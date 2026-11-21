#include "nurbscontact/contact_dynamics.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nc::dynamics {
namespace {

Vec rotated(Vec point, Vec axis, double angle) {
    const double cosine = std::cos(angle), sine = std::sin(angle);
    return add(add(scale(point, cosine), scale(cross3(axis, point), sine)),
               scale(axis, (1-cosine)*dot3(axis, point)));
}

Vec tangential(Vec point, Vec normal) {
    return sub(point, scale(normal, dot3(point, normal)));
}

double maximum_projection(TriangleBvh::Box box, Vec direction) {
    double maximum = 0;
    for (std::size_t i=0; i<3; ++i) {
        maximum += direction[i]*(direction[i]>=0 ? box.hi[i] : box.lo[i]);
    }
    return maximum;
}

struct WorldHit {
    double along;
    Vec point;
    Vec normal;
};

std::optional<WorldHit> hit(const ContactBodyState& body, Vec ray, Vec direction,
                            ContactWork& work, bool from_anchor=false) {
    const Vec local_ray = body.pose.inverse_rotate(sub(ray,body.pose.origin));
    const Vec local_direction = body.pose.inverse_rotate(direction);
    const auto box = body.surface->bounds();
    constexpr double margin = 1e-8;
    const double low = -maximum_projection(box, scale(local_direction,-1))-margin;
    const double high = maximum_projection(box, local_direction)+margin;
    const double projected = dot3(local_ray,local_direction);
    const double shift = from_anchor ? 0 : low-projected;
    const double distance = high-(from_anchor ? projected : low);
    ++work.rays;
    if (distance<0) {
        ++work.missing;
        return std::nullopt;
    }
    const auto result = body.surface->first_hit(add(local_ray,scale(local_direction,shift)),
                                                 local_direction,distance,work);
    if (!result) {
        ++work.missing;
        return std::nullopt;
    }
    const double along = shift+result->distance;
    return WorldHit{along,add(ray,scale(direction,along)),body.pose.rotate(result->normal)};
}

bool evaluate(Forces* forces, const ContactBodyState& a, const ContactBodyState& b,
              const ContactPair& pair, const Vector& velocity,
              ContactWork& work, ContactSample& sample, bool observe) {
    ++work.pairs;
    const Vec anchor_a = a.pose.point(pair.anchor_a);
    const Vec anchor_b = b.pose.point(pair.anchor_b);
    const Vec axis = sub(anchor_b,anchor_a);
    const double center_distance = length(axis);
    const bool radial = pair.direction==ContactDirection::contained_radial;
    const bool contained = pair.direction==ContactDirection::contained || radial;
    const bool moving_axis = (pair.direction==ContactDirection::anchors || contained)
                             && center_distance>1e-12;
    Vec normal = pair.normal;
    if (moving_axis) {
        normal = scale(axis, (contained ? -1 : 1)/center_distance);
    } else if (pair.direction==ContactDirection::anchors) {
        return false;
    } else if (pair.direction==ContactDirection::body_a || contained) {
        normal = a.pose.rotate(normal);
    }
    if (!observe && !contained) {
        const auto lower = dot3(sub(b.pose.origin,a.pose.origin),normal)
            -maximum_projection(a.surface->bounds(),a.pose.inverse_rotate(normal))
            -maximum_projection(b.surface->bounds(),b.pose.inverse_rotate(scale(normal,-1)));
        if (lower>1e-8) {
            ++work.culled;
            return true;
        }
    }
    // For containment the two anchors lie on the same ray line. Each ray
    // starts in its own region, so crossing the bore rim cannot switch walls.
    const Vec outward = scale(normal,-1);
    const auto left = hit(a,contained ? anchor_a : anchor_b,outward,work,contained);
    if (!left) return !contained;
    const auto right = hit(b,anchor_b,contained ? outward : normal,work,contained);
    if (!right) return !contained;
    const double lambda_a = (contained ? center_distance : 0)-left->along;
    const double lambda_b = contained ? -right->along : right->along;
    sample.gap = lambda_b-lambda_a;
    sample.resolved = true;
    sample.witness_a = left->point;
    sample.witness_b = right->point;
    if (sample.gap >= 0) return true;

    const auto& material = pair.material;
    const double depth = -sample.gap;
    if (depth > material.maximum_depth) return false;
    const Vec normal_a = radial ? normal : left->normal;
    const Vec normal_b = radial ? normal : right->normal;
    const double cos_a = dot3(normal_a,normal), cos_b = dot3(normal_b,normal);
    if (std::abs(cos_a)<1e-6 || std::abs(cos_b)<1e-6) return false;

    const std::size_t n = velocity.size();
    Vector jacobian(n);
    std::vector<Vec> tangent(n);
    const auto differential = [&](const Twist& da, const Twist& db) {
        const Vec ray_rate = db.at(sub(anchor_b,b.pose.origin));
        Vec normal_rate{};
        if (moving_axis) {
            const Vec offset_rate = sub(ray_rate,da.at(sub(anchor_a,a.pose.origin)));
            normal_rate = scale(tangential(offset_rate,normal),(contained ? -1 : 1)/center_distance);
        } else if (pair.direction==ContactDirection::body_a || contained) {
            normal_rate = cross3(da.angular,normal);
        }
        const Vec a_rate = da.at(sub(left->point,a.pose.origin));
        const Vec b_rate = db.at(sub(right->point,b.pose.origin));
        const double la = dot3(normal_a,sub(sub(a_rate,ray_rate),scale(normal_rate,lambda_a)))/cos_a;
        const double lb = dot3(normal_b,sub(sub(b_rate,ray_rate),scale(normal_rate,lambda_b)))/cos_b;
        return std::pair{lb-la,tangential(sub(b_rate,a_rate),normal)};
    };
    auto [rate,slip] = differential(a.prescribed,b.prescribed);
    for (std::size_t i=0;i<n;++i) {
        const auto [gradient,relative] = differential(a.derivative[i],b.derivative[i]);
        jacobian[i] = gradient;
        tangent[i] = relative;
        rate += jacobian[i]*velocity[i];
        slip = add(slip,scale(tangent[i],velocity[i]));
    }
    const double elastic = material.stiffness*std::pow(depth,material.exponent);
    const double damping_exponent = .5*(material.exponent-1);
    const double c = material.rate*std::pow(depth,damping_exponent);
    const double normal_force = elastic-c*rate;
    if (normal_force<=0) return true;
    const double tangent_speed = std::hypot(length(slip), material.slip_speed);
    const double friction = material.friction*normal_force/tangent_speed;
    sample.normal_force = normal_force;
    sample.friction = scale(slip,-friction);
    if (!forces) return true;
    const double k = (material.exponent*elastic-damping_exponent*c*rate)/depth;
    for (std::size_t i=0;i<n;++i) {
        forces->value[i] += jacobian[i]*normal_force + dot3(tangent[i],sample.friction);
        for (std::size_t j=0;j<n;++j) {
            const double product = jacobian[i]*jacobian[j];
            forces->stiffness[i*n+j] += k*product;
            const double projection = dot3(tangent[i],slip)*dot3(tangent[j],slip)/(tangent_speed*tangent_speed);
            forces->damping[i*n+j] += c*product + friction*(dot3(tangent[i],tangent[j])-projection);
        }
    }
    return true;
}

} // namespace

Vec ContactPose::rotate(Vec local) const {
    return angle==0 ? local : rotated(local,axis,angle);
}

Vec ContactPose::inverse_rotate(Vec world) const {
    return angle==0 ? world : rotated(world,axis,-angle);
}

void add_point_resistances(Forces& forces, std::span<const ContactBodyState> bodies,
                           std::span<const PointResistance> resistances, const Vector& velocity) {
    const auto n = velocity.size();
    Vector projections(n);
    std::vector<Vec> jacobian(n);
    for (const auto& item:resistances) {
        const auto& a = bodies[item.a];
        const auto& b = bodies[item.b];
        const Vec arm_a = a.pose.rotate(item.anchor_a);
        const Vec arm_b = b.pose.rotate(item.anchor_b);
        const auto relative = [&](const Twist& da, const Twist& db) {
            return tangential(sub(db.at(arm_b),da.at(arm_a)),item.normal);
        };
        Vec slip = relative(a.prescribed,b.prescribed);
        for (std::size_t i=0;i<n;++i) {
            jacobian[i] = relative(a.derivative[i],b.derivative[i]);
            slip = add(slip,scale(jacobian[i],velocity[i]));
        }
        const double speed = std::hypot(length(slip),item.slip_speed);
        const double coefficient = item.force_limit/speed;
        for (std::size_t i=0;i<n;++i) {
            const double along = dot3(jacobian[i],slip);
            projections[i] = along/speed;
            forces.value[i] -= coefficient*along;
        }
        for (std::size_t i=0;i<n;++i) {
            for (std::size_t j=0;j<n;++j) {
                forces.damping[i*n+j] += coefficient*
                    (dot3(jacobian[i],jacobian[j])-projections[i]*projections[j]);
            }
        }
    }
}

void add_translational_bushings(Forces& forces, std::span<const ContactBodyState> bodies,
                                std::span<const TranslationalBushing> bushings, const Vector& velocity) {
    const auto n = velocity.size();
    std::vector<Vec> jacobian(n);
    for (const auto& element:bushings) {
        const auto& a = bodies[element.a];
        const auto& b = bodies[element.b];
        const Vec arm_a = a.pose.rotate(element.anchor_a);
        const Vec arm_b = b.pose.rotate(element.anchor_b);
        const Vec difference = sub(add(b.pose.origin,arm_b),add(a.pose.origin,arm_a));
        const Vec strain = sub(a.pose.inverse_rotate(difference),element.rest);
        const auto relative = [&](const Twist& da, const Twist& db) {
            // Transport the separation vector with A's rotating frame.
            return a.pose.inverse_rotate(sub(sub(db.at(arm_b),da.at(arm_a)),
                                              cross3(da.angular,difference)));
        };
        Vec rate = relative(a.prescribed,b.prescribed);
        for (std::size_t i=0;i<n;++i) {
            jacobian[i] = relative(a.derivative[i],b.derivative[i]);
            rate = add(rate,scale(jacobian[i],velocity[i]));
        }
        Vec restoring{};
        for (std::size_t axis=0;axis<3;++axis)
            restoring[axis] = element.stiffness[axis]*strain[axis]+element.damping[axis]*rate[axis];
        for (std::size_t i=0;i<n;++i) {
            forces.value[i] -= dot3(jacobian[i],restoring);
            for (std::size_t j=0;j<n;++j) {
                for (std::size_t axis=0;axis<3;++axis) {
                    const double projection = jacobian[i][axis]*jacobian[j][axis];
                    forces.stiffness[i*n+j] += element.stiffness[axis]*projection;
                    forces.damping[i*n+j] += element.damping[axis]*projection;
                }
            }
        }
    }
}

bool add_contact_forces(Forces* forces, std::span<const ContactBodyState> bodies,
                        std::span<const ContactPair> pairs, const Vector& velocity,
                        ContactWork& work, std::vector<ContactSample>* observations) {
    if (observations) {
        observations->clear();
        observations->reserve(pairs.size());
    }
    for (const auto& pair:pairs) {
        ContactSample sample;
        if (!evaluate(forces,bodies[pair.a],bodies[pair.b],pair,velocity,work,sample,observations!=nullptr)) {
            return false;
        }
        if (observations) observations->push_back(sample);
    }
    return true;
}

} // namespace nc::dynamics
