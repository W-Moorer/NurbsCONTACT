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
                             ContactWork& work) {
    const Vec local_ray = body.pose.inverse_rotate(sub(ray,body.pose.origin));
    const Vec local_direction = body.pose.inverse_rotate(direction);
    const auto box = body.surface->bounds();
    constexpr double margin = 1e-8;
    const double low = -maximum_projection(box, scale(local_direction,-1))-margin;
    const double high = maximum_projection(box, local_direction)+margin;
    const double shift = low-dot3(local_ray,local_direction);
    ++work.rays;
    const auto result = body.surface->first_hit(add(local_ray,scale(local_direction,shift)),
                                                 local_direction,high-low,work);
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
    Vec normal = pair.normal;
    if (pair.direction==ContactDirection::anchors) {
        if (center_distance<=1e-12) return false;
        normal = scale(axis, 1/center_distance);
    } else if (pair.direction==ContactDirection::body_a) {
        normal = a.pose.rotate(normal);
    }
    if (!observe) {
        const auto lower = dot3(sub(b.pose.origin,a.pose.origin),normal)
            -maximum_projection(a.surface->bounds(),a.pose.inverse_rotate(normal))
            -maximum_projection(b.surface->bounds(),b.pose.inverse_rotate(scale(normal,-1)));
        if (lower>1e-8) {
            ++work.culled;
            return true;
        }
    }
    const auto left = hit(a,anchor_b,scale(normal,-1),work);
    if (!left) return true;
    const auto right = hit(b,anchor_b,normal,work);
    if (!right) return true;
    const double lambda_a = -left->along, lambda_b = right->along;
    sample.gap = lambda_b-lambda_a;
    sample.resolved = true;
    sample.witness_a = left->point;
    sample.witness_b = right->point;
    if (sample.gap >= 0) return true;

    const auto& material = pair.material;
    const double depth = -sample.gap;
    if (depth > material.maximum_depth) return false;
    const double cos_a = dot3(left->normal,normal), cos_b = dot3(right->normal,normal);
    if (std::abs(cos_a)<1e-6 || std::abs(cos_b)<1e-6) return false;

    const std::size_t n = velocity.size();
    Vector jacobian(n);
    std::vector<Vec> tangent(n);
    double rate=0;
    Vec slip{};
    for (std::size_t i=0;i<n;++i) {
        const auto& da = a.derivative[i];
        const auto& db = b.derivative[i];
        const Vec ray_rate = db.at(sub(anchor_b,b.pose.origin));
        Vec normal_rate{};
        if (pair.direction==ContactDirection::anchors) {
            const Vec offset_rate = sub(ray_rate,da.at(sub(anchor_a,a.pose.origin)));
            normal_rate = scale(tangential(offset_rate,normal),1/center_distance);
        } else if (pair.direction==ContactDirection::body_a) {
            normal_rate = cross3(da.angular,normal);
        }
        const Vec a_rate = da.at(sub(left->point,a.pose.origin));
        const Vec b_rate = db.at(sub(right->point,b.pose.origin));
        const double la = dot3(left->normal,sub(sub(a_rate,ray_rate),scale(normal_rate,lambda_a)))/cos_a;
        const double lb = dot3(right->normal,sub(sub(b_rate,ray_rate),scale(normal_rate,lambda_b)))/cos_b;
        jacobian[i] = lb-la;
        tangent[i] = tangential(sub(b_rate,a_rate),normal);
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
