#include "nurbscontact/contact_dynamics.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <exception>
#include <stdexcept>
#include <unordered_set>
#ifdef _OPENMP
#include <omp.h>
#endif

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

template<class Differential>
bool assemble_contact(Forces* forces, const ContactMaterial& material, ContactSample& sample,
                      const ContactBodyState& a, const ContactBodyState& b, const Vector& velocity,
                      const ImpactHistory& previous, const Differential& differential,Vec pressure_arm={}) {
    const double depth = -sample.gap;
    const bool incident_law = material.approach_floor > 0;
    const std::size_t n = velocity.size();
    Vector jacobian(n);
    std::vector<Vec> tangent(n);
    auto [rate,slip] = differential(a.prescribed,b.prescribed);
    for (std::size_t i=0;i<n;++i) {
        const auto [gradient,relative] = differential(a.derivative[i],b.derivative[i]);
        jacobian[i] = gradient;
        tangent[i] = relative;
        rate += jacobian[i]*velocity[i];
        slip = add(slip,scale(tangent[i],velocity[i]));
    }
    sample.gap_rate = rate;
    if (depth<=0) return true;

    const double elastic = material.stiffness*std::pow(depth,material.exponent);
    double damping_exponent = .5*(material.exponent-1);
    double c;
    if (incident_law) {
        if (previous.gap<0 && previous.impact_speed>0) {
            sample.impact_speed = previous.impact_speed;
        } else {
            double incidence = -rate;
            if (previous.gap>=0 && std::isfinite(previous.gap_rate)) {
                // Interpolate the end-point rates at the sign change of the gap.
                const double fraction = previous.gap/(previous.gap+depth);
                incidence = -std::lerp(previous.gap_rate,rate,fraction);
            }
            sample.impact_speed = std::max(material.approach_floor,incidence);
        }
        damping_exponent = material.exponent;
        c = elastic*material.impact_dissipation/sample.impact_speed;
    } else {
        c = material.rate*std::pow(depth,damping_exponent);
    }
    const double normal_force = elastic-c*rate;
    if (normal_force<=0) return true;
    const double tangent_speed = std::hypot(length(slip), material.slip_speed);
    const double friction = material.friction*normal_force/tangent_speed;
    sample.normal_force = normal_force;
    sample.friction = scale(slip,-friction);
    if (!forces) return true;
    const double k = (material.exponent*elastic-damping_exponent*c*rate)/depth;
    for (std::size_t i=0;i<n;++i) {
        const double normal_load=jacobian[i]+dot3(pressure_arm,sub(b.derivative[i].angular,a.derivative[i].angular));
        forces->value[i] += normal_load*normal_force + dot3(tangent[i],sample.friction);
        for (std::size_t j=0;j<n;++j) {
            const double product = normal_load*jacobian[j];
            forces->stiffness[i*n+j] += k*product;
            const double projection = dot3(tangent[i],slip)*dot3(tangent[j],slip)/(tangent_speed*tangent_speed);
            forces->damping[i*n+j] += c*product + friction*(dot3(tangent[i],tangent[j])-projection);
        }
    }
    return true;
}

bool evaluate_closest(Forces* forces, const ContactBodyState& a, const ContactBodyState& b,
                      const ContactPair& pair, const Vector& velocity, ContactWork& work,
                      ContactSample& sample, const ImpactHistory& previous) {
    ++work.closest_queries;
    const auto result = pair.closest->closest(a.pose, b.pose, pair.activation, work);
    if (result.status == PairStatus::separated) {
        ++work.culled;
        return true;
    }
    if (result.status == PairStatus::unresolved) {
        ++work.unresolved_queries;
        work.last_unresolved_distance = result.distance_bound_m;
        return false;
    }
    const auto& points = result.points;
    const Vec separation = sub(points.b, points.a);
    const double length_ab = length(separation);
    if (!(length_ab > 1e-12) || points.distance <= 1e-12) {
        ++work.unresolved_queries;
        return false;
    }
    sample.gap = points.distance - pair.activation;
    sample.witness_a = points.a;
    sample.witness_b = points.b;
    sample.resolved = true;
    if (-sample.gap > pair.material.maximum_depth) return false;
    if (sample.gap >= 0 && pair.material.approach_floor == 0) return true;

    const Vec normal = scale(separation, 1 / length_ab);
    const Vec arm_a = sub(points.a, a.pose.origin);
    const Vec arm_b = sub(points.b, b.pose.origin);
    const auto differential = [&](const Twist& da, const Twist& db) {
        const Vec relative = sub(db.at(arm_b), da.at(arm_a));
        return std::pair{dot3(normal, relative), tangential(relative, normal)};
    };
    return assemble_contact(forces, pair.material, sample, a, b, velocity, previous, differential);
}

struct RegionEvaluation {
    std::vector<ContactRegion> regions;
    ContactWork work;
    std::exception_ptr error;
};

#ifdef _OPENMP
// Each admitted pair owns its mutable geometry. Material response and HHT
// assembly keep their original per-pair and per-region accumulation order.
std::vector<RegionEvaluation> batch_mesh_regions(std::span<const ContactBodyState> bodies,
        std::span<const ContactPair> pairs,ContactWork& work) {
    const int workers=std::min(omp_get_max_threads(),static_cast<int>(std::min<std::size_t>(4,pairs.size())));
    if (workers<2 || omp_in_parallel()) return {};
    std::unordered_set<const PairRegionQuery*> seen;
    std::size_t faces=0;
    bool cad_patch=false;
    for (const auto& pair:pairs) {
        const auto* query=pair.regions.get();
        if (!query || (!query->independent_mesh_faces() && !query->independent_cad_patch()) ||
            !seen.insert(query).second) return {};
        cad_patch|=query->independent_cad_patch();
        faces+=query->independent_mesh_faces();
    }
    if (!cad_patch && faces<1024) return {};
    std::vector<RegionEvaluation> batch(pairs.size());
    const auto start=std::chrono::steady_clock::now();
    int team=1;
#pragma omp parallel num_threads(workers)
    {
#pragma omp single
        team=omp_get_num_threads();
#pragma omp for schedule(dynamic,1)
        for (std::ptrdiff_t i=0;i<static_cast<std::ptrdiff_t>(pairs.size());++i) {
            const auto& pair=pairs[i];
            auto& evaluation=batch[i];
            try {
                evaluation.regions=pair.regions->regions(bodies[pair.a].pose,bodies[pair.b].pose,evaluation.work);
            } catch (...) {evaluation.error=std::current_exception();}
        }
    }
    const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    if (cad_patch) {work.cad_patch_batch_s+=elapsed;++work.cad_patch_batches;}
    else {
        work.mesh_detection_s+=elapsed;
        ++work.mesh_parallel_batches;
        work.mesh_max_workers=std::max(work.mesh_max_workers,static_cast<std::size_t>(team));
    }
    for (const auto& entry:batch) {
        work.cad_patch_queries+=entry.work.cad_patch_queries;
        work.cad_window_builds+=entry.work.cad_window_builds;
        work.cad_window_reuses+=entry.work.cad_window_reuses;
        work.cad_path_reuses+=entry.work.cad_path_reuses;
        work.cad_skipped_area_nodes+=entry.work.cad_skipped_area_nodes;
        work.cad_pressure_clipped+=entry.work.cad_pressure_clipped;
        work.cad_pressure_full+=entry.work.cad_pressure_full;
        work.cad_region_select_s+=entry.work.cad_region_select_s;
        work.cad_patch_broad_s+=entry.work.cad_patch_broad_s;
        work.cad_patch_search_s+=entry.work.cad_patch_search_s;
        work.cad_patch_group_s+=entry.work.cad_patch_group_s;
        work.cad_patch_area_s+=entry.work.cad_patch_area_s;
        work.cad_boolean_queries+=entry.work.cad_boolean_queries;
        work.cad_region_common_s+=entry.work.cad_region_common_s;
        work.cad_region_extract_s+=entry.work.cad_region_extract_s;
        work.mesh.nodes+=entry.work.mesh.nodes;
        work.mesh.triangles+=entry.work.mesh.triangles;
        work.mesh_pair_bvh_builds+=entry.work.mesh_pair_bvh_builds;
        work.mesh_pair_bvh_reuses+=entry.work.mesh_pair_bvh_reuses;
        work.mesh_patch_builds+=entry.work.mesh_patch_builds;
        work.mesh_patch_reuses+=entry.work.mesh_patch_reuses;
    }
    return batch;
}
#endif

bool evaluate(Forces* forces, const ContactBodyState& a, const ContactBodyState& b,
              const ContactPair& pair, const Vector& velocity,
              ContactWork& work, ContactSample& sample, bool observe,
              const ImpactHistory& previous,RegionEvaluation* prepared) {
    ++work.pairs;
    if (pair.regions) {
        std::vector<ContactRegion> regions;
        try {
            if (prepared) {
                if (prepared->error) std::rethrow_exception(prepared->error);
                regions=std::move(prepared->regions);
            } else regions=pair.regions->regions(a.pose,b.pose,work);
        } catch (const UnresolvedContact&) {
            ++work.unresolved_queries;
            return false;
        }
        if (regions.empty()) { ++work.culled; return true; }
        if (pair.material.approach_floor>0 && regions.size()>1)
            throw std::runtime_error("incidence-dependent damping needs per-region impact histories");
        const bool shared=pair.share_penalty_stiffness && (regions.size()>1 || pair.regions->predictor_slots()>1);
        Forces local;
        if (shared && forces) {
            local.value.resize(velocity.size());
            local.stiffness.resize(velocity.size()*velocity.size());
            local.damping.resize(local.stiffness.size());
        }
        std::size_t loaded=0;
        for (const auto& region : regions) {
            ContactSample item;
            item.gap=-region.depth_m;
            if (region.depth_m>pair.material.maximum_depth) {
                ++work.rejected_region_depths;
                work.last_rejected_depth_m=region.depth_m;
                work.last_depth_limit_m=pair.material.maximum_depth;
                work.last_depth_bodies={pair.a,pair.b};
                return false;
            }
            const Vec ra=sub(region.point,a.pose.origin), rb=sub(region.point,b.pose.origin);
            const auto differential=[&](const Twist& da,const Twist& db) {
                const Vec relative=sub(db.at(rb),da.at(ra));
                return std::pair{dot3(region.normal,relative),tangential(relative,region.normal)};
            };
            if (!assemble_contact(shared && forces ? &local : forces,pair.material,item,a,b,velocity,previous,differential,region.pressure_arm)) return false;
            loaded+=item.normal_force>0;
            if (!sample.resolved || item.gap<sample.gap) {
                sample.gap=item.gap;
                sample.gap_rate=item.gap_rate;
                sample.impact_speed=item.impact_speed;
                sample.witness_a=sample.witness_b=region.point;
            }
            sample.resolved=true;
            sample.normal_force+=item.normal_force;
            sample.friction=add(sample.friction,item.friction);
        }
        if (const auto predicted=pair.regions->predictor_slots(); predicted) loaded=predicted;
        if (shared && loaded) {
            const double weight=1./static_cast<double>(loaded);
            sample.normal_force*=weight;
            sample.friction=scale(sample.friction,weight);
            if (forces) {
                for (std::size_t i=0;i<local.value.size();++i) forces->value[i]+=weight*local.value[i];
                for (std::size_t i=0;i<local.stiffness.size();++i) {
                    forces->stiffness[i]+=weight*local.stiffness[i];
                    forces->damping[i]+=weight*local.damping[i];
                }
            }
        }
        return true;
    }
    if (pair.closest) return evaluate_closest(forces, a, b, pair, velocity, work, sample, previous);
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
    const auto& material = pair.material;
    const double depth = -sample.gap;
    const bool incident_law = material.approach_floor>0;
    if (depth<=0 && !incident_law) return true;
    if (depth > material.maximum_depth) return false;
    const Vec normal_a = radial ? normal : left->normal;
    const Vec normal_b = radial ? normal : right->normal;
    const double cos_a = dot3(normal_a,normal), cos_b = dot3(normal_b,normal);
    if (std::abs(cos_a)<1e-6 || std::abs(cos_b)<1e-6) return depth<=0;

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
    return assemble_contact(forces, material, sample, a, b, velocity, previous, differential);
}

} // namespace

Vec ContactPose::rotate(Vec local) const {
    return angle==0 ? local : rotated(local,axis,angle);
}

Vec ContactPose::inverse_rotate(Vec world) const {
    return angle==0 ? world : rotated(world,axis,-angle);
}

ContactPairWrenches line_contact_wrenches(std::span<const ContactLineSample> samples, double activation_m,
                                         double stiffness_per_m, Vec center_a, Vec center_b) {
    if (!std::isfinite(activation_m) || activation_m < 0 || !std::isfinite(stiffness_per_m) || stiffness_per_m < 0) {
        throw std::invalid_argument("invalid line contact material");
    }
    ContactPairWrenches result;
    for (const auto& sample : samples) {
        const Vec delta = sub(sample.points.b, sample.points.a);
        const double distance = length(delta);
        if (!std::isfinite(distance) || distance <= 0 || !std::isfinite(sample.weight_m) || sample.weight_m <= 0) {
            throw std::invalid_argument("invalid line contact sample");
        }
        const double load = stiffness_per_m * sample.weight_m * std::max(0., activation_m - distance);
        const Vec force = scale(delta, load / distance);
        result.a.force = sub(result.a.force, force);
        result.b.force = add(result.b.force, force);
        result.a.moment = sub(result.a.moment, cross3(sub(sample.points.a, center_a), force));
        result.b.moment = add(result.b.moment, cross3(sub(sample.points.b, center_b), force));
    }
    return result;
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
                        ContactWork& work, std::vector<ContactSample>* observations,
                        std::span<const ImpactHistory> previous) {
    if (!previous.empty() && previous.size()!=pairs.size())
        throw std::invalid_argument("contact history must match the probe count");
    if (observations) {
        observations->clear();
        observations->reserve(pairs.size());
    }
    std::vector<RegionEvaluation> prepared;
#ifdef _OPENMP
    if (pairs.size()>1) prepared=batch_mesh_regions(bodies,pairs,work);
#endif
    for (std::size_t i=0; i<pairs.size(); ++i) {
        const auto& pair = pairs[i];
        ContactSample sample;
        const ImpactHistory start = previous.empty() ? ImpactHistory{} : previous[i];
        if (!evaluate(forces,bodies[pair.a],bodies[pair.b],pair,velocity,work,sample,observations!=nullptr,start,
                      prepared.empty()?nullptr:&prepared[i])) {
            return false;
        }
        if (observations) observations->push_back(sample);
    }
    return true;
}

} // namespace nc::dynamics
