#include "nurbscontact/cpu_mesh.hpp"
#include "cpu_mesh_solid.hpp"
#include "cpu_mesh_contours.hpp"
#ifdef NC_WITH_MESH_CCD
#include "cpu_mesh_ccd.hpp"
#include "cpu_mesh_motion.hpp"
#endif

#include <fcl/broadphase/broadphase_dynamic_AABB_tree.h>
#include <fcl/geometry/bvh/BVH_model.h>
#include <fcl/geometry/shape/box.h>
#include <fcl/math/bv/OBB.h>

#include <climits>
#include <map>
#include <numeric>
#include <tuple>
#include <variant>

namespace nc::cpu {
namespace {
using V = fcl::Vector3d;
using X = fcl::Transform3d;
using Bounds = fcl::AABBd;
using Oriented = fcl::OBBd;
template<class Box> using Tree = fcl::BVHModel<Box>;
using Trees = std::variant<std::shared_ptr<Tree<Bounds>>, std::shared_ptr<Tree<Oriented>>>;

V vector(Vec v) { return {v[0], v[1], v[2]}; }

X transform(const RigidPlacement& pose) {
    X result = X::Identity();
    result.translation() = vector(pose.origin);
    for (int col = 0; col < 3; ++col) result.linear().col(col) = vector(pose.basis[col]);
    if (!result.matrix().allFinite() || !result.linear().isUnitary(1e-12) || result.linear().determinant() <= 0)
        throw std::invalid_argument("a body pose must be a proper rigid transform");
    return result;
}

double roundoff(const Bounds& box, const X& pose) {
    return 64 * std::numeric_limits<double>::epsilon() *
        (1 + box.min_.cwiseAbs().maxCoeff() + box.max_.cwiseAbs().maxCoeff() + pose.translation().cwiseAbs().maxCoeff());
}

Bounds placed(const Bounds& box, const X& pose, double padding) {
    const V center = pose * box.center();
    const V radius = pose.linear().cwiseAbs() * ((box.max_ - box.min_) * .5) + V::Constant(padding);
    return {center - radius, center + radius};
}

Oriented placed(const Oriented& box, const X& pose, double padding) {
    return {pose.linear() * box.axis, pose * box.To, box.extent + V::Constant(padding)};
}

template<class Box>
std::shared_ptr<Tree<Box>> build_tree(const MeshData& mesh) {
    if (mesh.faces.empty() || mesh.faces.size() > INT_MAX / 2 || mesh.vertices.size() > INT_MAX)
        throw std::invalid_argument("unsupported triangle count");
    std::vector<V> vertices;
    for (const auto& point : mesh.vertices) {
        const V p = vector(point);
        if (!p.allFinite()) throw std::invalid_argument("non-finite mesh vertex");
        vertices.push_back(p);
    }
    std::vector<fcl::Triangle> faces;
    for (const auto& f : mesh.faces) {
        const V a = vertices.at(f[0]), b = vertices.at(f[1]), c = vertices.at(f[2]);
        if ((b-a).cross(c-a).squaredNorm() == 0) throw std::invalid_argument("degenerate input triangle");
        faces.emplace_back(f[0], f[1], f[2]);
    }
    auto tree = std::make_shared<Tree<Box>>();
    if (tree->beginModel(static_cast<int>(faces.size()), static_cast<int>(vertices.size())) != fcl::BVH_OK ||
        tree->addSubModel(vertices, faces) != fcl::BVH_OK || tree->endModel() != fcl::BVH_OK)
        throw std::runtime_error("FCL mesh construction failed");
    tree->computeLocalAABB();
    return tree;
}

class BodyProxy final : public fcl::CollisionObjectd {
public:
    BodyProxy(std::size_t index, const std::shared_ptr<fcl::CollisionGeometryd>& geometry)
        : fcl::CollisionObjectd(geometry), index(index) {}
    void bounds(Bounds value) { aabb = std::move(value); }
    const std::size_t index;
};

using Pairs = std::vector<std::pair<std::size_t, std::size_t>>;
bool collect(fcl::CollisionObjectd* left, fcl::CollisionObjectd* right, void* context) {
    const auto a = static_cast<BodyProxy*>(left)->index, b = static_cast<BodyProxy*>(right)->index;
    static_cast<Pairs*>(context)->emplace_back(std::min(a,b), std::max(a,b));
    return false; // Continue collecting other potentially intersecting bodies.
}

Triangle triangle(const MeshData& mesh, std::size_t id, const RigidPlacement& pose) {
    const auto& f = mesh.faces[id];
    return {pose.point(mesh.vertices[f[0]]), pose.point(mesh.vertices[f[1]]), pose.point(mesh.vertices[f[2]])};
}

struct Body {
    std::shared_ptr<const MeshData> mesh;
    Trees tree;
    std::shared_ptr<SolidBoundary> boundary;
    std::unique_ptr<BodyProxy> proxy;
    Bounds local;
    X world = X::Identity();
    double padding{};
};

Pairs sweep_and_prune(const std::vector<Body>& bodies, std::vector<std::size_t>& order) {
    Pairs candidates;
    if (bodies.empty()) return candidates;
    V lo = bodies.front().proxy->getAABB().center(), hi = lo;
    for (const auto& body : bodies) {
        const V center = body.proxy->getAABB().center();
        lo = lo.cwiseMin(center);
        hi = hi.cwiseMax(center);
    }
    Eigen::Index axis;
    (hi-lo).maxCoeff(&axis);
    std::sort(order.begin(),order.end(),[&](std::size_t i, std::size_t j) {
        const auto a = bodies[i].proxy->getAABB().min_[axis], b = bodies[j].proxy->getAABB().min_[axis];
        return a < b || (a == b && i < j);
    });
    for (std::size_t i=0; i<order.size(); ++i) {
        const auto& a = bodies[order[i]].proxy->getAABB();
        for (std::size_t j=i+1; j<order.size(); ++j) {
            const auto& b = bodies[order[j]].proxy->getAABB();
            if (b.min_[axis] > a.max_[axis]) break;
            if (a.overlap(b)) candidates.emplace_back(std::min(order[i],order[j]),std::max(order[i],order[j]));
        }
    }
    std::sort(candidates.begin(),candidates.end());
    return candidates;
}

void enclosed(const Body& a, const Body& b, std::span<const Intersection> hits, bool first, IntersectionResult& result) {
    const X relative = b.world.inverse() * a.world;
    const auto representatives = a.boundary->representatives();
    if (representatives.size()==1 && !hits.empty()) return;
    std::vector<bool> crossing(representatives.size());
    for (const auto& hit : hits) crossing[a.boundary->component(first ? hit.triangle_a : hit.triangle_b)] = true;
    for (std::size_t component = 0; component < representatives.size(); ++component) {
        if (crossing[component]) continue;
        // Without a crossing the entire connected boundary is on the same
        // side. One actual exterior vertex rules out enclosure.
        const auto& extremes=a.boundary->extreme_vertices(component);
        const double slack=a.padding+b.padding;
        const auto outside=[&](std::size_t vertex) {
            const V p=relative*vector(a.mesh->vertices[vertex]);
            return (p.array()<b.local.min_.array()-slack).any() ||
                   (p.array()>b.local.max_.array()+slack).any();
        };
        if (std::any_of(extremes.begin(),extremes.end(),outside)) continue;
        const auto id = representatives[component];
        const V p = relative * vector(a.mesh->vertices[id]);
        if (b.local.contain(p) && b.boundary->contains({p.x(),p.y(),p.z()}))
            result.enclosed.push_back({a.proxy->index,b.proxy->index,id});
    }
}

template<class Box>
void intersect_pair(const Body& a, const Body& b, std::span<const RigidPlacement> poses,
                    IntersectionResult& result, TriangleContours* contours) {
    const auto i = a.proxy->index, j = b.proxy->index;
    const auto& first = *std::get<std::shared_ptr<Tree<Box>>>(a.tree);
    const auto& second = *std::get<std::shared_ptr<Tree<Box>>>(b.tree);
    std::vector<std::pair<int,int>> pending{{0,0}};
    while (!pending.empty()) {
        const auto [ia, ib] = pending.back();
        pending.pop_back();
        ++result.node_pairs;
        const auto& na = first.getBV(ia);
        const auto& nb = second.getBV(ib);
        if (!placed(na.bv, a.world, a.padding).overlap(placed(nb.bv, b.world, b.padding))) continue;
        if (na.isLeaf() && nb.isLeaf()) {
            const auto fa = static_cast<std::size_t>(na.primitiveId());
            const auto fb = static_cast<std::size_t>(nb.primitiveId());
            ++result.triangle_tests;
            const auto ta=triangle(*a.mesh,fa,poses[i]), tb=triangle(*b.mesh,fb,poses[j]);
            auto points = contours ? contours->intersect(ta,tb) : intersect_triangles(ta,tb);
            if (!points.empty()) result.intersections.push_back({i,j,fa,fb,std::move(points)});
        } else if (nb.isLeaf() || (!na.isLeaf() && na.num_primitives >= nb.num_primitives)) {
            pending.emplace_back(na.leftChild(), ib);
            pending.emplace_back(na.rightChild(), ib);
        } else {
            pending.emplace_back(ia, nb.leftChild());
            pending.emplace_back(ia, nb.rightChild());
        }
    }
}
}

struct MeshScene::Impl {
    std::vector<Body> bodies;
    std::unique_ptr<fcl::BroadPhaseCollisionManagerd> manager;
    std::vector<std::size_t> order;
    bool registered{};
    MeshMethod method;

    Pairs candidates() {
        if (!manager) return sweep_and_prune(bodies,order);
        if (!registered) {
            for (const auto& body : bodies) manager->registerObject(body.proxy.get());
            manager->setup();
            registered = true;
        } else {
            manager->update();
        }
        Pairs pairs;
        manager->collide(&pairs,collect);
        std::sort(pairs.begin(),pairs.end());
        return pairs;
    }
};

MeshScene::MeshScene(std::span<const std::shared_ptr<const MeshData>> inputs, MeshMethod method)
    : impl_(std::make_unique<Impl>()) {
    impl_->method = method;
    if (method == MeshMethod::sap) {
        impl_->order.resize(inputs.size());
        std::iota(impl_->order.begin(),impl_->order.end(),std::size_t{0});
    } else impl_->manager = std::make_unique<fcl::DynamicAABBTreeCollisionManagerd>();
    std::map<const MeshData*, Trees> built;
    std::map<const MeshData*, std::shared_ptr<SolidBoundary>> solids;
    auto proxy_shape = std::make_shared<fcl::Boxd>(1.,1.,1.);
    for (const auto& mesh : inputs) {
        if (!mesh) throw std::invalid_argument("missing mesh");
        auto [it, fresh] = built.try_emplace(mesh.get());
        if (fresh) {
            if (method == MeshMethod::obb) it->second = build_tree<Oriented>(*mesh);
            else it->second = build_tree<Bounds>(*mesh);
            solids.emplace(mesh.get(),std::make_shared<SolidBoundary>(*mesh));
        }
        auto proxy = std::make_unique<BodyProxy>(impl_->bodies.size(), proxy_shape);
        auto local = std::visit([](const auto& tree) { return tree->aabb_local; }, it->second);
        impl_->bodies.push_back({mesh,it->second,solids.at(mesh.get()),std::move(proxy),local});
    }
}

MeshScene::~MeshScene() = default;
MeshScene::MeshScene(MeshScene&&) noexcept = default;
MeshScene& MeshScene::operator=(MeshScene&&) noexcept = default;

IntersectionResult MeshScene::intersect(std::span<const RigidPlacement> poses, bool collect_contours) {
    auto& state = *impl_;
    if (poses.size() != state.bodies.size()) throw std::invalid_argument("one pose required per body");
    for (std::size_t i = 0; i < poses.size(); ++i) {
        auto& body = state.bodies[i];
        body.world = transform(poses[i]);
        body.padding = roundoff(body.local, body.world);
        body.proxy->bounds(placed(body.local,body.world,body.padding));
    }
    const auto candidates=state.candidates();
    IntersectionResult result;
    result.body_pairs = candidates.size();
    for (const auto& [a,b] : candidates) {
        const auto before = result.intersections.size();
        auto contours=collect_contours ? std::make_unique<TriangleContours>() : nullptr;
        if (state.method == MeshMethod::obb) intersect_pair<Oriented>(state.bodies[a],state.bodies[b],poses,result,contours.get());
        else intersect_pair<Bounds>(state.bodies[a],state.bodies[b],poses,result,contours.get());
        const auto hits = std::span<const Intersection>(result.intersections).subspan(before);
        if (contours) {
            auto groups=contours->finish(hits);
            result.contours.insert(result.contours.end(),std::make_move_iterator(groups.begin()),std::make_move_iterator(groups.end()));
        }
        enclosed(state.bodies[a],state.bodies[b],hits,true,result);
        enclosed(state.bodies[b],state.bodies[a],hits,false,result);
    }
    std::sort(result.intersections.begin(), result.intersections.end(), [](const Intersection& a, const Intersection& b) {
        return std::tie(a.body_a,a.body_b,a.triangle_a,a.triangle_b) < std::tie(b.body_a,b.body_b,b.triangle_a,b.triangle_b);
    });
    return result;
}

#ifdef NC_WITH_MESH_CCD
namespace {
struct SweepBody {
    MeshMotion motion;
    X start, end;
    double radius, padding, sagitta;

    Bounds envelope(const Bounds& box) const {
        const double buffer=padding+sagitta;
        return placed(box,start,buffer)+placed(box,end,buffer);
    }

    Oriented envelope(const Oriented& box) const {
        // Accumulate both end boxes in a fixed frame aligned with the initial OBB.
        const Eigen::Matrix3d axes=start.linear()*box.axis;
        Bounds enclosure;
        for (const X* pose : {&start,&end}) {
            const V center=axes.transpose()*((*pose)*box.To);
            const V radius_in_frame=(axes.transpose()*pose->linear()*box.axis).cwiseAbs()*box.extent;
            enclosure+=center-radius_in_frame;
            enclosure+=center+radius_in_frame;
        }
        return {axes,axes*enclosure.center(),(enclosure.max_-enclosure.min_)*.5+V::Constant(sagitta+padding)};
    }
};

Triangle world_triangle(const Triangle& local,const X& pose) {
    Triangle world;
    for (std::size_t k=0;k<3;++k) {
        const V p=pose*vector(local[k]);
        world[k]={p.x(),p.y(),p.z()};
    }
    return world;
}

void sweep_triangles(const Triangle& a,const Triangle& b,const SweepBody& ma,const SweepBody& mb,
                     std::size_t ia,std::size_t ib,std::size_t fa,std::size_t fb,double tolerance,
                     MeshSweepResult& out) {
    const auto array = [](const V& v) -> Vec { return {v.x(),v.y(),v.z()}; };
    if ((ma.start.linear().array()==ma.end.linear().array()).all() &&
        (mb.start.linear().array()==mb.end.linear().array()).all()) {
        ++out.translation_tests;
        const auto hit = translated_triangle_ccd(world_triangle(a,ma.start),world_triangle(b,mb.start),
            array(ma.start.translation()),array(ma.end.translation()),array(mb.start.translation()),array(mb.end.translation()));
        if (std::isfinite(hit.time) && (!out.first || hit.time < out.first->fraction))
            out.first=ImpactCandidate{ia,ib,fa,fb,hit.time,hit.output_tolerance};
        return;
    }
    struct TimeRange { double lo,hi; };
    std::vector<TimeRange> pending{{0.,out.first ? out.first->fraction : 1.}};
    while (!pending.empty()) {
        const auto [lo,hi]=pending.back();
        pending.pop_back();
        if (out.first && lo >= out.first->fraction) continue;
        const double mid=std::midpoint(lo,hi);
        const auto distance=squared_distance_bounds(world_triangle(a,ma.motion.at(mid)),world_triangle(b,mb.motion.at(mid)));
        const auto excursion=[&](const SweepBody& body) {
            return (body.end.translation()-body.start.translation()).norm()*(hi-lo)*.5 +
                   2*body.radius*std::sin(body.motion.angle*(hi-lo)*.25)+body.padding;
        };
        const double movement=nc::up(excursion(ma)+excursion(mb));
        if (distance.first>nc::up(movement*movement)) continue;
        const double at_start_bound=nc::up(std::sqrt(std::max(0.,distance.second))+movement);
        if (at_start_bound<=tolerance) {
            out.first=ImpactCandidate{ia,ib,fa,fb,lo,at_start_bound};
            return;
        }
        const double chord=ma.motion.chord_error(ma.radius,hi-lo)+mb.motion.chord_error(mb.radius,hi-lo);
        const double margin=chord+ma.padding+mb.padding;
        if (margin <= tolerance*.125) {
            ++out.linear_tests;
            const auto candidate=linear_triangle_ccd(world_triangle(a,ma.motion.at(lo)),world_triangle(a,ma.motion.at(hi)),
                world_triangle(b,mb.motion.at(lo)),world_triangle(b,mb.motion.at(hi)),margin,tolerance*.25);
            if (!std::isfinite(candidate.time)) continue;
            // The linear solver measures Cartesian box widths. Convert its
            // tolerance and minimum-separation padding to a Euclidean bound,
            // then allow the displacement from the chord to the rigid motion.
            const double proximity=nc::up(std::sqrt(3.)*(candidate.output_tolerance+margin)+margin);
            if (proximity<=tolerance) {
                const double fraction=lo+(hi-lo)*candidate.time;
                if (!out.first || fraction < out.first->fraction)
                    out.first=ImpactCandidate{ia,ib,fa,fb,fraction,proximity};
                return;
            }
        }
        if (mid==lo || mid==hi || ma.padding+mb.padding>=tolerance)
            throw std::runtime_error("insufficient spatial or temporal precision for rigid mesh CCD");
        pending.push_back({mid,hi});
        pending.push_back({lo,mid});
    }
}

template<class Box>
void sweep_pair(const Body& a,const Body& b,const SweepBody& ma,const SweepBody& mb,double tolerance,MeshSweepResult& out) {
    const auto& ta=*std::get<std::shared_ptr<Tree<Box>>>(a.tree);
    const auto& tb=*std::get<std::shared_ptr<Tree<Box>>>(b.tree);
    std::vector<std::pair<int,int>> pending{{0,0}};
    while (!pending.empty() && (!out.first || out.first->fraction>0.)) {
        const auto [ia,ib]=pending.back();
        pending.pop_back();
        const auto& na=ta.getBV(ia);
        const auto& nb=tb.getBV(ib);
        ++out.node_pairs;
        if (!ma.envelope(na.bv).overlap(mb.envelope(nb.bv))) continue;
        if (na.isLeaf() && nb.isLeaf()) {
            ++out.triangle_pairs;
            const auto fa=static_cast<std::size_t>(na.primitiveId()), fb=static_cast<std::size_t>(nb.primitiveId());
            sweep_triangles(triangle(*a.mesh,fa,{}),triangle(*b.mesh,fb,{}),ma,mb,
                            a.proxy->index,b.proxy->index,fa,fb,tolerance,out);
        } else if (nb.isLeaf() || (!na.isLeaf() && na.num_primitives >= nb.num_primitives)) {
            pending.emplace_back(na.leftChild(),ib);
            pending.emplace_back(na.rightChild(),ib);
        } else {
            pending.emplace_back(ia,nb.leftChild());
            pending.emplace_back(ia,nb.rightChild());
        }
    }
}
}

MeshSweepResult MeshScene::sweep(std::span<const RigidPlacement> start,std::span<const RigidPlacement> end,
                                 double tolerance) {
    auto& state=*impl_;
    if (!(tolerance>0 && std::isfinite(tolerance)) || start.size()!=state.bodies.size() || end.size()!=start.size())
        throw std::invalid_argument("expected one start and end pose per body and a positive CCD tolerance");
    std::vector<SweepBody> motion;
    for (std::size_t i=0;i<start.size();++i) {
        auto& body=state.bodies[i];
        const auto from=transform(start[i]), to=transform(end[i]);
        const double radius=body.local.min_.cwiseAbs().cwiseMax(body.local.max_.cwiseAbs()).norm();
        const double padding=std::max(roundoff(body.local,from),roundoff(body.local,to));
        MeshMotion path(from,to);
        motion.push_back({path,from,to,radius,padding,path.chord_error(radius,1.)});
        body.proxy->bounds(motion.back().envelope(body.local));
    }
    MeshSweepResult result;
    auto pairs=state.candidates();
    result.body_pairs=pairs.size();
    for (const auto& [i,j] : pairs) {
        if (state.method==MeshMethod::obb)
            sweep_pair<Oriented>(state.bodies[i],state.bodies[j],motion[i],motion[j],tolerance,result);
        else sweep_pair<Bounds>(state.bodies[i],state.bodies[j],motion[i],motion[j],tolerance,result);
    }
    return result;
}
#endif
} // namespace nc::cpu
