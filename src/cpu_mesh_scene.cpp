#include "nurbscontact/cpu_mesh.hpp"
#include "cpu_mesh_solid.hpp"

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
        const auto id = representatives[component];
        const V p = relative * vector(a.mesh->vertices[id]);
        if (b.local.contain(p) && b.boundary->contains({p.x(),p.y(),p.z()}))
            result.enclosed.push_back({a.proxy->index,b.proxy->index,id});
    }
}

template<class Box>
void intersect_pair(const Body& a, const Body& b, std::span<const RigidPlacement> poses,
                    IntersectionResult& result) {
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
            auto points = intersect_triangles(triangle(*a.mesh, fa, poses[i]), triangle(*b.mesh, fb, poses[j]));
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

IntersectionResult MeshScene::intersect(std::span<const RigidPlacement> poses) {
    auto& state = *impl_;
    if (poses.size() != state.bodies.size()) throw std::invalid_argument("one pose required per body");
    for (std::size_t i = 0; i < poses.size(); ++i) {
        auto& body = state.bodies[i];
        body.world = transform(poses[i]);
        body.padding = roundoff(body.local, body.world);
        body.proxy->bounds(placed(body.local,body.world,body.padding));
    }
    Pairs candidates;
    if (!state.manager) {
        candidates=sweep_and_prune(state.bodies,state.order);
    } else if (!state.registered) {
        for (const auto& body : state.bodies) state.manager->registerObject(body.proxy.get());
        state.manager->setup();
        state.registered = true;
    } else {
        state.manager->update();
    }
    if (state.manager) state.manager->collide(&candidates, collect);
    std::sort(candidates.begin(), candidates.end());
    IntersectionResult result;
    result.body_pairs = candidates.size();
    for (const auto& [a,b] : candidates) {
        const auto before = result.intersections.size();
        if (state.method == MeshMethod::obb) intersect_pair<Oriented>(state.bodies[a],state.bodies[b],poses,result);
        else intersect_pair<Bounds>(state.bodies[a],state.bodies[b],poses,result);
        const auto hits = std::span<const Intersection>(result.intersections).subspan(before);
        enclosed(state.bodies[a],state.bodies[b],hits,true,result);
        enclosed(state.bodies[b],state.bodies[a],hits,false,result);
    }
    std::sort(result.intersections.begin(), result.intersections.end(), [](const Intersection& a, const Intersection& b) {
        return std::tie(a.body_a,a.body_b,a.triangle_a,a.triangle_b) < std::tie(b.body_a,b.body_b,b.triangle_a,b.triangle_b);
    });
    return result;
}
} // namespace nc::cpu
