#include "nurbscontact/cpu_mesh.hpp"

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Surface_mesh.h>

namespace nc::cpu {
namespace {
using namespace nc::dynamics;
namespace pmp = CGAL::Polygon_mesh_processing;
using K = CGAL::Exact_predicates_exact_constructions_kernel;
using Surface = CGAL::Surface_mesh<K::Point_3>;
using Face = Surface::Face_index;
using Owner = Surface::Property_map<Face,int>;

Surface solid(const MeshData& mesh) {
    Surface result;
    for (const Vec p : mesh.vertices) result.add_vertex({p[0],p[1],p[2]});
    for (const auto& f : mesh.faces) {
        using Index = Surface::Vertex_index;
        if (result.add_face(Index(f[0]),Index(f[1]),Index(f[2]))==Surface::null_face())
            throw std::invalid_argument("invalid solid mesh connectivity");
    }
    if (!CGAL::is_closed(result) || pmp::does_self_intersect(result) ||
        !pmp::does_bound_a_volume(result) || !pmp::is_outward_oriented(result))
        throw std::invalid_argument("contact response requires a closed outward solid boundary with correctly oriented cavities");
    return result;
}

RigidPlacement placement(ContactPose pose) {
    RigidPlacement result;
    result.origin=pose.origin;
    for (Vec& axis : result.basis) axis=pose.rotate(axis);
    return result;
}

Surface placed(const Surface& source,const MeshData& mesh,const RigidPlacement& pose) {
    Surface result=source;
    for (auto v : result.vertices()) {
        const Vec p=pose.point(mesh.vertices[v.idx()]);
        result.point(v)={p[0],p[1],p[2]};
    }
    return result;
}

// Track which solid contributes each piece of the overlap boundary.
// Coincident, equally oriented faces carry half ownership on each side;
// their contribution to the difference of the two area vectors is zero.
struct SourceFaces : pmp::Corefinement::Default_visitor<Surface> {
    const Surface* a;
    Owner first, second, output;
    void subface_of_coplanar_faces_intersection(Face f,const Surface& source) {
        (&source==a ? first : second)[f]=0;
    }
    void after_face_copy(Face f,const Surface& source,Face g,const Surface&) {
        output[g]=(&source==a ? first : second)[f];
    }
};

Vec point(const Surface& surface,Surface::Vertex_index v) {
    const auto& p=surface.point(v);
    return {CGAL::to_double(p.x()),CGAL::to_double(p.y()),CGAL::to_double(p.z())};
}

std::vector<ContactRegion> overlap_regions(Surface& a,Surface& b) {
    Surface overlap;
    auto owner=overlap.add_property_map<Face,int>("f:source",0).first;
    SourceFaces source{{},&a,a.add_property_map<Face,int>("f:source",1).first,
        b.add_property_map<Face,int>("f:source",-1).first,owner};
    if (!pmp::corefine_and_compute_intersection(a,b,overlap,CGAL::parameters::visitor(source)))
        throw UnresolvedContact("overlap has a non-manifold boundary");
    if (overlap.is_empty()) return {};
    auto id=overlap.add_property_map<Face,std::size_t>("f:volume",0).first;
    const auto count=pmp::volume_connected_components(overlap,id);
    std::vector<std::vector<Face>> components(count);
    for (Face f : overlap.faces()) components.at(id[f]).push_back(f);

    std::vector<ContactRegion> regions;
    for (const auto& faces : components) {
        if (faces.empty()) continue;
        const Vec origin=point(overlap,overlap.target(overlap.halfedge(faces.front())));
        Vec area{},moment{};
        double volume6=0,area_sum=0;
        for (const Face f : faces) {
            const auto h=overlap.halfedge(f);
            const Vec x=sub(point(overlap,overlap.source(h)),origin);
            const Vec y=sub(point(overlap,overlap.target(h)),origin);
            const Vec z=sub(point(overlap,overlap.target(overlap.next(h))),origin);
            const Vec cross=cross3(sub(y,x),sub(z,x));
            area=add(area,scale(cross,owner[f]));
            area_sum+=length(cross);
            const double v=dot3(x,cross3(y,z));
            volume6+=v;
            moment=add(moment,scale(add(x,add(y,z)),v));
        }
        const double magnitude=length(area);
        if (!(volume6>0 && magnitude>64*std::numeric_limits<double>::epsilon()*area_sum))
            throw UnresolvedContact("overlap has no resolved separation normal (enclosure or degenerate volume)");
        const Vec normal=scale(area,1/magnitude);
        double lo=infinity,hi=-infinity;
        for (Face f : faces) {
            for (auto v : CGAL::vertices_around_face(overlap.halfedge(f),overlap)) {
                const double height=dot3(sub(point(overlap,v),origin),normal);
                lo=std::min(lo,height);
                hi=std::max(hi,height);
            }
        }
        regions.push_back({add(origin,scale(moment,.25/volume6)),normal,hi-lo});
    }
    return regions;
}

class MeshRegions final : public PairRegionQuery {
public:
    MeshRegions(std::shared_ptr<const MeshData> a,std::shared_ptr<const MeshData> b,MeshMethod method,
                std::array<Vec,2> centers)
        : meshes_{std::move(a),std::move(b)},centers_(centers),scene_(meshes_,method),
          solids_{std::make_shared<Surface>(solid(*meshes_[0])),nullptr} {
        solids_[1]=meshes_[0]==meshes_[1] ? solids_[0] : std::make_shared<Surface>(solid(*meshes_[1]));
    }
    std::vector<ContactRegion> regions(ContactPose a,ContactPose b,ContactWork& work) const override {
        const auto poses=placements(a,b);
        const auto hit=scene_.overlaps(poses);
        work.mesh.triangles+=hit.triangle_tests;
        work.mesh.nodes+=hit.node_pairs;
        if (!hit.overlap) return {};
        auto first=placed(*solids_[0],*meshes_[0],poses[0]);
        auto second=placed(*solids_[1],*meshes_[1],poses[1]);
        return overlap_regions(first,second);
    }
    SurfaceTransition transition(ContactPose a0,ContactPose b0,ContactPose a1,ContactPose b1,
                             double tolerance) const override {
#ifdef NC_WITH_MESH_CCD
        const auto start=placements(a0,b0),end=placements(a1,b1);
        const auto initial=scene_.overlaps(start);
        const auto last=scene_.overlaps(end);
        SurfaceTransition result;
        result.start_in_contact=initial.overlap;
        result.end_in_contact=last.overlap;
        result.nodes=initial.node_pairs+last.node_pairs;
        if (result.start_in_contact && result.end_in_contact) return result;
        const auto sweep=result.start_in_contact ? scene_.sweep(end,start,tolerance) : scene_.sweep(start,end,tolerance);
        result.nodes+=sweep.node_pairs;
        if (sweep.first) result.fraction=result.start_in_contact ? 1.-sweep.first->fraction : sweep.first->fraction;
        else if (result.start_in_contact!=result.end_in_contact)
            throw std::runtime_error("mesh CCD missed a change of contact state");
        return result;
#else
        throw std::invalid_argument("surface event detection requires NC_WITH_MESH_CCD");
#endif
    }
private:
    std::array<RigidPlacement,2> placements(ContactPose a,ContactPose b) const {
        a.origin=a.point(centers_[0]);
        b.origin=b.point(centers_[1]);
        return {placement(a),placement(b)};
    }
    std::array<std::shared_ptr<const MeshData>,2> meshes_;
    std::array<Vec,2> centers_;
    mutable MeshScene scene_;
    std::array<std::shared_ptr<const Surface>,2> solids_;
};
}

std::shared_ptr<const dynamics::PairRegionQuery> mesh_region_query(std::shared_ptr<const MeshData> a,
    std::shared_ptr<const MeshData> b,MeshMethod method,std::array<Vec,2> centers) {
    const auto recenter=[](std::shared_ptr<const MeshData> mesh,Vec center) -> std::shared_ptr<const MeshData> {
        if (center==Vec{}) return mesh;
        auto local=std::make_shared<MeshData>(*mesh);
        for (Vec& point:local->vertices) point=sub(point,center);
        return local;
    };
    auto first=recenter(a,centers[0]);
    auto second=(a==b && centers[0]==centers[1]) ? first : recenter(b,centers[1]);
    return std::make_shared<MeshRegions>(std::move(first),std::move(second),method,centers);
}
} // namespace nc::cpu
