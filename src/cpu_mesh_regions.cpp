#include "nurbscontact/cpu_mesh.hpp"
#include "cpu_mesh_features.hpp"

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_mesh_processing/manifoldness.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Surface_mesh.h>
#include <chrono>

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

Surface placed(const Surface& source,std::span<const Vec> world,std::span<const std::size_t> ids={}) {
    Surface result=source;
    for (auto v : result.vertices()) {
        const Vec p=world[ids.empty() ? v.idx() : ids[v.idx()]];
        result.point(v)={p[0],p[1],p[2]};
    }
    return result;
}

struct WorldMesh {
    std::vector<Vec> points;
    Vec lo{},hi{};
    void place(const MeshData& mesh,const RigidPlacement& pose) {
        points.resize(mesh.vertices.size());
        lo={infinity,infinity,infinity}; hi={-infinity,-infinity,-infinity};
        for (std::size_t i=0;i<points.size();++i) {
            points[i]=pose.point(mesh.vertices[i]);
            for (std::size_t j=0;j<3;++j) {
                lo[j]=std::min(lo[j],points[i][j]);
                hi[j]=std::max(hi[j],points[i][j]);
            }
        }
    }
};

struct Cut {
    unsigned axis{};
    int sign{1};
    double height(Vec p) const { return sign*p[axis]; }
    double minimum(const WorldMesh& w) const { return sign>0 ? w.lo[axis] : -w.hi[axis]; }
    double maximum(const WorldMesh& w) const { return sign>0 ? w.hi[axis] : -w.lo[axis]; }
};

// Original triangles on the contact side are kept whole. The replacement faces
// use a single vertex from inside the far-side bounding box to close one loop.
struct LocalSolid {
    Surface topology;
    std::vector<std::size_t> retained,discarded;
    Cut cut;
    std::optional<Vec> cap;

    bool usable(const WorldMesh& own,const WorldMesh& other,const RigidPlacement& pose) const {
        const double limit=cut.minimum(other);
        for (auto i:discarded) if (!(cut.height(own.points[i])<limit)) return false;
        if (cap) {
            const Vec tip=pose.point(*cap);
            // All other vertices are original. Keeping the cap inside the original
            // world AABB makes two independently localized bodies compatible.
            for (unsigned j=0;j<3;++j) if (!(own.lo[j]<=tip[j] && tip[j]<=own.hi[j])) return false;
            if (!(cut.height(tip)<limit)) return false;
        }
        return true;
    }

    Surface placed_mesh(WorldMesh& world,const RigidPlacement& pose) const {
        if (cap) world.points.push_back(pose.point(*cap));
        auto result=placed(topology,world.points,retained);
        if (cap) world.points.pop_back();
        return result;
    }
};

std::optional<LocalSolid> local_solid(const MeshData& mesh,const WorldMesh& world,Cut cut,double level) {
    using Index=Surface::Vertex_index;
    LocalSolid part;
    part.cut=cut;
    std::vector<Index> indices(mesh.vertices.size(),Surface::null_vertex());
    std::vector<bool> discarded(mesh.vertices.size());
    for (const auto& f:mesh.faces) {
        if (std::all_of(f.begin(),f.end(),[&](auto i){return cut.height(world.points[i])<level;})) {
            for (auto i:f) discarded[i]=true;
            continue;
        }
        for (auto i:f) if (indices[i]==Surface::null_vertex()) {
            const auto& p=mesh.vertices[i];
            indices[i]=part.topology.add_vertex({p[0],p[1],p[2]});
            part.retained.push_back(i);
        }
        if (part.topology.add_face(indices[f[0]],indices[f[1]],indices[f[2]])==Surface::null_face()) return {};
    }
    if (part.topology.is_empty() || part.topology.number_of_faces()>.8*mesh.faces.size()) return {};
    for (std::size_t i=0;i<discarded.size();++i) if (discarded[i]) part.discarded.push_back(i);
    std::vector<Surface::Halfedge_index> border;
    for (auto h:part.topology.halfedges()) if (part.topology.is_border(h)) border.push_back(h);
    if (!border.empty()) {
        std::vector<std::pair<Index,Index>> edges;
        std::vector<bool> used(part.retained.size());
        for (auto h:CGAL::halfedges_around_face(border.front(),part.topology)) {
            Index a=part.topology.source(h),b=part.topology.target(h);
            if (used[a.idx()] || !discarded[part.retained[a.idx()]]) return {};
            used[a.idx()]=true;
            edges.emplace_back(a,b);
        }
        if (edges.size()!=border.size()) return {};
        // Average far-side source vertices to avoid a cap tip in an end face.
        Vec tip{};
        std::size_t count=0;
        const double far=cut.minimum(world)+.15*(level-cut.minimum(world));
        for (auto i:part.discarded) if (cut.height(world.points[i])<=far) {
            ++count;
            tip=add(tip,scale(sub(mesh.vertices[i],tip),1./count));
        }
        if (!count) return {};
        part.cap=tip;
        const Index apex=part.topology.add_vertex({tip[0],tip[1],tip[2]});
        part.retained.push_back(mesh.vertices.size());
        for (const auto [a,b]:edges)
            if (part.topology.add_face(a,b,apex)==Surface::null_face()) return {};
    }
    if (!CGAL::is_valid_polygon_mesh(part.topology) || !CGAL::is_closed(part.topology)) return {};
    std::vector<Surface::Halfedge_index> pinches;
    pmp::non_manifold_vertices(part.topology,std::back_inserter(pinches));
    if (!pinches.empty()) return {};
    for (auto f:part.topology.faces()) if (pmp::is_degenerate_triangle_face(f,part.topology)) return {};
    if (pmp::does_self_intersect(part.topology) || !pmp::does_bound_a_volume(part.topology) ||
        !pmp::is_outward_oriented(part.topology)) return {};
    return part;
}

struct LocalCache {
    std::optional<LocalSolid> part;
    std::optional<Vec> last_direction;
    double last_level{};

    const LocalSolid* select(const MeshData& mesh,const WorldMesh& own,const WorldMesh& other,
                             const RigidPlacement& pose,ContactWork& work) {
        if (mesh.faces.size()<128) return nullptr;
        if (part && part->usable(own,other,pose)) return &*part;
        Cut best;
        double best_fraction=.35;
        for (unsigned j=0;j<3;++j) for (int sign:{1,-1}) {
            Cut candidate{j,sign};
            const double width=candidate.maximum(own)-candidate.minimum(own);
            if (width<=0) continue;
            const double fraction=(candidate.minimum(other)-candidate.minimum(own))/width;
            if (fraction>best_fraction) { best_fraction=fraction; best=candidate; }
        }
        if (best_fraction==.35) return nullptr;
        const double width=best.maximum(own)-best.minimum(own);
        const double limit=best.minimum(other);
        Vec direction{};
        for (unsigned i=0;i<3;++i) direction[i]=best.sign*pose.basis[i][best.axis];
        const double local_level=limit-best.height(pose.origin);
        if (last_direction && dot3(direction,*last_direction)>.995 && std::abs(local_level-last_level)<.05*width)
            return nullptr;
        last_direction=direction;
        last_level=local_level;
        // Moving the cut farther from the overlap can join isolated strips
        // into a single ring. Every trial still passes the same solid tests.
        for (double margin:{.12,.18,.24,.30}) {
            ++work.local_solid_builds;
            part=local_solid(mesh,own,best,limit-margin*width);
            if (part && part->usable(own,other,pose)) return &*part;
        }
        return nullptr;
    }
};

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
    return {CGAL::to_double(CGAL::exact(p.x())),CGAL::to_double(CGAL::exact(p.y())),
            CGAL::to_double(CGAL::exact(p.z()))};
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
    // The Boolean may enumerate the same faces and vertices in different orders.
    // Convert each exact witness once and integrate with a geometry-based origin
    // and order, so repeated rigid dynamics trials see consistent moments.
    auto coordinates=overlap.add_property_map<Surface::Vertex_index,Vec>("v:coordinates").first;
    for (auto v:overlap.vertices()) coordinates[v]=point(overlap,v);
    using Facet=std::pair<std::array<Vec,3>,int>;
    std::vector<std::vector<Facet>> components(count);
    for (Face f : overlap.faces()) {
        const auto h=overlap.halfedge(f);
        std::array<Vec,3> corners{coordinates[overlap.source(h)],coordinates[overlap.target(h)],
                                  coordinates[overlap.target(overlap.next(h))]};
        std::rotate(corners.begin(),std::min_element(corners.begin(),corners.end()),corners.end());
        components.at(id[f]).emplace_back(corners,owner[f]);
    }

    std::vector<ContactRegion> regions;
    for (auto& faces : components) {
        if (faces.empty()) continue;
        std::sort(faces.begin(),faces.end());
        const Vec origin=faces.front().first.front();
        Vec area{},moment{};
        double volume6=0,area_sum=0;
        for (const auto& [corners,side] : faces) {
            const Vec x=sub(corners[0],origin), y=sub(corners[1],origin), z=sub(corners[2],origin);
            const Vec cross=cross3(sub(y,x),sub(z,x));
            area=add(area,scale(cross,side));
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
        for (const auto& f : faces) {
            for (Vec v : f.first) {
                const double height=dot3(sub(v,origin),normal);
                lo=std::min(lo,height);
                hi=std::max(hi,height);
            }
        }
        regions.push_back({add(origin,scale(moment,.25/volume6)),normal,hi-lo});
    }
    std::sort(regions.begin(),regions.end(),[](const auto& a,const auto& b) {
        return std::tie(a.point,a.normal,a.depth_m)<std::tie(b.point,b.normal,b.depth_m);
    });
    return regions;
}

class MeshRegions final : public PairRegionQuery {
public:
    MeshRegions(std::shared_ptr<const MeshData> a,std::shared_ptr<const MeshData> b,MeshMethod method,
                std::array<Vec,2> centers,MeshRegionMode mode,double max_depth)
        : meshes_{std::move(a),std::move(b)},centers_(centers),mode_(mode),scene_(meshes_,method,mode==MeshRegionMode::finite || mode==MeshRegionMode::contour_ray) {
        solids_[0]=std::make_shared<Surface>(solid(*meshes_[0]));
        solids_[1]=meshes_[0]==meshes_[1] ? solids_[0] : std::make_shared<Surface>(solid(*meshes_[1]));
        // The same one-time orientation and self-intersection checks apply to
        // leaf contact normals. Their query needs only the prepared mesh scene.
        if (mode_==MeshRegionMode::finite || mode_==MeshRegionMode::contour_ray)
            features_=std::make_unique<FiniteFeatures>(*meshes_[0],*meshes_[1],scene_,max_depth,mode_==MeshRegionMode::contour_ray);
        if (features_ || mode_==MeshRegionMode::fcl) solids_={};
    }
    std::size_t independent_mesh_faces() const override {
        return features_ ? meshes_[0]->faces.size()+meshes_[1]->faces.size() : 0;
    }
    std::vector<ContactRegion> regions(ContactPose a,ContactPose b,ContactWork& work) const override {
        using Clock=std::chrono::steady_clock;
        const auto start=Clock::now();
        const auto poses=placements(a,b);
        if (features_ || mode_==MeshRegionMode::fcl) {
            auto contacts=features_ ? features_->query(poses,work) : scene_.fcl_contacts(poses,work);
            work.mesh_detection_s+=std::chrono::duration<double>(Clock::now()-start).count();
            return contacts;
        }
        const auto hit=scene_.overlaps(poses);
        const auto detected=Clock::now();
        work.mesh_detection_s+=std::chrono::duration<double>(detected-start).count();
        work.mesh.triangles+=hit.triangle_tests;
        work.mesh.nodes+=hit.node_pairs;
        if (!hit.overlap) return {};
        for (unsigned i=0;i<2;++i) world_[i].place(*meshes_[i],poses[i]);
        bool shortened=false;
        const auto select=[&](unsigned i) {
            if (mode_==MeshRegionMode::local_volume)
                if (const auto* part=cache_[i].select(*meshes_[i],world_[i],world_[1-i],poses[i],work)) {
                    ++work.local_solid_uses;
                    shortened=true;
                    return part->placed_mesh(world_[i],poses[i]);
                }
            return placed(*solids_[i],world_[i].points);
        };
        auto first=select(0),second=select(1);
        ++work.boolean_queries;
        work.boolean_input_faces+=first.number_of_faces()+second.number_of_faces();
        const auto positioned=Clock::now();
        work.mesh_placement_s+=std::chrono::duration<double>(positioned-detected).count();
        std::vector<ContactRegion> result;
        try {
            result=overlap_regions(first,second);
        } catch (const UnresolvedContact&) {
            if (!shortened) throw;
            first=placed(*solids_[0],world_[0].points);
            second=placed(*solids_[1],world_[1].points);
            ++work.boolean_queries;
            work.boolean_input_faces+=first.number_of_faces()+second.number_of_faces();
            result=overlap_regions(first,second);
        }
        work.mesh_boolean_s+=std::chrono::duration<double>(Clock::now()-positioned).count();
        return result;
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
    MeshRegionMode mode_;
    mutable std::array<WorldMesh,2> world_;
    mutable std::array<LocalCache,2> cache_;
    mutable MeshScene scene_;
    std::unique_ptr<FiniteFeatures> features_;
    std::array<std::shared_ptr<const Surface>,2> solids_;
};
}

std::shared_ptr<const dynamics::PairRegionQuery> mesh_region_query(std::shared_ptr<const MeshData> a,
    std::shared_ptr<const MeshData> b,MeshMethod method,std::array<Vec,2> centers,bool localize) {
    return mesh_region_query(std::move(a),std::move(b),method,centers,
                             localize ? MeshRegionMode::local_volume : MeshRegionMode::full_volume);
}

std::shared_ptr<const dynamics::PairRegionQuery> mesh_region_query(std::shared_ptr<const MeshData> a,
    std::shared_ptr<const MeshData> b,MeshMethod method,std::array<Vec,2> centers,MeshRegionMode mode,double max_depth) {
    const auto recenter=[](std::shared_ptr<const MeshData> mesh,Vec center) -> std::shared_ptr<const MeshData> {
        if (center==Vec{}) return mesh;
        auto local=std::make_shared<MeshData>(*mesh);
        for (Vec& point:local->vertices) point=sub(point,center);
        return local;
    };
    auto first=recenter(a,centers[0]);
    auto second=(a==b && centers[0]==centers[1]) ? first : recenter(b,centers[1]);
    return std::make_shared<MeshRegions>(std::move(first),std::move(second),method,centers,mode,max_depth);
}
} // namespace nc::cpu
