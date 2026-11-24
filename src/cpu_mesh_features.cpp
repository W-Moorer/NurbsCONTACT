#include "cpu_mesh_features.hpp"
#include "cpu_mesh_contours.hpp"

#include <numeric>
#include <unordered_map>
#include <unordered_set>

#include <algorithm>
#include <map>
#include <ranges>
#include <tuple>

namespace nc::cpu {
namespace {
using namespace dynamics;
using Face=std::array<std::size_t,3>;
struct Edge {
    std::size_t a,b,first,second;
};
struct SurfaceFeatures {
    const MeshData& mesh;
    std::vector<Vec> normals;
    std::vector<Edge> edges;
    std::vector<Face> face_edges;
    std::vector<std::size_t> panel;
    std::vector<std::vector<std::size_t>> panel_faces;
    std::vector<std::vector<std::size_t>> ridges_at;
    std::vector<std::size_t> edge_ridge;
    std::vector<std::array<std::size_t,2>> ridges;
    std::vector<std::vector<Vec>> rays;
    std::vector<std::vector<std::size_t>> neighbors;
    std::vector<std::size_t> surface_vertices;
    std::vector<std::size_t> planes;
    Vec lo{},hi{};
    double radius{};
    bool convex{true};
    bool connected{};
    explicit SurfaceFeatures(const MeshData& m):mesh(m),face_edges(m.faces.size()),rays(m.vertices.size()),neighbors(m.vertices.size()) {
        std::map<std::pair<std::size_t,std::size_t>,std::size_t> lookup;
        for (std::size_t i=0;i<m.faces.size();++i) {
            const auto f=m.faces[i];
            const Vec n=cross3(sub(m.vertices[f[1]],m.vertices[f[0]]),sub(m.vertices[f[2]],m.vertices[f[0]]));
            normals.push_back(scale(n,1/length(n)));
            for (int j=0;j<3;++j) {
                const auto a=f[j],b=f[(j+1)%3];
                auto [it,fresh]=lookup.emplace(std::minmax(a,b),edges.size());
                if (fresh) {
                    edges.push_back({a,b,i,m.faces.size()});
                    Vec d=sub(m.vertices[b],m.vertices[a]);
                    d=scale(d,1/length(d));
                    rays[a].push_back(d);
                    rays[b].push_back(scale(d,-1));
                    neighbors[a].push_back(b);
                    neighbors[b].push_back(a);
                } else edges[it->second].second=i;
                face_edges[i][j]=it->second;
            }
        }
        lo=hi=m.vertices[m.faces.front().front()];
        for (std::size_t i=0;i<m.vertices.size();++i) if (!neighbors[i].empty()) {
            surface_vertices.push_back(i);
            const auto p=m.vertices[i];
            radius=std::max(radius,length(p));
            for (int k=0;k<3;++k) {lo[k]=std::min(lo[k],p[k]); hi[k]=std::max(hi[k],p[k]);}
        }
        std::vector<bool> visited(m.faces.size());
        std::vector<std::size_t> pending{0};
        visited[0]=true;
        for (std::size_t next=0;next<pending.size();++next) {
            const auto face=pending[next];
            for (auto id:face_edges[face]) {
                const auto& e=edges[id];
                const auto other=e.first==face ? e.second : e.first;
                if (other>=m.faces.size()) throw std::invalid_argument("open contact mesh");
                if (!visited[other]) {visited[other]=true; pending.push_back(other);}
            }
        }
        connected=pending.size()==m.faces.size();
        convex=connected;
        for (const auto& e:edges) {
            if (e.second==m.faces.size()) throw std::invalid_argument("open contact mesh");
            const auto other=m.faces[e.second];
            for (auto v:other) {
                if (v==e.a || v==e.b) continue;
                if (dot3(normals[e.first],sub(m.vertices[v],m.vertices[e.a]))>2e-12*radius) convex=false;
            }
        }
        if (convex) for (std::size_t i=0;i<m.faces.size();++i) {
            if (std::ranges::none_of(planes,[&](auto j){return squared(sub(normals[i],normals[j]))<1e-24;}))
                planes.push_back(i);
            // A direct half-space check is economical only for few-plane hulls.
            if (planes.size()>32) {planes.clear(); break;}
        }
    }
    void build_support_topology() {
        // A panel is one connected, coplanar piece. Parallel faces separated by
        // a gap retain different IDs, even if their supporting planes coincide.
        panel.assign(mesh.faces.size(),mesh.faces.size());
        for (std::size_t f=0;f<mesh.faces.size();++f) {
            if (panel[f]!=mesh.faces.size()) continue;
            const auto id=panel_faces.size();
            auto& members=panel_faces.emplace_back();
            members.push_back(f); panel[f]=id;
            const Vec origin=mesh.vertices[mesh.faces[f][0]];
            for (std::size_t j=0;j<members.size();++j) {
                const auto current=members[j];
                for (auto index:face_edges[current]) {
                    const auto& edge=edges[index];
                    const auto adj=edge.first==current?edge.second:edge.first;
                    if (panel[adj]!=mesh.faces.size() || squared(sub(normals[f],normals[adj]))>1e-18) continue;
                    if (std::ranges::any_of(mesh.faces[adj],[&](auto v){
                        return std::abs(dot3(sub(mesh.vertices[v],origin),normals[f]))>radius*1e-12;
                    })) continue;
                    panel[adj]=id; members.push_back(adj);
                }
            }
        }
        std::vector<std::vector<std::size_t>> incident(mesh.vertices.size());
        for (std::size_t i=0;i<edges.size();++i) {
            const auto& e=edges[i];
            if (panel[e.first]==panel[e.second]) continue;
            incident[e.a].push_back(i); incident[e.b].push_back(i);
        }
        // Edges between the same two nonparallel planes lie on their common
        // line. Subdivision along that line must not create additional springs.
        ridges_at.resize(mesh.vertices.size());
        edge_ridge.assign(edges.size(),edges.size());
        std::vector<char> assigned(edges.size());
        for (std::size_t i=0;i<edges.size();++i) {
            const auto& seed=edges[i];
            const auto sides=std::minmax(panel[seed.first],panel[seed.second]);
            if (assigned[i] || sides.first==sides.second ||
                squared(cross3(normals[seed.first],normals[seed.second]))<1e-18) continue;
            const Vec origin=mesh.vertices[seed.a], dir=sub(mesh.vertices[seed.b],origin);
            std::vector<std::size_t> stack{i},vertices;
            assigned[i]=1;
            std::array<std::size_t,2> ends{seed.a,seed.b};
            double lo=0, hi=squared(dir);
            while (!stack.empty()) {
                edge_ridge[stack.back()]=ridges.size();
                const auto e=edges[stack.back()]; stack.pop_back();
                for (auto v:{e.a,e.b}) {
                    vertices.push_back(v);
                    const double t=dot3(dir,sub(mesh.vertices[v],origin));
                    if (t<lo) {lo=t; ends[0]=v;}
                    if (t>hi) {hi=t; ends[1]=v;}
                    for (auto index:incident[v]) {
                        const auto& other=edges[index];
                        if (!assigned[index] && sides==std::minmax(panel[other.first],panel[other.second])) {
                            assigned[index]=1;stack.push_back(index);
                        }
                    }
                }
            }
            std::sort(vertices.begin(),vertices.end());
            vertices.erase(std::unique(vertices.begin(),vertices.end()),vertices.end());
            for (auto v:vertices) ridges_at[v].push_back(ridges.size());
            ridges.push_back(ends);
        }
    }
};
Vec inverse_vector(const RigidPlacement& p,Vec v) {
    return {dot3(p.basis[0],v),dot3(p.basis[1],v),dot3(p.basis[2],v)};
}
RigidPlacement relative(const RigidPlacement& a,const RigidPlacement& b) {
    RigidPlacement r;
    r.origin=inverse_vector(a,sub(b.origin,a.origin));
    for (int k=0;k<3;++k) r.basis[k]=inverse_vector(a,b.basis[k]);
    return r;
}
double motion_bound(const RigidPlacement& a,const RigidPlacement& b,double radius) {
    double norm2=0;
    for (int k=0;k<3;++k) norm2+=squared(sub(a.basis[k],b.basis[k]));
    return length(sub(a.origin,b.origin))+radius*std::sqrt(norm2);
}
struct Geometry {
    Triangle tri;
    Vec n;
    Vec lo,hi;
    unsigned u{},v{},drop{};
    struct Side {double x,y,lower,edge_max,altitude,scale;};
    std::array<Side,3> sides{};
};
Geometry facet(const SurfaceFeatures& s,std::size_t id,const RigidPlacement& p) {
    const auto& f=s.mesh.faces[id];
    Geometry g{{p.point(s.mesh.vertices[f[0]]),p.point(s.mesh.vertices[f[1]]),p.point(s.mesh.vertices[f[2]])},
               p.rotate(s.normals[id]),{},{}};
    for (unsigned k=0;k<3;++k) {
        g.lo[k]=std::min({g.tri[0][k],g.tri[1][k],g.tri[2][k]});
        g.hi[k]=std::max({g.tri[0][k],g.tri[1][k],g.tri[2][k]});
        if (std::abs(g.n[k])>std::abs(g.n[g.drop])) g.drop=k;
    }
    g.u=(g.drop+1)%3; g.v=(g.drop+2)%3;
    const double sign=g.n[g.drop]>=0 ? 1. : -1.;
    for (int i=0;i<3;++i) {
        const auto& a=g.tri[i]; const auto& b=g.tri[(i+1)%3];
        const double x=sign*(a[g.v]-b[g.v]),y=sign*(b[g.u]-a[g.u]);
        // Use the minimum over all three vertices, including very thin facets.
        const auto project=[&](const Vec& p){return x*p[g.u]+y*p[g.v];};
        const double p0=project(a),p1=project(b),p2=project(g.tri[(i+2)%3]);
        const double edge_max=std::max(p0,p1);
        g.sides[i]={x,y,std::min({p0,p1,p2}),edge_max,p2-edge_max,std::abs(x)+std::abs(y)};
    }
    return g;
}

// The narrow phase already established intersection. Extract the common segment
// from the overlap of the two finite triangle/plane slices.
std::optional<std::pair<Vec,Vec>> transverse_segment(const Geometry& a,const Geometry& b,double eps) {
    Vec direction=cross3(a.n,b.n);
    const double norm=length(direction);
    // Coincident planar caps have area overlap. Their internal triangulation
    // does not contribute to the length of a transverse contact boundary.
    // Purely tangent or coplanar groups use the finite support fallback.
    if (norm<1e-10) return {};
    direction=scale(direction,1/norm);
    const auto slice=[&](const Geometry& f,const Geometry& plane) {
        std::array<double,3> heights;
        for (int i=0;i<3;++i) heights[i]=dot3(sub(f.tri[i],plane.tri[0]),plane.n);
        std::pair<double,double> band{INFINITY,-INFINITY};
        const auto include=[&](Vec p) {
            const double t=dot3(sub(p,a.tri[0]),direction);
            band={std::min(band.first,t),std::max(band.second,t)};
        };
        for (int i=0;i<3;++i) {
            const int j=(i+1)%3;
            if (std::abs(heights[i])<=eps) include(f.tri[i]);
            if ((heights[i]<0)!=(heights[j]<0))
                include(add(f.tri[i],scale(sub(f.tri[j],f.tri[i]),heights[i]/(heights[i]-heights[j]))));
        }
        return band;
    };
    const auto u=slice(a,b),v=slice(b,a);
    const double lo=std::max(u.first,v.first),hi=std::min(u.second,v.second);
    if (hi-lo<=eps) return {};
    const Vec transverse=cross3(direction,a.n);
    const double offset=dot3(sub(b.tri[0],a.tri[0]),b.n)/dot3(transverse,b.n);
    const Vec line=add(a.tri[0],scale(transverse,offset));
    return std::pair{add(line,scale(direction,lo)),add(line,scale(direction,hi))};
}

// A separating edge in either two-dimensional projection also separates
// the spatial triangles. Strict agreement in a shared coordinate plane proves
// overlap there; touching and roundoff-sensitive cases use the exact predicate.
int projected_relation(const Geometry& a,const Geometry& b,double allowance) {
    bool strict=true;
    for (const auto& side:a.sides) {
        const auto project=[&](const Vec& p){return side.x*p[a.u]+side.y*p[a.v];};
        const double upper=std::max({project(b.tri[0]),project(b.tri[1]),project(b.tri[2])});
        const double tol=allowance*side.scale;
        if (upper-side.lower < -tol) return -1;
        if (side.altitude<=tol || upper-side.edge_max<=tol) strict=false;
    }
    return strict ? 1 : 0;
}
double segment_distance2(Vec p,Vec a,Vec b) {
    Vec ab=sub(b,a);
    return squared(sub(sub(p,a),scale(ab,std::clamp(dot3(sub(p,a),ab)/squared(ab),0.,1.))));
}
bool inside(const Geometry& face,Vec point) {
    for (int i=0;i<3;++i) {
        const Vec edge=sub(face.tri[(i+1)%3],face.tri[i]);
        if (dot3(face.n,cross3(edge,sub(point,face.tri[i]))) < -1e-13*squared(edge)) return false;
    }
    return true;
}
double point_distance2(const Geometry& face,Vec p) {
    if (inside(face,p)) {
        const double gap=dot3(sub(p,face.tri[0]),face.n);
        return gap*gap;
    }
    return std::min({segment_distance2(p,face.tri[0],face.tri[1]),segment_distance2(p,face.tri[1],face.tri[2]),
                     segment_distance2(p,face.tri[2],face.tri[0])});
}
struct Segment {Vec a,b;};
Segment segment(const SurfaceFeatures& s,std::size_t id,const RigidPlacement& p) {
    const auto& e=s.edges[id];
    return {p.point(s.mesh.vertices[e.a]),p.point(s.mesh.vertices[e.b])};
}
struct EdgePair {
    Vec normal,x,y;
    bool interior{};
};
EdgePair common_normal(Segment a,Segment b) {
    Vec u=sub(a.b,a.a),v=sub(b.b,b.a),offset=sub(b.a,a.a);
    const Vec cross=cross3(u,v);
    const double n2=squared(cross);
    if (n2<=1e-20*squared(u)*squared(v)) return {};
    const double s=dot3(cross3(offset,v),cross)/n2,t=dot3(cross3(offset,u),cross)/n2;
    if (s<0 || s>1 || t<0 || t>1) return {};
    return {scale(cross,1/std::sqrt(n2)),add(a.a,scale(u,s)),add(b.a,scale(v,t)),true};
}
double edges_distance2(Segment a,Segment b) {
    const auto q=common_normal(a,b);
    if (q.interior) return squared(sub(q.x,q.y));
    return std::min({segment_distance2(a.a,b.a,b.b),segment_distance2(a.b,b.a,b.b),
                     segment_distance2(b.a,a.a,a.b),segment_distance2(b.b,a.a,a.b)});
}
// The two incident triangles must lie behind the support plane of a convex edge.
bool supports_edge(const SurfaceFeatures& s,std::size_t id,Vec direction) {
    const auto& e=s.edges[id];
    const Vec u=sub(s.mesh.vertices[e.b],s.mesh.vertices[e.a]);
    const double tol=1e-9*length(u);
    const Vec n=s.normals[e.first],m=s.normals[e.second];
    return dot3(direction,cross3(n,u))<=tol && dot3(direction,cross3(u,m))<=tol &&
           dot3(direction,add(n,m))>0;
}
struct FeaturePair {
    unsigned kind;
    std::size_t a,b;
    auto operator<=>(const FeaturePair&) const=default;
};
using FacePair=std::array<std::size_t,2>;
using FaceGroup=std::vector<FacePair>;
struct PatchTopology {
    std::array<std::vector<std::size_t>,2> faces;
    std::vector<FeaturePair> features;
};

// Mesh indices are dense. Only rows touched by this contact patch are reset.
struct PairRows {
    std::vector<std::vector<std::size_t>> rows;
    std::vector<std::size_t> touched;
    std::unordered_map<std::size_t,std::unordered_set<std::size_t>> large_rows;
    bool insert(std::size_t a,std::size_t b) {
        auto& row=rows[a];
        if (row.size()<32) {
            if (std::ranges::find(row,b)!=row.end()) return false;
        } else {
            auto [it,fresh]=large_rows.try_emplace(a);
            if (fresh) it->second.insert(row.begin(),row.end());
            if (!it->second.insert(b).second) return false;
        }
        if (row.empty()) touched.push_back(a);
        row.push_back(b);
        return true;
    }
    void clear() {
        for (auto a:touched) rows[a].clear();
        touched.clear();
        large_rows.clear();
    }
};
}

struct FiniteFeatures::Data {
    std::array<SurfaceFeatures,2> surfaces;
    MeshScene& scene;
    std::array<std::unique_ptr<TriangleBvh>,2> ray_trees;
    double reach,skin,limit,initial_reach;
    bool initialized{};
    RigidPlacement saved;
    std::vector<FeaturePair> list;
    std::vector<std::array<std::size_t,2>> nearby;
    std::vector<Geometry> fixed_faces,moving_faces;
    std::vector<std::size_t> versions;
    std::size_t version{};
    std::array<std::size_t,2> last_plane{};
    PairRows face_rows;
    std::array<PairRows,3> feature_rows;
    std::map<FaceGroup,PatchTopology> previous_patches;
    Data(const MeshData& a,const MeshData& b,MeshScene& scene_,double maximum,bool contour_ray)
        :surfaces{SurfaceFeatures(a),SurfaceFeatures(b)},scene(scene_) {
        const double radius=std::min(surfaces[0].radius,surfaces[1].radius);
        for (std::size_t i=0;i<a.faces.size();++i) fixed_faces.push_back(facet(surfaces[0],i,{}));
        if (contour_ray) {
            for (unsigned s=0;s<2;++s) {
                const auto& mesh=surfaces[s].mesh;
                std::vector<Triangle> triangles;
                triangles.reserve(mesh.faces.size());
                for (auto f:mesh.faces) triangles.push_back({mesh.vertices[f[0]],mesh.vertices[f[1]],mesh.vertices[f[2]]});
                ray_trees[s]=std::make_unique<TriangleBvh>(std::move(triangles));
            }
        }
        moving_faces.resize(b.faces.size());
        versions.resize(b.faces.size());
        limit=maximum>0 && std::isfinite(maximum) ? 4*maximum : radius*.08;
        initial_reach=std::min(limit,radius*.006);
        reach=initial_reach;
        skin=std::max(radius*.006,initial_reach*.25);
        if (!surfaces[0].convex || !surfaces[1].convex) {
            face_rows.rows.resize(a.faces.size());
            for (auto& surface:surfaces) surface.build_support_topology();
        }
        feature_rows[0].rows.resize(a.vertices.size());
        feature_rows[1].rows.resize(b.vertices.size());
        feature_rows[2].rows.resize(surfaces[0].edges.size());
    }
    void rebuild(const std::array<RigidPlacement,2>& poses,const RigidPlacement& rel,ContactWork& work) {
        const double band=reach+skin;
        const double band2=band*band;
        const auto ids=scene.near_pairs(poses,band,work);
        const std::array<RigidPlacement,2> local{RigidPlacement{},rel};
        list.clear();
        for (auto& rows:feature_rows) rows.clear();
        for (auto [a,b]:ids) {
            std::array<std::size_t,2> face{a,b};
            for (unsigned source=0;source<2;++source) {
                const auto target=1-source;
                const auto& f=target ? moving(face[target],rel) : fixed_faces[face[target]];
                for (auto id:surfaces[source].mesh.faces[face[source]]) {
                    if (feature_rows[source].insert(id,face[target]) &&
                        point_distance2(f,local[source].point(surfaces[source].mesh.vertices[id]))<=band2)
                        list.push_back({source,id,face[target]});
                }
            }
            for (auto i:surfaces[0].face_edges[a]) {
                const auto seg=segment(surfaces[0],i,local[0]);
                for (auto j:surfaces[1].face_edges[b])
                    if (feature_rows[2].insert(i,j) && edges_distance2(seg,segment(surfaces[1],j,local[1]))<=band2)
                        list.push_back({2,i,j});
            }
        }
        std::sort(list.begin(),list.end());
        saved=rel;
        initialized=true;
    }
    // Locate the connected part of a straight support ridge over one planar
    // target panel. Gaps in a nonconvex panel split the parameter intervals.
    std::optional<Vec> line_center(unsigned s,std::size_t ridge,std::size_t face,Vec on_source,
                                   const RigidPlacement& rel,double depth) {
        const unsigned t=1-s;
        const auto& source=surfaces[s]; const auto& target=surfaces[t];
        const auto ends=source.ridges[ridge];
        const RigidPlacement local=s?rel:RigidPlacement{};
        const Vec first=local.point(source.mesh.vertices[ends[0]]);
        const Vec delta=local.rotate(sub(source.mesh.vertices[ends[1]],source.mesh.vertices[ends[0]]));
        const double len=length(delta);
        const auto& plane=t?moving(face,rel):fixed_faces[face];
        const double tol=1e-12*(source.radius+target.radius+length(rel.origin));
        if (len<=tol || std::abs(dot3(delta,plane.n))>tol) return {};
        const double parameter=dot3(sub(on_source,first),delta)/(len*len);
        const Vec projected=add(first,scale(plane.n,depth));
        std::vector<std::pair<double,double>> intervals;
        for (auto member:target.panel_faces[target.panel[face]]) {
            const auto& tri=t?moving(member,rel):fixed_faces[member];
            double lo=0,hi=1;
            for (unsigned j=0;j<3 && lo<=hi;++j) {
                const Vec edge=sub(tri.tri[(j+1)%3],tri.tri[j]);
                const Vec inward=cross3(tri.n,edge);
                const double h=dot3(inward,sub(projected,tri.tri[j]));
                const double slope=dot3(inward,delta);
                const double eps=tol*length(edge);
                if (std::abs(slope)<=eps) {if(h < -eps) hi=-1;}
                else if (slope>0) lo=std::max(lo,-h/slope);
                else hi=std::min(hi,-h/slope);
            }
            if (lo<=hi) intervals.emplace_back(lo,hi);
        }
        std::sort(intervals.begin(),intervals.end());
        const double eps=tol/len;
        for (std::size_t i=0;i<intervals.size();) {
            auto [lo,hi]=intervals[i++];
            while (i<intervals.size() && intervals[i].first<=hi+eps)
                hi=std::max(hi,intervals[i++].second);
            if (lo-eps<=parameter && parameter<=hi+eps && (hi-lo)*len>tol)
                return add(add(first,scale(delta,.5*(lo+hi))),scale(plane.n,.5*depth));
        }
        return {};
    }
    ContactRegion witness(const RigidPlacement& rel,const std::vector<FeaturePair>& candidates,bool convex,double cutoff) {
        std::array<RigidPlacement,2> local{RigidPlacement{},rel};
        ContactRegion best{};
        FeaturePair winner{};
        if (convex) best.depth_m=std::numeric_limits<double>::infinity();
        const double epsilon=1e-13*std::min(surfaces[0].radius,surfaces[1].radius);
        const auto acceptable=[&](double d) {
            return d>=-epsilon && d<=cutoff && (convex ? d<best.depth_m : d>best.depth_m);
        };
        for (const auto& f:candidates) {
            if (f.kind<2) {
                const unsigned s=f.kind,t=1-s;
                const Vec p=local[s].point(surfaces[s].mesh.vertices[f.a]);
                const auto& face=t ? moving(f.b,rel) : fixed_faces[f.b];
                const double d=dot3(sub(face.tri[0],p),face.n);
                if (!acceptable(d) || !inside(face,p)) continue;
                const Vec n=inverse_vector(local[s],face.n);
                if (std::ranges::any_of(surfaces[s].rays[f.a],[&](Vec edge){return dot3(n,edge)<-1e-9;})) continue;
                best={add(p,scale(face.n,.5*d)),scale(face.n,s?1.:-1.),d};
                winner=f;
            } else {
                const auto q=common_normal(segment(surfaces[0],f.a,local[0]),segment(surfaces[1],f.b,local[1]));
                if (!q.interior) continue;
                const double gap=dot3(q.normal,sub(q.x,q.y)),d=std::abs(gap);
                if (!acceptable(d)) continue;
                const Vec n=scale(q.normal,gap>0 ? 1. : -1.);
                if (!supports_edge(surfaces[0],f.a,n) ||
                    !supports_edge(surfaces[1],f.b,inverse_vector(rel,scale(n,-1)))) continue;
                best={scale(add(q.x,q.y),.5),n,d};
                winner=f;
            }
        }
        if (!convex && best.depth_m>epsilon) {
            if (winner.kind<2) {
                const unsigned s=winner.kind;
                const Vec p=local[s].point(surfaces[s].mesh.vertices[winner.a]);
                std::optional<Vec> center;
                for (auto ridge:surfaces[s].ridges_at[winner.a]) {
                    const auto point=line_center(s,ridge,winner.b,p,rel,best.depth_m);
                    if (!point) continue;
                    if (center) return best; // A corner can support two independent line directions.
                    center=point;
                }
                if (center) best.point=*center;
            } else {
                for (unsigned s=0;s<2;++s) {
                    const unsigned t=1-s;
                    const auto edge=s?winner.b:winner.a;
                    const auto ridge=surfaces[s].edge_ridge[edge];
                    if (ridge>=surfaces[s].ridges.size()) continue;
                    const auto& target=surfaces[t].edges[s?winner.a:winner.b];
                    const Vec outward=scale(best.normal,s?1.:-1.);
                    const Vec p=sub(best.point,scale(outward,.5*best.depth_m));
                    for (auto face:{target.first,target.second}) {
                        if (squared(sub(local[t].rotate(surfaces[t].normals[face]),outward))>1e-18) continue;
                        if (const auto center=line_center(s,ridge,face,p,rel,best.depth_m)) {
                            best.point=*center; return best;
                        }
                    }
                }
            }
        }
        return best;
    }
    ContactRegion world(ContactRegion p,const RigidPlacement& pose) const {
        return {pose.point(p.point),pose.rotate(p.normal),p.depth_m};
    }
    // If the entire opposing footprint fits inside a convex support facet,
    // each connected cap has exactly the vertex-to-plane normal depth.
    std::optional<std::vector<ContactRegion>> planar_cap(unsigned side,const std::array<RigidPlacement,2>& poses) {
        const auto& base=surfaces[side];
        const auto& other=surfaces[1-side];
        const auto transform=relative(poses[side],poses[1-side]);
        const double eps=2e-12*(base.radius+other.radius);
        for (std::size_t k=0;k<base.planes.size();++k) {
            const auto index=(last_plane[side]+k)%base.planes.size();
            const auto id=base.planes[index];
            const auto n=base.normals[id];
            const auto origin=base.mesh.vertices[base.mesh.faces[id][0]];
            const double offset=dot3(sub(transform.origin,origin),n);
            const auto dir=inverse_vector(transform,n);
            const double box_min=offset+dot3(dir,{dir[0]>0?other.lo[0]:other.hi[0],dir[1]>0?other.lo[1]:other.hi[1],dir[2]>0?other.lo[2]:other.hi[2]});
            if (box_min>=0) continue;
            std::vector<char> inside_cap(other.mesh.vertices.size());
            std::size_t tips=0;
            double depth=0;
            for (auto i:other.surface_vertices) {
                const double d=-offset-dot3(dir,other.mesh.vertices[i]);
                if (d<=0) continue;
                inside_cap[i]=1; ++tips;
                depth=std::max(depth,d);
            }
            if (!tips || tips==other.surface_vertices.size()) continue;
            const auto fits=[&](Vec vertex) {
                const auto x=transform.point(vertex);
                const auto projected=sub(x,scale(n,dot3(sub(x,origin),n)));
                for (const auto j:base.planes) if (j!=id) {
                    const auto m=base.normals[j];
                    const double h=dot3(m,sub(projected,base.mesh.vertices[base.mesh.faces[j][0]]));
                    if (h+std::max(0.,-depth*dot3(n,m))>=-eps) return false;
                }
                return true;
            };
            bool box_fits=true;
            for (int corner=0;corner<8 && box_fits;++corner)
                box_fits=fits({corner&1?other.hi[0]:other.lo[0],corner&2?other.hi[1]:other.lo[1],corner&4?other.hi[2]:other.lo[2]});
            if (!box_fits && !std::ranges::all_of(other.surface_vertices,[&](auto i){return fits(other.mesh.vertices[i]);})) continue;
            std::vector<ContactRegion> regions;
            std::vector<std::size_t> reached;
            for (auto i:other.surface_vertices) {
                if (!inside_cap[i]) continue;
                reached.assign(1,i);
                inside_cap[i]=0;
                double deepest=0;
                Vec tip{};
                for (std::size_t h=0;h<reached.size();++h) {
                    const auto p=other.mesh.vertices[reached[h]];
                    const double d=-offset-dot3(dir,p);
                    if (d>deepest) {deepest=d; tip=p;}
                    for (auto j:other.neighbors[reached[h]]) if (inside_cap[j]) {
                        inside_cap[j]=0; reached.push_back(j);
                    }
                }
                const auto x=transform.point(tip);
                regions.push_back(world({add(x,scale(n,.5*deepest)),scale(n,side?-1.:1.),deepest},poses[side]));
            }
            last_plane[side]=index;
            return regions;
        }
        return {};
    }
    const Geometry& moving(std::size_t id,const RigidPlacement& rel) {
        if (version!=versions[id]) {
            moving_faces[id]=facet(surfaces[1],id,rel);
            versions[id]=version;
        }
        return moving_faces[id];
    }
    bool planes_separate(const Geometry& a,const Geometry& b,double margin) const {
        const double g0=dot3(a.n,sub(b.tri[0],a.tri[0]));
        const double g1=dot3(a.n,sub(b.tri[1],a.tri[0]));
        const double g2=dot3(a.n,sub(b.tri[2],a.tri[0]));
        return std::min({g0,g1,g2})>margin || std::max({g0,g1,g2}) < -margin;
    }
    bool possible_pair(const Geometry& a,const Geometry& b,double margin) const {
        for (int k=0;k<3;++k)
            if (a.lo[k]>b.hi[k]+margin || b.lo[k]>a.hi[k]+margin) return false;
        return !planes_separate(a,b,margin) && !planes_separate(b,a,margin);
    }
    // Nodes are actual intersecting finite triangle pairs. Two neighboring
    // nodes are connected only if their common mesh edge also hits the other
    // triangle. Thus a shared triangle ID alone never merges separate patches.
    std::vector<std::vector<std::array<std::size_t,2>>> crossing_groups(
            const std::array<RigidPlacement,2>& poses,const RigidPlacement& rel,ContactWork& work) {
        const double slack=1e-13*(surfaces[0].radius+surfaces[1].radius+length(rel.origin));
        if (!initialized || motion_bound(rel,saved,surfaces[1].radius)>skin) {
            nearby=scene.near_pairs(poses,skin,work);
            std::erase_if(nearby,[&](FacePair p){
                const auto& a=fixed_faces[p[0]]; const auto& b=moving(p[1],rel);
                const double margin=skin+slack;
                return !possible_pair(a,b,margin) || projected_relation(a,b,margin)<0 || projected_relation(b,a,margin)<0;
            });
            initialized=true; saved=rel;
        }
        using Pair=std::array<std::size_t,2>;
        std::vector<Pair> hits;
        std::unordered_map<std::uint64_t,std::size_t> index;
        const auto key=[](Pair p) {return (std::uint64_t(p[0])<<32)|p[1];};
        for (auto pair:nearby) {
            const auto& a=fixed_faces[pair[0]]; const auto& b=moving(pair[1],rel);
            if (!possible_pair(a,b,slack)) continue;
            const int first=projected_relation(a,b,slack);
            if (first<0) continue;
            const int second=projected_relation(b,a,slack);
            if (second<0) continue;
            ++work.mesh.triangles;
            const bool planar_overlap=first && second && a.drop==b.drop && a.lo[a.drop]==a.hi[a.drop] &&
                b.lo[a.drop]==a.lo[a.drop] && b.hi[a.drop]==a.hi[a.drop];
            if (planar_overlap || triangle_overlap(a.tri,b.tri)) hits.push_back(pair);
        }
        // Contact forces are accumulated in component order. Make both the
        // representative triangle and the accumulation independent of BVH traversal.
        std::sort(hits.begin(),hits.end());
        for (std::size_t i=0;i<hits.size();++i) index.emplace(key(hits[i]),i);
        std::vector<std::size_t> parent(hits.size());
        std::iota(parent.begin(),parent.end(),0);
        auto root=[&](std::size_t i) {while (i!=parent[i]) {parent[i]=parent[parent[i]]; i=parent[i];} return i;};
        for (std::size_t i=0;i<hits.size();++i) for (unsigned side=0;side<2;++side) {
            const auto pair=hits[i]; const auto face=pair[side];
            const auto& target=side ? fixed_faces[pair[0]] : moving(pair[1],rel);
            const auto& source=side ? moving(pair[1],rel) : fixed_faces[pair[0]];
            for (int j=0;j<3;++j) {
                const auto& e=surfaces[side].edges[surfaces[side].face_edges[face][j]];
                Pair adjacent=pair;
                adjacent[side]=e.first==face ? e.second : e.first;
                const auto it=index.find(key(adjacent));
                if (it==index.end() || it->second>=i || root(i)==root(it->second)) continue;
                if (segment_hits_face(source.tri[j],source.tri[(j+1)%3],target.tri)) parent[root(i)]=root(it->second);
            }
        }
        std::map<std::size_t,std::vector<Pair>> components;
        for (std::size_t i=0;i<hits.size();++i) components[root(i)].push_back(hits[i]);
        std::vector<std::vector<Pair>> result;
        for (auto& [id,group]:components) result.push_back(std::move(group));
        return result;
    }
    PatchTopology build_patch(const FaceGroup& group) {
        face_rows.clear();
        for (auto& rows:feature_rows) rows.clear();
        PatchTopology patch;
        const auto add_faces=[&](std::size_t a,std::size_t b) {
            if (!face_rows.insert(a,b)) return;
            patch.faces[0].push_back(a);
            patch.faces[1].push_back(b);
            for (auto v:surfaces[0].mesh.faces[a]) feature_rows[0].insert(v,b);
            for (auto v:surfaces[1].mesh.faces[b]) feature_rows[1].insert(v,a);
            for (auto i:surfaces[0].face_edges[a])
                for (auto j:surfaces[1].face_edges[b]) feature_rows[2].insert(i,j);
        };
        for (auto [a,b]:group) {
            add_faces(a,b);
            for (auto i:surfaces[0].face_edges[a]) {
                const auto& e=surfaces[0].edges[i]; add_faces(e.first==a ? e.second : e.first,b);
            }
            for (auto j:surfaces[1].face_edges[b]) {
                const auto& e=surfaces[1].edges[j]; add_faces(a,e.first==b ? e.second : e.first);
            }
        }
        for (unsigned kind=0;kind<feature_rows.size();++kind) {
            auto& rows=feature_rows[kind];
            std::sort(rows.touched.begin(),rows.touched.end());
            for (auto a:rows.touched) {
                auto& partners=rows.rows[a];
                std::sort(partners.begin(),partners.end());
                for (auto b:partners) patch.features.push_back({kind,a,b});
            }
        }
        for (auto& faces:patch.faces) {
            std::sort(faces.begin(),faces.end());
            faces.erase(std::unique(faces.begin(),faces.end()),faces.end());
        }
        return patch;
    }
    std::optional<ContactRegion> contour_contact(const FaceGroup& group,const RigidPlacement& rel,ContactWork& work) {
        const double eps=1e-12*(surfaces[0].radius+surfaces[1].radius+length(rel.origin));
        // If all opposed facets share a tangent, their intersection can be
        // split at coplanar end caps without splitting the physical line
        // support. Resolve that support with the finite feature witnesses.
        Vec first_normal{},tangent{};
        bool spans_space=false;
        for (auto [ia,ib]:group) {
            const Vec a=fixed_faces[ia].n,b=moving(ib,rel).n;
            if (dot3(a,b)>=-.8) continue;
            for (Vec n:{a,b}) {
                if (squared(first_normal)==0) first_normal=n;
                if (squared(tangent)==0) {
                    const auto c=cross3(first_normal,n);
                    if (length(c)>1e-7) tangent=scale(c,1/length(c));
                } else if (std::abs(dot3(tangent,n))>1e-7) spans_space=true;
            }
            if (spans_space) break;
        }
        if (!spans_space) return {};
        double perimeter=0;
        Vec center{},normal{};
        for (auto [ia,ib]:group) {
            const auto& a=fixed_faces[ia];
            const auto& b=moving(ib,rel);
            const auto interval=transverse_segment(a,b,eps);
            if (!interval) continue;
            const double weight=length(sub(interval->second,interval->first));
            center=add(center,scale(add(interval->first,interval->second),.5*weight));
            perimeter+=weight;
            if (dot3(a.n,b.n)<-.8) normal=add(normal,scale(sub(a.n,b.n),weight));
        }
        if (perimeter<=eps || length(normal)<=eps) return {};
        center=scale(center,1/perimeter);
        normal=scale(normal,1/length(normal));
        std::array<double,2> offsets;
        for (unsigned s=0;s<2;++s) {
            const Vec local_point=s?inverse_vector(rel,sub(center,rel.origin)):center;
            const Vec outward=s?inverse_vector(rel,scale(normal,-1)):normal;
            Vec hit_normal{};
            // Start outside the contact slab and trace inward to this material's surface.
            const auto t=ray_trees[s]->ray_first(add(local_point,scale(outward,limit)),scale(outward,-1),
                                                  2*limit,work.mesh,&hit_normal);
            if (!t || dot3(hit_normal,outward)<=0) return {};
            offsets[s]=limit-*t;
        }
        const double depth=offsets[0]+offsets[1];
        if (depth<=eps || depth>limit) return {};
        return ContactRegion{add(center,scale(normal,.5*(offsets[0]-offsets[1]))),normal,depth};
    }
    std::vector<ContactRegion> partitioned(const std::array<RigidPlacement,2>& poses,const RigidPlacement& rel,
                                           ContactWork& work) {
        const auto groups=crossing_groups(poses,rel,work);
        if (groups.empty()) {
            const auto exact=scene.intersect(poses,true);
            if (!exact.enclosed.empty() || !exact.contours.empty())
                throw UnresolvedContact("enclosed or unresolved mesh intersection without a finite boundary patch");
            return {};
        }
        if (!surfaces[0].connected || !surfaces[1].connected) {
            if (!scene.intersect(poses,true).enclosed.empty())
                throw UnresolvedContact("an enclosed mesh component has no boundary contact patch");
        }
        std::vector<ContactRegion> contacts;
        decltype(previous_patches) next_patches;
        for (const auto& group:groups) {
            if (ray_trees[0]) {
                if (const auto contact=contour_contact(group,rel,work)) {
                    contacts.push_back(*contact);
                    continue;
                }
            }
            auto node=previous_patches.extract(group);
            decltype(next_patches)::iterator entry;
            if (node.empty()) {
                entry=next_patches.emplace(group,build_patch(group)).first;
                ++work.mesh_patch_builds;
            } else {
                entry=next_patches.insert(std::move(node)).position;
                ++work.mesh_patch_reuses;
            }
            const auto& patch=entry->second;
            // Translating either local surface patch past the other along
            // a facet normal separates them. A longer witness (e.g. from
            // the front rim through the back of a tooth) cannot describe
            // their local penetration. This bound uses only mesh geometry.
            std::array<std::vector<Vec>,2> patch_vertices;
            std::vector<Vec> axes;
            for (unsigned s=0;s<2;++s) {
                for (auto id:patch.faces[s]) {
                    const auto& triangle=s?moving(id,rel):fixed_faces[id];
                    patch_vertices[s].insert(patch_vertices[s].end(),triangle.tri.begin(),triangle.tri.end());
                    if (std::ranges::none_of(axes,[&](Vec n){return std::abs(dot3(n,triangle.n))>1.-1e-12;}))
                        axes.push_back(triangle.n);
                }
            }
            double bound=std::numeric_limits<double>::infinity();
            for (auto n:axes) {
                std::array<double,2> lo{},hi{};
                for (unsigned s=0;s<2;++s) {
                    lo[s]=hi[s]=dot3(n,patch_vertices[s].front());
                    for (auto p:patch_vertices[s]) {
                        const double h=dot3(n,p);
                        lo[s]=std::min(lo[s],h); hi[s]=std::max(hi[s],h);
                    }
                }
                bound=std::min({bound,hi[0]-lo[1],hi[1]-lo[0]});
            }
            const double roundoff=1e-12*(surfaces[0].radius+surfaces[1].radius+length(rel.origin));
            auto region=witness(rel,patch.features,false,bound+roundoff);
            if (region.depth_m<=roundoff && groups.size()==1) {
                // A face ring can surround the deepest vertex without
                // containing it. Flood only the part inside the other solid.
                // Each projection is onto the actual closed triangle boundary.
                for (unsigned s=0;s<2;++s) {
                    const unsigned t=1-s;
                    const auto to_target=relative(poses[t],poses[s]);
                    const auto to_first=s?rel:RigidPlacement{};
                    std::vector<bool> visited(surfaces[s].mesh.vertices.size());
                    std::vector<std::size_t> queue;
                    const auto visit=[&](std::size_t v) {
                        if (visited[v]) return;
                        visited[v]=true;
                        const Vec p=surfaces[s].mesh.vertices[v];
                        const Vec x=to_target.point(p);
                        const auto foot=scene.interior_projection(t,x);
                        if (!foot) return;
                        queue.push_back(v);
                        Vec delta=sub(*foot,x);
                        const double depth=length(delta);
                        if (depth<=region.depth_m) return;
                        const auto outward=inverse_vector(to_target,scale(delta,1/depth));
                        region={to_first.point(add(p,scale(outward,.5*depth))),
                                to_first.rotate(scale(outward,s?1.:-1.)),depth};
                    };
                    for (const auto& pair:group)
                        for (auto v:surfaces[s].mesh.faces[pair[s]]) visit(v);
                    for (std::size_t i=0;i<queue.size();++i)
                        for (auto v:surfaces[s].neighbors[queue[i]]) visit(v);
                }
            }
            if (region.depth_m>1e-13*std::min(surfaces[0].radius,surfaces[1].radius)) {
                const auto same=[&](const ContactRegion& p) {
                    return std::abs(p.depth_m-region.depth_m)<=roundoff &&
                           squared(sub(p.point,region.point))<=roundoff*roundoff &&
                           squared(sub(p.normal,region.normal))<=1e-18;
                };
                if (std::ranges::none_of(contacts,same)) contacts.push_back(region);
            }
            else {
                const auto exact=scene.intersect(poses,true);
                for (const auto& c:exact.contours)
                    if (c.vector_area_a_m2 && squared(*c.vector_area_a_m2)>0 &&
                        std::ranges::find(c.triangles,group.front())!=c.triangles.end())
                        throw UnresolvedContact("intersecting patch has no finite feature witness");
            }
        }
        previous_patches.swap(next_patches);
        for (auto& contact:contacts) contact=world(contact,poses[0]);
        return contacts;
    }
    std::vector<ContactRegion> query(const std::array<RigidPlacement,2>& poses,ContactWork& work) {
        ++version;
        const auto hit=scene.overlaps(poses);
        work.mesh.nodes+=hit.node_pairs;
        work.mesh.triangles+=hit.triangle_tests;
        if (!hit.overlap) return {};
        const auto rel=relative(poses[0],poses[1]);
        if (!surfaces[0].convex || !surfaces[1].convex) {
            for (unsigned i=0;i<2;++i) if (auto cap=planar_cap(i,poses)) return *cap;
            return partitioned(poses,rel,work);
        }
        if (!initialized || motion_bound(rel,saved,surfaces[1].radius)>skin) rebuild(poses,rel,work);
        auto region=witness(rel,list,true,reach);
        while (!std::isfinite(region.depth_m) && reach<limit) {
            reach=std::min(limit,2*reach);
            rebuild(poses,rel,work);
            region=witness(rel,list,true,reach);
        }
        if (!std::isfinite(region.depth_m)) throw UnresolvedContact("convex overlap outside finite feature search band");
        if (region.depth_m<=1e-13*std::min(surfaces[0].radius,surfaces[1].radius)) return {};
        return {world(region,poses[0])};
    }
};
FiniteFeatures::FiniteFeatures(const MeshData& a,const MeshData& b,MeshScene& scene,double max_depth,bool contour_ray)
    :data_(std::make_unique<Data>(a,b,scene,max_depth,contour_ray)) {}
FiniteFeatures::~FiniteFeatures()=default;
std::vector<dynamics::ContactRegion> FiniteFeatures::query(const std::array<RigidPlacement,2>& poses,
                                                           dynamics::ContactWork& work) {
    return data_->query(poses,work);
}
}
