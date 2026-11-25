// Independent CAD-only proposals for the region-discovery replay. A finite
// set of witnesses proves local overlap, but an empty set cannot prove separation.
#include "nurbscontact/cad_contact_regions.hpp"
#include "cad_compiled_chart.hpp"
#include "cad_rigid_path.hpp"
#include "cad_patch_search.hpp"
#include "cad_local_solid.hpp"
#include <BRepAdaptor_Surface.hxx>
#include <BRepTopAdaptor_FClass2d.hxx>
#include <Bnd_Box.hxx>
#include <NCollection_UBTreeFiller.hxx>
#include <OSD_ThreadPool.hxx>
#include <TopExp.hxx>
#include <TopoDS.hxx>
#include <chrono>
#include <map>
#include <memory>
#include <numeric>

namespace {
using namespace nc;
using namespace nc::dynamics;
using Tree=NCollection_UBTree<std::size_t,Bnd_Box>;
using Clock=std::chrono::steady_clock;
struct Chart {
    gp_Pnt point;
    gp_Vec normal;
    Bnd_Box search_box;
    std::size_t face;
    bool interior;
    double u,v;
};
class Selection final : public Tree::Selector {
public:
    explicit Selection(Bnd_Box b): box(b) {}
    bool Reject(const Bnd_Box& branch) const override {return box.IsOut(branch);}
    bool Accept(const std::size_t& id) override {ids.push_back(id); return true;}
    Bnd_Box box;
    std::vector<std::size_t> ids;
};
struct Box {
    Vec center,half;
    explicit Box(const Bnd_Box& b) {
        const auto lo=cad::coordinates(b.CornerMin()),hi=cad::coordinates(b.CornerMax());
        center=scale(add(lo,hi),.5); half=scale(sub(hi,lo),.5);
    }
    bool overlaps(const Box& b) const {
        for (int i=0;i<3;++i) if (std::abs(center[i]-b.center[i])>half[i]+b.half[i]) return false;
        return true;
    }
};
// Keep tile and point bounds in contiguous median-split trees. A transformed
// parent bounds every transformed child AABB, including the search margin.
struct ChartTree {
    struct Node {Box box; int left=-1,right=-1; std::size_t chart{};};
    std::vector<Node> nodes;
    void build(std::vector<std::size_t>& ids,const std::vector<Chart>& charts,bool points) {
        const auto recurse=[&](auto&& self,auto first,auto last)->int {
            Bnd_Box bounds;
            for (auto it=first;it!=last;++it) {
                if (points) bounds.Add(charts[*it].point); else bounds.Add(charts[*it].search_box);
            }
            int id=static_cast<int>(nodes.size());
            nodes.push_back({Box(bounds)});
            if (last-first==1) nodes[id].chart=*first;
            else {
                auto& h=nodes[id].box.half;
                int axis=static_cast<int>(std::max_element(h.begin(),h.end())-h.begin())+1;
                auto middle=first+(last-first)/2;
                std::nth_element(first,middle,last,[&](auto a,auto b) {
                    return charts[a].point.Coord(axis)<charts[b].point.Coord(axis);
                });
                int l=self(self,first,middle),r=self(self,middle,last);
                nodes[id].left=l; nodes[id].right=r;
            }
            return id;
        };
        if (!ids.empty()) recurse(recurse,ids.begin(),ids.end());
    }
};
struct Atlas {
    CadContactIndex index;
    CadExitQuery rays;
    std::vector<Chart> charts;
    Tree sites,tiles;
    Bnd_Box body_box;
    ChartTree point_index,tile_index;
    std::vector<std::size_t> point_order,tile_order;
    using Face=cad::CompiledChart;
    std::vector<std::unique_ptr<Face>> surfaces;
    Atlas(const TopoDS_Shape& shape,int count):index(shape),rays(index) {
        TopTools_IndexedMapOfShape faces;
        TopExp::MapShapes(shape,TopAbs_FACE,faces);
        NCollection_UBTreeFiller<std::size_t,Bnd_Box> points(sites),boxes(tiles);
        const auto face_bounds=index.face_bounds();
        for (int f=1;f<=faces.Extent();++f) {
            for (const auto xyz:face_bounds[f-1]) body_box.Add(cad::point(scale(xyz,1000)));
            const auto face=TopoDS::Face(faces(f));
            surfaces.emplace_back(std::make_unique<Face>(face));
            const auto& surface=surfaces.back()->surface;
            const auto& trims=surfaces.back()->trims;
            const double u0=surface.FirstUParameter(),v0=surface.FirstVParameter();
            const double du=(surface.LastUParameter()-u0)/count,dv=(surface.LastVParameter()-v0)/count;
            for (int i=0;i<count;++i) for (int j=0;j<count;++j) {
                const double u=u0+(i+.5)*du,v=v0+(j+.5)*dv;
                gp_Pnt p;
                gp_Vec tu,tv;
                surface.D1(u,v,p,tu,tv);
                auto n=tu.Crossed(tv);
                if (n.SquareMagnitude()<1e-24) continue;
                n.Normalize();
                if (face.Orientation()==TopAbs_REVERSED) n.Reverse();
                Bnd_Box box;
                box.Add(p);
                for (double a:{-.5,.5}) for (double b:{-.5,.5}) box.Add(surface.Value(u+a*du,v+b*dv));
                const bool interior=trims.Perform(gp_Pnt2d(u,v))==TopAbs_IN;
                const auto id=charts.size();
                charts.push_back({p,n,box,std::size_t(f),interior,u,v});
                boxes.Add(id,box);
                if (interior) { Bnd_Box site; site.Add(p); points.Add(id,site); }
            }
        }
        points.Fill(); boxes.Fill();
        Bnd_Box all; all.SetWhole();
        Selection p(all),t(all);
        sites.Select(p); tiles.Select(t);
        point_order.resize(charts.size()); tile_order.resize(charts.size());
        for (std::size_t i=0;i<p.ids.size();++i) point_order[p.ids[i]]=i;
        for (std::size_t i=0;i<t.ids.size();++i) tile_order[t.ids[i]]=i;
        point_index.build(p.ids,charts,true); tile_index.build(t.ids,charts,false);
    }
};
struct Proposal {
    double score;
    std::size_t site,tile;
    bool reverse;
};
using FacePair=std::array<std::size_t,2>;
using Proposals=std::map<FacePair,std::vector<Proposal>>;

void collect_indexed(const Atlas& source,const Atlas& target,const gp_Trsf& pose,
                     double reach,bool reverse,Proposals& candidates) {
    const auto& sites=source.point_index.nodes;
    const auto& tiles=target.tile_index.nodes;
    if (sites.empty() || tiles.empty()) return;
    const auto m=pose.VectorialPart();
    auto window=source.body_box.Transformed(pose.Inverted()); window.Enlarge(reach);
    const auto visit_tile=[&](auto&& self,int t,int s)->void {
        const auto& tile=tiles[t];
        Box box=tile.box;
        box.center=cad::coordinates(cad::point(box.center).Transformed(pose));
        for (int i=0;i<3;++i) {
            box.half[i]=reach+1e-10;
            for (int j=0;j<3;++j) box.half[i]+=std::abs(m.Value(i+1,j+1))*tile.box.half[j];
        }
        const auto visit_sites=[&](auto&& descend,int k)->void {
            const auto& site=sites[k];
            if (!box.overlaps(site.box)) return;
            if (tile.left>=0 && (site.left<0 || squared(box.half)>squared(site.box.half))) {
                self(self,tile.left,k); self(self,tile.right,k); return;
            }
            if (site.left>=0) {descend(descend,site.left); descend(descend,site.right); return;}
            const auto& f=target.charts[tile.chart];
            const auto& p=source.charts[site.chart];
            if (window.IsOut(f.search_box)) return;
            auto old_box=f.search_box.Transformed(pose); old_box.Enlarge(reach);
            if (old_box.IsOut(p.point)) return;
            const auto n=f.normal.Transformed(pose);
            const double cosine=-p.normal.Dot(n);
            if (cosine<.5) return;
            const double d=-gp_Vec(f.point.Transformed(pose),p.point).Dot(n)/cosine;
            if (std::abs(d)>reach) return;
            const FacePair pair=reverse ? FacePair{f.face,p.face} : FacePair{p.face,f.face};
            candidates[pair].push_back({d,site.chart,tile.chart,reverse});
        };
        visit_sites(visit_sites,s);
    };
    visit_tile(visit_tile,0,0);
}

void collect(const Atlas& source,const Atlas& target,const gp_Trsf& target_in_source,
             double reach_mm,bool reverse,Proposals& candidates) {
    auto window=source.body_box.Transformed(target_in_source.Inverted());
    window.Enlarge(reach_mm);
    Selection tiles(window);
    target.tiles.Select(tiles);
    for (const auto id:tiles.ids) {
        const auto& tile=target.charts[id];
        auto box=tile.search_box.Transformed(target_in_source);
        box.Enlarge(reach_mm);
        Selection sites(box);
        source.sites.Select(sites);
        if (sites.ids.empty()) continue;
        const auto origin=tile.point.Transformed(target_in_source);
        const auto outward=tile.normal.Transformed(target_in_source);
        for (auto s:sites.ids) {
            const auto& site=source.charts[s];
            const double cosine=-site.normal.Dot(outward);
            if (cosine<.5) continue;
            const double estimated=-gp_Vec(origin,site.point).Dot(outward)/cosine;
            if (std::abs(estimated)>reach_mm) continue;
            const FacePair pair=reverse ? FacePair{tile.face,site.face} : FacePair{site.face,tile.face};
            candidates[pair].push_back({estimated,s,id,reverse});
        }
    }
}

// Intersect the supporting surface near the candidate chart in millimetres.
// This inexpensive proposal may fail at folds. The accepted overlap still
// requires both nearest oriented, trim-valid exits from the full CAD face tree.
std::optional<double> propose_depth(const Atlas& target,const Chart& tile,
                                    gp_Pnt p,gp_Vec direction,double reach,bool require_exit=false) {
    const auto& face=*target.surfaces[tile.face-1];
    const auto& surf=face.surface;
    double u=tile.u,v=tile.v;
    for (int iter=0;iter<8;++iter) {
        gp_Pnt q;
        gp_Vec du,dv;
        face.d1(u,v,q,du,dv);
        const auto delta=gp_Vec(p,q);
        const double t=delta.Dot(direction);
        const auto residual=delta-direction*t;
        if (residual.SquareMagnitude()<1e-14) {
            if (std::abs(t)>reach || !face.inside(u,v)) return {};
            if (require_exit) {
                const auto normal=du.Crossed(dv);
                const double sign=surf.Face().Orientation()==TopAbs_REVERSED ? -1. : 1.;
                if (sign*direction.Dot(normal)<=1e-8*normal.Magnitude()) return {};
            }
            return t;
        }
        const auto c=dv.Crossed(direction),e=direction.Crossed(du);
        const double det=du.Dot(c);
        if (std::abs(det)<1e-11*du.Magnitude()*dv.Magnitude() || det==0) return {};
        u-=residual.Dot(c)/det;
        v-=residual.Dot(e)/det;
        if (!(u>=surf.FirstUParameter() && u<=surf.LastUParameter()
              && v>=surf.FirstVParameter() && v<=surf.LastVParameter())) return {};
    }
    return {};
}
struct Seed {
    Proposal proposal;
    Chart site;
};
struct Search {
    Atlas& a;
    Atlas& b;
    CadExitQuery &ray_a,&ray_b;
    const std::array<ContactPose,2>& poses;
    double reach;
    int grid;
    std::size_t tests{},attempts{};
    std::array<gp_Trsf,2> world,relative;
    Search(Atlas& a,Atlas& b,CadExitQuery& ray_a,CadExitQuery& ray_b,
           const std::array<ContactPose,2>& poses,double reach,int grid):
        a(a),b(b),ray_a(ray_a),ray_b(ray_b),poses(poses),reach(reach),grid(grid),
        world{cad::placement(poses[0]),cad::placement(poses[1])},
        relative{world[1].Inverted()*world[0],world[0].Inverted()*world[1]} {}

    double evaluate(const Seed& seed, FacePair pair, std::optional<CadChord>& hit) {
        const bool reverse=seed.proposal.reverse;
        const auto& other=reverse ? a : b;
        const auto& site=seed.site;
        const auto local=site.point.Transformed(relative[reverse]);
        const auto dir=-site.normal.Transformed(relative[reverse]);
        ++attempts;
        const auto depth=propose_depth(other,other.charts[seed.proposal.tile],local,dir,reach*1000);
        if (!depth) return -infinity;
        if (*depth>2*Precision::Confusion()) {
            const Vec p=scale(cad::coordinates(site.point.Transformed(world[reverse])),.001);
            const auto outward=site.normal.Transformed(world[reverse]);
            const Vec n{outward.X(),outward.Y(),outward.Z()};
            auto chord=local_chord(ray_a,ray_b,poses,sub(p,scale(n,.0005 * *depth)),
                                   scale(n,reverse ? -1. : 1.),reach,pair,tests);
            if (chord && chord->exits[0].face==pair[0] && chord->exits[1].face==pair[1]) hit=chord;
        }
        return *depth;
    }

    // Follow a near miss in the native parameter domain. Only a verified
    // positive chord stops the search. Searching a finite stencil is heuristic.
    void refine(Seed& best,double score,FacePair pair,std::optional<CadChord>& hit) {
        const auto& body=best.proposal.reverse ? b : a;
        const auto& face=*body.surfaces[best.site.face-1];
        const auto& surf=face.surface;
        double du=(surf.LastUParameter()-surf.FirstUParameter())/(2*grid);
        double dv=(surf.LastVParameter()-surf.FirstVParameter())/(2*grid);
        for (int k=0;k<5 && !hit;++k,du*=.5,dv*=.5) {
            const auto center=best.site;
            for (const auto offset:std::array<std::array<int,2>,8>{{{1,0},{-1,0},{0,1},{0,-1},{1,1},{-1,1},{1,-1},{-1,-1}}}) {
                Seed trial=best;
                auto& s=trial.site;
                s.u=center.u+du*offset[0]; s.v=center.v+dv*offset[1];
                if (s.u<surf.FirstUParameter() || s.u>surf.LastUParameter() || s.v<surf.FirstVParameter() ||
                    s.v>surf.LastVParameter() || !face.inside(s.u,s.v)) continue;
                gp_Vec tu,tv;
                face.d1(s.u,s.v,s.point,tu,tv);
                s.normal=tu.Crossed(tv);
                if (s.normal.SquareMagnitude()<1e-24) continue;
                s.normal.Normalize();
                if (surf.Face().Orientation()==TopAbs_REVERSED) s.normal.Reverse();
                const auto depth=evaluate(trial,pair,hit);
                if (depth>score || hit) {best=trial; score=depth;}
                if (hit) break;
            }
        }
    }
};

// Midpoint surface quadrature on the original UV charts. Each material-side
// face contributes once per group, including when several opposing faces
// cover it. The opposing trim and positive thickness clip its loaded part.
struct AreaNode {Vec point,area;};
struct SurfaceQuadrature {
    Atlas& body;
    int order;
    std::map<std::size_t,std::vector<AreaNode>> saved;
    const std::vector<AreaNode>& nodes(std::size_t id) {
        auto [it,fresh]=saved.try_emplace(id);
        if (!fresh) return it->second;
        const auto& f=*body.surfaces[id-1];
        const auto& s=f.surface;
        const double u0=s.FirstUParameter(),v0=s.FirstVParameter();
        const double du=(s.LastUParameter()-u0)/order,dv=(s.LastVParameter()-v0)/order;
        const double sign=s.Face().Orientation()==TopAbs_REVERSED ? -1. : 1.;
        for (int i=0;i<order;++i) for (int j=0;j<order;++j) {
            const double u=u0+(i+.5)*du,v=v0+(j+.5)*dv;
            if (!f.inside(u,v)) continue;
            gp_Pnt p;gp_Vec tu,tv;
            f.d1(u,v,p,tu,tv);
            const auto n=tu.Crossed(tv);
            it->second.push_back({scale(cad::coordinates(p),.001),
                                  scale(Vec{n.X(),n.Y(),n.Z()},sign*du*dv*1e-6)});
        }
        return it->second;
    }
};
struct PressurePatch {Vec center,normal,arm;double depth;};
std::optional<PressurePatch> integrate_patch(std::array<SurfaceQuadrature,2>& grids,
    CadExitQuery& ray_a,CadExitQuery& ray_b,const std::array<ContactPose,2>& poses,
    const std::vector<CadChord>& group,double reach) {
    Vec axis{};
    double best=-2.;
    for (const auto& w:group) {
        const auto a=poses[0].rotate(w.exits[0].outward),b=poses[1].rotate(w.exits[1].outward);
        const double opposed=-dot3(a,b);
        if (opposed>best) {best=opposed;axis=sub(a,b);}
    }
    if (length(axis)<1e-8) return {};
    axis=scale(axis,1/length(axis));
    Vec area{},moment{},centroid{};
    const Vec origin=group.front().region.point;
    double volume=0;
    for (int side=0;side<2;++side) {
        const int other=1-side;
        const double sign=side ? -1. : 1.;
        const auto ray=scale(axis,-sign);
        const auto direction=poses[other].inverse_rotate(ray);
        const auto world=cad::placement(poses[side]);
        const auto to_target=cad::placement(poses[other]).Inverted()*world;
        const gp_Vec cast(direction[0],direction[1],direction[2]);
        const auto segment=cast.XYZ()*(1000*reach);
        std::map<std::size_t,std::map<std::size_t,Chart>> faces;
        for (const auto& w:group) {
            const auto& t=w.exits[other];
            auto& c=faces[w.exits[side].face][t.face];
            c.face=t.face;c.u=t.uv[0];c.v=t.uv[1];
        }
        for (const auto& [id,targets]:faces) for (const auto& sample:grids[side].nodes(id)) {
            const auto native=cad::point(scale(sample.point,1000));
            const auto local=native.Transformed(to_target);
            double depth=infinity;
            for (const auto& [face,seed]:targets) {
                if (!grids[other].body.surfaces[face-1]->column_possible(local.XYZ(),segment)) continue;
                const auto d=propose_depth(grids[other].body,seed,local,cast,reach*1000,true);
                if (d && *d>0.) depth=std::min(depth,*d*.001);
            }
            if (!std::isfinite(depth)) continue;
            const auto point=scale(cad::coordinates(native.Transformed(world)),.001);
            const auto area_element=scale(poses[side].rotate(sample.area),sign);
            const auto r=sub(point,origin);
            area=add(area,area_element);
            moment=add(moment,cross3(r,area_element));
            const double dv=depth*std::max(0.,dot3(area_element,axis));
            centroid=add(centroid,scale(add(r,scale(ray,.5*depth)),dv));
            volume+=dv;
        }
    }
    const double magnitude=length(area);
    if (!(volume>0 && magnitude>0)) return {};
    const auto center=add(origin,scale(centroid,1/volume));
    const auto normal=scale(area,1/magnitude);
    std::size_t tests=0;
    auto chord=local_chord(ray_a,ray_b,poses,center,normal,reach,{},tests);
    if (!chord) return {};
    return PressurePatch{center,normal,sub(scale(moment,1/magnitude),cross3(sub(center,origin),normal)),
                         chord->region.depth_m};
}


}
namespace nc::cad {
using namespace dynamics;
struct PatchSearch::Data {
    Atlas a,b;
    int grid;
    double reach;
    std::map<FacePair,Seed> previous;
    struct Worker {
        CadExitQuery a,b;
        std::map<int,std::array<SurfaceQuadrature,2>> levels;
        Worker(const Atlas& x,const Atlas& y):a(x.index),b(y.index) {}
    };
    static constexpr int worker_limit=4;
    std::array<std::unique_ptr<Worker>,worker_limit> workers;
    Data(const TopoDS_Shape& first,const TopoDS_Shape& second,int grid,double reach):
        a(first,grid),b(second,grid),grid(grid),reach(reach) {
        for (auto& w:workers) w=std::make_unique<Worker>(a,b);
    }
    PatchDiscovery discover(const std::array<ContactPose,2>& poses,bool indexed,bool group) {
        const auto t=Clock::now();
        const auto b_in_a=cad::placement(poses[0]).Inverted()*cad::placement(poses[1]);
        std::array<Proposals,2> sides;
        const auto discover=indexed ? collect_indexed : collect;
        OSD_ThreadPool::Launcher pool(*OSD_ThreadPool::DefaultPool(),indexed ? worker_limit : 1);
        pool.Perform(0,2,[&](int,int side) {
            if (side) discover(b,a,b_in_a.Inverted(),1000*reach,true,sides[1]);
            else discover(a,b,b_in_a,1000*reach,false,sides[0]);
        });
        auto& proposals=sides[0];
        for (const auto& [pair,sites]:sides[1]) {
            auto& dest=proposals[pair]; dest.insert(dest.end(),sites.begin(),sites.end());
        }
        if (indexed) for (auto& [pair,sites]:proposals) {
            const auto key=[&](const Proposal& s) {
                return std::tuple{s.reverse,(s.reverse?a:b).tile_order[s.tile],(s.reverse?b:a).point_order[s.site]};
            };
            std::sort(sites.begin(),sites.end(),[&](const auto& l,const auto& r){return key(l)<key(r);});
        }
        for (const auto& [pair,seed]:previous) proposals.try_emplace(pair);
        const auto shortlisted=Clock::now();
        struct Job {
            Proposals::iterator where;
            std::optional<CadChord> hit;
            std::optional<Seed> best;
            std::size_t tracked{},attempts{},tests{};
        };
        std::vector<Job> jobs;
        for (auto it=proposals.begin();it!=proposals.end();++it) jobs.push_back({it,{},{}});
        pool.Perform(0,static_cast<int>(jobs.size()),[&](int idx,int n) {
            auto& job=jobs[n];
            auto& [pair,sites]=*job.where;
            auto& hit=job.hit;
            auto& best=job.best;
            auto& w=*workers[idx];
            Search search{a,b,w.a,w.b,poses,reach,grid};
            double score=-infinity;
            const auto probe=[&](const Seed& seed) {
                const double d=search.evaluate(seed,pair,hit);
                if (d>score || hit) {score=d; best=seed;}
            };
            if (const auto old=previous.find(pair);old!=previous.end()) {
                probe(old->second);
                job.tracked=hit.has_value();
            }
            std::sort(sites.begin(),sites.end(),[](const auto& l,const auto& r){return l.score>r.score;});
            std::vector<std::pair<bool,std::size_t>> tried;
            for (const auto& s:sites) {
                if (hit) break;
                const auto key=std::pair{s.reverse,s.site};
                if (std::find(tried.begin(),tried.end(),key)!=tried.end()) continue;
                tried.push_back(key);
                probe({s,(s.reverse ? b : a).charts[s.site]});
            }
            if (!hit && best) search.refine(*best,score,pair,hit);
            job.attempts=search.attempts;job.tests=search.tests;
        });
        std::map<FacePair,Seed> next;
        std::vector<std::optional<CadChord>> found;
        std::size_t attempts=0,tracked=0,tests=0;
        for (const auto& job:jobs) {
            if (job.hit) {found.push_back(job.hit); next.emplace(job.where->first,*job.best);}
            attempts+=job.attempts; tracked+=job.tracked; tests+=job.tests;
        }
        previous=std::move(next);
        const auto search_end=Clock::now();
        std::vector<int> labels;
        if (group && found.size()>=3 && found.size()<=24 && pool.HasThreads()) {
            std::vector<std::pair<int,int>> edges;
            for (int i=1;i<static_cast<int>(found.size());++i)
                for (int j=0;j<i;++j) edges.emplace_back(i,j);
            std::vector<unsigned char> connected(edges.size());
            pool.Perform(0,static_cast<int>(edges.size()),[&](int idx,int k) {
                const auto [i,j]=edges[k]; auto& w=*workers[idx];
                connected[k]=connected_chords(w.a,w.b,poses,*found[i],*found[j],reach);
            });
            labels.resize(found.size());std::iota(labels.begin(),labels.end(),0);
            for (std::size_t k=0;k<edges.size();++k) if (connected[k]) {
                const auto [i,j]=edges[k]; const int old=labels[i],replacement=labels[j];
                std::replace(labels.begin(),labels.end(),old,replacement);
            }
        } else if (group) labels=chord_groups(a.rays,b.rays,poses,found,reach);
        const auto end=Clock::now();
        const double seconds=std::chrono::duration<double>(end-t).count();

        return {std::move(found),std::move(labels),proposals.size(),attempts,tracked,tests,
                seconds,std::chrono::duration<double>(shortlisted-t).count(),std::chrono::duration<double>(end-search_end).count()};
    }
};
PatchSearch::PatchSearch(const TopoDS_Shape& a,const TopoDS_Shape& b,int grid,double reach):
    data_(std::make_unique<Data>(a,b,grid,reach)) {}
PatchSearch::~PatchSearch()=default;
bool PatchSearch::boxes_separate(const std::array<ContactPose,2>& poses) const {
    auto a=data_->a.body_box.Transformed(placement(poses[0]));
    a.Enlarge(a.GetGap()+2*Precision::Confusion());
    return a.IsOut(data_->b.body_box.Transformed(placement(poses[1])));
}
PatchDiscovery PatchSearch::discover(const std::array<ContactPose,2>& poses,bool indexed,bool group) {
    return data_->discover(poses,indexed,group);
}
std::optional<CadChord> PatchSearch::chord(const std::array<ContactPose,2>& poses,Vec p,Vec n,std::array<std::size_t,2> ids) {
    std::size_t tests=0;
    return local_chord(data_->a.rays,data_->b.rays,poses,p,n,data_->reach,ids,tests);
}
std::optional<ContactRegion> PatchSearch::integrate(const std::array<ContactPose,2>& poses,
                                                    const std::vector<CadChord>& group,int order,int worker) {
    if (order==0) {
        for (int level:{12,24,48,96}) if (auto r=integrate(poses,group,level,worker)) return r;
        return {};
    }
    auto& w=*data_->workers.at(worker);
    auto& levels=w.levels;
    if (!levels.contains(order)) levels.emplace(order,std::array<SurfaceQuadrature,2>{{{data_->a,order,{}},{data_->b,order,{}}}});
    const auto p=integrate_patch(levels.at(order),w.a,w.b,poses,group,data_->reach);
    if (!p) return {};
    return ContactRegion{p->center,p->normal,p->depth,p->arm};
}
}

namespace nc {
namespace {
using namespace dynamics;
class PatchRegionQuery final : public PairRegionQuery {
public:
    PatchRegionQuery(const TopoDS_Shape& a,const TopoDS_Shape& b,std::array<Vec,2> centers,double maximum):
        a_(a),b_(b),search_(a,b,6,2*maximum),events_(cad_local_region_query(a,b,centers,maximum)) {}
    bool independent_cad_patch() const override {return true;}
    std::vector<ContactRegion> regions(ContactPose a,ContactPose b,ContactWork& work) const override {
        const std::array poses{a,b};
        ++work.cad_patch_queries;
        if (search_.boxes_separate(poses)) return {};
        const auto search=search_.discover(poses,true,true);
        work.cad_patch_broad_s+=search.broad_s;
        work.cad_patch_search_s+=search.query_s-search.group_s-search.broad_s;
        work.cad_patch_group_s+=search.group_s;
        if (search.witnesses.empty()) {
            const auto tic=Clock::now();
            bool separated=false;
            try {
                if (!boundary_a_) boundary_a_=cad::boundary_index(a_);
                if (!boundary_b_) boundary_b_=cad::boundary_index(b_);
                const auto relative=cad::placement(b).Inverted()*cad::placement(a);
                SurfaceTransition stats;
                const double tol=2*Precision::Confusion();
                separated=cad::bounded_distance(boundary_a_,boundary_b_,relative,4*tol,tol,stats,&witness_)>tol;
            } catch (const UnresolvedContact&) { /* Resolve directly on the placed solids. */ }
              catch (const Standard_Failure&) {}
            work.cad_patch_search_s+=std::chrono::duration<double>(Clock::now()-tic).count();
            if (separated) return {};
            return whole_pressure(poses,work);
        }
        std::map<int,std::vector<CadChord>> groups;
        for (std::size_t i=0;i<search.labels.size();++i) groups[search.labels[i]].push_back(*search.witnesses[i]);
        const auto start=Clock::now();
        std::vector<const std::vector<CadChord>*> components;
        for (const auto& [label,group]:groups) components.push_back(&group);
        std::vector<std::optional<ContactRegion>> patches(groups.size());
        if (!groups.empty()) {
            OSD_ThreadPool::Launcher pool(*OSD_ThreadPool::DefaultPool(),std::min(4,static_cast<int>(groups.size())));
            pool.Perform(0,static_cast<int>(groups.size()),[&](int worker,int i) {
                patches[i]=search_.integrate(poses,*components[i],0,worker);
            });
        }
        std::vector<ContactRegion> result;
        for (const auto& patch:patches) {
            if (!patch) {
                work.cad_patch_area_s+=std::chrono::duration<double>(Clock::now()-start).count();
                return whole_pressure(poses,work);
            }
            result.push_back(*patch);
        }
        work.cad_patch_area_s+=std::chrono::duration<double>(Clock::now()-start).count();
        return result;
    }
    SurfaceTransition transition(ContactPose a,ContactPose b,ContactPose c,ContactPose d,double eps) const override {
        return events_->transition(a,b,c,d,eps);
    }
    SurfaceTransition transition(ContactPose a,ContactPose b,ContactPose c,ContactPose d,double eps,double dt) const override {
        return events_->transition(a,b,c,d,eps,dt);
    }
private:
    std::vector<ContactRegion> whole_pressure(const std::array<ContactPose,2>& poses,ContactWork& work) const {
        std::vector<Vec> arms;
        auto result=cad::solid_regions(a_.Moved(TopLoc_Location(cad::placement(poses[0]))),
                                       b_.Moved(TopLoc_Location(cad::placement(poses[1]))),work,false,&arms);
        for (std::size_t i=0;i<result.size();++i) {
            auto chord=search_.chord(poses,result[i].point,result[i].normal,{});
            if (chord) result[i].depth_m=chord->region.depth_m;
            else result[i].depth_m=limiting_depth(poses,result[i]);
            result[i].pressure_arm=arms[i];
        }
        return result;
    }
    double limiting_depth(const std::array<ContactPose,2>& poses,const ContactRegion& patch) const {
        // At an isolated chart pole the centroid ray can terminate on a
        // degenerate boundary. Check the two-sided limit on two shrinking
        // rings. A missing sample or a depth jump leaves the pose unresolved.
        const Vec n=patch.normal;
        Vec u=cross3(n,std::abs(n[0])<.75 ? Vec{1,0,0} : Vec{0,1,0});
        u=scale(u,1/length(u));
        const Vec v=cross3(n,u);
        for (double phase:{.371,1.083}) {
            double lo=infinity,hi=0,sum=0;
            int hits=0;
            for (double radius:{2e-6,1e-6}) for (int j=0;j<4;++j) {
                const double angle=phase+j*1.5707963267948966;
                const auto offset=add(scale(u,radius*std::cos(angle)),scale(v,radius*std::sin(angle)));
                const auto c=search_.chord(poses,add(patch.point,offset),n,{});
                if (!c) continue;
                const double depth=c->region.depth_m;
                lo=std::min(lo,depth);hi=std::max(hi,depth);
                if (radius==1e-6) sum+=depth;
                ++hits;
            }
            if (hits==8 && hi-lo<=std::max(2e-9,1e-5*lo)) return sum*.25;
        }
        throw UnresolvedContact("unresolved pressure depth in STEP overlap");
    }
    TopoDS_Shape a_,b_;
    mutable std::shared_ptr<const cad::BoundaryIndex> boundary_a_,boundary_b_;
    mutable cad::BoundaryWitness witness_;
    mutable cad::PatchSearch search_;
    std::shared_ptr<const PairRegionQuery> events_;
};
}
std::shared_ptr<const dynamics::PairRegionQuery> cad_patch_region_query(
    const TopoDS_Shape& a,const TopoDS_Shape& b,std::array<Vec,2> centers,double maximum) {
    if (!(maximum>0 && std::isfinite(maximum))) throw std::invalid_argument("invalid maximum depth");
    return std::make_shared<PatchRegionQuery>(a,b,centers,maximum);
}
}
