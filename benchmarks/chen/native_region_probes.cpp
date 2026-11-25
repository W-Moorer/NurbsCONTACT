// Independent CAD-only proposals for the region-discovery replay. A finite
// set of witnesses proves local overlap, but an empty set cannot prove separation.
#include "nurbscontact/cad_contact_regions.hpp"
#include "../../src/cad_rigid_path.hpp"
#include <BRepAdaptor_Surface.hxx>
#include <BRepTopAdaptor_FClass2d.hxx>
#include <Bnd_Box.hxx>
#include <NCollection_UBTreeFiller.hxx>
#include <TopExp.hxx>
#include <TopoDS.hxx>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>

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
struct Atlas {
    CadContactIndex index;
    CadExitQuery rays;
    std::vector<Chart> charts;
    Tree sites,tiles;
    Bnd_Box body_box;
    struct Face {
        BRepAdaptor_Surface surface;
        BRepTopAdaptor_FClass2d trims;
        explicit Face(const TopoDS_Face& f):surface(f),trims(f,Precision::PConfusion()) {}
    };
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
    }
};
struct Proposal {
    double score;
    std::size_t site,tile;
    bool reverse;
};
using FacePair=std::array<std::size_t,2>;
using Proposals=std::map<FacePair,std::vector<Proposal>>;

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
                                    gp_Pnt p,gp_Vec direction,double reach) {
    const auto& face=*target.surfaces[tile.face-1];
    const auto& surf=face.surface;
    double u=tile.u,v=tile.v;
    for (int iter=0;iter<8;++iter) {
        gp_Pnt q;
        gp_Vec du,dv;
        surf.D1(u,v,q,du,dv);
        const auto delta=gp_Vec(p,q);
        const double t=delta.Dot(direction);
        const auto residual=delta-direction*t;
        if (residual.SquareMagnitude()<1e-14) {
            if (std::abs(t)>reach || face.trims.Perform(gp_Pnt2d(u,v))!=TopAbs_IN) return {};
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
ContactPose read_pose() {
    ContactPose pose;
    for (auto& x:pose.origin) std::cin>>x;
    for (auto& x:pose.axis) std::cin>>x;
    if (!(std::cin>>pose.angle)) throw std::runtime_error("incomplete pose");
    return pose;
}

struct Seed {
    Proposal proposal;
    Chart site;
};
struct Search {
    Atlas& a;
    Atlas& b;
    const std::array<ContactPose,2>& poses;
    double reach;
    int grid;
    std::size_t tests{},attempts{};

    double evaluate(const Seed& seed, FacePair pair, std::optional<CadChord>& hit) {
        const bool reverse=seed.proposal.reverse;
        const auto& other=reverse ? a : b;
        const auto& source=poses[reverse ? 1 : 0];
        const auto& target=poses[reverse ? 0 : 1];
        const auto& site=seed.site;
        const Vec p=source.point(scale(cad::coordinates(site.point),.001));
        const Vec n=source.rotate({site.normal.X(),site.normal.Y(),site.normal.Z()});
        const auto local=cad::point(scale(target.inverse_rotate(sub(p,target.origin)),1000));
        const auto dir=target.inverse_rotate(scale(n,-1));
        ++attempts;
        const auto depth=propose_depth(other,other.charts[seed.proposal.tile],local,gp_Vec(dir[0],dir[1],dir[2]),reach*1000);
        if (!depth) return -infinity;
        if (*depth>2*Precision::Confusion()) {
            auto chord=local_chord(a.rays,b.rays,poses,sub(p,scale(n,.0005 * *depth)),
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
                    s.v>surf.LastVParameter() || face.trims.Perform(gp_Pnt2d(s.u,s.v))!=TopAbs_IN) continue;
                gp_Vec tu,tv;
                surf.D1(s.u,s.v,s.point,tu,tv);
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
}

int main(int argc,char** argv) try {
    const bool group=argc==6 && std::string_view(argv[5])=="--groups";
    if (argc!=5 && !group) throw std::invalid_argument("nc_cad_probes A.step B.step sites_per_uv_axis reach_m [--groups]");
    const int grid=std::stoi(argv[3]);
    const double reach=std::stod(argv[4]);
    if (grid<1 || grid>128 || !(reach>0 && std::isfinite(reach))) throw std::invalid_argument("invalid probe grid or reach");
    const auto start=Clock::now();
    Atlas a(read_step_shape(argv[1]),grid),b(read_step_shape(argv[2]),grid);
    std::cerr<<"prepare_s "<<std::chrono::duration<double>(Clock::now()-start).count()<<'\n';
    std::cout<<std::setprecision(17);
    std::map<FacePair,Seed> previous;
    while ((std::cin>>std::ws).peek()!=std::char_traits<char>::eof()) {
        const std::array poses{read_pose(),read_pose()};
        const auto t=Clock::now();
        const auto b_in_a=cad::placement(poses[0]).Inverted()*cad::placement(poses[1]);
        Proposals proposals;
        collect(a,b,b_in_a,1000*reach,false,proposals);
        collect(b,a,b_in_a.Inverted(),1000*reach,true,proposals);
        for (const auto& [pair,seed]:previous) proposals.try_emplace(pair);
        const auto shortlisted=Clock::now();
        Search search{a,b,poses,reach,grid};
        std::vector<std::optional<CadChord>> found;
        std::map<FacePair,Seed> next;
        std::size_t tracked=0;
        for (auto& [pair,sites]:proposals) {
            std::optional<CadChord> hit;
            std::optional<Seed> best;
            double score=-infinity;
            const auto probe=[&](const Seed& seed) {
                const double d=search.evaluate(seed,pair,hit);
                if (d>score || hit) {score=d; best=seed;}
            };
            if (const auto old=previous.find(pair);old!=previous.end()) {
                probe(old->second);
                if (hit) ++tracked;
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
            if (hit) {found.push_back(*hit); next.emplace(pair,*best);}
        }
        previous=std::move(next);
        const auto search_end=Clock::now();
        std::vector<int> labels;
        if (group) labels=chord_groups(a.rays,b.rays,poses,found,reach);
        const auto end=Clock::now();
        const double seconds=std::chrono::duration<double>(end-t).count();
        std::cout<<"{\"query_s\":"<<seconds<<",\"broad_s\":"<<std::chrono::duration<double>(shortlisted-t).count()
                 <<",\"candidate_face_pairs\":"<<proposals.size()
                 <<",\"attempts\":"<<search.attempts<<",\"tracked\":"<<tracked<<",\"face_tests\":"<<search.tests<<",\"witnesses\":[";
        for (std::size_t i=0;i<found.size();++i) {
            if (i) std::cout<<',';
            const auto& w=*found[i];
            std::cout<<'['<<w.exits[0].face<<','<<w.exits[1].face;
            for (double x:w.region.point) std::cout<<','<<x;
            for (double x:w.region.normal) std::cout<<','<<x;
            std::cout<<','<<w.region.depth_m<<']';
        }
        std::cout<<']';
        if (group) {
            std::cout<<",\"group_s\":"<<std::chrono::duration<double>(end-search_end).count()<<",\"labels\":[";
            for (std::size_t i=0;i<labels.size();++i) {if (i) std::cout<<','; std::cout<<labels[i];}
            std::cout<<']';
        }
        std::cout<<"}"<<std::endl;
    }
    return 0;
} catch (const Standard_Failure& e) {std::cerr<<e.GetMessageString()<<'\n'; return 1;}
  catch (const std::exception& e) {std::cerr<<e.what()<<'\n'; return 1;}
