#include "../../src/cad_boundary_distance.hpp"
#include "../../src/cad_local_solid.hpp"
#include "../../src/cad_rigid_path.hpp"
#include "nurbscontact/occt.hpp"
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepClass3d_SolidClassifier.hxx>
#include <BRep_Builder.hxx>
#include <IntCurvesFace_ShapeIntersector.hxx>
#include <gp_Lin.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS_Compound.hxx>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <string_view>

namespace {
using namespace nc;
TopoDS_Shape shells(const TopoDS_Shape& shape) {
    BRep_Builder builder;
    TopoDS_Compound compound;
    builder.MakeCompound(compound);
    for (TopExp_Explorer it(shape,TopAbs_SHELL);it.More();it.Next()) builder.Add(compound,it.Current());
    return compound;
}
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point t) {return std::chrono::duration<double>(Clock::now()-t).count();}

dynamics::ContactPose read_pose() {
    dynamics::ContactPose input;
    for (double& x:input.origin) std::cin>>x;
    for (double& x:input.axis) std::cin>>x;
    if (!(std::cin>>input.angle)) throw std::invalid_argument("incomplete pose");
    return input;
}
int regions(const TopoDS_Shape& a,const TopoDS_Shape& b) {
    const auto query=cad_solid_region_query(a,b,{},CadEventMethod::conservative_advancement);
    dynamics::ContactWork work;
    std::cout<<std::setprecision(17)<<"pose,scope,count,point_x_m,point_y_m,point_z_m,nx,ny,nz,depth_m\n";
    for (int i=0;(std::cin>>std::ws).peek()!=std::char_traits<char>::eof();++i) {
        const auto pa=read_pose(),pb=read_pose();
        const auto full=cad::solid_regions(a.Moved(TopLoc_Location(cad::placement(pa))),
                                           b.Moved(TopLoc_Location(cad::placement(pb))),work);
        const auto local=query->regions(pa,pb,work);
        const auto write=[&](const char* scope,const std::vector<dynamics::ContactRegion>& rows) {
            if (rows.empty()) std::cout<<i<<','<<scope<<",0,,,,,,,\n";
            for (const auto& r:rows)
                std::cout<<i<<','<<scope<<','<<rows.size()<<','<<r.point[0]<<','<<r.point[1]<<','<<r.point[2]
                         <<','<<r.normal[0]<<','<<r.normal[1]<<','<<r.normal[2]<<','<<r.depth_m<<'\n';
        };
        write("full",full); write("local",local);
    }
    return 0;
}

// Time the fixed-pose geometry used by nc_contact, without CCD or the integrator.
int profile_regions(const TopoDS_Shape& a,const TopoDS_Shape& b) {
    const auto query=cad_solid_region_query(a,b,{},CadEventMethod::conservative_advancement);
    std::cout<<std::setprecision(17);
    while ((std::cin>>std::ws).peek()!=std::char_traits<char>::eof()) {
        const auto pa=read_pose(),pb=read_pose();
        dynamics::ContactWork work;
        const auto start=Clock::now();
        std::vector<dynamics::ContactRegion> found;
        bool resolved=true;
        try {
            found=query->regions(pa,pb,work);
        } catch (const dynamics::UnresolvedContact& error) {
            resolved=false;
            std::cerr<<error.what()<<'\n';
        }
        const auto seconds=elapsed(start);
        std::cout<<"{\"query_s\":"<<seconds<<",\"resolved\":"<<std::boolalpha<<resolved
                 <<",\"index_s\":"<<work.cad_region_index_s<<",\"select_s\":"<<work.cad_region_select_s
                 <<",\"common_s\":"<<work.cad_region_common_s<<",\"extract_s\":"<<work.cad_region_extract_s
                 <<",\"boolean_queries\":"<<work.cad_boolean_queries
                 <<",\"cache_hits\":"<<work.cad_region_cache_hits<<",\"regions\":[";
        for (std::size_t i=0;i<found.size();++i) {
            const auto& r=found[i];
            if (i) std::cout<<',';
            std::cout<<'['<<r.point[0]<<','<<r.point[1]<<','<<r.point[2]<<','
                     <<r.normal[0]<<','<<r.normal[1]<<','<<r.normal[2]<<','<<r.depth_m<<']';
        }
        std::cout<<"]}"<<std::endl;
    }
    return 0;
}

// Independent local thickness check at a supplied point and normal, in SI units.
// Reject ambiguous edge/tangent hits and require material on the inside of both exit faces.
int normal_chords(const TopoDS_Shape& a,const TopoDS_Shape& b) {
    std::array<BRepClass3d_SolidClassifier,2> inside;
    std::array<IntCurvesFace_ShapeIntersector,2> rays;
    const double tolerance=Precision::Confusion(); // OCCT lengths are mm.
    for (int j=0;j<2;j++) {
        const auto& shape=j==0 ? a : b;
        inside[j].Load(shape);
        rays[j].Load(shape,tolerance);
    }
    std::cout<<std::setprecision(17);
    while ((std::cin>>std::ws).peek()!=std::char_traits<char>::eof()) {
        const std::array poses{read_pose(),read_pose()};
        Vec point{},normal{};
        double reach;
        for (double& x:point) std::cin>>x;
        for (double& x:normal) std::cin>>x;
        if (!(std::cin>>reach) || !(reach>0) || std::abs(dynamics::length(normal)-1)>1e-8)
            throw std::invalid_argument("expected point, unit normal and positive reach");
        const auto start=Clock::now();
        double depth=0;
        const char* status="resolved";
        for (int j=0;j<2;j++) {
            const auto& pose=poses[j];
            const auto origin=cad::point(dynamics::scale(pose.inverse_rotate(dynamics::sub(point,pose.origin)),1000));
            const auto d=pose.inverse_rotate(dynamics::scale(normal,j==0 ? 1. : -1.));
            const gp_Dir direction(d[0],d[1],d[2]);
            inside[j].Perform(origin,tolerance);
            if (inside[j].State()!=TopAbs_IN) {status="seed_not_strictly_inside"; break;}
            auto& ray=rays[j];
            ray.Perform(gp_Lin(origin,direction),0,1000*reach);
            if (!ray.IsDone() || !ray.NbPnt()) {status="no_boundary_hit"; break;}
            int first=1;
            for (int k=2;k<=ray.NbPnt();k++) if (ray.WParameter(k)<ray.WParameter(first)) first=k;
            if (ray.State(first)!=TopAbs_IN || ray.Transition(first)==IntCurveSurface_Tangent) {
                status="ambiguous_boundary_hit"; break;
            }
            inside[j].Perform(ray.Pnt(first).Translated(gp_Vec(direction)*(10*tolerance)),tolerance);
            if (inside[j].State()!=TopAbs_OUT) {status="exit_not_verified"; break;}
            depth+=.001*ray.WParameter(first);
        }
        const auto seconds=elapsed(start);
        std::cout<<"{\"status\":\""<<status<<"\",\"query_s\":"<<seconds<<",\"depth_m\":";
        if (std::string_view(status)=="resolved") std::cout<<depth;
        else std::cout<<"null";
        std::cout<<"}"<<std::endl;
    }
    return 0;
}

// Compare the two BVH starts on a recorded clear-to-loaded interval. The
// supplied poses interpolate their origins linearly and rotate on the short arc.
int entry(const TopoDS_Shape& a,const TopoDS_Shape& b) {
    const auto a0=read_pose(),b0=read_pose(),a1=read_pose(),b1=read_pose();
    double tol,cap;
    int splits;
    if (!(std::cin>>tol>>cap>>splits) || !(tol>0 && cap>tol) || splits<0 || splits>30)
        throw std::invalid_argument("expected positive tolerance and larger cap in mm, then bisection count");
    cad::RigidPath pa(a0,a1,{}),pb(b0,b1,{});
    std::cout<<std::setprecision(17)<<"start,lo,hi,extrema,witness_tests,witness_improvements,status\n";
    bool unresolved_volume=false;
    const auto bracket=[&](const auto& contact,bool keep_resolved_ends=false) {
        double lo=0.,hi=1.;
        if (contact(lo) || !contact(hi)) throw std::runtime_error("entry needs clear and loaded endpoints");
        for (int i=0;i<splits;++i) {
            const double mid=std::midpoint(lo,hi);
            try {
                if (contact(mid)) hi=mid;
                else lo=mid;
            } catch (const dynamics::UnresolvedContact&) {
                if (!keep_resolved_ends) throw;
                unresolved_volume=true;
                break;
            }
        }
        return std::pair{lo,hi};
    };
    for (bool reuse:{false,true}) {
        cad::LocalSolidSearch search(a,b);
        cad::BoundaryWitness witness;
        dynamics::SurfaceTransition work;
        const auto contact=[&](double t) {
            const auto relative=pb.at(t).Inverted()*pa.at(t);
            const auto parts=search.select(relative,cap);
            return !parts.separated && cad::bounded_distance(parts.a.boundary,parts.b.boundary,relative,
                                                              cap,tol,work,reuse ? &witness : nullptr)<=tol;
        };
        const auto [lo,hi]=bracket(contact);
        std::cout<<(reuse ? "previous_feature" : "cold")<<','<<lo<<','<<hi<<','<<work.cad_extrema<<','
                 <<work.cad_witness_tests<<','<<work.cad_witness_improvements<<",resolved\n";
    }
    dynamics::ContactWork work;
    const auto [lo,hi]=bracket([&](double t) {
        return !cad::solid_regions(a.Moved(TopLoc_Location(pa.at(t))),b.Moved(TopLoc_Location(pb.at(t))),work).empty();
    },true);
    std::cout<<"whole_solid_load,"<<lo<<','<<hi<<",0,0,0,"
             <<(unresolved_volume ? "unresolved_midpoint" : "resolved")<<'\n';
    return 0;
}
}

int main(int argc,char** argv) {
    try {
        const bool local=argc==5 && std::string_view(argv[4])=="--local";
        if (argc!=4 && !local) throw std::invalid_argument("nc_cad_distance_benchmark A.step B.step CAP_MM [--local] | --regions | --profile-regions | --normal-chords | --entry < poses.txt");
        const auto a=read_step_shape(argv[1]),b=read_step_shape(argv[2]);
        if (std::string_view(argv[3])=="--regions") return regions(a,b);
        if (std::string_view(argv[3])=="--profile-regions") return profile_regions(a,b);
        if (std::string_view(argv[3])=="--normal-chords") return normal_chords(a,b);
        if (std::string_view(argv[3])=="--entry") return entry(a,b);
        const double cap=std::stod(argv[3]),tol=1e-6;
        if (!(std::isfinite(cap) && cap>tol)) throw std::invalid_argument("invalid cap");
        const auto start=Clock::now();
        std::optional<cad::LocalSolidSearch> search;
        cad::LocalSolidPair pair;
        if (local) search.emplace(a,b);
        else pair={{a,cad::boundary_index(a)},{b,cad::boundary_index(b)},false};
        const double setup=elapsed(start);
        const auto sa=shells(a),sb=shells(b);
        std::cout<<std::setprecision(17)<<"setup_s,indexed_s,distance_mm,shell_s,shell_distance_mm,boxes,extrema,classifications,clipped_bodies\n";
        while ((std::cin>>std::ws).peek()!=std::char_traits<char>::eof()) {
            gp_Trsf pose[2];
            for (auto& transform:pose) transform=cad::placement(read_pose());
            dynamics::SurfaceTransition stats;
            const auto t=Clock::now();
            const auto relative=pose[1].Inverted()*pose[0];
            if (search) pair=search->select(relative,cap);
            const auto d=pair.separated ? cap : cad::bounded_distance(pair.a.boundary,pair.b.boundary,
                                                                      relative,cap,tol,stats);
            const auto indexed=elapsed(t);
            const auto s=Clock::now();
            BRepExtrema_DistShapeShape control(sa.Moved(TopLoc_Location(pose[0])),
                                               sb.Moved(TopLoc_Location(pose[1])),1e-7);
            if (!control.IsDone()) throw std::runtime_error("OCCT shell distance failed");
            std::cout<<setup<<','<<indexed<<','<<d<<','<<elapsed(s)<<','<<control.Value()<<','
                     <<stats.cad_box_tests<<','<<stats.cad_extrema<<','<<stats.cad_classifications<<','
                     <<(!pair.separated ? !pair.a.shape.IsSame(a)+!pair.b.shape.IsSame(b) : 0)<<std::endl;
        }
    } catch (const Standard_Failure& e) {std::cerr<<e.GetMessageString()<<'\n';return 1;}
      catch (const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
