#include "../../src/cad_boundary_distance.hpp"
#include "../../src/cad_local_solid.hpp"
#include "../../src/cad_rigid_path.hpp"
#include "nurbscontact/occt.hpp"
#include "nurbscontact/cad_contact_regions.hpp"
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepAlgoAPI_Common.hxx>
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

int pressure_patches(const TopoDS_Shape& a,const TopoDS_Shape& b,double reach) {
    using namespace dynamics;
    CadExitQuery ray_a{CadContactIndex(a)},ray_b{CadContactIndex(b)};
    std::cout<<std::setprecision(17);
    while ((std::cin>>std::ws).peek()!=std::char_traits<char>::eof()) {
        const std::array poses{read_pose(),read_pose()};
        ContactWork work;
        const auto start=Clock::now();
        std::vector<Vec> arms;
        std::vector<ContactRegion> regions;
        bool resolved=true;
        try {
            regions=cad::solid_regions(a.Moved(TopLoc_Location(cad::placement(poses[0]))),
                                       b.Moved(TopLoc_Location(cad::placement(poses[1]))),work,false,&arms);
            for (auto& region:regions) {
                std::size_t tests=0;
                auto chord=local_chord(ray_a,ray_b,poses,region.point,region.normal,reach,{},tests);
                if (!chord) {resolved=false;break;}
                region.depth_m=chord->region.depth_m;
            }
        } catch (const UnresolvedContact&) {resolved=false;}
        std::cout<<"{\"query_s\":"<<elapsed(start)<<",\"resolved\":"<<(resolved?"true":"false")<<",\"patches\":[";
        if (resolved) for (std::size_t i=0;i<regions.size();++i) {
            if (i) std::cout<<',';
            std::cout<<'[';
            for (double x:regions[i].point) std::cout<<x<<',';
            for (double x:regions[i].normal) std::cout<<x<<',';
            for (double x:arms[i]) std::cout<<x<<',';
            std::cout<<regions[i].depth_m<<']';
        }
        std::cout<<"]}"<<std::endl;
    }
    return 0;
}

// Time the fixed-pose geometry used by nc_contact, without CCD or the integrator.
int profile_regions(const TopoDS_Shape& a,const TopoDS_Shape& b,double maximum_depth=0.) {
    const auto query=maximum_depth ? cad_local_region_query(a,b,{},maximum_depth)
                                  : cad_solid_region_query(a,b,{},CadEventMethod::conservative_advancement);
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
        std::array<double,2> offsets{};
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
            offsets[j]=.001*ray.WParameter(first);
            depth+=offsets[j];
        }
        const auto seconds=elapsed(start);
        std::cout<<"{\"status\":\""<<status<<"\",\"query_s\":"<<seconds<<",\"depth_m\":";
        if (std::string_view(status)=="resolved")
            std::cout<<depth<<",\"offsets_m\":["<<offsets[0]<<','<<offsets[1]<<']';
        else std::cout<<"null";
        std::cout<<"}"<<std::endl;
    }
    return 0;
}

// A row supplies one pose and all region seeds. Only the previous pose's
// successful face IDs are hints. Each segment still traverses the face tree.
int local_chords(const TopoDS_Shape& a,const TopoDS_Shape& b,bool group=false) {
    const auto setup=Clock::now();
    CadExitQuery first{CadContactIndex(a)}, second{CadContactIndex(b)};
    std::cerr<<"index_s "<<std::setprecision(17)<<elapsed(setup)<<'\n';
    struct Saved { Vec anchor; std::array<std::size_t,2> faces; };
    std::vector<Saved> previous;
    std::cout<<std::setprecision(17);
    while ((std::cin>>std::ws).peek()!=std::char_traits<char>::eof()) {
        const std::array poses{read_pose(),read_pose()};
        double reach;
        int count;
        if (!(std::cin>>reach>>count) || !(reach>0) || count<0) throw std::invalid_argument("invalid seed group");
        std::vector<Saved> next;
        std::vector<std::optional<CadChord>> witnesses;
        if (group) std::cout<<"{\"chords\":";
        std::cout<<"[";
        for (int i=0;i<count;++i) {
            Vec point{},normal{};
            for (auto& x:point) std::cin>>x;
            for (auto& x:normal) std::cin>>x;
            if (!std::cin) throw std::invalid_argument("incomplete region seed");
            const auto start=Clock::now();
            const auto anchor=poses[0].inverse_rotate(dynamics::sub(point,poses[0].origin));
            std::array<std::size_t,2> hint{};
            double best=infinity;
            for (const auto& old:previous) {
                const auto d=dynamics::squared(dynamics::sub(anchor,old.anchor));
                if (d<best) { best=d; hint=old.faces; }
            }
            std::size_t tests=0;
            const auto chord=local_chord(first,second,poses,point,normal,reach,hint,tests);
            const double t=elapsed(start);
            if (group) witnesses.push_back(chord);
            if (i) std::cout<<',';
            std::cout<<"{\"query_s\":"<<t<<",\"face_tests\":"<<tests<<",\"depth_m\":";
            if (chord) {
                const auto& x=chord->exits[0]; const auto& y=chord->exits[1];
                next.push_back({anchor,{x.face,y.face}});
                const auto na=poses[0].rotate(x.outward),nb=poses[1].rotate(y.outward);
                const auto n=dynamics::sub(na,nb);
                const auto bisector=dynamics::scale(n,1./dynamics::length(n));
                std::cout<<chord->region.depth_m<<",\"offsets_m\":["<<x.distance_m<<','<<y.distance_m
                         <<"],\"faces\":["<<x.face<<','<<y.face<<"],\"hint_matches\":"
                         <<int(hint[0]==x.face)+int(hint[1]==y.face)
                         <<",\"normal_cos\":"<<dynamics::dot3(bisector,normal)
                         <<",\"opposition\":"<<-dynamics::dot3(na,nb)
                         <<",\"bisector\":["<<bisector[0]<<','<<bisector[1]<<','<<bisector[2]<<']';
            } else std::cout<<"null";
            std::cout<<"}";
        }
        previous=std::move(next);
        std::cout<<"]";
        if (group) {
            const auto t=Clock::now();
            const auto labels=chord_groups(first,second,poses,witnesses,reach);
            std::cout<<",\"group_s\":"<<elapsed(t)<<",\"labels\":[";
            for (int i=0;i<count;++i) {if (i) std::cout<<','; std::cout<<labels[i];}
            std::cout<<"]}";
        }
        std::cout<<std::endl;
    }
    return 0;
}

// Independent audit of witness coverage. Label using the actual connected
// volumes of the two complete STEP solids, without a face-ID correspondence.
int memberships(const TopoDS_Shape& a,const TopoDS_Shape& b) {
    while ((std::cin>>std::ws).peek()!=std::char_traits<char>::eof()) {
        const auto pa=read_pose(),pb=read_pose();
        int count;
        if (!(std::cin>>count) || count<0) throw std::invalid_argument("invalid number of witnesses");
        std::vector<gp_Pnt> points;
        for (int i=0;i<count;++i) {
            Vec v{};
            for (auto& x:v) if (!(std::cin>>x)) throw std::invalid_argument("incomplete witness");
            points.push_back(cad::point(dynamics::scale(v,1000)));
        }
        BRepAlgoAPI_Common op;
        TopTools_ListOfShape args,tools;
        args.Append(a.Moved(TopLoc_Location(cad::placement(pa))));
        tools.Append(b.Moved(TopLoc_Location(cad::placement(pb))));
        op.SetArguments(args); op.SetTools(tools);
        op.SetNonDestructive(true); op.SetUseOBB(true); op.Build();
        if (!op.IsDone() || op.HasErrors() || op.HasWarnings()) {
            std::cout<<"{\"resolved\":false}"<<std::endl; continue;
        }
        int components=0;
        std::vector<int> labels(count,-1); // -1 outside, -2 on boundary or ambiguous
        for (TopExp_Explorer it(op.Shape(),TopAbs_SOLID);it.More();it.Next(),++components) {
            BRepClass3d_SolidClassifier classifier(it.Current());
            for (int i=0;i<count;++i) {
                classifier.Perform(points[i],Precision::Confusion());
                const auto state=classifier.State();
                if (state==TopAbs_IN) labels[i]=labels[i]==-1 ? components : -2;
                else if (state!=TopAbs_OUT) labels[i]=-2;
            }
        }
        std::cout<<"{\"resolved\":true,\"count\":"<<components<<",\"labels\":[";
        for (int i=0;i<count;++i) {if (i) std::cout<<','; std::cout<<labels[i];}
        std::cout<<"]}"<<std::endl;
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
        const bool local_profile=argc==5 && std::string_view(argv[3])=="--profile-local-regions";
        const bool pressure=argc==5 && std::string_view(argv[3])=="--pressure-patches";
        if (argc!=4 && !local && !local_profile && !pressure) throw std::invalid_argument("nc_cad_distance_benchmark A.step B.step CAP_MM [--local] | --regions | --profile-regions | --profile-local-regions MAX_DEPTH_M | --pressure-patches REACH_M | --normal-chords | --local-chords | --entry < poses.txt");
        const auto a=read_step_shape(argv[1]),b=read_step_shape(argv[2]);
        if (std::string_view(argv[3])=="--regions") return regions(a,b);
        if (pressure) {
            const auto reach=std::stod(argv[4]);
            if (!(reach>0 && std::isfinite(reach))) throw std::invalid_argument("invalid patch reach");
            return pressure_patches(a,b,reach);
        }
        if (local_profile) {
            const auto limit=std::stod(argv[4]);
            if (!(limit>0 && std::isfinite(limit))) throw std::invalid_argument("invalid local depth limit");
            return profile_regions(a,b,limit);
        }
        if (std::string_view(argv[3])=="--profile-regions") return profile_regions(a,b);
        if (std::string_view(argv[3])=="--normal-chords") return normal_chords(a,b);
        if (std::string_view(argv[3])=="--local-chords") return local_chords(a,b);
        if (std::string_view(argv[3])=="--group-chords") return local_chords(a,b,true);
        if (std::string_view(argv[3])=="--memberships") return memberships(a,b);
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
