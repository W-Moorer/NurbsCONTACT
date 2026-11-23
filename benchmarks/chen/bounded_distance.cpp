#include "../../src/cad_boundary_distance.hpp"
#include "../../src/cad_local_solid.hpp"
#include "../../src/cad_rigid_path.hpp"
#include "nurbscontact/occt.hpp"
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRep_Builder.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS_Compound.hxx>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
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
}

int main(int argc,char** argv) {
    try {
        const bool local=argc==5 && std::string_view(argv[4])=="--local";
        if (argc!=4 && !local) throw std::invalid_argument("nc_cad_distance_benchmark A.step B.step CAP_MM [--local] | --regions < poses.txt");
        const auto a=read_step_shape(argv[1]),b=read_step_shape(argv[2]);
        if (std::string_view(argv[3])=="--regions") return regions(a,b);
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
            const auto d=pair.separated ? cap : cad::bounded_distance(*pair.a.boundary,*pair.b.boundary,
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
