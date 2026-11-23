#include "../../src/cad_boundary_distance.hpp"
#include "../../src/cad_rigid_path.hpp"
#include "nurbscontact/occt.hpp"
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRep_Builder.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS_Compound.hxx>
#include <chrono>
#include <iomanip>
#include <iostream>

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
}

int main(int argc,char** argv) {
    try {
        if (argc!=4) throw std::invalid_argument("nc_cad_distance_benchmark A.step B.step CAP_MM < poses.txt");
        const double cap=std::stod(argv[3]),tol=1e-6;
        if (!(std::isfinite(cap) && cap>tol)) throw std::invalid_argument("invalid cap");
        const auto a=read_step_shape(argv[1]),b=read_step_shape(argv[2]);
        const auto start=Clock::now();
        const auto ia=cad::boundary_index(a),ib=cad::boundary_index(b);
        const double setup=elapsed(start);
        const auto sa=shells(a),sb=shells(b);
        std::cout<<std::setprecision(17)<<"setup_s,indexed_s,distance_mm,shell_s,shell_distance_mm,boxes,extrema,classifications\n";
        while ((std::cin>>std::ws).peek()!=std::char_traits<char>::eof()) {
            gp_Trsf pose[2];
            for (auto& transform:pose) {
                dynamics::ContactPose input;
                for (double& x:input.origin) std::cin>>x;
                for (double& x:input.axis) std::cin>>x;
                if (!(std::cin>>input.angle)) throw std::invalid_argument("incomplete pose");
                transform=cad::placement(input);
            }
            dynamics::SurfaceTransition stats;
            const auto t=Clock::now();
            const auto d=cad::bounded_distance(*ia,*ib,pose[1].Inverted()*pose[0],cap,tol,stats);
            const auto indexed=elapsed(t);
            const auto s=Clock::now();
            BRepExtrema_DistShapeShape control(sa.Moved(TopLoc_Location(pose[0])),
                                               sb.Moved(TopLoc_Location(pose[1])),1e-7);
            if (!control.IsDone()) throw std::runtime_error("OCCT shell distance failed");
            std::cout<<setup<<','<<indexed<<','<<d<<','<<elapsed(s)<<','<<control.Value()<<','
                     <<stats.cad_box_tests<<','<<stats.cad_extrema<<','<<stats.cad_classifications<<std::endl;
        }
    } catch (const Standard_Failure& e) {std::cerr<<e.GetMessageString()<<'\n';return 1;}
      catch (const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
