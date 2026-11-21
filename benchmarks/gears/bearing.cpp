#include "../step_triangles.hpp"
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include "nurbscontact/hht.hpp"
#include "nurbscontact/occt.hpp"

#include <BRepAdaptor_Surface.hxx>
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <STEPControl_Writer.hxx>
#include <gp_Cylinder.hxx>

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string_view>

namespace {

using namespace nc;
using namespace nc::dynamics;
using Path=std::filesystem::path;
using Clock=std::chrono::steady_clock;
constexpr auto pi=std::numbers::pi;
constexpr std::size_t count=14;
constexpr double inner_race=27.965, outer_race=36.035, roller_radius=4;
constexpr double radial_skin=.052, stiffness=2.7e8;
constexpr double shaft_speed=1800.*2*pi/60;
constexpr double cage_speed=shaft_speed/2*(1-roller_radius/32.);
constexpr double roller_spin=-shaft_speed*inner_race/(2*roller_radius);

std::ofstream output(const Path& path) {
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream f(path);
    f.exceptions(std::ios::badbit|std::ios::failbit);
    f << std::setprecision(17);
    return f;
}

TopoDS_Shape cylinder(double r,double width) {
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0,0,-width/2),gp_Dir(0,0,1)),r,width).Shape();
}

void write_step(const TopoDS_Shape& model,const Path& file) {
    STEPControl_Writer writer;
    if (!BRepCheck_Analyzer(model).IsValid() ||
        writer.Transfer(model,STEPControl_AsIs)!=IFSelect_RetDone ||
        writer.Write(file.string().c_str())!=IFSelect_RetDone) {
        throw std::runtime_error("failed to export "+file.string());
    }
}

void make_models(const Path& folder) {
    std::filesystem::create_directories(folder);
    write_step(BRepAlgoAPI_Cut(cylinder(inner_race,19),cylinder(22.5,21)).Shape(),folder/"inner.step");
    write_step(BRepAlgoAPI_Cut(cylinder(42.5,19),cylinder(outer_race,21)).Shape(),folder/"outer.step");
    write_step(cylinder(roller_radius,15),folder/"roller.step");
}

struct Race {
    double radius{};
    std::optional<TriangleBvh> mesh;

    Race(const Path& file,double nominal,double deflection) {
        TopoDS_Shape model=read_step_shape(file);
        for (TopExp_Explorer i(model,TopAbs_FACE);i.More();i.Next()) {
            const TopoDS_Face face=TopoDS::Face(i.Current());
            const BRepAdaptor_Surface surface(face);
            if (surface.GetType()!=GeomAbs_Cylinder) continue;
            const auto cylinder=surface.Cylinder();
            if (std::abs(cylinder.Radius()-nominal)>1e-6) continue;
            if (cylinder.Axis().Direction().Dot(gp_Dir(0,0,1))<.999999 ||
                std::hypot(cylinder.Location().X(),cylinder.Location().Y())>1e-6) {
                throw std::runtime_error("bearing axes differ from the model");
            }
            radius=cylinder.Radius();
            if (deflection>0) mesh.emplace(bench::step_triangles(face,deflection),4);
            return;
        }
        throw std::runtime_error("missing cylindrical STEP working face");
    }
};

struct BearingSample {
    std::array<double,count> forces{},radii{};
    double smallest=std::numeric_limits<double>::infinity();
};

struct Bearing {
    Race inner,outer,roller;
    TriangleStatistics work;
    std::size_t rays{}, misses{};
    BearingSample sample;

    Bearing(const Path& folder,double deflection)
        : inner(folder/"inner.step",inner_race,deflection),outer(folder/"outer.step",outer_race,deflection),
          roller(folder/"roller.step",roller_radius,deflection) {}

    double ray(const Race& surface,Vec root,Vec e,Pose shape) {
        ++rays;
        const auto hit=surface.mesh->ray_first(shape.inverse_rotate(sub(root,shape.origin)),
                                               shape.inverse_rotate(e),surface.radius+2,work);
        if (hit) return *hit;
        ++misses;
        return std::numeric_limits<double>::quiet_NaN();
    }

    static Vec load(double time) {
        const double modulation=.5*(1+std::tanh((time-.17)/.012));
        const double t=shaft_speed*time;
        return {2400+700*modulation+310*std::cos(t),450+310*std::sin(t),0};
    }

    Forces evaluate(const Vector& q,const Vector& v,double time) {
        const Vec center{1000*q[0],1000*q[1],0};
        const Vec applied=load(time);
        constexpr double damping=140;
        Forces force{{applied[0]-damping*v[0],applied[1]-damping*v[1]},Vector(4),{damping,0,0,damping}};
        const Pose inner_pose{center,std::cos(shaft_speed*time),std::sin(shaft_speed*time)};
        const Pose roll_pose{{},std::cos(roller_spin*time),std::sin(roller_spin*time)};
        sample={};
        for (std::size_t i=0;i<count;++i) {
            const double theta=cage_speed*time+2*pi*i/count;
            const Vec e{std::cos(theta),std::sin(theta),0};
            const double travel=dot3(center,e);
            const Vec cross=sub(center,scale(e,travel));
            const double radius=std::sqrt(inner.radius*inner.radius-squared(cross));
            double inner_hit=travel+radius;
            double outer_hit=outer.radius;
            double diameter=2*roller.radius;
            if (inner.mesh) {
                inner_hit=ray(inner,{},e,inner_pose);
                outer_hit=ray(outer,{},e,{});
                diameter=ray(roller,{},e,roll_pose)+ray(roller,{},scale(e,-1),roll_pose);
            }
            const double gap=outer_hit-inner_hit-diameter;
            if (!std::isfinite(gap) || gap<=.002) {
                force.value[0]=std::numeric_limits<double>::quiet_NaN();
                break;
            }
            sample.radii[i]=(inner_hit+outer_hit)/2;
            sample.smallest=std::min(sample.smallest,gap);
            const double f=stiffness*.001*std::max(0.,radial_skin-gap);
            sample.forces[i]=f;
            if (f==0) continue;
            const Vec jac=sub(scale(cross,1/radius),e);
            for (std::size_t j=0;j<2;++j) {
                force.value[j]+=jac[j]*f;
                for (std::size_t k=0;k<2;++k) force.stiffness[2*j+k]+=stiffness*jac[j]*jac[k];
            }
        }
        return force;
    }
};

struct Frame { double t; Vector q,v; BearingSample sample; };

void run(const Path& folder,double deflection,double dt,double duration,const Path& prefix,unsigned repeats) {
    if (!(dt>0 && duration>0 && repeats>0)) throw std::invalid_argument("invalid bearing protocol");
    const auto steps=static_cast<std::size_t>(std::llround(duration/dt));
    if (std::abs(steps*dt-duration)>1e-12) throw std::invalid_argument("nonintegral bearing step count");
    const auto prep_start=Clock::now();
    Bearing bearing(folder,deflection);
    const double setup=std::chrono::duration<double>(Clock::now()-prep_start).count();
    HhtOptions opts;
    opts.absolute_tolerance=1e-6;
    opts.relative_tolerance=1e-7;
    opts.minimum_clearance=.002;
    const auto clearance=[&](const Vector& a,const Vector& b) {
        // The norm of the linearly reconstructed center displacement attains
        // its upper bound at an endpoint. Distances here are in millimeters.
        const auto bound=[](const Vector& q) {
            return (norm(exact(Vec{q[0],q[1],0}))*1000).hi;
        };
        return (Interval{bearing.outer.radius}-bearing.inner.radius-2*bearing.roller.radius
                -std::max(bound(a),bound(b))).lo;
    };
    Hht hht({3.5,3.5},[&](const Vector& q,const Vector& v,double t){return bearing.evaluate(q,v,t);},opts,clearance);
    std::vector<Frame> frames;
    frames.reserve(steps+1);
    auto stats_out=output(prefix.string()+".stats.csv");
    stats_out << "repeat,deflection_mm,dt_s,duration_s,steps,wall_s,setup_s,accepted,subdivisions,"
                 "queries,misses,bvh_nodes,triangle_tests,triangles,min_interval_gap_mm\n";
    for (unsigned rep=0;rep<repeats;++rep) {
        bearing.work={};
        bearing.rays=bearing.misses=0;
        HhtStats stats;
        const auto start=Clock::now();
        State state=hht.initialize({21e-6,4e-6},{0,0});
        if (rep==0) frames.push_back({state.time,state.q,state.v,bearing.sample});
        for (std::size_t i=0;i<steps;++i) {
            hht.step(state,dt,stats);
            if (rep==0) {
                // The bare CAD guard has no force callback, so the last
                // converged HHT evaluation also supplies the current loads.
                frames.push_back({state.time,state.q,state.v,bearing.sample});
            }
        }
        const double wall=std::chrono::duration<double>(Clock::now()-start).count();
        const std::size_t triangles=bearing.inner.mesh ? bearing.inner.mesh->triangles().size()+
            bearing.outer.mesh->triangles().size()+bearing.roller.mesh->triangles().size() : 0;
        stats_out << rep << ',' << deflection << ',' << dt << ',' << duration << ',' << steps << ',' << wall << ','
            << setup << ',' << stats.accepted_steps << ',' << stats.subdivisions << ',' << bearing.rays << ','
            << bearing.misses << ',' << bearing.work.nodes << ',' << bearing.work.triangles << ',' << triangles << ','
            << stats.minimum_clearance << '\n' << std::flush;
        std::cout << "bearing " << deflection << " mm pass " << rep+1 << ' ' << wall << " s\n" << std::flush;
    }
    auto trajectory=output(prefix.string()+".csv");
    trajectory << "time,x_um,y_um,vx_m_s,vy_m_s,load_x_N,load_y_N,peak_roller_N,active,native_bound_um,backend_gap_um";
    for (std::size_t i=0;i<count;++i) trajectory << ",roller" << i << "_N,radius" << i << "_mm";
    trajectory << '\n';
    for (const Frame& frame:frames) {
        const Vec force=Bearing::load(frame.t);
        const auto& f=frame.sample.forces;
        trajectory << frame.t << ',' << frame.q[0]*1e6 << ',' << frame.q[1]*1e6 << ',' << frame.v[0] << ',' << frame.v[1]
            << ',' << force[0] << ',' << force[1] << ',' << *std::max_element(f.begin(),f.end()) << ','
            << std::count_if(f.begin(),f.end(),[](double value){return value>0;}) << ','
            << clearance(frame.q,frame.q)*1000 << ',' << frame.sample.smallest*1000;
        for (std::size_t i=0;i<count;++i) trajectory << ',' << f[i] << ',' << frame.sample.radii[i];
        trajectory << '\n';
    }
}

}  // namespace

int main(int argc,char** argv) {
    try {
        if (argc==3 && std::string_view(argv[1])=="models") make_models(argv[2]);
        else if (argc==8 && std::string_view(argv[1])=="run") {
            run(argv[2],std::stod(argv[3]),std::stod(argv[4]),std::stod(argv[5]),argv[6],
                static_cast<unsigned>(std::stoul(argv[7])));
        } else {
            std::cerr << "nc_roller_bearing models MODEL_DIR\n"
                         "nc_roller_bearing run MODEL_DIR DEFLECTION_MM DT_S DURATION_S PREFIX REPEATS\n";
            return 2;
        }
    } catch (const Standard_Failure& error) {
        std::cerr << error.GetMessageString() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
