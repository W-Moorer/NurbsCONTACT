#include "nurbscontact/occt_contact.hpp"

#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeSphere.hxx>
#include <STEPControl_Writer.hxx>
#include <Standard_Failure.hxx>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using namespace nc;
using namespace nc::dynamics;
using Path = std::filesystem::path;
using Clock = std::chrono::steady_clock;

struct Coordinate {
    double mass, position, velocity, stiffness, damping, rest, load;
};

struct Translation {
    std::size_t coordinate;
    Vec direction;
};

struct Body {
    std::shared_ptr<const ContactSurface> surface;
    double mass;
    ContactPose reference;
    Vec center_of_mass;
    std::vector<Translation> translations;
    int hinge;
    Vec pivot;
};

struct Input {
    Vec gravity;
    std::vector<Coordinate> coordinates;
    std::vector<Body> bodies;
    std::vector<ContactPair> pairs;
};

template<class T> T take(std::istream& in) {
    T x{};
    if (!(in>>x)) throw std::invalid_argument("invalid or missing scene field");
    return x;
}

double number(std::istream& in) {
    const double x=take<double>(in);
    if (!std::isfinite(x)) throw std::invalid_argument("scene values must be finite");
    return x;
}

Vec vector(std::istream& in) { return {number(in),number(in),number(in)}; }

Vec unit(Vec v) {
    const double norm=length(v);
    if (norm<1e-12) throw std::invalid_argument("axis must be nonzero");
    return scale(v,1/norm);
}

void expect(std::istream& in,std::string_view expected) {
    if (take<std::string>(in)!=expected) throw std::invalid_argument("expected "+std::string(expected));
}

Input read_scene(const Path& path,double deflection) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("could not open scene "+path.string());
    expect(in,"nc-contact-v1");
    expect(in,"gravity");
    Input input;
    input.gravity=vector(in);
    const auto count=take<unsigned>(in);
    const auto bodies=take<unsigned>(in);
    const auto pairs=take<unsigned>(in);
    if (!(count>0 && bodies>=2 && pairs>0) || count>1024 || bodies>1024 || pairs>16384)
        throw std::invalid_argument("invalid scene counts");
    for (unsigned i=0;i<count;++i) {
        expect(in,"coordinate");
        Coordinate data{number(in),number(in),number(in),number(in),number(in),number(in),number(in)};
        if (!(data.mass>0 && data.stiffness>=0 && data.damping>=0))
            throw std::invalid_argument("invalid coordinate inertia or spring");
        input.coordinates.push_back(data);
    }
    std::map<Path,std::shared_ptr<const ContactSurface>> geometry;
    for (unsigned i=0;i<bodies;++i) {
        expect(in,"body");
        std::string filename;
        if (!(in>>std::quoted(filename))) throw std::invalid_argument("missing body STEP file");
        const auto file=path.parent_path()/filename;
        auto found=geometry.find(file);
        if (found==geometry.end()) found=geometry.emplace(file,contact_surface(read_step_shape(file),deflection)).first;
        Body body{found->second,number(in),{vector(in),unit(vector(in)),number(in)},vector(in),{},
                  take<int>(in),vector(in)};
        if (body.mass<0 || body.hinge<-1 || body.hinge>=static_cast<int>(count))
            throw std::invalid_argument("invalid body mass or hinge");
        const auto translations=take<unsigned>(in);
        if (translations>count) throw std::invalid_argument("invalid translation count");
        for (unsigned j=0;j<translations;++j) {
            const auto coordinate=take<unsigned>(in);
            if (coordinate>=count) throw std::invalid_argument("invalid translation coordinate");
            body.translations.push_back({coordinate,vector(in)});
        }
        input.bodies.push_back(std::move(body));
    }
    for (unsigned i=0;i<pairs;++i) {
        expect(in,"probe");
        const auto a=take<unsigned>(in),b=take<unsigned>(in);
        const auto mode=take<std::string>(in);
        if (a>=bodies || b>=bodies || a==b) throw std::invalid_argument("invalid probe pair");
        ContactDirection direction;
        if (mode=="world") direction=ContactDirection::world;
        else if (mode=="body-a") direction=ContactDirection::body_a;
        else if (mode=="anchors") direction=ContactDirection::anchors;
        else throw std::invalid_argument("unknown probe direction");
        ContactPair pair{a,b,vector(in),vector(in),direction,unit(vector(in)),
                         {number(in),number(in),number(in),number(in),number(in),number(in)}};
        const auto& m=pair.material;
        if (!(m.stiffness>0 && m.exponent>=1 && m.rate>=0 && m.friction>=0 && m.slip_speed>0 &&
                m.maximum_depth>0)) throw std::invalid_argument("invalid contact material");
        input.pairs.push_back(pair);
    }
    if (!(in>>std::ws).eof()) throw std::invalid_argument("extra fields in scene");
    return input;
}

std::vector<ContactBodyState> kinematics(const Input& input,const Vector& q) {
    std::vector<ContactBodyState> result;
    result.reserve(input.bodies.size());
    for (const Body& body:input.bodies) {
        ContactBodyState state{body.surface.get(),body.reference,std::vector<Twist>(q.size())};
        if (body.hinge>=0) {
            const auto i=static_cast<std::size_t>(body.hinge);
            const ContactPose orbit{body.pivot,body.reference.axis,q[i]};
            const Vec arm=orbit.rotate(sub(body.reference.origin,body.pivot));
            state.pose.origin=add(body.pivot,arm);
            state.pose.angle+=q[i];
            state.derivative[i]={cross3(body.reference.axis,arm),body.reference.axis};
        }
        for (const auto& item:body.translations) {
            state.pose.origin=add(state.pose.origin,scale(item.direction,q[item.coordinate]));
            auto& data=state.derivative[item.coordinate].linear;
            data=add(data,item.direction);
        }
        result.push_back(std::move(state));
    }
    return result;
}

std::ofstream open_output(const Path& path) {
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write "+path.string());
    out<<std::setprecision(17);
    return out;
}

void simulate(const Path& scene,double deflection,double dt,double duration,unsigned stride,const Path& output) {
    if (!(std::isfinite(dt) && dt>0 && std::isfinite(duration) && duration>=dt && stride>0))
        throw std::invalid_argument("invalid step, duration or stride");
    const auto steps=static_cast<std::size_t>(std::ceil(duration/dt));
    const auto setup=Clock::now();
    const Input input=read_scene(scene,deflection);
    const double setup_s=std::chrono::duration<double>(Clock::now()-setup).count();
    const auto n=input.coordinates.size();
    Vector mass,q,v;
    for (const auto& coordinate:input.coordinates) {
        mass.push_back(coordinate.mass);
        q.push_back(coordinate.position);
        v.push_back(coordinate.velocity);
    }
    ContactWork work;
    const auto force=[&](const Vector& q_,const Vector& v_,double) {
        const auto state=kinematics(input,q_);
        Forces out{Vector(n),Vector(n*n),Vector(n*n)};
        for (std::size_t i=0;i<n;++i) {
            const auto& coordinate=input.coordinates[i];
            out.value[i]=coordinate.load-coordinate.stiffness*(q_[i]-coordinate.rest)-coordinate.damping*v_[i];
            out.stiffness[i*n+i]=coordinate.stiffness;
            out.damping[i*n+i]=coordinate.damping;
            for (std::size_t j=0;j<state.size();++j) {
                const auto& body=input.bodies[j];
                const auto offset=state[j].pose.rotate(body.center_of_mass);
                out.value[i]+=body.mass*dot3(input.gravity,state[j].derivative[i].at(offset));
            }
        }
        if (!add_contact_forces(&out,state,input.pairs,v_,work))
            out.value[0]=std::numeric_limits<double>::quiet_NaN();
        return out;
    };
    HhtOptions options;
    options.absolute_tolerance=1e-9;
    options.relative_tolerance=1e-8;
    Hht solver(mass,force,options);
    HhtStats stats;
    std::vector<State> samples;
    samples.reserve(steps/stride+2);
    const auto start=Clock::now();
    auto state=solver.initialize(q,v);
    samples.push_back(state);
    for (std::size_t s=1;s<=steps;++s) {
        solver.step(state,dt,stats);
        if (s%stride==0 || s==steps) samples.push_back(state);
    }
    const double wall=std::chrono::duration<double>(Clock::now()-start).count();
    auto trace=open_output(output.string()+".csv");
    trace<<"time_s";
    for (std::size_t i=0;i<n;++i) trace<<",q"<<i<<",v"<<i<<",a"<<i;
    for (std::size_t j=0;j<input.bodies.size();++j)
        trace<<",b"<<j<<"_x_m,b"<<j<<"_y_m,b"<<j<<"_z_m,b"<<j<<"_angle_rad";
    for (std::size_t p=0;p<input.pairs.size();++p)
        trace<<",p"<<p<<"_gap_m,p"<<p<<"_normal_N,p"<<p<<"_friction_N,p"<<p<<"_hit";
    trace<<'\n';
    ContactWork audit;
    std::vector<ContactSample> values;
    for (const auto& item:samples) {
        trace<<item.time;
        for (std::size_t i=0;i<n;++i) trace<<','<<item.q[i]<<','<<item.v[i]<<','<<item.a[i];
        const auto poses=kinematics(input,item.q);
        for (const auto& body:poses)
            trace<<','<<body.pose.origin[0]<<','<<body.pose.origin[1]<<','<<body.pose.origin[2]<<','<<body.pose.angle;
        if (!add_contact_forces(nullptr,poses,input.pairs,item.v,audit,&values))
            throw std::runtime_error("invalid saved contact state");
        for (const auto& value:values)
            trace<<','<<value.gap<<','<<value.normal_force<<','<<length(value.friction)<<','<<value.resolved;
        trace<<'\n';
    }
    auto report=open_output(output.string()+".stats.csv");
    report<<"wall_s,setup_s,steps,accepted,subdivisions,force_evaluations,max_residual,pairs,culled,rays,missing,triangle_tests\n"
        <<wall<<','<<setup_s<<','<<steps<<','<<stats.accepted_steps<<','<<stats.subdivisions<<','<<stats.evaluations
        <<','<<stats.maximum_residual<<','<<work.pairs<<','<<work.culled<<','<<work.rays<<','<<work.missing
        <<','<<work.mesh.triangles<<'\n';
    if (!trace || !report) throw std::runtime_error("failed to write contact run");
}

void export_fixture(int argc,char** argv) {
    const std::string_view kind=argv[2];
    TopoDS_Shape shape;
    Path file;
    if (kind=="sphere" && argc==5) {
        const double radius=std::stod(argv[3]);
        if (!(radius>0 && std::isfinite(radius))) throw std::invalid_argument("invalid sphere radius");
        const gp_Ax2 axis(gp_Pnt(0,0,0),gp_Dir(.24,.21,1));
        shape=BRepPrimAPI_MakeSphere(axis,radius).Shape();
        file=argv[4];
    } else if (kind=="box" && argc==8) {
        const double x=std::stod(argv[3]),y=std::stod(argv[4]),z=std::stod(argv[5]);
        const double z_center=std::stod(argv[6]);
        if (!(x>0 && y>0 && z>0 && std::isfinite(x+y+z+z_center)))
            throw std::invalid_argument("invalid box dimensions");
        shape=BRepPrimAPI_MakeBox(gp_Pnt(-x/2,-y/2,z_center-z/2),x,y,z).Shape();
        file=argv[7];
    } else {
        throw std::invalid_argument("fixture syntax: make sphere R_MM FILE or make box X_MM Y_MM Z_MM CENTER_Z_MM FILE");
    }
    if (!file.parent_path().empty()) std::filesystem::create_directories(file.parent_path());
    STEPControl_Writer writer;
    if (writer.Transfer(shape,STEPControl_AsIs)!=IFSelect_RetDone || writer.Write(file.string().c_str())!=IFSelect_RetDone)
        throw std::runtime_error("could not write STEP fixture");
}

} // namespace

int main(int argc,char** argv) {
    try {
        if (argc>=3 && std::string_view(argv[1])=="make") {
            export_fixture(argc,argv);
            return 0;
        }
        if (argc!=7) {
            std::cerr<<"nc_contact SCENE DEFLECTION_M DT_S DURATION_S OUTPUT_STRIDE OUTPUT_PREFIX\n";
            return 2;
        }
        simulate(argv[1],std::stod(argv[2]),std::stod(argv[3]),std::stod(argv[4]),
                 static_cast<unsigned>(std::stoul(argv[5])),argv[6]);
    } catch (const Standard_Failure& error) {
        std::cerr<<error.GetMessageString()<<'\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr<<error.what()<<'\n';
        return 1;
    }
}
