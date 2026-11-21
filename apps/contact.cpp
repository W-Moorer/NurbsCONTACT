#include "nurbscontact/occt_contact.hpp"

#include <BRepAlgoAPI_Cut.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakeSphere.hxx>
#include <STEPControl_Writer.hxx>
#include <Standard_Failure.hxx>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>

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

struct Orbit {
    Vec arm, axis;
    double speed;
};

struct TranslationSample {
    double time;
    Vec displacement;
};

using Drive = std::variant<std::monostate,Orbit,std::vector<TranslationSample>>;

struct Body {
    std::shared_ptr<const ContactSurface> surface;
    double mass;
    ContactPose reference;
    Vec center_of_mass;
    std::vector<Translation> translations;
    int hinge;
    Vec pivot;
    Drive drive{};
};

struct Input {
    Vec gravity;
    std::vector<Coordinate> coordinates;
    std::vector<Body> bodies;
    std::vector<ContactPair> pairs;
    std::vector<PointResistance> resistances;
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
    const auto version=take<std::string>(in);
    const bool has_drives=version=="nc-contact-v2";
    if (version!="nc-contact-v1" && !has_drives) throw std::invalid_argument("unknown scene version");
    expect(in,"gravity");
    Input input;
    input.gravity=vector(in);
    const auto count=take<unsigned>(in);
    const auto bodies=take<unsigned>(in);
    const auto pairs=take<unsigned>(in);
    if ((!has_drives && count==0) || bodies<2 || pairs==0 || count>1024 || bodies>1024 || pairs>16384)
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
        else if (mode=="contained") direction=ContactDirection::contained;
        else if (mode=="contained-radial") direction=ContactDirection::contained_radial;
        else throw std::invalid_argument("unknown probe direction");
        ContactPair pair{a,b,vector(in),vector(in),direction,unit(vector(in)),
                         {number(in),number(in),number(in),number(in),number(in),number(in)}};
        const auto& m=pair.material;
        if (!(m.stiffness>0 && m.exponent>=1 && m.rate>=0 && m.friction>=0 && m.slip_speed>0 &&
                m.maximum_depth>0)) throw std::invalid_argument("invalid contact material");
        input.pairs.push_back(pair);
    }
    if (has_drives) {
        expect(in,"drives");
        const auto drives=take<unsigned>(in);
        if (drives>bodies) throw std::invalid_argument("too many drives");
        for (unsigned i=0;i<drives;++i) {
            const auto index=take<unsigned>(in);
            if (index>=bodies) throw std::invalid_argument("invalid drive body");
            auto& body=input.bodies[index];
            if (!std::holds_alternative<std::monostate>(body.drive))
                throw std::invalid_argument("duplicate body drive");
            const auto kind=take<std::string>(in);
            if (kind=="orbit") {
                body.drive=Orbit{vector(in),unit(vector(in)),number(in)};
            } else if (kind=="table") {
                if (body.hinge>=0 || !body.translations.empty())
                    throw std::invalid_argument("tabulated drive requires a prescribed body");
                const auto size=take<unsigned>(in);
                if (size<2 || size>10000000) throw std::invalid_argument("invalid drive table size");
                std::vector<TranslationSample> table;
                table.reserve(size);
                for (unsigned j=0;j<size;++j) {
                    TranslationSample item{number(in),vector(in)};
                    if ((j==0 && item.time!=0) || (j>0 && item.time<=table.back().time))
                        throw std::invalid_argument("table times must increase from zero");
                    table.push_back(item);
                }
                body.drive=std::move(table);
            } else {
                throw std::invalid_argument("unknown translation drive");
            }
        }
    }
    if (has_drives && !(in>>std::ws).eof()) {
        expect(in,"resistances");
        const auto resistances=take<unsigned>(in);
        if (resistances>16384) throw std::invalid_argument("too many resistances");
        for (unsigned i=0;i<resistances;++i) {
            expect(in,"slide");
            const auto a=take<unsigned>(in),b=take<unsigned>(in);
            if (a>=bodies || b>=bodies || a==b) throw std::invalid_argument("invalid resistance pair");
            PointResistance item{a,b,vector(in),vector(in),unit(vector(in)),number(in),number(in)};
            if (item.force_limit<0 || item.slip_speed<=0)
                throw std::invalid_argument("invalid resistance force or slip speed");
            input.resistances.push_back(item);
        }
    }
    if (!(in>>std::ws).eof()) throw std::invalid_argument("extra fields in scene");
    return input;
}

std::vector<ContactBodyState> kinematics(const Input& input,const Vector& q,double time,
                                         std::vector<Vec>* base_accelerations=nullptr) {
    std::vector<ContactBodyState> result;
    result.reserve(input.bodies.size());
    if (base_accelerations) base_accelerations->clear();
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
        Vec acceleration{};
        if (const auto* orbit=std::get_if<Orbit>(&body.drive)) {
            const Vec offset=ContactPose{{},orbit->axis,time*orbit->speed}.rotate(orbit->arm);
            const Vec spin=scale(orbit->axis,orbit->speed);
            state.pose.origin=add(state.pose.origin,offset);
            state.prescribed.linear=cross3(spin,offset);
            acceleration=cross3(spin,state.prescribed.linear);
        } else if (const auto* table=std::get_if<std::vector<TranslationSample>>(&body.drive)) {
            const auto end=std::upper_bound(table->begin(),table->end()-1,time,
                [](double t,const TranslationSample& row) { return t<row.time; });
            const auto& a=*(end-1);
            const auto& b=*end;
            state.prescribed.linear=scale(sub(b.displacement,a.displacement),1/(b.time-a.time));
            const Vec offset=add(a.displacement,scale(state.prescribed.linear,std::min(time,b.time)-a.time));
            state.pose.origin=add(state.pose.origin,offset);
        }
        if (base_accelerations) base_accelerations->push_back(acceleration);
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
    const double step_count=duration/dt;
    const double roundoff=8*std::numeric_limits<double>::epsilon()*step_count;
    const auto steps=static_cast<std::size_t>(std::ceil(step_count-roundoff));
    const auto setup=Clock::now();
    const Input input=read_scene(scene,deflection);
    for (const auto& body:input.bodies) {
        const auto* table=std::get_if<std::vector<TranslationSample>>(&body.drive);
        if (table && steps*dt>table->back().time+1e-12)
            throw std::invalid_argument("simulation extends beyond prescribed translation table");
    }
    const double setup_s=std::chrono::duration<double>(Clock::now()-setup).count();
    const auto n=input.coordinates.size();
    Vector mass,q,v;
    for (const auto& coordinate:input.coordinates) {
        mass.push_back(coordinate.mass);
        q.push_back(coordinate.position);
        v.push_back(coordinate.velocity);
    }
    ContactWork work;
    std::vector<Vec> base_acceleration;
    const auto force=[&](const Vector& q_,const Vector& v_,double time) {
        const auto state=kinematics(input,q_,time,&base_acceleration);
        Forces out{Vector(n),Vector(n*n),Vector(n*n)};
        for (std::size_t i=0;i<n;++i) {
            const auto& coordinate=input.coordinates[i];
            out.value[i]=coordinate.load-coordinate.stiffness*(q_[i]-coordinate.rest)-coordinate.damping*v_[i];
            out.stiffness[i*n+i]=coordinate.stiffness;
            out.damping[i*n+i]=coordinate.damping;
            for (std::size_t j=0;j<state.size();++j) {
                const auto& body=input.bodies[j];
                const auto offset=state[j].pose.rotate(body.center_of_mass);
                out.value[i]+=body.mass*dot3(sub(input.gravity,base_acceleration[j]),
                                             state[j].derivative[i].at(offset));
            }
        }
        if (!input.resistances.empty()) add_point_resistances(out,state,input.resistances,v_);
        if (!add_contact_forces(&out,state,input.pairs,v_,work))
            out.value[0]=std::numeric_limits<double>::quiet_NaN();
        return out;
    };
    HhtOptions options;
    options.absolute_tolerance=1e-9;
    options.relative_tolerance=1e-8;
    HhtStats stats;
    std::vector<State> samples;
    samples.reserve(steps/stride+2);
    const auto start=Clock::now();
    if (n>0) {
        Hht solver(mass,force,options);
        auto state=solver.initialize(q,v);
        samples.push_back(state);
        for (std::size_t s=1;s<=steps;++s) {
            solver.step(state,dt,stats);
            if (s%stride==0 || s==steps) samples.push_back(state);
        }
    } else {
        for (std::size_t s=0;s<=steps;++s) {
            if (s%stride==0 || s==steps) samples.push_back({s*dt,{},{},{},{}});
        }
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
    double replay_s=0;
    for (const auto& item:samples) {
        trace<<item.time;
        for (std::size_t i=0;i<n;++i) trace<<','<<item.q[i]<<','<<item.v[i]<<','<<item.a[i];
        const auto poses=kinematics(input,item.q,item.time);
        for (const auto& body:poses)
            trace<<','<<body.pose.origin[0]<<','<<body.pose.origin[1]<<','<<body.pose.origin[2]<<','<<body.pose.angle;
        if (n) {
            if (!add_contact_forces(nullptr,poses,input.pairs,item.v,audit,&values))
                throw std::runtime_error("invalid saved contact state");
        } else {
            const auto replay_start=Clock::now();
            if (!add_contact_forces(nullptr,poses,input.pairs,item.v,work,&values))
                throw std::runtime_error("invalid prescribed contact state");
            replay_s+=std::chrono::duration<double>(Clock::now()-replay_start).count();
        }
        for (const auto& value:values)
            trace<<','<<value.gap<<','<<value.normal_force<<','<<length(value.friction)<<','<<value.resolved;
        trace<<'\n';
    }
    auto report=open_output(output.string()+".stats.csv");
    report<<"wall_s,setup_s,steps,accepted,subdivisions,force_evaluations,max_residual,pairs,culled,rays,missing,triangle_tests\n"
        <<(n ? wall : replay_s)<<','<<setup_s<<','<<steps<<','<<stats.accepted_steps<<','<<stats.subdivisions<<','<<stats.evaluations
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
    } else if ((kind=="cylinder" || kind=="annulus") && argc==(kind=="cylinder" ? 6 : 7)) {
        const double outer=std::stod(argv[3]), height=std::stod(argv[4]);
        const double inner=kind=="annulus" ? std::stod(argv[5]) : 0;
        if (!(outer>inner && inner>=0 && height>0 && std::isfinite(outer+height)))
            throw std::invalid_argument("invalid cylinder dimensions");
        const gp_Ax2 axis(gp_Pnt(0,0,-height/2),gp_Dir(0,0,1));
        shape=BRepPrimAPI_MakeCylinder(axis,outer,height).Shape();
        if (inner>0) {
            const gp_Ax2 extended(gp_Pnt(0,0,-height/2-1),gp_Dir(0,0,1));
            BRepAlgoAPI_Cut cut(shape,BRepPrimAPI_MakeCylinder(extended,inner,height+2).Shape());
            if (!cut.IsDone()) throw std::runtime_error("could not cut fixture bore");
            shape=cut.Shape();
        }
        file=argv[argc-1];
    } else {
        throw std::invalid_argument("fixture syntax: make sphere R FILE, make box X Y Z CENTER_Z FILE, "
                                    "make cylinder R H FILE or make annulus R H BORE_R FILE (mm)");
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
