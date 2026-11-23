#include "nurbscontact/cad_contact_regions.hpp"
#include "nurbscontact/free_rigid.hpp"
#include "nurbscontact/occt_contact.hpp"
#ifdef NC_WITH_CPU_MESH
#include "nurbscontact/cpu_mesh.hpp"
#endif

#include <BRepAlgoAPI_Cut.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <BRepBndLib.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepClass_FaceClassifier.hxx>
#include <BRepClass3d_SolidClassifier.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakeSphere.hxx>
#include <STEPControl_Writer.hxx>
#include <Standard_Failure.hxx>
#include <TopExp.hxx>
#include <TopoDS.hxx>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
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

struct LoadSample {
    double time, force;
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
    std::optional<FreeRigidBody> free{};
};

struct Accelerometer {
    std::size_t body;
    Vec point, axis;
};

struct Input {
    Vec gravity;
    std::vector<Coordinate> coordinates;
    std::vector<Body> bodies;
    std::vector<ContactPair> pairs;
    std::vector<std::pair<std::shared_ptr<const PairGeometry>,std::shared_ptr<const PairGeometry>>> event_geometry;
    std::vector<PointResistance> resistances;
    std::vector<TranslationalBushing> bushings;
    std::vector<Accelerometer> accelerometers;
    std::vector<std::vector<LoadSample>> loads;
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

struct SurfaceOptions {
    std::string method;
    Path mesh_directory;
    double distance_tolerance{1e-7};
};

Input read_scene(const Path& path,double deflection,std::optional<PairBackend> pair_backend,
                 PairSearchOptions pair_options,bool detect_events,const SurfaceOptions& surface_options) {
    if (!(deflection >= 0) || !std::isfinite(deflection)) throw std::invalid_argument("invalid mesh deflection");
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
    std::map<Path,TopoDS_Shape> shapes;
    std::vector<Path> files;
#ifdef NC_WITH_CPU_MESH
    std::map<Path,std::shared_ptr<const cpu::MeshData>> mesh_cache;
    std::vector<std::shared_ptr<const cpu::MeshData>> meshes;
#endif
    for (unsigned i=0;i<bodies;++i) {
        expect(in,"body");
        std::string filename;
        if (!(in>>std::quoted(filename))) throw std::invalid_argument("missing body STEP file");
        const auto file=path.parent_path()/filename;
        if (surface_options.method.empty()) {
            if (!shapes.contains(file)) shapes.emplace(file,read_step_shape(file));
        } else {
#ifdef NC_WITH_CPU_MESH
            Path asset=filename;
            if (asset.is_absolute()) asset=std::filesystem::relative(asset,path.parent_path());
            asset=(surface_options.mesh_directory / asset).replace_extension(".msh");
            auto& mesh=mesh_cache[asset];
            if (!mesh) mesh=std::make_shared<cpu::MeshData>(cpu::read_gmsh(asset));
            meshes.push_back(mesh);
#else
            throw std::invalid_argument("rebuild nc_contact with NC_WITH_CPU_MESH for mesh contact");
#endif
        }
        files.push_back(file);
        Body body{{},number(in),{vector(in),unit(vector(in)),number(in)},vector(in),{},
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
    std::map<Path,std::shared_ptr<const ContactSurface>> ray_geometry;
    std::map<Path,std::shared_ptr<const PairGeometry>> closest_geometry;
    std::map<Path,std::shared_ptr<const PairGeometry>> native_geometry;
    const auto load_pair_geometry=[&](std::size_t id,PairBackend backend) {
        const auto& file=files[id];
        auto& cache=backend==PairBackend::brep ? native_geometry : closest_geometry;
        auto it=cache.find(file);
        if (it==cache.end()) {
            it=cache.emplace(file,pair_geometry(shapes.at(file),backend,deflection)).first;
        }
        return it->second;
    };
    for (unsigned i=0;i<pairs;++i) {
        const auto type=take<std::string>(in);
        const auto a=take<unsigned>(in),b=take<unsigned>(in);
        if (a>=bodies || b>=bodies || a==b) throw std::invalid_argument("invalid probe pair");
        ContactPair pair{a,b,{},{},ContactDirection::world,{},{}};
        if (type=="surface") {
            if (surface_options.method.empty()) throw std::invalid_argument("surface pair requires a surface method");
        } else if (type=="closest") {
            if (!has_drives || !pair_backend) throw std::invalid_argument("closest requires a v2 scene and a pair backend");
            pair.activation=number(in);
            if (!(pair.activation>0)) throw std::invalid_argument("closest activation distance must be positive");
            pair.closest=pair_contact_query(load_pair_geometry(a,*pair_backend),load_pair_geometry(b,*pair_backend),pair_options);
        } else if (type=="probe") {
            const auto mode=take<std::string>(in);
            if (mode=="world") pair.direction=ContactDirection::world;
            else if (mode=="body-a") pair.direction=ContactDirection::body_a;
            else if (mode=="anchors") pair.direction=ContactDirection::anchors;
            else if (mode=="contained") pair.direction=ContactDirection::contained;
            else if (mode=="contained-radial") pair.direction=ContactDirection::contained_radial;
            else throw std::invalid_argument("unknown probe direction");
            pair.anchor_a=vector(in);
            pair.anchor_b=vector(in);
            pair.normal=unit(vector(in));
            if (surface_options.method.empty()) {
                for (auto id : {a,b}) {
                    const auto& file=files[id];
                    auto it=ray_geometry.find(file);
                    if (it==ray_geometry.end())
                        it=ray_geometry.emplace(file,contact_surface(shapes.at(file),deflection)).first;
                    input.bodies[id].surface=it->second;
                }
            }
        } else throw std::invalid_argument("expected surface, probe or closest contact");
        pair.material={number(in),number(in),number(in),number(in),number(in),number(in)};
        const auto& m=pair.material;
        if (!(m.stiffness>0 && m.exponent>=1 && m.rate>=0 && m.friction>=0 && m.slip_speed>0 &&
                m.maximum_depth>0)) throw std::invalid_argument("invalid contact material");
        if (detect_events && surface_options.method.empty()) {
            if (!pair.closest || count==0) throw std::invalid_argument("step CCD requires dynamic closest-pair contacts");
            input.event_geometry.emplace_back(load_pair_geometry(a,PairBackend::brep),load_pair_geometry(b,PairBackend::brep));
        }
        input.pairs.push_back(std::move(pair));
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
    bool read_resistances=false, read_bushings=false, read_accelerometers=false, read_impact_laws=false;
    bool read_free=false;
    while (has_drives && !(in>>std::ws).eof()) {
        const auto block=take<std::string>(in);
        if (block=="free-bodies" && !read_free) {
            read_free=true;
            const auto entries=take<unsigned>(in);
            if (entries>bodies || count+7*entries>1024) throw std::invalid_argument("too many free bodies");
            for (unsigned j=0;j<entries;++j) {
                expect(in,"free");
                const auto id=take<unsigned>(in);
                if (id>=bodies) throw std::invalid_argument("invalid free body index");
                auto& body=input.bodies[id];
                if (body.free || body.mass<=0 || body.hinge>=0 || !body.translations.empty() ||
                    !std::holds_alternative<std::monostate>(body.drive))
                    throw std::invalid_argument("free body requires positive mass and independent motion");
                const Vec diagonal=vector(in),off_diagonal=vector(in);
                const Vec velocity=vector(in),omega=vector(in);
                body.free.emplace(input.coordinates.size(),diagonal,off_diagonal);
                const Vec center=body.reference.point(body.center_of_mass);
                const auto q=body.free->initial_attitude(body.reference);
                const auto rate=body.free->initial_rate(q,omega);
                for (std::size_t k=0;k<3;++k)
                    input.coordinates.push_back({body.mass,center[k],velocity[k],0,0,0,0});
                for (std::size_t k=0;k<4;++k)
                    input.coordinates.push_back({1.,q[k],rate[k],0,0,0,0});
            }
            continue;
        }
        if (block=="loads" && input.loads.empty()) {
            const auto tables=take<unsigned>(in);
            if (count==0 || tables>count)
                throw std::invalid_argument("load tables require dynamic coordinates");
            input.loads.resize(count);
            for (unsigned i=0;i<tables;++i) {
                expect(in,"load");
                const auto index=take<unsigned>(in), size=take<unsigned>(in);
                if (index>=count || size<2 || size>10000000 || !input.loads[index].empty())
                    throw std::invalid_argument("invalid or duplicate coordinate load table");
                auto& table=input.loads[index];
                table.reserve(size);
                for (unsigned j=0;j<size;++j) {
                    const LoadSample sample{number(in),number(in)};
                    if ((j==0 && sample.time!=0) || (j>0 && sample.time<=table.back().time))
                        throw std::invalid_argument("load times must increase from zero");
                    table.push_back(sample);
                }
            }
            continue;
        }
        if (block=="impact-laws" && !read_impact_laws) {
            read_impact_laws=true;
            const auto laws=take<unsigned>(in);
            if (laws>pairs) throw std::invalid_argument("too many impact laws");
            for (unsigned i=0;i<laws;++i) {
                expect(in,"impact");
                const auto probe=take<unsigned>(in);
                const auto dissipation=number(in), floor=number(in);
                if (probe>=pairs || input.pairs[probe].material.approach_floor>0 || dissipation<0 || floor<=0)
                    throw std::invalid_argument("invalid or duplicate impact law");
                input.pairs[probe].material.impact_dissipation=dissipation;
                input.pairs[probe].material.approach_floor=floor;
            }
            continue;
        }
        if (block=="accelerometers" && !read_accelerometers) {
            read_accelerometers=true;
            const auto sensors=take<unsigned>(in);
            if (sensors>16384) throw std::invalid_argument("too many accelerometers");
            for (unsigned i=0;i<sensors;++i) {
                expect(in,"accelerometer");
                const auto body=take<unsigned>(in);
                if (body>=bodies) throw std::invalid_argument("invalid accelerometer body");
                input.accelerometers.push_back({body,vector(in),unit(vector(in))});
            }
            continue;
        }
        const bool sliding=block=="resistances" && !read_resistances;
        const bool elastic=block=="bushings" && !read_bushings;
        if (!sliding && !elastic) throw std::invalid_argument("unknown or duplicate element block");
        read_resistances |= sliding;
        read_bushings |= elastic;
        const auto elements=take<unsigned>(in);
        if (elements>16384) throw std::invalid_argument("too many elements in scene block");
        for (unsigned i=0;i<elements;++i) {
            expect(in,sliding ? "slide" : "bushing");
            const auto a=take<unsigned>(in),b=take<unsigned>(in);
            if (a>=bodies || b>=bodies || a==b) throw std::invalid_argument("invalid element pair");
            const auto anchor_a=vector(in),anchor_b=vector(in);
            if (sliding) {
                PointResistance item{a,b,anchor_a,anchor_b,unit(vector(in)),number(in),number(in)};
                if (item.force_limit<0 || item.slip_speed<=0)
                    throw std::invalid_argument("invalid resistance force or slip speed");
                input.resistances.push_back(item);
            } else {
                TranslationalBushing item{a,b,anchor_a,anchor_b,vector(in),vector(in),vector(in)};
                for (std::size_t axis=0;axis<3;++axis) {
                    if (item.stiffness[axis]<0 || item.damping[axis]<0)
                        throw std::invalid_argument("negative bushing stiffness or damping");
                }
                input.bushings.push_back(item);
            }
        }
    }
    if (!(in>>std::ws).eof()) throw std::invalid_argument("extra fields in scene");
    if (read_impact_laws && input.coordinates.empty()) throw std::invalid_argument("impact laws require dynamic coordinates");
    if (detect_events && !surface_options.method.empty()) {
        for (const auto& body : input.bodies) {
            if (!std::holds_alternative<std::monostate>(body.drive))
                throw std::invalid_argument("mesh entry checks currently require affine body origins between step endpoints");
        }
    }
#ifdef NC_WITH_CPU_MESH
    if (!surface_options.method.empty()) {
        const auto method=cpu::mesh_method(surface_options.method);
        const auto center=[&](std::size_t id) -> Vec {
            const auto& body=input.bodies[id];
            if (!detect_events) return {};
            if (body.free) return body.center_of_mass;
            if (body.hinge>=0) return body.reference.inverse_rotate(sub(body.pivot,body.reference.origin));
            return {};
        };
        for (auto& pair:input.pairs)
            pair.regions=cpu::mesh_region_query(meshes[pair.a],meshes[pair.b],method,
                                                {center(pair.a),center(pair.b)});
    }
#endif
    for (std::size_t i=0;i<input.event_geometry.size();++i) {
        const auto& pair=input.pairs[i];
        const auto& a=input.bodies[pair.a];
        const auto& b=input.bodies[pair.b];
        if (a.free || b.free || a.hinge>=0 || (b.hinge>=0 && b.pivot!=b.reference.origin) ||
            !std::holds_alternative<std::monostate>(a.drive) || !std::holds_alternative<std::monostate>(b.drive)) {
            throw std::invalid_argument("step CCD requires affine origin motion with no rotation of body A");
        }
    }
    return input;
}

std::vector<ContactBodyState> kinematics(const Input& input,const Vector& q,double time,
                                         std::vector<Vec>* base_accelerations=nullptr) {
    std::vector<ContactBodyState> result;
    result.reserve(input.bodies.size());
    if (base_accelerations) base_accelerations->clear();
    for (const Body& body:input.bodies) {
        ContactBodyState state{body.surface.get(),body.reference,std::vector<Twist>(q.size())};
        if (body.free) body.free->kinematics(q,body.center_of_mass,state);
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

double specific_force(const Accelerometer& sensor,const Body& body,const ContactBodyState& pose,
                      const State& state,Vec base_acceleration,Vec gravity) {
    if (body.free) {
        const auto i=body.free->first;
        const Vec a{state.a[i],state.a[i+1],state.a[i+2]};
        const Vec offset=sub(sensor.point,body.center_of_mass);
        const Vec omega=body.free->body_angular_velocity(state.q,state.v);
        const Vec alpha=body.free->body_angular_acceleration(state);
        const Vec local_acc=add(cross3(alpha,offset),cross3(omega,cross3(omega,offset)));
        return dot3(add(pose.pose.inverse_rotate(sub(a,gravity)),local_acc),sensor.axis);
    }
    const Vec arm=pose.pose.rotate(sensor.point);
    Vec acceleration=sub(base_acceleration,gravity);
    for (std::size_t i=0;i<state.a.size();++i) {
        acceleration=add(acceleration,scale(pose.derivative[i].at(arm),state.a[i]));
    }
    if (body.hinge>=0) {
        const auto i=static_cast<std::size_t>(body.hinge);
        const ContactPose hinge{body.pivot,body.reference.axis,state.q[i]};
        const Vec pivot_arm=hinge.rotate(sub(body.reference.origin,body.pivot));
        const Vec spin=scale(body.reference.axis,state.v[i]);
        acceleration=add(acceleration,cross3(spin,cross3(spin,add(pivot_arm,arm))));
    }
    return dot3(acceleration,pose.pose.rotate(sensor.axis));
}

std::vector<ImpactHistory> impact_history(const Vector& history) {
    std::vector<ImpactHistory> result;
    result.reserve(history.size()/3);
    for (std::size_t i=0;i<history.size();i+=3)
        result.push_back({history[i],history[i+1],history[i+2]});
    return result;
}

std::ofstream open_output(const Path& path) {
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write "+path.string());
    out<<std::setprecision(17);
    return out;
}

void report_line_loads(const CadContactIndex& a,const CadContactIndex& b,ContactPose pose,double cutoff,
                       double stiffness,const std::string& prefix,std::span<const CadContactSeed> seeds) {
    const auto query_start=Clock::now();
    auto regions=a.query(b,pose,cutoff);
    const double query_s=std::chrono::duration<double>(Clock::now()-query_start).count();
    auto loads=open_output(prefix+".csv");
    loads<<"mode,face_a,face_b,length_m,samples,force_ax_N,force_ay_N,force_az_N,"
           "moment_ax_Nm,moment_ay_Nm,moment_az_Nm,force_bx_N,force_by_N,force_bz_N,"
           "moment_bx_Nm,moment_by_Nm,moment_bz_Nm\n";
    auto stats=open_output(prefix+".stats.csv");
    stats<<"mode,lines,unresolved_face_pairs,unresolved_seeds,replaced_witnesses,"
           "max_witness_shift_m,cold_query_s,tracking_s,assembly_s\n";
    const auto write_loads=[&](std::string_view mode,std::size_t unresolved_seeds,std::size_t replaced,
                               double shift,double tracking_s) {
        const auto start=Clock::now();
        const auto lines=a.straight_lines(b,pose,cutoff,1e-7,regions);
        for (const auto& line:lines.lines) {
            double length=0;
            for (const auto& sample:line.samples) length+=sample.weight_m;
            const auto wrenches=line_contact_wrenches(line.samples,cutoff,stiffness,{},pose.origin);
            loads<<mode<<','<<line.face_a<<','<<line.face_b<<','<<length<<','<<line.samples.size();
            for (const auto* wrench:{&wrenches.a,&wrenches.b}) {
                for (double component:wrench->force) loads<<','<<component;
                for (double component:wrench->moment) loads<<','<<component;
            }
            loads<<'\n';
        }
        stats<<mode<<','<<lines.lines.size()<<','<<lines.unresolved_face_pairs<<','<<unresolved_seeds<<','
             <<replaced<<','<<shift<<','<<query_s<<','<<tracking_s<<','
             <<std::chrono::duration<double>(Clock::now()-start).count()<<'\n';
    };
    write_loads("cold",0,0,0,0);
    if (!seeds.empty()) {
        const auto start=Clock::now();
        const auto tracked=a.track_interiors(b,pose,cutoff,seeds);
        const double tracking_s=std::chrono::duration<double>(Clock::now()-start).count();
        std::size_t replaced=0,unresolved=tracked.unresolved_face_pairs;
        double shift=0;
        for (const auto& witness:tracked.candidates) {
            const auto matches=[&](const auto& candidate) {
                return candidate.facing_interiors && candidate.a.face==witness.a.face
                                                   && candidate.b.face==witness.b.face;
            };
            const auto cold=std::find_if(regions.candidates.begin(),regions.candidates.end(),matches);
            if (cold==regions.candidates.end()) {
                ++unresolved;
                continue;
            }
            const Vec difference=sub(witness.points.a,cold->points.a);
            shift=std::max(shift,length(difference));
            std::erase_if(regions.candidates,matches);
            regions.candidates.push_back(witness);
            ++replaced;
        }
        write_loads("tracked",unresolved,replaced,shift,tracking_s);
    }
    if (!loads || !stats) throw std::runtime_error("failed to write CAD line loads");
}

void contact_regions(int argc,char** argv,bool line_load) {
    const int shift=line_load ? 1 : 0;
    if (argc!=13+shift && argc!=14+shift) {
        throw std::invalid_argument(line_load ?
            "nc_contact line-load A.step B.step CUTOFF_M STIFFNESS_N_PER_M2 TX TY TZ AX AY AZ ANGLE_RAD OUTPUT_PREFIX [SEEDS.txt]" :
            "nc_contact regions A.step B.step CUTOFF_M TX TY TZ AX AY AZ ANGLE_RAD OUTPUT_PREFIX [SEEDS.txt]");
    }
    const auto scalar=[&](int i) {
        std::size_t end=0;
        const std::string value(argv[i]);
        const double x=std::stod(value,&end);
        if (end!=value.size() || !std::isfinite(x)) throw std::invalid_argument("invalid contact region argument");
        return x;
    };
    const double cutoff=scalar(4);
    const ContactPose pose{{scalar(5+shift),scalar(6+shift),scalar(7+shift)},
                           unit({scalar(8+shift),scalar(9+shift),scalar(10+shift)}),scalar(11+shift)};
    std::vector<CadContactSeed> seeds;
    if (argc==14+shift) {
        std::ifstream input(argv[13+shift]);
        if (!input) throw std::invalid_argument("could not open CAD contact seed file");
        while (!(input>>std::ws).eof()) {
            CadContactSeed seed;
            if (!(input>>seed.face_a>>seed.face_b>>seed.uv_a[0]>>seed.uv_a[1]>>seed.uv_b[0]>>seed.uv_b[1])) {
                throw std::invalid_argument("invalid CAD contact seed");
            }
            seeds.push_back(seed);
        }
        if (seeds.empty()) throw std::invalid_argument("CAD contact seed file is empty");
    }
    const auto setup_start=Clock::now();
    const CadContactIndex a{read_step_shape(argv[2])},b{read_step_shape(argv[3])};
    if (line_load) {
        const double stiffness=scalar(5);
        if (stiffness<0) throw std::invalid_argument("invalid line stiffness");
        return report_line_loads(a,b,pose,cutoff,stiffness,argv[13],seeds);
    }
    const auto start=Clock::now();
    const auto report=seeds.empty() ? a.query(b,pose,cutoff) : a.track_interiors(b,pose,cutoff,seeds);
    const auto end=Clock::now();

    const auto kind=[](FeatureKind feature) {
        switch (feature) {
        case FeatureKind::face: return "face";
        case FeatureKind::edge: return "edge";
        case FeatureKind::vertex: return "vertex";
        }
        return "";
    };
    auto points=open_output(std::string(argv[12])+".csv");
    points<<"face_a,face_b,support_a,support_b,distance_m,"
             "ax_m,ay_m,az_m,bx_m,by_m,bz_m,ua,va,ub,vb,facing_interiors\n";
    for (const auto& candidate:report.candidates) {
        points<<candidate.a.face<<','<<candidate.b.face<<','<<kind(candidate.a.support)<<','
              <<kind(candidate.b.support)<<','<<candidate.points.distance;
        for (double coordinate:candidate.points.a) points<<','<<coordinate;
        for (double coordinate:candidate.points.b) points<<','<<coordinate;
        for (const auto* sample:{&candidate.a,&candidate.b}) {
            points<<',';
            if (sample->uv) points<<(*sample->uv)[0];
            points<<',';
            if (sample->uv) points<<(*sample->uv)[1];
        }
        points<<','<<candidate.facing_interiors<<'\n';
    }
    auto stats=open_output(std::string(argv[12])+".stats.csv");
    stats<<"faces_a,faces_b,tested_face_pairs,unresolved_face_pairs,witnesses,setup_s,query_s\n"
         <<a.face_count()<<','<<b.face_count()<<','<<report.tested_face_pairs<<','<<report.unresolved_face_pairs<<','
         <<report.candidates.size()<<','<<std::chrono::duration<double>(start-setup_start).count()<<','
         <<std::chrono::duration<double>(end-start).count()<<'\n';
    if (!points || !stats) throw std::runtime_error("failed to write CAD contact region query");
}

void replay_contact_regions(int argc,char** argv) {
    const bool write_samples=argc==10 && std::string_view(argv[9])=="--samples";
    if (argc!=9 && !write_samples) throw std::invalid_argument(
        "nc_contact regions-replay A.step B.step CUTOFF_M STIFFNESS_N_PER_M2 POSES.txt OUTPUT_PREFIX cold|reuse [--samples]");
    const std::string_view mode(argv[8]);
    if (mode!="cold" && mode!="reuse") throw std::invalid_argument("invalid contact replay mode");
    const double cutoff=std::stod(argv[4]), stiffness=std::stod(argv[5]);
    if (!std::isfinite(cutoff) || cutoff<=0 || !std::isfinite(stiffness) || stiffness<0)
        throw std::invalid_argument("invalid contact replay cutoff or stiffness");
    std::ifstream poses(argv[6]);
    if (!poses) throw std::invalid_argument("could not open CAD contact poses");
    const auto start=Clock::now();
    const CadContactIndex a{read_step_shape(argv[2])}, b{read_step_shape(argv[3])};
    const double setup_s=std::chrono::duration<double>(Clock::now()-start).count();
    CadContactCache cache;
    const std::string output(argv[7]);
    auto witnesses=open_output(output+".csv");
    auto loads=open_output(output+".loads.csv");
    auto stats=open_output(output+".stats.csv");
    std::ofstream samples;
    if (write_samples) {
        samples=open_output(output+".samples.csv");
        samples<<"time_s,face_pair,weight_m,x_mm,y_mm,z_mm,cad_distance_mm\n";
    }
    witnesses<<"time_s,face_a,face_b,distance_m,facing_interiors\n";
    loads<<"time_s,face_a,face_b,length_m,force_bx_N,force_by_N,force_bz_N,"
           "moment_bx_Nm,moment_by_Nm,moment_bz_Nm\n";
    stats<<"time_s,tested_face_pairs,reused_face_exclusions,attempted_face_exclusions,unresolved_face_pairs,"
           "unresolved_line_pairs,setup_s,query_s,assembly_s\n";
    while (!(poses>>std::ws).eof()) {
        double time;
        ContactPose pose;
        if (!(poses>>time>>pose.origin[0]>>pose.origin[1]>>pose.origin[2]>>
                        pose.axis[0]>>pose.axis[1]>>pose.axis[2]>>pose.angle) || !std::isfinite(time)) {
            throw std::invalid_argument("invalid CAD contact replay pose");
        }
        pose.axis=unit(pose.axis);
        const auto query_start=Clock::now();
        const auto regions=a.query(b,pose,cutoff,mode=="reuse" ? &cache : nullptr);
        const auto assembly_start=Clock::now();
        const auto lines=a.straight_lines(b,pose,cutoff,1e-7,regions);
        for (const auto& line:lines.lines) {
            double length=0;
            for (const auto& sample:line.samples) {
                length+=sample.weight_m;
                if (write_samples) {
                    samples<<time<<','<<line.face_a<<'/'<<line.face_b<<','<<sample.weight_m;
                    for (double v:sample.points.b) samples<<','<<v*1000;
                    samples<<','<<sample.points.distance*1000<<'\n';
                }
            }
            const auto wrenches=line_contact_wrenches(line.samples,cutoff,stiffness,{},pose.origin);
            loads<<time<<','<<line.face_a<<','<<line.face_b<<','<<length;
            for (const auto& vector:{wrenches.b.force,wrenches.b.moment}) {
                for (double component:vector) loads<<','<<component;
            }
            loads<<'\n';
        }
        const auto end=Clock::now();
        for (const auto& witness:regions.candidates) {
            witnesses<<time<<','<<witness.a.face<<','<<witness.b.face<<','<<witness.points.distance<<','
                     <<witness.facing_interiors<<'\n';
        }
        stats<<time<<','<<regions.tested_face_pairs<<','<<regions.reused_face_exclusions<<','
             <<regions.attempted_face_exclusions<<','
             <<regions.unresolved_face_pairs<<','<<lines.unresolved_face_pairs-regions.unresolved_face_pairs<<','
             <<setup_s<<','<<std::chrono::duration<double>(assembly_start-query_start).count()<<','
             <<std::chrono::duration<double>(end-assembly_start).count()<<'\n';
    }
    if (!witnesses || !loads || !stats || (write_samples && !samples))
        throw std::runtime_error("failed to write CAD contact replay");
}

void cad_point_distances(int argc,char** argv) {
    if (argc!=5 && argc!=6) throw std::invalid_argument(
        "nc_contact cad-point-distance A.step POINTS_MM.txt OUTPUT.csv [TIMINGS.csv]");
    const bool measure=argc==6;
    const auto setup_start=Clock::now();
    const auto shape=read_step_shape(argv[2]);
    TopTools_IndexedMapOfShape solids, faces;
    TopExp::MapShapes(shape,TopAbs_SOLID,solids);
    if (solids.Extent()!=1) throw std::invalid_argument("CAD distances require one solid");
    const auto& solid=solids(1);
    TopExp::MapShapes(solid,TopAbs_FACE,faces);
    std::vector<Bnd_Box> boxes(faces.Extent());
    for (int i=1;i<=faces.Extent();++i) {
        BRepBndLib::AddOptimal(faces(i),boxes[i-1],false,true);
        if (boxes[i-1].IsVoid() || boxes[i-1].IsOpen()) throw std::invalid_argument("invalid CAD face bounds");
    }
    BRepClass3d_SolidClassifier classifier(solid);
    const double setup_s=std::chrono::duration<double>(Clock::now()-setup_start).count();
    std::ifstream points(argv[3]);
    if (!points) throw std::invalid_argument("could not open CAD distance points");
    auto output=open_output(argv[4]);
    output<<"signed_distance_mm,nx,ny,nz\n";
    std::vector<std::pair<double,int>> order(boxes.size());
    std::ofstream timings;
    if (measure) {
        timings=open_output(argv[5]);
        timings<<"setup_s,query_s\n";
    }
    constexpr double tolerance=1e-7;
    while (!(points>>std::ws).eof()) {
        double x,y,z;
        if (!(points>>x>>y>>z) || !(std::isfinite(x) && std::isfinite(y) && std::isfinite(z)))
            throw std::invalid_argument("invalid CAD distance point");
        const auto query_start=measure?Clock::now():Clock::time_point{};
        const gp_Pnt point{x,y,z};
        const auto vertex=BRepBuilderAPI_MakeVertex(point).Vertex();
        Bnd_Box point_box;
        point_box.Add(point);
        for (int i=1;i<=faces.Extent();++i) order[i-1]={boxes[i-1].Distance(point_box),i};
        std::sort(order.begin(),order.end());
        double distance=std::numeric_limits<double>::infinity();
        gp_Pnt nearest;
        bool unique=false;
        for (const auto& [lower,i]:order) {
            if (lower>distance+tolerance) break;
            const BRepExtrema_DistShapeShape closest(vertex,faces(i),tolerance);
            if (!closest.IsDone() || !closest.NbSolution())
                throw std::runtime_error("unresolved CAD point-to-face distance");
            for (int j=1;j<=closest.NbSolution();++j) {
                const auto& candidate=closest.PointOnShape2(j);
                const double gap=point.Distance(candidate);
                if (gap < distance-tolerance) {
                    nearest=candidate;
                    unique=true;
                } else if (gap <= distance+tolerance && nearest.Distance(candidate)>tolerance) {
                    unique=false;
                }
                distance=std::min(distance,gap);
            }
        }
        classifier.Perform(point,tolerance);
        const auto state=classifier.State();
        if (measure) timings<<setup_s<<','<<std::chrono::duration<double>(Clock::now()-query_start).count()<<'\n';
        if (state==TopAbs_ON && distance<=1e-6) {
            output<<"0,,,\n";
            continue;
        }
        if (!std::isfinite(distance) || (state!=TopAbs_IN && state!=TopAbs_OUT))
            throw std::runtime_error("unresolved CAD point classification");
        const double sign=state==TopAbs_IN ? -1 : 1;
        output<<sign*distance;
        if (unique && distance>tolerance) {
            const auto n=gp_Vec{nearest,point}.Normalized()*sign;
            output<<','<<n.X()<<','<<n.Y()<<','<<n.Z()<<'\n';
        } else {
            output<<",,,\n";
        }
    }
    if (!output) throw std::runtime_error("failed to write CAD distances");
    if (measure && !timings) throw std::runtime_error("failed to write CAD timings");
}

void cad_surface_samples(int argc,char** argv) {
    if (argc!=5) throw std::invalid_argument(
        "nc_contact cad-surface-samples A.step FACE_UV.txt OUTPUT.csv");
    const auto shape=read_step_shape(argv[2]);
    TopTools_IndexedMapOfShape faces;
    TopExp::MapShapes(shape,TopAbs_FACE,faces);
    std::vector<BRepAdaptor_Surface> surfaces;
    surfaces.reserve(faces.Extent());
    for (int i=1;i<=faces.Extent();++i) surfaces.emplace_back(TopoDS::Face(faces(i)));
    std::ifstream parameters(argv[3]);
    if (!parameters) throw std::invalid_argument("could not open CAD sampling parameters");
    auto output=open_output(argv[4]);
    output<<"inside_trim,x_mm,y_mm,z_mm,nx,ny,nz\n";
    while (!(parameters>>std::ws).eof()) {
        int id;
        double u,v;
        if (!(parameters>>id>>u>>v) || id<1 || id>faces.Extent() || !std::isfinite(u) || !std::isfinite(v))
            throw std::invalid_argument("invalid CAD surface sampling parameter");
        const auto& face=TopoDS::Face(faces(id));
        const auto state=BRepClass_FaceClassifier(face,gp_Pnt2d{u,v},1e-9).State();
        if (state!=TopAbs_IN && state!=TopAbs_ON) {
            output<<"0,,,,,,\n";
            continue;
        }
        gp_Pnt point;
        gp_Vec du,dv;
        surfaces[id-1].D1(u,v,point,du,dv);
        auto normal=du.Crossed(dv).Normalized();
        if (face.Orientation()==TopAbs_REVERSED) normal.Reverse();
        output<<1<<','<<point.X()<<','<<point.Y()<<','<<point.Z()<<','
              <<normal.X()<<','<<normal.Y()<<','<<normal.Z()<<'\n';
    }
    if (!output) throw std::runtime_error("failed to write CAD surface samples");
}

double sampled_load(const std::vector<LoadSample>& table,double time) {
    if (table.empty()) return 0;
    const auto right=std::upper_bound(table.begin()+1,table.end()-1,time,
                                      [](double t,const LoadSample& sample) {return t<sample.time;});
    const auto& left=*(right-1);
    const double fraction=std::clamp((time-left.time)/(right->time-left.time),0.,1.);
    return std::lerp(left.force,right->force,fraction);
}

void simulate(const Path& scene,double deflection,double dt,double duration,unsigned stride,const Path& output,
              std::optional<PairBackend> pair_backend,PairSearchOptions pair_options,double event_tolerance,
              const SurfaceOptions& surface={}) {
    if (!(std::isfinite(dt) && dt>0 && std::isfinite(duration) && duration>=dt && stride>0))
        throw std::invalid_argument("invalid step, duration or stride");
    const double step_count=duration/dt;
    const double roundoff=8*std::numeric_limits<double>::epsilon()*step_count;
    const auto steps=static_cast<std::size_t>(std::ceil(step_count-roundoff));
    const auto setup=Clock::now();
    const Input input=read_scene(scene,deflection,pair_backend,pair_options,event_tolerance>0,surface);
    for (const auto& body:input.bodies) {
        const auto* table=std::get_if<std::vector<TranslationSample>>(&body.drive);
        if (table && steps*dt>table->back().time+1e-12)
            throw std::invalid_argument("simulation extends beyond prescribed translation table");
    }
    for (const auto& table:input.loads) {
        if (!table.empty() && steps*dt>table.back().time+1e-12)
            throw std::invalid_argument("simulation extends beyond prescribed coordinate load table");
    }
    const double setup_s=std::chrono::duration<double>(Clock::now()-setup).count();
    const auto n=input.coordinates.size();
    const bool has_history=std::any_of(input.pairs.begin(),input.pairs.end(),
                                      [](const auto& pair) {return pair.material.approach_floor>0;});
    Vector mass,q,v;
    for (const auto& coordinate:input.coordinates) {
        mass.push_back(coordinate.mass);
        q.push_back(coordinate.position);
        v.push_back(coordinate.velocity);
    }
    ContactWork work;
    std::vector<Vec> base_acceleration;
    const bool reuse_mesh_samples=!surface.method.empty() && !has_history;
    bool record_sample=reuse_mesh_samples;
    double last_sample_time{};
    Vector last_sample_q,last_sample_v;
    std::vector<ContactSample> last_contacts;
    const auto evaluate_force=[&](const Vector& q_,const Vector& v_,double time,const Vector& previous) {
        const auto state=kinematics(input,q_,time,&base_acceleration);
        Forces out{Vector(n),Vector(n*n),Vector(n*n)};
        for (std::size_t i=0;i<n;++i) {
            const auto& coordinate=input.coordinates[i];
            out.value[i]=coordinate.load-coordinate.stiffness*(q_[i]-coordinate.rest)-coordinate.damping*v_[i];
            if (i<input.loads.size()) out.value[i]+=sampled_load(input.loads[i],time);
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
        if (!input.bushings.empty()) add_translational_bushings(out,state,input.bushings,v_);
        std::vector<ContactSample> contacts;
        if (!add_contact_forces(&out,state,input.pairs,v_,work,has_history || record_sample ? &contacts : nullptr,
                                impact_history(previous))) {
            out.value[0]=std::numeric_limits<double>::quiet_NaN();
        } else if (has_history) {
            out.history.reserve(3*contacts.size());
            for (const auto& contact:contacts) {
                out.history.push_back(contact.gap);
                out.history.push_back(contact.gap_rate);
                out.history.push_back(contact.impact_speed);
            }
        } else if (record_sample) {
            last_sample_time=time;
            last_sample_q=q_;
            last_sample_v=v_;
            last_contacts=std::move(contacts);
        }
        for (const auto& body:input.bodies)
            if (body.free) body.free->apply_euler_equations(q_,v_,out);
        return out;
    };
    HhtOptions options;
    options.nonsymmetric_tangent=std::any_of(input.bodies.begin(),input.bodies.end(),
                                               [](const Body& b) { return b.free.has_value(); });
    options.absolute_tolerance=1e-9;
    options.relative_tolerance=1e-8;
    HhtStats stats;
    std::size_t event_nodes=0, unresolved_events=0;
    StepAcceptanceFunction accept_step;
    if (event_tolerance>0) {
        accept_step=[&](const State& start,const State& end) -> std::optional<double> {
            const double dt_trial=end.time-start.time;
            const auto first=kinematics(input,start.q,start.time);
            const auto last=kinematics(input,end.q,end.time);
            for (std::size_t i=0;i<input.pairs.size();++i) {
                const auto& pair=input.pairs[i];
                if (pair.regions) {
                    const auto& a0=first[pair.a].pose; const auto& a1=last[pair.a].pose;
                    const auto& b0=first[pair.b].pose; const auto& b1=last[pair.b].pose;
                    const auto large_turn=[&](std::size_t id) {
                        const auto& body=input.bodies[id];
                        if (!body.free) return std::abs(last[id].pose.angle-first[id].pose.angle)>=std::acos(-1.);
                        const auto i=body.free->first+3;
                        double alignment=0.;
                        for (std::size_t k=i;k<i+4;++k) alignment+=start.q[k]*end.q[k];
                        return alignment<=0 || dt_trial*std::max(length(body.free->body_angular_velocity(start.q,start.v)),
                            length(body.free->body_angular_velocity(end.q,end.v)))>=std::acos(-1.);
                    };
                    if (large_turn(pair.a) || large_turn(pair.b)) return .5;
                    const auto hit=pair.regions->transition(a0,b0,a1,b1,surface.distance_tolerance);
                    event_nodes+=hit.nodes;
                    if (!hit.fraction) continue;
                    if (dt_trial>event_tolerance)
                        return *hit.fraction+(hit.start_in_contact ? -.5 : .5)*event_tolerance/dt_trial;
                    if (!hit.start_in_contact && !hit.end_in_contact) {
                        ++unresolved_events;
                        return .5;
                    }
                    continue;
                }
                const auto& [a,b]=input.event_geometry[i];
                const auto result=fixed_axis_pair_sweep(*a,*b,first[pair.a].pose,last[pair.a].pose,
                    first[pair.b].pose,last[pair.b].pose,pair.activation,std::min(.25,event_tolerance/dt_trial),pair_options);
                event_nodes+=result.statistics.visited;
                if (result.status==SweepStatus::clear || result.witness_time==0) continue;
                if (result.status==SweepStatus::unresolved) {
                    ++unresolved_events;
                    return .5;
                }
                if (dt_trial>event_tolerance) {
                    // Put the first substep just past a witnessed event so an
                    // exact threshold at the midpoint does not straddle two steps.
                    return *result.witness_time + .5*event_tolerance/dt_trial;
                }
                const auto endpoint=pair.closest->closest(last[pair.a].pose,last[pair.b].pose,pair.activation,work);
                if (endpoint.status!=PairStatus::within_cutoff) return .5;
            }
            return std::nullopt;
        };
    }
    std::vector<State> samples;
    samples.reserve(steps/stride+2);
    std::vector<std::vector<ContactSample>> observed_contacts;
    const auto save_state=[&](const State& state) {
        samples.push_back(state);
        if (reuse_mesh_samples) {
            if (state.time==last_sample_time && state.q==last_sample_q && state.v==last_sample_v)
                observed_contacts.push_back(std::move(last_contacts));
            else observed_contacts.emplace_back();
        }
    };
    const auto start=Clock::now();
    if (n>0) {
        ForceFunction force;
        HistoryForceFunction historical_force;
        if (has_history) historical_force=evaluate_force;
        else force=[&](const Vector& q_,const Vector& v_,double time) {
            return evaluate_force(q_,v_,time,{});
        };
        Hht solver(mass,std::move(force),options,{},std::move(historical_force),{},std::move(accept_step));
        try {
            auto state=solver.initialize(q,v);
            save_state(state);
            for (std::size_t s=1;s<=steps;++s) {
                const bool save=s%stride==0 || s==steps;
                record_sample=reuse_mesh_samples && save;
                solver.step(state,dt,stats);
                for (const auto& body:input.bodies) if (body.free) body.free->normalize(state);
                if (save) save_state(state);
            }
        } catch (const std::exception& error) {
            if (!work.unresolved_queries && !unresolved_events) throw;
            std::ostringstream detail;
            detail<<error.what()<<" ("<<work.unresolved_queries<<" unresolved pair force evaluations, "
                  <<unresolved_events<<" unresolved step sweeps)";
            if (const auto bound=work.last_unresolved_distance) {
                detail<<", last distance interval ["<<std::setprecision(9)<<bound->lo<<", "<<bound->hi<<"] m";
            }
            throw std::runtime_error(detail.str());
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
    for (std::size_t j=0;j<input.bodies.size();++j) {
        if (input.bodies[j].free) {
            for (auto label:{"_qw","_qx","_qy","_qz","_wx_rad_s","_wy_rad_s","_wz_rad_s"})
                trace<<",b"<<j<<label;
        }
    }
    for (std::size_t p=0;p<input.pairs.size();++p)
        trace<<",p"<<p<<"_gap_m,p"<<p<<"_normal_N,p"<<p<<"_friction_N,p"<<p<<"_hit";
    for (std::size_t s=0;s<input.accelerometers.size();++s) trace<<",s"<<s<<"_a_m_s2";
    if (has_history) {
        for (std::size_t p=0;p<input.pairs.size();++p) {
            if (input.pairs[p].material.approach_floor>0)
                trace<<",p"<<p<<"_gap_rate_m_s,p"<<p<<"_impact_speed_m_s";
        }
    }
    trace<<'\n';
    ContactWork audit;
    std::vector<ContactSample> values;
    double replay_s=0;
    std::size_t reused=0;
    for (std::size_t index=0;index<samples.size();++index) {
        const auto& item=samples[index];
        trace<<item.time;
        for (std::size_t i=0;i<n;++i) trace<<','<<item.q[i]<<','<<item.v[i]<<','<<item.a[i];
        const auto poses=kinematics(input,item.q,item.time,
                                    input.accelerometers.empty() ? nullptr : &base_acceleration);
        for (const auto& body:poses)
            trace<<','<<body.pose.origin[0]<<','<<body.pose.origin[1]<<','<<body.pose.origin[2]<<','<<body.pose.angle;
        for (std::size_t j=0;j<poses.size();++j) {
            if (const auto& free=input.bodies[j].free) {
                for (std::size_t k=free->first+3;k<free->first+7;++k) trace<<','<<item.q[k];
                const Vec omega=poses[j].pose.rotate(free->body_angular_velocity(item.q,item.v));
                for (double component:omega) trace<<','<<component;
            }
        }
        if (index<observed_contacts.size() && !observed_contacts[index].empty()) {
            values=std::move(observed_contacts[index]);
            ++reused;
        } else if (n) {
            if (!add_contact_forces(nullptr,poses,input.pairs,item.v,audit,&values,impact_history(item.history)))
                throw std::runtime_error("invalid saved contact state");
        } else {
            const auto replay_start=Clock::now();
            if (!add_contact_forces(nullptr,poses,input.pairs,item.v,work,&values))
                throw std::runtime_error("invalid prescribed contact state");
            replay_s+=std::chrono::duration<double>(Clock::now()-replay_start).count();
        }
        for (const auto& value:values)
            trace<<','<<value.gap<<','<<value.normal_force<<','<<length(value.friction)<<','<<value.resolved;
        for (const auto& sensor:input.accelerometers) {
            const auto body=sensor.body;
            trace<<','<<specific_force(sensor,input.bodies[body],poses[body],item,
                                       base_acceleration[body],input.gravity);
        }
        if (has_history) {
            for (std::size_t p=0;p<input.pairs.size();++p) {
                if (input.pairs[p].material.approach_floor>0)
                    trace<<','<<values[p].gap_rate<<','<<values[p].impact_speed;
            }
        }
        trace<<'\n';
    }
    auto report=open_output(output.string()+".stats.csv");
    report<<"wall_s,setup_s,steps,accepted,subdivisions,force_evaluations,max_residual,pairs,culled,rays,missing,triangle_tests,"
            "closest_queries,unresolved_queries,native_nodes,ccd_checks,ccd_rejected,ccd_nodes,ccd_unresolved,saved_contact_reuses\n"
        <<(n ? wall : replay_s)<<','<<setup_s<<','<<steps<<','<<stats.accepted_steps<<','<<stats.subdivisions<<','<<stats.evaluations
        <<','<<stats.maximum_residual<<','<<work.pairs<<','<<work.culled<<','<<work.rays<<','<<work.missing
        <<','<<work.mesh.triangles<<','<<work.closest_queries<<','<<work.unresolved_queries<<','<<work.native_nodes
        <<','<<stats.step_checks<<','<<stats.rejected_steps<<','<<event_nodes<<','<<unresolved_events<<','<<reused<<'\n';
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
        if (argc>=2 && std::string_view(argv[1])=="cad-surface-samples") {
            cad_surface_samples(argc,argv);
            return 0;
        }
        if (argc>=2 && std::string_view(argv[1])=="cad-point-distance") {
            cad_point_distances(argc,argv);
            return 0;
        }
        if (argc>=2 && std::string_view(argv[1])=="regions-replay") {
            replay_contact_regions(argc,argv);
            return 0;
        }
        if (argc>=2 && (std::string_view(argv[1])=="regions" || std::string_view(argv[1])=="line-load")) {
            contact_regions(argc,argv,std::string_view(argv[1])=="line-load");
            return 0;
        }
        if (argc>=3 && std::string_view(argv[1])=="make") {
            export_fixture(argc,argv);
            return 0;
        }
        const bool named=argc>7 && std::string_view(argv[7]).starts_with("--");
        if (argc<7 || (!named && argc>11)) {
            std::cerr<<"nc_contact SCENE DEFLECTION_M DT_S DURATION_S OUTPUT_STRIDE OUTPUT_PREFIX "
                        "[pair-occt|pair-mesh|pair-brep [DISTANCE_TOL_M [MAX_NODES [EVENT_TOL_S]]]]\n"
                        "  or append --method mesh-aabb|mesh-obb|mesh-sap --mesh-dir DIR [--solver penalty] "
                        "[--event-tol-s SECONDS --ccd-tol-m METRES]\n";
            return 2;
        }
        std::optional<PairBackend> pair_backend;
        PairSearchOptions settings;
        SurfaceOptions surface;
        double event_tolerance=0.;
        if (named) {
            for (int i=7;i<argc;i+=2) {
                if (i+1==argc) throw std::invalid_argument("missing method, solver or mesh directory value");
                const std::string_view key=argv[i], value=argv[i+1];
                if (key=="--method") surface.method=value;
                else if (key=="--mesh-dir") surface.mesh_directory=argv[i+1];
                else if (key=="--event-tol-s") event_tolerance=std::stod(argv[i+1]);
                else if (key=="--ccd-tol-m") surface.distance_tolerance=std::stod(argv[i+1]);
                else if (key=="--solver") {
                    if (value!="penalty") throw std::invalid_argument("unsupported surface contact solver");
                }
                else throw std::invalid_argument("unsupported surface option or solver");
            }
            if (surface.method.empty() || surface.mesh_directory.empty())
                throw std::invalid_argument("surface dynamics requires --method mesh-aabb|mesh-obb|mesh-sap and --mesh-dir DIR");
            if (!(std::isfinite(surface.distance_tolerance) && surface.distance_tolerance>0))
                throw std::invalid_argument("invalid mesh CCD tolerance");
        } else if (argc>=8) {
            const std::string_view option(argv[7]);
            if (option=="pair-occt") pair_backend=PairBackend::occt;
            else if (option=="pair-mesh") pair_backend=PairBackend::mesh;
            else if (option=="pair-brep") pair_backend=PairBackend::brep;
            else throw std::invalid_argument("unknown closest-pair backend");
        }
        if (!named && argc>=9) settings.tolerance_m=std::stod(argv[8]);
        if (!named && argc>=10) {
            const auto nodes=std::stoll(argv[9]);
            if (nodes<1 || nodes>100'000'000) throw std::invalid_argument("invalid closest-pair node budget");
            settings.max_nodes=static_cast<std::size_t>(nodes);
        }
        if (!named && argc>=11) event_tolerance=std::stod(argv[10]);
        if (!std::isfinite(event_tolerance) || event_tolerance<0) throw std::invalid_argument("invalid event tolerance");
        simulate(argv[1],std::stod(argv[2]),std::stod(argv[3]),std::stod(argv[4]),
                 static_cast<unsigned>(std::stoul(argv[5])),argv[6],pair_backend,settings,event_tolerance,surface);
    } catch (const Standard_Failure& error) {
        std::cerr<<error.GetMessageString()<<'\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr<<error.what()<<'\n';
        return 1;
    }
}
