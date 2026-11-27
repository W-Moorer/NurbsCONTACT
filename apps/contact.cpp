#include "nurbscontact/predictor_regions.hpp"
#include "nurbscontact/compiled_part.hpp"
#ifdef NC_WITH_CUDA_COLUMNS
#include "../src/compiled_geometry.hpp"
#include "../src/cuda_columns.hpp"
#endif
#include "nurbscontact/pair_options.hpp"
#ifndef NC_COMPILED_ONLY
#include "nurbscontact/cad_contact_regions.hpp"
#include "nurbscontact/cad_solid_contact.hpp"
#endif
#include "nurbscontact/free_rigid.hpp"
#include "nurbscontact/surface_impulse.hpp"
#ifndef NC_COMPILED_ONLY
#include "nurbscontact/occt_contact.hpp"
#endif
#ifdef NC_WITH_CPU_MESH
#include "nurbscontact/cpu_mesh.hpp"
#endif

#ifndef NC_COMPILED_ONLY
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

#endif
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
    std::vector<std::shared_ptr<const compiled::Solid>> compiled_parts;
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
    bool cuda_columns{};
    bool resident{};
    std::string region_reduction;
    std::string gpu_execution;
    bool gpu_profile{};
    std::optional<ImpulseLaw> impulse_law;
    Path mesh_directory,compiled_directory;
    Path progress_file;
    Path replay_file;
    double distance_tolerance{1e-7};
    std::string mesh_regions{"local"};
    bool share_penalty_stiffness{};
    bool predictor_geometry{};
    std::optional<double> spectral_radius;
    std::optional<double> force_tolerance, moment_tolerance;
    std::optional<bool> fixed_cubature;
    bool chen() const { return method=="chen-2024" || method=="chen-2026"; }
    bool compiled() const { return method=="cad-compiled" || method=="cad-boundary"; }
    bool patch() const { return compiled() || method=="cad-patch" || method=="cad-bezier" || method=="cad-bezier-bounds" || method=="cad-bezier-relocate"; }
    bool cad() const { return method=="cad-occt" || method=="cad-local" || patch() || chen(); }
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
#ifndef NC_COMPILED_ONLY
    std::map<Path,TopoDS_Shape> shapes;
#endif
    std::map<Path,std::shared_ptr<const compiled::Solid>> parts;
    std::map<Path,Path> part_sources;
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
        if (surface_options.compiled()) {
            auto asset=surface_options.compiled_directory.empty() ? file : surface_options.compiled_directory/file.filename();
            asset.replace_extension(".ncpart");
            const auto source=file.lexically_normal();
            const auto [owner,inserted]=part_sources.emplace(asset.lexically_normal(),source);
            if (!inserted && owner->second!=source)
                throw std::invalid_argument("distinct model files resolve to the same compiled part: "+asset.string());
            const auto sites=surface_options.method=="cad-boundary" ? compiled::DiscoverySites::trim_boundary : compiled::DiscoverySites::face_grid;
            if (!parts.contains(file)) parts.emplace(file,compiled::load_solid(asset,sites));
            if (surface_options.resident) input.compiled_parts.push_back(parts.at(file));
        }
#ifndef NC_COMPILED_ONLY
        else if (surface_options.method.empty() || surface_options.cad()) {
            if (!shapes.contains(file)) shapes.emplace(file,read_step_shape(file));
        } else {
#ifdef NC_WITH_CPU_MESH
            Path asset=file;
            if (!surface_options.mesh_directory.empty()) {
                Path relative=filename;
                if (relative.is_absolute()) relative=std::filesystem::relative(file,path.parent_path());
                if (relative.empty()) throw std::invalid_argument("cannot resolve the STEP asset under --mesh-dir");
                asset=surface_options.mesh_directory/relative;
            }
            asset.replace_extension(".msh");
            auto& mesh=mesh_cache[asset];
            if (!mesh) mesh=std::make_shared<cpu::MeshData>(cpu::read_gmsh(asset));
            meshes.push_back(mesh);
#else
            throw std::invalid_argument("rebuild nc_contact with NC_WITH_CPU_MESH for mesh contact");
#endif
        }
#endif
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
#ifndef NC_COMPILED_ONLY
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
#endif
    for (unsigned i=0;i<pairs;++i) {
        const auto type=take<std::string>(in);
        const auto a=take<unsigned>(in),b=take<unsigned>(in);
        if (a>=bodies || b>=bodies || a==b) throw std::invalid_argument("invalid probe pair");
        ContactPair pair{a,b,{},{},ContactDirection::world,{},{}};
        if (type=="surface") {
            if (surface_options.method.empty()) throw std::invalid_argument("surface pair requires a surface method");
            pair.share_penalty_stiffness=surface_options.share_penalty_stiffness;
#ifndef NC_COMPILED_ONLY
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
#endif
        } else throw std::invalid_argument("expected surface, probe or closest contact");
        pair.material={number(in),number(in),number(in),number(in),number(in),number(in)};
        const auto& m=pair.material;
        if (!(m.stiffness>0 && m.exponent>=1 && m.rate>=0 && m.friction>=0 && m.slip_speed>0 &&
                m.maximum_depth>0)) throw std::invalid_argument("invalid contact material");
#ifndef NC_COMPILED_ONLY
        if (detect_events && surface_options.method.empty()) {
            if (!pair.closest || count==0) throw std::invalid_argument("step CCD requires dynamic closest-pair contacts");
            input.event_geometry.emplace_back(load_pair_geometry(a,PairBackend::brep),load_pair_geometry(b,PairBackend::brep));
        }
#endif
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
                throw std::invalid_argument("surface event checks require affine pivot motion between step endpoints");
        }
    }
    if (!surface_options.method.empty()) {
        const auto center=[&](std::size_t id) -> Vec {
            const auto& body=input.bodies[id];
            if (!detect_events) return {};
            if (body.free) return body.center_of_mass;
            if (body.hinge>=0) return body.reference.inverse_rotate(sub(body.pivot,body.reference.origin));
            return {};
        };
        for (auto& pair:input.pairs) {
            const std::array centers{center(pair.a),center(pair.b)};
            if (surface_options.compiled()) {
                std::optional<compiled::LoadQuadrature> rule;
                if (surface_options.force_tolerance || surface_options.fixed_cubature.value_or(false)) {
                    const auto &m=pair.material;
                    if (m.rate!=0 || m.friction!=0 || m.impact_dissipation!=0)
                        throw std::invalid_argument("load quadrature requires frictionless elastic penalty contact");
                    rule=compiled::LoadQuadrature{m.stiffness,m.exponent,surface_options.force_tolerance.value_or(0),
                                                  surface_options.moment_tolerance.value_or(0),surface_options.fixed_cubature.value_or(false) ? 96 : 0,
                                                  surface_options.share_penalty_stiffness};
                }
                pair.regions=compiled::region_query(parts.at(files[pair.a]),parts.at(files[pair.b]),pair.material.maximum_depth,rule,surface_options.cuda_columns);
            }
#ifndef NC_COMPILED_ONLY
            else if (surface_options.patch()) {
                pair.regions=cad_patch_region_query(shapes.at(files[pair.a]),shapes.at(files[pair.b]),centers,
                                                    pair.material.maximum_depth,surface_options.method!="cad-patch",
                                                    surface_options.method=="cad-bezier-bounds" || surface_options.method=="cad-bezier-relocate",
                                                    surface_options.method=="cad-bezier-relocate");
            } else if (surface_options.method=="cad-local") {
                pair.regions=cad_local_region_query(shapes.at(files[pair.a]),shapes.at(files[pair.b]),centers,
                                                    pair.material.maximum_depth);
            } else if (surface_options.cad()) {
                pair.regions=cad_solid_region_query(shapes.at(files[pair.a]),shapes.at(files[pair.b]),centers,
                    surface_options.method=="chen-2026" ? CadEventMethod::chen_2026 :
                    surface_options.method=="chen-2024" ? CadEventMethod::chen_2024 : CadEventMethod::conservative_advancement);
            } else {
#ifdef NC_WITH_CPU_MESH
                pair.regions=cpu::mesh_region_query(meshes[pair.a],meshes[pair.b],cpu::mesh_method(surface_options.method),
                                                    centers,cpu::mesh_region_mode(surface_options.mesh_regions),pair.material.maximum_depth);
#endif
            }
#endif
        }
    }
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

#ifndef NC_COMPILED_ONLY
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

#endif
double sampled_load(const std::vector<LoadSample>& table,double time) {
    if (table.empty()) return 0;
    const auto right=std::upper_bound(table.begin()+1,table.end()-1,time,
                                      [](double t,const LoadSample& sample) {return t<sample.time;});
    const auto& left=*(right-1);
    const double fraction=std::clamp((time-left.time)/(right->time-left.time),0.,1.);
    return std::lerp(left.force,right->force,fraction);
}

struct ImpulseSteps {
    ImpulseLaw law;
    const Input& input;
    const Vector& mass;
    const Hht* smooth;
    const StepAcceptanceFunction& guard;
    ContactWork& work;
    HhtStats& stats;
    HhtStats smooth_stats{};
    std::size_t solves{},regions{},iterations{};
    double residual{};
    bool checking_correction{};

    std::optional<double> check(const State& start,const State& end) {
        if (!guard) return {};
        ++stats.step_checks;
        return guard(start,end);
    }

    std::optional<double> attempt(State& state,std::vector<ContactSample>& previous,double h,
                                 Vector& normal,std::vector<Vec>& friction) {
        State trial=state;
        if (smooth) {
            // The previous contact impulse has already changed the starting
            // velocity. Only external and gyroscopic loads enter this predictor.
            trial=smooth->initialize(state.q,state.v,state.time);
            smooth->step(trial,h,smooth_stats);
        } else trial.time+=h;
        checking_correction=false;
        if (auto split=check(state,trial)) return split;
        const auto inverse=[&](const Vector& row) {
            Vector dv(row.size());
            for (std::size_t j=0;j<dv.size();++j) dv[j]=row[j]/mass[j];
            for (const auto& body:input.bodies)
                if (body.free) body.free->angular_impulse_response(trial.q,row,dv);
            return dv;
        };
        auto answer=surface_impulses(law,kinematics(input,trial.q,trial.time),input.pairs,
                                     trial.v,h,inverse,work,previous);
        ++solves;
        regions+=answer.regions;
        iterations+=answer.iterations;
        residual=std::max(residual,answer.residual_m_s);
        for (std::size_t j=0;j<mass.size();++j) {
            const double dv=answer.velocity_change[j];
            trial.v[j]+=dv;
            trial.q[j]+=h*dv;
            trial.a[j]+=dv/h;
            trial.force[j]=mass[j]*trial.a[j];
        }
        for (const auto& body:input.bodies) if (body.free) body.free->normalize(trial);
        if (answer.regions) {
            checking_correction=true;
            if (auto split=check(state,trial)) return split;
        }
        state=std::move(trial);
        previous=std::move(answer.contacts);
        for (std::size_t i=0;i<previous.size();++i) {
            normal[i]+=h*previous[i].normal_force;
            friction[i]=add(friction[i],scale(previous[i].friction,h));
        }
        ++stats.accepted_steps;
        return {};
    }

    void advance(State& state,std::vector<ContactSample>& previous,double h,Vector& normal,
                 std::vector<Vec>& friction,unsigned remaining) {
        double fraction=.5;
        try {
            const auto split=attempt(state,previous,h,normal,friction);
            if (!split) return;
            fraction=std::clamp(*split,.125,.875);
            ++stats.rejected_steps;
        } catch (const UnresolvedContact&) {
            ++work.unresolved_queries;
            if (!remaining) throw;
        }
        if (!remaining) {
            std::ostringstream message;
            message<<std::setprecision(12)<<"impulse step subdivision limit during "
                   <<(checking_correction ? "corrected" : "free")<<" motion at "<<state.time<<" s, h="<<h;
            throw std::runtime_error(message.str());
        }
        ++stats.subdivisions;
        const double first=h*fraction;
        advance(state,previous,first,normal,friction,remaining-1);
        advance(state,previous,h-first,normal,friction,remaining-1);
    }
};

void simulate(const Path& scene,double deflection,double dt,double duration,unsigned stride,const Path& output,
              std::optional<PairBackend> pair_backend,PairSearchOptions pair_options,double event_tolerance,
              const SurfaceOptions& surface={}) {
    if (!(std::isfinite(dt) && dt>0 && std::isfinite(duration) && duration>=dt && stride>0))
        throw std::invalid_argument("invalid step, duration or stride");
    const double step_count=duration/dt;
    const double roundoff=8*std::numeric_limits<double>::epsilon()*step_count;
    const auto steps=static_cast<std::size_t>(std::ceil(step_count-roundoff));
    const auto setup=Clock::now();
    Input input=read_scene(scene,deflection,pair_backend,pair_options,event_tolerance>0,surface);
    for (const auto& body:input.bodies) {
        const auto* table=std::get_if<std::vector<TranslationSample>>(&body.drive);
        if (table && steps*dt>table->back().time+1e-12)
            throw std::invalid_argument("simulation extends beyond prescribed translation table");
    }
    for (const auto& table:input.loads) {
        if (!table.empty() && steps*dt>table.back().time+1e-12)
            throw std::invalid_argument("simulation extends beyond prescribed coordinate load table");
    }
    std::vector<std::shared_ptr<PredictorRegions>> predictors;
    if (surface.predictor_geometry && input.coordinates.empty())
        throw std::invalid_argument("predictor geometry requires integrated coordinates");
    if (!surface.replay_file.empty() && (!surface.predictor_geometry || surface.impulse_law))
        throw std::invalid_argument("HHT step replay requires predictor penalty contact");
    if (surface.predictor_geometry) for (auto& pair:input.pairs) {
        if (!pair.regions) throw std::invalid_argument("predictor updates require surface regions");
        auto proxy=std::make_shared<PredictorRegions>(pair.regions);
        predictors.push_back(proxy);
        pair.regions=std::move(proxy);
    }
    double setup_s=std::chrono::duration<double>(Clock::now()-setup).count();
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
        if (!surface.impulse_law && !add_contact_forces(&out,state,input.pairs,v_,work,has_history || record_sample ? &contacts : nullptr,
                                impact_history(previous))) {
            out.value[0]=std::numeric_limits<double>::quiet_NaN();
        } else if (!surface.impulse_law && has_history) {
            out.history.reserve(3*contacts.size());
            for (const auto& contact:contacts) {
                out.history.push_back(contact.gap);
                out.history.push_back(contact.gap_rate);
                out.history.push_back(contact.impact_speed);
            }
        } else if (!surface.impulse_law && record_sample) {
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
    options.spectral_radius=surface.spectral_radius;
    options.nonsymmetric_tangent=std::any_of(input.bodies.begin(),input.bodies.end(),
                                               [](const Body& b) { return b.free.has_value(); });
    options.absolute_tolerance=1e-9;
    options.relative_tolerance=1e-8;
    HhtStats stats;
    std::size_t event_nodes=0, unresolved_events=0, spatial_refinements=0, refined_clear=0;
    std::size_t cad_distances=0,chen_queries=0,chen_confirmed=0;
    std::size_t tdib_cells=0,tdib_candidates=0,tdib_unresolved=0;
    std::size_t cad_box_tests=0,cad_extrema=0,cad_classifications=0;
    std::size_t cad_distance_hits=0;
    std::size_t cad_witness_tests=0,cad_witness_improvements=0;
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
                    SurfaceTransition hit;
                    try {
                        const auto sweep=[&](double tolerance) {
                            hit=pair.regions->transition(a0,b0,a1,b1,tolerance,
                                                        std::min(1.,.25*event_tolerance/dt_trial));
                            cad_distances+=hit.cad_distances;
                            chen_queries+=hit.chen_queries;
                            chen_confirmed+=hit.chen_confirmed;
                            tdib_cells+=hit.tdib_cells;
                            tdib_candidates+=hit.tdib_candidates;
                            tdib_unresolved+=hit.tdib_unresolved;
                            cad_box_tests+=hit.cad_box_tests;
                            cad_extrema+=hit.cad_extrema;
                            cad_classifications+=hit.cad_classifications;
                            cad_distance_hits+=hit.cad_distance_cache_hits;
                            cad_witness_tests+=hit.cad_witness_tests;
                            cad_witness_improvements+=hit.cad_witness_improvements;
                            event_nodes+=hit.nodes;
                        };
                        sweep(surface.distance_tolerance);
                        // Hard contacts can leave two separated endpoints extremely close
                        // to touch. A tighter spatial sweep can resolve a near miss before
                        // bisecting time again. A surviving candidate still requires a split.
                        if (surface.impulse_law && dt_trial<=event_tolerance) {
                            double tolerance=surface.distance_tolerance;
                            for (int k=0;k<2 && hit.fraction && !hit.start_in_contact && !hit.end_in_contact;++k) {
                                tolerance/=32.;
                                if (tolerance<1e-11) break;
                                ++spatial_refinements;
                                sweep(tolerance);
                                if (!hit.fraction) ++refined_clear;
                            }
                        }
                    } catch (const UnresolvedContact&) {
                        ++unresolved_events;
                        return .5;
                    }
                    if (!hit.fraction) continue;
                    if (dt_trial>event_tolerance)
                        return *hit.fraction+(hit.start_in_contact ? -.5 : .5)*event_tolerance/dt_trial;
                    if (!hit.start_in_contact && !hit.end_in_contact) {
                        ++unresolved_events;
                        return .5;
                    }
                    continue;
                }
#ifndef NC_COMPILED_ONLY
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
#endif
            }
            return std::nullopt;
        };
    }
    std::vector<State> samples;
    samples.reserve(steps/stride+2);
    std::vector<std::vector<ContactSample>> observed_contacts;
    const auto save_state=[&](const State& state) {
        samples.push_back(state);
        if (reuse_mesh_samples || !predictors.empty()) {
            if (reuse_mesh_samples && state.time==last_sample_time && state.q==last_sample_q && state.v==last_sample_v)
                observed_contacts.push_back(std::move(last_contacts));
            else if (!predictors.empty()) {
                auto& sample=observed_contacts.emplace_back();
                const auto poses=kinematics(input,state.q,state.time);
                if (!add_contact_forces(nullptr,poses,input.pairs,state.v,work,&sample,impact_history(state.history)))
                    throw std::runtime_error("invalid accepted predictor contact");
            }
            else observed_contacts.emplace_back();
        }
    };
    const auto start=Clock::now();
    std::ofstream progress;
    if (!surface.progress_file.empty()) {
        progress=open_output(surface.progress_file);
        progress<<"step,time_s,integration_s,mesh_triangle_tests,mesh_booleans,cad_booleans,"
                    "mesh_detection_s,mesh_placement_s,mesh_boolean_s";
        for (std::size_t i=0;i<n;++i) progress<<",q"<<i<<",v"<<i;
        progress<<",mesh_pair_bvh_builds,mesh_pair_bvh_reuses,mesh_patch_builds,mesh_patch_reuses";
        if (surface.patch()) progress<<",cad_patch_queries,cad_patch_batches,cad_patch_broad_s,cad_patch_search_s,cad_patch_group_s,cad_patch_area_s";
        progress<<std::endl;
    }
    const auto progress_step=[&](std::size_t step,const State& accepted) {
        if (!progress.is_open()) return;
        progress<<step<<','<<accepted.time<<','<<std::chrono::duration<double>(Clock::now()-start).count()
                <<','<<work.mesh.triangles<<','<<work.boolean_queries<<','<<work.cad_boolean_queries
                <<','<<work.mesh_detection_s<<','<<work.mesh_placement_s<<','<<work.mesh_boolean_s;
        for (std::size_t i=0;i<n;++i) progress<<','<<accepted.q[i]<<','<<accepted.v[i];
        progress<<','<<work.mesh_pair_bvh_builds<<','<<work.mesh_pair_bvh_reuses
                <<','<<work.mesh_patch_builds<<','<<work.mesh_patch_reuses;
        if (surface.patch()) progress<<','<<work.cad_patch_queries<<','<<work.cad_patch_batches
            <<','<<work.cad_patch_broad_s<<','<<work.cad_patch_search_s<<','<<work.cad_patch_group_s<<','<<work.cad_patch_area_s;
        progress<<std::endl;
    };
    std::size_t impulse_solves=0,impulse_regions=0,impulse_iterations=0;
    double impulse_residual=0.;
    std::optional<double> device_wall;
    if (surface.resident) {
#ifdef NC_WITH_CUDA_COLUMNS
        namespace dev=compiled::device;
        if (surface.method!="cad-compiled" || !surface.predictor_geometry || surface.spectral_radius!=0.0 ||
            surface.impulse_law || has_history || !input.bushings.empty() || !input.resistances.empty() ||
            !surface.replay_file.empty() || event_tolerance>0 || surface.force_tolerance ||
            surface.moment_tolerance || surface.fixed_cubature || steps>std::numeric_limits<int>::max())
            throw std::invalid_argument("cuda-resident requires compiled CAD, predictor penalty and spectral radius zero");
        dev::ResidentScene model;
        auto vec=[](Vec v){return dev::V{{v[0],v[1],v[2]}};};
        for (std::size_t i=0;i<n;++i) {
            const auto &c=input.coordinates[i];
            int first=static_cast<int>(model.loads.size());
            if (i<input.loads.size()) for (auto &sample:input.loads[i]) model.loads.push_back({sample.time,sample.force});
            model.coordinates.push_back({c.mass,c.position,c.velocity,c.stiffness,c.damping,c.rest,c.load,
                                         first,static_cast<int>(model.loads.size())});
        }
        std::map<const compiled::Solid*,std::shared_ptr<dev::Columns>> uploads;
        std::map<const compiled::Solid*,double> reach;
        for(const auto &p:input.pairs)for(auto body:{p.a,p.b}){
            auto &r=reach[input.compiled_parts.at(body).get()];
            r=std::max(r,2000*p.material.maximum_depth);
        }
        for (std::size_t i=0;i<input.bodies.size();++i) {
            const auto &b=input.bodies[i];
            if (b.free || !std::holds_alternative<std::monostate>(b.drive) || b.translations.size()>1 ||
                (b.mass!=0 && length(input.gravity)>0))
                throw std::invalid_argument("unsupported resident joint, drive or gravitational loading");
            int axis=b.translations.empty() ? -1 : static_cast<int>(b.translations.front().coordinate);
            auto direction=b.translations.empty() ? Vec{} : b.translations.front().direction;
            model.bodies.push_back({vec(b.reference.origin),vec(b.reference.axis),vec(b.pivot),b.reference.angle,
                                    b.hinge,axis,vec(direction)});
            auto &part=input.compiled_parts.at(i);
            auto &upload=uploads[part.get()];
            if (!upload) upload=compiled::cuda_columns(*part,true,reach[part.get()]);
            model.geometry.push_back(upload);
        }
        for (auto &p:input.pairs) {
            const auto &m=p.material;
            if (m.friction!=0 || p.activation!=0) throw std::invalid_argument("unsupported resident friction or activation");
            model.pairs.push_back({static_cast<int>(p.a),static_cast<int>(p.b),m.stiffness,m.exponent,m.rate,
                                   m.maximum_depth,p.share_penalty_stiffness});
        }
        auto reduction=dev::RegionReduction::ordered;
        if (surface.region_reduction=="serial") reduction=dev::RegionReduction::serial;
        if (surface.region_reduction=="warp") reduction=dev::RegionReduction::warp;
        if (surface.region_reduction=="audit-warp") reduction=dev::RegionReduction::audit_warp;
        if (surface.region_reduction=="audit") reduction=dev::RegionReduction::audit;
        const auto execution=surface.gpu_execution=="batched" ? dev::ResidentExecution::batched :
            surface.gpu_execution=="cached" ? dev::ResidentExecution::cached : dev::ResidentExecution::cooperative;
        auto result=dev::advance_resident(model,dt,static_cast<int>(steps),reduction,execution,surface.gpu_profile);
        device_wall=result.online_s;
        setup_s+=std::chrono::duration<double>(Clock::now()-start).count()-*device_wall;
        stats.evaluations=result.forces;
        stats.maximum_residual=result.maximum_residual;
        stats.subdivisions=result.splits;
        stats.accepted_steps=steps+result.splits;
        work.cad_patch_queries=result.queries;
        auto gpu_stats=open_output(output.string()+".gpu.csv");
        gpu_stats<<"online_s,broad_s,sort_s,search_s,group_s,integral_s,p50_step_s,p99_step_s,max_step_s,missed_deadlines,steps,reduction_s,reductions,checked_links,execution";
        for (int i=0;i<9;++i) gpu_stats<<",accepted_level_"<<i;
        gpu_stats<<",proposals,face_groups,confirmed_groups,seed_hits,maximum_proposals,maximum_groups,blocker_checks,blocker_hits,warp_segments,scalar_segments,separated_groups,path_checks,path_hits,batched_rays,ray_batches";
        gpu_stats<<",fused_math,profiled_links";
        for (const char *name:{"cache","endpoints","direct","recursive"}) gpu_stats<<",link_"<<name<<"_work_s,link_"<<name<<"_calls";
        gpu_stats<<'\n';
        gpu_stats<<*device_wall;
        for(double t:result.phase_s)gpu_stats<<','<<t;
        gpu_stats<<','<<result.p50_s<<','<<result.p99_s<<','<<result.maximum_step_s<<','<<result.missed_deadlines<<','<<steps
                 <<','<<result.reduction_s<<','<<result.reductions<<','<<result.checked_links
                 <<','<<(execution==dev::ResidentExecution::batched ? "batched" :
                          execution==dev::ResidentExecution::cached ? "cached" : "cooperative");
        for (auto count:result.accepted_by_level) {
            gpu_stats<<',';
            if (execution==dev::ResidentExecution::batched) gpu_stats<<count;
        }
        gpu_stats<<','<<result.proposals<<','<<result.face_groups<<','<<result.confirmed_groups
                 <<','<<result.seed_hits<<','<<result.maximum_proposals<<','<<result.maximum_groups
                 <<','<<result.blocker_checks<<','<<result.blocker_hits<<','<<result.warp_segments<<','<<result.scalar_segments
                 <<','<<result.separated_groups<<','<<result.path_checks<<','<<result.path_hits
                 <<','<<result.batched_rays<<','<<result.ray_batches;
        gpu_stats<<','<<result.fused_math<<','<<result.profiled_links;
        for (int k=0;k<4;++k) gpu_stats<<','<<result.link_work_s[k]<<','<<result.link_calls[k];
        gpu_stats<<'\n';
        if (!result.region_audit.empty()) {
            auto audit=open_output(output.string()+".regions.csv");
            audit<<"step,pair,queries,regions,entered_face_pairs,exited_face_pairs,region_count_changes,"
                    "Fx_error_N,Fy_error_N,Fz_error_N,Ax_error_Nm,Ay_error_Nm,Az_error_Nm,Bx_error_Nm,By_error_Nm,Bz_error_Nm\n";
            for (size_t i=0;i<result.region_audit.size();++i) {
                const auto &r=result.region_audit[i];
                audit<<i/model.pairs.size()<<','<<i%model.pairs.size()<<','<<r.queries<<','<<r.regions
                     <<','<<r.entered_faces<<','<<r.exited_faces<<','<<r.count_changes;
                for (auto v:{r.force_error,r.torque_a_error,r.torque_b_error})
                    for (int j=0;j<3;++j) audit<<','<<v[j];
                audit<<'\n';
            }
        }
        const auto width=1+4*n+3*input.pairs.size();
        for (std::size_t i=0;i<=steps;++i) {
            if (i%stride && i!=steps) continue;
            auto f=result.frames.data()+i*width;
            State s{*f++,Vector(n),Vector(n),Vector(n),Vector(n)};
            for (std::size_t j=0;j<n;++j) {s.q[j]=*f++; s.v[j]=*f++; s.a[j]=*f++; s.force[j]=*f++;}
            samples.push_back(std::move(s));
            auto &row=observed_contacts.emplace_back(input.pairs.size());
            for (auto &p:row) {p.gap=*f++; p.normal_force=*f++; p.resolved=*f++!=0;}
        }
#else
        throw std::invalid_argument("rebuild with NC_WITH_CUDA_COLUMNS for cuda-resident");
#endif
    } else if (surface.impulse_law) {
        std::optional<Hht> smooth;
        std::size_t smooth_evaluations=0;
        if (n) smooth.emplace(mass,[&](const Vector& x,const Vector& u,double t) {
            ++smooth_evaluations;
            return evaluate_force(x,u,t,{});
        },options);
        State state{0,q,v,Vector(n),Vector(n)};
        if (smooth) state=smooth->initialize(q,v);
        std::vector<ContactSample> previous(input.pairs.size());
        const auto initial=kinematics(input,q,0);
        for (std::size_t i=0;i<input.pairs.size();++i) {
            const auto& pair=input.pairs[i];
            if (!pair.regions) throw std::invalid_argument("impulse solver requires surface geometry");
            for (const auto& region:pair.regions->regions(initial[pair.a].pose,initial[pair.b].pose,work)) {
                previous[i].gap=previous[i].resolved ? std::min(previous[i].gap,-region.depth_m) : -region.depth_m;
                previous[i].resolved=true;
            }
        }
        samples.push_back(state);
        observed_contacts.push_back(previous);
        progress_step(0,state);
        ImpulseSteps integrator{*surface.impulse_law,input,mass,smooth ? &*smooth : nullptr,accept_step,work,stats};
        for (std::size_t s=1;s<=steps;++s) {
            Vector normal(input.pairs.size());
            std::vector<Vec> friction(input.pairs.size());
            integrator.advance(state,previous,dt,normal,friction,options.max_subdivisions);
            if (s%stride==0 || s==steps) {
                samples.push_back(state);
                auto observations=previous;
                for (std::size_t i=0;i<observations.size();++i) {
                    observations[i].normal_force=normal[i]/dt;
                    observations[i].friction=scale(friction[i],1/dt);
                }
                observed_contacts.push_back(std::move(observations));
                progress_step(s,state);
            }
        }
        stats.evaluations=smooth_evaluations;
        stats.maximum_residual=integrator.smooth_stats.maximum_residual;
        impulse_solves=integrator.solves;
        impulse_regions=integrator.regions;
        impulse_iterations=integrator.iterations;
        impulse_residual=integrator.residual;
    } else if (n>0) {
        ForceFunction force;
        HistoryForceFunction historical_force;
        if (has_history) historical_force=evaluate_force;
        else force=[&](const Vector& q_,const Vector& v_,double time) {
            return evaluate_force(q_,v_,time,{});
        };
        Hht solver(mass,std::move(force),options,{},std::move(historical_force),{},std::move(accept_step),
                   [&] {for (auto& proxy:predictors) proxy->reset();});
        if (!surface.replay_file.empty()) {
            // Numerical audit of a single interval from a previously accepted
            // state. Preserve absolute time, inertia history and applied loads.
            if (has_history || predictors.empty())
                throw std::invalid_argument("HHT step replay requires history-free predictor contact");
            std::ifstream checkpoints(surface.replay_file);
            if (!checkpoints) throw std::invalid_argument("cannot open HHT replay file");
            auto log=open_output(output);
            log<<"case,time_s,substeps,subdivisions,rejected_trials";
            for (std::size_t i=0;i<n;++i) log<<",q"<<i<<",v"<<i<<",a"<<i<<",force"<<i;
            log<<'\n';
            for (std::size_t row=0;(checkpoints>>std::ws).peek()!=std::char_traits<char>::eof();++row) {
                const double time=take<double>(checkpoints);
                const auto count=take<int>(checkpoints);
                if (!std::isfinite(time) || count<1 || count>1024)
                    throw std::invalid_argument("invalid HHT replay interval");
                State state{time,Vector(n),Vector(n),Vector(n),Vector(n)};
                for (auto field:{&state.q,&state.v,&state.a,&state.force})
                    for (auto& value:*field) value=take<double>(checkpoints);
                HhtStats local;
                for (int i=0;i<count;++i) solver.step(state,dt/count,local);
                log<<row<<','<<state.time<<','<<count<<','<<local.subdivisions<<','<<local.rejected_trials;
                for (std::size_t i=0;i<n;++i) log<<','<<state.q[i]<<','<<state.v[i]<<','<<state.a[i]<<','<<state.force[i];
                log<<'\n';
            }
            return;
        }
        std::size_t depth_rejections_before=work.rejected_region_depths;
        try {
            auto state=solver.initialize(q,v);
            save_state(state);
            progress_step(0,state);
            for (std::size_t s=1;s<=steps;++s) {
                const bool save=s%stride==0 || s==steps;
                record_sample=reuse_mesh_samples && save;
                depth_rejections_before=work.rejected_region_depths;
                solver.step(state,dt,stats);
                for (const auto& body:input.bodies) if (body.free) body.free->normalize(state);
                if (save) {
                    save_state(state);
                    progress_step(s,state);
                }
            }
        } catch (const std::exception& error) {
            const auto depth_rejections=work.rejected_region_depths-depth_rejections_before;
            if (!work.unresolved_queries && !unresolved_events && !depth_rejections) throw;
            std::ostringstream detail;
            detail<<error.what()<<" ("<<work.unresolved_queries<<" unresolved pair force evaluations, "
                  <<unresolved_events<<" unresolved step sweeps)";
            if (const auto bound=work.last_unresolved_distance) {
                detail<<", last distance interval ["<<std::setprecision(9)<<bound->lo<<", "<<bound->hi<<"] m";
            }
            if (depth_rejections) {
                detail<<", "<<depth_rejections<<" trial depths exceeded the model limit during the failed step"
                      <<", last "<<work.last_rejected_depth_m*1e6<<" um versus "<<work.last_depth_limit_m*1e6
                      <<" um for bodies "<<work.last_depth_bodies[0]<<"/"<<work.last_depth_bodies[1];
            }
            throw std::runtime_error(detail.str());
        }
    } else {
        for (std::size_t s=0;s<=steps;++s) {
            if (s%stride==0 || s==steps) samples.push_back({s*dt,{},{},{},{}});
        }
    }
    const double wall=device_wall.value_or(std::chrono::duration<double>(Clock::now()-start).count());
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
    if (!predictors.empty()) for (std::size_t i=0;i<n;++i) trace<<",generalized_force"<<i;
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
        if (!predictors.empty()) for (double f:item.force) trace<<','<<f;
        trace<<'\n';
    }
    auto report=open_output(output.string()+".stats.csv");
    report<<"wall_s,setup_s,steps,accepted,subdivisions,force_evaluations,max_residual,pairs,culled,rays,missing,triangle_tests,"
            "closest_queries,unresolved_queries,native_nodes,ccd_checks,ccd_rejected,ccd_nodes,ccd_unresolved,saved_contact_reuses,"
            "mesh_boolean_queries,mesh_boolean_input_faces,mesh_local_solid_uses,mesh_local_solid_builds,"
            "mesh_detection_s,mesh_placement_s,mesh_boolean_s,cad_boolean_queries,cad_event_distances,chen_patch_queries,chen_confirmed_candidates,event_tolerance_s,"
            "solver,impulse_solves,impulse_regions,impulse_iterations,impulse_residual_m_s,ccd_refinements,ccd_refined_clear,"
            "tdib_cells,tdib_candidates,tdib_unresolved,cad_box_tests,cad_extrema,cad_classifications,"
            "cad_region_cache_hits,cad_distance_cache_hits,cad_witness_tests,cad_witness_improvements,"
            "mesh_pair_bvh_builds,mesh_pair_bvh_reuses,mesh_patch_builds,mesh_patch_reuses,mesh_parallel_batches,mesh_max_workers,"
            "cad_region_index_s,cad_region_select_s,cad_region_common_s,cad_region_extract_s,"
            "cad_patch_queries,cad_patch_batches,cad_patch_batch_s,cad_patch_broad_s,cad_patch_search_s,cad_patch_group_s,cad_patch_area_s,cad_window_builds,cad_window_reuses,cad_path_reuses,cad_skipped_area_nodes,cad_pressure_clipped,cad_pressure_full,cad_full_blocks,cad_partial_blocks,cad_shared_columns,"
            "cad_depth_tests,cad_depth_excluded,cad_depth_refinements,cad_depth_accepted,cad_depth_empty_blocks,cad_depth_interior_blocks,cad_quadrature_rules,cad_quadrature_converged,cuda_column_rays,cuda_column_subdivisions\n"
        <<(n || surface.impulse_law ? wall : replay_s)<<','<<setup_s<<','<<steps<<','<<stats.accepted_steps<<','<<stats.subdivisions<<','<<stats.evaluations
        <<','<<stats.maximum_residual<<','<<work.pairs<<','<<work.culled<<','<<work.rays<<','<<work.missing
        <<','<<work.mesh.triangles<<','<<work.closest_queries<<','<<work.unresolved_queries<<','<<work.native_nodes
        <<','<<stats.step_checks<<','<<stats.rejected_steps<<','<<event_nodes<<','<<unresolved_events<<','<<reused
        <<','<<work.boolean_queries<<','<<work.boolean_input_faces<<','<<work.local_solid_uses<<','<<work.local_solid_builds
        <<','<<work.mesh_detection_s<<','<<work.mesh_placement_s<<','<<work.mesh_boolean_s
        <<','<<work.cad_boolean_queries<<','<<cad_distances<<','<<chen_queries<<','<<chen_confirmed<<','<<event_tolerance
        <<','<<(surface.impulse_law ? impulse_law_name(*surface.impulse_law) : "penalty")
        <<','<<impulse_solves<<','<<impulse_regions<<','<<impulse_iterations<<','<<impulse_residual
        <<','<<spatial_refinements<<','<<refined_clear<<','<<tdib_cells<<','<<tdib_candidates<<','<<tdib_unresolved
        <<','<<cad_box_tests<<','<<cad_extrema<<','<<cad_classifications<<','<<work.cad_region_cache_hits
        <<','<<cad_distance_hits<<','<<cad_witness_tests<<','<<cad_witness_improvements
        <<','<<work.mesh_pair_bvh_builds<<','<<work.mesh_pair_bvh_reuses
        <<','<<work.mesh_patch_builds<<','<<work.mesh_patch_reuses<<','<<work.mesh_parallel_batches<<','<<work.mesh_max_workers
        <<','<<work.cad_region_index_s<<','<<work.cad_region_select_s<<','<<work.cad_region_common_s<<','<<work.cad_region_extract_s
        <<','<<work.cad_patch_queries<<','<<work.cad_patch_batches<<','<<work.cad_patch_batch_s<<','<<work.cad_patch_broad_s
        <<','<<work.cad_patch_search_s<<','<<work.cad_patch_group_s<<','<<work.cad_patch_area_s<<','<<work.cad_window_builds<<','<<work.cad_window_reuses<<','<<work.cad_path_reuses<<','<<work.cad_skipped_area_nodes<<','<<work.cad_pressure_clipped<<','<<work.cad_pressure_full<<','<<work.cad_full_blocks<<','<<work.cad_partial_blocks<<','<<work.cad_shared_columns;
    for (auto value:work.cad_depth_counts) report<<','<<value;
    report<<','<<work.cad_quadrature_rules<<','<<work.cad_quadrature_converged;
    report<<','<<work.cuda_column_rays<<','<<work.cuda_column_subdivisions;
    report<<'\n';
    if (!trace || !report) throw std::runtime_error("failed to write contact run");
}

#ifndef NC_COMPILED_ONLY
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

#endif
} // namespace

int main(int argc,char** argv) {
    try {
#ifndef NC_COMPILED_ONLY
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
#endif
        const bool named=argc>7 && std::string_view(argv[7]).starts_with("--");
        if (argc<7 || (!named && argc>11)) {
            std::cerr<<"nc_contact SCENE DEFLECTION_M DT_S DURATION_S OUTPUT_STRIDE OUTPUT_PREFIX "
                        "[pair-occt|pair-mesh|pair-brep [DISTANCE_TOL_M [MAX_NODES [EVENT_TOL_S]]]]\n"
                        "  or append --method bvh-aabb|bvh-obb|mesh-aabb|mesh-obb|mesh-sap [--mesh-dir DIR] [--solver LAW] "
                        "[--event-tol-s SECONDS --ccd-tol-m METRES] [--mesh-region-scope local|full|fcl|finite|contour-ray] [--penalty-stiffness per-region|shared] [--geometry-update newton|predictor] [--spectral-radius RHO]\n"
                        "  or append --method cad-compiled|cad-boundary [--device cpu|cuda-columns|cuda-resident] [--compiled-dir DIR] [--force-tol-n N --moment-tol-nm NM --cubature adaptive|fixed96]\n"
                        "     cuda-resident supports --region-reduction ordered|serial|audit|warp|audit-warp [--gpu-execution cooperative|cached|batched] [--gpu-profile on|off]\n"
                        "  or append --method cad-occt|cad-local|cad-patch|cad-bezier|cad-bezier-bounds|cad-bezier-relocate|chen-2024|chen-2026 [--solver LAW] [--event-tol-s SECONDS --ccd-tol-m METRES]\n"
                        "  LAW: penalty|sap|lagrange|augmented-lagrange|lcp|ncp|soccp [--progress FILE]\n";
            return 2;
        }
        std::optional<PairBackend> pair_backend;
        PairSearchOptions settings;
        SurfaceOptions surface;
        double event_tolerance=0.;
        if (named) {
            bool event_specified=false,region_specified=false;
            for (int i=7;i<argc;i+=2) {
                if (i+1==argc) throw std::invalid_argument("missing method, solver or mesh directory value");
                const std::string_view key=argv[i], value=argv[i+1];
                if (key=="--method") surface.method=value;
                else if (key=="--device") {
                    if(value!="cpu" && value!="cuda-columns" && value!="cuda-resident") throw std::invalid_argument("unknown contact device");
                    surface.cuda_columns=value=="cuda-columns";
                    surface.resident=value=="cuda-resident";
                }
                else if (key=="--progress") surface.progress_file=argv[i+1];
                else if (key=="--hht-replay") surface.replay_file=argv[i+1];
                else if (key=="--compiled-dir") surface.compiled_directory=argv[i+1];
                else if (key=="--force-tol-n") surface.force_tolerance=std::stod(argv[i+1]);
                else if (key=="--moment-tol-nm") surface.moment_tolerance=std::stod(argv[i+1]);
                else if (key=="--cubature") {
                    if (value!="adaptive" && value!="fixed96") throw std::invalid_argument("invalid cubature policy");
                    surface.fixed_cubature=value=="fixed96";
                }
                else if (key=="--mesh-dir") surface.mesh_directory=argv[i+1];
                else if (key=="--spectral-radius") surface.spectral_radius=std::stod(argv[i+1]);
                else if (key=="--geometry-update") {
                    if (value!="newton" && value!="predictor") throw std::invalid_argument("expected newton or predictor geometry");
                    surface.predictor_geometry=value=="predictor";
                }
                else if (key=="--penalty-stiffness") {
                    if (value!="per-region" && value!="shared") throw std::invalid_argument("expected per-region or shared stiffness");
                    surface.share_penalty_stiffness=value=="shared";
                }
                else if (key=="--mesh-region-scope") {
                    region_specified=true;
                    if (value!="local" && value!="full" && value!="fcl" && value!="finite" && value!="contour-ray")
                        throw std::invalid_argument("expected local, full, fcl, finite or contour-ray mesh contact geometry");
                    surface.mesh_regions=value;
                }
                else if (key=="--event-tol-s") {
                    event_tolerance=std::stod(argv[i+1]);
                    event_specified=true;
                }
                else if (key=="--ccd-tol-m") surface.distance_tolerance=std::stod(argv[i+1]);
                else if (key=="--region-reduction") {
                    if(value!="ordered" && value!="serial" && value!="warp" && value!="audit" && value!="audit-warp")
                        throw std::invalid_argument("unknown region reduction mode");
                    surface.region_reduction=value;
                }
                else if (key=="--gpu-execution") {
                    if(value!="cooperative" && value!="cached" && value!="batched") throw std::invalid_argument("unknown GPU execution mode");
                    surface.gpu_execution=value;
                }
                else if (key=="--gpu-profile") {
                    if (value!="on" && value!="off") throw std::invalid_argument("GPU profile expects on or off");
                    surface.gpu_profile=value=="on";
                }
                else if (key=="--solver") {
                    if (value=="penalty") surface.impulse_law.reset();
                    else surface.impulse_law=parse_impulse_law(value);
                }
                else throw std::invalid_argument("unsupported surface option or solver");
            }
            if (!surface.region_reduction.empty() && !surface.resident)
                throw std::invalid_argument("region reduction requires cuda-resident");
            if (!surface.gpu_execution.empty() && !surface.resident)
                throw std::invalid_argument("GPU execution mode requires cuda-resident");
            if (surface.gpu_profile && !surface.resident)
                throw std::invalid_argument("GPU profiling requires cuda-resident");
            if (surface.method=="bvh-aabb" || surface.method=="bvh-obb") {
                surface.method=surface.method=="bvh-aabb" ? "mesh-aabb" : "mesh-obb";
                if (!region_specified) surface.mesh_regions="finite";
            }
            if (surface.patch() && surface.impulse_law)
                throw std::invalid_argument("distributed CAD pressure requires the penalty solver");
            if (surface.force_tolerance.has_value()!=surface.moment_tolerance.has_value() ||
                ((surface.force_tolerance || surface.fixed_cubature) && surface.method!="cad-boundary") ||
                (surface.fixed_cubature && !*surface.fixed_cubature && !surface.force_tolerance))
                throw std::invalid_argument("adaptive CAD quadrature requires both load tolerances; cubature requires cad-boundary");
            if (surface.predictor_geometry && (surface.impulse_law || (surface.cad() && !surface.patch())))
                throw std::invalid_argument("predictor geometry requires a mesh or cad-patch penalty method");
            if (surface.share_penalty_stiffness && surface.impulse_law)
                throw std::invalid_argument("shared penalty stiffness requires the penalty solver");
            if (!surface.cad() && surface.method!="mesh-aabb" && surface.method!="mesh-obb" && surface.method!="mesh-sap")
                throw std::invalid_argument("unknown surface contact method");
            if (surface.chen() && !event_specified) event_tolerance=.2*std::stod(argv[3]);
            if (surface.chen() && !(event_tolerance>0))
                throw std::invalid_argument("Chen CCD requires positive --event-tol-s");
            if (!(std::isfinite(surface.distance_tolerance) && surface.distance_tolerance>0))
                throw std::invalid_argument("expected a positive finite CCD spatial tolerance in metres");
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
        if (surface.cuda_columns && !surface.compiled()) throw std::invalid_argument("cuda-columns requires compiled CAD");
#ifdef NC_COMPILED_ONLY
        if (!surface.compiled()) throw std::invalid_argument("expected --method cad-compiled or cad-boundary");
#endif
        simulate(argv[1],std::stod(argv[2]),std::stod(argv[3]),std::stod(argv[4]),
                 static_cast<unsigned>(std::stoul(argv[5])),argv[6],pair_backend,settings,event_tolerance,surface);
    }
#ifndef NC_COMPILED_ONLY
    catch (const Standard_Failure& error) {
        std::cerr<<error.GetMessageString()<<'\n';
        return 1;
    }
#endif
    catch (const std::exception& error) {
        std::cerr<<error.what()<<'\n';
        return 1;
    }
}
