#include "flank.hpp"
#include "nurbscontact/hht.hpp"

#include <Standard_Failure.hxx>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace nc::gears {
namespace {

using namespace dynamics;
using Path = std::filesystem::path;
using Clock = std::chrono::steady_clock;
constexpr double mm = .001;
constexpr double skin = .055;
constexpr double stiffness = 1.6e8;
constexpr double minimum_probe_gap = .002;
constexpr double tip_margin = .12;
constexpr double tip_transition = .75;

struct Stage {
    Gear driver, wheel;
    double z;
    double ratio() const { return double(wheel.teeth)/driver.teeth; }
    double center() const { return driver.pitch_radius()+wheel.pitch_radius(); }
    double beta_base() const {
        return std::atan(std::tan(driver.helix)*std::cos(driver.transverse_pressure()));
    }
    double low() const {
        const double a = driver.transverse_pressure();
        return std::max(driver.base_radius()*driver.roll_min()-driver.pitch_radius()*std::sin(a),
                        wheel.pitch_radius()*std::sin(a)-wheel.base_radius()*wheel.roll_max());
    }
    double high() const {
        const double a = driver.transverse_pressure();
        return std::min(driver.base_radius()*driver.roll_max()-driver.pitch_radius()*std::sin(a),
                        wheel.pitch_radius()*std::sin(a)-wheel.base_radius()*wheel.roll_min());
    }
    double base_pitch() const { return 2*pi*driver.base_radius()/driver.teeth; }
};

struct Case {
    std::string name;
    double rpm;
    double torque;
    std::vector<Stage> stages;
};

void read_profiles(Stage& stage, const Path& file) {
    std::ifstream input(file);
    std::string header;
    if (!std::getline(input,header) || header!="roll_deg,driver_um,driven_um")
        throw std::runtime_error("invalid NASA profile file: "+file.string());
    std::string line;
    while (std::getline(input,line)) {
        std::istringstream row(line);
        double roll, driver, driven;
        char first, second;
        auto& points=stage.driver.measured_relief;
        if (!(row>>roll>>first>>driver>>second>>driven) || !(row>>std::ws).eof()
                || first!=',' || second!=',' || !std::isfinite(roll+driver+driven)
                || (!points.empty() && radians(roll)<=points.back()[0]))
            throw std::runtime_error("invalid NASA profile row");
        points.push_back({radians(roll),driver*.001});
        stage.wheel.measured_relief.push_back({radians(roll),driven*.001});
    }
    if (!input.eof() || stage.driver.measured_relief.size()<2)
        throw std::runtime_error("incomplete NASA profile");
}

Case configuration(std::string_view name, const Path& folder={}) {
    constexpr double alpha = radians(20);
    constexpr double backlash = .18;
    const auto stage = [&](int a, int b, double module, double helix, double width, double z) {
        return Stage{{a,module,alpha,radians(helix),width,backlash},
                     {b,module,alpha,-radians(helix),width,backlash},z};
    };
    if (name == "spur") return {"spur",1800,56,{stage(24,48,2.5,0,24,0)}};
    if (name == "helical") return {"helical",1500,72,{stage(25,45,2.5,25,28,0)}};
    if (name == "reducer") return {"reducer",1800,148,{stage(24,48,2.5,0,24,0),stage(22,44,3.,22,32,38)}};
    if (name == "rig28" || name == "rig28-flex") {
        auto gears=stage(28,28,3.175,0,6.35,0);
        gears.driver.tip_relief=.023;
        gears.wheel.tip_relief=.025;
        if (name=="rig28-flex") {
            read_profiles(gears,folder/"rig28-profiles.csv");
        }
        return {std::string(name),2000,71.77,{gears}};
    }
    if (name == "unsw") {
        auto gears=stage(27,44,2.,0,5,0);
        gears.driver.bore_radius=10;
        gears.wheel.bore_radius=10;
        return {"unsw",300,10*44./27,{gears}};
    }
    throw std::invalid_argument("unknown gear case");
}

std::ofstream output(const Path& path) {
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path);
    file.exceptions(std::ios::badbit|std::ios::failbit);
    file << std::setprecision(17);
    return file;
}

Path component(const Path& folder, const Case& scene, std::size_t stage, char side, bool flank) {
    return folder/(scene.name+"-"+std::to_string(stage+1)+side+(flank ? "-flank.step" : ".step"));
}

void make_models(const Case& scene, const Path& target) {
    std::filesystem::create_directories(target);
    double center = 0;
    auto layout = output(target/(scene.name+".csv"));
    layout << "stage,side,shaft,direction,center_x_mm,z_mm,teeth,normal_module_mm,helix_deg,width_mm,"
              "angle_zero_rad,start_rpm,pitch_radius_mm,base_radius_mm,center_distance_mm,transverse_ratio,overlap_ratio,step\n";
    double speed = scene.rpm;
    for (std::size_t i=0; i<scene.stages.size(); ++i) {
        const Stage& stage = scene.stages[i];
        const double side_sign = i%2 ? -1. : 1.;
        const double overlap = stage.driver.width*std::abs(stage.driver.twist())*stage.driver.base_radius()/stage.base_pitch();
        for (const char side : {'a','b'}) {
            const bool driver = side=='a';
            const Gear& gear = driver ? stage.driver : stage.wheel;
            const Path solid = component(target,scene,i,side,false);
            const Path flank = component(target,scene,i,side,true);
            export_gear(gear,solid,flank,i%2!=0);
            const Flank imported(flank,0);
            const double fit = flank_fit(gear,imported);
            std::cout << flank.filename().string() << " STEP flank fit " << fit << " mm\n" << std::flush;
            if (fit > .002) throw std::runtime_error("STEP flank fit exceeded 2 micrometers");
            const double angle = side_sign*((driver?0.:pi)-pi/(2*gear.teeth));
            layout << i+1 << ',' << side << ',' << i+!driver << ',' << (driver?side_sign:-side_sign)
                << ',' << center+(driver?0.:stage.center()) << ',' << stage.z << ',' << gear.teeth << ','
                << gear.module << ',' << side_sign*gear.helix*180/pi << ',' << gear.width << ',' << angle << ','
                << speed/(driver?1.:stage.ratio()) << ',' << gear.pitch_radius() << ',' << gear.base_radius()
                << ',' << stage.center() << ',' << (stage.high()-stage.low())/stage.base_pitch() << ','
                << overlap << ',' << solid.filename().string() << '\n';
        }
        center += stage.center();
        speed /= stage.ratio();
    }
}

struct Measurements {
    double normal{};
    double axial{};
    double min_gap{std::numeric_limits<double>::infinity()};
    double peak_slice{};
    double tooth_weight{};
    double first_tooth{std::numeric_limits<double>::quiet_NaN()};
    double first_z{};
    bool valid{true};
};

struct ToothLoad {
    long index;
    double roll_degrees;
    double force;
};

// Bending, shear, radial compression and line-contact compliance in m/N.
double tooth_compliance(const Gear& gear, double roll) {
    constexpr double young=206.8427e9, poisson=.30;
    constexpr double shear=young/(2*(1+poisson));
    constexpr int slices=32;
    const double width=gear.width*mm;
    const double base=gear.base_radius()*mm;
    const double root=gear.root_radius()*mm;
    const double radius=base*std::hypot(1.,roll);
    const double tangential=base/radius;
    const double radial_squared=1-tangential*tangential;
    const double dx=(radius-root)/slices;
    double compliance=0;
    for (int i=0;i<slices;++i) {
        const double section=root+(i+.5)*dx;
        const double ratio=std::max(1.,section/base);
        const double angle=gear.half_angle(std::sqrt(ratio*ratio-1));
        const double thickness=2*section*std::sin(angle);
        const double area=width*thickness;
        const double moment=area*thickness*thickness/12;
        const double arm=radius-section;
        compliance+=dx*(tangential*tangential*(arm*arm/(young*moment)+1.2/(shear*area))
                        +radial_squared/(young*area));
    }
    return compliance;
}

struct FlexCompliance {
    static constexpr std::size_t size=513;
    std::array<double,size> stiffness;
    double low, high, pitch_stiffness;
    double mass;

    FlexCompliance(const Stage& stage, double requested_pitch_stiffness, double reduced_mass)
        : low(stage.low()),high(stage.high()),mass(reduced_mass) {
        constexpr double young=206.8427e9, poisson=.30;
        const auto at=[&](double s) {
            const double alpha=stage.driver.transverse_pressure();
            const auto& a=stage.driver;
            const auto& b=stage.wheel;
            const double roll_a=(a.pitch_radius()*std::sin(alpha)+s)/a.base_radius();
            const double roll_b=(b.pitch_radius()*std::sin(alpha)-s)/b.base_radius();
            const double hertz=4*(1-poisson*poisson)/(pi*young*a.width*mm);
            return 1/(tooth_compliance(a,roll_a)+tooth_compliance(b,roll_b)+hertz);
        };
        const double physical=at(0);
        pitch_stiffness=requested_pitch_stiffness>0 ? requested_pitch_stiffness : physical;
        if (pitch_stiffness>physical)
            throw std::invalid_argument("pitch stiffness cannot exceed the uncoupled tooth pair");
        const double series_compliance=1/pitch_stiffness-1/physical;
        for (std::size_t i=0;i<size;++i)
            stiffness[i]=1/(1/at(std::lerp(low,high,double(i)/(size-1)))+series_compliance);
    }

    double at(double s) const {
        const double u=std::clamp((s-low)/(high-low),0.,1.)*(size-1);
        const auto i=std::min(static_cast<std::size_t>(u),size-2);
        return std::lerp(stiffness[i],stiffness[i+1],u-i);
    }
};

struct ExperimentalContact {
    double stiffness;
    double damping;
    const Vector& velocity;
    std::vector<ToothLoad>& report;
    const FlexCompliance* flex{};
    double damping_ratio{};
};

double shoulder(double x, double width=tip_transition) {
    const double u = std::clamp((x-tip_margin)/width,0.,1.);
    return u*u*(3-2*u);
}

double trimmed_weight(Vec p, Vec n, Vec start, Vec end) {
    const Vec transverse{n[1],-n[0],0};
    const double first=dot3(sub(start,p),transverse);
    const double last=dot3(sub(end,p),transverse);
    const double margin=std::min(-std::min(first,last),std::max(first,last));
    const double u=std::clamp((margin-.03)/.18,0.,1.);
    return u*u*(3-2*u);
}

struct BoundedContactRequest {
    double target;
    bool reuse;
};

struct Pair {
    Stage stage;
    unsigned axial_stations;
    double omega;
    Flank a,b;
    std::vector<std::array<Vec,4>> ends;
    std::vector<std::array<std::optional<Flank::Hint>, 4>> cached_a, cached_b;
    QueryWork work;
    double activation_mm{skin};

    Pair(Stage data, const Path& folder, const Case& scene, std::size_t index, double speed,
         double deflection, unsigned axial)
        : stage(std::move(data)), axial_stations(stage.driver.helix == 0 ? 1u : axial), omega(speed),
          a(component(folder,scene,index,'a',true),deflection,axial_stations),
          b(component(folder,scene,index,'b',true),deflection,axial_stations),
          cached_a(axial_stations), cached_b(axial_stations) {
        for (unsigned i=0; i<axial_stations; ++i) {
            const double v=(i+.5)/axial_stations;
            ends.push_back({stage.driver.flank(0,v),stage.driver.flank(1,v),
                            stage.wheel.flank(0,v),stage.wheel.flank(1,v)});
        }
    }

    Measurements evaluate(double q1, double q2, double time, Forces* force = nullptr,
                          std::size_t shaft = 0, std::size_t n = 0,
                          ExperimentalContact* experimental = nullptr,
                          std::optional<BoundedContactRequest> bounded = std::nullopt) {
        const double alpha = stage.driver.transverse_pressure();
        const double cosine_base = std::cos(stage.beta_base());
        const Vec normal{std::sin(alpha),std::cos(alpha),0};
        const Vec tangent{normal[1],-normal[0],0};
        const Vec center_b{stage.center(),0,0};
        const double r1 = stage.driver.base_radius(), r2 = stage.wheel.base_radius();
        const double pitch = stage.base_pitch();
        const double lo = stage.low(), hi = stage.high();
        const double max_candidates = bounded ? (std::ceil((hi - lo) / pitch) + 1) * axial_stations : 1;
        const double angle = std::remainder(omega*time,2*pi);
        const double angle_b = angle/stage.ratio();
        const double turn = (omega*time-angle)/(2*pi);
        const double a0 = -pi/(2*stage.driver.teeth);
        const double b0 = pi-pi/(2*stage.wheel.teeth);
        const FlexCompliance* flex=experimental ? experimental->flex : nullptr;
        Measurements out;
        double active_weight = 0;
        for (unsigned station=0; station<axial_stations; ++station) {
            const double v = (station+.5)/axial_stations;
            const double z = stage.driver.width*(v-.5);
            const double phase = r1*(angle + stage.driver.twist()*z);
            const auto j0 = static_cast<long>(std::ceil((lo+tip_margin-phase)/pitch));
            for (long j=j0; ; ++j) {
                const double s = phase+j*pitch;
                if (s >= hi-tip_margin) break;
                const double theta_a = a0+angle+q1+j*2*pi/stage.driver.teeth;
                const double theta_b = b0-angle_b-q2-j*2*pi/stage.wheel.teeth;
                const Pose pa{{},std::cos(theta_a),std::sin(theta_a)};
                const Pose pb{center_b,std::cos(theta_b),std::sin(theta_b)};
                const Vec probe = add(Vec{stage.driver.pitch_radius(),0,z},scale(normal,s));
                const Vec local_a=pa.inverse_rotate(probe);
                const Vec local_b=pb.inverse_rotate(sub(probe,center_b));
                const Vec normal_a=pa.inverse_rotate(normal);
                const Vec normal_b=pb.inverse_rotate(normal);
                const auto& end=ends[station];
                const double transition=flex ? .06 : tip_transition;
                const double weight = shoulder(s-lo,transition)*shoulder(hi-s,transition)/axial_stations
                    * trimmed_weight(local_a,normal_a,end[0],end[1])
                    * trimmed_weight(local_b,normal_b,end[2],end[3]);
                if (weight <= 0) continue;
                const double rpa = stage.driver.pitch_radius()*std::sin(alpha);
                const double rpb = stage.wheel.pitch_radius()*std::sin(alpha);
                const double u1 = ((rpa+s)/r1-stage.driver.roll_min())/(stage.driver.roll_max()-stage.driver.roll_min());
                const double u2 = ((rpb-s)/r2-stage.wheel.roll_min())/(stage.wheel.roll_max()-stage.wheel.roll_min());
                double slope_a, slope_b;
                std::optional<double> sa, sb;
                Interval gap_range;
                if (bounded) {
                    // Allocate the stage's torque budget over both rays and
                    // all possible stations. The returned force bound uses
                    // the actual brackets rather than assuming this target.
                    const double target = bounded->target / (4 * max_candidates * weight * stiffness * mm
                        * std::max(r1,r2) * mm * cosine_base * cosine_base);
                    const auto slot = static_cast<std::size_t>((j % 4 + 4) % 4);
                    auto* hint_a = bounded->reuse ? &cached_a[station][slot] : nullptr;
                    auto* hint_b = bounded->reuse ? &cached_b[station][slot] : nullptr;
                    auto ra = a.probe_bound(local_a,normal_a,u1,v,station,target,work,hint_a);
                    auto rb = b.probe_bound(local_b,normal_b,u2,v,station,target,work,hint_b);
                    if (ra && rb && bounded->reuse && activation_mm>0 && ((*rb-*ra)*cosine_base).lo <= minimum_probe_gap) {
                        // A low-weight ray may meet the force budget while its
                        // bracket is too wide for the independent gap check.
                        ra = a.probe_bound(local_a,normal_a,u1,v,station,0,work,hint_a);
                        rb = b.probe_bound(local_b,normal_b,u2,v,station,0,work,hint_b);
                    }
                    if (ra && rb) {
                        sa = ra->center();
                        sb = rb->center();
                        gap_range = (*rb - *ra) * cosine_base;
                    }
                } else {
                    sa = a.probe(local_a,normal_a,u1,v,station,work,flex ? &slope_a : nullptr);
                    sb = b.probe(local_b,normal_b,u2,v,station,work,flex ? &slope_b : nullptr);
                }
                if (!sa || !sb) {
                    out.valid=false;
                    continue;
                }
                const double gap = (*sb-*sa)*cosine_base;
                out.min_gap = std::min(out.min_gap,bounded ? gap_range.lo : gap);
                active_weight += weight;
                std::array jacobian{-r1*cosine_base*mm,r2*cosine_base*mm};
                const double k0=flex ? flex->at(s) : (experimental ? experimental->stiffness : stiffness);
                const double k=weight*k0;
                double damping=0, explicit_rate=0;
                if (flex) {
                    const auto lever=[&](Vec origin,double ray,double slope) {
                        const Vec radial=sub(add(probe,scale(normal,ray)),origin);
                        const Vec rotation{-radial[1],radial[0],0};
                        return dot3(sub(normal,scale(tangent,slope)),rotation)*cosine_base*mm;
                    };
                    jacobian={-lever({},*sa,slope_a),-lever(center_b,*sb,slope_b)};
                    explicit_rate=omega*(jacobian[0]+jacobian[1]/stage.ratio());
                    damping=weight*2*experimental->damping_ratio*std::sqrt(k0*flex->mass);
                } else if (experimental) {
                    damping=weight*experimental->damping;
                }
                double normal_force = k*mm*std::max(0.,activation_mm-gap);
                Interval normal_range;
                if (bounded) {
                    const Interval depth = Interval{activation_mm} - gap_range;
                    normal_range = Interval{k} * mm * Interval{std::max(0.0,depth.lo), std::max(0.0,depth.hi)};
                    normal_force = normal_range.center();
                }
                if (experimental) {
                    if (normal_force > 0) {
                        const auto& v=experimental->velocity;
                        const double rate=explicit_rate+jacobian[0]*v[shaft]+jacobian[1]*v[shaft+1];
                        normal_force=std::max(0.,normal_force-damping*rate);
                    }
                    const double roll=(rpb-s-*sb)/r2;
                    const long index=j-std::lround(turn)*stage.driver.teeth;
                    experimental->report.push_back({index,roll*180/pi,normal_force});
                }
                out.normal += normal_force;
                if (normal_force <= 0 && (!bounded || normal_range.hi <= 0)) continue;
                out.tooth_weight += 1./axial_stations;
                if (normal_force > out.peak_slice) {
                    out.peak_slice = normal_force;
                    out.first_tooth = s+(*sa+*sb)/2;
                    out.first_z = z;
                }
                if (force) {
                    const std::array indices{shaft,shaft+1};
                    double added_error = 0;
                    for (std::size_t i=0; i<2; ++i) {
                        const double previous = force->value[indices[i]];
                        force->value[indices[i]] += jacobian[i]*normal_force;
                        if (bounded) {
                            const Interval allowed = Interval{previous} + jacobian[i] * normal_range;
                            added_error = std::max(added_error, magnitude(allowed - force->value[indices[i]]));
                        }
                        for (std::size_t j=0; j<2; ++j) {
                            force->stiffness[indices[i]*n+indices[j]] += k*jacobian[i]*jacobian[j];
                            if (experimental) {
                                force->damping[indices[i]*n+indices[j]] +=
                                    damping*jacobian[i]*jacobian[j];
                            }
                        }
                    }
                    if (bounded) force->error_bound = (Interval{force->error_bound} + added_error).hi;
                }
            }
        }
        out.axial = -out.normal*std::sin(stage.beta_base());
        out.valid = out.valid && active_weight > 0;
        return out;
    }
};

double inertia(const Gear& g) {
    constexpr double density = 7850;
    const double radius = g.pitch_radius()*mm;
    const double bore = g.bore()*mm;
    return .5*pi*density*(g.width*mm)*(std::pow(radius,4)-std::pow(bore,4));
}

struct Dynamics {
    std::vector<Pair> pairs;
    Vector mass;
    std::vector<Measurements> last;
    double brake;
    double speed;
    double input_torque;

    Dynamics(const Case& scene,const Path& folder,double deflection,unsigned axial,double activation_mm=skin)
        : mass(scene.stages.size()+1,.0003), last(scene.stages.size()), brake(scene.torque),
          speed(scene.rpm*2*pi/60), input_torque(brake) {
        double omega = speed;
        for (std::size_t i=0; i<scene.stages.size(); ++i) {
            const auto& s=scene.stages[i];
            pairs.emplace_back(s,folder,scene,i,omega,deflection,axial);
            pairs.back().activation_mm=activation_mm;
            mass[i] += inertia(s.driver);
            mass[i+1] += inertia(s.wheel);
            omega /= s.ratio();
            input_torque /= s.ratio();
        }
    }

    Vector initial() const {
        Vector q(mass.size(),0);
        double drive_torque = input_torque;
        for (std::size_t i=0; i<pairs.size(); ++i) {
            const Stage& stage=pairs[i].stage;
            const double c=std::cos(stage.beta_base());
            const double force=drive_torque/(stage.driver.base_radius()*mm*c);
            // Preserve the original initial shaft positions when comparing contact laws.
            const double closed = stage.driver.backlash/2-(skin-force/stiffness/mm)/c;
            q[i+1]=(stage.driver.base_radius()*q[i]-closed)/stage.wheel.base_radius();
            drive_torque*=stage.ratio();
        }
        return q;
    }

    double load(double time) const {
        const double raised = .5*(1+std::tanh((time-.10)/.018));
        const double released = .5*(1+std::tanh((time-.24)/.028));
        return brake*(1+.26*raised-.34*released+.06*std::sin(2*pi*13*time));
    }

    Forces evaluate(const Vector& q,const Vector& v,double time,
                    std::optional<BoundedContactRequest> bounded = std::nullopt) {
        const std::size_t n=q.size();
        Forces f{Vector(n),Vector(n*n),Vector(n*n)};
        for (std::size_t i=0; i<n; ++i) {
            f.value[i]=-.09*v[i];
            f.damping[i*n+i]=.09;
        }
        f.value[0] += input_torque + .10*input_torque*std::sin(2*pi*23*time)-1800*q[0]-.6*v[0];
        f.stiffness[0]=1800;
        f.damping[0]+=.6;
        f.value[n-1] -= load(time);
        if (bounded) bounded->target /= static_cast<double>(pairs.size());
        for (std::size_t i=0; i<pairs.size(); ++i) {
            last[i]=pairs[i].evaluate(q[i],q[i+1],time,&f,i,n,nullptr,bounded);
            if (!last[i].valid || (pairs[i].activation_mm>0 && last[i].min_gap < minimum_probe_gap)) {
                f.value[0]=std::numeric_limits<double>::quiet_NaN();
            }
        }
        return f;
    }
};

struct Frame { double time; Vector q,v; std::vector<Measurements> mesh; };

void integrate(const Case& scene,const Path& geometry,double deflection,double dt,double duration,
               const Path& prefix,unsigned repeats,unsigned stations,unsigned trace_stride,
               std::string_view force_strategy, double activation_mm,const Path& trial_path) {
    if (!(dt>0 && duration>0 && repeats && stations && trace_stride)) {
        throw std::invalid_argument("invalid run protocol");
    }
    if (force_strategy != "fixed" && deflection != 0) throw std::invalid_argument("bounded probes require CAD geometry");
    const auto steps = static_cast<std::size_t>(std::llround(duration/dt));
    if (std::abs(steps*dt-duration)>1e-12) throw std::invalid_argument("duration must be divisible by dt");
    const auto prep_start=Clock::now();
    Dynamics system(scene,geometry,deflection,stations,activation_mm);
    const double setup=std::chrono::duration<double>(Clock::now()-prep_start).count();
    auto stats_out = output(prefix.string()+".stats.csv");
    stats_out << "repeat,deflection_mm,dt_s,duration_s,steps,wall_s,setup_s,accepted,subdivisions,evaluations,newton,"
                 "probe_calls,surface_evaluations,probe_misses,bvh_nodes,triangle_tests,triangles,max_residual_Nm,"
                 "trace_stride,force_strategy,force_refinements,certificate_hits,activation_mm\n";
    std::vector<Frame> frames;
    frames.reserve(steps/trace_stride+2);
    HhtOptions opts;
    opts.absolute_tolerance=1e-7;
    opts.relative_tolerance=1e-7;
    opts.max_subdivisions=6;
    std::optional<std::ofstream> trials;
    unsigned repetition=0;
    if (!trial_path.empty()) {
        trials.emplace(output(trial_path));
        *trials << "repeat,time";
        for (std::size_t i=0; i<system.mass.size(); ++i) *trials << ",q" << i;
        *trials << '\n';
    }
    const auto record_trial=[&](double time,const Vector& q) {
        if (!trials) return;
        *trials << repetition << ',' << time;
        for (const double value:q) *trials << ',' << value;
        *trials << '\n';
    };
    BoundedForceFunction bounded;
    if (force_strategy != "fixed") {
        bounded = [&](const Vector& q, const Vector& v, double t, const Vector&, double target) {
            record_trial(t,q);
            return system.evaluate(q, v, t, BoundedContactRequest{
                force_strategy == "bounded-exact" ? 0 : target, force_strategy == "bounded-cache"});
        };
    }
    const Hht integrator(system.mass,[&](const Vector& q,const Vector& v,double t){
                             record_trial(t,q);
                             return system.evaluate(q,v,t);},
                         opts, {}, {}, std::move(bounded));
    for (repetition=0; repetition<repeats; ++repetition) {
        for (auto& pair:system.pairs) {
            pair.work={};
            if (force_strategy == "bounded-cache") {
                std::fill(pair.cached_a.begin(), pair.cached_a.end(), std::array<std::optional<Flank::Hint>, 4>{});
                std::fill(pair.cached_b.begin(), pair.cached_b.end(), std::array<std::optional<Flank::Hint>, 4>{});
            }
        }
        HhtStats stats;
        const auto start=Clock::now();
        State state=integrator.initialize(system.initial(),Vector(system.mass.size()));
        if (repetition==0) frames.push_back({state.time,state.q,state.v,system.last});
        for (std::size_t i=0; i<steps; ++i) {
            try {
                integrator.step(state,dt,stats);
            } catch (const std::exception& error) {
                std::cerr << scene.name << " fails at " << state.time << " s, q:";
                for (double q:state.q) std::cerr << ' ' << q;
                for (const auto& m:system.last) std::cerr << " | valid " << m.valid << " gap " << m.min_gap;
                std::cerr << '\n';
                throw;
            }
            if (repetition==0 && ((i+1)%trace_stride==0 || i+1==steps)) {
                frames.push_back({state.time,state.q,state.v,system.last});
            }
        }
        const double wall=std::chrono::duration<double>(Clock::now()-start).count();
        std::size_t calls=0, evaluations=0, misses=0, nodes=0, tests=0, triangles=0, hits=0;
        for (const auto& p:system.pairs) {
            calls+=p.work.rays;
            evaluations+=p.work.evaluations;
            misses+=p.work.misses;
            hits+=p.work.certificate_hits;
            nodes+=p.work.mesh.nodes;
            tests+=p.work.mesh.triangles;
            if (p.a.mesh) triangles+=p.a.mesh->triangles().size()+p.b.mesh->triangles().size();
        }
        stats_out << repetition << ',' << deflection << ',' << dt << ',' << duration << ',' << steps << ',' << wall << ',' << setup
            << ',' << stats.accepted_steps << ',' << stats.subdivisions << ',' << stats.evaluations << ','
            << stats.newton_iterations << ',' << calls << ',' << evaluations << ',' << misses << ',' << nodes
            << ',' << tests << ',' << triangles << ',' << stats.maximum_residual << ',' << trace_stride
            << ',' << force_strategy << ',' << stats.force_refinements << ',' << hits << ',' << activation_mm << '\n' << std::flush;
        std::cout << scene.name << ' ' << deflection << " mm pass " << repetition+1 << " " << wall << " s\n" << std::flush;
    }
    Dynamics audit(scene,geometry,0,stations,activation_mm);
    auto trace=output(prefix.string()+".csv");
    trace << "time,load_Nm";
    for (std::size_t i=0; i<system.mass.size(); ++i) trace << ",q" << i << ",v" << i;
    for (std::size_t i=0; i<system.pairs.size(); ++i) {
        trace << ",normal" << i << "_N,axial" << i << "_N,te" << i << "_um,native_gap" << i
              << "_mm,mesh_gap" << i << "_mm,active" << i << ",station" << i << "_mm,z" << i << "_mm";
    }
    trace << '\n';
    for (const Frame& frame:frames) {
        trace << frame.time << ',' << system.load(frame.time);
        for (std::size_t i=0; i<frame.q.size(); ++i) trace << ',' << frame.q[i] << ',' << frame.v[i];
        for (std::size_t i=0; i<system.pairs.size(); ++i) {
            const auto& stage = system.pairs[i].stage;
            const double te = (frame.q[i+1]-frame.q[i]/stage.ratio())*stage.wheel.base_radius()*1000;
            const auto d=audit.pairs[i].evaluate(frame.q[i],frame.q[i+1],frame.time);
            // In bounded runs report the native load at the accepted state,
            // independent of the last force evaluation used in line search.
            const auto& m=force_strategy == "fixed" ? frame.mesh[i] : d;
            trace << ',' << m.normal << ',' << m.axial << ',' << te << ','
                << (d.valid ? d.min_gap : std::numeric_limits<double>::quiet_NaN()) << ',' << m.min_gap << ','
                << m.tooth_weight << ',' << m.first_tooth << ',' << m.first_z;
        }
        trace << '\n';
    }
}

struct StaticPose {
    double angle;
    Measurements measurement;
    double torque;
};

StaticPose static_pose(Pair& pair, const FlexCompliance& flex, double time, double output_torque,
                       std::vector<ToothLoad>& teeth) {
    const double radius=pair.stage.wheel.base_radius();
    const double nominal=pair.stage.wheel.backlash/2-skin;
    const Vector zero(2);
    ExperimentalContact contact{0,0,zero,teeth,&flex,0};
    const auto evaluate=[&](double angle) {
        teeth.clear();
        Forces forces{Vector(2),Vector(4),Vector(4)};
        const auto result=pair.evaluate(0,angle,time,&forces,0,2,&contact);
        if (!result.valid) throw std::runtime_error("unresolved static gear flank");
        return StaticPose{angle,result,forces.value[1]};
    };
    double low=-(nominal+.07)/radius, high=-(nominal-.07)/radius;
    if (evaluate(low).torque<output_torque || evaluate(high).torque>output_torque)
        throw std::runtime_error("static gear load is outside the bracket");
    for (unsigned i=0;i<40;++i) {
        const double mid=(low+high)/2;
        if (evaluate(mid).torque>output_torque) low=mid;
        else high=mid;
    }
    const auto result=evaluate((low+high)/2);
    if (result.measurement.min_gap<.002 || std::abs(result.torque-output_torque)>1e-6)
        throw std::runtime_error("static gear load or clearance did not converge");
    return result;
}

void static_experiment(const Case& scene,const Path& geometry,double deflection,double torque,
                       double k,const Path& file) {
    if (scene.name!="rig28-flex" || !(torque>0 && k>=0))
        throw std::invalid_argument("invalid static gear settings");
    const auto& stage=scene.stages.front();
    Pair pair(stage,geometry,scene,0,1,deflection,1);
    const FlexCompliance flex(stage,k,0);
    constexpr unsigned samples=224;
    auto csv=output(file);
    csv << "mesh_cycle,te_um,normal_N,gap_mm,loaded,pitch_stiffness_N_m,output_torque_Nm";
    for (unsigned i=0;i<2;++i) csv << ",id" << i << ",roll" << i << "_deg,force" << i << "_N";
    csv << '\n';
    std::vector<ToothLoad> teeth;
    for (unsigned i=0;i<samples;++i) {
        const double phase=double(i)/samples;
        const double time=phase*2*pi/stage.driver.teeth;
        const auto result=static_pose(pair,flex,time,torque*stage.ratio(),teeth);
        const auto& m=result.measurement;
        csv << phase << ',' << result.angle*stage.wheel.base_radius()*1000 << ',' << m.normal
            << ',' << m.min_gap << ',' << m.tooth_weight << ',' << flex.pitch_stiffness << ',' << result.torque;
        if (teeth.size()>2) throw std::logic_error("unexpected static spur tooth count");
        for (const auto& tooth:teeth)
            csv << ',' << tooth.index << ',' << tooth.roll_degrees << ',' << tooth.force;
        for (std::size_t j=teeth.size();j<2;++j) csv << ",,,";
        csv << '\n';
    }
}

void experiment(Case scene,const Path& geometry,double deflection,double rpm,double input_torque,
                double dt,double duration,double k,double zeta,const Path& prefix,double inertia_scale) {
    const bool flexible=scene.name=="rig28-flex";
    const bool nasa = scene.name == "rig28" || flexible;
    if ((!nasa && scene.name!="unsw") || !(rpm>0 && input_torque>0 && dt>0 &&
            duration>0 && (k>0 || (flexible && k==0)) && zeta>=0 && inertia_scale>0)) {
        throw std::invalid_argument("invalid experimental gear settings");
    }
    const auto stage=scene.stages.front();
    const double omega=rpm*2*pi/60;
    const double ratio=stage.ratio();
    const double output_torque=input_torque*ratio;
    const double r1=stage.driver.base_radius()*mm, r2=stage.wheel.base_radius()*mm;
    const auto steps = static_cast<std::size_t>(std::llround(duration/dt));
    const auto preparation=Clock::now();
    Pair pair(stage,geometry,scene,0,omega,deflection,1);
    constexpr double pound_force_inch=.1129848290276167;
    Vector mass = flexible ? Vector{2.10*pound_force_inch,.00322*pound_force_inch,
                                  .00322*pound_force_inch,7.5*pound_force_inch}
        : nasa ? Vector{.0237,.0000364,.0000364,.085}
               : Vector{inertia(stage.driver)+.0003,inertia(stage.wheel)+.0003};
    if (nasa) for (double& value:mass) value*=inertia_scale;
    const double setup=std::chrono::duration<double>(Clock::now()-preparation).count();
    const std::size_t n=mass.size(), shaft=nasa ? 1 : 0;
    const double reduced=1/(r1*r1/mass[shaft]+r2*r2/mass[shaft+1]);
    std::optional<FlexCompliance> flex;
    if (flexible) {
        flex.emplace(stage,k,reduced);
        k=flex->pitch_stiffness;
    }
    const double dashpot=2*zeta*std::sqrt(k*reduced);
    constexpr double drive_shaft=341, load_shaft=12700;
    const double drive_damp=.10*std::sqrt(drive_shaft/(1/mass[0]+1/mass[1]));
    const double load_damp=.10*std::sqrt(load_shaft/(1/mass[n-2]+1/mass[n-1]));
    std::vector<ToothLoad> teeth;
    teeth.reserve(4);
    Measurements current;
    auto force = [&](const Vector& q,const Vector& v,double time) {
        Forces f{Vector(n),Vector(n*n),Vector(n*n)};
        f.value.front()=input_torque;
        f.value.back()=-output_torque;
        const auto spring=[&](std::size_t a,std::size_t b,double stiffness_value,double damping_value) {
            const double torque=stiffness_value*(q[a]-q[b])+damping_value*(v[a]-v[b]);
            f.value[a]-=torque;
            f.value[b]+=torque;
            for (const std::size_t i : {a,b}) {
                f.stiffness[i*n+i]+=stiffness_value;
                f.damping[i*n+i]+=damping_value;
            }
            f.stiffness[a*n+b]-=stiffness_value;
            f.stiffness[b*n+a]-=stiffness_value;
            f.damping[a*n+b]-=damping_value;
            f.damping[b*n+a]-=damping_value;
        };
        if (nasa) {
            spring(0,1,drive_shaft,drive_damp);
            spring(2,3,load_shaft,load_damp);
        } else {
            f.value[0]-=1800*q[0]+.6*v[0];
            f.stiffness[0]+=1800;
            f.damping[0]+=.6;
        }
        teeth.clear();
        ExperimentalContact contact{k,dashpot,v,teeth,flex ? &*flex : nullptr,zeta};
        current=pair.evaluate(q[shaft],q[shaft+1],time,&f,shaft,n,&contact);
        if (!current.valid || current.min_gap < .002) {
            f.value[0]=std::numeric_limits<double>::quiet_NaN();
        }
        return f;
    };
    const double closed=stage.driver.backlash/2-skin+input_torque/(k*r1*mm);
    Vector q(n);
    q[shaft+1]=-closed*mm/r2;
    if (flex) q[shaft+1]=static_pose(pair,*flex,0,output_torque,teeth).angle;
    if (nasa) {
        q[0]=input_torque/drive_shaft;
        q[3]=q[2]-output_torque/load_shaft;
    }
    HhtOptions options;
    options.absolute_tolerance=1e-7;
    options.relative_tolerance=1e-7;
    Hht solver(mass,force,options);
    HhtStats stats;
    struct Record {double time, angle, te, min_gap; std::array<ToothLoad,4> teeth; std::size_t count;};
    std::vector<Record> records;
    const auto start=Clock::now();
    State state=solver.initialize(q,Vector(n));
    for (std::size_t i=0;i<steps;++i) {
        solver.step(state,dt,stats);
        if (i % 4!=0 || i<steps/2) continue;
        Record record{state.time,omega*state.time+state.q[shaft],
                      (state.q[shaft+1]*r2-state.q[shaft]*r1)*1e6,current.min_gap,{},teeth.size()};
        if (teeth.size()>record.teeth.size()) throw std::logic_error("unexpected active spur tooth count");
        std::copy(teeth.begin(),teeth.end(),record.teeth.begin());
        records.push_back(record);
    }
    const double wall=std::chrono::duration<double>(Clock::now()-start).count();
    auto trace=output(prefix.string()+".csv");
    trace << "time,pinion_angle_rad,te_um,gap_mm";
    for (std::size_t i=0;i<4;++i) trace << ",id" << i << ",roll" << i << "_deg,force" << i << "_N";
    trace << '\n';
    for (const auto& record:records) {
        trace << record.time << ',' << record.angle << ',' << record.te << ',' << record.min_gap;
        for (std::size_t j=0;j<record.teeth.size();++j) {
            if (j < record.count) {
                const auto& tooth=record.teeth[j];
                trace << ',' << tooth.index << ',' << tooth.roll_degrees << ',' << tooth.force;
            } else {
                trace << ",,,";
            }
        }
        trace << '\n';
    }
    auto summary=output(prefix.string()+".stats.csv");
    summary << "rpm,input_torque_Nm,axial_stations,dt_s,duration_s,steps,accepted,subdivisions,wall_s,setup_s,"
               "stiffness_N_m,damping_ratio,deflection_mm,probe_calls,probe_misses,triangles_tested,max_residual_Nm\n";
    summary << rpm << ',' << input_torque << ",1," << dt << ',' << duration << ',' << steps << ','
        << stats.accepted_steps << ',' << stats.subdivisions << ',' << wall << ',' << setup << ','
        << k << ',' << zeta << ',' << deflection << ',' << pair.work.rays << ',' << pair.work.misses << ','
        << pair.work.mesh.triangles << ',' << stats.maximum_residual << '\n';
}

}  // namespace
}  // namespace nc::gears

int main(int argc,char** argv) {
    using namespace nc::gears;
    try {
        if (argc==4 && std::string_view(argv[1])=="models") {
            make_models(configuration(argv[2],argv[3]),argv[3]);
        } else if (argc>=10 && argc<=14 && std::string_view(argv[1])=="run") {
            const int trace_stride = argc>=11 ? std::stoi(argv[10]) : 1;
            if (trace_stride<=0) throw std::invalid_argument("trace stride must be positive");
            const std::string_view force_strategy = argc>=12 ? argv[11] : "fixed";
            const std::string_view activation = argc>=13 ? argv[12] : "skin";
            if (activation!="skin" && activation!="zero-gap") throw std::invalid_argument("unknown contact activation");
            if (force_strategy!="fixed" && force_strategy!="bounded-exact" && force_strategy!="bounded-cache") {
                throw std::invalid_argument("unknown gear force strategy");
            }
            integrate(configuration(argv[2]),argv[3],std::stod(argv[4]),std::stod(argv[5]),std::stod(argv[6]),
                      argv[7],static_cast<unsigned>(std::stoi(argv[8])),static_cast<unsigned>(std::stoi(argv[9])),
                      static_cast<unsigned>(trace_stride),force_strategy,activation=="skin" ? skin : 0.,
                      argc==14 ? Path(argv[13]) : Path{});
        } else if ((argc==12 || argc==13) && std::string_view(argv[1])=="experiment") {
            experiment(configuration(argv[2],argv[3]),argv[3],std::stod(argv[4]),std::stod(argv[5]),std::stod(argv[6]),
                       std::stod(argv[7]),std::stod(argv[8]),std::stod(argv[9]),std::stod(argv[10]),argv[11],
                       argc==13 ? std::stod(argv[12]) : 1.);
        } else if (argc==8 && std::string_view(argv[1])=="statics") {
            static_experiment(configuration(argv[2],argv[3]),argv[3],std::stod(argv[4]),std::stod(argv[5]),
                              std::stod(argv[6]),argv[7]);
        } else {
            std::cerr << "nc_gears models {spur|helical|reducer|rig28|rig28-flex|unsw} MODEL_DIR\n"
                         "nc_gears run CASE MODEL_DIR DEFLECTION_MM DT_S DURATION_S PREFIX REPEATS AXIAL_STATIONS "
                         "[TRACE_STRIDE [fixed|bounded-exact|bounded-cache [skin|zero-gap [TRIAL_STATES.csv]]]]\n"
                         "nc_gears experiment CASE MODEL_DIR DEFLECTION_MM RPM INPUT_NM DT_S DURATION_S K_N_M ZETA PREFIX [INERTIA_SCALE]\n"
                         "nc_gears statics rig28-flex MODEL_DIR DEFLECTION_MM INPUT_NM K_N_M FILE\n";
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
