#include "scene.hpp"
#include "../step_triangles.hpp"

#include "nurbscontact/occt.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <string>

namespace {

using namespace nc;
using namespace nc::dynamics;
using namespace nc::hht_bench;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

double microseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end-start).count();
}

std::ofstream output(const fs::path& path) {
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
    std::ofstream file(path);
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file << std::setprecision(17);
    return file;
}

Backend parse_backend(std::string_view value) {
    if (value == "brep") return Backend::cad;
    if (value == "mesh") return Backend::mesh;
    if (value == "mesh-support") return Backend::mesh_support;
    if (value == "general-brep") return Backend::general_cad;
    throw std::invalid_argument("unknown geometry backend");
}

unsigned facets_arg(const char* value) {
    const int n = std::stoi(value);
    if (n < 8 || n > 8192 || n % 2) throw std::invalid_argument("even facet count must be between 8 and 8192");
    return static_cast<unsigned>(n);
}

void oscillator(const fs::path& file) {
    constexpr double omega = 10;
    constexpr double duration = 1.2;
    Hht stepper({1}, [](const Vector& q, const Vector&, double) {
        return Forces{{-omega*omega*q[0]}, {omega*omega}, {0}};
    });
    auto out = output(file);
    out << "dt,steps,rms_displacement,end_displacement_error,max_residual\n";
    for (int steps : {60,120,240,480}) {
        HhtStats stats;
        auto state = stepper.initialize({1},{0});
        const double h = duration/steps;
        double error = 0;
        for (int i = 0; i < steps; ++i) {
            stepper.step(state,h,stats);
            const double delta = state.q[0]-std::cos(omega*state.time);
            error += delta*delta;
        }
        out << h << ',' << steps << ',' << std::sqrt(error/steps) << ','
            << std::abs(state.q[0]-std::cos(omega*duration)) << ',' << stats.maximum_residual << '\n';
    }
}

struct Snapshot { double t; Vector q; Vector v; double peak_force; };

void simulation(std::string_view scene_name, Backend backend, unsigned facets, double step, double duration,
                const fs::path& models, const fs::path& prefix, int repeats, bool warm, bool cache_guard,
                std::string_view force_strategy) {
    if (!(std::isfinite(step) && step > 0 && std::isfinite(duration) && duration >= step) ||
        duration / step > 1'000'000 || repeats < 1) throw std::invalid_argument("invalid simulation duration");
    if (force_strategy != "fixed" && backend != Backend::cad) {
        throw std::invalid_argument("bounded force strategies require the analytic CAD fixture backend");
    }
    const auto prep_start = Clock::now();
    const Scene scene = make_scene(scene_name, models, facets);
    const double prep = microseconds(prep_start,Clock::now());
    const auto steps = static_cast<std::size_t>(std::ceil(duration/step-1e-10));
    const double h = duration/static_cast<double>(steps);
    auto summary = output(prefix.string()+".stats.csv");
    summary << "scene,backend,facets,warm,cache_guard,repeat,duration,dt,steps,prep_us,wall_us,dofs,triangles,accepted,"
               "subdivisions,evaluations,newton,rejected,max_residual,min_guard,queries,analytic,cad_fallback,"
               "bvh_nodes,triangle_tests,guard_endpoint_evaluations,force_strategy,force_refinements,"
               "bounded_pairs,approximated_pairs,reused_exclusions\n";
    auto description = output(prefix.string()+".json");
    write_description(description, scene);
    std::vector<Snapshot> trajectory;
    trajectory.reserve(steps+1);

    std::size_t triangles = 0;
    for (const Body& body : scene.bodies) triangles += scene.geometry[body.geometry].mesh->triangles().size();
    for (int repetition = 0; repetition < repeats; ++repetition) {
        Dynamics system(scene,backend,warm);
        ClearanceGuard guard(scene,cache_guard);
        HhtOptions options;
        options.minimum_clearance = 0.0005;
        BoundedForceFunction bounded;
        if (force_strategy != "fixed") {
            bounded = [&](const Vector& q, const Vector& v, double t, const Vector&, double target) {
                return system.bounded_forces(q, v, t, force_strategy == "bounded-exact" ? 0 : target,
                                             force_strategy == "bounded-cache");
            };
        }
        Hht stepper(scene.mass, [&](const Vector& q,const Vector& v,double t){return system.forces(q,v,t);}, options,
            [&](const Vector& a,const Vector& b){return guard.check(a,b);}, {}, std::move(bounded));
        State state = stepper.initialize(scene.q,scene.v);
        HhtStats stats;
        const auto start = Clock::now();
        const auto record = [&] {
            const double peak = force_strategy == "fixed" ? system.peak_force() : state.history.at(0);
            trajectory.push_back({state.time,state.q,state.v,peak});
        };
        if (repetition == 0) record();
        for (std::size_t i = 0; i < steps; ++i) {
            const double target = h*static_cast<double>(i+1);
            stepper.step(state,target-state.time,stats);
            if (repetition == 0) record();
        }
        const double elapsed = microseconds(start,Clock::now());
        const Statistics& q = system.statistics();
        summary << scene.name << ',' << name(backend) << ',' << facets << ',' << warm << ',' << cache_guard << ',' << repetition
                << ',' << duration << ',' << h << ',' << steps << ',' << prep << ',' << elapsed << ','
                << scene.mass.size() << ',' << triangles << ',' << stats.accepted_steps << ',' << stats.subdivisions
                << ',' << stats.evaluations << ',' << stats.newton_iterations << ',' << stats.rejected_trials
                << ',' << stats.maximum_residual << ',' << stats.minimum_clearance << ',' << q.queries << ','
                << q.analytic << ',' << q.cad_fallback << ',' << q.mesh.nodes << ',' << q.mesh.triangles
                << ',' << guard.endpoint_evaluations() << ',' << force_strategy << ',' << stats.force_refinements
                << ',' << system.bounded_pairs() << ',' << system.approximated_pairs() << ','
                << system.excluded_pairs() << '\n';
        summary.flush();
        std::cout << scene.name << ' ' << name(backend) << " pass " << repetition+1 << ' ' << elapsed/1e6 << " s\n";
    }
    auto out = output(prefix.string()+".csv");
    out << "time,cad_gap,cad_peak_force,solver_peak_force,stored_energy,contacts";
    for (std::size_t i = 0; i < scene.mass.size(); ++i) out << ",q" << i;
    for (std::size_t i = 0; i < scene.mass.size(); ++i) out << ",v" << i;
    out << '\n';
    for (const auto& frame : trajectory) {
        const auto d = diagnostics(scene, {frame.t,frame.q,frame.v,{},{}});
        out << frame.t << ',' << d.cad_gap << ',' << d.cad_peak_force << ',' << frame.peak_force
            << ',' << d.stored_energy << ',' << d.contacts;
        for (double value : frame.q) out << ',' << value;
        for (double value : frame.v) out << ',' << value;
        out << '\n';
    }
}

void cam_queries(const fs::path& models, const fs::path& file) {
    auto out = output(file);
    out << "backend,facets,phase,reference,gap,latency_us,nodes,triangles,status\n";
    constexpr double tau = 2 * std::numbers::pi;
    constexpr double height = 0.143;
    const Pose plate{{0,height,0},1,0};

    const auto run = [&](Backend backend,unsigned facets,unsigned count) {
        const auto a = load_geometry(models/"follower.step",facets);
        const auto b = load_geometry(models/"cam.step",facets);
        TriangleHint previous;
        for (unsigned i = 0; i < count; ++i) {
            const double angle = .173 + tau*i/count;
            const Pose cam{{},std::cos(angle),std::sin(angle)};
            const double exact_gap = height-.105-.024*std::sin(angle);
            Statistics stats;
            const auto start = Clock::now();
            std::optional<PointPair> result;
            try {
                result = closest(a,plate,b,cam,backend,.12,&previous,stats);
            } catch (const std::runtime_error&) {
                if (backend != Backend::general_cad) throw;
            }
            const double us = microseconds(start,Clock::now());
            out << name(backend) << ',' << facets << ',' << angle << ',' << exact_gap << ',';
            if (result) out << result->distance;
            out << ',' << us << ',' << stats.mesh.nodes << ',' << stats.mesh.triangles << ','
                << (result ? "ok" : "unresolved") << '\n';
        }
        out.flush();
    };
    run(Backend::cad,24,360);
    for (unsigned n : {12,24,48,96}) run(Backend::mesh,n,360);
    run(Backend::general_cad,24,12);
}

void trimmed_queries(const fs::path& models, const fs::path& file) {
    const auto bore = read_step_shape(models/"bore.step");
    const auto pin = read_step_shape(models/"pin.step");
    const PreparedModel a{adapt_occt(bore)}, b{adapt_occt(pin)};
    auto out = output(file);
    out << "backend,deflection,triangles,phase,reference,gap,lower,latency_us,nodes,primitive_tests,status\n";
    for (double deflection : {0.0,0.04,0.01,0.0025,0.000625}) {
        std::optional<TriangleBvh> mesh_a,mesh_b;
        if (deflection > 0) {
            mesh_a.emplace(nc::bench::step_triangles(bore,deflection),2);
            mesh_b.emplace(nc::bench::step_triangles(pin,deflection),2);
        }
        const auto count = deflection == 0 ? 0 : mesh_a->triangles().size()+mesh_b->triangles().size();
        TriangleHint hint;
        for (unsigned i=0;i<=24;++i) {
            const double t = .33*i/24;
            const double d = std::hypot(.07,.4-1.2*t);
            const Vec translation{.68,0,1.2*t};
            double result = infinity, lower = 0;
            bool resolved = false;
            std::size_t visits=0, tests=0;
            const auto start = Clock::now();
            if (deflection == 0) {
                Options options;
                options.max_nodes=150'000;
                options.distance_tolerance=0.001;
                Motion state;
                state.translation=translation;
                const auto hit=nc::distance(a,b,state,0,options);
                visits=hit.statistics.visited;
                result=hit.upper;
                lower=hit.lower;
                resolved=std::isfinite(result) && result-lower<=options.distance_tolerance;
            } else {
                TriangleStatistics stats;
                const auto pair=mesh_a->closest_within(*mesh_b,Pose{translation},infinity,&hint,stats);
                visits=stats.nodes;
                tests=stats.triangles;
                if (pair) { resolved=true; result=lower=pair->distance; }
            }
            const double us=microseconds(start,Clock::now());
            out << (deflection==0 ? "brep" : "mesh") << ',' << deflection << ',' << count << ','
                << t << ',' << d << ',';
            if (std::isfinite(result)) out << result;
            out << ',' << lower << ',' << us << ',' << visits << ',' << tests << ','
                << (resolved?"ok":"budget") << '\n';
        }
        out.flush();
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string_view action = argc > 1 ? argv[1] : "";
        if (action == "models" && argc == 3) make_step_models(argv[2]);
        else if (action == "oscillator" && argc == 3) oscillator(argv[2]);
        else if (action == "cam-queries" && argc == 4) cam_queries(argv[2],argv[3]);
        else if (action == "trimmed-queries" && argc == 4) trimmed_queries(argv[2],argv[3]);
        else if (action == "run" && (argc == 10 || argc == 11)) {
            const std::string_view variant = argc == 11 ? argv[10] : "";
            const bool bounded = variant == "bounded" || variant == "bounded-cache" || variant == "bounded-exact";
            if (variant != "" && variant != "cold" && variant != "uncached" && !bounded) {
                throw std::invalid_argument("unsupported timing variant");
            }
            simulation(argv[2],parse_backend(argv[3]),facets_arg(argv[4]),std::stod(argv[5]),
                       std::stod(argv[6]),argv[7],argv[8],std::stoi(argv[9]),variant != "cold",variant != "uncached",
                       bounded ? variant : "fixed");
        } else {
            std::cerr << "nc_hht models MODEL_DIR\n"
                         "nc_hht oscillator OUTPUT.csv\n"
                         "nc_hht cam-queries MODEL_DIR OUTPUT.csv\n"
                         "nc_hht trimmed-queries BORE_STEP_DIR OUTPUT.csv\n"
                         "nc_hht run SCENE BACKEND FACETS DT DURATION MODEL_DIR PREFIX REPEATS "
                         "[cold|uncached|bounded-exact|bounded|bounded-cache]\n";
            return 2;
        }
    } catch (const Standard_Failure& e) {
        std::cerr << e.GetMessageString() << '\n';
        return 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
