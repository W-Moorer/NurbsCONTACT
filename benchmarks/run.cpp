#include "nurbscontact/occt.hpp"
#include "nurbscontact/query.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <string_view>

namespace {

using namespace nc;
namespace fs = std::filesystem;

struct Scenario {
    std::string_view name;
    std::string_view a;
    std::string_view b;
    Motion motion;
    double clearance;
    std::optional<double> reference;
};

std::vector<Scenario> scenarios() {
    constexpr double omega = std::numbers::pi / 2;
    const Motion rotor{{}, {}, {0, 0, 1}, 0, omega};
    const double r = 0.9 * std::sqrt(2.0);
    const double angle = std::numbers::pi / 4 - std::acos((1 + r * r - 0.304 * 0.304) / (2 * r));
    return {
        {"box-entry", "block-a.step", "block-b.step", {{}, {0, 0, -1}}, 0.1, 0.4},
        {"bore-passage", "bore.step", "pin.step", {{}, {0, 0, 1.2}}, 0.09, std::nullopt},
        {"bore-rim-entry", "bore.step", "pin.step", {{0.68, 0, 0}, {0, 0, 1.2}}, 0.09,
            (0.4 - std::sqrt(0.09 * 0.09 - 0.07 * 0.07)) / 1.2},
        {"rotation-clear", "guard.step", "rotor.step", rotor, 0.26, std::nullopt},
        {"rotation-entry", "guard.step", "rotor.step", rotor, 0.30, angle / omega},
        {"spline-clear", "freeform.step", "block-b.step", {{}, {0, 0, -0.20}, {0, 0, 1}, 0, omega},
            0.15, std::nullopt},
    };
}

const char* status(SweepStatus result) {
    switch (result) {
    case SweepStatus::clear: return "clear";
    case SweepStatus::hit: return "hit";
    case SweepStatus::unresolved: return "unresolved";
    }
    return "unknown";
}

struct Variant {
    std::string_view name;
    Options options;
};

std::vector<Variant> variants() {
    Options base;
    base.distance_tolerance = 2e-4;
    base.time_tolerance = 5e-4;
    base.max_nodes = 200'000;
    auto axial = base;
    base.axial_time = false;
    auto swept = base;
    swept.directional_time = false;
    auto chord = swept;
    chord.chord_box = true;
    auto no_inheritance = base;
    no_inheritance.inherit_time = false;
    auto late_trim = base;
    late_trim.eager_trim = false;
    return {{"axial", axial}, {"directional", base}, {"swept-box", swept}, {"chord-box", chord},
            {"no-inheritance", no_inheritance}, {"late-trim", late_trim}};
}

bool verifies(const Scenario& scenario, const SweepResult& result) {
    constexpr double reference_uncertainty = 1e-10;
    if (!scenario.reference) return result.status == SweepStatus::clear && result.safe_prefix == 1;
    return result.status == SweepStatus::hit && result.witness_time && result.witness &&
           result.safe_prefix <= *scenario.reference + reference_uncertainty &&
           *result.witness_time >= *scenario.reference - reference_uncertainty &&
           *result.witness_time - result.safe_prefix <= 5e-4;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: nc_bench STEP_FIXTURE_DIR OUTPUT_CSV REPEATS\n";
        return 2;
    }
    try {
        const int repeats = std::stoi(argv[3]);
        if (repeats <= 0) throw std::invalid_argument("positive repetition count required");
        const fs::path folder = argv[1];
        std::ofstream output(argv[2]);
        output.exceptions(std::ios::badbit | std::ios::failbit);
        output << std::setprecision(17) << "scene,variant,repeat,status,reference,safe_prefix,witness_time,"
            "bracket_width,query_us,nodes,spatial_splits,time_splits,trim_rejected,direction_rejected,direction_windows,"
            "axial_rejected,axial_clipped\n";
        const auto configs = variants();
        bool valid = true;
        for (const Scenario& scene : scenarios()) {
            const PreparedModel a{read_step(folder / scene.a)};
            const PreparedModel b{read_step(folder / scene.b)};
            const auto perform = [&](const Variant& variant) {
                return sweep(a, b, scene.motion, {0, 1}, scene.clearance, variant.options);
            };
            for (const auto& variant : configs) valid = verifies(scene, perform(variant)) && valid;
            for (int iteration = 0; iteration < repeats; ++iteration) {
                for (std::size_t k = 0; k < configs.size(); ++k) {
                    const Variant& variant = configs[(k + static_cast<std::size_t>(iteration)) % configs.size()];
                    const auto start = std::chrono::steady_clock::now();
                    const SweepResult result = perform(variant);
                    const auto stop = std::chrono::steady_clock::now();
                    const double micros = std::chrono::duration<double, std::micro>(stop - start).count();
                    valid = verifies(scene, result) && valid;
                    output << scene.name << ',' << variant.name << ',' << iteration << ',' << status(result.status) << ',';
                    if (scene.reference) output << *scene.reference;
                    output << ',' << result.safe_prefix << ',';
                    if (result.witness_time) output << *result.witness_time;
                    output << ',';
                    if (result.witness_time) output << *result.witness_time - result.safe_prefix;
                    output << ',' << micros << ',' << result.statistics.visited << ',' << result.statistics.spatial_splits
                           << ',' << result.statistics.time_splits << ',' << result.statistics.trim_rejected
                           << ',' << result.statistics.direction_rejected << ',' << result.statistics.direction_windows
                           << ',' << result.statistics.axial_rejected << ',' << result.statistics.axial_clipped << '\n';
                }
            }
            output.flush();
            std::cout << scene.name << " finished\n" << std::flush;
        }
        if (!valid) throw std::runtime_error("a result contradicted an analytic reference");
    } catch (const Standard_Failure& e) {
        std::cerr << e.GetMessageString() << '\n';
        return 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
