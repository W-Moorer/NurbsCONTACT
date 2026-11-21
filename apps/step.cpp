#include "nurbscontact/occt.hpp"
#include "nurbscontact/query.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>

namespace {

using namespace nc;

const char* label(FeatureKind kind) {
    switch (kind) {
    case FeatureKind::face: return "face";
    case FeatureKind::edge: return "edge";
    case FeatureKind::vertex: return "vertex";
    }
    return "unknown";
}

const char* label(SweepStatus status) {
    switch (status) {
    case SweepStatus::clear: return "clear";
    case SweepStatus::hit: return "hit";
    case SweepStatus::unresolved: return "unresolved";
    }
    return "unknown";
}

void number(double x) {
    if (std::isfinite(x)) std::cout << x;
    else std::cout << "null";
}

void interval(Interval x) {
    std::cout << '[';
    number(x.lo);
    std::cout << ',';
    number(x.hi);
    std::cout << ']';
}

void sample(const Sample& sample) {
    std::cout << "{\"kind\":\"" << label(sample.feature.kind) << "\",\"id\":" << sample.feature.id << ",\"uv\":[";
    interval(sample.uv[0]);
    std::cout << ',';
    interval(sample.uv[1]);
    std::cout << "],\"xyz\":[";
    interval(sample.position[0]);
    std::cout << ',';
    interval(sample.position[1]);
    std::cout << ',';
    interval(sample.position[2]);
    std::cout << "]}";
}

void details(const Stats& stats, const std::optional<Witness>& witness) {
    std::cout << ",\"visited\":" << stats.visited << ",\"spatial_splits\":" << stats.spatial_splits
              << ",\"time_splits\":" << stats.time_splits << ",\"trim_rejected\":" << stats.trim_rejected
              << ",\"direction_rejected\":" << stats.direction_rejected
              << ",\"direction_windows\":" << stats.direction_windows
              << ",\"axial_rejected\":" << stats.axial_rejected << ",\"axial_clipped\":" << stats.axial_clipped
              << ",\"witness\":";
    if (witness) {
        std::cout << "{\"time\":" << witness->time << ",\"a\":";
        sample(witness->a);
        std::cout << ",\"b\":";
        sample(witness->b);
        std::cout << '}';
    } else {
        std::cout << "null";
    }
    std::cout << "}\n";
}

void usage() {
    std::cerr << "Usage: nc_step distance|sweep A.step B.step [options]\n"
              << "  --translation x y z  --velocity x y z  --axis x y z\n"
              << "  --angle radians  --omega radians-per-time  --time t0 t1\n"
              << "  --clearance length  --distance-tol length  --time-tol time\n"
              << "  --max-nodes count  --swept-box-only  --chord-box  --no-inheritance  --late-trim  --no-axial\n";
}

}  // namespace

int main(int argc, char** argv) {
    using namespace nc;
    if (argc < 4) {
        usage();
        return 2;
    }
    try {
        const std::string_view command{argv[1]};
        if (command != "distance" && command != "sweep") throw std::invalid_argument("expected distance or sweep");
        Motion motion;
        Interval time{0, 1};
        double clearance = 0;
        Options options;
        for (int i = 4; i < argc; ++i) {
            const std::string_view option{argv[i]};
            const auto scalar = [&]() {
                if (++i >= argc) throw std::invalid_argument("missing argument for " + std::string(option));
                char* end = nullptr;
                const double result = std::strtod(argv[i], &end);
                if (end == argv[i] || *end != '\0' || !std::isfinite(result)) {
                    throw std::invalid_argument("invalid value for " + std::string(option));
                }
                return result;
            };
            const auto vector = [&] { return Vec{scalar(), scalar(), scalar()}; };
            if (option == "--translation") motion.translation = vector();
            else if (option == "--velocity") motion.velocity = vector();
            else if (option == "--axis") motion.axis = vector();
            else if (option == "--angle") motion.angle = scalar();
            else if (option == "--omega") motion.angular_velocity = scalar();
            else if (option == "--time") time = {scalar(), scalar()};
            else if (option == "--clearance") clearance = scalar();
            else if (option == "--distance-tol") options.distance_tolerance = scalar();
            else if (option == "--time-tol") options.time_tolerance = scalar();
            else if (option == "--max-nodes") {
                const double value = scalar();
                if (value < 1 || value > 1e8 || value != std::floor(value)) throw std::invalid_argument("invalid node count");
                options.max_nodes = static_cast<std::size_t>(value);
            } else if (option == "--swept-box-only") {
                options.directional_time = false;
                options.axial_time = false;
            }
            else if (option == "--chord-box") {
                options.chord_box = true;
                options.directional_time = false;
                options.axial_time = false;
            }
            else if (option == "--no-inheritance") options.inherit_time = false;
            else if (option == "--late-trim") options.eager_trim = false;
            else if (option == "--no-axial") options.axial_time = false;
            else throw std::invalid_argument("unknown option " + std::string(option));
        }
        const PreparedModel a{read_step(argv[2])};
        const PreparedModel b{read_step(argv[3])};
        std::cout << std::setprecision(17);
        if (command == "distance") {
            const auto result = distance(a, b, motion, time.lo, options);
            std::cout << "{\"lower\":";
            number(result.lower);
            std::cout << ",\"upper\":";
            number(result.upper);
            details(result.statistics, result.witness);
        } else {
            const auto result = sweep(a, b, motion, time, clearance, options);
            std::cout << "{\"status\":\"" << label(result.status) << "\",\"safe_prefix\":" << result.safe_prefix
                      << ",\"witness_time\":";
            number(result.witness_time.value_or(infinity));
            details(result.statistics, result.witness);
        }
    } catch (const Standard_Failure& e) {
        std::cerr << e.GetMessageString() << '\n';
        return 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
