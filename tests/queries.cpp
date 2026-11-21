#include "nurbscontact/query.hpp"

#include "../src/axial.hpp"

#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string_view>

using namespace nc;

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

TrimCurve line(Vec a, Vec b, std::size_t id) {
    return {BezierCurve{{homogeneous(exact(a)), homogeneous(exact(b))}},
            {FeatureKind::edge, id}, {FeatureKind::vertex, id}, {FeatureKind::vertex, id + 1}};
}

std::vector<TrimCurve> loop(Interval u, Interval v) {
    return {line({u.lo, v.lo, 0}, {u.hi, v.lo, 0}, 1),
            line({u.hi, v.lo, 0}, {u.hi, v.hi, 0}, 2),
            line({u.hi, v.hi, 0}, {u.lo, v.hi, 0}, 3),
            line({u.lo, v.hi, 0}, {u.lo, v.lo, 0}, 4)};
}

PreparedModel plane(Rect rect, Vec origin = {}, TrimDomain trim = {}) {
    Model model;
    model.trims.push_back(std::move(trim));
    model.patches.push_back({Plane{exact(origin), exact({1, 0, 0}), exact({0, 1, 0})}, rect, 1, 0});
    return PreparedModel{std::move(model)};
}

void test_interval() {
    for (int i = -200; i <= 200; ++i) {
        const double x = i * 0.13;
        const auto sc = sin_cos(x);
        require(sc.sin.contains(std::sin(x)), "sine not enclosed");
        require(sc.cos.contains(std::cos(x)), "cosine not enclosed");
    }
    const auto sc = sin_cos({-0.25, 0.25});
    require(sc.sin.contains(std::sin(-0.25)) && sc.sin.contains(std::sin(0.25)),
            "trigonometric interval endpoints not enclosed");
}

void test_rational_patch() {
    // A quarter circle swept in z, with the native Bezier parameter in u.
    const double w = std::sqrt(0.5);
    BezierSurface patch{3, 2, {
        homogeneous(exact({1, 0, 0})), homogeneous(exact({1, 0, 1})),
        homogeneous(exact({1, 1, 0}), w), homogeneous(exact({1, 1, 1}), w),
        homogeneous(exact({0, 1, 0})), homogeneous(exact({0, 1, 1})),
    }, {{0, 1}, {0, 1}}};
    const Rect clipped{{0.25, 0.625}, {0.375, 0.75}};
    const auto box = bounding_box(patch.support(clipped));
    for (double u : {0.25, 0.4, 0.5, 0.625}) {
        const auto value = patch.eval({u, 0.5});
        for (std::size_t i = 0; i < 3; ++i) {
            require(box[i].contains(value[i].center()), "restricted rational hull misses sample");
        }
    }
    const BezierCurve linear{{homogeneous(exact({0, 0, 0})), homogeneous(exact({1, -1, 0}))}};
    const auto extension = linear.reparameterize(-0.125, 1.5);
    require(extension.eval(0)[0].contains(-0.125) && extension.eval(1)[1].contains(-1.5),
            "Bezier continuation changed the endpoint polynomial");
}

void test_deformed_rational_patch() {
    const Rect chart{{0, 1}, {0, 1}};
    const Rect trimmed{{0.25, 0.75}, {0.25, 0.75}};
    BezierSurface surface{3, 2, {}, chart};
    SurfaceMode mode;
    for (const double u : {0., 0.5, 1.}) {
        for (const double v : {0., 1.}) {
            surface.poles.push_back(homogeneous(exact({u, v, 0}), u == 1. ? 2. : 1.));
            mode.displacements.push_back(exact({0, 0, u == 1. ? 1. : 0.}));
        }
    }
    const std::array modes{mode};
    const std::array amplitude_window{Interval{0.01, 0.03}};
    const auto swept = surface.deformed(modes, amplitude_window);
    const auto box = bounding_box(swept.support(trimmed));
    const auto reference = [](double u, double amplitude) {
        return amplitude * 2 * u * u / (1 + u * u);
    };
    for (const double u : {0.25, 0.5, 0.75}) {
        for (const double amplitude : {0.01, 0.02, 0.03}) {
            const std::array point_amplitude{Interval{amplitude}};
            const auto deformed = surface.deformed(modes, point_amplitude);
            const double z = reference(u, amplitude);
            require(deformed.eval({u, 0.5})[2].contains(z), "wrong rational modal deformation");
            require(box[2].contains(z), "restricted hull misses modal deformation");
        }
    }

    const std::array amplitude{Interval{0.02}};
    Model model;
    model.trims.push_back(TrimDomain{{loop(trimmed.u, trimmed.v)}});
    model.patches.push_back({surface.deformed(modes, amplitude), chart, 1, 0});
    const auto stop = plane(chart, {0, 0, 0.06});
    Options options;
    options.distance_tolerance = 0.001;
    const auto gap = distance(PreparedModel{std::move(model)}, stop, {}, 0, options);
    const double exact_gap = 0.06 - reference(trimmed.u.hi, amplitude[0].lo);
    require(gap.lower <= exact_gap && exact_gap <= gap.upper &&
            gap.upper - gap.lower <= options.distance_tolerance && gap.witness.has_value(),
            "deformed trim-edge gap not certified");
}

void test_trim_distance() {
    TrimDomain trim{{loop({-2, 2}, {-2, 2}), loop({-1, 1}, {-1, 1})}};
    require(trim.classify(Vec{0, 0, 0}) == TrimClass::outside, "hole membership");
    require(trim.classify(Vec{1.5, 0, 0}) == TrimClass::inside, "plate membership");
    require(trim.classify(Rect{{-0.25, 0.25}, {-0.25, 0.25}}) == TrimClass::outside,
            "interior hole rectangle");
    require(trim.classify(Rect{{0.75, 1.25}, {-0.1, 0.1}}) == TrimClass::mixed,
            "trim crossing rectangle");
    const auto plate = plane({{-2, 2}, {-2, 2}}, {}, trim);
    const auto probe = plane({{-0.25, 0.25}, {-0.25, 0.25}}, {0, 0, 0.5});
    Options options;
    options.distance_tolerance = 0.015;
    const auto result = distance(plate, probe, {}, 0, options);
    const double reference = std::hypot(0.75, 0.5);
    std::cout << "Trimmed plate: [" << result.lower << ", " << result.upper
              << "] after " << result.statistics.visited << " pairs\n";
    require(result.lower <= reference && reference <= result.upper, "trim distance bounds");
    require(result.upper - result.lower <= options.distance_tolerance, "trim distance tolerance");
    require(result.witness.has_value(), "trim distance witness");
    require(result.statistics.trim_rejected > 0, "no trimmed cells rejected");
}

void test_translation() {
    const auto a = plane({{-0.1, 0.1}, {-0.1, 0.1}});
    const auto b = plane({{-0.1, 0.1}, {-0.1, 0.1}});
    const Motion path{{0, 0, 1}, {0, 0, -2}, {0, 0, 1}, 0, 0};
    Options options;
    for (bool directional : {false, true}) {
        options.directional_time = directional;
        const auto hit = sweep(a, b, path, {0, 1}, 0.125, options);
        std::cout << "Translation " << directional << ": [" << hit.safe_prefix << ", "
                  << hit.witness_time.value_or(-1) << "] after " << hit.statistics.visited << " pairs\n";
        const double reference = 0.4375;
        require(hit.status == SweepStatus::hit && hit.witness_time.has_value(), "interior TOI not confirmed");
        require(hit.safe_prefix <= reference && reference <= *hit.witness_time, "invalid TOI bracket");
        require(*hit.witness_time - hit.safe_prefix <= options.time_tolerance, "TOI tolerance");
    }
    const Motion negative_time{{0, 0, 1}, {0, 0, 2}, {0, 0, 1}, 0, 0};
    const auto earlier = sweep(a, b, negative_time, {-1, 0}, 0.125, options);
    require(earlier.status == SweepStatus::hit && earlier.safe_prefix <= -0.5625 &&
                earlier.witness_time.value_or(-1) >= -0.5625, "negative-time clipping missed an event");
    require(*earlier.witness_time - earlier.safe_prefix <= options.time_tolerance, "negative TOI tolerance");
    options.max_nodes = 1;
    const auto limited = sweep(a, b, path, {0, 1}, 0, options);
    require(limited.status != SweepStatus::clear && limited.safe_prefix <= 0.5,
            "budget exhaustion incorrectly declared clear");
}

void test_rotation() {
    const auto a = plane({{-0.002, 0.002}, {-0.002, 0.002}}, {0.9, 0.9, 0});
    const auto b = plane({{-0.002, 0.002}, {-0.002, 0.002}}, {1, 0, 0});
    const Motion path{{}, {}, {0, 0, 1}, 0, std::numbers::pi / 2};
    const auto clear = sweep(a, b, path, {0, 1}, 0.26);
    require(clear.status == SweepStatus::clear, "rotational near miss not excluded");
    const auto hit = sweep(a, b, path, {0, 1}, 0.3);
    require(hit.status == SweepStatus::hit && hit.witness_time.value_or(1) < 0.5,
            "rotational interior approach not confirmed");
}

void test_tilted_envelope() {
    using detail::AxialEnvelope;
    const IVec tilted = exact({0.6, 0, 0.8});
    const AxialEnvelope a{exact({0, 0, 0}), exact({0, 0, 1}), {0.85, 1.1}, {-0.04, 0.09}};
    const AxialEnvelope b{exact({0.7, 0.13, 0.55}), tilted / norm(tilted), {0.12, 0.3}, {-0.12, 0.2}};
    const IVec velocity{0.1, -0.02, -0.8};
    constexpr double clearance = 0.30;
    const auto window = detail::axial_window(a, b, velocity, {-1, 1}, clearance);
    int feasible = 0;
    for (int k = -12; k <= 12; ++k) {
        const double t = k / 12.0;
        for (int i = 0; i < 12; ++i) {
            const double u = i * std::numbers::pi / 6;
            const IVec on_a = exact({std::cos(u), std::sin(u), 0.02});
            for (int j = 0; j < 12; ++j) {
                const double v = j * std::numbers::pi / 6;
                const IVec radial = exact({0.8, 0, -0.6}) * (0.22 * std::cos(v)) +
                                    exact({0, 1, 0}) * (0.22 * std::sin(v));
                const IVec on_b = b.origin + velocity * t + radial + b.axis * 0.1;
                if (norm(on_b - on_a).hi + 1e-9 >= clearance) continue;
                ++feasible;
                require(window && window->contains(t), "tilted envelope excluded a feasible point pair");
            }
        }
    }
    require(feasible > 0 && window->lo > -1, "tilted envelope test does not exercise clipping");
}

}  // namespace

int main() {
    try {
        test_interval();
        test_rational_patch();
        test_deformed_rational_patch();
        test_trim_distance();
        test_translation();
        test_rotation();
        test_tilted_envelope();
        std::cout << "Native geometric queries passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
