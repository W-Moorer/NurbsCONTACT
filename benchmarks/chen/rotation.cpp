#include "nurbscontact/query.hpp"

#include "config.h"
#include "recBezier.h"
#include "solverTD_margin.h"

#include <chrono>
#include <iomanip>

namespace {
using namespace nc;
using Clock = std::chrono::steady_clock;
using Patch = RecQuadBezier;
using Solver = SolverTD<Patch, Patch, RecParamBound, RecParamBound>;

constexpr double clearance = 0.03125;
constexpr Options settings{2e-4, 5e-4, 200'000};
constexpr double param_target = 5e-4;
constexpr double missing = std::numeric_limits<double>::quiet_NaN();
constexpr Rect chart{{0, 1}, {0, 1}};

Vec from_eigen(const Vector3d& p) { return {p.x(), p.y(), p.z()}; }

BezierSurface native_patch(const Patch& p) {
    BezierSurface surface{3, 3, {}, chart};
    for (const auto& pole : p.ctrlp) surface.poles.push_back(homogeneous(exact(from_eigen(pole))));
    return surface;
}

struct Fixture {
    const char* name;
    double height, angle, curvature;
};

Patch rectangle(double xmin, double xmax, double ymin, double ymax, double curvature) {
    Patch result;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result.ctrlp[3*i+j] = {std::lerp(xmin, xmax, .5*i), std::lerp(ymin, ymax, .5*j),
                                  curvature*(j == 1 ? -.25 : .25)};
        }
    }
    return result;
}

IVec transform(IVec point, const Fixture& scene, double t) {
    const auto [s, c] = sin_cos(Interval{scene.angle}*t);
    return {point[0]*c + point[2]*s, point[1], scene.height - point[0]*s + point[2]*c};
}

double acceleration_bound(const Patch& patch, const Fixture& scene) {
    double bound = 0;
    for (const auto& pole : patch.ctrlp) {
        const Interval radius = sqrt(square(pole.x()) + square(pole.z()));
        bound = std::max(bound, (radius*square(scene.angle)).hi);
    }
    return bound;
}

struct Chord {
    Patch start, velocity;
    double padding;
};

Chord chord(const Patch& patch, const Fixture& scene, Interval time, double interpolation_error) {
    Chord result;
    double rounding = 0;
    for (std::size_t i = 0; i < patch.ctrlp.size(); ++i) {
        const IVec point = exact(from_eigen(patch.ctrlp[i]));
        const IVec first = transform(point, scene, time.lo), last = transform(point, scene, time.hi);
        const Vec a = midpoint(first), b = midpoint(last);
        result.start.ctrlp[i] = {a[0], a[1], a[2]};
        result.velocity.ctrlp[i] = (Vector3d{b[0], b[1], b[2]} - result.start.ctrlp[i])/time.width();
        const IVec origin = exact(a), velocity = exact(from_eigen(result.velocity.ctrlp[i]));
        rounding = std::max({rounding, norm(first-origin).hi,
                             norm(last-(origin+velocity*time.width())).hi});
    }
    result.padding = (Interval{interpolation_error} + rounding).hi;
    return result;
}

bool verify_exclusion(const Patch& fixed, const Chord& linear, double duration, std::size_t& work) {
    std::vector<Vector3d> axes;
    setAxes<Patch, Patch>(fixed.ctrlp, linear.start.ctrlp, axes, BoundingBoxType::OBB);
    std::vector<std::pair<IVec, Interval>> projections;
    const Interval margin = Interval{clearance} + linear.padding;
    for (const auto& candidate : axes) {
        if (candidate.squaredNorm() == 0) continue;
        const IVec n = exact(from_eigen(candidate));
        Interval a{infinity, -infinity};
        for (const auto& pole : fixed.ctrlp) a = hull(a, dot(n, exact(from_eigen(pole))));
        const double threshold = (margin*norm(n)).hi;
        projections.emplace_back(n, a+Interval{-threshold, threshold});
    }
    const BezierSurface first = native_patch(linear.start);
    BezierSurface last{3, 3, {}, chart};
    for (std::size_t i = 0; i < linear.start.ctrlp.size(); ++i) {
        const IVec x = exact(from_eigen(linear.start.ctrlp[i]));
        const IVec v = exact(from_eigen(linear.velocity.ctrlp[i]));
        last.poles.push_back(homogeneous(x+v*duration));
    }
    const auto limit = std::min(settings.max_nodes, work+128);
    std::vector<Rect> remaining{chart};
    while (!remaining.empty() && work < limit) {
        const auto cell = remaining.back();
        remaining.pop_back();
        ++work;
        const auto a = first.support(cell), b = last.support(cell);
        const bool separated = std::any_of(projections.begin(), projections.end(), [&](const auto& projection) {
            const auto& [n, occupied] = projection;
            Interval moved{infinity, -infinity};
            for (std::size_t i = 0; i < a.size(); ++i) moved = hull(moved, hull(dot(n, a[i]), dot(n, b[i])));
            return disjoint(occupied, moved);
        });
        if (!separated) {
            const auto [left, right] = nc::split(cell, cell.u.width() >= cell.v.width());
            remaining.push_back(right);
            remaining.push_back(left);
        }
    }
    return remaining.empty();
}

double verify_pair(const BezierSurface& moving, const Fixture& scene, double time, double u, double v) {
    const IVec b = transform(moving.eval({u, v}), scene, time);
    // The orthogonal projection is on the original fixed rectangle if its x
    // and y enclosures are inside. The pair then differs only in z.
    if (b[0].lo < -1.5 || b[0].hi > 1.5 || b[1].lo < -.25 || b[1].hi > .25) return infinity;
    return magnitude(b[2]);
}

struct Result {
    SweepStatus status{SweepStatus::unresolved};
    double lower{}, upper{missing}, residual{missing}, candidate{missing}, error{};
    std::size_t nodes{}, segments{}, rechecked_negatives{};
};

Result segmented(const Patch& fixed, const Patch& moving, const BezierSurface& moving_surface,
                 const Fixture& scene) {
    const double acceleration = acceleration_bound(moving, scene);
    Patch stationary;
    for (auto& pole : stationary.ctrlp) pole.setZero();
    std::vector<Interval> work{{0, 1}};
    double first_open = infinity;
    Result out;
    while (!work.empty() && out.nodes < settings.max_nodes) {
        const Interval item = work.back();
        work.pop_back();
        const auto split = [&] {
            work.push_back({item.center(), item.hi});
            work.push_back({item.lo, item.center()});
        };
        const double error = (Interval{acceleration}*square(item.width())/8).hi;
        if (error > settings.distance_tolerance) {
            split();
            continue;
        }
        const Chord linear = chord(moving, scene, item, error);
        out.error = std::max(out.error, linear.padding);
        ++out.segments;
        ++out.nodes;
        const double margin = (Interval{clearance} + linear.padding).hi;
        const Array2d local_time{0, item.width()};
        Array2d candidate_window;
        const bool maybe = Solver::primitiveCheck(fixed, stationary, linear.start, linear.velocity,
            {}, {}, candidate_window, BoundingBoxType::OBB, local_time, margin);
        if (!maybe) {
            ++out.rechecked_negatives;
            if (verify_exclusion(fixed, linear, item.width(), out.nodes)) continue;
        }
        if (item.width() > settings.time_tolerance/4) {
            split();
            continue;
        }
        Array2d uv_a, uv_b;
        const double proposed = maybe ? Solver::solveCCD(fixed, stationary, linear.start, linear.velocity,
            uv_a, uv_b, BoundingBoxType::OBB, param_target, item.width(), margin,
            &out.nodes, settings.max_nodes) : -1;
        if (maybe && proposed == -1) {
            ++out.rechecked_negatives;
            if (verify_exclusion(fixed, linear, item.width(), out.nodes)) continue;
        }
        if (proposed >= 0 && std::isnan(out.candidate)) out.candidate = item.lo + proposed;
        first_open = std::min(first_open, item.lo);
        for (double time : {item.lo, item.center(), item.hi}) {
            double residual = proposed >= 0 ?
                verify_pair(moving_surface, scene, time, uv_b[0], uv_b[1]) : infinity;
            for (double u : {0., .5, 1.}) {
                for (double v : {0., .5, 1.}) {
                    residual = std::min(residual, verify_pair(moving_surface, scene, time, u, v));
                }
            }
            if (residual <= clearance) {
                out.status = SweepStatus::hit;
                out.lower = first_open;
                out.upper = time;
                out.residual = residual;
                return out;
            }
        }
    }
    out.lower = std::min({1., first_open, work.empty() ? infinity : work.back().lo});
    if (out.lower == 1) out.status = SweepStatus::clear;
    return out;
}

void print(const Fixture& scene, const char* method, int repeat, const Result& result, double us) {
    const char* status = result.status == SweepStatus::hit ? "hit" :
        result.status == SweepStatus::clear ? "clear" : "unresolved";
    std::cout << scene.name << ',' << method << ',' << repeat << ',' << scene.height << ','
              << scene.angle << ',' << scene.curvature << ',' << status << ',' << result.lower << ','
              << result.upper << ',' << result.residual << ',' << result.candidate << ','
              << result.error << ',' << result.nodes << ',' << result.segments << ','
              << result.rechecked_negatives << ',' << us << '\n';
}
}

int main(int argc, char** argv) {
    const int repeats = argc == 2 ? std::stoi(argv[1]) : 9;
    if (argc > 2 || repeats < 1) return 2;
    const Patch fixed = rectangle(-1.5, 1.5, -.25, .25, 0);
    const PreparedModel a({{{native_patch(fixed), chart, 0, 0}}, {{}}, 0});
    std::cout << std::setprecision(17)
              << "scene,method,repeat,height,angle,curvature,status,safe_prefix,witness_time,"
              << "point_residual,chen_candidate,max_chord_padding,nodes,segments,obb_negatives_rechecked,query_us\n";
    for (const Fixture& scene : {Fixture{"planar-tilt", .7, 1., 0.},
                                 Fixture{"curved-tilt", .55, 1., .25},
                                 Fixture{"interior-entry", .95, 3.125, 0.},
                                 Fixture{"rotating-near-miss", 1.03225, 3.125, 0.}}) {
        const Patch moving = rectangle(.25, 1., -.125, .125, scene.curvature);
        const BezierSurface surface = native_patch(moving);
        const PreparedModel b({{{surface, chart, 0, 0}}, {{}}, 0});
        const Motion motion{{0, 0, scene.height}, {}, {0, 1, 0}, 0, scene.angle};
        const auto run = [&](bool exact_path, int repeat) {
            const auto start = Clock::now();
            Result output;
            if (exact_path) {
                const auto r = sweep(a, b, motion, {0, 1}, clearance, settings);
                output.status = r.status;
                output.lower = r.safe_prefix;
                output.upper = r.witness_time.value_or(missing);
                output.residual = r.witness ? norm(r.witness->b.position-r.witness->a.position).hi : missing;
                output.nodes = r.statistics.visited;
            } else {
                output = segmented(fixed, moving, surface, scene);
            }
            const double us = std::chrono::duration<double, std::micro>(Clock::now()-start).count();
            if (repeat >= 0) print(scene, exact_path ? "native" : "chen-obb-chord", repeat, output, us);
        };
        run(true, -1);
        run(false, -1);
        for (int i = 0; i < repeats; ++i) {
            run(i%2 == 0, i);
            run(i%2 != 0, i);
        }
        std::cout.flush();
    }
}
