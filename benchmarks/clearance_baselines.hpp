#pragma once

#include "nurbscontact/query.hpp"
#include "nurbscontact/triangle_bvh.hpp"

namespace nc::bench {

struct ClearanceTiming {
    SweepStatus status{SweepStatus::unresolved};
    double lower{};
    std::optional<double> upper;
    std::size_t calls{};
    std::size_t nodes{};
    std::size_t primitives{};
};

// A static distance lower bound d at t excludes all times strictly closer
// than (d-clearance)/speed to t. Use both ends and then bisect only what
// remains. A distance upper bound can establish an event even if the static
// search did not close its own distance interval.
template<class Distance>
ClearanceTiming distance_advancement(Distance sample, double speed, double clearance,
                                    double time_tolerance, std::size_t max_calls,
                                    std::optional<double> candidate = std::nullopt) {
    ClearanceTiming result;
    double confirmed = infinity;
    double terminal = infinity;
    const auto observe = [&](double t, double width) {
        ++result.calls;
        const DistanceResult d = sample(t, width);
        result.nodes += d.statistics.visited;
        if (d.upper <= clearance) confirmed = std::min(confirmed, t);
        if (!(d.lower > clearance)) return Interval{t};
        if (speed == 0) return Interval{-infinity, infinity};
        const double radius = down(((Interval{d.lower} - clearance) / speed).lo);
        if (!(radius > 0)) return Interval{t};
        return Interval{(Interval{t} - radius).hi, (Interval{t} + radius).lo};
    };

    const Interval start = observe(0, 1);
    if (start.hi >= 1 || confirmed == 0) {
        result.status = confirmed == 0 ? SweepStatus::hit : SweepStatus::clear;
        result.lower = confirmed == 0 ? 0 : 1;
        if (confirmed == 0) result.upper = 0;
        return result;
    }
    const Interval end = observe(1, 1 - start.hi);
    std::vector<Interval> pending{{start.hi, end.lo}};
    if (candidate && start.hi < *candidate && *candidate < end.lo) {
        const Interval gap = observe(*candidate, time_tolerance);
        if (gap.hi > gap.lo) {
            pending = {{gap.hi, end.lo}, {start.hi, gap.lo}};
        }
    }

    while (!pending.empty() && result.calls < max_calls) {
        const Interval next = pending.back();
        if (next.lo >= confirmed || up(confirmed - std::min(next.lo, terminal)) <= time_tolerance) break;
        pending.pop_back();
        const Interval span{next.lo, std::min(next.hi, confirmed)};
        if (span.lo > span.hi) continue;
        const double middle = span.center();
        const Interval gap = observe(middle, span.width());
        if (gap.lo <= span.lo && gap.hi >= span.hi && gap.lo < gap.hi) continue;
        if (span.width() <= time_tolerance / 4 || middle == span.lo || middle == span.hi) {
            terminal = std::min(terminal, span.lo);
            continue;
        }
        if (gap.hi < span.hi && gap.hi < confirmed) pending.push_back({gap.hi, span.hi});
        if (span.lo < gap.lo) pending.push_back({span.lo, gap.lo});
    }
    result.lower = std::min({1.0, confirmed, terminal, pending.empty() ? infinity : pending.back().lo});
    if (std::isfinite(confirmed)) {
        result.status = SweepStatus::hit;
        result.upper = confirmed;
    } else if (pending.empty() && !std::isfinite(terminal)) {
        result.status = SweepStatus::clear;
    }
    return result;
}

inline ClearanceTiming mesh_ccd(const dynamics::TriangleBvh& a, const dynamics::TriangleBvh& b,
                                const Motion& motion, double clearance, double time_tolerance) {
    ClearanceTiming result;
    dynamics::TriangleStatistics counts;
    dynamics::TriangleHint hint;
    const auto translation = [&](double t) {
        return dynamics::add(motion.translation, dynamics::scale(motion.velocity, t));
    };
    const auto test = [&](double lo, double hi) {
        ++result.calls;
        return a.swept_within(b, translation(lo), translation(hi), clearance, hint, counts);
    };
    if (!test(0, 1)) {
        result.status = SweepStatus::clear;
        result.lower = 1;
    } else {
        Interval bracket{0, 1};
        while (up(bracket.width()) > time_tolerance) {
            const double middle = bracket.center();
            if (test(bracket.lo, middle)) bracket.hi = middle;
            else bracket.lo = middle;
        }
        result.status = SweepStatus::hit;
        result.lower = bracket.lo;
        result.upper = bracket.hi;
    }
    result.nodes = counts.nodes;
    result.primitives = counts.triangles;
    return result;
}

}  // namespace nc::bench
