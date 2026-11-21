#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace nc {

// IEEE-754 binary64 under round-to-nearest with contraction disabled.
static_assert(std::numeric_limits<double>::is_iec559);

inline constexpr double infinity = std::numeric_limits<double>::infinity();
inline double down(double x) { return std::nextafter(x, -infinity); }
inline double up(double x) { return std::nextafter(x, infinity); }

struct Interval {
    double lo{};
    double hi{};

    constexpr Interval() = default;
    constexpr Interval(double value) : lo(value), hi(value) {}
    constexpr Interval(double lower, double upper) : lo(lower), hi(upper) {}

    double center() const { return std::midpoint(lo, hi); }
    double width() const { return hi - lo; }
    bool contains(double x) const { return lo <= x && x <= hi; }
    bool is_zero() const { return lo == 0 && hi == 0; }
};

inline Interval operator+(Interval a, Interval b) {
    if (a.is_zero()) return b;
    if (b.is_zero()) return a;
    if (a.lo == a.hi && b.lo == b.hi && a.lo == -b.lo) return 0;
    return {down(a.lo + b.lo), up(a.hi + b.hi)};
}
inline Interval operator-(Interval a) { return {-a.hi, -a.lo}; }
inline Interval operator-(Interval a, Interval b) { return a + (-b); }
inline Interval operator*(Interval a, Interval b) {
    if (a.is_zero() || b.is_zero()) return 0;
    if (a.lo == 1 && a.hi == 1) return b;
    if (b.lo == 1 && b.hi == 1) return a;
    if (a.lo == -1 && a.hi == -1) return -b;
    if (b.lo == -1 && b.hi == -1) return -a;
    const auto [lo, hi] = std::minmax({a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi});
    return {down(lo), up(hi)};
}
inline Interval operator/(Interval a, Interval b) {
    if (b.contains(0)) throw std::domain_error("interval divisor contains zero");
    if (a.is_zero()) return 0;
    if (b.lo == 1 && b.hi == 1) return a;
    if (b.lo == -1 && b.hi == -1) return -a;
    const auto [lo, hi] = std::minmax({a.lo / b.lo, a.lo / b.hi, a.hi / b.lo, a.hi / b.hi});
    return {down(lo), up(hi)};
}
inline Interval hull(Interval a, Interval b) {
    return {std::min(a.lo, b.lo), std::max(a.hi, b.hi)};
}
inline bool disjoint(Interval a, Interval b) { return a.hi < b.lo || b.hi < a.lo; }
inline double magnitude(Interval a) { return std::max(std::abs(a.lo), std::abs(a.hi)); }
inline Interval square(Interval a) {
    const Interval b = a * a;
    return {std::max(0.0, b.lo), b.hi};
}
inline Interval sqrt(Interval a) {
    if (a.hi < 0) throw std::domain_error("negative square root");
    return {a.lo <= 0 ? 0 : down(std::sqrt(a.lo)), a.hi <= 0 ? 0 : up(std::sqrt(a.hi))};
}

struct SinCos { Interval sin; Interval cos; };

inline SinCos bounded_trig_point(double x) {
    // Caller supplies |x| <= 4.25. On |y| <= 1.0625 the noninitial Taylor
    // terms contract by at least 0.565. With binary64 rounding, each term
    // has absolute error < 2^-48 and each initial sum differs by < 2^-43
    // including the alternating-series remainder. The two angle doublings
    // keep the final error inside 2^-38 even at cancellation.
    const double y = x * 0.25;
    const double z = -y * y;
    double sin_term = y;
    double cos_term = 1;
    double s = y;
    double c = 1;
    for (int k = 1; k <= 10; ++k) {
        sin_term = sin_term * z / ((2 * k) * (2 * k + 1));
        cos_term = cos_term * z / ((2 * k - 1) * (2 * k));
        s += sin_term;
        c += cos_term;
    }
    for (int i = 0; i < 2; ++i) {
        const double next = 2 * s * c;
        c = c * c - s * s;
        s = next;
    }
    constexpr double error = 0x1p-38;
    return {Interval{s} + Interval{-error, error}, Interval{c} + Interval{-error, error}};
}

inline SinCos sin_cos(Interval x) {
    if (x.is_zero()) return {0, 1};
    if (!std::isfinite(x.lo) || !std::isfinite(x.hi) || x.width() >= 2) {
        return {{-1, 1}, {-1, 1}};
    }
    // Reduce relative to an explicit enclosure of 2*pi, then apply alternating
    // Taylor bounds. The integer is a performance choice and need not be optimal.
    constexpr Interval pi{0x1.921fb54442d18p+1, 0x1.921fb54442d19p+1};
    constexpr Interval two_pi{0x1.921fb54442d18p+2, 0x1.921fb54442d19p+2};
    constexpr Interval half_pi{0x1.921fb54442d18p+0, 0x1.921fb54442d19p+0};
    const double turns = std::round(x.center() / two_pi.lo);
    if (std::abs(turns) > 1e10) return {{-1, 1}, {-1, 1}};
    const Interval reduced = x - Interval{turns} * two_pi;
    if (magnitude(reduced) > 4.25) return {{-1, 1}, {-1, 1}};
    const auto a = bounded_trig_point(reduced.lo);
    const auto b = reduced.lo == reduced.hi ? a : bounded_trig_point(reduced.hi);
    Interval s = hull(a.sin, b.sin);
    Interval c = hull(a.cos, b.cos);
    if (!disjoint(reduced, half_pi)) s.hi = 1;
    if (!disjoint(reduced, -half_pi)) s.lo = -1;
    if (!disjoint(reduced, pi) || !disjoint(reduced, -pi)) c.lo = -1;
    if (reduced.contains(0)) c.hi = 1;
    return {{std::max(-1.0, s.lo), std::min(1.0, s.hi)},
            {std::max(-1.0, c.lo), std::min(1.0, c.hi)}};
}

using Vec = std::array<double, 3>;
using IVec = std::array<Interval, 3>;

inline IVec exact(Vec x) { return {x[0], x[1], x[2]}; }
inline Vec midpoint(IVec x) { return {x[0].center(), x[1].center(), x[2].center()}; }
inline IVec operator+(IVec a, IVec b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
inline IVec operator-(IVec a, IVec b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
inline IVec operator*(IVec a, Interval b) { return {a[0] * b, a[1] * b, a[2] * b}; }
inline IVec operator/(IVec a, Interval b) { return {a[0] / b, a[1] / b, a[2] / b}; }
inline Interval dot(IVec a, IVec b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline IVec cross(IVec a, IVec b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
inline Interval norm(IVec a) { return sqrt(square(a[0]) + square(a[1]) + square(a[2])); }
inline IVec hull(IVec a, IVec b) { return {hull(a[0], b[0]), hull(a[1], b[1]), hull(a[2], b[2])}; }

}  // namespace nc
