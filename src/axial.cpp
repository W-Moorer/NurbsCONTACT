#include "axial.hpp"

namespace nc::detail {
namespace {

bool finite(Interval x) {
    return std::isfinite(x.lo) && std::isfinite(x.hi) && x.lo <= x.hi;
}

Interval absolute(Interval x) {
    return {x.contains(0) ? 0 : std::min(std::abs(x.lo), std::abs(x.hi)), magnitude(x)};
}

// The eigenvalues of the 2x2 Gram matrix are bounded by its Gershgorin discs.
Interval singular_values(IVec x, IVec y) {
    const Interval xx = square(x[0]) + square(x[1]) + square(x[2]);
    const Interval yy = square(y[0]) + square(y[1]) + square(y[2]);
    const Interval off = absolute(dot(x, y));
    const double lo = (Interval{std::min(xx.lo, yy.lo)} - off).lo;
    const double hi = (Interval{std::max(xx.hi, yy.hi)} + off).hi;
    return sqrt({std::max(0.0, lo), hi});
}

Interval trim_radii(const TrimDomain& trim, Vec pivot) {
    const IVec p = exact(pivot);
    Interval radial{infinity, 0};
    for (const auto& loop : trim.loops) {
        for (const TrimCurve& edge : loop) {
            Interval gap = norm(bounds(edge.geometry) - p);
            if (const auto* arc = std::get_if<Arc>(&edge.geometry)) {
                const Interval radius = singular_values(arc->x, arc->y);
                const double offset = norm(arc->center - p).hi;
                // Both the UV box and the ellipse's singular values enclose
                // the radius. Intersecting tightens bounds at a circular rim.
                gap.lo = std::max(gap.lo, (Interval{radius.lo} - offset).lo);
                gap.hi = std::min(gap.hi, (Interval{radius.hi} + offset).hi);
            }
            if (!finite(gap)) return {0, infinity};
            radial.lo = std::min(radial.lo, gap.lo);
            radial.hi = std::max(radial.hi, gap.hi);
        }
    }
    return radial;
}

bool outside_disk_center(const TrimDomain& trim, Vec pivot, double radius) {
    const TrimClass center = trim.classify(pivot);
    if (center != TrimClass::mixed) return center == TrimClass::outside;
    // The disk contains no trim edge, so all these probes have the same
    // parity. An offset avoids rays passing exactly through wire vertices.
    for (Vec probe : {Vec{pivot[0], pivot[1] + radius / 4, 0},
                      Vec{pivot[0] + radius / 4, pivot[1] - radius / 8, 0}}) {
        if (!(norm(exact(probe) - exact(pivot)).hi < radius)) continue;
        const TrimClass status = trim.classify(probe);
        if (status != TrimClass::mixed) return status == TrimClass::outside;
    }
    return false;
}

std::optional<AxialEnvelope> plane_envelope(const Plane& plane, const TrimDomain& trim) {
    const IVec normal = cross(plane.x, plane.y);
    const Interval length = norm(normal);
    if (!finite(length) || length.lo <= 0) return std::nullopt;
    const IVec axis = normal / length;
    const Interval metric = singular_values(plane.x, plane.y);
    std::optional<AxialEnvelope> fallback;
    for (const auto& loop : trim.loops) {
        // Arc centres provide useful pivots in bores and on disks. One arc
        // per wire is enough to propose a pivot. The entire boundary is then checked.
        const auto found = std::find_if(loop.begin(), loop.end(), [](const TrimCurve& edge) {
            return std::holds_alternative<Arc>(edge.geometry);
        });
        if (found == loop.end()) continue;
        const Arc& arc = std::get<Arc>(found->geometry);
        const Vec pivot = midpoint(arc.center);
        Interval radial = trim_radii(trim, pivot);
        if (!finite(radial)) continue;
        if (radial.lo > 0 && !outside_disk_center(trim, pivot, radial.lo)) radial.lo = 0;
        radial = radial * metric;
        radial.lo = std::max(0.0, radial.lo);
        if (!finite(radial)) continue;
        const AxialEnvelope envelope{plane.eval({pivot[0], pivot[1]}), axis, radial, 0};
        if (radial.lo > 0) return envelope;
        if (!fallback || radial.hi < fallback->radial.hi) fallback = envelope;
    }
    return fallback;
}

std::optional<AxialEnvelope> cylinder_envelope(const Cylinder& cylinder, Rect domain) {
    const Interval length = norm(cylinder.z);
    if (!finite(length) || length.lo <= 0) return std::nullopt;
    const IVec axis = cylinder.z / length;
    Interval radial = singular_values(cross(axis, cylinder.x), cross(axis, cylinder.y)) * absolute(cylinder.radius);
    radial.lo = std::max(0.0, radial.lo);
    const double radial_axial = (sqrt(square(dot(axis, cylinder.x)) + square(dot(axis, cylinder.y))) *
                                 absolute(cylinder.radius)).hi;
    const Interval axial = domain.v * length + Interval{-radial_axial, radial_axial};
    if (!finite(radial) || !finite(axial)) return std::nullopt;
    return AxialEnvelope{cylinder.origin, axis, radial, axial};
}

double gap(Interval a, Interval b) {
    return std::max({0.0, (Interval{a.lo} - b.hi).lo, (Interval{b.lo} - a.hi).lo});
}

bool clip_nonpositive(Interval& span, double intercept, double slope) {
    if (!std::isfinite(intercept) || !std::isfinite(slope)) return true;
    if (slope == 0) return intercept <= 0;
    const Interval root = Interval{-intercept} / slope;
    if (slope > 0) span.hi = std::min(span.hi, root.hi);
    else span.lo = std::max(span.lo, root.lo);
    return span.lo <= span.hi;
}

}  // namespace

std::optional<AxialEnvelope> axial_envelope(const FacePatch& patch, const TrimDomain& trim) {
    if (const auto* cylinder = std::get_if<Cylinder>(&patch.surface)) {
        return cylinder_envelope(*cylinder, patch.domain);
    }
    if (const auto* plane = std::get_if<Plane>(&patch.surface)) return plane_envelope(*plane, trim);
    return std::nullopt;
}

std::optional<Interval> axial_window(
    const AxialEnvelope& a, const AxialEnvelope& b, IVec velocity, Interval time, double clearance
) {
    const IVec offset = b.origin - a.origin;
    const Interval cosine = dot(a.axis, b.axis);
    const double sine = norm(cross(a.axis, b.axis)).hi;
    const double lateral = norm(cross(a.axis, offset) + cross(a.axis, velocity) * time).hi;
    const double radial_error = (Interval{lateral} + Interval{sine} * magnitude(b.axial)).hi;
    Interval b_radial{
        std::max(0.0, (Interval{b.radial.lo} * absolute(cosine) - radial_error).lo),
        (Interval{b.radial.hi} + radial_error).hi
    };
    if (!finite(b_radial)) return time;
    const double radial_gap = gap(a.radial, b_radial);
    if (radial_gap > clearance) return std::nullopt;
    const Interval remaining = square(Interval{clearance}) - square(Interval{radial_gap});
    if (!finite(remaining)) return time;
    const double allowable = sqrt({0, std::max(0.0, remaining.hi)}).hi;
    const double tilt = (Interval{sine} * b.radial.hi).hi;
    const Interval intercept = dot(a.axis, offset) + cosine * b.axial + Interval{-tilt, tilt};
    const Interval slope = dot(a.axis, velocity);
    if (!finite(intercept) || !finite(slope) || !std::isfinite(allowable)) return time;

    const double below = (Interval{a.axial.lo} - allowable - intercept.hi).lo;
    const double above = (Interval{intercept.lo} - allowable - a.axial.hi).lo;
    std::optional<Interval> result;
    for (Interval span : {Interval{time.lo, std::min(0.0, time.hi)}, Interval{std::max(0.0, time.lo), time.hi}}) {
        if (span.lo > span.hi) continue;
        const bool negative = span.hi <= 0;
        if (!clip_nonpositive(span, below, negative ? -slope.lo : -slope.hi) ||
            !clip_nonpositive(span, above, negative ? slope.hi : slope.lo)) continue;
        result = result ? hull(*result, span) : span;
    }
    return result;
}

}  // namespace nc::detail
