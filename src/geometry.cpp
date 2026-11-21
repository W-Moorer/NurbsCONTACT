#include "nurbscontact/geometry.hpp"

#include <algorithm>
#include <utility>

namespace nc {
namespace {

HPoint blend(const HPoint& a, const HPoint& b, Interval t) {
    if (t.lo == 0 && t.hi == 0) return a;
    if (t.lo == 1 && t.hi == 1) return b;
    const Interval opposite = Interval{1} - t;
    HPoint result;
    for (std::size_t k = 0; k < result.size(); ++k) {
        result[k] = opposite * a[k] + t * b[k];
    }
    return result;
}

Interval unit(Interval t) {
    return {std::max(0.0, t.lo), std::min(1.0, t.hi)};
}

Interval normalize(Interval p, Interval domain) {
    if (p.lo == domain.lo && p.hi == domain.lo) return 0;
    if (p.lo == domain.hi && p.hi == domain.hi) return 1;
    return unit((p - domain.lo) / (Interval{domain.hi} - domain.lo));
}

HPoint casteljau(std::vector<HPoint> poles, Interval t) {
    if (poles.empty()) throw std::invalid_argument("empty Bezier control polygon");
    for (std::size_t count = poles.size(); count > 1; --count) {
        for (std::size_t i = 0; i + 1 < count; ++i) poles[i] = blend(poles[i], poles[i + 1], t);
    }
    return poles.front();
}

BezierCurve restrict_curve(const BezierCurve& curve, Interval a, Interval b) {
    if (a.hi == 0 && b.lo == 1) return curve;
    if (b.hi == 0) return {std::vector<HPoint>(curve.poles.size(), curve.poles.front())};
    BezierCurve prefix = curve.split(b).first;
    if (a.hi == 0) return prefix;
    return prefix.split(b.lo == 1 ? a : unit(a / b)).second;
}

IVec uv_box(Rect rect) { return {rect.u, rect.v, 0}; }

bool overlaps_uv(IVec a, IVec b) {
    return !disjoint(a[0], b[0]) && !disjoint(a[1], b[1]);
}

bool contained_uv(IVec a, IVec b) {
    return b[0].lo <= a[0].lo && a[0].hi <= b[0].hi &&
           b[1].lo <= a[1].lo && a[1].hi <= b[1].hi;
}

std::vector<SinCos> circular_support(Interval angles) {
    const Interval half = (Interval{angles.hi} - angles.lo) / 2;
    if (half.hi > 1.4) {
        return {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
    }
    const Interval mid = (Interval{angles.lo} + angles.hi) / 2;
    const SinCos center = sin_cos(mid);
    const Interval scale = sin_cos(half).cos;
    return {sin_cos(angles.lo), {center.sin / scale, center.cos / scale}, sin_cos(angles.hi)};
}

IVec curve_box(const Curve& curve, Interval t) {
    return std::visit([t](const auto& item) -> IVec {
        if constexpr (std::is_same_v<std::decay_t<decltype(item)>, Arc>) {
            return item.box(t);
        } else {
            return item.restrict(t).box();
        }
    }, curve);
}

enum class RaySide { below, above, unknown };

RaySide ray_side(Interval y, double level) {
    if (y.hi <= level) return RaySide::below;
    if (y.lo > level) return RaySide::above;
    return RaySide::unknown;
}

const BezierCurve* linear_curve(const Curve& curve) {
    const auto* bezier = std::get_if<BezierCurve>(&curve);
    return bezier && bezier->poles.size() == 2 ? bezier : nullptr;
}

std::optional<int> line_winding(const BezierCurve& line, Vec p) {
    const IVec a = cartesian(line.poles.front());
    const IVec b = cartesian(line.poles.back());
    const RaySide first = ray_side(a[1], p[1]);
    const RaySide last = ray_side(b[1], p[1]);
    if (std::max(a[0].hi, b[0].hi) < p[0]) return 0;
    if (first == RaySide::unknown || last == RaySide::unknown) return std::nullopt;
    if (first == last) return 0;
    const IVec ab = b - a;
    const IVec ap = exact(p) - a;
    const Interval side = ab[0] * ap[1] - ab[1] * ap[0];
    if (side.contains(0)) return std::nullopt;
    return last == RaySide::above ? (side.lo > 0 ? 1 : 0) : (side.hi < 0 ? -1 : 0);
}

bool line_may_cross(const BezierCurve& line, IVec rect) {
    const IVec a = cartesian(line.poles.front());
    const IVec b = cartesian(line.poles.back());
    if (!overlaps_uv(hull(a, b), rect)) return false;
    const IVec ab = b - a;
    const IVec ar = rect - a;
    return (ab[0] * ar[1] - ab[1] * ar[0]).contains(0);
}

std::optional<int> winding(const Curve& curve, Interval span, Vec point, int depth) {
    const IVec enclosure = curve_box(curve, span);
    if (!enclosure[1].contains(point[1]) || enclosure[0].hi < point[0]) return 0;
    if (enclosure[0].lo > point[0]) {
        // The whole segment is to the right of the ray origin. Its signed
        // crossing count is exactly the difference of the endpoint sides.
        const RaySide first = ray_side(evaluate(curve, span.lo)[1], point[1]);
        const RaySide last = ray_side(evaluate(curve, span.hi)[1], point[1]);
        if (first == RaySide::unknown || last == RaySide::unknown) return std::nullopt;
        return static_cast<int>(last == RaySide::above) - static_cast<int>(first == RaySide::above);
    }
    if (depth == 0) return std::nullopt;
    const double middle = span.center();
    const auto first = winding(curve, {span.lo, middle}, point, depth - 1);
    if (!first) return std::nullopt;
    const auto last = winding(curve, {middle, span.hi}, point, depth - 1);
    return last ? std::optional{*first + *last} : std::nullopt;
}

bool may_cross(const Curve& curve, Interval span, IVec rect, int depth) {
    const IVec enclosure = curve_box(curve, span);
    if (!overlaps_uv(enclosure, rect)) return false;
    if (depth == 0 || contained_uv(enclosure, rect)) return true;
    const double middle = span.center();
    return may_cross(curve, {span.lo, middle}, rect, depth - 1) ||
           may_cross(curve, {middle, span.hi}, rect, depth - 1);
}

void sample_curve(
    std::vector<Sample>& result, const FacePatch& patch, const TrimCurve& trim,
    Interval span, IVec rect, int depth
) {
    if (!overlaps_uv(curve_box(trim.geometry, span), rect)) return;
    for (double s : {span.lo, span.center(), span.hi}) {
        const IVec p = evaluate(trim.geometry, s);
        if (!overlaps_uv(p, rect)) continue;
        const auto* bezier = std::get_if<BezierSurface>(&patch.surface);
        if (bezier && !contained_uv(p, uv_box(bezier->domain))) continue;
        const Feature feature = s == 0 ? trim.start : s == 1 ? trim.end : trim.feature;
        const UV uv{p[0], p[1]};
        result.push_back({uv, evaluate(patch.surface, uv), feature});
        return;
    }
    if (depth == 0) return;
    const double middle = span.center();
    sample_curve(result, patch, trim, {span.lo, middle}, rect, depth - 1);
    sample_curve(result, patch, trim, {middle, span.hi}, rect, depth - 1);
}

}  // namespace

HPoint homogeneous(IVec point, Interval weight) {
    return {point[0] * weight, point[1] * weight, point[2] * weight, weight};
}

IVec cartesian(HPoint point) {
    if (point[3].lo <= 0) throw std::domain_error("Bezier weight cannot be bounded away from zero");
    return {point[0] / point[3], point[1] / point[3], point[2] / point[3]};
}

IVec BezierCurve::eval(Interval parameter) const { return cartesian(casteljau(poles, parameter)); }

std::pair<BezierCurve, BezierCurve> BezierCurve::split(Interval parameter) const {
    if (poles.empty()) throw std::invalid_argument("empty Bezier control polygon");
    std::vector<HPoint> left(poles.size());
    std::vector<HPoint> right(poles.size());
    std::vector<HPoint> work = poles;
    left.front() = poles.front();
    right.back() = poles.back();
    for (std::size_t level = 1; level < poles.size(); ++level) {
        const std::size_t count = poles.size() - level;
        for (std::size_t i = 0; i < count; ++i) work[i] = blend(work[i], work[i + 1], parameter);
        left[level] = work.front();
        right[count - 1] = work[count - 1];
    }
    return {{std::move(left)}, {std::move(right)}};
}

BezierCurve BezierCurve::restrict(Interval parameters) const {
    return restrict_curve(*this, parameters.lo, parameters.hi);
}

BezierCurve BezierCurve::restrict(Interval first, Interval last) const {
    return restrict_curve(*this, first, last);
}

BezierCurve BezierCurve::reparameterize(Interval first, Interval last) const {
    if (poles.empty()) throw std::invalid_argument("empty Bezier control polygon");
    BezierCurve result;
    for (std::size_t i = 0; i < poles.size(); ++i) {
        auto work = poles;
        // The blossom is multiaffine in all arguments. It also represents
        // the same polynomial outside the source span when weights stay positive.
        for (std::size_t count = poles.size(); count > 1; --count) {
            const Interval t = count <= i + 1 ? last : first;
            for (std::size_t j = 0; j + 1 < count; ++j) work[j] = blend(work[j], work[j + 1], t);
        }
        if (work.front()[3].lo <= 0) {
            throw std::runtime_error("p-curve continuation has a nonpositive enclosure weight");
        }
        result.poles.push_back(work.front());
    }
    return result;
}

IVec BezierCurve::box() const {
    if (poles.empty()) throw std::invalid_argument("empty Bezier control polygon");
    IVec result = cartesian(poles.front());
    for (std::size_t i = 1; i < poles.size(); ++i) result = hull(result, cartesian(poles[i]));
    return result;
}

IVec BezierSurface::eval(UV uv) const {
    const Interval u = normalize(uv[0], domain.u);
    const Interval v = normalize(uv[1], domain.v);
    std::vector<HPoint> column;
    column.reserve(nu);
    for (std::size_t i = 0; i < nu; ++i) {
        const auto start = poles.begin() + static_cast<std::ptrdiff_t>(i * nv);
        column.push_back(casteljau({start, start + static_cast<std::ptrdiff_t>(nv)}, v));
    }
    return cartesian(casteljau(std::move(column), u));
}

std::vector<IVec> BezierSurface::support(Rect rect) const {
    const Interval va = normalize(rect.v.lo, domain.v);
    const Interval vb = normalize(rect.v.hi, domain.v);
    const Interval ua = normalize(rect.u.lo, domain.u);
    const Interval ub = normalize(rect.u.hi, domain.u);
    std::vector<BezierCurve> columns(nv);
    for (std::size_t i = 0; i < nu; ++i) {
        const auto start = poles.begin() + static_cast<std::ptrdiff_t>(i * nv);
        const BezierCurve row = restrict_curve({{start, start + static_cast<std::ptrdiff_t>(nv)}}, va, vb);
        for (std::size_t j = 0; j < nv; ++j) columns[j].poles.push_back(row.poles[j]);
    }
    std::vector<IVec> result;
    result.reserve(nu * nv);
    for (const BezierCurve& column : columns) {
        for (HPoint pole : restrict_curve(column, ua, ub).poles) result.push_back(cartesian(pole));
    }
    return result;
}

IVec Plane::eval(UV uv) const { return origin + x * uv[0] + y * uv[1]; }

std::vector<IVec> Plane::support(Rect rect) const {
    return {eval({rect.u.lo, rect.v.lo}), eval({rect.u.lo, rect.v.hi}),
            eval({rect.u.hi, rect.v.lo}), eval({rect.u.hi, rect.v.hi})};
}

IVec Cylinder::eval(UV uv) const {
    const SinCos sc = sin_cos(uv[0]);
    return origin + (x * sc.cos + y * sc.sin) * radius + z * uv[1];
}

std::vector<IVec> Cylinder::support(Rect rect) const {
    std::vector<IVec> result;
    for (SinCos sc : circular_support(rect.u)) {
        const IVec point = origin + (x * sc.cos + y * sc.sin) * radius;
        result.push_back(point + z * rect.v.lo);
        result.push_back(point + z * rect.v.hi);
    }
    return result;
}

IVec Revolved::eval(UV uv) const {
    const SinCos u = sin_cos(uv[0]);
    const SinCos v = meridian_radius.is_zero() ? SinCos{0, 1} : sin_cos(uv[1]);
    const Interval r = radius + radial_slope * uv[1] + meridian_radius * v.cos;
    const Interval h = height_slope * uv[1] + meridian_radius * v.sin;
    return origin + (x * u.cos + y * u.sin) * r + z * h;
}

std::vector<IVec> Revolved::support(Rect rect) const {
    const auto meridian = meridian_radius.is_zero() ? std::vector<SinCos>{{0, 1}} : circular_support(rect.v);
    const auto linear = radial_slope.is_zero() && height_slope.is_zero()
        ? std::vector<double>{0} : std::vector<double>{rect.v.lo, rect.v.hi};
    std::vector<IVec> result;
    for (SinCos u : circular_support(rect.u)) {
        const IVec radial = x * u.cos + y * u.sin;
        for (SinCos v : meridian) {
            for (double t : linear) {
                const Interval r = radius + radial_slope * t + meridian_radius * v.cos;
                const Interval h = height_slope * t + meridian_radius * v.sin;
                result.push_back(origin + radial * r + z * h);
            }
        }
    }
    return result;
}

IVec Arc::eval(Interval parameter) const {
    const Interval angle = reversed ? Interval{angles.hi} + (Interval{angles.lo} - angles.hi) * parameter
                                    : Interval{angles.lo} + (Interval{angles.hi} - angles.lo) * parameter;
    const SinCos sc = sin_cos(angle);
    return center + x * sc.cos + y * sc.sin;
}

IVec Arc::box(Interval parameters) const {
    return eval(parameters);
}

TrimClass TrimDomain::classify(Vec uv) const {
    if (loops.empty()) return TrimClass::inside;
    int crossings = 0;
    for (const auto& loop : loops) {
        for (const TrimCurve& trim : loop) {
            const auto* line = linear_curve(trim.geometry);
            const auto value = line ? line_winding(*line, uv) : winding(trim.geometry, {0, 1}, uv, 22);
            if (!value) return TrimClass::mixed;
            crossings += *value;
        }
    }
    return crossings % 2 != 0 ? TrimClass::inside : TrimClass::outside;
}

TrimClass TrimDomain::classify(Rect rect) const {
    if (loops.empty()) return TrimClass::inside;
    const IVec box = uv_box(rect);
    for (const auto& loop : loops) {
        for (const TrimCurve& trim : loop) {
            const auto* line = linear_curve(trim.geometry);
            if (line ? line_may_cross(*line, box) : may_cross(trim.geometry, {0, 1}, box, 12)) {
                return TrimClass::mixed;
            }
        }
    }
    for (double y : {rect.v.center(), std::midpoint(rect.v.lo, rect.v.center()),
                     std::midpoint(rect.v.center(), rect.v.hi)}) {
        const TrimClass result = classify(Vec{rect.u.center(), y, 0});
        if (result != TrimClass::mixed) return result;
    }
    return TrimClass::mixed;
}

IVec evaluate(const Surface& surface, UV uv) {
    return std::visit([&](const auto& item) { return item.eval(uv); }, surface);
}

IVec evaluate(const Curve& curve, Interval parameter) {
    return std::visit([&](const auto& item) { return item.eval(parameter); }, curve);
}

IVec bounds(const Curve& curve, Interval parameters) {
    return curve_box(curve, parameters);
}

std::vector<IVec> support(const Surface& surface, Rect rect) {
    return std::visit([&](const auto& item) { return item.support(rect); }, surface);
}

IVec bounding_box(const std::vector<IVec>& support_points) {
    if (support_points.empty()) throw std::invalid_argument("empty support set");
    IVec result = support_points.front();
    for (std::size_t i = 1; i < support_points.size(); ++i) result = hull(result, support_points[i]);
    return result;
}

std::vector<IVec> corners(IVec box) {
    std::vector<IVec> result;
    result.reserve(8);
    for (double x : {box[0].lo, box[0].hi}) {
        for (double y : {box[1].lo, box[1].hi}) {
            for (double z : {box[2].lo, box[2].hi}) result.push_back({x, y, z});
        }
    }
    return result;
}

std::pair<Rect, Rect> split(Rect rect, bool along_u) {
    if (along_u) {
        const double middle = rect.u.center();
        return {{{rect.u.lo, middle}, rect.v}, {{middle, rect.u.hi}, rect.v}};
    }
    const double middle = rect.v.center();
    return {{rect.u, {rect.v.lo, middle}}, {rect.u, {middle, rect.v.hi}}};
}

std::vector<Sample> samples(const FacePatch& patch, const TrimDomain& trim, Rect rect, TrimClass kind) {
    std::vector<Sample> result;
    if (kind == TrimClass::outside) return result;
    const Vec center{rect.u.center(), rect.v.center(), 0};
    if (kind == TrimClass::inside || trim.classify(center) == TrimClass::inside) {
        const UV uv{center[0], center[1]};
        result.push_back({uv, evaluate(patch.surface, uv), {FeatureKind::face, patch.face_id}});
    }
    if (kind == TrimClass::inside) {
        for (double u : {rect.u.lo, rect.u.hi}) {
            for (double v : {rect.v.lo, rect.v.hi}) {
                const UV uv{u, v};
                result.push_back({uv, evaluate(patch.surface, uv), {FeatureKind::face, patch.face_id}});
            }
        }
    } else {
        for (const auto& loop : trim.loops) {
            for (const TrimCurve& curve : loop) sample_curve(result, patch, curve, {0, 1}, uv_box(rect), 6);
        }
    }
    return result;
}

}  // namespace nc
