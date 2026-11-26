#include "compiled_geometry.hpp"
#include <algorithm>
#include <cmath>

namespace nc::compiled {
namespace {
using H = std::array<double, 4>;
using Net = std::vector<H>;
H blend(H a, H b, double t) {
    for (int c = 0; c < 4; ++c)
        a[c] += t * (b[c] - a[c]);
    return a;
}
H curve(H *p, int n, double t) {
    for (int k = n - 1; k > 0; --k)
        for (int j = 0; j < k; ++j)
            p[j] = blend(p[j], p[j + 1], t);
    return p[0];
}
H homogeneous_at(const Net &net, int nu, int nv, double x, double y, int derivative) {
    Net rows(nu), line(nv);
    for (int i = 0; i < nu; ++i) {
        std::copy_n(net.begin() + i * nv, nv, line.begin());
        if (derivative == 2)
            for (int j = 0; j < nv - 1; ++j)
                for (int k = 0; k < 4; ++k)
                    line[j][k] = (nv - 1) * (line[j + 1][k] - line[j][k]);
        rows[i] = curve(line.data(), nv - (derivative == 2), y);
    }
    if (derivative == 1)
        for (int i = 0; i < nu - 1; ++i)
            for (int k = 0; k < 4; ++k)
                rows[i][k] = (nu - 1) * (rows[i + 1][k] - rows[i][k]);
    return curve(rows.data(), nu - (derivative == 1), x);
}
Vec affine(const H &q) {
    return {q[0] / q[3], q[1] / q[3], q[2] / q[3]};
}
Vec project(const H &q, Vec origin) {
    return {q[0] - q[3] * origin[0], q[1] - q[3] * origin[1], q[2] - q[3] * origin[2]};
}
using Cubic = std::array<Vec, 16>;
struct ClipRange {
    double lo = 0, hi = 1;
};
// In a tensor Bernstein basis both the normalized parameter and a signed
// transverse coordinate are convex combinations of the same control points.
// Intersect their 2D control hull with the strip surrounding the ray.
bool clip_parameter(const std::array<double, 4> &low, const std::array<double, 4> &high, double pad,
                    ClipRange &range) {
    double l = 1, h = 0;
    for (int i = 0; i < 4; ++i) {
        if (low[i] <= pad && high[i] >= -pad) {
            l = std::min(l, i / 3.);
            h = std::max(h, i / 3.);
        }
        for (int j = i + 1; j < 4; ++j) {
            if ((low[i] > pad && low[j] > pad) || (high[i] < -pad && high[j] < -pad))
                continue;
            const auto include = [&](double x) {
                l = std::min(l, x);
                h = std::max(h, x);
            };
            // The hull across two vertical intervals is a trapezoid.
            // Its first and last admissible abscissae bound every edge crossing.
            double a = 0, b = 1;
            if (low[i] > pad)
                a = (low[i] - pad) / (low[i] - low[j]);
            if (high[i] < -pad)
                a = (-pad - high[i]) / (high[j] - high[i]);
            if (low[j] > pad)
                b = (pad - low[i]) / (low[j] - low[i]);
            if (high[j] < -pad)
                b = (high[i] + pad) / (high[i] - high[j]);
            include((i + (j - i) * a) / 3.);
            include((i + (j - i) * b) / 3.);
        }
    }
    range.lo = std::max(range.lo, l - 1e-10);
    range.hi = std::min(range.hi, h + 1e-10);
    return range.lo <= range.hi;
}
Cubic restrict_cubic(const Cubic &net, bool along, ClipRange range) {
    Cubic out;
    const auto lerp = [](Vec a, Vec b, double t) { return add(a, scale(sub(b, a), t)); };
    for (int k = 0; k < 4; ++k) {
        const auto idx = [&](int j) { return along ? 4 * j + k : 4 * k + j; };
        auto evaluate = [&](double t) {
            auto a = lerp(net[idx(0)], net[idx(1)], t), b = lerp(net[idx(1)], net[idx(2)], t),
                 c = lerp(net[idx(2)], net[idx(3)], t);
            auto x = lerp(a, b, t), y = lerp(b, c, t);
            return std::pair{lerp(x, y, t), sub(y, x)};
        };
        auto [p, dp] = evaluate(range.lo);
        auto [q, dq] = evaluate(range.hi);
        double w = range.hi - range.lo;
        out[idx(0)] = p;
        out[idx(1)] = add(p, scale(dp, w));
        out[idx(2)] = sub(q, scale(dq, w));
        out[idx(3)] = q;
    }
    return out;
}
struct RootSolver {
    const Patch &patch;
    Vec p, d;
    double end, tol;
    std::vector<RayRoot>& roots;
    std::size_t first_root;
    void record(Param uv) {
        const auto j = patch.at(uv);
        const double t = dot3(sub(j.p, p), d);
        Vec n = cross3(j.u, j.v);
        if (t < -tol || t > end + tol || length(n) < 1e-22)
            return;
        for (auto i=first_root; i<roots.size(); ++i)
            if (std::abs(roots[i].t - t) <= tol)
                return;
        roots.push_back({t, scale(n, 1 / length(n)), uv, patch.face});
    }
    bool projected_root(const Cubic &points, Rect rect, Vec axis_x, Vec axis_y, double pad) {
        std::array<Box, 2> deriv;
        Vec center{};
        constexpr double basis[]{.125, .375, .375, .125};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                const auto &q = points[4 * i + j];
                center = add(center, scale(q, basis[i] * basis[j]));
                if (i < 3)
                    deriv[0].include(scale(sub(points[4 * i + j + 4], q), 3));
                if (j < 3)
                    deriv[1].include(scale(sub(points[4 * i + j + 1], q), 3));
            }
        const Vec a = scale(add(deriv[0].lo, deriv[0].hi), .5), b = scale(add(deriv[1].lo, deriv[1].hi), .5);
        const double det = a[0] * b[1] - a[1] * b[0];
        if (std::abs(det) < 1e-18 || !std::isfinite(det))
            return false;
        const Vec rows[]{{b[1] / det, -b[0] / det, 0}, {-a[1] / det, a[0] / det, 0}};
        Param xy{};
        for (int k = 0; k < 2; ++k) {
            double error = 0;
            const auto row = rows[k];
            for (const auto &v : deriv)
                for (int j = 0; j < 2; ++j)
                    error += std::abs(row[j]) * (.5 * (v.hi[j] - v.lo[j]) + 8 * pad);
            if (error >= .93)
                return false;
            xy[k] = .5 - dot3(row, center);
            const double radius = .5 * error + pad * (std::abs(row[0]) + std::abs(row[1])) + 1e-11;
            if (xy[k] + radius < 0 || xy[k] - radius > 1)
                return true;
            xy[k] = std::clamp(xy[k], 0., 1.);
        }
        for (int i = 0; i < 8; ++i) {
            const Param uv{rect.u.lo + xy[0] * rect.u.width(), rect.v.lo + xy[1] * rect.v.width()};
            const auto jet = patch.at(uv);
            const auto rel = sub(jet.p, p);
            const double x = dot3(rel, axis_x), y = dot3(rel, axis_y);
            if (x * x + y * y < 1e-20) {
                record(uv);
                return true;
            }
            const double ux = dot3(jet.u, axis_x), uy = dot3(jet.u, axis_y), vx = dot3(jet.v, axis_x),
                         vy = dot3(jet.v, axis_y);
            const double jac = ux * vy - uy * vx;
            if (jac == 0)
                return false;
            xy[0] -= (vy * x - vx * y) / (jac * rect.u.width());
            xy[1] -= (ux * y - uy * x) / (jac * rect.v.width());
            if (xy[0] < -1e-10 || xy[0] > 1 + 1e-10 || xy[1] < -1e-10 || xy[1] > 1 + 1e-10)
                return false;
        }
        return false;
    }
    bool clip_cubic(const Cubic &points, Rect rect, Vec axis_x, Vec axis_y, double pad, int depth) {
        const auto a = sub(points[12], points[0]), b = sub(points[3], points[0]);
        const auto width = [](Vec v) { return v[0] * v[0] + v[1] * v[1]; };
        const Vec tangent = width(a) > width(b) ? a : b;
        const double mag = std::sqrt(width(tangent));
        ClipRange u, v;
        double zlo = infinity, zhi = -infinity;
        for (const auto &p : points) {
            zlo = std::min(zlo, p[2]);
            zhi = std::max(zhi, p[2]);
        }
        if (zlo > end + tol || zhi < -tol)
            return true;
        for (int row = 0; row < 3; ++row) {
            if (row == 0 && mag < 1e-16)
                continue;
            std::array<double, 4> ul, uh, vl, vh;
            ul.fill(infinity);
            vl = ul;
            uh.fill(-infinity);
            vh = uh;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) {
                    const auto &q = points[4 * i + j];
                    const double val = row ? q[row - 1] : (q[0] * tangent[1] - q[1] * tangent[0]) / mag;
                    ul[i] = std::min(ul[i], val);
                    uh[i] = std::max(uh[i], val);
                    vl[j] = std::min(vl[j], val);
                    vh[j] = std::max(vh[j], val);
                }
            if (!clip_parameter(ul, uh, pad, u) || !clip_parameter(vl, vh, pad, v))
                return true;
        }
        // Difficult or multiple roots retain the general enclosure solver.
        if (projected_root(points, rect, axis_x, axis_y, pad))
            return true;
        if (depth >= 18)
            return false;
        if (std::min(u.hi - u.lo, v.hi - v.lo) > .72) {
            const bool along = width(a) > width(b);
            auto [l, h] = split(rect, along);
            return clip_cubic(restrict_cubic(points, along, {0, .5}), l, axis_x, axis_y, pad, depth + 1) &&
                   clip_cubic(restrict_cubic(points, along, {.5, 1}), h, axis_x, axis_y, pad, depth + 1);
        }
        auto restricted = points;
        for (int k = 0; k < 2; ++k) {
            const auto h = k ? v : u;
            if (h.hi - h.lo > .95)
                continue;
            auto &axis = k ? rect.v : rect.u;
            const Interval next{axis.lo + axis.width() * h.lo, axis.lo + axis.width() * h.hi};
            if (next.width() <= 1e-13 * (1 + std::abs(axis.center())))
                return false;
            restricted = restrict_cubic(restricted, k == 0, h);
            axis = next;
        }
        return clip_cubic(restricted, rect, axis_x, axis_y, pad, depth + 1);
    }
    // A contraction of the projected root equation bounds all roots in this rectangle.
    // A seed outside the rectangle can prove exclusion; otherwise subdivide it.
    bool isolate(Rect r, Vec su, Vec sv, Vec ru, Vec rv, double weight, double minweight,
                 const Vec *midpoint = nullptr) {
        Vec a = cross3(sv, d), b = cross3(d, su);
        double det = dot3(su, a);
        if (std::abs(det) <= 1e-14 * length(su) * length(sv) || det == 0)
            return false;
        a = scale(a, 1 / det);
        b = scale(b, 1 / det);
        auto spread = [](Vec row, Vec e) {
            return std::abs(row[0]) * e[0] + std::abs(row[1]) * e[1] + std::abs(row[2]) * e[2];
        };
        const double eu = spread(a, ru) + spread(a, rv) + 1e-12, ev = spread(b, ru) + spread(b, rv) + 1e-12;
        if (std::max(eu, ev) >= .93)
            return false;
        Param uv{r.u.center(), r.v.center()};
        // The convex-hull subdivision uses homogeneous derivatives. The midpoint numerator
        // is the Cartesian residual multiplied by its local homogeneous weight.
        const Vec f = scale(sub(midpoint ? *midpoint : patch.at(uv).p, p), weight);
        double x = .5 - dot3(a, f), y = .5 - dot3(b, f);
        const double pu = tol * length(a) * minweight + 1e-11, pv = tol * length(b) * minweight + 1e-11;
        if (x + eu * .5 < -pu || x - eu * .5 > 1 + pu || y + ev * .5 < -pv || y - ev * .5 > 1 + pv)
            return true;
        if (x < 0 || x > 1 || y < 0 || y > 1) {
            double l[2]{0, 0}, h[2]{1, 1};
            const double e[2][2]{{spread(a, ru) + 1e-12, spread(a, rv) + 1e-12},
                                 {spread(b, ru) + 1e-12, spread(b, rv) + 1e-12}};
            // The preconditioned interval Newton image encloses every root
            // of a polynomial span; the rational case uses subdivision.
            if (patch.cubic)
                for (int pass = 0; pass < 4; ++pass) {
                    double cx = (l[0] + h[0]) * .5, cy = (l[1] + h[1]) * .5, hx = (h[0] - l[0]) * .5,
                           hy = (h[1] - l[1]) * .5;
                    const auto delta =
                        sub(patch.at({r.u.lo + r.u.width() * cx, r.v.lo + r.v.width() * cy}).p, p);
                    const double centers[]{cx - dot3(a, delta), cy - dot3(b, delta)};
                    for (int k = 0; k < 2; ++k) {
                        double radius = e[k][0] * hx + e[k][1] * hy + (k ? pv : pu);
                        l[k] = std::max(l[k], centers[k] - radius);
                        h[k] = std::min(h[k], centers[k] + radius);
                        if (l[k] > h[k])
                            return true;
                    }
                }
            x = std::clamp(x, l[0], h[0]);
            y = std::clamp(y, l[1], h[1]);
        }
        for (int i = 0; i < 9; ++i) {
            uv = {r.u.lo + x * r.u.width(), r.v.lo + y * r.v.width()};
            const auto j = patch.at(uv);
            Vec f0 = sub(j.p, p);
            f0 = sub(f0, scale(d, dot3(f0, d)));
            if (squared(f0) < 1e-20) {
                // Closed knot rectangles share boundary roots; collect and merge by depth.
                if (x >= -1e-10 && x <= 1 + 1e-10 && y >= -1e-10 && y <= 1 + 1e-10) {
                    record(uv);
                    return true;
                }
                return false;
            }
            const Vec cu = cross3(j.v, d), cv = cross3(d, j.u);
            const double jac = dot3(j.u, cu);
            if (jac == 0)
                return false;
            x -= dot3(f0, cu) / (jac * r.u.width());
            y -= dot3(f0, cv) / (jac * r.v.width());
            if (x < -1e-10 || x > 1 + 1e-10 || y < -1e-10 || y > 1 + 1e-10)
                return false;
        }
        return false;
    }
    void visit(const Net &net, Rect r, int depth) {
        const int nu = patch.nu, nv = patch.nv;
        Vec lateral = std::abs(d[0]) < .7 ? cross3(d, {1, 0, 0}) : cross3(d, {0, 1, 0});
        lateral = scale(lateral, 1 / length(lateral));
        Vec other = cross3(d, lateral);
        Vec lo{infinity, infinity, infinity}, hi{-infinity, -infinity, -infinity};
        Vec dl[2]{{infinity, infinity, infinity}, {infinity, infinity, infinity}};
        Vec dh[2]{{-infinity, -infinity, -infinity}, {-infinity, -infinity, -infinity}};
        double wmin = infinity, wmax = 0;
        for (int i = 0; i < nu; ++i)
            for (int j = 0; j < nv; ++j) {
                auto &h = net[i * nv + j];
                Vec num = project(h, p), q = scale(num, 1 / h[3]);
                Vec coord{dot3(q, lateral), dot3(q, other), dot3(q, d)};
                for (int k = 0; k < 3; ++k) {
                    lo[k] = std::min(lo[k], coord[k]);
                    hi[k] = std::max(hi[k], coord[k]);
                }
                wmin = std::min(wmin, h[3]);
                wmax = std::max(wmax, h[3]);
                for (int s = 0; s < 2; ++s) {
                    if (s ? j + 1 == nv : i + 1 == nu)
                        continue;
                    Vec v = scale(sub(project(net[i * nv + j + (s ? 1 : nv)], p), num), s ? nv - 1 : nu - 1);
                    for (int k = 0; k < 3; ++k) {
                        dl[s][k] = std::min(dl[s][k], v[k]);
                        dh[s][k] = std::max(dh[s][k], v[k]);
                    }
                }
            }
        const double roundoff =
            256 * std::numeric_limits<double>::epsilon() * (1 + length(p) + length(patch.anchor));
        if (lo[0] > roundoff || hi[0] < -roundoff || lo[1] > roundoff || hi[1] < -roundoff ||
            lo[2] > end + tol || hi[2] < -tol)
            return;
        const double w = wmin == wmax ? wmin : homogeneous_at(net, nu, nv, .5, .5, 0)[3];
        if (isolate(r, scale(add(dl[0], dh[0]), .5), scale(add(dl[1], dh[1]), .5), scale(sub(dh[0], dl[0]), .5),
                    scale(sub(dh[1], dl[1]), .5), w, wmax))
            return;
        // A ray lying on a planar support has no transversal root.
        if (std::abs(dot3(patch.axis, d)) < 1e-12 && patch.far_plane - patch.near_plane < 1e-10)
            return;
        if (depth >= 80 || (hi[2] - lo[2] <= tol && hi[0] - lo[0] <= tol && hi[1] - lo[1] <= tol)) {
            const Param uv{r.u.center(), r.v.center()};
            auto j = patch.at(uv);
            Vec v = sub(j.p, p);
            if (length(sub(v, scale(d, dot3(v, d)))) <= 2 * tol)
                record(uv);
            else if (depth >= 80)
                throw UnresolvedContact("compiled CAD root box exceeded resolution");
            return;
        }
        Net left(net), right(net), line(std::max(nu, nv));
        const bool along = length(add(dl[0], dh[0])) + length(sub(dh[0], dl[0])) >
                           length(add(dl[1], dh[1])) + length(sub(dh[1], dl[1]));
        const int n = along ? nu : nv, m = along ? nv : nu;
        for (int i = 0; i < m; ++i) {
            const auto index = [&](int j) { return along ? j * nv + i : i * nv + j; };
            for (int j = 0; j < n; ++j)
                line[j] = net[index(j)];
            for (int k = 0; k < n; ++k) {
                left[index(k)] = line[0];
                right[index(n - 1 - k)] = line[n - 1 - k];
                for (int j = 0; j < n - 1 - k; ++j)
                    line[j] = blend(line[j], line[j + 1], .5);
            }
        }
        auto halves = split(r, along);
        visit(left, halves.first, depth + 1);
        visit(right, halves.second, depth + 1);
    }
};
} // namespace
void Patch::prepare_cells() {
    // In the patch frame these control hulls also enclose nearly tangential
    // crossings. Each derivative bound is for the cell's unit square.
    constexpr int levels = 3;
    cells.resize((1 << (levels + 1)) - 1);
    using CubicNet = std::array<Vec, 16>;
    CubicNet initial;
    for (int k = 0; k < 16; ++k)
        initial[k] = affine(net[k]);
    auto build = [&](auto &&self, const CubicNet &points, Rect rect, int i, int level) -> void {
        auto &cell = cells[i];
        cell.rect = rect;
        cell.center = {};
        constexpr double basis[]{.125, .375, .375, .125};
        Box derivatives[2];
        for (int u = 0; u < 4; ++u)
            for (int v = 0; v < 4; ++v) {
                const auto &p = points[4 * u + v];
                const auto delta = sub(p, anchor);
                cell.bounds.include(Vec{dot3(frame[0], delta), dot3(frame[1], delta), dot3(frame[2], delta)});
                cell.center = add(cell.center, scale(p, basis[u] * basis[v]));
                if (u < 3)
                    derivatives[0].include(scale(sub(points[4 * u + v + 4], p), 3));
                if (v < 3)
                    derivatives[1].include(scale(sub(points[4 * u + v + 1], p), 3));
            }
        for (int j = 0; j < 2; ++j) {
            cell.slope[j] = scale(add(derivatives[j].lo, derivatives[j].hi), .5);
            cell.radius[j] = scale(sub(derivatives[j].hi, derivatives[j].lo), .5);
        }
        if (level == levels)
            return;
        const bool along =
            length(cell.slope[0]) + length(cell.radius[0]) > length(cell.slope[1]) + length(cell.radius[1]);
        CubicNet left, right;
        for (int strip = 0; strip < 4; ++strip) {
            auto index = [&](int j) { return along ? 4 * j + strip : 4 * strip + j; };
            std::array<Vec, 4> v;
            for (int j = 0; j < 4; ++j)
                v[j] = points[index(j)];
            for (int k = 0; k < 4; ++k) {
                left[index(k)] = v[0];
                right[index(3 - k)] = v[3 - k];
                for (int j = 0; j < 3 - k; ++j)
                    v[j] = scale(add(v[j], v[j + 1]), .5);
            }
        }
        const auto [a, b] = split(rect, along);
        self(self, left, a, 2 * i + 1, level + 1);
        self(self, right, b, 2 * i + 2, level + 1);
    };
    build(build, initial, full, 0, 0);
}
Jet Patch::at(Param uv) const {
    const double h = full.u.width(), k = full.v.width(), x = (uv[0] - full.u.lo) / h,
                 y = (uv[1] - full.v.lo) / k;
    if (cubic) {
        Jet q{};
        for (int c = 0; c < 3; ++c) {
            double a[4], b[4];
            for (int i = 0; i < 4; ++i) {
                const auto *v = &power[i * 4];
                a[i] = ((v[3][c] * y + v[2][c]) * y + v[1][c]) * y + v[0][c];
                b[i] = (3 * v[3][c] * y + 2 * v[2][c]) * y + v[1][c];
            }
            q.p[c] = ((a[3] * x + a[2]) * x + a[1]) * x + a[0];
            q.u[c] = ((3 * a[3] * x + 2 * a[2]) * x + a[1]) / h;
            q.v[c] = (((b[3] * x + b[2]) * x + b[1]) * x + b[0]) / k;
        }
        return q;
    }
    H q = homogeneous_at(net, nu, nv, x, y, 0), a = homogeneous_at(net, nu, nv, x, y, 1),
      b = homogeneous_at(net, nu, nv, x, y, 2);
    Vec p = affine(q);
    return {p, scale(project(a, p), 1 / (h * q[3])), scale(project(b, p), 1 / (k * q[3]))};
}
void Patch::roots(Vec p, Vec d, double end, double tol, std::vector<RayRoot>& out) const {
    double a = dot3(axis, sub(p, anchor)), b = a + dot3(axis, d) * end;
    if (std::min(a, b) > far_plane + tol || std::max(a, b) < near_plane - tol)
        return;
    const auto rel = sub(p, anchor);
    const Vec origin{dot3(frame[0], rel), dot3(frame[1], rel), a};
    const Vec direction{dot3(frame[0], d), dot3(frame[1], d), dot3(axis, d)};
    if (!frame_box.ray(origin, direction, end, 2 * tol))
        return;
    RootSolver solver{*this, p, d, end, tol, out, out.size()};
    if (cubic) {
        const SlabRay ray{origin, direction};
        const auto visit = [&](auto &&self, std::size_t i) -> bool {
            const auto &c = cells[i];
            if (!ray.intersects(c.bounds, end, 2 * tol))
                return true;
            if (solver.isolate(c.rect, c.slope[0], c.slope[1], c.radius[0], c.radius[1], 1, 1, &c.center))
                return true;
            return 2 * i + 2 < cells.size() && self(self, 2 * i + 1) && self(self, 2 * i + 2);
        };
        if (visit(visit, 0))
            return;
        if (std::abs(dot3(axis, d)) < 1e-12 && far_plane - near_plane < 1e-10)
            return;
        // Clip the projected control hull before the more costly 3D subdivision,
        // especially for short nearly tangential segments between load regions.
        Vec x = std::abs(d[0]) < .7 ? cross3(d, {1, 0, 0}) : cross3(d, {0, 1, 0});
        x = scale(x, 1 / length(x));
        const Vec y = cross3(d, x);
        Cubic transverse;
        for (int i = 0; i < 16; ++i) {
            const auto q = sub(affine(net[i]), p);
            transverse[i] = {dot3(x, q), dot3(y, q), dot3(d, q)};
        }
        double pad = 4096 * std::numeric_limits<double>::epsilon() * (1 + length(p) + length(anchor));
        if (solver.clip_cubic(transverse, full, x, y, pad, 0))
            return;
    }
    solver.visit(net, full, 0);
}
} // namespace nc::compiled
