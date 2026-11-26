#include "compiled_geometry.hpp"
#include <algorithm>
#include <numbers>

namespace nc::compiled {
namespace {
Vec middle(IVec p) {
    return {p[0].center(), p[1].center(), p[2].center()};
}
Vec unit(Vec v) {
    const double l = length(v);
    return l ? scale(v, 1 / l) : Vec{};
}
std::array<Vec, 4> power_curve(std::array<Vec, 4> v) {
    return {v[0], scale(sub(v[1], v[0]), 3), scale(add(sub(v[2], scale(v[1], 2)), v[0]), 3),
            add(sub(v[3], v[0]), scale(sub(v[1], v[2]), 3))};
}
Patch make_patch(const FacePatch &p) {
    const auto &s = std::get<BezierSurface>(p.surface);
    Patch q{};
    q.domain = p.domain;
    q.full = s.domain;
    q.face = int(p.face_id) - 1;
    q.nu = int(s.nu);
    q.nv = int(s.nv);
    q.cubic = q.nu == 4 && q.nv == 4;
    const double weight = s.poles.front()[3].center();
    for (auto &h : s.poles) {
        std::array<double, 4> v;
        for (int k = 0; k < 4; ++k)
            v[k] = h[k].center() / weight;
        const Vec xyz{v[0] / v[3], v[1] / v[3], v[2] / v[3]};
        q.net.push_back(v);
        q.box.include(xyz);
        q.cubic &= std::abs(v[3] - 1) < 2e-14;
    }
    auto point = [&](int i, int j) {
        const auto &v = q.net[i * q.nv + j];
        return Vec{v[0] / v[3], v[1] / v[3], v[2] / v[3]};
    };
    q.anchor = point(0, 0);
    if (q.cubic) {
        std::array<Vec, 16> tmp;
        for (int i = 0; i < 4; ++i) {
            const auto c = power_curve({point(i, 0), point(i, 1), point(i, 2), point(i, 3)});
            for (int j = 0; j < 4; ++j)
                tmp[i * 4 + j] = c[j];
        }
        for (int j = 0; j < 4; ++j) {
            const auto c = power_curve({tmp[j], tmp[4 + j], tmp[8 + j], tmp[12 + j]});
            for (int i = 0; i < 4; ++i)
                q.power[i * 4 + j] = c[i];
        }
    }
    const auto j = q.at({q.full.u.center(), q.full.v.center()});
    q.axis = unit(cross3(j.u, j.v));
    q.frame = {unit(j.u), unit(cross3(q.axis, j.u)), q.axis};
    for (int i = 0; i < q.nu; ++i)
        for (int k = 0; k < q.nv; ++k) {
            const auto v = sub(point(i, k), q.anchor);
            const auto z = dot3(q.axis, v);
            q.near_plane = std::min(z, q.near_plane);
            q.far_plane = std::max(z, q.far_plane);
            q.frame_box.include(Vec{dot3(q.frame[0], v), dot3(q.frame[1], v), z});
        }
    if (q.cubic)
        q.prepare_cells();
    return q;
}
Jet analytic_jet(const Surface &s, Param uv) {
    const double u = uv[0], v = uv[1];
    if (const auto *p = std::get_if<Plane>(&s)) {
        Vec x = middle(p->x), y = middle(p->y);
        return {add(middle(p->origin), add(scale(x, u), scale(y, v))), x, y};
    }
    Vec o, x, y, z;
    double r, dr = 0, h = v, dh = 1;
    if (const auto *p = std::get_if<Cylinder>(&s)) {
        o = middle(p->origin);
        x = middle(p->x);
        y = middle(p->y);
        z = middle(p->z);
        r = p->radius.center();
    } else {
        const auto &q = std::get<Revolved>(s);
        o = middle(q.origin);
        x = middle(q.x);
        y = middle(q.y);
        z = middle(q.z);
        r = q.radius.center() + q.radial_slope.center() * v + q.meridian_radius.center() * std::cos(v);
        dr = q.radial_slope.center() - q.meridian_radius.center() * std::sin(v);
        h = q.height_slope.center() * v + q.meridian_radius.center() * std::sin(v);
        dh = q.height_slope.center() + q.meridian_radius.center() * std::cos(v);
    }
    Vec radial = add(scale(x, std::cos(u)), scale(y, std::sin(u))),
        tangent = add(scale(x, -std::sin(u)), scale(y, std::cos(u)));
    return {add(o, add(scale(radial, r), scale(z, h))), scale(tangent, r),
            add(scale(radial, dr), scale(z, dh))};
}
double periodic(double v, Interval bounds) {
    const double period = 2 * std::numbers::pi;
    return v + period * std::ceil((bounds.lo - v - 1e-12) / period);
}
std::vector<double> quadratic(double a, double b, double c) {
    if (std::abs(a) < 1e-24)
        return std::abs(b) < 1e-24 ? std::vector<double>{} : std::vector<double>{-c / b};
    const double disc = b * b - 4 * a * c;
    if (disc < 0)
        return {};
    const double q = -.5 * (b + std::copysign(std::sqrt(disc), b));
    if (q == 0)
        return {-b / (2 * a)};
    return {q / a, c / q};
}
// Isolate real polynomial roots on [lo,hi] between stationary points (torus ray).
std::vector<double> real_roots(const std::vector<double> &c, double lo, double hi) {
    if (c.size() == 2)
        return c[1] == 0 ? std::vector<double>{} : std::vector<double>{-c[0] / c[1]};
    std::vector<double> dc;
    for (std::size_t i = 1; i < c.size(); ++i)
        dc.push_back(i * c[i]);
    auto cuts = real_roots(dc, lo, hi);
    std::erase_if(cuts, [&](double t) { return !(t > lo && t < hi); });
    cuts.push_back(lo);
    cuts.push_back(hi);
    std::sort(cuts.begin(), cuts.end());
    const auto eval = [&](double t) {
        double s = 0;
        for (auto i = c.rbegin(); i != c.rend(); ++i)
            s = s * t + *i;
        return s;
    };
    std::vector<double> roots;
    for (auto x : cuts)
        if (std::abs(eval(x)) <= 1e-12)
            roots.push_back(x);
    for (std::size_t i = 1; i < cuts.size(); ++i) {
        double a = cuts[i - 1], b = cuts[i], fa = eval(a);
        if (fa * eval(b) >= 0)
            continue;
        for (int n = 0; n < 60; ++n) {
            double m = (a + b) * .5;
            if (fa * eval(m) > 0)
                a = m;
            else
                b = m;
        }
        roots.push_back((a + b) * .5);
    }
    return roots;
}
} // namespace
Placement Placement::between(const ContactPose &a, const ContactPose &b) {
    Placement r;
    for (int k = 0; k < 3; ++k) {
        Vec e{};
        e[k] = 1;
        r.cols[k] = b.inverse_rotate(a.rotate(e));
    }
    r.offset = scale(b.inverse_rotate(sub(a.origin, b.origin)), 1000);
    return r;
}
void Box::include(Vec p) {
    for (int k = 0; k < 3; ++k) {
        lo[k] = std::min(lo[k], p[k]);
        hi[k] = std::max(hi[k], p[k]);
    }
}
bool Box::overlaps(const Box &b, double pad) const {
    for (int k = 0; k < 3; ++k)
        if (lo[k] > b.hi[k] + pad || hi[k] + pad < b.lo[k])
            return false;
    return true;
}
bool Box::ray(Vec p, Vec d, double end, double pad) const {
    double t0 = -pad, t1 = end + pad;
    for (int k = 0; k < 3; ++k) {
        const double a = lo[k] - pad - p[k], b = hi[k] + pad - p[k];
        if (std::abs(d[k]) < 1e-18) {
            if (a > 0 || b < 0)
                return false;
            continue;
        }
        double x = a / d[k], y = b / d[k];
        if (x > y)
            std::swap(x, y);
        t0 = std::max(t0, x);
        t1 = std::min(t1, y);
        if (t0 > t1)
            return false;
    }
    return true;
}
Box Box::placed(const Placement &t) const {
    const auto c = t.point(scale(add(lo, hi), .5)), h = scale(sub(hi, lo), .5);
    Vec e{};
    for (int k = 0; k < 3; ++k)
        for (int j = 0; j < 3; ++j)
            e[k] += std::abs(t.cols[j][k]) * h[j];
    return {sub(c, e), add(c, e)};
}
void Tree::build(const std::vector<Box> &boxes) {
    if (boxes.empty())
        return;
    std::vector<int> index(boxes.size());
    std::iota(index.begin(), index.end(), 0);
    auto make = [&](auto &&self, auto begin, auto end) -> int {
        int i = int(nodes.size());
        nodes.emplace_back();
        Box box;
        for (auto j = begin; j != end; ++j)
            box.include(boxes[*j]);
        nodes[i].box = box;
        if (end - begin == 1)
            nodes[i].leaf = *begin;
        else {
            Vec w = sub(box.hi, box.lo);
            int axis = int(std::max_element(w.begin(), w.end()) - w.begin());
            auto mid = begin + (end - begin) / 2;
            std::nth_element(begin, mid, end, [&](int a, int b) {
                return boxes[a].lo[axis] + boxes[a].hi[axis] < boxes[b].lo[axis] + boxes[b].hi[axis];
            });
            const int l = self(self, begin, mid), r = self(self, mid, end);
            nodes[i].left = l;
            nodes[i].right = r;
        }
        return i;
    };
    make(make, index.begin(), index.end());
}
const Patch *Face::patch(Param uv) const {
    if (grid.empty())
        return nullptr;
    const auto find = [](const auto &k, double x) {
        return std::upper_bound(k.begin() + 1, k.end() - 1, x) - k.begin() - 1;
    };
    const auto idx = grid[find(u, uv[0]) * (v.size() - 1) + find(v, uv[1])];
    return idx < 0 ? nullptr : &patches[idx];
}
Jet Face::at(Param uv) const {
    if (analytic)
        return analytic_jet(*analytic, uv);
    if (const auto *q = patch(uv))
        return q->at(uv);
    throw UnresolvedContact("uncovered compiled CAD parameter cell");
}
bool Face::inside(Param uv) const {
    auto &r = info.domain;
    if (uv[0] < r.u.lo - 1e-10 || uv[0] > r.u.hi + 1e-10 || uv[1] < r.v.lo - 1e-10 || uv[1] > r.v.hi + 1e-10)
        return false;
    if (info.rectangle || boundary.empty())
        return true;
    const Vec p{uv[0], uv[1], 0};
    const Box ray{p, {infinity, p[1], 0}};
    int winding = 0;
    bool on_boundary = false;
    auto test = [&](int i) {
        if (on_boundary)
            return;
        const auto &edge = boundary[i];
        if (edge.box.lo[0] > p[0] && !edge.first_y.contains(p[1]) && !edge.last_y.contains(p[1])) {
            winding += int(edge.last_y.lo > p[1]) - int(edge.first_y.lo > p[1]);
        } else if (auto count = trim_crossings(*edge.curve, p)) {
            winding += *count;
        } else
            on_boundary = true;
    };
    trim_tree.visit(0, ray, test);
    return on_boundary || winding % 2 != 0;
}
void Face::span_roots(int i, Vec p, Vec d, double end, double tol, std::vector<RayRoot> &out) const {
    const auto first = out.size();
    patches[i].roots(p, d, end, tol, out);
    auto kept = first;
    for (auto j = first; j < out.size(); ++j) {
        if (!inside(out[j].uv)) continue;
        out[j].normal = scale(out[j].normal, info.orientation);
        if (kept != j) out[kept] = out[j];
        ++kept;
    }
    out.resize(kept);
}
void Face::roots(Vec p, Vec d, double end, double tol, int id, std::vector<RayRoot> &out) const {
    if (!analytic) {
        auto test = [&](int i) { span_roots(i, p, d, end, tol, out); };
        if (!tree.nodes.empty())
            tree.ray(0, in_frame(p), in_frame(d), end, 2 * tol, test);
        return;
    }
    const auto emit = [&](double t, Param uv) {
        if (t < -tol || t > end + tol || !inside(uv))
            return;
        auto q = at(uv);
        if (length(sub(q.p, add(p, scale(d, t)))) > std::max(2e-7, 2 * tol))
            return;
        Vec n = unit(cross3(q.u, q.v));
        out.push_back({t, scale(n, info.orientation), uv, id});
    };
    if (auto s = std::get_if<Plane>(analytic)) {
        auto delta = sub(p, middle(s->origin)), x = middle(s->x), y = middle(s->y), n = cross3(x, y);
        const double den = dot3(n, d);
        if (std::abs(den) < 1e-14)
            return;
        const double t = -dot3(n, delta) / den;
        if (t < -tol || t > end + tol)
            return;
        auto q = add(delta, scale(d, t));
        emit(t, {dot3(q, x), dot3(q, y)});
        return;
    }
    Vec origin, x, y, z;
    double R = 0, r = 0, rs = 0, hs = 1;
    if (auto s = std::get_if<Cylinder>(analytic)) {
        origin = middle(s->origin);
        x = middle(s->x);
        y = middle(s->y);
        z = middle(s->z);
        R = s->radius.center();
    } else {
        const auto &s2 = std::get<Revolved>(*analytic);
        origin = middle(s2.origin);
        x = middle(s2.x);
        y = middle(s2.y);
        z = middle(s2.z);
        R = s2.radius.center();
        r = s2.meridian_radius.center();
        rs = s2.radial_slope.center();
        hs = s2.height_slope.center();
    }
    Vec q = sub(p, origin), o{dot3(q, x), dot3(q, y), dot3(q, z)}, vdir{dot3(d, x), dot3(d, y), dot3(d, z)};
    std::vector<double> roots;
    if (r == 0) {
        double slope = rs / hs, h = R + slope * o[2], dh = slope * vdir[2];
        roots = quadratic(squared(vdir) - vdir[2] * vdir[2] - dh * dh,
                          2 * (o[0] * vdir[0] + o[1] * vdir[1] - h * dh), o[0] * o[0] + o[1] * o[1] - h * h);
    } else if (R == 0)
        roots = quadratic(1, 2 * dot3(o, vdir), squared(o) - r * r);
    else {
        double b = 2 * dot3(o, vdir), c = squared(o) + R * R - r * r;
        auto s = 4 * R * R;
        roots = real_roots({c * c - s * (o[0] * o[0] + o[1] * o[1]),
                            2 * b * c - 2 * s * (o[0] * vdir[0] + o[1] * vdir[1]),
                            b * b + 2 * c - s * (1 - vdir[2] * vdir[2]), 2 * b, 1},
                           -tol, end + tol);
    }
    for (auto t : roots)
        if (t >= -tol && t <= end + tol) {
            Vec h = add(o, scale(vdir, t));
            double a = periodic(std::atan2(h[1], h[0]), info.domain.u);
            double b =
                r == 0 ? h[2] / hs : periodic(std::atan2(h[2], std::hypot(h[0], h[1]) - R), info.domain.v);
            emit(t, {a, b});
        }
}
Rule Face::quadrature(int order) const {
    Rule rule;
    const auto &r = info.domain;
    const double du = r.u.width() / order, dv = r.v.width() / order;
    for (int i = 0; i < order; ++i)
        for (int j = 0; j < order; ++j) {
            Param uv{r.u.lo + (i + .5) * du, r.v.lo + (j + .5) * dv};
            if (!inside(uv))
                continue;
            auto q = at(uv);
            auto n = cross3(q.u, q.v);
            double a = length(n);
            if (a == 0)
                continue;
            if (!(j % 8) || rule.blocks.empty() || rule.blocks.back().end - rule.blocks.back().first >= 8)
                rule.blocks.push_back({{}, {1, 1, 1}, {-1, -1, -1}, rule.sites.size(), rule.sites.size()});
            auto &b = rule.blocks.back();
            n = scale(n, info.orientation / a);
            b.box.include(q.p);
            for (int k = 0; k < 3; ++k) {
                b.normal_lo[k] = std::min(b.normal_lo[k], n[k]);
                b.normal_hi[k] = std::max(b.normal_hi[k], n[k]);
            }
            rule.sites.push_back({q.p, n, uv, a * du * dv});
            ++b.end;
        }
    return rule;
}
Solid::Solid(PartFile f, DiscoverySites discovery) : file_(std::move(f)), faces(file_.faces.size()) {
    std::vector<Box> boxes;
    for (std::size_t i = 0; i < faces.size(); ++i) {
        auto &face = faces[i];
        face.info = file_.faces[i];
        face.box = {face.info.lo, face.info.hi};
        boxes.push_back(face.box);
    }
    for (auto &p : file_.geometry.patches) {
        auto &f = faces.at(p.face_id - 1);
        f.trim = &file_.geometry.trims.at(p.trim_id);
        if (std::holds_alternative<BezierSurface>(p.surface))
            f.patches.push_back(make_patch(p));
        else
            f.analytic = &p.surface;
    }
    for (auto &f : faces) {
        if (!f.trim)
            throw std::runtime_error("uncompiled CAD face");
        if (!f.info.rectangle) {
            std::vector<Box> enclosures;
            for (const auto &loop : f.trim->loops)
                for (const auto &edge : loop) {
                    const auto b = bounds(edge.geometry);
                    Box box{{b[0].lo, b[1].lo, 0}, {b[0].hi, b[1].hi, 0}};
                    f.boundary.push_back(
                        {&edge.geometry, box, evaluate(edge.geometry, 0)[1], evaluate(edge.geometry, 1)[1]});
                    enclosures.push_back(box);
                }
            f.trim_tree.build(enclosures);
        }
        if (!f.analytic) {
            std::vector<Box> spans;
            f.span_frame = f.patches[f.patches.size() / 2].frame;
            if (squared(f.span_frame[0]) < .9 || squared(f.span_frame[1]) < .9 || squared(f.span_frame[2]) < .9)
                f.span_frame = {Vec{1, 0, 0}, Vec{0, 1, 0}, Vec{0, 0, 1}};
            for (auto &p : f.patches) {
                f.u.push_back(p.full.u.lo);
                f.u.push_back(p.full.u.hi);
                f.v.push_back(p.full.v.lo);
                f.v.push_back(p.full.v.hi);
                Box box;
                for (const auto &h : p.net)
                    box.include(f.in_frame({h[0] / h[3], h[1] / h[3], h[2] / h[3]}));
                spans.push_back(box);
            }
            for (auto k : {&f.u, &f.v}) {
                std::sort(k->begin(), k->end());
                k->erase(std::unique(k->begin(), k->end()), k->end());
            }
            f.grid.assign((f.u.size() - 1) * (f.v.size() - 1), -1);
            for (int i = 0; i < int(f.patches.size()); ++i) {
                auto &p = f.patches[i];
                auto u = std::lower_bound(f.u.begin(), f.u.end(), p.full.u.lo) - f.u.begin();
                auto v = std::lower_bound(f.v.begin(), f.v.end(), p.full.v.lo) - f.v.begin();
                f.grid[u * (f.v.size() - 1) + v] = i;
            }
            f.tree.build(spans);
        }
    }
    tree.build(boxes);
    std::vector<Box> point_boxes, tile_boxes;
    for (int id = 0; id < int(faces.size()); ++id) {
        const auto &f = faces[id];
        const auto &r = f.info.domain;
        double du = r.u.width() / 6, dv = r.v.width() / 6;
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j) {
                Param uv{r.u.lo + (i + .5) * du, r.v.lo + (j + .5) * dv};
                if (!f.analytic && !f.patch(uv))
                    continue;
                auto jet = f.at(uv);
                auto n = cross3(jet.u, jet.v);
                if (squared(n) < 1e-24)
                    continue;
                Box box;
                box.include(jet.p);
                for (double h : {-.5, .5})
                    for (double k : {-.5, .5}) {
                        const Param corner{uv[0] + du * h, uv[1] + dv * k};
                        if (f.analytic || f.patch(corner))
                            box.include(f.at(corner).p);
                        else
                            box.include(f.box);
                    }
                const bool interior = f.inside(uv);
                if (interior) {
                    point_boxes.push_back({jet.p, jet.p});
                    material_sites.push_back(int(tiles.size()));
                }
                tiles.push_back({{jet.p, scale(n, f.info.orientation / length(n)), uv, 0}, box, id, interior});
                tile_boxes.push_back(box);
            }
        if (discovery == DiscoverySites::audit_chart_edges && f.info.rectangle) {
            // Sensitivity audit for slivers at the edge of a bounded chart.
            // These sites propose a search. Only verified material exits carry force.
            auto insert = [&](double u, double v) {
                Param uv{r.u.lo + u * r.u.width(), r.v.lo + v * r.v.width()};
                auto jet = f.at(uv);
                const Vec normal = cross3(jet.u, jet.v);
                if (length(normal) < 1e-14)
                    return;
                Box box{jet.p, jet.p};
                point_boxes.push_back(box);
                tile_boxes.push_back(box);
                material_sites.push_back(static_cast<int>(tiles.size()));
                tiles.push_back(
                    {{jet.p, scale(normal, f.info.orientation / length(normal)), uv, 0}, box, id, true});
            };
            for (int ring = 0; ring < 5; ++ring) {
                const double inset = std::ldexp(1. / 24., -2 * ring);
                for (double edge : {inset, 1 - inset}) {
                    for (int j = 0; j < 12; ++j) {
                        const double t = (j + .5) / 12;
                        insert(edge, t);
                        insert(t, edge);
                    }
                    for (double other : {inset, 1 - inset})
                        insert(edge, other);
                }
            }
        }
    }
    tile_tree.build(tile_boxes);
    // Boundary proposals are sources only. Newton still starts on the target's
    // finite chart tiles; adding degenerate target boxes multiplies pair work.
    if (discovery == DiscoverySites::trim_boundary) {
        for (int id = 0; id < int(faces.size()); ++id) {
            const auto &f = faces[id];
            const auto &r = f.info.domain;
            auto propose = [&](Param uv) {
                if (!f.inside(uv) || (!f.analytic && !f.patch(uv)))
                    return;
                const auto jet = f.at(uv);
                const Vec normal = cross3(jet.u, jet.v);
                const double mag = length(normal);
                if (mag < 1e-14)
                    return;
                Box box{jet.p, jet.p};
                point_boxes.push_back(box);
                material_sites.push_back(int(tiles.size()));
                tiles.push_back({{jet.p, scale(normal, f.info.orientation / mag), uv, 0}, box, id, true});
            };
            constexpr double inset = 1. / 6144.;
            if (f.info.rectangle) {
                for (double edge : {inset, 1 - inset}) {
                    for (int i = 0; i < 12; ++i) {
                        const double t = (i + .5) / 12;
                        propose({r.u.lo + edge * r.u.width(), r.v.lo + t * r.v.width()});
                        propose({r.u.lo + t * r.u.width(), r.v.lo + edge * r.v.width()});
                    }
                    for (double v : {inset, 1 - inset})
                        propose({r.u.lo + edge * r.u.width(), r.v.lo + v * r.v.width()});
                }
            } else {
                for (const auto &edge : f.boundary)
                    for (double t : {inset, .125, .375, .625, .875, 1 - inset}) {
                        const auto uv = middle(evaluate(*edge.curve, t));
                        const auto before = middle(evaluate(*edge.curve, t - inset));
                        const auto after = middle(evaluate(*edge.curve, t + inset));
                        const double du = (after[0] - before[0]) / r.u.width();
                        const double dv = (after[1] - before[1]) / r.v.width();
                        const double mag = std::hypot(du, dv);
                        if (mag > 0)
                            for (double sign : {-1., 1.})
                                propose({uv[0] - sign * inset * dv / mag * r.u.width(),
                                         uv[1] + sign * inset * du / mag * r.v.width()});
                    }
            }
        }
    }
    site_tree.build(point_boxes);
}
namespace {
template <class Visit> std::optional<RayRoot> select_exit(Vec d, double end, Visit visit) {
    std::optional<RayRoot> best;
    bool ambiguous = false;
    const auto record = [&](const std::vector<RayRoot> &hits, double tol) {
        for (auto &h : hits) {
            if (h.t < -tol || h.t > end)
                continue;
            if (!best || h.t < best->t - tol) {
                best = h;
                end = h.t + tol;
                ambiguous = false;
            } else if (std::abs(best->t - h.t) <= tol && dot3(best->normal, h.normal) < 1. - 1e-9)
                ambiguous = true;
        }
    };
    visit(end, record);
    if (!best || ambiguous || best->t <= 1e-7 || dot3(best->normal, d) <= 1e-8)
        return {};
    return best;
}
} // namespace
std::optional<RayRoot> Solid::first(Vec p, Vec d, double maximum) const {
    return select_exit(d, maximum, [&](double &end, const auto &record) {
        std::vector<RayRoot> hits;
        auto test = [&](int i) {
            hits.clear();
            const double tol = std::max(1e-7, faces[i].info.tolerance);
            faces[i].roots(p, d, end, tol, i, hits);
            record(hits, tol);
        };
        tree.ray(0, p, d, end, 1e-6, test);
    });
}
Solid::ColumnBlock Solid::column_block(const Box &swept) const {
    ColumnBlock block;
    auto face = [&](int id) {
        const auto &f = faces[id];
        BlockFace record{id, {}};
        if (!f.analytic) {
            const double pad = 2 * std::max(1e-6, f.info.tolerance);
            const Vec center = f.in_frame(scale(add(swept.lo, swept.hi), .5)),
                      half = scale(sub(swept.hi, swept.lo), .5);
            Vec r{pad, pad, pad};
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    r[i] += std::abs(f.span_frame[i][j]) * half[j];
            f.tree.shortlist(0, {sub(center, r), add(center, r)}, record.spans, 9);
            if (record.spans.empty())
                return;
        }
        block.candidates.push_back(std::move(record));
    };
    tree.visit(0, swept, face);
    return block;
}
std::optional<RayRoot> Solid::first(Vec p, Vec d, double maximum, const ColumnBlock &block,
                                   std::vector<RayRoot>& hits) const {
    const SlabRay ray{p, d};
    return select_exit(d, maximum, [&](double &end, const auto &record) {
        for (const auto &c : block.candidates) {
            const auto &f = faces[c.face];
            if (!ray.intersects(f.box, end, 1e-6))
                continue;
            const double tol = std::max(1e-7, f.info.tolerance);
            hits.clear();
            if (f.analytic || c.spans.size() > 8)
                f.roots(p, d, end, tol, c.face, hits);
            else
                for (int s : c.spans)
                    if (ray.intersects(f.patches[s].box, end, tol))
                        f.span_roots(s, p, d, end, tol, hits);
            record(hits, tol);
        }
    });
}
std::optional<RayRoot> Solid::crossing(Vec a, Vec b) const {
    Vec d = sub(b, a);
    double end = length(d);
    if (end < 1e-8)
        return {};
    d = scale(d, 1 / end);
    std::optional<RayRoot> cross;
    std::vector<RayRoot> hits;
    auto test = [&](int i) {
        hits.clear();
        faces[i].roots(a, d, end, std::max(1e-7, faces[i].info.tolerance), i, hits);
        for (const auto &h : hits)
            if (h.t >= 0 && h.t <= end) {
                cross = h;
                end = -1;
                break;
            }
    };
    tree.ray(0, a, d, end, 1e-7, test);
    return cross;
}
std::shared_ptr<const Solid> load_solid(const std::filesystem::path &p, DiscoverySites discovery) {
    return std::make_shared<Solid>(load(p), discovery);
}
} // namespace nc::compiled
