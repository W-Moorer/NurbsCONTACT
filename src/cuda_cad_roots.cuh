// Included inside the CUDA geometry namespace. All lengths are in millimetres.
__device__ V sum3(V a, V b) { return {{a.x[0] + b.x[0], a.x[1] + b.x[1], a.x[2] + b.x[2]}}; }
__device__ V trim_position(View g, const TrimPiece &c, double t) {
    const auto *p = g.trim_poles + c.start;
    if (!c.count) {
        double a = c.angle + c.turn * t;
        return {{p[0].x + p[1].x * cos(a) + p[2].x * sin(a), p[0].y + p[1].y * cos(a) + p[2].y * sin(a), 0}};
    }
    if (t <= 0 || t >= 1) {
        const auto &q = p[t <= 0 ? 0 : c.count - 1];
        return {{q.x / q.w, q.y / q.w, 0}};
    }
    // Start from the closer endpoint to avoid large powers of t/(1-t).
    bool rev = t > .5;
    double b = pow(rev ? t : 1 - t, c.count - 1), ratio = rev ? (1 - t) / t : t / (1 - t);
    double x = 0, y = 0, w = 0;
    for (int i = 0; i < c.count; ++i) {
        auto q = p[rev ? c.count - 1 - i : i];
        x += b * q.x;
        y += b * q.y;
        w += b * q.w;
        b *= ratio * (c.count - 1 - i) / (i + 1);
    }
    return {{x / w, y / w, 0}};
}
// Signed winding against the original p-curve with conservative derivative enclosures.
struct Bound {
    double lo, hi;
};
#include "cuda_trim_intervals.cuh"
// Preserve the closed-face boundary convention used by Face::inside for linear trims.
__device__ __noinline__ int face_contains(View g, const FaceData &f, double u, double v) {
    const auto r = f.domain;
    if (u < r.u - 1e-10 || u > r.U + 1e-10 || v < r.v - 1e-10 || v > r.Vv + 1e-10)
        return 0;
    if (f.rectangle || f.trim_end == f.trim_first)
        return 1;
    int winding = 0;
    for (int j = f.trim_first; j < f.trim_end;) {
        const auto &node = g.trim_tree[j];
        if (node.box.hi.x[0] < u || node.box.hi.x[1] < v || node.box.lo.x[1] > v) {
            j = node.next;
            continue;
        }
        ++j;
        if (node.leaf < 0)
            continue;
        const auto &e = g.trims[node.leaf];
        if (node.box.lo.x[0] > u && fabs(v - e.first_y) > e.roundoff && fabs(v - e.last_y) > e.roundoff) {
            winding += int(e.last_y > v) - int(e.first_y > v);
            continue;
        }
        if (e.count == 2) {
            const auto a = e.endpoints[0], b = e.endpoints[1];
            if (fmax(a.hi.x[0], b.hi.x[0]) < u)
                continue;
            int ay = a.hi.x[1] <= v ? 0 : (a.lo.x[1] > v ? 1 : -1);
            int by = b.hi.x[1] <= v ? 0 : (b.lo.x[1] > v ? 1 : -1);
            if (ay < 0 || by < 0)
                return 1;
            if (ay == by)
                continue;
            Bound ax{a.lo.x[0], a.hi.x[0]}, az{a.lo.x[1], a.hi.x[1]};
            auto dx = minus({b.lo.x[0], b.hi.x[0]}, ax), dy = minus({b.lo.x[1], b.hi.x[1]}, az);
            Bound side = minus(times(dx, minus({v, v}, az)), times(dy, minus({u, u}, ax)));
            if (side.lo <= 0 && side.hi >= 0)
                return 1;
            winding += by ? (side.lo > 0) : -(side.hi < 0);
            continue;
        }
        unsigned int cell = 1;
        const int previous_winding = winding;
        bool uncertain = false;
        int level = 0;
        for (;;) {
            double h = ldexp(1., -level), left = cell * h - 1, mid = left + h * .5;
            V p = trim_position(g, e, mid);
            double dx = h * .5 * e.du + e.roundoff, dy = h * .5 * e.dv + e.roundoff;
            if (!isfinite(p.x[0]) || !isfinite(p.x[1])) {
                uncertain = true;
                break;
            }
            if (p.x[0] + dx >= u && fabs(p.x[1] - v) <= dy) {
                if (p.x[0] - dx > u) {
                    double a = trim_position(g, e, left).x[1], b = trim_position(g, e, left + h).x[1];
                    if (fabs(a - v) <= e.roundoff || fabs(b - v) <= e.roundoff) {
                        uncertain = true;
                        break;
                    }
                    winding += int(b > v) - int(a > v);
                } else if (level < 22) {
                    cell *= 2;
                    ++level;
                    continue;
                } else {
                    uncertain = true;
                    break;
                }
            }
            while (cell > 1 && (cell & 1)) {
                cell /= 2;
                --level;
            }
            if (cell == 1)
                break;
            ++cell;
        }
        if (uncertain) {
            winding = previous_winding;
            int state = interval_winding(g, e, u, v, winding);
            if (state != 0)
                return state;
        }
    }
    return winding % 2 != 0;
}
__device__ int knot_slot(const double *knots, int count, double x) {
    int lo = 1, hi = count - 1;
    while (lo < hi) {
        int m = (lo + hi) / 2;
        if (knots[m] <= x)
            lo = m + 1;
        else
            hi = m;
    }
    return lo - 1;
}
__device__ bool face_jet(View g, int id, double u, double v, Jet &j) {
    const auto &f = g.charts[id];
    if (f.kind == 1) {
        j = {sum3(f.origin, sum3(scale(f.frame[0], u), scale(f.frame[1], v))), f.frame[0], f.frame[1]};
        return true;
    }
    if (f.kind == 2) {
        V radial = sum3(scale(f.frame[0], cos(u)), scale(f.frame[1], sin(u)));
        V du = scale(sum3(scale(f.frame[0], -sin(u)), scale(f.frame[1], cos(u))), f.radius);
        j = {sum3(f.origin, sum3(scale(radial, f.radius), scale(f.frame[2], v))), du, f.frame[2]};
        return true;
    }
    if (f.kind == 3) {
        V radial = sum3(scale(f.frame[0], cos(u)), scale(f.frame[1], sin(u)));
        V tangent = sum3(scale(f.frame[0], -sin(u)), scale(f.frame[1], cos(u)));
        const double r = f.radius + f.radial_slope * v + f.meridian_radius * cos(v);
        const double h = f.height_slope * v + f.meridian_radius * sin(v);
        j = {sum3(f.origin, sum3(scale(radial, r), scale(f.frame[2], h))), scale(tangent, r),
             sum3(scale(radial, f.radial_slope - f.meridian_radius * sin(v)),
                  scale(f.frame[2], f.height_slope + f.meridian_radius * cos(v)))};
        return true;
    }
    if (f.nu < 2 || f.nv < 2)
        return false;
    int i = knot_slot(g.knots + f.knots_u, f.nu, u), k = knot_slot(g.knots + f.knots_v, f.nv, v);
    int span = g.grid[f.grid + i * (f.nv - 1) + k];
    if (span < 0 || g.patches[span].cell < 0)
        return false;
    j = at(g.patches[span], u, v);
    return true;
}
struct RootSelection {
    Hit hit{0, -1};
    bool ambiguous = false;
    __device__ void insert(Hit h, double tol, double &end) {
        if (h.t < -tol || h.t > end)
            return;
        if (hit.face < 0 || h.t < hit.t - tol) {
            hit = h;
            ambiguous = false;
            end = h.t + tol;
        } else if (fabs(hit.t - h.t) <= tol && dot(hit.normal, h.normal) < 1. - 1e-9)
            ambiguous = true;
    }
};
// Subdivide the interval between derivative zeros. Include repeated roots at stationary points.
template <int Degree>
__device__ int polynomial_roots(const double *coeff, double lo, double hi, double *roots) {
    if constexpr (Degree == 1) {
        if (coeff[1] == 0)
            return 0;
        roots[0] = -coeff[0] / coeff[1];
        return 1;
    } else {
        double derivative[Degree], stationary[32];
        for (int k = 1; k <= Degree; ++k)
            derivative[k - 1] = k * coeff[k];
        int m = polynomial_roots<Degree - 1>(derivative, lo, hi, stationary);
        if (m < 0)
            return -1;
        double cuts[34];
        int n = 1;
        cuts[0] = lo;
        for (int i = 0; i < m; ++i)
            if (stationary[i] > lo && stationary[i] < hi)
                cuts[n++] = stationary[i];
        cuts[n++] = hi;
        for (int i = 1; i < n; ++i) {
            double v = cuts[i];
            int j = i;
            while (j && cuts[j - 1] > v) {
                cuts[j] = cuts[j - 1];
                --j;
            }
            cuts[j] = v;
        }
        auto eval = [&](double t) {
            double y = 0;
            for (int k = Degree; k >= 0; --k)
                y = y * t + coeff[k];
            return y;
        };
        int count = 0;
        for (int i = 0; i < n; ++i)
            if (fabs(eval(cuts[i])) <= 1e-12) {
                if (count == 32)
                    return -1;
                roots[count++] = cuts[i];
            }
        for (int i = 1; i < n; ++i) {
            double a = cuts[i - 1], b = cuts[i], fa = eval(a);
            if (fa * eval(b) >= 0)
                continue;
            for (int j = 0; j < 60; ++j) {
                double mid = .5 * (a + b);
                if (fa * eval(mid) > 0)
                    a = mid;
                else
                    b = mid;
            }
            if (count == 32)
                return -1;
            roots[count++] = .5 * (a + b);
        }
        return count;
    }
}
__device__ int analytic_hits(View g, int id, Ray ray, double &end, RootSelection &selection, bool any) {
    const auto &f = g.charts[id];
    double roots[32];
    int count = 0;
    V delta = sub(ray.p, f.origin);
    if (f.kind == 1) {
        V n = cross(f.frame[0], f.frame[1]);
        double den = dot(n, ray.d);
        if (fabs(den) < 1e-14)
            return 0;
        roots[count++] = -dot(n, delta) / den;
    } else if (f.kind == 2 || f.kind == 3) {
        V o = basis(f.frame, delta), d = basis(f.frame, ray.d);
        double a = dot(d, d) - d.x[2] * d.x[2], b = 2 * (o.x[0] * d.x[0] + o.x[1] * d.x[1]),
               c = o.x[0] * o.x[0] + o.x[1] * o.x[1] - f.radius * f.radius;
        if (f.kind == 3 && f.meridian_radius == 0) {
            if (f.height_slope == 0)
                return -2;
            double slope = f.radial_slope / f.height_slope, h = f.radius + slope * o.x[2],
                   dh = slope * d.x[2];
            a -= dh * dh;
            b = 2 * (o.x[0] * d.x[0] + o.x[1] * d.x[1] - h * dh);
            c = o.x[0] * o.x[0] + o.x[1] * o.x[1] - h * h;
        } else if (f.kind == 3 && f.radius == 0) {
            a = 1;
            b = 2 * dot(o, d);
            c = dot(o, o) - f.meridian_radius * f.meridian_radius;
        }
        if (f.kind == 3 && f.radius != 0 && f.meridian_radius != 0) {
            double linear = 2 * dot(o, d),
                   constant = dot(o, o) + f.radius * f.radius - f.meridian_radius * f.meridian_radius;
            double s = 4 * f.radius * f.radius;
            double coeff[5]{constant * constant - s * (o.x[0] * o.x[0] + o.x[1] * o.x[1]),
                            2 * linear * constant - 2 * s * (o.x[0] * d.x[0] + o.x[1] * d.x[1]),
                            linear * linear + 2 * constant - s * (1 - d.x[2] * d.x[2]), 2 * linear, 1};
            count = polynomial_roots<4>(coeff, -f.tol, end + f.tol, roots);
            if (count < 0)
                return -2;
        } else if (fabs(a) < 1e-24) {
            if (fabs(b) >= 1e-24)
                roots[count++] = -c / b;
        } else if (double disc = b * b - 4 * a * c; disc >= 0) {
            double q = -.5 * (b + copysign(sqrt(disc), b));
            if (q == 0)
                roots[count++] = -b / (2 * a);
            else {
                roots[count++] = q / a;
                roots[count++] = c / q;
            }
        }
    } else
        return -2;
    for (int k = 0; k < count; ++k) {
        double t = roots[k];
        if (t < -f.tol || t > end || (any && t < 0))
            continue;
        V q = sum3(delta, scale(ray.d, t));
        double u = dot(q, f.frame[0]), v = dot(q, f.frame[1]);
        if (f.kind >= 2) {
            constexpr double period = 6.2831853071795864769;
            const double radius = hypot(u, v);
            u = atan2(v, u);
            u += period * ceil((f.domain.u - u - 1e-12) / period);
            v = dot(q, f.frame[2]);
            if (f.kind == 3) {
                if (f.meridian_radius == 0)
                    v /= f.height_slope;
                else {
                    v = atan2(v, radius - f.radius);
                    v += period * ceil((f.domain.v - v - 1e-12) / period);
                }
            }
        }
        int inside = face_contains(g, f, u, v);
        if (inside < 0)
            return -2;
        if (!inside)
            continue;
        Jet j;
        face_jet(g, id, u, v, j);
        if (norm(sub(j.p, sum3(ray.p, scale(ray.d, t)))) > fmax(2e-7, 2 * f.tol))
            continue;
        auto normal = cross(j.u, j.v);
        normal = scale(normal, f.orientation / norm(normal));
        selection.insert({t, id, u, v, normal}, f.tol, end);
        if (any && t >= 0)
            return 1;
    }
    return 0;
}
// Re-expand a bicubic on a smaller UV rectangle. Bounding each term of the
// finite Taylor polynomial on [-1/2,1/2]^2 encloses its first derivatives.
__device__ __noinline__ bool restrict_cell(const Span &s, Range r, Ray ray, double end, double tol, Cell &c) {
    constexpr int choose[4][4] = {{1}, {1, 1}, {1, 2, 1}, {1, 3, 3, 1}};
    double du = (r.U - r.u) / (s.range.U - s.range.u), dv = (r.Vv - r.v) / (s.range.Vv - s.range.v);
    double x = ((r.u + r.U) * .5 - s.range.u) / (s.range.U - s.range.u),
           y = ((r.v + r.Vv) * .5 - s.range.v) / (s.range.Vv - s.range.v);
    double xp[4] = {1, x, x * x, x * x * x}, yp[4] = {1, y, y * y, y * y * y};
    double up[4] = {1, du, du * du, du * du * du}, vp[4] = {1, dv, dv * dv, dv * dv * dv};
    c = {};
    c.range = r;
    V extent{}, coeff[16]{};
    double pad = 0;
    for (int k = 0; k < 3; ++k) {
        double mag = 0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                mag += fabs(s.power[i * 4 + j].x[k]);
        double roundoff = 4096. * 2.2204460492503131e-16 * (1 + mag + fabs(ray.p.x[k]));
        pad += roundoff;
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b) {
                double term = 0;
                for (int i = a; i < 4; ++i)
                    for (int j = b; j < 4; ++j)
                        term += s.power[i * 4 + j].x[k] * choose[i][a] * choose[j][b] * xp[i - a] * yp[j - b];
                term *= up[a] * vp[b];
                coeff[4 * a + b].x[k] = term;
                if (!a && !b) {
                    c.center.x[k] = term;
                    continue;
                }
                double bound = ldexp(fabs(term), -a - b);
                extent.x[k] += bound;
                if (a + b == 1) {
                    if (a)
                        c.su.x[k] = term;
                    else
                        c.sv.x[k] = term;
                } else {
                    c.ru.x[k] += 2 * a * bound;
                    c.rv.x[k] += 2 * b * bound;
                }
            }
        extent.x[k] += roundoff;
        c.ru.x[k] += roundoff * du;
        c.rv.x[k] += roundoff * dv;
        c.box.lo.x[k] = c.center.x[k] - extent.x[k];
        c.box.hi.x[k] = c.center.x[k] + extent.x[k];
    }
    if (!intersects(c.box, ray, end, 2 * tol))
        return false;
    V helper = fabs(ray.d.x[0]) < .7 ? V{{1, 0, 0}} : V{{0, 1, 0}};
    V xaxis = cross(ray.d, helper);
    xaxis = scale(xaxis, 1 / norm(xaxis));
    V yaxis = cross(ray.d, xaxis);
    V tangent = norm(c.su) > norm(c.sv) ? c.su : c.sv;
    V normal = cross(ray.d, tangent);
    double n = norm(normal);
    V axes[3]{xaxis, yaxis, n > 1e-16 ? scale(normal, 1 / n) : xaxis};
    for (int k = 0; k < 3; ++k) {
        double radius = pad;
        for (int i = 1; i < 16; ++i)
            radius += ldexp(fabs(dot(coeff[i], axes[k])), -(i / 4 + i % 4));
        if (fabs(dot(sub(c.center, ray.p), axes[k])) > radius)
            return false;
    }
    return true;
}
__device__ __noinline__ int subdivide(View g, int id, const Span &s, Range domain, Ray ray, double &end,
                                      RootSelection &selection, bool any) {
    const auto &f = g.charts[id];
    Range pending[37];
    int depth[37];
    int n = 1;
    pending[0] = domain;
    depth[0] = 0;
    for (int visits = 0; n && visits < 8192; ++visits) {
        --n;
        Range r = pending[n];
        int level = depth[n];
        Cell c{};
        if (!restrict_cell(s, r, ray, end, f.tol, c))
            continue;
        double t, u, v;
        V normal;
        int state = isolate(s, c, ray, f.tol, end, t, normal, u, v);
        if (state == -1)
            continue;
        if (state == 0) {
            if (t > end || (any && t < 0))
                continue;
            int inside = face_contains(g, f, u, v);
            if (inside < 0)
                return -2;
            if (inside) {
                selection.insert({t, id, u, v, scale(normal, f.orientation)}, f.tol, end);
                if (any)
                    return 1;
            }
            continue;
        }
        if (level >= 35)
            return -2;
        Range l = r, h = r;
        if (norm(c.su) >= norm(c.sv)) {
            l.U = h.u = .5 * (r.u + r.U);
        } else {
            l.Vv = h.v = .5 * (r.v + r.Vv);
        }
        pending[n] = h;
        depth[n++] = level + 1;
        pending[n] = l;
        depth[n++] = level + 1;
    }
    return n ? -2 : 0;
}
__device__ __noinline__ int span_hits(View g, int face, int span, Ray ray, double &end, bool any,
                                      RootSelection &selection) {
    const auto &f = g.charts[face];
    const auto &s = g.patches[span];
    if (s.cell < 0)
        return -2;
    const auto &frame = g.frames[span];
    Ray cellray{basis(frame.axes, sub(ray.p, frame.origin)), basis(frame.axes, ray.d)};
    int k = 0;
    for (;;) {
        const auto &c = g.cells[s.cell + k];
        if (intersects(c.box, cellray, end, 2 * f.tol)) {
            double t, u, v;
            V normal;
            int code = isolate(s, c, ray, f.tol, end, t, normal, u, v);
            if (code == -2) {
                if (2 * k + 1 < s.count) {
                    k = 2 * k + 1;
                    continue;
                }
                if (fabs(dot(s.axis, ray.d)) >= 1e-12 || s.plane_width >= 1e-10) {
                    int pass = subdivide(g, face, s, c.range, ray, end, selection, any);
                    if (pass < 0)
                        return -3;
                    if (pass > 0)
                        return 1;
                }
            } else if (code == 0 && t <= end && (!any || t >= 0)) {
                int inside = face_contains(g, f, u, v);
                if (inside < 0)
                    return -4;
                if (inside) {
                    selection.insert({t, face, u, v, scale(normal, f.orientation)}, f.tol, end);
                    if (any && t >= 0)
                        return 1;
                }
            }
        }
        while (k && !(k & 1))
            k = (k - 1) / 2;
        if (!k)
            break;
        ++k;
    }
    return 0;
}
__device__ Hit final_hit(const RootSelection &selection, Ray ray, bool any) {
    if (!any && selection.hit.face >= 0 &&
        (selection.ambiguous || selection.hit.t <= 1e-7 || dot(selection.hit.normal, ray.d) <= 1e-8))
        return {0, -1};
    return selection.hit;
}
__device__ __noinline__ Hit first(View g, Ray ray, double end, bool any = false) {
    RootSelection selection;
    if (g.ray_grid && end <= g.ray_grid->reach && fabs(dot(ray.d, ray.d) - 1) < 1e-12) {
        const auto &cells = *g.ray_grid;
        int id = 0;
        bool in_grid = true;
        for (int k = 0; k < 3; ++k) {
            double index = floor((ray.p.x[k] - cells.low.x[k]) * cells.inverse_width.x[k]);
            if (!(index >= 0 && index < cells.size[k])) {
                in_grid = false;
                break;
            }
            id = id * cells.size[k] + int(index);
        }
        if (in_grid) {
            const int from = cells.offsets[id], to = cells.offsets[id + 1];
            if (to - from <= 96) {
                int previous = -1;
                bool active = false;
                Ray local{};
                for (int k = from; k < to; ++k) {
                    const auto primitive = cells.primitives[cells.ids[k]];
                    const int face = primitive.face;
                    const auto &f = g.charts[face];
                    if (face != previous) {
                        active = intersects(f.box, ray, end, 1e-6);
                        if (active && primitive.span_node >= 0)
                            local = {basis(f.frame, ray.p), basis(f.frame, ray.d)};
                        previous = face;
                    }
                    if (!active)
                        continue;
                    int code = 0;
                    if (primitive.span_node < 0) {
                        code = analytic_hits(g, face, ray, end, selection, any);
                        if (code < 0)
                            return {0, -2, double(face), 1};
                    } else {
                        const auto &node = g.spans[primitive.span_node];
                        if (intersects(node.box, local, end, 2 * f.tol))
                            code = span_hits(g, face, node.leaf, ray, end, any, selection);
                        if (code < 0)
                            return {0, -2, double(face), double(-code)};
                    }
                    if (code > 0)
                        return selection.hit;
                }
                return final_hit(selection, ray, any);
            }
        }
    }
    for (int i = 0; i < g.n;) {
        const auto &n = g.faces[i];
        if (!intersects(n.box, ray, end, 1e-6)) {
            i = n.next;
            continue;
        }
        ++i;
        if (n.leaf < 0)
            continue;
        const auto &f = g.charts[n.leaf];
        if (f.first < 0) {
            int code = analytic_hits(g, n.leaf, ray, end, selection, any);
            if (code < 0)
                return {0, -2, double(n.leaf), 1};
            if (code > 0)
                return selection.hit;
            continue;
        }
        Ray local{basis(f.frame, ray.p), basis(f.frame, ray.d)};
        for (int j = f.first; j < f.end;) {
            const auto &node = g.spans[j];
            if (!intersects(node.box, local, end, 2 * f.tol)) {
                j = node.next;
                continue;
            }
            ++j;
            if (node.leaf < 0)
                continue;
            int code = span_hits(g, n.leaf, node.leaf, ray, end, any, selection);
            if (code < 0)
                return {0, -2, double(n.leaf), double(-code)};
            if (code > 0)
                return selection.hit;
        }
    }
    return final_hit(selection, ray, any);
}
