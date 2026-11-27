// Binary64 interval operations and trim winding follow interval.hpp and geometry.cpp.
// This path is used when the cheaper UV derivative enclosure straddles a boundary.
__device__ bool singleton(Bound a, double x) { return a.lo == x && a.hi == x; }
__device__ Bound exact_value(double x) { return {x, x}; }
__device__ Bound negative(Bound x) { return {-x.hi, -x.lo}; }
__device__ Bound add_bound(Bound x, Bound y) {
    if (singleton(x, 0))
        return y;
    if (singleton(y, 0))
        return x;
    if (x.lo == x.hi && y.lo == y.hi && x.lo == -y.lo)
        return {0, 0};
    return {nextafter(x.lo + y.lo, -DBL_MAX), nextafter(x.hi + y.hi, DBL_MAX)};
}
__device__ Bound minus(Bound x, Bound y) { return add_bound(x, negative(y)); }
__device__ Bound outward(double a, double b, double c, double d) {
    return {nextafter(fmin(fmin(a, b), fmin(c, d)), -DBL_MAX),
            nextafter(fmax(fmax(a, b), fmax(c, d)), DBL_MAX)};
}
__device__ Bound times(Bound x, Bound y) {
    if (singleton(x, 0) || singleton(y, 0))
        return {0, 0};
    if (singleton(x, 1))
        return y;
    if (singleton(y, 1))
        return x;
    if (singleton(x, -1))
        return negative(y);
    if (singleton(y, -1))
        return negative(x);
    return outward(x.lo * y.lo, x.lo * y.hi, x.hi * y.lo, x.hi * y.hi);
}
__device__ Bound divide_bound(Bound x, Bound y) {
    if (y.lo <= 0 && y.hi >= 0)
        return {-HUGE_VAL, HUGE_VAL};
    if (singleton(x, 0))
        return {0, 0};
    if (singleton(y, 1))
        return x;
    if (singleton(y, -1))
        return negative(x);
    return outward(x.lo / y.lo, x.lo / y.hi, x.hi / y.lo, x.hi / y.hi);
}
__device__ Bound unite(Bound a, Bound b) { return {fmin(a.lo, b.lo), fmax(a.hi, b.hi)}; }
__device__ Bound component(Bounds b, int k) { return {b.lo.x[k], b.hi.x[k]}; }
__device__ Bound mix_bound(Bound a, Bound b, Bound t) {
    if (singleton(t, 0))
        return a;
    if (singleton(t, 1))
        return b;
    return add_bound(times(minus({1, 1}, t), a), times(t, b));
}
struct TrigPair {
    Bound s, c;
};
__device__ TrigPair point_trig(double a) {
    double y = .25 * a, z = -y * y, st = y, ct = 1, s = y, c = 1;
    for (int i = 1; i <= 10; ++i) {
        st = st * z / ((2 * i) * (2 * i + 1));
        ct = ct * z / ((2 * i - 1) * (2 * i));
        s += st;
        c += ct;
    }
    for (int i = 0; i < 2; ++i) {
        double next = 2 * s * c;
        c = c * c - s * s;
        s = next;
    }
    constexpr double e = 0x1p-38;
    return {add_bound({s, s}, {-e, e}), add_bound({c, c}, {-e, e})};
}
__device__ bool intersect_bound(Bound a, Bound b) { return a.hi >= b.lo && a.lo <= b.hi; }
__device__ TrigPair bounded_trig(Bound x) {
    if (singleton(x, 0))
        return {{0, 0}, {1, 1}};
    TrigPair all{{-1, 1}, {-1, 1}};
    if (!isfinite(x.lo) || !isfinite(x.hi) || x.hi - x.lo >= 2)
        return all;
    constexpr Bound twopi{0x1.921fb54442d18p+2, 0x1.921fb54442d19p+2},
        pi{0x1.921fb54442d18p+1, 0x1.921fb54442d19p+1}, half{0x1.921fb54442d18p+0, 0x1.921fb54442d19p+0};
    double turns = round(.5 * (x.lo + x.hi) / twopi.lo);
    if (fabs(turns) > 1e10)
        return all;
    x = minus(x, times(exact_value(turns), twopi));
    if (fmax(fabs(x.lo), fabs(x.hi)) > 4.25)
        return all;
    auto a = point_trig(x.lo), b = x.lo == x.hi ? a : point_trig(x.hi);
    auto s = unite(a.s, b.s), c = unite(a.c, b.c);
    if (intersect_bound(x, half))
        s.hi = 1;
    if (intersect_bound(x, negative(half)))
        s.lo = -1;
    if (intersect_bound(x, pi) || intersect_bound(x, negative(pi)))
        c.lo = -1;
    if (intersect_bound(x, {0, 0}))
        c.hi = 1;
    return {{fmax(-1., s.lo), fmin(1., s.hi)}, {fmax(-1., c.lo), fmin(1., c.hi)}};
}
constexpr int trim_order_limit = 32;
struct UVBox {
    Bound x, y;
};
__device__ __noinline__ UVBox trim_box(const View &g, const TrimPiece &e, double lo, double hi) {
    const auto *p = g.trim_intervals + e.start;
    if (!e.count) {
        Bound start = exact_value(e.angle);
        auto sc = bounded_trig(add_bound(start, times(minus(exact_value(e.last_angle), start), {lo, hi})));
        auto axis = [&](int k) {
            return add_bound(add_bound(component(p[0], k), times(component(p[1], k), sc.c)),
                             times(component(p[2], k), sc.s));
        };
        return {axis(0), axis(1)};
    }
    // Form the left control polygon at hi, then the right control polygon at lo/hi.
    // Restricting the original span on every visit reproduces the CPU interval widths.
    Bound cart[3][trim_order_limit], work[trim_order_limit];
    for (int k = 0; k < 3; ++k) {
        for (int j = 0; j < e.count; ++j)
            work[j] = component(p[j], k);
        if (lo == hi) {
            for (int n = e.count; n > 1; --n)
                for (int j = 0; j < n - 1; ++j)
                    work[j] = mix_bound(work[j], work[j + 1], {lo, lo});
            cart[k][0] = work[0];
            continue;
        }
        if (hi == 1) {
            for (int j = 0; j < e.count; ++j)
                cart[k][j] = work[j];
        } else {
            cart[k][0] = work[0];
            for (int n = e.count - 1; n > 0; --n) {
                for (int j = 0; j < n; ++j)
                    work[j] = mix_bound(work[j], work[j + 1], {hi, hi});
                cart[k][e.count - n] = work[0];
            }
        }
        if (lo > 0) {
            Bound t = hi == 1 ? exact_value(lo) : divide_bound({lo, lo}, {hi, hi});
            t = {fmax(0., t.lo), fmin(1., t.hi)};
            for (int j = 0; j < e.count; ++j)
                work[j] = cart[k][j];
            for (int n = e.count - 1; n > 0; --n) {
                for (int j = 0; j < n; ++j)
                    work[j] = mix_bound(work[j], work[j + 1], t);
                cart[k][n - 1] = work[n - 1];
            }
        }
    }
    UVBox box{divide_bound(cart[0][0], cart[2][0]), divide_bound(cart[1][0], cart[2][0])};
    if (lo != hi)
        for (int i = 1; i < e.count; ++i) {
            box.x = unite(box.x, divide_bound(cart[0][i], cart[2][i]));
            box.y = unite(box.y, divide_bound(cart[1][i], cart[2][i]));
        }
    return box;
}
__device__ int ray_side(Bound y, double level) { return y.hi <= level ? 0 : (y.lo > level ? 1 : -1); }
// Return 1 for the CPU's closed boundary, 0 for a winding count, -1 for unsupported input.
__device__ __noinline__ int interval_winding(const View &g, const TrimPiece &edge, double u, double v,
                                             int &winding) {
    if (edge.count > trim_order_limit)
        return -1;
    unsigned cell = 1;
    int depth = 0;
    for (;;) {
        const double width = ldexp(1., -depth), left = cell * width - 1, right = left + width;
        auto box = trim_box(g, edge, left, right);
        if (!isfinite(box.x.lo) || !isfinite(box.x.hi) || !isfinite(box.y.lo) || !isfinite(box.y.hi))
            return -1;
        if (box.x.hi >= u && box.y.lo <= v && box.y.hi >= v) {
            if (box.x.lo > u) {
                int first = ray_side(trim_box(g, edge, left, left).y, v);
                int last = ray_side(trim_box(g, edge, right, right).y, v);
                if (first < 0 || last < 0)
                    return 1;
                winding += last - first;
            } else if (depth == 22)
                return 1;
            else {
                cell *= 2;
                ++depth;
                continue;
            }
        }
        while (cell > 1 && (cell & 1)) {
            cell /= 2;
            --depth;
        }
        if (cell == 1)
            return 0;
        ++cell;
    }
}
