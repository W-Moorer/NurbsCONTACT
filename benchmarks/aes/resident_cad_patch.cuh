// Six-node quadratic interpolation of the native face and common trim curve.
__device__ P cad_vec(double4 v) { return {v.x, v.y, v.z}; }
__device__ P cad_cross(P a, P b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
__device__ double2 triangle_uv(P ab, P ac, P ap) {
    const double d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0 && d2 <= 0)
        return {0, 0};
    const P bp = sub(ap, ab);
    const double d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0 && d4 <= d3)
        return {1, 0};
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0)
        return {d1 / (d1 - d3), 0};
    const P cp = sub(ap, ac);
    const double d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0 && d5 <= d6)
        return {0, 1};
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0)
        return {0, d2 / (d2 - d6)};
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0 && d4 >= d3 && d5 >= d6) {
        const double v = (d4 - d3) / (d4 - d3 + d5 - d6);
        return {1 - v, v};
    }
    return {vb / (va + vb + vc), vc / (va + vb + vc)};
}
struct CadPatch {
    P a, ab, ac, e, f, g;
    __device__ P value(double u, double v) const {
        double w = 1 - u - v;
        P p = along(along(a, u, ab), v, ac);
        return along(along(along(p, 4 * w * u, e), 4 * u * v, f), 4 * v * w, g);
    }
    __device__ void tangents(double u, double v, P &du, P &dv) const {
        double w = 1 - u - v;
        du = along(along(ab, 4 * (w - u), e), 4 * v, sub(f, g));
        dv = along(along(ac, 4 * (w - v), g), 4 * u, sub(f, e));
    }
};
struct CadWitness {
    P delta, normal;
    double sq;
    int feature;
};
__device__ CadPatch read_patch(const P *t, const double4 *c) {
    return {t[0], sub(t[1], t[0]), sub(t[2], t[0]), cad_vec(c[0]), cad_vec(c[1]), cad_vec(c[2])};
}
struct PatchMinimum {
    double sq;
    int id;
    double2 uv;
};
__device__ PatchMinimum project_patch(P p, int id, const P *t, const double4 *c) {
    CadPatch s = read_patch(t, c);
    double2 uv = triangle_uv(s.ab, s.ac, sub(p, s.a));
    P q = s.value(uv.x, uv.y), delta = sub(q, p);
    double sq = dot(delta, delta);
    if (c[0].w > 1.e-10) {
        for (int iter = 0; iter < 6; ++iter) {
            P du, dv;
            s.tangents(uv.x, uv.y, du, dv);
            // Closest point on the tangent triangle solves the constrained
            // Gauss-Newton subproblem, including all three trim segments.
            P rhs = along(along(sub(p, q), uv.x, du), uv.y, dv);
            double2 next = triangle_uv(du, dv, rhs);
            P nq = s.value(next.x, next.y), nd = sub(nq, p);
            double ns = dot(nd, nd);
            P step = sub(nq, q);
            if (dot(step, step) < 1.e-20)
                break;
            for (int k = 0; ns > sq && k < 3; ++k) {
                next = {(next.x + uv.x) * .5, (next.y + uv.y) * .5};
                nq = s.value(next.x, next.y);
                nd = sub(nq, p);
                ns = dot(nd, nd);
            }
            if (!(ns < sq))
                break;
            P update = sub(nq, q);
            uv = next;
            q = nq;
            delta = nd;
            sq = ns;
            if (dot(update, update) < 1.e-20)
                break;
        }
    }
    return {sq, id, uv};
}
__device__ CadWitness patch_witness(P p, CadPatch s, PatchMinimum m) {
    auto uv = m.uv;
    const double tol = 1.e-10;
    bool a = uv.x < tol, b = uv.y < tol, h = 1 - uv.x - uv.y < tol;
    int feature = a && b ? 4 : b && h ? 5 : h && a ? 6 : b ? 1 : h ? 2 : a ? 3 : 0;
    P du, dv;
    s.tangents(uv.x, uv.y, du, dv);
    P n = cad_cross(du, dv);
    return {sub(s.value(uv.x, uv.y), p), n, m.sq, feature};
}
__device__ double patch_lower(P p, int j, const P *tri, const double4 *controls) {
    int region;
    P d = sub(closest(p, tri + 3 * j, region), p);
    return fmax(0., sqrt(dot(d, d)) - controls[3 * j].w - 1.e-9);
}
__device__ void patch_test(P p, int j, const P *tri, const double4 *c, PatchMinimum &best) {
    auto candidate = project_patch(p, j, tri + 3 * j, c + 3 * j);
    if (candidate.sq < best.sq || (candidate.sq == best.sq && j < best.id))
        best = candidate;
}
template <int W> __device__ PatchMinimum closest_patch_group(PatchMinimum m) {
    if constexpr (W > 1) {
        unsigned group = query_mask<W>();
        Best closest = nearest_group<W>({m.sq, m.id});
        unsigned owners = __ballot_sync(group, m.id == closest.id && m.sq == closest.sq);
        int owner = (__ffs(owners) - 1) % W;
        m.uv.x = __shfl_sync(group, m.uv.x, owner, W);
        m.uv.y = __shfl_sync(group, m.uv.y, owner, W);
        m.id = closest.id;
        m.sq = closest.sq;
    }
    return m;
}
template <int W, bool countStages>
__device__ void curved_contact(P p, int i, int cell, int ix, int iy, int iz, const double *pose, GEOM,
                               unsigned long long *stages) {
    const int lane = threadIdx.x % W;
    const double4 previous = outside[i];
    P change = sub(p, cad_vec(previous));
    if (previous.w > 0 && dot(change, change) < previous.w * previous.w)
        return;
    const float4 plane = planes[cell];
    double h = 1 / bounds[6];
    double sheet = plane.w + plane.x * (p.x - bounds[0] - (ix + .5) * h) +
                   plane.y * (p.y - bounds[1] - (iy + .5) * h) + plane.z * (p.z - bounds[2] - (iz + .5) * h);
    if (sheet > 0.) {
        if (!lane)
            outside[i] = make_double4(p.x, p.y, p.z, sheet);
        return;
    }
    count_search_stage<W, countStages>(stages, 3);
    double4 prev = nearby[i];
    PatchMinimum best = {1.e300, 0x7fffffff, {0, 0}};
    double exclusion = 0.;
    bool reused = false;
#ifndef NC_NO_PATCH_REUSE
    if (prev.w > 0) {
        change = sub(p, cad_vec(prev));
        double travel = sqrt(dot(change, change));
        if (travel < prev.w) {
            int j = neighbors[W * i];
            patch_test(p, j, tri, controls, best);
            exclusion = prev.w - travel;
            reused = sqrt(best.sq) < exclusion;
        }
    }
#endif
    if (reused)
        count_search_stage<W, countStages>(stages, 5);
    if (!reused) {
        // The carrier distance +/- its displacement bounds the patch
        // distance. Find an inexpensive upper bound before any curved solve.
        double limit = sqrt(best.sq);
        double smallest = 1.e300, second = 1.e300;
        int first = -1;
#ifdef NC_DIRECT_PATCH_SEARCH
        int seed = seeds[cell];
        if (seed >= 0) {
            patch_test(p, seed, tri, controls, best);
            limit = sqrt(best.sq);
        }
        for (int k = offsets[cell] + lane; k < ends[cell]; k += W) {
            int j = ids[k];
            double lower = patch_lower(p, j, tri, controls);
            if (lower < smallest) {
                second = smallest;
                smallest = lower;
                first = j;
            } else
                second = fmin(second, lower);
            if (j != seed && lower <= limit) {
                patch_test(p, j, tri, controls, best);
                limit = sqrt(best.sq);
            }
        }
#else
#ifdef NC_CARRIER_FILTER
        Best flatFirst = {1.e300, 0x7fffffff};
        double flatSecond = 1.e300;
#endif
        for (int k = offsets[cell] + lane; k < ends[cell]; k += W) {
            int j = ids[k];
            double d = sqrtf(quick_distance(p, axes + 4 * j));
            double pad = controls[3 * j].w + fpPad;
#ifdef NC_CARRIER_FILTER
            if (d < flatFirst.sq) {
                flatSecond = flatFirst.sq;
                flatFirst = {d, j};
            } else
                flatSecond = fmin(d, flatSecond);
#endif
            double lower = fmax(0., d - pad);
            if (lower < smallest) {
                second = smallest;
                smallest = lower;
                first = j;
            } else
                second = fmin(second, lower);
            limit = fmin(limit, d + pad);
        }
#ifdef NC_CARRIER_FILTER
        // Deforming the flat carrier continuously into the quadratic sheet
        // cannot change inside/outside at a point outside every displacement
        // envelope. Only use a locally complete and unambiguous flat witness.
        Best flatOwner = nearest_group<W>(flatFirst);
        double runner = flatFirst.id == flatOwner.id ? flatSecond : flatFirst.sq;
        double separation = fmin(smallest, .866025403784438 * h);
        if constexpr (W > 1) {
            unsigned group = query_mask<W>();
            for (int k = W / 2; k; k /= 2) {
                runner = fmin(runner, __shfl_xor_sync(group, runner, k, W));
                separation = fmin(separation, __shfl_xor_sync(group, separation, k, W));
            }
        }
        if (separation > 0. && flatOwner.sq + 2 * fpPad < runner &&
            flatOwner.sq + fpPad < bounds[7] - 2 * bounds[8] - 1.e-8) {
            int j = flatOwner.id, feature;
            P delta = sub(closest(p, tri + 3 * j, feature), p);
            if (dot(delta, features[7 * j + feature]) < 0.) {
                if (!lane)
                    outside[i] = make_double4(p.x, p.y, p.z, separation);
                return;
            }
        }
#endif
        if constexpr (W > 1) {
            unsigned group = query_mask<W>();
            for (int k = W / 2; k; k /= 2)
                limit = fmin(limit, __shfl_xor_sync(group, limit, k, W));
        }
        for (int k = offsets[cell] + lane; k < ends[cell]; k += W) {
            int j = ids[k];
            double lower = sqrtf(quick_distance(p, axes + 4 * j)) - controls[3 * j].w - fpPad;
            if (lower <= limit) {
                patch_test(p, j, tri, controls, best);
                limit = fmin(limit, sqrt(best.sq));
            }
        }
#endif
        best = closest_patch_group<W>(best);
        exclusion = fmin(first == best.id ? second : smallest, .866025403784438 * h);
        if constexpr (W > 1) {
            unsigned group = query_mask<W>();
            for (int k = W / 2; k; k /= 2)
                exclusion = fmin(exclusion, __shfl_xor_sync(group, exclusion, k, W));
        }
        if (sqrt(best.sq) >= bounds[7] - bounds[8] - 1.e-8) {
            count_search_stage<W, countStages>(stages, 4);
            // Cells only store a finite band. Deep interior witnesses still
            // use the complete tree, with boxes expanded by the patch bound.
            for (int j = 0; j < nodes[0].x;) {
                int4 node = nodes[j];
                double reach = sqrt(best.sq) + bounds[8] + 1.e-8;
                if (point_box(p, boxes + 6 * j) > reach * reach) {
                    j = node.x;
                    continue;
                }
                ++j;
                if (node.y < 0)
                    continue;
                for (int k = node.y + lane; k < node.z; k += W)
                    if (patch_lower(p, k, tri, controls) <= reach)
                        patch_test(p, k, tri, controls, best);
                best = closest_patch_group<W>(best);
            }
            exclusion = 0.;
        }
        if (!lane) {
            nearby[i] = make_double4(p.x, p.y, p.z, exclusion);
            neighbors[W * i] = best.id;
        }
    }
    if (lane || best.id == 0x7fffffff)
        return;
    int j = best.id;
    auto hit = patch_witness(p, read_patch(tri + 3 * j, controls + 3 * j), best);
    if (hit.sq <= NC_DEPTH_FLOOR_MM * NC_DEPTH_FLOOR_MM)
        return;
    P side = hit.feature ? features[7 * j + hit.feature] : hit.normal;
    if (dot(hit.delta, side) <= 0) {
        double clear = fmin(exclusion, patch_lower(p, j, tri, controls));
        outside[i] = make_double4(p.x, p.y, p.z, clear);
        return;
    }
    double depth = sqrt(hit.sq);
    P dir = hit.feature ? hit.delta : hit.normal;
    double length = sqrt(dot(dir, dir));
    if (!(length > 0))
        return;
    P unit = {dir.x / length, dir.y / length, dir.z / length};
    double f = dot(unit, rotate(pose, normals[i])),
           o = fb[i] == alt[i] ? f : dot(unit, rotate(pose, other[i]));
    if (fmin(f, o) < -1.e-10) {
        count_search_stage<W, countStages>(stages, 6);
        contact_hit(i, fa[j], o < f ? alt[i] : fb[i], depth, p, unit, fmin(f, o), keys, peaks, mask, slots,
                    count, witness);
    }
}
