// Compiled together with gpu_triangle_intersections.cu for its finite-triangle predicates.
namespace {
__device__ bool distant(V axis, const V *a, const V *b, double skin) {
    double a0 = proj(axis, a[0]), a1 = proj(axis, a[1]), a2 = proj(axis, a[2]);
    double b0 = proj(axis, b[0]), b1 = proj(axis, b[1]), b2 = proj(axis, b[2]);
    double gap = fmax(min3(a0, a1, a2) - max3(b0, b1, b2), min3(b0, b1, b2) - max3(a0, a1, a2));
    return gap > 0 && gap * gap > skin * skin * proj(axis, axis);
}
__device__ bool nearby(const V *a, const V *b, double skin) {
    V ea[3], eb[3];
    for (int j = 0; j < 3; j++) {
        ea[j] = minus_v(a[(j + 1) % 3], a[j]);
        eb[j] = minus_v(b[(j + 1) % 3], b[j]);
    }
    V na = cross_v(ea[0], ea[1]), nb = cross_v(eb[0], eb[1]);
    if (distant(na, a, b, skin) || distant(nb, a, b, skin))
        return false;
    for (int j = 0; j < 3; j++) {
        if (distant(cross_v(na, ea[j]), a, b, skin) || distant(cross_v(nb, eb[j]), a, b, skin))
            return false;
        for (int k = 0; k < 3; k++)
            if (distant(cross_v(ea[j], eb[k]), a, b, skin))
                return false;
    }
    return true;
}
} // namespace
extern "C" __global__ void window_pairs(const V *source, int n, const V *target, const int *ids,
                                        const double *boxes, const int *links, int nodes,
                                        const double *pose, double skin, int start, int *counts,
                                        const int *offsets, int2 *candidates, int write) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, bin = start + blockIdx.y;
    if (i >= n)
        return;
    V a[3];
    for (int j = 0; j < 3; j++)
        a[j] = transform(pose + 12 * bin, source[3 * i + j]);
    V lo = make_double3(min3(a[0].x, a[1].x, a[2].x) - skin, min3(a[0].y, a[1].y, a[2].y) - skin,
                        min3(a[0].z, a[1].z, a[2].z) - skin);
    V hi = make_double3(max3(a[0].x, a[1].x, a[2].x) + skin, max3(a[0].y, a[1].y, a[2].y) + skin,
                        max3(a[0].z, a[1].z, a[2].z) + skin);
    for (int k = 0; k < nodes;) {
        const int *link = links + 3 * k;
        const double *box = boxes + 6 * k;
        if (hi.x < box[0] || hi.y < box[1] || hi.z < box[2] || lo.x > box[3] || lo.y > box[4] ||
            lo.z > box[5]) {
            k = link[0];
            continue;
        }
        for (int j = link[1]; j < link[1] + link[2]; j++)
            if (nearby(a, target + 3 * j, skin)) {
                int pos = atomicAdd(counts + bin, 1);
                if (write)
                    candidates[offsets[bin] + pos] = make_int2(i, ids[j]);
            }
        ++k;
    }
}

namespace {
struct HingePair {
    double ax, ay, az, bx, by, bz, sa, sb, oa, ob, period, ratio, origin, band, stiff;
};
struct RelativePose {
    double c, s, x, y, z;
};
__device__ RelativePose relative_pose(const HingePair &p, double a, double b) {
    double ca, sa, cb, sb;
    sincos(p.sa * (a + p.oa), &sa, &ca);
    sincos(p.sb * (b + p.ob), &sb, &cb);
    double x = p.ax - p.bx, y = p.ay - p.by;
    return {ca * cb + sa * sb, sa * cb - ca * sb, cb * x + sb * y, cb * y - sb * x, p.az - p.bz};
}
__device__ V apply(RelativePose m, V p) {
    return make_double3(m.c * p.x - m.s * p.y + m.x, m.s * p.x + m.c * p.y + m.y, p.z + m.z);
}
__device__ bool crosses(int2 pair, const V *a, const V *b, RelativePose m) {
    V tri[3];
    for (int i = 0; i < 3; i++)
        tri[i] = apply(m, a[3 * pair.x + i]);
    const V *other = b + 3 * pair.y;
    if (max3(tri[0].z, tri[1].z, tri[2].z) < min3(other[0].z, other[1].z, other[2].z) ||
        max3(other[0].z, other[1].z, other[2].z) < min3(tri[0].z, tri[1].z, tri[2].z))
        return false;
    // Orthogonal projections provide cheap separating axes for thin facets and caps.
    // Translate to a common origin to retain the SAT predicate's scale.
    V u[3], v[3];
    for (int j = 0; j < 3; j++) {
        u[j] = minus_v(tri[j], tri[0]);
        v[j] = minus_v(other[j], tri[0]);
    }
    for (int j = 0; j < 3; j++) {
        V e = minus_v(u[(j + 1) % 3], u[j]), f = minus_v(v[(j + 1) % 3], v[j]);
        if (separates(make_double3(-e.y, e.x, 0), u, v) || separates(make_double3(-f.y, f.x, 0), u, v))
            return false;
    }
    if (tri[0].z == tri[1].z && tri[1].z == tri[2].z && other[0].z == other[1].z &&
        other[1].z == other[2].z)
        return true;
    return intersect(tri, other);
}
} // namespace
// Rank finite witnesses by their margin in the XY projection. A large interior
// overlap of coplanar faces makes a useful seed even after the hinges move.
__device__ double seed_margin(int2 pair, const V *a, const V *b, RelativePose m) {
    V u[3], v[3];
    for (int j = 0; j < 3; j++) {
        u[j] = apply(m, a[3 * pair.x + j]);
        v[j] = b[3 * pair.y + j];
    }
    if (u[0].z != u[1].z || u[1].z != u[2].z || v[0].z != v[1].z || v[1].z != v[2].z || u[0].z != v[0].z)
        return 0;
    double margin = 1.;
    for (int side = 0; side < 2; side++)
        for (int j = 0; j < 3; j++) {
            const V *tri = side ? u : v;
            V edge = minus_v(tri[(j + 1) % 3], tri[j]), n = make_double3(-edge.y, edge.x, 0);
            double x0 = proj(n, u[0]), x1 = proj(n, u[1]), x2 = proj(n, u[2]);
            double y0 = proj(n, v[0]), y1 = proj(n, v[1]), y2 = proj(n, v[2]);
            double overlap =
                fmin(max3(x0, x1, x2) - min3(y0, y1, y2), max3(y0, y1, y2) - min3(x0, x1, x2));
            margin = fmin(margin, overlap / norm_v(n));
        }
    return margin;
}
// Each seed is an actual triangle pair in the conservative list. Failure of
// all seeds always falls back to the entire list in the same configuration.
extern "C" __global__ void seed_windows(const HingePair *params, int bins, const int *offsets,
                                        const int2 *pairs, const V *mesh_a, const V *mesh_b, int *seeds) {
    int bin = blockIdx.x, lane = threadIdx.x;
    const HingePair p = *params;
    int first = offsets[bin], last = offsets[bin + 1];
    for (int slot = 0; slot < 8; slot++) {
        double qa = (bin + ((slot % 3) - 1) * .40) * p.period / bins;
        double qb = qa * p.ratio + p.origin + (slot / 3 - 1) * .5 * p.band;
        RelativePose m = relative_pose(p, qa, qb);
        int found = -1;
        double best = -1;
        for (int i = first + lane; i < last; i += blockDim.x) {
            if (!crosses(pairs[i], mesh_a, mesh_b, m))
                continue;
            double score = seed_margin(pairs[i], mesh_a, mesh_b, m);
            if (score > best) {
                found = i;
                best = score;
            }
        }
        for (int shift = 16; shift; shift /= 2) {
            int other = __shfl_down_sync(0xffffffff, found, shift);
            double merit = __shfl_down_sync(0xffffffff, best, shift);
            if (merit > best) {
                found = other;
                best = merit;
            }
        }
        if (lane % 32 == 0)
            seeds[bin * 32 + slot * 4 + lane / 32] = found;
    }
}

namespace {
struct DeviceStage {
    const HingePair *p;
    const V *a;
    const V *b;
    const int2 *pairs;
    const int *offsets;
    const int *seeds;
    const double *map;
    const int *anchor_offsets;
    const V *anchors;
};
// One warp per pair. The exact Boolean gate has no sampled-point tolerance.
__device__ bool has_contact(DeviceStage d, int bin, RelativePose pose, int *cached, int *work) {
    const int lane = threadIdx.x % 32;
    int first = d.offsets[bin], last = d.offsets[bin + 1];
    int slot = d.seeds[bin * 32 + lane];
    if (lane == 0 && *cached >= 0)
        slot = *cached;
    bool hit = slot >= 0 && crosses(d.pairs[slot], d.a, d.b, pose);
    unsigned votes = __ballot_sync(0xffffffff, hit);
    if (votes) {
        int i = __shfl_sync(0xffffffff, slot, __ffs(votes) - 1);
        if (lane == 0)
            *cached = i;
        return true;
    }
    for (int i = first + lane; i - lane < last; i += 32) {
        hit = i < last && crosses(d.pairs[i], d.a, d.b, pose);
        votes = __ballot_sync(0xffffffff, hit);
        if (lane == 0)
            (*work)++;
        if (votes) {
            int id = __shfl_sync(0xffffffff, i, __ffs(votes) - 1);
            if (lane == 0)
                *cached = id;
            return true;
        }
    }
    return false;
}
// All groups reconverge here, including when only one pair needs the fallback.
__device__ bool parallel_contact(DeviceStage d, int bin, RelativePose pose, int *cache, int *work) {
#if NC_WARPS == 1
    return has_contact(d, bin, pose, cache, work);
#else
    constexpr int width = 32 * NC_WARPS;
    int local = threadIdx.x % width, group = threadIdx.x / width, lane = local % 32;
    __shared__ int match[NC_PAIRS];
    if (local == 0)
        match[group] = -1;
    __syncthreads();
    if (local < 32) {
        int index = d.seeds[bin * 32 + lane];
        if (lane == 0 && *cache >= 0)
            index = *cache;
        unsigned votes = __ballot_sync(0xffffffff, index >= 0 && crosses(d.pairs[index], d.a, d.b, pose));
        if (votes) {
            int witness = __shfl_sync(0xffffffff, index, __ffs(votes) - 1);
            if (lane == 0)
                match[group] = witness;
        }
    }
    __syncthreads();
    int last = d.offsets[bin + 1];
    for (int first = d.offsets[bin];; first += width) {
        bool search = match[group] < 0 && first < last;
        if (!__syncthreads_or(search))
            break;
        if (search) {
            int i = first + local;
            unsigned votes = __ballot_sync(0xffffffff, i < last && crosses(d.pairs[i], d.a, d.b, pose));
            if (votes && lane == 0)
                atomicCAS(match + group, -1, first + local + __ffs(votes) - 1);
            if (local == 0)
                *work += NC_WARPS;
        }
        __syncthreads();
    }
    int witness = match[group];
    if (local == 0)
        *cache = witness;
    __syncthreads();
    return witness >= 0;
#endif
}
// Return [load, shaft A torque, shaft B torque] and their two angle derivatives.
__device__ void contact_map(DeviceStage d, double qa, double qb, int bins, int phases, int layers,
                            double slip0, double ds, int method, int *cached, int *work, double *response,
                            int *status) {
    const HingePair p = *d.p;
    const int lane = threadIdx.x % 32;
    double u = qa / p.period;
    u -= floor(u);
    double slip = qb - p.ratio * qa - p.origin;
    double y = (slip - slip0) / ds;
    if (!(fabs(slip) <= p.band && y >= 0 && y <= layers - 1)) {
        if (lane == 0)
            atomicCAS(status, 0, 1);
        return;
    }
    bool active = method == 0;
    if (lane >= 3)
        return;
    double value = 0, da = 0, db = 0;
    if (active) {
        double x = u * phases;
        int i = (int)floor(x), j = min((int)floor(y), layers - 2);
        double fx = x - i, fy = y - j;
        int k0 = (i * layers + j) * 3 + lane, k1 = (((i + 1) % phases) * layers + j) * 3 + lane;
        double v00 = d.map[k0], v01 = d.map[k0 + 3], v10 = d.map[k1], v11 = d.map[k1 + 3];
        db = ((1 - fx) * (v01 - v00) + fx * (v11 - v10)) / ds;
        double vi = v00 + fy * (v01 - v00), vj = v10 + fy * (v11 - v10);
        value = vi + fx * (vj - vi);
        da = (vj - vi) * phases / p.period - p.ratio * db;
        if (!isfinite(value + da + db))
            atomicCAS(status, 0, 2);
    }
    response[lane] = p.stiff * value;
    response[3 + lane] = p.stiff * da;
    response[6 + lane] = p.stiff * db;
}
// Each of four neighboring configurations supplies material surface points and normals.
// Transport and apply the shared penalty separately in each physical contact region.
__device__ V rotz(V v, double c, double s) {
    return make_double3(c * v.x - s * v.y, s * v.x + c * v.y, v.z);
}
__device__ void anchor_response(DeviceStage d, double qa, double qb, int phases, int layers, double slip0,
                                double ds, double source_a, double source_b, bool active,
                                double *response, int *status) {
    const int lane = threadIdx.x % 32;
    const HingePair p = *d.p;
    double u = source_a / p.period;
    u = (u - floor(u)) * phases;
    double slip = qb - p.ratio * qa - p.origin, source_slip = source_b - p.ratio * source_a - p.origin;
    double y = (source_slip - slip0) / ds;
    if (!(fabs(slip) <= p.band && y >= 0 && y <= layers - 1)) {
        if (lane == 0)
            atomicCAS(status, 0, 1);
        return;
    }
    double result[9] = {};
    if (active && lane < 4) {
        int i = (int)floor(u), j = min((int)floor(y), layers - 2);
        double fx = u - i, fy = y - j;
        int sample = ((i + lane / 2) % phases) * layers + j + lane % 2;
        double weight = (lane / 2 ? fx : 1 - fx) * (lane % 2 ? fy : 1 - fy);
        if (weight > 0 && !isfinite(d.map[3 * sample]))
            atomicCAS(status, 0, 2);
        double ca, sa, cb, sb;
        sincos(p.sa * (qa + p.oa), &sa, &ca);
        sincos(p.sb * (qb + p.ob), &sb, &cb);
        V between = make_double3(p.ax - p.bx, p.ay - p.by, p.az - p.bz);
        int loaded = 0;
        for (int k = d.anchor_offsets[sample]; k < d.anchor_offsets[sample + 1]; k++) {
            const V *anchor = d.anchors + 4 * k;
            V a = rotz(anchor[0], ca, sa), b = rotz(anchor[2], cb, sb);
            V na = rotz(anchor[1], ca, sa), nb = rotz(anchor[3], cb, sb);
            V n = plus_v(na, nb);
            double norm = norm_v(n);
            if (norm < 1) {
                atomicCAS(status, 0, 4);
                continue;
            }
            n = times_v(n, 1 / norm);
            double depth = proj(plus_v(between, minus_v(a, b)), n);
            if (depth <= 0)
                continue;
            loaded++;
            V arm_a = times_v(plus_v(a, minus_v(b, between)), .5);
            V arm_b = plus_v(arm_a, between);
            double la = p.sa * (arm_a.x * n.y - arm_a.y * n.x);
            double lb = p.sb * (arm_b.x * n.y - arm_b.y * n.x);
            result[0] += depth;
            result[1] -= la * depth;
            result[2] += lb * depth;
            for (int side = 0; side < 2; side++) {
                double sign = side ? p.sb : p.sa;
                V arm = side ? b : a, face = side ? nb : na;
                V dr = make_double3(-sign * arm.y, sign * arm.x, 0);
                V dn = make_double3(-sign * face.y, sign * face.x, 0);
                dn = times_v(minus_v(dn, times_v(n, proj(dn, n))), 1 / norm);
                double dd = (side ? -1 : 1) * proj(dr, n) + proj(plus_v(between, minus_v(a, b)), dn);
                double rotation = .5 * (dr.x * n.y - dr.y * n.x);
                double dla = p.sa * (rotation + arm_a.x * dn.y - arm_a.y * dn.x);
                double dlb = p.sb * (rotation + arm_b.x * dn.y - arm_b.y * dn.x);
                result[3 + 3 * side] += dd;
                result[4 + 3 * side] -= la * dd + depth * dla;
                result[5 + 3 * side] += lb * dd + depth * dlb;
            }
        }
        if (loaded)
            for (int k = 0; k < 9; k++)
                result[k] *= weight * p.stiff / loaded;
    }
    for (int k = 0; k < 9; k++) {
        double sum = result[k];
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);
        if (lane == 0)
            response[k] = sum;
    }
}

__device__ void solve(double mat[NC_COORDS][NC_COORDS], double rhs[NC_COORDS]) {
    for (int c = 0; c < NC_COORDS - 1; c++) {
        int pivot = c;
        for (int r = c + 1; r < NC_COORDS; r++)
            if (fabs(mat[r][c]) > fabs(mat[pivot][c]))
                pivot = r;
        if (pivot != c) {
            for (int j = c; j < NC_COORDS; j++) {
                double v = mat[c][j];
                mat[c][j] = mat[pivot][j];
                mat[pivot][j] = v;
            }
            double v = rhs[c];
            rhs[c] = rhs[pivot];
            rhs[pivot] = v;
        }
        for (int r = c + 1; r < NC_COORDS; r++) {
            double f = mat[r][c] / mat[c][c];
            for (int j = c + 1; j < NC_COORDS; j++)
                mat[r][j] -= f * mat[c][j];
            rhs[r] -= f * rhs[c];
        }
    }
    for (int c = NC_COORDS - 1; c >= 0; c--) {
        for (int j = c + 1; j < NC_COORDS; j++)
            rhs[c] -= mat[c][j] * rhs[j];
        rhs[c] /= mat[c][c];
    }
}
} // namespace
namespace {
__device__ DeviceStage stage_at(const unsigned long long *ptrs, int pair) {
    const unsigned long long *row = ptrs + 9 * pair;
    return {(const HingePair *)row[0], (const V *)row[1],   (const V *)row[2],
            (const int2 *)row[3],      (const int *)row[4], (const int *)row[5],
            (const double *)row[6],    (const int *)row[7], (const V *)row[8]};
}
} // namespace

extern "C" __global__ void initialize_state(const unsigned long long *pointers, const int *mapping,
                                            const double *coord, const double *nominal,
                                            const double *loads, int bins, int phases, int layers,
                                            double slip0, double ds, int method, double *trace,
                                            int *status) {
    constexpr int n = NC_COORDS, pairs = NC_PAIRS;
    __shared__ double response[pairs][9];
    __shared__ int cache[pairs], work[pairs];
    int tid = threadIdx.x, pair = tid / (32 * NC_WARPS), lane = tid % (32 * NC_WARPS);
    if (lane == 0) {
        cache[pair] = -1;
        work[pair] = 0;
    }
    __syncthreads();
    DeviceStage d = stage_at(pointers, pair);
    double qa = coord[4 * mapping[2 * pair] + 3], qb = coord[4 * mapping[2 * pair + 1] + 3];
    double u = qa / d.p->period;
    u -= floor(u);
    int bin = (int)floor(u * bins + .5) % bins;
    bool active =
        method == 0 || parallel_contact(d, bin, relative_pose(*d.p, qa, qb), cache + pair, work + pair);
    if (lane < 32) {
        if (method == 2)
            anchor_response(d, qa, qb, phases, layers, slip0, ds, qa, qb, active, response[pair], status);
        else
            contact_map(d, qa, qb, bins, phases, layers, slip0, ds, active ? 0 : -1, cache + pair,
                        work + pair, response[pair], status);
    }
    __syncthreads();
    if (lane < 3)
        trace[3 * n + 3 * pair + lane] = response[pair][lane];
    if (tid < n) {
        double q = coord[4 * tid + 3], tau = loads[tid] - coord[4 * tid + 1] * q;
        for (int k = 0; k < 2 * pairs; k++)
            if (mapping[k] == tid)
                tau += response[k / 2][1 + k % 2];
        trace[tid] = q;
        trace[n + tid] = nominal[tid];
        trace[2 * n + tid] = tau / coord[4 * tid];
    }
}

extern "C" __global__ void check_window(const HingePair *p, const V *a, const V *b, const int2 *pairs,
                                        int start, int count, double qa, double qb, int *mask) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < count)
        mask[i] = crosses(pairs[start + i], a, b, relative_pose(*p, qa, qb));
}
// Naturally aligned acquire/release flags are shared with one host producer.
__device__ unsigned load_command(char *page) {
    unsigned value;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(page) : "memory");
    return value;
}
__device__ void publish_reply(char *page, int step) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(page + 64), "r"(step) : "memory");
}
// States and initial velocities come from the independent model. The comparison
// archive is never passed to this kernel. One trajectory advances sequentially.
extern "C" __global__ void advance(const unsigned long long *pointers, const int *mapping,
                                   const double *coord, const double *nominal, const double *loads,
                                   double h, int bins, int phases, int layers, double slip0, double ds,
                                   int method, int begin, int end, double *trace, int *history,
                                   unsigned long long *ticks, int *status, unsigned long long channel) {
    constexpr int n = NC_COORDS, pairs = NC_PAIRS, stride = 3 * n + 3 * pairs;
    __shared__ double st[3][n], q[n], v[n], acc[n], predictor[n], response[pairs][9];
#if NC_EXCHANGE
    __shared__ double step_load[n];
#endif
    __shared__ int cached[pairs], work[pairs], active[pairs], done;
    int lane = threadIdx.x % (32 * NC_WARPS), pair = threadIdx.x / (32 * NC_WARPS);
    DeviceStage d = stage_at(pointers, pair);
    const int a = mapping[2 * pair], b = mapping[2 * pair + 1];
    int tid = threadIdx.x;
    for (int k = tid; k < 3 * n; k += blockDim.x)
        st[k / n][k % n] = trace[begin * stride + k];
    if (lane == 0) {
        cached[pair] = -1;
        work[pair] = 0;
        active[pair] = method == 0 || trace[stride * begin + 3 * n + 3 * pair] > 0;
    }
    __syncthreads();
    char *page = (char *)channel;
    for (int step = begin + 1; step <= end; step++) {
        if (*status)
            return;
        unsigned long long started;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(started));
#if NC_EXCHANGE
        if (tid == 0) {
            while (load_command(page) != (unsigned)step) {
                unsigned long long now;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
                if (now - started > 250000000ULL) {
                    atomicCAS(status, 0, 5);
                    break;
                }
            }
        }
        __syncthreads();
        if (*status)
            return;
#endif
        if (tid < n) {
#if NC_EXCHANGE
            // The producer holds this command until it receives this step's wrench.
            step_load[tid] = ((const volatile double *)(page + 128))[tid];
#endif
            acc[tid] = st[2][tid];
            predictor[tid] = st[0][tid] + h * st[1][tid] + .5 * h * h * st[2][tid];
        }
        if (lane == 0)
            work[pair] = 0;
        __syncthreads();
        int iter;
        for (iter = 0; iter < 20; iter++) {
            if (tid < n) {
                q[tid] = st[0][tid] + h * st[1][tid] + h * h * (acc[tid] - .5 * st[2][tid]);
                v[tid] = st[1][tid] + h * (1.5 * acc[tid] - .5 * st[2][tid]);
            }
            __syncthreads();
            if (lane < 32) {
                if (method == 2)
                    anchor_response(d, q[a], q[b], phases, layers, slip0, ds, predictor[a], predictor[b],
                                    active[pair], response[pair], status);
                else
                    contact_map(d, q[a], q[b], bins, phases, layers, slip0, ds, active[pair] ? 0 : -1,
                                cached + pair, work + pair, response[pair], status);
            }
            __syncthreads();
            if (*status)
                return;
            if (tid == 0) {
                double jac[n][n] = {}, rhs[n], tau[n] = {};
                for (int s = 0; s < pairs; s++) {
                    tau[mapping[2 * s]] += response[s][1];
                    tau[mapping[2 * s + 1]] += response[s][2];
                }
                double norm = 0;
                for (int j = 0; j < n; j++) {
                    double m = coord[4 * j], k = coord[4 * j + 1], c = coord[4 * j + 2];
#if NC_EXCHANGE
                    const double drive = step_load[j];
#else
                    const double drive = loads[n * step + j];
#endif
                    rhs[j] = drive - k * (q[j] - nominal[j] * (step * h)) - c * (v[j] - nominal[j]) +
                             tau[j] - m * (2 * acc[j] - st[2][j]);
                    jac[j][j] = 2 * m + h * h * k + 1.5 * h * c;
                    norm = fmax(norm, fabs(rhs[j]));
                }
                for (int s = 0; s < pairs; s++)
                    for (int e = 0; e < 2; e++)
                        for (int j = 0; j < 2; j++)
                            jac[mapping[2 * s + e]][mapping[2 * s + j]] -=
                                h * h * response[s][(1 + j) * 3 + 1 + e];
                done = norm < 2e-6;
                if (!done) {
                    solve(jac, rhs);
                    for (int j = 0; j < n; j++)
                        acc[j] += rhs[j];
                }
            }
            __syncthreads();
            const bool verify = done && method;
            __syncthreads();
            if (verify) {
                const HingePair p = *d.p;
                double u = q[a] / p.period;
                u -= floor(u);
                int bin = (int)floor(u * bins + .5) % bins;
                bool present =
                    parallel_contact(d, bin, relative_pose(p, q[a], q[b]), cached + pair, work + pair);
                if (lane == 0 && present != active[pair]) {
                    active[pair] = present;
                    atomicExch(&done, 0);
                }
                __syncthreads();
            }
            if (done)
                break;
        }
        if (iter == 20) {
            if (tid == 0)
                atomicCAS(status, 0, 3);
            return;
        }
        if (tid < n) {
            st[0][tid] = q[tid];
            st[1][tid] = v[tid];
            st[2][tid] = acc[tid];
        }
        __syncthreads();
        for (int k = tid; k < 3 * n; k += blockDim.x)
            trace[stride * step + k] = st[k / n][k % n];
        if (lane < 3)
            trace[stride * step + 3 * n + 3 * pair + lane] = response[pair][lane];
        if (tid == 0) {
#if NC_EXCHANGE
            volatile double *wrench = (volatile double *)(page + 384);
            for (int s = 0; s < pairs; s++)
                for (int j = 0; j < 3; j++)
                    wrench[3 * s + j] = response[s][j];
            publish_reply(page, step);
#endif
            history[2 * step] = iter + 1;
            int scanned = 0;
            for (int s = 0; s < pairs; s++)
                scanned += work[s];
            history[2 * step + 1] = scanned;
            unsigned long long now;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
            ticks[step] = now - started;
        }
        __syncthreads();
    }
}

// Pass the identical floating-point transform to the independent full-BVH audit.
extern "C" __global__ void audit_pose(const HingePair *param, double a, double b, double *matrix) {
    RelativePose m = relative_pose(*param, a, b);
    double pose[12] = {m.c, -m.s, 0, m.x, m.s, m.c, 0, m.y, 0, 0, 1, m.z};
    for (int i = 0; i < 12; i++)
        matrix[i] = pose[i];
}
