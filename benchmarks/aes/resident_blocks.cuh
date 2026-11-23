// Compiled after resident_contact.cu. Distances and relative translations are in mm.
struct PatchIndex {
    const double *parents, *leaves;
    const int *ids;
    const unsigned char *occupied;
    const int4 *levels;
    int *active;
    double radius, guard;
    int parentsCount, levelsCount;
    unsigned long long visits;
    const unsigned char *halo;
    int refinement;
    double haloMargin;
};
extern "C" __global__ void bind_patches(PatchIndex *p, const double *parents, const double *leaves,
                                        const int *ids, const unsigned char *occupied, const int4 *levels,
                                        int *active, double radius, double guard, int np, int nl,
                                        const unsigned char *halo, int refinement, double haloMargin) {
    static_assert(sizeof(PatchIndex) == 104);
    *p = {parents, leaves, ids, occupied, levels, active, radius, guard, np, nl, 0, halo, refinement, haloMargin};
}

__device__ void prepare_patches(State &f, const PatchIndex &p) {
    f.cold = !f.rebuilds || p.guard == 0.;
    // Bound every probe's displacement in each target coordinate from the saved pose.
    for (int i = 0; i < 3 && !f.cold; ++i) {
        const double *a = f.pose + 4 * i, *b = f.anchor + 4 * i;
        P row = {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
        f.cold = fabs(a[3] - b[3]) + p.radius * sqrt(dot(row, row)) >= p.guard;
    }
    if (f.cold) {
        ++f.rebuilds;
        f.nc = 0;
        for (int i = 0; i < 12; ++i)
            f.anchor[i] = f.pose[i];
    }
}

// Query octree boxes in the target frame. A large source box visits up to eight cells
// at its chosen level. At probe scale, test the same fine-cell half-space as the force kernel.
template <bool probe> __device__ bool patch_overlap(P c, P e, const MeshView &m, const PatchIndex &p) {
    int lo[3], hi[3], span = 0;
    for (int a = 0; a < 3; ++a) {
        double l = c.at(a) - e.at(a), h = c.at(a) + e.at(a);
        if (h < m.bounds[a] || l > m.bounds[a + 3])
            return false;
        lo[a] = int(fmax(0., (l - m.bounds[a]) * m.bounds[6]));
        hi[a] = int(fmin(double(m.grid[a] - 1), (h - m.bounds[a]) * m.bounds[6]));
        // The last grid cell can start exactly at the upper model bound.
        lo[a] = min(lo[a], hi[a]);
        span = max(span, hi[a] - lo[a]);
    }
    int level = span <= 1 ? 0 : min(32 - __clz((unsigned)span), p.levelsCount - 1);
    if (probe && span <= 2)
        level = 0;
    int4 dim = p.levels[level];
    for (int z = lo[2] >> level; z <= (hi[2] >> level); ++z)
        for (int y = lo[1] >> level; y <= (hi[1] >> level); ++y)
            for (int x = lo[0] >> level; x <= (hi[0] >> level); ++x) {
                int cell = dim.x + (z * dim.z + y) * dim.y + x;
                if (!p.occupied[cell])
                    continue;
                if (probe && !level) {
                    float4 n = m.planes[cell];
                    double inv = m.bounds[6];
                    double lower = n.w + n.x * (c.x - m.bounds[0] - (x + .5) / inv)
                                       + n.y * (c.y - m.bounds[1] - (y + .5) / inv)
                                       + n.z * (c.z - m.bounds[2] - (z + .5) / inv);
                    if (lower > fabs(n.x) * e.x + fabs(n.y) * e.y + fabs(n.z) * e.z)
                        continue;
                }
                return true;
            }
    return false;
}

__device__ bool source_patch(const double *box, const State &f, const MeshView &m, const PatchIndex &p,
                             double margin) {
    P c = move(f.anchor, {box[0], box[1], box[2]});
    const double *r = f.anchor;
    P e = {fabs(r[0])*box[3] + fabs(r[1])*box[4] + fabs(r[2])*box[5] + margin,
           fabs(r[4])*box[3] + fabs(r[5])*box[4] + fabs(r[6])*box[5] + margin,
           fabs(r[8])*box[3] + fabs(r[9])*box[4] + fabs(r[10])*box[5] + margin};
    return patch_overlap<false>(c, e, m, p);
}

__device__ bool near_occupied(P c, double margin, const MeshView &m, const PatchIndex &p) {
    if (margin > p.haloMargin)
        return patch_overlap<true>(c, {margin, margin, margin}, m, p);
    int cell = 0;
    for (int a = 2; a >= 0; --a) {
        if (c.at(a) < m.bounds[a] - margin || c.at(a) > m.bounds[a+3] + margin)
            return false;
        int n = m.grid[a] * p.refinement;
        int k = int(fmin(double(n-1), fmax(0., (c.at(a)-m.bounds[a]) * m.bounds[6] * p.refinement)));
        cell = cell * n + k;
    }
    return p.halo[cell];
}

__device__ void rebuild_patches(State &f, const MeshView &m, const PatchIndex &p, int worker) {
    int lane = worker % 32;
    double scale = p.radius + fabs(f.anchor[3]) + fabs(f.anchor[7]) + fabs(f.anchor[11])
                   + fabs(m.bounds[0]) + fabs(m.bounds[1]) + fabs(m.bounds[2]) + 1.;
    double margin = p.guard + 1.e-10 + 128. * 2.220446049250313e-16 * scale;
    for (int top = worker / 32; top < p.parentsCount; top += gridDim.x * blockDim.x / 32) {
        bool parent = !lane && source_patch(p.parents + 6 * top, f, m, p, margin);
        if (!__any_sync(0xffffffff, parent))
            continue;
        int leaf = 16 * top + lane;
        bool hit = lane < 16 && 32 * leaf < m.n && source_patch(p.leaves + 6 * leaf, f, m, p, margin);
        unsigned hits = __ballot_sync(0xffffffff, hit);
        while (hits) {
            int k = 512 * top + 32 * (__ffs(hits) - 1) + lane;
            hits &= hits - 1;
            int id = k < m.n ? p.ids[k] : -1;
            bool keep = false;
            if (id >= 0) {
                P c = move(f.anchor, m.points[id]);
                keep = near_occupied(c, margin, m, p);
                if (keep && QUERY_LANES > 1) {
                    double4 old = m.outside[id];
                    P shift = {c.x - old.x, c.y - old.y, c.z - old.z};
                    keep = !(sqrt(dot(shift, shift)) + 1.732050807568878 * margin < old.w);
                }
            }
            unsigned mask = __ballot_sync(0xffffffff, keep);
            int base = 0;
            if (!lane && mask)
                base = atomicAdd(&f.nc, __popc(mask));
            base = __shfl_sync(0xffffffff, base, 0);
            if (keep)
                p.active[base + __popc(mask & ((1u << lane) - 1))] = id;
        }
    }
}
extern "C" __global__ void patch_counts(const State *frames, const PatchIndex *p, unsigned long long *counts,
                                        int directions) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < directions) {
        counts[2*i] = frames[i].rebuilds;
        counts[2*i+1] = p[i].visits;
    }
}

#ifdef NC_INNER_GUARD
struct MotionCandidates {
    double anchor[12];
    int *ids;
    int count, cold;
    U64 builds;
};
extern "C" __global__ void bind_motion_candidates(MotionCandidates *v, int *ids) {
    static_assert(sizeof(MotionCandidates) == 120);
    v->ids = ids;
}
__device__ void prepare_motion_candidates(MotionCandidates &v, const State &f, double radius) {
    v.cold = !v.builds || f.cold;
    for (int i = 0; i < 3 && !v.cold; ++i) {
        const double *a = f.pose + 4 * i, *b = v.anchor + 4 * i;
        P row = {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
        v.cold = fabs(a[3] - b[3]) + radius * sqrt(dot(row, row)) >= NC_INNER_GUARD;
    }
    if (v.cold) {
        ++v.builds;
        v.count = 0;
        for (int i = 0; i < 12; ++i)
            v.anchor[i] = f.pose[i];
    }
}
__device__ void compact_motion_candidates(MotionCandidates &v, const State &f, const MeshView &m,
                                          const PatchIndex &outer, int worker) {
    double scale = 1. + outer.radius + fabs(v.anchor[3]) + fabs(v.anchor[7]) + fabs(v.anchor[11]);
    double e = NC_INNER_GUARD + 1.e-10 + 128. * 2.220446049250313e-16 * scale;
    int lane = worker % 32;
    for (int row = worker - lane; row < f.nc; row += blockDim.x * gridDim.x) {
        int id = row + lane < f.nc ? outer.active[row + lane] : -1;
        bool keep = false;
        if (id >= 0) {
            P c = move(v.anchor, m.points[id]);
            double4 previous = m.outside[id];
            P motion = {c.x - previous.x, c.y - previous.y, c.z - previous.z};
            if (!(previous.w > sqrt(dot(motion, motion)) + 1.732050807568878 * e))
                keep = patch_overlap<true>(c, {e, e, e}, m, outer);
        }
        unsigned chosen = __ballot_sync(0xffffffff, keep);
        int base = 0;
        if (!lane && chosen)
            base = atomicAdd(&v.count, __popc(chosen));
        base = __shfl_sync(0xffffffff, base, 0);
        if (keep)
            v.ids[base + __popc(chosen & ((1u << lane) - 1))] = id;
    }
}
extern "C" __global__ void motion_counts(const MotionCandidates *views, U64 *counts, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        counts[i] = views[i].builds;
}
#endif
