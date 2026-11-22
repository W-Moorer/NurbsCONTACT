#include <cooperative_groups.h>
namespace cg = cooperative_groups;
using U64 = unsigned long long;
constexpr U64 vacant = ~0ULL;

struct P {
    double x, y, z;
    __device__ double at(int d) const { return d == 0 ? x : (d == 1 ? y : z); }
};
__device__ P rotate(const double *r, P x) {
    return {r[0] * x.x + r[1] * x.y + r[2] * x.z, r[4] * x.x + r[5] * x.y + r[6] * x.z,
            r[8] * x.x + r[9] * x.y + r[10] * x.z};
}
__device__ P move(const double *r, P x) {
    P a = rotate(r, x);
    return {a.x + r[3], a.y + r[7], a.z + r[11]};
}
__device__ double dot(P a, P b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

#ifdef NC_SDF
#define FIELD_ARGS , const float *sdf, const int *sdfFace
#define FIELD_PASS , sdf, sdfFace
#define VIEW_FIELD(v) , (v).sdf, (v).sdfFace
#else
#define FIELD_ARGS
#define FIELD_PASS
#define VIEW_FIELD(v)
#endif
#define GEOM                                                                                                 \
    const P *tri, const P *features, const int *fa, const int *offsets, const int *ids, const int *grid,     \
        const double *bounds, const int4 *nodes, const double *boxes, const unsigned char *occupied,         \
        const float4 *planes, const int *ends, const int *seeds, const float4 *axes, const P *points,        \
        const P *normals, const int *fb, const P *other, const int *alt, int n, double4 *outside,            \
        double4 *nearby, int *neighbors, double fpPad, U64 *keys, U64 *peaks, unsigned mask, int *slots,     \
        int *count, double *witness FIELD_ARGS
#define PASS                                                                                                 \
    tri, features, fa, offsets, ids, grid, bounds, nodes, boxes, occupied, planes, ends, seeds, axes,        \
        points, normals, fb, other, alt, n, outside, nearby, neighbors, fpPad, keys, peaks, mask, slots,     \
        count, witness FIELD_PASS

struct MeshView {
    const P *tri, *features;
    const int *fa, *offsets, *ids, *grid;
    const double *bounds;
    const int4 *nodes;
    const double *boxes;
    const unsigned char *occupied;
    const float4 *planes;
    const int *ends, *seeds;
    const float4 *axes;
    const P *points, *normals;
    const int *fb;
    const P *other;
    const int *alt;
    int n;
    double4 *outside, *nearby;
    int *neighbors;
    double fpPad;
    U64 *keys, *peaks;
    unsigned mask;
    int *slots, *count;
    double *witness;
#ifdef NC_SDF
    const float *sdf;
    const int *sdfFace;
#endif
};
#define VIEW(v)                                                                                              \
    (v).tri, (v).features, (v).fa, (v).offsets, (v).ids, (v).grid, (v).bounds, (v).nodes, (v).boxes,         \
        (v).occupied, (v).planes, (v).ends, (v).seeds, (v).axes, (v).points, (v).normals, (v).fb, (v).other, \
        (v).alt, (v).n, (v).outside, (v).nearby, (v).neighbors, (v).fpPad, (v).keys, (v).peaks, (v).mask,    \
        (v).slots, (v).count, (v).witness VIEW_FIELD(v)
extern "C" __global__ void bind_mesh(GEOM, MeshView *v) {
    static_assert(sizeof(MeshView) <= 320, "Increase host mesh-view storage");
    *v = MeshView{PASS};
}

__device__ P sub(P a, P b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__device__ P along(P a, double t, P v) { return {a.x + t * v.x, a.y + t * v.y, a.z + t * v.z}; }
// Feature 0 is the face, 1..3 its cyclic edges and 4..6 its vertices.
__device__ P closest(P p, const P *t, int &feature) {
    P a = t[0], b = t[1], c = t[2], ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
    double d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0 && d2 <= 0) {
        feature = 4;
        return a;
    }
    P bp = sub(p, b);
    double d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) {
        feature = 5;
        return b;
    }
    double vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        feature = 1;
        return along(a, d1 / (d1 - d3), ab);
    }
    P cp = sub(p, c);
    double d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) {
        feature = 6;
        return c;
    }
    double vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        feature = 3;
        return along(a, d2 / (d2 - d6), ac);
    }
    double va = d3 * d6 - d5 * d4;
    if (va <= 0 && d4 >= d3 && d5 >= d6) {
        feature = 2;
        return along(b, (d4 - d3) / (d4 - d3 + d5 - d6), sub(c, b));
    }
    feature = 0;
    double inv = 1. / (va + vb + vc);
    return along(along(a, vb * inv, ab), vc * inv, ac);
}
struct Best {
    double sq;
    int id;
};
__device__ void test_tri(P p, const P *tri, int j, Best &b) {
    int region;
    P delta = sub(p, closest(p, tri + 3 * j, region));
    double sq = dot(delta, delta);
    if (sq < b.sq || (sq == b.sq && j < b.id))
        b = {sq, j};
}
template <int W> __device__ Best nearest_group(Best b) {
    if constexpr (W > 1) {
        unsigned group = __activemask();
        for (int k = W / 2; k; k /= 2) {
            double sq = __shfl_xor_sync(group, b.sq, k, W);
            int id = __shfl_xor_sync(group, b.id, k, W);
            if (sq < b.sq || (sq == b.sq && id < b.id))
                b = {sq, id};
        }
    }
    return b;
}
__device__ double point_box(P p, const double *box) {
    double sq = 0;
    for (int k = 0; k < 3; ++k) {
        double gap = fmax(0., fmax(box[k] - p.at(k), p.at(k) - box[k + 3]));
        sq += gap * gap;
    }
    return sq;
}
__device__ void contact_hit(int i, int target, int source, double depth, P p, P normal, double alignment,
                            U64 *keys, U64 *peaks, unsigned mask, int *slots, int *count, double *witness) {
    U64 key = ((U64)target << 32) | (unsigned)source;
    unsigned slot = (unsigned)((key ^ (key >> 32)) * 11400714819323198485ULL >> 32) & mask;
    while (true) {
        U64 old = atomicCAS(keys + slot, vacant, key);
        if (old == vacant)
            slots[atomicAdd(count, 1)] = slot;
        if (old == vacant || old == key)
            break;
        slot = (slot + 1) & mask;
    }
    double *w = witness + 8 * i;
    w[0] = depth;
    w[1] = p.x;
    w[2] = p.y;
    w[3] = p.z;
    w[4] = normal.x;
    w[5] = normal.y;
    w[6] = normal.z;
    w[7] = alignment;
    atomicMax(peaks + slot, ((U64)__float_as_uint((float)depth) << 32) | (unsigned)i);
}
__device__ float scalar(float4 n, float x, float y, float z) {
    return fmaf(n.x, x, fmaf(n.y, y, fmaf(n.z, z, n.w)));
}
__device__ float segment2(float x, float y, float dx, float dy) {
    float t = fmaxf(0, fminf(1, __fdividef(x * dx + y * dy, dx * dx + dy * dy)));
    x = fmaf(-t, dx, x);
    y = fmaf(-t, dy, y);
    return x * x + y * y;
}
__device__ float quick_distance(P p, const float4 *axes) {
    float x = p.x, y = p.y, z = p.z;
    float a = scalar(axes[0], x, y, z), b = scalar(axes[1], x, y, z), c = scalar(axes[2], x, y, z);
    float4 shape = axes[3];
    float l = shape.x, u = shape.y, v = shape.z;
    bool base = b < 0, left = a * v < b * u, right = (l - a) * v < b * (l - u);
    if (!base && !left && !right)
        return c * c;
    // The Euclidean projection is on a supporting edge visible from the point.
    float planar = 3.4028234e38f;
    if (base) {
        float dx = fmaxf(0, fmaxf(-a, a - l));
        planar = dx * dx + b * b;
    }
    if (left)
        planar = fminf(planar, segment2(a, b, u, v));
    if (right)
        planar = fminf(planar, segment2(a - l, b, u - l, v));
    return c * c + planar;
}
template <int W> __device__ void sample_contact(int i, const double *pose, GEOM) {
    P p = move(pose, points[i]);
    for (int d = 0; d < 3; ++d)
        if (p.at(d) <= bounds[d] + 1.e-10 || p.at(d) >= bounds[d + 3] - 1.e-10)
            return;
    int ix = (int)((p.x - bounds[0]) * bounds[6]), iy = (int)((p.y - bounds[1]) * bounds[6]),
        iz = (int)((p.z - bounds[2]) * bounds[6]);
    int cell = (iz * grid[1] + iy) * grid[0] + ix;
    if (!occupied[cell])
        return;
#ifdef NC_SDF
    // Analytic derivative of the trilinear interpolant. Face tags come from the
    // closest grid node and are categorical, so they are never interpolated.
    int nx = grid[0] + 1, ny = grid[1] + 1;
    int base = (iz * ny + iy) * nx + ix;
    double x = (p.x - bounds[0]) * bounds[6] - ix, y = (p.y - bounds[1]) * bounds[6] - iy;
    double z = (p.z - bounds[2]) * bounds[6] - iz;
    double value = 0.;
    P g = {};
    for (int k = 0; k < 2; ++k)
        for (int j = 0; j < 2; ++j)
            for (int l = 0; l < 2; ++l) {
                double v = sdf[base + (k * ny + j) * nx + l];
                double wx = l ? x : 1 - x, wy = j ? y : 1 - y, wz = k ? z : 1 - z;
                value += wx * wy * wz * v;
                g.x += (l ? 1 : -1) * wy * wz * v;
                g.y += wx * (j ? 1 : -1) * wz * v;
                g.z += wx * wy * (k ? 1 : -1) * v;
            }
    double length = sqrt(dot(g, g));
    if (value >= -1e-10 || length <= 1e-14)
        return;
    P nrm = {g.x / length, g.y / length, g.z / length};
    double f = dot(nrm, rotate(pose, normals[i])), o = dot(nrm, rotate(pose, other[i]));
    if (fmin(f, o) < -1e-10) {
        int tag = sdfFace[base + ((int)(z > .5) * ny + (int)(y > .5)) * nx + (int)(x > .5)];
        contact_hit(i, tag, o < f ? alt[i] : fb[i], -value, p, nrm, fmin(f, o), keys, peaks, mask, slots,
                    count, witness);
    }
    return;
#endif
    int lane = threadIdx.x % W;
    Best best = {1.e300, 0x7fffffff};
#ifdef NC_BVH_CONTROL
    if (seeds[cell] >= 0)
        test_tri(p, tri, seeds[cell], best);
#else
    if constexpr (W > 1) {
        double4 prev = outside[i];
        P shift = {p.x - prev.x, p.y - prev.y, p.z - prev.z};
        if (prev.w > 0 && dot(shift, shift) < prev.w * prev.w)
            return;
    }
    float4 sheet = planes[cell];
    double inv = bounds[6];
    double lower = sheet.w + sheet.x * (p.x - bounds[0] - (ix + .5) / inv) +
                   sheet.y * (p.y - bounds[1] - (iy + .5) / inv) +
                   sheet.z * (p.z - bounds[2] - (iz + .5) / inv);
    if (lower > 0.) {
        if constexpr (W > 1)
            if (!lane)
                outside[i] = make_double4(p.x, p.y, p.z, lower);
        return;
    }

    bool reused = false;
    if constexpr (W > 1) {
        double4 prev = nearby[i];
        P shift = {p.x - prev.x, p.y - prev.y, p.z - prev.z};
        double travel = sqrt(dot(shift, shift));
        if (prev.w > travel) {
            int j = neighbors[W * i + lane];
            if (j >= 0)
                test_tri(p, tri, j, best);
            best = nearest_group<W>(best);
            reused = sqrt(best.sq) + travel < prev.w;
        }
    }
    if (!reused) {
        Best first = {1.e300, 0x7fffffff};
        double second = 1.e300;
        for (int k = offsets[cell] + lane; k < ends[cell]; k += W) {
            int j = ids[k];
            double d = quick_distance(p, axes + 4 * j);
            if (d < first.sq || (d == first.sq && j < first.id)) {
                second = first.sq;
                first = {d, j};
            } else
                second = fmin(second, d);
        }
        Best approximate = nearest_group<W>(first);
        double limit = sqrt(approximate.sq) + 2 * fpPad;
        limit *= limit;
        best = {1.e300, 0x7fffffff};
        if (first.id != 0x7fffffff && first.sq <= limit)
            test_tri(p, tri, first.id, best);
        if (second <= limit) {
            for (int k = offsets[cell] + lane; k < ends[cell]; k += W) {
                int j = ids[k];
                if (j != first.id && quick_distance(p, axes + 4 * j) <= limit)
                    test_tri(p, tri, j, best);
            }
        }
        best = nearest_group<W>(best);
        if constexpr (W > 1) {
            neighbors[i * W + lane] = first.id == 0x7fffffff ? -1 : first.id;
            unsigned group = __activemask();
            for (int k = W / 2; k; k /= 2)
                second = fmin(second, __shfl_xor_sync(group, second, k, W));
            if (!lane)
                // Cell preparation drops triangles farther than d_center + two cell radii.
                // At the previous probe these omitted triangles were at least one cell radius away.
                nearby[i] =
                    make_double4(p.x, p.y, p.z, fmin(.866025403784438 / bounds[6], sqrt(second) - fpPad));
        }
    }
#endif
#ifdef NC_BVH_CONTROL
    {
#else
    if (best.sq >= bounds[7] * bounds[7]) {
#endif
        // Interior points of arbitrary depth and near-boundary cells lacking
        // a local witness take the unrestricted closest-triangle path.
        for (int j = 0; j < nodes[0].x;) {
            int4 node = nodes[j];
            if (point_box(p, boxes + 6 * j) > best.sq) {
                j = node.x;
                continue;
            }
            ++j;
            if (node.y < 0)
                continue;
            for (int k = node.y + lane; k < node.z; k += W)
                test_tri(p, tri, k, best);
            best = nearest_group<W>(best);
        }
    }
    if (lane || best.sq <= 1.e-20)
        return;
    int id = best.id, region;
    P delta = sub(closest(p, tri + 3 * id, region), p);
    if (dot(delta, features[7 * id + region]) <= 0) {
        if constexpr (W > 1)
            outside[i] = make_double4(p.x, p.y, p.z, fmax(0., sqrt(best.sq) - 1e-9));
        return;
    }
    double depth = sqrt(best.sq);
    // On a face use the geometric normal directly; subtracting nearly coincident
    // points can turn parallel sliding into a small spurious compressive component.
    P normal = region == 0 ? features[7 * id] : P{delta.x / depth, delta.y / depth, delta.z / depth};
    double facing = dot(normal, rotate(pose, normals[i])),
           opposite = fb[i] == alt[i] ? facing : dot(normal, rotate(pose, other[i]));
    if (fmin(facing, opposite) >= -1.e-10)
        return;
    contact_hit(i, fa[id], opposite < facing ? alt[i] : fb[i], depth, p, normal, fmin(facing, opposite), keys,
                peaks, mask, slots, count, witness);
}

extern "C" __global__ void probe(GEOM, const double *pose) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += gridDim.x * blockDim.x)
        sample_contact<1>(i, pose, PASS);
}
extern "C" __global__ void clear_active(U64 *keys, U64 *best, const int *slots, const int *count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < *count; i += gridDim.x * blockDim.x) {
        keys[slots[i]] = vacant;
        best[slots[i]] = 0;
    }
}

// Each body has one revolute or prismatic coordinate in this benchmark driver.
// The triangle query above accepts any rigid 3D relative pose.
struct Joint {
    double kind, axis[3], origin[3], offset, speed, mass, spring, damping, initial;
};
struct State {
    double q[2], v[2], a[2], f[2], trial[2], pos[2], vel[2], pose[12], rot[18], origin[6];
    double error, scale, correction[2], base[2], peak_error;
    double anchor[12];
    int tick, span, splits, deepest, total_substeps;
    unsigned long long begin;
    double last_error;
#ifdef NC_TWO_SIDED
    double inverse[12];
#endif
    int firstProbe, probeCount;
#ifdef NC_IMPACT_SAMPLES
    double refine_until, peak_fn, peak_t, first_fn;
    int accepted_contacts, impact_windows, impact_span, timed_step;
#endif
    int step, iteration, backtrack, trials, fail, max_iterations, cold, nc, rebuilds, max_candidates;
};

__device__ void set_pose(State &s, const Joint *joints, double t) {
    for (int j = 0; j < 2; ++j) {
        const Joint &b = joints[j];
        double value = b.offset + b.speed * t + s.pos[j];
        double angle = b.kind == 0 ? value : 0, sn, cs;
        sincos(angle, &sn, &cs);
        double x = b.axis[0], y = b.axis[1], z = b.axis[2], u = 1 - cs;
        double r[9] = {cs + u * x * x,     u * x * y - sn * z, u * x * z + sn * y,
                       u * x * y + sn * z, cs + u * y * y,     u * y * z - sn * x,
                       u * x * z - sn * y, u * y * z + sn * x, cs + u * z * z};
        for (int k = 0; k < 9; ++k)
            s.rot[9 * j + k] = r[k];
        for (int k = 0; k < 3; ++k)
            s.origin[3 * j + k] = b.origin[k] + (b.kind == 1 ? 1000 * value * b.axis[k] : 0.);
    }
    for (int i = 0; i < 3; ++i) {
        double shift = 0.;
        for (int k = 0; k < 3; ++k)
            shift += s.rot[3 * k + i] * (s.origin[3 + k] - s.origin[k]);
        s.pose[4 * i + 3] = shift;
        for (int j = 0; j < 3; ++j) {
            double v = 0.;
            for (int k = 0; k < 3; ++k)
                v += s.rot[3 * k + i] * s.rot[9 + 3 * k + j];
            s.pose[4 * i + j] = v;
        }
    }
#ifdef NC_TWO_SIDED
    for (int i = 0; i < 3; ++i) {
        double shift = 0.;
        for (int j = 0; j < 3; ++j) {
            s.inverse[4 * i + j] = s.pose[4 * j + i];
            shift -= s.pose[4 * j + i] * s.pose[4 * j + 3];
        }
        s.inverse[4 * i + 3] = shift;
    }
#endif
}
__device__ P from_a(const State &s, P a) {
    return {s.rot[0] * a.x + s.rot[1] * a.y + s.rot[2] * a.z,
            s.rot[3] * a.x + s.rot[4] * a.y + s.rot[5] * a.z,
            s.rot[6] * a.x + s.rot[7] * a.y + s.rot[8] * a.z};
}
__device__ double arm(const Joint &j, P normal, P r) {
    P axis = {j.axis[0], j.axis[1], j.axis[2]};
    if (j.kind == 1)
        return dot(axis, normal);
    P cross = {r.y * normal.z - r.z * normal.y, r.z * normal.x - r.x * normal.z,
               r.x * normal.y - r.y * normal.x};
    return .001 * dot(cross, axis);
}
__device__ void trial_pose(State &s, const Joint *joints, double h) {
    for (int j = 0; j < 2; ++j) {
        s.pos[j] = s.q[j] + h * s.v[j] + h * h * (.1975 * s.a[j] + .3025 * s.trial[j]);
        s.vel[j] = s.v[j] + h * (.4 * s.a[j] + .6 * s.trial[j]);
    }
}
__device__ unsigned long long gpu_time() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ void prepare_trial(State &s, const Joint *joints, double h, double guard, double radius) {

#ifdef NC_IMPACT_SAMPLES
    if (!s.trials || s.timed_step != s.step) {
        s.timed_step = s.step;
        s.begin = gpu_time();
    }
#else
    if (s.tick == 0 && s.iteration == 0 && s.backtrack == 0 && (s.span == 64 || !s.step))
        s.begin = gpu_time();
#endif
    if (s.step == 0) {
        if (!s.trials) {
            s.span = 64;
            for (int j = 0; j < 2; ++j) {
                s.pos[j] = s.q[j] = joints[j].initial;
                s.vel[j] = s.v[j] = 0.;
            }
        }
        set_pose(s, joints, 0.);
    } else {
        trial_pose(s, joints, h * (s.span / 64.));
        set_pose(s, joints, (s.step - 1 + (s.tick + s.span) / 64.) * h);
    }
    double dr = 0, dt = 0;
    for (int i = 0; i < 12; ++i) {
        double delta = s.pose[i] - s.anchor[i];
        if (i % 4 == 3)
            dt += delta * delta;
        else
            dr += delta * delta;
    }
    s.cold = !s.rebuilds || guard == 0 || sqrt(dt) + radius * sqrt(dr) > guard;
    if (s.cold) {
        ++s.rebuilds;
        s.nc = 0;
        for (int i = 0; i < 12; ++i)
            s.anchor[i] = s.pose[i];
    }
}
// Project the target bound onto a source-local direction. The resulting
// sorted interval contains every source sample that can reach the target.
__device__ void source_interval(State &s, const double *box, const double *directions, const int *bins, int n,
                                double radius, double guard) {
    if (!s.cold)
        return;
    P center = {(box[0] + box[3]) * .5 - s.pose[3], (box[1] + box[4]) * .5 - s.pose[7],
                (box[2] + box[5]) * .5 - s.pose[11]};
    P half = {(box[3] - box[0]) * .5, (box[4] - box[1]) * .5, (box[5] - box[2]) * .5};
    int best = n, first = 0;
    for (int axis = 0; axis < 13; ++axis) {
        const double *d = directions + 5 * axis;
        P dir = rotate(s.pose, {d[0], d[1], d[2]});
        double mid = dot(dir, center),
               width = fmin(radius, fabs(dir.x) * half.x + fabs(dir.y) * half.y + fabs(dir.z) * half.z) +
                       guard + 1.e-8;
        int lo = (int)fmax(0., fmin(256., floor((mid - width - d[3]) * d[4])));
        int hi = (int)fmax(0., fmin(256., 1. + floor((mid + width - d[3]) * d[4])));
        int a = bins[axis * 257 + lo], b = bins[axis * 257 + hi];
        if (b - a < best) {
            best = b - a;
            first = axis * n + a;
        }
    }
    s.firstProbe = first;
    s.probeCount = best;
}
#ifdef NC_TWO_SIDED
// Express reverse witnesses in the forward target frame and keep one spring
// for each unordered pair of contact regions. Called by the solver's single
// assembly thread after both directions have completed their parallel query.
__device__ void merge_reverse(const MeshView &back, const double *pose, int n, bool preferBack, U64 *keys,
                              U64 *peaks, unsigned mask, int *slots, int *count, double *witness) {
    for (int k = 0; k < *back.count; ++k) {
        int old = back.slots[k];
        U64 id = back.keys[old], key = (id << 32) | (id >> 32);
        int src = (unsigned)back.peaks[old];
        back.keys[old] = vacant;
        back.peaks[old] = 0;
        const double *rw = back.witness + 8 * src;
        unsigned slot = (unsigned)((key ^ (key >> 32)) * 11400714819323198485ULL >> 32) & mask;
        while (keys[slot] != vacant && keys[slot] != key)
            slot = (slot + 1) & mask;
        if (keys[slot] == vacant) {
            keys[slot] = key;
            slots[(*count)++] = slot;
        } else {
            double depth = witness[8 * (unsigned)peaks[slot]];
            if (rw[0] < depth - 1e-12 || (fabs(rw[0] - depth) <= 1e-12 && !preferBack))
                continue;
        }
        P p = move(pose, {rw[1], rw[2], rw[3]});
        P normal = rotate(pose, {rw[4], rw[5], rw[6]});
        int dest = n + src;
        double *w = witness + 8 * dest;
        w[0] = rw[0];
        w[1] = p.x;
        w[2] = p.y;
        w[3] = p.z;
        w[4] = -normal.x;
        w[5] = -normal.y;
        w[6] = -normal.z;
        w[7] = rw[7];
        peaks[slot] = (unsigned)dest;
    }
    *back.count = 0;
}
#endif
extern "C" __global__ void advance(GEOM, const Joint *joints, State *state, const double *loads, double h,
                                   double stiffness, int stop, int ntime, int predict, double *trace,
                                   const unsigned char *halo, int *candidates, double guard, double radius,
                                   unsigned long long *elapsed, const double *dirs, const int *bins,
                                   const int *order, double targetRadius
#ifdef NC_TWO_SIDED
                                   ,
                                   const MeshView *reverse, int preferReverse
#endif
) {
    auto all = cg::this_grid();
    int worker = blockIdx.x * blockDim.x + threadIdx.x;
    State &s = *state;
    if (worker == 0 && s.step < stop && !s.fail) {
        prepare_trial(s, joints, h, guard, radius);
        source_interval(s, bounds, dirs, bins, n, targetRadius, guard);
    }
    all.sync();
    while (s.step < stop && !s.fail) {
        if (s.cold) {
            for (int begin = worker & ~31; begin < s.probeCount; begin += blockDim.x * gridDim.x) {
                int lane = threadIdx.x & 31;
                int k = begin + lane, i = k < s.probeCount ? order[s.firstProbe + k] : n;
                bool keep = false;
                if (i < n) {
                    P p = move(s.pose, points[i]);
                    if (p.x >= bounds[0] - guard && p.x <= bounds[3] + guard && p.y >= bounds[1] - guard &&
                        p.y <= bounds[4] + guard && p.z >= bounds[2] - guard && p.z <= bounds[5] + guard) {
                        int x = max(0, min(grid[0] - 1, (int)((p.x - bounds[0]) * bounds[6])));
                        int y = max(0, min(grid[1] - 1, (int)((p.y - bounds[1]) * bounds[6])));
                        int z = max(0, min(grid[2] - 1, (int)((p.z - bounds[2]) * bounds[6])));
                        keep = halo[(z * grid[1] + y) * grid[0] + x] != 0;
                    }
                }
                unsigned selected = __ballot_sync(0xffffffff, keep);
                int slot = 0;
                if (lane == 0 && selected)
                    slot = atomicAdd(&s.nc, __popc(selected));
                slot = __shfl_sync(0xffffffff, slot, 0);
                if (keep)
                    candidates[slot + __popc(selected & ((1u << lane) - 1))] = i;
            }
            all.sync();
        }
        for (int i = worker / QUERY_LANES; i < s.nc; i += blockDim.x * gridDim.x / QUERY_LANES)
            sample_contact<QUERY_LANES>(candidates[i], s.pose, PASS);
#ifdef NC_TWO_SIDED
        for (int i = worker; i < reverse->n; i += blockDim.x * gridDim.x)
            sample_contact<1>(i, s.inverse, VIEW(*reverse));
#endif
        all.sync();
        if (worker == 0) {
#ifdef NC_TWO_SIDED
            merge_reverse(*reverse, s.pose, n, preferReverse, keys, peaks, mask, slots, count, witness);
#endif
            double force[2], k00 = 0, k01 = 0, k11 = 0, total = 0, nextForce[2] = {}, nextK[3] = {};
            for (int j = 0; j < 2; ++j) {
                double load = loads[2 * s.step + j];
                if (s.step && s.tick + s.span < 64) {
                    double left = loads[2 * (s.step - 1) + j];
                    load = left + (load - left) * ((s.tick + s.span) / 64.);
                }
                force[j] = load - joints[j].spring * s.pos[j] - joints[j].damping * s.vel[j];
            }
            s.max_candidates = max(s.max_candidates, s.nc);
            int contacts = *count;
            // A fixed accumulation order also makes Newton retries reproducible.
            for (int i = 1; i < contacts; ++i) {
                int v = slots[i], j = i;
                while (j > 0 && keys[slots[j - 1]] > keys[v]) {
                    slots[j] = slots[j - 1];
                    --j;
                }
                slots[j] = v;
            }
            for (int i = 0; i < contacts; ++i) {
                int slot = slots[i];
                unsigned winner = (unsigned)peaks[slot];
                double *w = witness + 8 * winner;
                P normal = from_a(s, {w[4], w[5], w[6]});
                P ra = from_a(s, {w[1], w[2], w[3]});
                P rb = {ra.x + s.origin[0] - s.origin[3], ra.y + s.origin[1] - s.origin[4],
                        ra.z + s.origin[2] - s.origin[5]};
                double ja = -arm(joints[0], normal, ra), jb = arm(joints[1], normal, rb);
                double fn = .001 * stiffness * w[0];
                total += fn;
                force[0] += ja * fn;
                force[1] += jb * fn;
                if (predict) {
                    double dq0 = h * (s.vel[0] + joints[0].speed) + .5 * h * h * s.trial[0];
                    double dq1 = h * (s.vel[1] + joints[1].speed) + .5 * h * h * s.trial[1];
                    double next = fn - stiffness * (ja * dq0 + jb * dq1);
                    if (next > 0) {
                        nextForce[0] += ja * next;
                        nextForce[1] += jb * next;
                        nextK[0] += stiffness * ja * ja;
                        nextK[1] += stiffness * ja * jb;
                        nextK[2] += stiffness * jb * jb;
                    }
                }
                k00 += stiffness * ja * ja;
                k01 += stiffness * ja * jb;
                k11 += stiffness * jb * jb;
                keys[slot] = vacant;
                peaks[slot] = 0;
            }
            *count = 0;
            ++s.trials;
            double res[2], scale = 0;
            for (int j = 0; j < 2; ++j) {
                res[j] = joints[j].mass * s.trial[j] - .9 * force[j] - .1 * s.f[j];
                scale = fmax(scale, fmax(fabs(force[j]), fabs(s.f[j])));
            }
            double err = fmax(fabs(res[0]), fabs(res[1]));
            s.last_error = err;
#ifdef NC_IMPACT_SAMPLES
            // An observed first load starts a local window at the contact time
            // scale. trace(M^-1 K) bounds the largest contact eigenvalue.
            double t0 = (s.step - 1 + s.tick / 64.) * h;
            double half_period = 0.;
            int desired = 64;
            bool enter = s.step && contacts && !s.accepted_contacts;
            if (enter && (k00 > 0 || k11 > 0)) {
                half_period = 3.141592653589793 / sqrt(k00 / joints[0].mass + k11 / joints[1].mass);
                while (desired > 1 && h * (desired / 64.) * NC_IMPACT_SAMPLES > half_period)
                    desired /= 2;
            }
            if (enter && s.span > desired) {
                ++s.impact_windows;
                s.refine_until = fmax(s.refine_until, t0 + half_period);
                s.impact_span = desired;
                s.span = desired;
                s.deepest = max(s.deepest, 64 / desired);
                s.iteration = s.backtrack = 0;
                for (int j = 0; j < 2; ++j)
                    s.trial[j] = s.a[j];
            } else
#endif
                if (!isfinite(err) || !isfinite(total))
                s.fail = 1;
            else if (!s.step || err <= 1.e-7 * (1 + scale)) {
                s.max_iterations = max(s.max_iterations, s.iteration);
                if (s.step)
                    s.peak_error = fmax(s.peak_error, err);
                for (int j = 0; j < 2; ++j) {
                    s.q[j] = s.pos[j];
                    s.v[j] = s.vel[j];
                    s.f[j] = force[j];
                    s.a[j] = s.trial[j] = s.step ? s.trial[j] : force[j] / joints[j].mass;
                }
                ++s.total_substeps;
#ifdef NC_IMPACT_SAMPLES
                double t1 = s.step ? t0 + h * (s.span / 64.) : 0.;
                if (enter && desired < 64) {
                    s.refine_until = fmax(s.refine_until, t1 + half_period);
                    s.impact_span = desired;
                }
                if (total > 0 && s.peak_fn == 0)
                    s.first_fn = t1;
                if (total > s.peak_fn) {
                    s.peak_fn = total;
                    s.peak_t = t1;
                }
                s.accepted_contacts = contacts;
#endif
                s.tick += s.span;
                if (s.tick == 64) {
                    elapsed[s.step] = gpu_time() - s.begin;
                    double *row = trace + 8 * s.step;
                    row[0] = s.q[0];
                    row[1] = s.q[1];
                    row[2] = s.v[0];
                    row[3] = s.v[1];
                    row[4] = total;
                    row[5] = contacts;
                    row[6] = s.iteration;
                    row[7] = s.step ? err : 0.;
                    ++s.step;
                    s.tick = 0;
                    s.span = 64;
#ifdef NC_IMPACT_SAMPLES
                    if ((s.step - 1) * h < s.refine_until)
                        s.span = s.impact_span;
#endif
                    if (predict && s.step < ntime && s.span == 64) {
                        double r[2];
                        for (int j = 0; j < 2; ++j) {
                            double q = s.q[j] + h * s.v[j] + .5 * h * h * s.a[j];
                            double v = s.v[j] + h * s.a[j];
                            double f = loads[2 * s.step + j] - joints[j].spring * q - joints[j].damping * v +
                                       nextForce[j];
                            r[j] = joints[j].mass * s.a[j] - .9 * f - .1 * s.f[j];
                        }
                        double fac = .9 * .3025 * h * h;
                        double a = joints[0].mass + fac * (joints[0].spring + nextK[0]) +
                                   .54 * h * joints[0].damping;
                        double b = fac * nextK[1];
                        double c = joints[1].mass + fac * (joints[1].spring + nextK[2]) +
                                   .54 * h * joints[1].damping;
                        double det = a * c - b * b;
                        s.trial[0] += (b * r[1] - c * r[0]) / det;
                        s.trial[1] += (b * r[0] - a * r[1]) / det;
                    }
                } else
                    s.span = min(s.span, 64 - s.tick);
                s.iteration = 0;
                s.backtrack = 0;
            } else if (s.backtrack && err >= s.error) {
                if (++s.backtrack > 10)
                    s.fail = 2;
                double f = ldexp(1., 1 - s.backtrack);
                for (int j = 0; j < 2; ++j)
                    s.trial[j] = s.base[j] + f * s.correction[j];
            } else {
                if (++s.iteration > 20)
                    s.fail = 3;
                double dt = h * (s.span / 64.);
                double fac = .9 * .3025 * dt * dt;
                double a = joints[0].mass + fac * (joints[0].spring + k00) + .9 * .6 * dt * joints[0].damping;
                double c = joints[1].mass + fac * (joints[1].spring + k11) + .9 * .6 * dt * joints[1].damping;
                double b = fac * k01, det = a * c - b * b;
                s.correction[0] = (b * res[1] - c * res[0]) / det;
                s.correction[1] = (b * res[0] - a * res[1]) / det;
                for (int j = 0; j < 2; ++j) {
                    s.base[j] = s.trial[j];
                    s.trial[j] += s.correction[j];
                }
                s.backtrack = 1;
                s.error = err;
            }
            if (s.fail >= 2 && s.span > 1) {
                s.fail = 0;
                s.span /= 2;
                ++s.splits;
                s.deepest = max(s.deepest, 64 / s.span);
                s.iteration = 0;
                s.backtrack = 0;
                for (int j = 0; j < 2; ++j)
                    s.trial[j] = s.a[j];
            }
            if (s.step < stop && !s.fail) {
                prepare_trial(s, joints, h, guard, radius);
                source_interval(s, bounds, dirs, bins, n, targetRadius, guard);
            }
        }
        all.sync();
    }
}
extern "C" __global__ void status(const State *s, int *counters) {
    counters[0] = s->step;
    counters[1] = s->trials;
    counters[2] = s->fail;
    counters[3] = s->max_iterations;
    counters[4] = sizeof(State);
    counters[5] = s->rebuilds;
    counters[6] = s->max_candidates;
    counters[7] = s->splits;
    counters[8] = s->deepest;
    counters[9] = s->total_substeps;
}
#ifdef NC_IMPACT_SAMPLES
extern "C" __global__ void impact_stats(const State *s, double *out) {
    out[0] = s->impact_windows;
    out[1] = s->peak_fn;
    out[2] = s->peak_t;
    out[3] = s->first_fn;
}
#endif
extern "C" __global__ void cell_planes(const int *work, int ncell, const P *tri, const P *features,
                                       const double *bounds, const int *grid, const int *offsets, int *ids,
                                       int *finish, float4 *plane, int *seed) {
    int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    for (; w < ncell; w += gridDim.x * blockDim.x / 32) {
        int cell = work[w];
        int x = cell % grid[0], y = (cell / grid[0]) % grid[1], z = cell / (grid[0] * grid[1]);
        double h = 1. / bounds[6], radius = .866025403784439 * h;
        P p = {bounds[0] + (x + .5) * h, bounds[1] + (y + .5) * h, bounds[2] + (z + .5) * h};
        int first = offsets[cell], end = offsets[cell + 1];
        Best b = {1.e300, 0x7fffffff};
        for (int k = first + lane; k < end; k += 32)
            test_tri(p, tri, ids[k], b);
        b = nearest_group<32>(b);
        P normal = features[7 * b.id];
        double limit = sqrt(b.sq) + 2 * radius + 1.e-8, margin = 1.e300;
        int next = first;
        for (int row = first; row < end; row += 32) {
            int j = row + lane < end ? ids[row + lane] : -1;
            bool keep = false;
            if (j >= 0) {
                int reg;
                P delta = sub(p, closest(p, tri + 3 * j, reg));
                keep = dot(delta, delta) <= limit * limit;
            }
            unsigned chosen = __ballot_sync(0xffffffff, keep);
            if (keep) {
                P n = features[7 * j];
                P diff = sub(n, normal);
                double m = dot(n, sub(p, tri[3 * j])) - .5 * h * (fabs(diff.x) + fabs(diff.y) + fabs(diff.z));
                margin = fmin(margin, m);
                ids[next + __popc(chosen & ((1u << lane) - 1))] = j;
            }
            next += __popc(chosen);
            __syncwarp();
        }
        for (int s = 16; s; s /= 2)
            margin = fmin(margin, __shfl_xor_sync(0xffffffff, margin, s));
        if (!lane) {
            finish[cell] = next;
            seed[cell] = b.id;
            if (sqrt(b.sq) + radius + 1.e-8 < bounds[7])
                plane[cell] =
                    make_float4(normal.x, normal.y, normal.z, margin - 1.e-6 * (1. + h + fabs(margin)));
        }
    }
}
