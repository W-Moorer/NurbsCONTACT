// Mesh contact uses the same finite-triangle detector as IntersectionBatch.
using V = double3;
using U64 = unsigned long long;
constexpr U64 empty_key = ~U64(0);
__device__ V plus(V a, V b) { return make_double3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ V sub(V a, V b) { return make_double3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ V scale(V a, double b) { return make_double3(a.x * b, a.y * b, a.z * b); }
__device__ double dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ V cross(V a, V b) { return mesh_detect::cross_v(a, b); }
__device__ double norm(V a) { return sqrt(dot(a, a)); }
__device__ V unit(V a) { return scale(a, 1 / norm(a)); }
__device__ double component(V a, int k) { return k == 0 ? a.x : k == 1 ? a.y : a.z; }
__device__ int component(int3 a, int k) { return k == 0 ? a.x : k == 1 ? a.y : a.z; }
struct Pose {
    V cols[3], offset;
};
struct MatPose {
    V origin, axis;
    double angle;
};
__device__ V rotate(MatPose p, V x, double sign = 1) {
    if (p.angle == 0)
        return x;
    double c = cos(sign * p.angle), s = sin(sign * p.angle);
    return plus(plus(scale(x, c), scale(cross(p.axis, x), s)), scale(p.axis, (1 - c) * dot(p.axis, x)));
}
__device__ V world(MatPose p, V x) { return plus(p.origin, rotate(p, x)); }
__device__ V local(MatPose p, V x) { return rotate(p, sub(x, p.origin), -1); }
__device__ V direction(Pose p, V x) {
    return plus(plus(scale(p.cols[0], x.x), scale(p.cols[1], x.y)), scale(p.cols[2], x.z));
}
__device__ V point(Pose p, V x) { return plus(p.offset, direction(p, x)); }
__device__ Pose between(MatPose a, MatPose b) {
    V ac[3] = {rotate(a, make_double3(1, 0, 0)), rotate(a, make_double3(0, 1, 0)),
               rotate(a, make_double3(0, 0, 1))};
    V bc[3] = {rotate(b, make_double3(1, 0, 0)), rotate(b, make_double3(0, 1, 0)),
               rotate(b, make_double3(0, 0, 1))};
    V off = sub(a.origin, b.origin);
    Pose p{};
    p.offset = make_double3(dot(bc[0], off), dot(bc[1], off), dot(bc[2], off));
    for (int k = 0; k < 3; ++k)
        p.cols[k] = make_double3(dot(bc[0], ac[k]), dot(bc[1], ac[k]), dot(bc[2], ac[k]));
    p.cols[0] = unit(p.cols[0]);
    p.cols[1] = unit(sub(p.cols[1], scale(p.cols[0], dot(p.cols[0], p.cols[1]))));
    p.cols[2] = unit(cross(p.cols[0], p.cols[1]));
    return p;
}
__device__ V inverse_direction(Pose p, V v) {
    return make_double3(dot(p.cols[0], v), dot(p.cols[1], v), dot(p.cols[2], v));
}
__device__ V inverse_point(Pose p, V v) { return inverse_direction(p, sub(v, p.offset)); }
__device__ unsigned long long timer_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
struct Region {
    V p, n, arm;
    double depth;
};
struct Anchor {
    V a, b, na, nb, ca, cb;
};
struct PairScratch {
    int anchors, ready;
    Anchor *anchor;
};
struct BodyView {
    const V *vertices, *normals, *rays, *packed, *vertex_normals;
    const int3 *faces, *face_edges;
    const int4 *edges;
    const int2 *ridges;
    const int *panels, *panel_offsets, *panel_faces, *ray_offsets, *ridge_offsets, *ridge_ids, *edge_ridge;
    const double *bounds;
    const int *links, *packed_ids;
    const int *neighbors, *components, *component_seeds;
    double radius;
    int nv, nf, ne, nr, convex, ncomponents;
};
struct AxisCoordinate {
    double mass, q, v, k, c, rest, load;
    int first, end;
};
struct LoadPoint {
    double t, value;
};
struct JointBody {
    V origin, axis, pivot;
    double angle;
    int hinge, translation;
    V direction;
};
struct ContactLaw {
    int a, b;
    double k, exponent, rate, max_depth;
    bool shared;
};
struct Workspace {
    const mesh_detect::MeshPairView *pairs;
    int *status, *parents, *region_valid;
    U64 *patch_hash, *feature_hash, *patches, *features;
    int capacity, patch_capacity, feature_capacity, pair, regions, selected;
    int counts[2], filtered, refresh;
    int *visited, *queue;
    int vertex_capacity, groups;
    V *placed_vertices, *placed_normals;
    double pose[12];
    Region *result, *unsorted, *contributions;
    unsigned long long queries, subdivisions, evaluations, integral_ns, broad_ns, narrow_ns, group_ns,
        support_ns;
    double maximum_residual;
};
struct Query {
    BodyView b[2];
    Pose rel[2];
    double reach;
    Workspace *work;
};

extern "C" __global__ void
bind_mesh_body(BodyView *out, const V *vertices, const int3 *faces, const V *normals, const int4 *edges,
               const int3 *face_edges, const int *panels, const int *panel_offsets, const int *panel_faces,
               const int *ray_offsets, const V *rays, const int2 *ridges, const int *edge_ridge,
               const int *ridge_offsets, const int *ridge_ids, const V *packed, const double *bounds,
               const int *links, const int *packed_ids, const V *vertex_normals, const int *neighbors,
               const int *components, const int *component_seeds, double radius, int nv, int nf, int ne,
               int nr, int convex, int ncomponents) {
    static_assert(sizeof(BodyView) <= 256);
    *out = {vertices,
            normals,
            rays,
            packed,
            vertex_normals,
            faces,
            face_edges,
            edges,
            ridges,
            panels,
            panel_offsets,
            panel_faces,
            ray_offsets,
            ridge_offsets,
            ridge_ids,
            edge_ridge,
            bounds,
            links,
            packed_ids,
            neighbors,
            components,
            component_seeds,
            radius,
            nv,
            nf,
            ne,
            nr,
            convex,
            ncomponents};
}
__device__ V vertex(const Query &q, int side, int id) {
    return side ? q.work->placed_vertices[id] : q.b[0].vertices[id];
}
__device__ V normal(const Query &q, int side, int id) {
    return side ? q.work->placed_normals[id] : q.b[0].normals[id];
}
__device__ void triangle(const Query &q, int side, int id, V *p) {
    int3 f = q.b[side].faces[id];
    for (int j = 0; j < 3; ++j)
        p[j] = vertex(q, side, component(f, j));
}
__device__ bool inside_triangle(const V *p, V n, V v) {
    for (int j = 0; j < 3; ++j) {
        V e = sub(p[(j + 1) % 3], p[j]);
        if (dot(n, cross(e, sub(v, p[j]))) < -1e-13 * dot(e, e))
            return false;
    }
    return true;
}
__device__ int root(int *parent, int i) {
    while (parent[i] != i)
        i = parent[i];
    return i;
}
__device__ void unite(int *p, int a, int b) {
    for (;;) {
        a = root(p, a);
        b = root(p, b);
        if (a == b)
            return;
        int hi = max(a, b), lo = min(a, b);
        if (atomicCAS(p + hi, hi, lo) == hi)
            return;
    }
}
__device__ U64 face_key(int2 p) { return (U64(unsigned(p.x)) << 32) | unsigned(p.y); }
__device__ int find_pair(const int2 *p, int n, int2 target) {
    U64 key = face_key(target);
    int a = 0, b = n;
    while (a < b) {
        int m = (a + b) / 2;
        if (face_key(p[m]) < key)
            a = m + 1;
        else
            b = m;
    }
    return a < n && face_key(p[a]) == key ? a : -1;
}
__device__ bool insert(U64 *hash, int size, U64 key) {
    unsigned i = unsigned((key ^ (key >> 32)) * 11400714819323198485ULL >> 32) & (size - 1);
    for (int k = 0; k < size; ++k) {
        U64 old = atomicCAS(hash + i, empty_key, key);
        if (old == key)
            return false;
        if (old == empty_key)
            return true;
        i = (i + 1) & (size - 1);
    }
    return false;
}
__device__ int power_two(int n) {
    int s = 1;
    while (s < n)
        s *= 2;
    return s;
}
__device__ bool segment_triangle(V x, V y, const V *tri) {
    V a = sub(x, tri[0]), b = sub(y, tri[0]), edge = sub(tri[1], tri[0]), other = sub(tri[2], tri[0]);
    V n = unit(cross(edge, other));
    double tol = 64 * 2.2204460492503131e-16 * (norm(a) + norm(b) + norm(edge) + norm(other));
    double lo = 0, hi = 1, h0 = dot(n, a), h1 = dot(n, b);
    if (fmin(h0, h1) > tol || fmax(h0, h1) < -tol)
        return false;
    if (fabs(h1 - h0) > tol) {
        double t = -h0 / (h1 - h0);
        lo = fmax(lo, t - tol / fabs(h1 - h0));
        hi = fmin(hi, t + tol / fabs(h1 - h0));
    }
    for (int j = 0; j < 3 && lo <= hi; ++j) {
        V e = sub(tri[(j + 1) % 3], tri[j]), in = cross(n, e);
        double first = dot(in, sub(x, tri[j])), slope = dot(in, sub(y, x)), pad = tol * norm(e);
        if (fabs(slope) <= pad) {
            if (first < -pad)
                return false;
        } else if (slope > 0)
            lo = fmax(lo, (-pad - first) / slope);
        else
            hi = fmin(hi, (-pad - first) / slope);
    }
    return lo <= hi;
}
struct MeshTwofold {
    double high, low;
};
__device__ MeshTwofold mesh_exact_difference(double a, double b) {
    double high = a - b, virtual_b = a - high;
    return {high, (a - (high + virtual_b)) + (virtual_b - b)};
}
__device__ MeshTwofold mesh_twofold_add(MeshTwofold a, MeshTwofold b) {
    double s = a.high + b.high, v = s - a.high;
    double e = (a.high - (s - v)) + (b.high - v);
    e += a.low + b.low;
    double high = s + e;
    return {high, e - (high - s)};
}
__device__ MeshTwofold mesh_twofold_product(MeshTwofold a, MeshTwofold b) {
    double p = a.high * b.high, e = fma(a.high, b.high, -p);
    e += a.high * b.low + a.low * b.high;
    double high = p + e;
    return {high, e - (high - p)};
}
__device__ int mesh_plane_side(const V *tri, V p) {
    V a = sub(tri[0], p), b = sub(tri[1], p), c = sub(tri[2], p);
    double permanent = (fabs(b.y * c.z) + fabs(b.z * c.y)) * fabs(a.x) +
                       (fabs(b.z * c.x) + fabs(b.x * c.z)) * fabs(a.y) +
                       (fabs(b.x * c.y) + fabs(b.y * c.x)) * fabs(a.z);
    double value = dot(a, cross(b, c));
    if (fabs(value) > 32 * 2.2204460492503131e-16 * permanent)
        return value > 0 ? 1 : -1;
    // A twofold filter resolves near-coplanar signs without claiming an exact
    // predicate. Leave signs inside its conservative error margin uncertain.
    MeshTwofold d[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            d[i][j] = mesh_exact_difference(component(tri[i], j), component(p, j));
    MeshTwofold total{};
    for (int j = 0; j < 3; ++j) {
        int k = (j + 1) % 3, l = (j + 2) % 3;
        auto x = mesh_twofold_product(d[1][k], d[2][l]);
        auto y = mesh_twofold_product(d[1][l], d[2][k]);
        y.high = -y.high;
        y.low = -y.low;
        total = mesh_twofold_add(total, mesh_twofold_product(d[0][j], mesh_twofold_add(x, y)));
    }
    value = total.high + total.low;
    double bound = 512 * 2.2204460492503131e-16 * 2.2204460492503131e-16 * permanent;
    if (!isfinite(permanent) || permanent < 1e-270 || fabs(value) <= bound)
        return 0;
    return value > 0 ? 1 : -1;
}
__device__ bool mesh_hit_plane_separated(Query &q, int2 hit) {
    V a[3], b[3];
    int3 fa = q.b[0].faces[hit.x], fb = q.b[1].faces[hit.y];
    for (int k = 0; k < 3; ++k) {
        a[k] = q.b[0].vertices[component(fa, k)];
        b[k] = point(q.rel[1], q.b[1].vertices[component(fb, k)]);
    }
    for (int side = 0; side < 2; ++side) {
        const V *plane = side ? b : a, *points = side ? a : b;
        int first = mesh_plane_side(plane, points[0]);
        if (first && mesh_plane_side(plane, points[1]) == first && mesh_plane_side(plane, points[2]) == first)
            return true;
    }
    return false;
}
__device__ void sort_hits(Query &q) {
    auto &w = *q.work;
    auto &m = w.pairs[w.pair];
    int n = w.counts[0];
    if (blockIdx.x)
        return;
    __shared__ int2 local_hits[512];
    __shared__ int local_parents[512];
    int2 *hits = n <= 512 ? local_hits : m.hits;
    int *parents = n <= 512 ? local_parents : w.parents;
    int count = power_two(n);
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        hits[i] = i < n ? m.hits[i] : make_int2(-1, -1);
        parents[i] = i;
    }
    __syncthreads();
    for (int size = 2; size <= count; size *= 2)
        for (int d = size / 2; d; d /= 2) {
            for (int i = threadIdx.x; i < count; i += blockDim.x) {
                int j = i ^ d;
                if (j <= i)
                    continue;
                U64 a = face_key(hits[i]), b = face_key(hits[j]);
                if ((i & size) ? a < b : a > b) {
                    int2 p = hits[i];
                    hits[i] = hits[j];
                    hits[j] = p;
                }
            }
            __syncthreads();
        }
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int2 hit = hits[i];
        for (int side = 0; side < 2; ++side) {
            int f = side ? hit.y : hit.x;
            V a[3], b[3];
            triangle(q, side, f, a);
            triangle(q, 1 - side, side ? hit.x : hit.y, b);
            int3 eids = q.b[side].face_edges[f];
            for (int k = 0; k < 3; ++k) {
                int4 e = q.b[side].edges[component(eids, k)];
                int other = e.z == f ? e.w : e.z;
                int2 adj = hit;
                if (side)
                    adj.y = other;
                else
                    adj.x = other;
                int j = find_pair(hits, n, adj);
                if (j < 0 || j >= i)
                    continue;
                if (segment_triangle(a[k], a[(k + 1) % 3], b))
                    unite(parents, i, j);
            }
        }
    }
    __syncthreads();
    if (n <= 512)
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            m.hits[i] = hits[i];
            w.parents[i] = root(parents, i);
        }
    __syncthreads();
}

__device__ bool edge_support(const BodyView &b, int id, V n) {
    int4 e = b.edges[id];
    V u = sub(b.vertices[e.y], b.vertices[e.x]);
    double tol = 1e-9 * norm(u);
    V a = b.normals[e.z], c = b.normals[e.w];
    return dot(n, cross(a, u)) <= tol && dot(n, cross(u, c)) <= tol && dot(n, plus(a, c)) > 0;
}
__device__ Region feature(const Query &q, U64 key, double bound) {
    Region out{};
    out.depth = NAN;
    int kind = key >> 62, a = (key >> 31) & 0x7fffffff, b = key & 0x7fffffff;
    double eps = 1e-13 * fmin(q.b[0].radius, q.b[1].radius);
    if (kind < 2) {
        int s = kind, t = 1 - s;
        V p = vertex(q, s, a), tri[3];
        triangle(q, t, b, tri);
        V n = normal(q, t, b);
        double d = dot(sub(tri[0], p), n);
        if (d < -eps || d > bound || !inside_triangle(tri, n, p))
            return out;
        V local_n = s ? inverse_direction(q.rel[1], n) : n;
        for (int i = q.b[s].ray_offsets[a]; i < q.b[s].ray_offsets[a + 1]; ++i)
            if (dot(local_n, q.b[s].rays[i]) < -1e-9)
                return out;
        return {plus(p, scale(n, .5 * d)), scale(n, s ? 1. : -1.), {}, d};
    }
    int4 ea = q.b[0].edges[a], eb = q.b[1].edges[b];
    V x = vertex(q, 0, ea.x), y = vertex(q, 1, eb.x), u = sub(vertex(q, 0, ea.y), x),
      v = sub(vertex(q, 1, eb.y), y), off = sub(y, x), c = cross(u, v);
    double n2 = dot(c, c);
    if (n2 <= 1e-20 * dot(u, u) * dot(v, v))
        return out;
    double s = dot(cross(off, v), c) / n2, t = dot(cross(off, u), c) / n2;
    if (s < 0 || s > 1 || t < 0 || t > 1)
        return out;
    x = plus(x, scale(u, s));
    y = plus(y, scale(v, t));
    c = scale(c, 1 / sqrt(n2));
    double gap = dot(c, sub(x, y)), d = fabs(gap);
    if (d > bound)
        return out;
    V n = scale(c, gap > 0 ? 1. : -1.);
    if (!edge_support(q.b[0], a, n) || !edge_support(q.b[1], b, inverse_direction(q.rel[1], scale(n, -1))))
        return out;
    return {scale(plus(x, y), .5), n, {}, d};
}
__device__ void convex_partial(Query &q) {
    auto &w = *q.work;
    auto &p = w.pairs[w.pair];
    int tid = threadIdx.x;
    __shared__ double depth[128];
    __shared__ U64 key[128];
    double best = INFINITY;
    U64 winner = empty_key;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < w.counts[1]; i += gridDim.x * blockDim.x) {
        int2 pair = p.candidates[i];
        const int *source = p.a_links + 3 * pair.x, *target = p.b_links + 3 * pair.y;
        for (int b = source[1]; b < source[1] + source[2]; ++b)
            for (int a = target[1]; a < target[1] + target[2]; ++a) {
                int ia = p.b_ids[a], ib = p.a_ids[b];
                int3 fa = q.b[0].faces[ia], fb = q.b[1].faces[ib], ea = q.b[0].face_edges[ia],
                     eb = q.b[1].face_edges[ib];
                for (int k = 0; k < 15; ++k) {
                    U64 f;
                    if (k < 3)
                        f = (U64(unsigned(component(fa, k))) << 31) | unsigned(ib);
                    else if (k < 6)
                        f = (U64(1) << 62) | (U64(unsigned(component(fb, k - 3))) << 31) | unsigned(ia);
                    else
                        f = (U64(2) << 62) | (U64(unsigned(component(ea, (k - 6) / 3))) << 31) |
                            unsigned(component(eb, (k - 6) % 3));
                    double d = feature(q, f, q.reach).depth;
                    if (d < best || (d == best && f < winner)) {
                        best = d;
                        winner = f;
                    }
                }
            }
    }
    depth[tid] = best;
    key[tid] = winner;
    __syncthreads();
    for (int d = blockDim.x / 2; d; d /= 2) {
        if (tid < d &&
            (depth[tid + d] < depth[tid] || (depth[tid + d] == depth[tid] && key[tid + d] < key[tid]))) {
            depth[tid] = depth[tid + d];
            key[tid] = key[tid + d];
        }
        __syncthreads();
    }
    if (!tid) {
        w.unsorted[blockIdx.x] =
            key[0] == empty_key ? Region{{}, {}, {}, INFINITY} : feature(q, key[0], q.reach);
        w.features[blockIdx.x] = key[0];
    }
}
__device__ void convex_finish(Query &q, int blocks) {
    auto &w = *q.work;
    double best = INFINITY;
    U64 winner = empty_key;
    Region selected{};
    for (int i = 0; i < blocks; ++i) {
        auto r = w.unsorted[i];
        U64 f = w.features[i];
        if (r.depth < best || (r.depth == best && f < winner)) {
            selected = r;
            best = r.depth;
            winner = f;
        }
    }
    w.regions = 0;
    if (!isfinite(best))
        atomicCAS(w.status, 0, 26);
    else if (best > 1e-13 * fmin(q.b[0].radius, q.b[1].radius)) {
        w.result[0] = selected;
        w.regions = 1;
    }
}
template <class Grid> __device__ void convex_support(Grid grid, Query &q) {
    convex_partial(q);
    grid.sync();
    if (!grid.thread_rank())
        convex_finish(q, gridDim.x);
    grid.sync();
}

// Reconstruct a connected straight-ridge support, including nonconvex panel gaps.
__device__ bool line_center(const Query &q, int side, int ridge, int face, V on, double depth, V &out) {
    auto &a = q.b[side];
    auto &b = q.b[1 - side];
    int2 ends = a.ridges[ridge];
    V first = vertex(q, side, ends.x), delta = sub(vertex(q, side, ends.y), first),
      n = normal(q, 1 - side, face);
    double len = norm(delta), tol = 1e-12 * (a.radius + b.radius + norm(q.rel[1].offset));
    if (len <= tol || fabs(dot(delta, n)) > tol)
        return false;
    double p = dot(sub(on, first), delta) / (len * len), eps = tol / len;
    V projected = plus(first, scale(n, depth));
    double low = p, high = p;
    bool found = false, changed = true;
    int panel = b.panels[face], begin = b.panel_offsets[panel], end = b.panel_offsets[panel + 1];
    for (int pass = 0; changed && pass <= end - begin; ++pass) {
        changed = false;
        for (int i = begin; i < end; ++i) {
            V tri[3];
            triangle(q, 1 - side, b.panel_faces[i], tri);
            double lo = 0, hi = 1;
            for (int j = 0; j < 3 && lo <= hi; ++j) {
                V e = sub(tri[(j + 1) % 3], tri[j]), inside = cross(n, e);
                double h = dot(inside, sub(projected, tri[j])), slope = dot(inside, delta),
                       et = tol * norm(e);
                if (fabs(slope) <= et) {
                    if (h < -et)
                        hi = -1;
                } else if (slope > 0)
                    lo = fmax(lo, -h / slope);
                else
                    hi = fmin(hi, -h / slope);
            }
            if (lo <= hi && lo <= high + eps && hi >= low - eps) {
                found = true;
                if (lo < low) {
                    low = lo;
                    changed = true;
                }
                if (hi > high) {
                    high = hi;
                    changed = true;
                }
            }
        }
    }
    if (!found || (high - low) * len <= tol)
        return false;
    out = plus(plus(first, scale(delta, .5 * (low + high))), scale(n, .5 * depth));
    return true;
}
__device__ void center_support(const Query &q, U64 key, Region &r) {
    int kind = key >> 62, a = (key >> 31) & 0x7fffffff, b = key & 0x7fffffff;
    if (kind < 2) {
        int s = kind;
        V p = vertex(q, s, a), center{};
        bool found = false;
        for (int i = q.b[s].ridge_offsets[a]; i < q.b[s].ridge_offsets[a + 1]; ++i) {
            V x;
            if (!line_center(q, s, q.b[s].ridge_ids[i], b, p, r.depth, x))
                continue;
            if (found)
                return;
            center = x;
            found = true;
        }
        if (found)
            r.p = center;
    } else
        for (int s = 0; s < 2; ++s) {
            int ridge = q.b[s].edge_ridge[s ? b : a];
            if (ridge < 0)
                continue;
            int4 edge = q.b[1 - s].edges[s ? a : b];
            V outward = scale(r.n, s ? 1. : -1.), p = sub(r.p, scale(outward, .5 * r.depth));
            for (int j = 0; j < 2; ++j) {
                int face = j ? edge.w : edge.z;
                V d = sub(normal(q, 1 - s, face), outward), center;
                if (dot(d, d) > 1e-18)
                    continue;
                if (line_center(q, s, ridge, face, p, r.depth, center)) {
                    r.p = center;
                    return;
                }
            }
        }
}

__device__ bool ray_first(const BodyView &b, V p, V d, double end, double &distance, V &n) {
    distance = end;
    bool found = false;
    for (int j = 0; j < b.links[0];) {
        const double *box = b.bounds + 6 * j;
        double lo = 0, hi = distance;
        for (int k = 0; k < 3 && lo <= hi; ++k) {
            double x = component(p, k), v = component(d, k);
            if (fabs(v) < 1e-15) {
                if (x < box[k] - 1e-10 || x > box[k + 3] + 1e-10)
                    hi = -1;
            } else {
                double a = (box[k] - 1e-10 - x) / v, c = (box[k + 3] + 1e-10 - x) / v;
                lo = fmax(lo, fmin(a, c));
                hi = fmin(hi, fmax(a, c));
            }
        }
        const int *node = b.links + 3 * j;
        if (lo > hi) {
            j = node[0];
            continue;
        }
        ++j;
        for (int k = node[1]; node[2] && k < node[1] + node[2]; ++k) {
            const V *t = b.packed + 3 * k;
            V e = sub(t[1], t[0]), f = sub(t[2], t[0]), h = cross(d, f);
            double det = dot(e, h);
            if (fabs(det) <= 1e-14 * norm(e) * norm(f))
                continue;
            V off = sub(p, t[0]);
            double u = dot(off, h) / det;
            V c = cross(off, e);
            double v = dot(d, c) / det, s = dot(f, c) / det;
            if (u >= -2e-10 && v >= -2e-10 && u + v <= 1 + 2e-10 && s >= 0 && s <= distance) {
                found = true;
                distance = s;
                n = unit(cross(e, f));
            }
        }
    }
    return found;
}
// Closest finite triangle feature. Angle-weighted vertex and edge pseudonormals
// determine the side of a consistently oriented closed triangle boundary.
struct Foot {
    V point, normal;
    double distance;
    int face;
};
__device__ Foot triangle_foot(const BodyView &b, int face, V p) {
    int3 ids = b.faces[face];
    V a = b.vertices[ids.x], c = b.vertices[ids.y], d = b.vertices[ids.z];
    V u = sub(c, a), v = sub(d, a), w = sub(p, a);
    double d1 = dot(u, w), d2 = dot(v, w);
    V point{}, n{};
    int vertex_id = -1, edge_id = -1;
    if (d1 <= 0 && d2 <= 0) {
        point = a;
        vertex_id = ids.x;
    } else {
        w = sub(p, c);
        double d3 = dot(u, w), d4 = dot(v, w);
        if (d3 >= 0 && d4 <= d3) {
            point = c;
            vertex_id = ids.y;
        } else {
            double vc = d1 * d4 - d3 * d2;
            if (vc <= 0 && d1 >= 0 && d3 <= 0) {
                point = plus(a, scale(u, d1 / (d1 - d3)));
                edge_id = b.face_edges[face].x;
            } else {
                w = sub(p, d);
                double d5 = dot(u, w), d6 = dot(v, w);
                if (d6 >= 0 && d5 <= d6) {
                    point = d;
                    vertex_id = ids.z;
                } else {
                    double vb = d5 * d2 - d1 * d6;
                    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
                        point = plus(a, scale(v, d2 / (d2 - d6)));
                        edge_id = b.face_edges[face].z;
                    } else {
                        double va = d3 * d6 - d5 * d4;
                        if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
                            point = plus(c, scale(sub(d, c), (d4 - d3) / ((d4 - d3) + (d5 - d6))));
                            edge_id = b.face_edges[face].y;
                        } else {
                            double inv = 1 / (va + vb + vc);
                            point = plus(a, plus(scale(u, vb * inv), scale(v, vc * inv)));
                        }
                    }
                }
            }
        }
    }
    if (vertex_id >= 0)
        n = b.vertex_normals[vertex_id];
    else if (edge_id >= 0) {
        auto e = b.edges[edge_id];
        n = plus(b.normals[e.z], b.normals[e.w]);
    } else
        n = b.normals[face];
    V delta = sub(p, point);
    return {point, n, dot(delta, delta), face};
}
__device__ __noinline__ Foot closest_boundary(const BodyView &b, V p) {
    Foot best{{}, {}, INFINITY, -1};
    for (int j = 0; j < b.links[0];) {
        const double *box = b.bounds + 6 * j;
        double distance = 0;
        for (int k = 0; k < 3; ++k) {
            double x = component(p, k), d = fmax(0., fmax(box[k] - x, x - box[k + 3]));
            distance += d * d;
        }
        const int *node = b.links + 3 * j;
        if (distance > best.distance) {
            j = node[0];
            continue;
        }
        ++j;
        for (int k = node[1]; node[2] && k < node[1] + node[2]; ++k) {
            auto f = triangle_foot(b, b.packed_ids[k], p);
            if (f.distance < best.distance || (f.distance == best.distance && f.face < best.face))
                best = f;
        }
    }
    return best;
}
__device__ void check_enclosure(Query &q) {
    auto &w = *q.work;
    int worker = blockIdx.x * blockDim.x + threadIdx.x;
    for (int side = 0; side < 2; ++side) {
        const auto &b = q.b[side];
        if (w.counts[0] && b.ncomponents == 1)
            continue;
        for (int c = worker; c < b.ncomponents; c += gridDim.x * blockDim.x) {
            bool crosses = false;
            for (int i = 0; i < w.counts[0]; ++i) {
                auto h = w.pairs[w.pair].hits[i];
                if (b.components[side ? h.y : h.x] == c) {
                    crosses = true;
                    break;
                }
            }
            if (crosses)
                continue;
            V local = b.vertices[b.component_seeds[c]], target = point(q.rel[side], local);
            auto f = closest_boundary(q.b[1 - side], target);
            V delta = sub(target, f.point);
            double eps = 1e-12 * (q.b[0].radius + q.b[1].radius + norm(q.rel[1].offset));
            if (f.distance > eps * eps && dot(delta, f.normal) < 0)
                atomicCAS(w.status, 0, 25);
        }
    }
}
__device__ __noinline__ Region interior_flood(Query &q, int root_id) {
    auto &w = *q.work;
    int tid = threadIdx.x;
    auto hits = w.pairs[w.pair].hits;
    __shared__ int head, tail, end;
    __shared__ double depths[128];
    __shared__ U64 choices[128];
    __shared__ Region result;
    if (!tid)
        result = {};
    __syncthreads();
    for (int side = 0; side < 2; ++side) {
        auto &b = q.b[side];
        for (int v = tid; v < b.nv; v += blockDim.x)
            w.visited[v] = 0;
        if (!tid)
            head = tail = 0;
        __syncthreads();
        for (int i = tid; i < w.counts[0]; i += blockDim.x)
            if (root(w.parents, i) == root_id) {
                auto h = hits[i];
                auto f = b.faces[side ? h.y : h.x];
                for (int j = 0; j < 3; ++j) {
                    int v = component(f, j);
                    if (!atomicCAS(w.visited + v, 0, 1))
                        w.queue[atomicAdd(&tail, 1)] = v;
                }
            }
        __syncthreads();
        double best = 0;
        U64 choice = empty_key;
        while (head < tail) {
            if (!tid)
                end = tail;
            __syncthreads();
            for (int i = head + tid; i < end; i += blockDim.x) {
                int v = w.queue[i];
                V p = point(q.rel[side], b.vertices[v]);
                auto f = closest_boundary(q.b[1 - side], p);
                if (dot(sub(p, f.point), f.normal) >= 0)
                    continue;
                double depth = sqrt(f.distance);
                if (depth > best || (depth == best && U64(v) < choice)) {
                    best = depth;
                    choice = v;
                }
                for (int j = b.ray_offsets[v]; j < b.ray_offsets[v + 1]; ++j) {
                    int neighbor = b.neighbors[j];
                    if (!atomicCAS(w.visited + neighbor, 0, 1))
                        w.queue[atomicAdd(&tail, 1)] = neighbor;
                }
            }
            __syncthreads();
            if (!tid)
                head = end;
            __syncthreads();
        }
        depths[tid] = best;
        choices[tid] = choice;
        __syncthreads();
        for (int d = blockDim.x / 2; d; d /= 2) {
            if (tid < d && (depths[tid + d] > depths[tid] ||
                            (depths[tid + d] == depths[tid] && choices[tid + d] < choices[tid]))) {
                depths[tid] = depths[tid + d];
                choices[tid] = choices[tid + d];
            }
            __syncthreads();
        }
        if (!tid && depths[0] > result.depth) {
            V local = b.vertices[choices[0]], p = point(q.rel[side], local);
            auto f = closest_boundary(q.b[1 - side], p);
            V outward = inverse_direction(q.rel[side], scale(sub(f.point, p), 1 / depths[0]));
            V center = plus(local, scale(outward, .5 * depths[0]));
            result = {side ? point(q.rel[1], center) : center,
                      side ? direction(q.rel[1], outward) : scale(outward, -1),
                      {},
                      depths[0]};
        }
        __syncthreads();
    }
    return result;
}
__device__ bool spatial_group(Query &q, int root_id) {
    auto &w = *q.work;
    auto hits = w.pairs[w.pair].hits;
    int count = w.counts[0];
    V first{}, tangent{};
    bool spatial = false;
    for (int i = 0; i < count && !spatial; ++i)
        if (root(w.parents, i) == root_id) {
            int2 h = hits[i];
            V a = normal(q, 0, h.x), b = normal(q, 1, h.y);
            if (dot(a, b) >= -.8)
                continue;
            for (int j = 0; j < 2; ++j) {
                V n = j ? b : a;
                if (dot(first, first) == 0)
                    first = n;
                if (dot(tangent, tangent) == 0) {
                    V c = cross(first, n);
                    if (norm(c) > 1e-7)
                        tangent = unit(c);
                } else if (fabs(dot(tangent, n)) > 1e-7)
                    spatial = true;
            }
        }
    return spatial;
}
__device__ Region contour_contribution(Query &q, int2 h) {
    double eps = 1e-12 * (q.b[0].radius + q.b[1].radius + norm(q.rel[1].offset));
    V a[3], b[3];
    triangle(q, 0, h.x, a);
    triangle(q, 1, h.y, b);
    V na = normal(q, 0, h.x), nb = normal(q, 1, h.y), axis = cross(na, nb);
    if (norm(axis) < 1e-10)
        return {};
    axis = unit(axis);
    double low[2] = {INFINITY, INFINITY}, high[2] = {-INFINITY, -INFINITY};
    for (int side = 0; side < 2; ++side) {
        V *t = side ? b : a, *plane = side ? a : b;
        V n = side ? na : nb;
        double heights[3];
        for (int k = 0; k < 3; ++k)
            heights[k] = dot(sub(t[k], plane[0]), n);
        for (int k = 0; k < 3; ++k) {
            int j = (k + 1) % 3;
            if (fabs(heights[k]) <= eps) {
                double x = dot(sub(t[k], a[0]), axis);
                low[side] = fmin(low[side], x);
                high[side] = fmax(high[side], x);
            }
            if ((heights[k] < 0) != (heights[j] < 0)) {
                V p = plus(t[k], scale(sub(t[j], t[k]), heights[k] / (heights[k] - heights[j])));
                double x = dot(sub(p, a[0]), axis);
                low[side] = fmin(low[side], x);
                high[side] = fmax(high[side], x);
            }
        }
    }
    double lo = fmax(low[0], low[1]), hi = fmin(high[0], high[1]), length = hi - lo;
    if (!(length > eps) || !isfinite(length))
        return {};
    V transverse = cross(axis, na);
    double offset = dot(sub(b[0], a[0]), nb) / dot(transverse, nb);
    V mid = plus(plus(a[0], scale(transverse, offset)), scale(axis, .5 * (lo + hi)));
    return {scale(mid, length), dot(na, nb) < -.8 ? scale(sub(na, nb), length) : V{}, {}, length};
}
__device__ bool finish_contour(Query &q, V center, V combined, double perimeter, Region &out) {
    double eps = 1e-12 * (q.b[0].radius + q.b[1].radius + norm(q.rel[1].offset));
    if (perimeter <= eps || norm(combined) <= eps)
        return false;
    center = scale(center, 1 / perimeter);
    combined = unit(combined);
    double offset[2];
    for (int side = 0; side < 2; ++side) {
        V p = side ? inverse_point(q.rel[1], center) : center,
          n = side ? inverse_direction(q.rel[1], scale(combined, -1)) : combined, hit_n;
        double t;
        if (!ray_first(q.b[side], plus(p, scale(n, q.reach)), scale(n, -1), 2 * q.reach, t, hit_n) ||
            dot(hit_n, n) <= 0)
            return false;
        offset[side] = q.reach - t;
    }
    double d = offset[0] + offset[1];
    if (d <= eps || d > q.reach)
        return false;
    out = {plus(center, scale(combined, .5 * (offset[0] - offset[1]))), combined, {}, d};
    return true;
}
__device__ __noinline__ bool contour_contact(Query &q, int root_id, Region &out) {
    auto &w = *q.work;
    auto hits = w.pairs[w.pair].hits;
    int tid = threadIdx.x;
    __shared__ int spatial, valid;
    __shared__ Region answer;
    Region *values = w.contributions + blockIdx.x * w.capacity;
    if (!tid) {
        spatial = spatial_group(q, root_id);
        valid = 0;
    }
    __syncthreads();
    if (!spatial)
        return false;
#ifdef NC_SERIAL_MESH_CONTOUR
    if (!tid)
        for (int i = 0; i < w.counts[0]; ++i)
            if (root(w.parents, i) == root_id)
                values[i] = contour_contribution(q, hits[i]);
#else
    for (int i = tid; i < w.counts[0]; i += blockDim.x)
        if (root(w.parents, i) == root_id)
            values[i] = contour_contribution(q, hits[i]);
#endif
    __syncthreads();
    if (!tid) {
        double perimeter = 0;
        V center{}, combined{};
        for (int i = 0; i < w.counts[0]; ++i)
            if (root(w.parents, i) == root_id) {
                auto v = values[i];
                center = plus(center, v.p);
                combined = plus(combined, v.n);
                perimeter += v.depth;
            }
        valid = finish_contour(q, center, combined, perimeter, answer);
    }
    __syncthreads();
    if (valid)
        out = answer;
    return valid;
}
__device__ __noinline__ void component_support(Query &q, int id) {
    auto &w = *q.work;
    auto hits = w.pairs[w.pair].hits;
    int tid = threadIdx.x;
    __shared__ int nh, np, nf, npanels, done;
    __shared__ double reduced[128];
    __shared__ U64 winners[128];
    __shared__ double cutoff;
    if (!tid) {
        nh = np = nf = npanels = done = 0;
        w.region_valid[id] = 0;
        for (int i = 0; i < w.counts[0]; ++i)
            nh += root(w.parents, i) == id;
    }
    __syncthreads();
    Region r;
    bool contact = contour_contact(q, id, r);
    if (!tid && contact) {
        w.unsorted[id] = r;
        w.region_valid[id] = 1;
        done = 1;
    }
    __syncthreads();
    if (done)
        return;
    if (7 * nh > w.patch_capacity) {
        if (!tid)
            atomicCAS(w.status, 0, 21);
        return;
    }
    U64 *ph = w.patch_hash + blockIdx.x * 2 * w.patch_capacity,
        *patch = w.patches + blockIdx.x * w.patch_capacity;
    U64 *fh = w.feature_hash + blockIdx.x * 2 * w.feature_capacity,
        *features = w.features + blockIdx.x * w.feature_capacity;
    int hash_size = power_two(14 * nh);
    for (int i = tid; i < hash_size; i += blockDim.x)
        ph[i] = empty_key;
    __syncthreads();
    for (int i = tid; i < w.counts[0]; i += blockDim.x)
        if (root(w.parents, i) == id) {
            int2 pair = hits[i];
            for (int k = 0; k < 7; ++k) {
                int2 p = pair;
                if (k) {
                    int side = (k - 1) / 3, j = (k - 1) % 3, f = side ? pair.y : pair.x;
                    int4 e = q.b[side].edges[component(q.b[side].face_edges[f], j)];
                    int other = e.z == f ? e.w : e.z;
                    if (side)
                        p.y = other;
                    else
                        p.x = other;
                }
                U64 key = face_key(p);
                if (insert(ph, hash_size, key))
                    patch[atomicAdd(&np, 1)] = key;
            }
        }
    __syncthreads();
    // Unique planar panels avoid testing the same separating direction per triangle.
    int panel_hash = power_two(4 * np);
    for (int i = tid; i < panel_hash; i += blockDim.x)
        fh[i] = empty_key;
    __syncthreads();
    for (int i = tid; i < np; i += blockDim.x)
        for (int side = 0; side < 2; ++side) {
            int f = side ? unsigned(patch[i]) : patch[i] >> 32;
            U64 key = (U64(side) << 63) | unsigned(q.b[side].panels[f]);
            if (insert(fh, panel_hash, key))
                features[atomicAdd(&npanels, 1)] = key;
        }
    __syncthreads();
    double bound = INFINITY;
    for (int i = tid; i < npanels; i += blockDim.x) {
        U64 key = features[i];
        int side = key >> 63, panel = unsigned(key);
        int face = q.b[side].panel_faces[q.b[side].panel_offsets[panel]];
        V n = normal(q, side, face);
        double low[2] = {INFINITY, INFINITY}, high[2] = {-INFINITY, -INFINITY};
        for (int j = 0; j < np; ++j)
            for (int s = 0; s < 2; ++s) {
                int f = s ? unsigned(patch[j]) : patch[j] >> 32;
                V tri[3];
                triangle(q, s, f, tri);
                for (V p : tri) {
                    double h = dot(n, p);
                    low[s] = fmin(low[s], h);
                    high[s] = fmax(high[s], h);
                }
            }
        bound = fmin(bound, fmin(high[0] - low[1], high[1] - low[0]));
    }
    reduced[tid] = bound;
    __syncthreads();
    for (int s = blockDim.x / 2; s; s /= 2) {
        if (tid < s)
            reduced[tid] = fmin(reduced[tid], reduced[tid + s]);
        __syncthreads();
    }
    if (!tid)
        cutoff = reduced[0] + 1e-12 * (q.b[0].radius + q.b[1].radius + norm(q.rel[1].offset));
    if (15 * np > w.feature_capacity) {
        if (!tid)
            atomicCAS(w.status, 0, 22);
        return;
    }
    int feature_hash = power_two(30 * np);
    for (int i = tid; i < feature_hash; i += blockDim.x)
        fh[i] = empty_key;
    __syncthreads();
    for (int i = tid; i < np; i += blockDim.x) {
        int a = patch[i] >> 32, b = unsigned(patch[i]);
        int3 fa = q.b[0].faces[a], fb = q.b[1].faces[b], ea = q.b[0].face_edges[a], eb = q.b[1].face_edges[b];
        for (int k = 0; k < 15; ++k) {
            U64 key;
            if (k < 3)
                key = (U64(unsigned(component(fa, k))) << 31) | unsigned(b);
            else if (k < 6)
                key = (U64(1) << 62) | (U64(unsigned(component(fb, k - 3))) << 31) | unsigned(a);
            else
                key = (U64(2) << 62) | (U64(unsigned(component(ea, (k - 6) / 3))) << 31) |
                      unsigned(component(eb, (k - 6) % 3));
            if (insert(fh, feature_hash, key))
                features[atomicAdd(&nf, 1)] = key;
        }
    }
    __syncthreads();
    double best = 0;
    U64 choice = empty_key;
    for (int i = tid; i < nf; i += blockDim.x) {
        auto r = feature(q, features[i], cutoff);
        if (r.depth > best || (r.depth == best && features[i] < choice)) {
            best = r.depth;
            choice = features[i];
        }
    }
    reduced[tid] = best;
    winners[tid] = choice;
    __syncthreads();
    for (int s = blockDim.x / 2; s; s /= 2) {
        if (tid < s && (reduced[tid + s] > reduced[tid] ||
                        (reduced[tid + s] == reduced[tid] && winners[tid + s] < winners[tid]))) {
            reduced[tid] = reduced[tid + s];
            winners[tid] = winners[tid + s];
        }
        __syncthreads();
    }
    if (!tid && reduced[0] > 1e-13 * fmin(q.b[0].radius, q.b[1].radius)) {
        Region r = feature(q, winners[0], cutoff);
        center_support(q, winners[0], r);
        w.unsorted[id] = r;
        w.region_valid[id] = 1;
    }
    __syncthreads();
    if (!w.region_valid[id] && w.groups == 1) {
        auto r = interior_flood(q, id);
        if (!tid && r.depth > 1e-13 * fmin(q.b[0].radius, q.b[1].radius)) {
            if (r.depth > q.reach)
                atomicCAS(w.status, 0, 26);
            else {
                w.unsorted[id] = r;
                w.region_valid[id] = 1;
            }
        }
    }
    __syncthreads();
    // An intersecting component without a finite or zero-depth witness is
    // unresolved. Never silently drop it and publish a partial contact load.
    // A nonpositive local separating-plane bound certifies a zero-depth
    // tangent patch, even when no vertex/edge witness has positive depth.
    double zero_tolerance = 1e-12 * (q.b[0].radius + q.b[1].radius + norm(q.rel[1].offset));
    if (!tid && !w.region_valid[id] && winners[0] == empty_key && cutoff > 2 * zero_tolerance)
        atomicCAS(w.status, 0, 27);
}

template <class Grid> __device__ void detect(Grid grid, Query &q, PairScratch &) {
    auto &w = *q.work;
    int worker = grid.thread_rank();
    auto &p = w.pairs[w.pair];
    unsigned long long stamp = worker ? 0 : timer_ns();
    bool convex = q.b[0].convex && q.b[1].convex;
    if (!worker) {
        ++w.queries;
        w.counts[0] = w.counts[1] = w.filtered = w.refresh = 0;
        for (int k = 0; k < 3; ++k) {
            for (int j = 0; j < 3; ++j)
                w.pose[4 * k + j] = component(q.rel[1].cols[j], k);
            w.pose[4 * k + 3] = component(q.rel[1].offset, k);
        }
    }
    grid.sync();
    for (int first = 0; first < p.leaves_n; first += gridDim.x * blockDim.x)
        mesh_detect::collect_leaves(p.leaves + first, p.leaves_n - first, p.bounds, p.traversal, p.nodes,
                                    w.pose, p.candidates, p.candidate_n, w.counts, convex ? q.reach : 0.);
    grid.sync();
    if (!worker) {
        w.broad_ns += timer_ns() - stamp;
        stamp = timer_ns();
    }
    if (w.counts[1] > p.candidate_n) {
        if (!worker)
            atomicCAS(w.status, 0, 23);
        grid.sync();
        return;
    }
    mesh_detect::batch_filter(&p, w.pose, w.counts, &w.filtered, &w.refresh, 0.);
    grid.sync();
    mesh_detect::batch_group_intersections(&p, w.pose, w.counts, &w.filtered, 0., 8);
    grid.sync();
    if (!worker) {
        w.narrow_ns += timer_ns() - stamp;
        stamp = timer_ns();
    }
    if (w.counts[0] > p.hit_n) {
        if (!worker)
            atomicCAS(w.status, 0, 24);
        grid.sync();
        return;
    }
    for (int i = worker; i < w.counts[0]; i += grid.size()) {
        auto hit = p.hits[i];
        p.hits[i] = make_int2(hit.y, hit.x);
    }
    grid.sync();
    for (int i = worker; i < w.counts[0]; i += grid.size())
        if (!mesh_hit_plane_separated(q, p.hits[i]))
            p.filtered[atomicAdd(&w.refresh, 1)] = p.hits[i];
    grid.sync();
    if (!worker)
        w.counts[0] = w.refresh;
    grid.sync();
    for (int i = worker; i < w.counts[0]; i += grid.size())
        p.hits[i] = p.filtered[i];
    grid.sync();
    check_enclosure(q);
    grid.sync();
    if (*w.status)
        return;
    if (!w.counts[0]) {
        if (!worker)
            w.regions = 0;
        grid.sync();
        return;
    }
    for (int i = worker; i < q.b[1].nv; i += grid.size())
        w.placed_vertices[i] = point(q.rel[1], q.b[1].vertices[i]);
    for (int i = worker; i < q.b[1].nf; i += grid.size())
        w.placed_normals[i] = direction(q.rel[1], q.b[1].normals[i]);
    grid.sync();
    if (convex) {
        convex_support(grid, q);
        if (!worker)
            w.support_ns += timer_ns() - stamp;
        grid.sync();
        return;
    }
    sort_hits(q);
    grid.sync();
    if (!worker) {
        w.group_ns += timer_ns() - stamp;
        stamp = timer_ns();
    }
    if (!worker) {
        w.groups = 0;
        for (int i = 0; i < w.counts[0]; ++i)
            w.groups += root(w.parents, i) == i;
    }
    grid.sync();
    int support_blocks = min(int(gridDim.x), 8);
    if (blockIdx.x < support_blocks)
        for (int i = blockIdx.x; i < w.counts[0]; i += support_blocks) {
            if (root(w.parents, i) == i)
                component_support(q, i);
        }
    grid.sync();
    if (!worker) {
        w.regions = 0;
        double eps = 1e-12 * (q.b[0].radius + q.b[1].radius + norm(q.rel[1].offset));
        for (int i = 0; i < w.counts[0]; ++i)
            if (root(w.parents, i) == i && w.region_valid[i]) {
                auto r = w.unsorted[i];
                bool same = false;
                for (int j = 0; j < w.regions; ++j) {
                    auto old = w.result[j];
                    V p = sub(r.p, old.p), n = sub(r.n, old.n);
                    same |= fabs(r.depth - old.depth) <= eps && dot(p, p) <= eps * eps && dot(n, n) <= 1e-18;
                }
                if (!same)
                    w.result[w.regions++] = r;
            }
        w.support_ns += timer_ns() - stamp;
    }
    grid.sync();
}
template <class Grid> __device__ void integrate_regions(Grid, Query &) {}
