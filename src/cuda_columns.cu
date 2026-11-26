#include "cuda_columns.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace nc::compiled::device {
namespace {
void check(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA CAD: ") + cudaGetErrorString(e));
}
template <class T> struct Buffer {
    T *ptr{};
    size_t capacity{};
    ~Buffer() { cudaFree(ptr); }
    void reserve(size_t n) {
        if (n <= capacity)
            return;
        T *replacement{};
        check(cudaMalloc(&replacement, n * sizeof(T)));
        cudaFree(ptr);
        ptr = replacement;
        capacity = n;
    }
    void upload(const std::vector<T> &src) {
        reserve(src.size());
        if (!src.empty())
            check(cudaMemcpy(ptr, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
};
struct Stream {
    cudaStream_t id{};
    Stream() { check(cudaStreamCreateWithFlags(&id, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(id); }
};
struct View {
    const Node *faces, *spans;
    const FaceData *charts;
    const Span *patches;
    const SpanFrame *frames;
    const Cell *cells;
    int n;
};
__device__ V sub(V a, V b) { return {{a.x[0] - b.x[0], a.x[1] - b.x[1], a.x[2] - b.x[2]}}; }
__device__ V scale(V a, double b) { return {{a.x[0] * b, a.x[1] * b, a.x[2] * b}}; }
__device__ double dot(V a, V b) { return a.x[0] * b.x[0] + a.x[1] * b.x[1] + a.x[2] * b.x[2]; }
__device__ double norm(V a) { return sqrt(dot(a, a)); }
__device__ V cross(V a, V b) {
    return {{a.x[1] * b.x[2] - a.x[2] * b.x[1], a.x[2] * b.x[0] - a.x[0] * b.x[2],
             a.x[0] * b.x[1] - a.x[1] * b.x[0]}};
}
__device__ V basis(const V *f, V a) { return {{dot(f[0], a), dot(f[1], a), dot(f[2], a)}}; }
__device__ bool intersects(Bounds b, Ray r, double end, double pad) {
    double lo = -pad, hi = end + pad;
    for (int i = 0; i < 3; ++i) {
        double l = b.lo.x[i] - r.p.x[i] - pad, h = b.hi.x[i] - r.p.x[i] + pad;
        if (fabs(r.d.x[i]) < 1e-18) {
            if (l > 0 || h < 0)
                return false;
        } else {
            double inv = 1 / r.d.x[i], a = l * inv, c = h * inv;
            lo = fmax(lo, fmin(a, c));
            hi = fmin(hi, fmax(a, c));
            if (lo > hi)
                return false;
        }
    }
    return true;
}
struct Jet {
    V p, u, v;
};
__device__ Jet at(const Span &s, double u, double v) {
    const auto r = s.range;
    const double du = r.U - r.u, dv = r.Vv - r.v, x = (u - r.u) / du, y = (v - r.v) / dv;
    Jet q;
    for (int c = 0; c < 3; ++c) {
        double a[4], b[4];
        for (int i = 0; i < 4; ++i) {
            const V *v0 = s.power + 4 * i;
            a[i] = ((v0[3].x[c] * y + v0[2].x[c]) * y + v0[1].x[c]) * y + v0[0].x[c];
            b[i] = (3 * v0[3].x[c] * y + 2 * v0[2].x[c]) * y + v0[1].x[c];
        }
        q.p.x[c] = ((a[3] * x + a[2]) * x + a[1]) * x + a[0];
        q.u.x[c] = ((3 * a[3] * x + 2 * a[2]) * x + a[1]) / du;
        q.v.x[c] = (((b[3] * x + b[2]) * x + b[1]) * x + b[0]) / dv;
    }
    return q;
}
__device__ double spread(V a, V b) {
    return fabs(a.x[0]) * b.x[0] + fabs(a.x[1]) * b.x[1] + fabs(a.x[2]) * b.x[2];
}
// The unit-cell contraction encloses every root. When it cannot decide the
// column, retain its exact source/target ray for the general subdivision path.
__device__ int isolate(const Span &s, const Cell &c, Ray ray, double tol, double end, double &t, V &normal) {
    V a = cross(c.sv, ray.d), b = cross(ray.d, c.su);
    double det = dot(c.su, a);
    if (det == 0 || fabs(det) <= 1e-14 * norm(c.su) * norm(c.sv))
        return -2;
    a = scale(a, 1 / det);
    b = scale(b, 1 / det);
    double eu = spread(a, c.ru) + spread(a, c.rv) + 1e-12, ev = spread(b, c.ru) + spread(b, c.rv) + 1e-12;
    if (fmax(eu, ev) >= .93)
        return -2;
    V f = sub(c.center, ray.p);
    double x = .5 - dot(a, f), y = .5 - dot(b, f);
    double pu = tol * norm(a) + 1e-11, pv = tol * norm(b) + 1e-11;
    if (x + eu * .5 < -pu || x - eu * .5 > 1 + pu || y + ev * .5 < -pv || y - ev * .5 > 1 + pv)
        return -1;
    if (x < 0 || x > 1 || y < 0 || y > 1) {
        double lo[2]{0, 0}, hi[2]{1, 1};
        const double e[2][2]{{spread(a, c.ru) + 1e-12, spread(a, c.rv) + 1e-12},
                             {spread(b, c.ru) + 1e-12, spread(b, c.rv) + 1e-12}};
        for (int pass = 0; pass < 4; ++pass) {
            double cx = (lo[0] + hi[0]) * .5, cy = (lo[1] + hi[1]) * .5, hx = (hi[0] - lo[0]) * .5,
                   hy = (hi[1] - lo[1]) * .5;
            const auto r = c.range;
            const auto delta = sub(at(s, r.u + (r.U - r.u) * cx, r.v + (r.Vv - r.v) * cy).p, ray.p);
            double centers[2]{cx - dot(a, delta), cy - dot(b, delta)};
            for (int k = 0; k < 2; ++k) {
                double radius = e[k][0] * hx + e[k][1] * hy + (k ? pv : pu);
                lo[k] = fmax(lo[k], centers[k] - radius);
                hi[k] = fmin(hi[k], centers[k] + radius);
                if (lo[k] > hi[k])
                    return -1;
            }
        }
        x = fmin(fmax(x, lo[0]), hi[0]);
        y = fmin(fmax(y, lo[1]), hi[1]);
    }
    for (int i = 0; i < 9; ++i) {
        const auto r = c.range;
        double du = r.U - r.u, dv = r.Vv - r.v;
        auto jet = at(s, r.u + x * du, r.v + y * dv);
        f = sub(jet.p, ray.p);
        t = dot(f, ray.d);
        f = sub(f, scale(ray.d, t));
        if (dot(f, f) < 1e-20) {
            if (t < -tol || t > end + tol)
                return -1;
            normal = cross(jet.u, jet.v);
            double mag = norm(normal);
            if (mag < 1e-22)
                return -1;
            normal = scale(normal, 1 / mag);
            return 0;
        }
        a = cross(jet.v, ray.d);
        b = cross(ray.d, jet.u);
        det = dot(jet.u, a);
        if (det == 0)
            return -2;
        x -= dot(f, a) / (det * du);
        y -= dot(f, b) / (det * dv);
        if (x < 0 || x > 1 || y < 0 || y > 1)
            return -2;
    }
    return -2;
}
__device__ Hit first(View g, Ray ray, double end) {
    Hit best{0, -1};
    V normal{};
    for (int i = 0; i < g.n;) {
        const auto n = g.faces[i];
        if (!intersects(n.box, ray, end, 1e-6)) {
            i = n.next;
            continue;
        }
        ++i;
        if (n.leaf < 0)
            continue;
        const auto f = g.charts[n.leaf];
        if (f.first < 0) {
            if (f.kind == 1) {
                auto normal = cross(f.frame[0], f.frame[1]);
                double den = dot(normal, ray.d);
                if (fabs(den) < 1e-14)
                    continue;
                double t = -dot(normal, sub(ray.p, f.origin)) / den;
                if (t < -f.tol || t > end + f.tol)
                    continue;
            } else if (f.kind == 2) {
                auto p = basis(f.frame, sub(ray.p, f.origin)), d = basis(f.frame, ray.d);
                double a = p.x[0], b = p.x[1], h = hypot(a, b), speed = hypot(d.x[0], d.x[1]);
                if (fabs(h - f.radius) > speed * (end + 2 * f.tol) + 2 * f.tol)
                    continue;
            }
            return {0, -2};
        }
        Ray local{basis(f.frame, ray.p), basis(f.frame, ray.d)};
        for (int j = f.first; j < f.end;) {
            const auto node = g.spans[j];
            if (!intersects(node.box, local, end, 2 * f.tol)) {
                j = node.next;
                continue;
            }
            ++j;
            if (node.leaf < 0)
                continue;
            const auto &s = g.patches[node.leaf];
            if (s.cell < 0)
                return {0, -2};
            const auto frame = g.frames[node.leaf];
            const auto &c = g.cells[s.cell];
            Ray cellray{basis(frame.axes, sub(ray.p, frame.origin)), basis(frame.axes, ray.d)};
            if (!intersects(c.box, cellray, end, 2 * f.tol))
                continue;
            double t;
            V new_normal;
            int result = isolate(s, c, ray, f.tol, end, t, new_normal);
            if (result == -2)
                return {0, -2};
            if (result == -1 || t > end)
                continue;
            if (best.face >= 0 && fabs(best.t - t) <= f.tol)
                return {0, -2};
            if (best.face < 0 || t < best.t - f.tol) {
                normal = scale(new_normal, f.orientation);
                best = {t, n.leaf};
                end = t + f.tol;
            }
        }
    }
    if (best.face >= 0 && (best.t <= 1e-7 || dot(normal, ray.d) <= 1e-8))
        return {0, -1};
    return best;
}
__global__ void columns(View view, const Ray *rays, double reach, Hit *hits, int n) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        hits[i] = first(view, rays[i], reach);
}
struct Leaf {
    int site, tile;
};
__device__ V direction(Pose pose, V a) {
    V b;
    for (int k = 0; k < 3; ++k)
        b.x[k] = pose.cols[0].x[k] * a.x[0] + pose.cols[1].x[k] * a.x[1] + pose.cols[2].x[k] * a.x[2];
    return b;
}
__device__ V plus(V a, V b) { return {{a.x[0] + b.x[0], a.x[1] + b.x[1], a.x[2] + b.x[2]}}; }
__device__ V point(Pose p, V a) { return plus(p.offset, direction(p, a)); }
__device__ Bounds placed(Pose p, Bounds b) {
    V c = point(p, scale(plus(b.lo, b.hi), .5)), h = scale(sub(b.hi, b.lo), .5), e{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            e.x[i] += fabs(p.cols[j].x[i]) * h.x[j];
    return {sub(c, e), plus(c, e)};
}
__device__ bool overlaps(Bounds b, Bounds a, double pad) {
    for (int k = 0; k < 3; ++k)
        if (b.lo.x[k] > a.hi.x[k] + pad || b.hi.x[k] + pad < a.lo.x[k])
            return false;
    return true;
}
__global__ void build_window(const Tile *source, const Node *tree, int nodes, const Tile *target,
                             const int *ids, int n, Pose pose, double reach, double turn, Leaf *out,
                             int capacity, int *count) {
    int t = threadIdx.x + blockIdx.x * blockDim.x;
    if (t >= n)
        return;
    int tile = ids[t];
    const auto ref = target[tile];
    auto box = placed(pose, ref.box);
    auto normal = direction(pose, ref.normal), p = point(pose, ref.point);
    for (int i = 0; i < nodes;) {
        const auto node = tree[i];
        if (!overlaps(box, node.box, 2 * reach + 1e-7)) {
            i = node.next;
            continue;
        }
        ++i;
        if (node.leaf < 0)
            continue;
        const auto s = source[node.leaf];
        const auto v = sub(s.point, p);
        if (-dot(normal, s.normal) + turn + 1e-12 >= .5 &&
            fabs(dot(v, normal)) <= reach * (1 + sqrt(3.)) + norm(v) * turn + 1e-7) {
            int slot = atomicAdd(count, 1);
            if (slot < capacity)
                out[slot] = {node.leaf, tile};
        }
    }
}
__global__ void score_window(const Tile *source, const Tile *target, const Leaf *list, int n, Pose pose,
                             double reach, Candidate *out, int capacity, int *count) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n)
        return;
    auto leaf = list[i];
    const auto s = source[leaf.site], t = target[leaf.tile];
    if (!overlaps(placed(pose, t.box), {s.point, s.point}, reach + 1e-7))
        return;
    V normal = direction(pose, t.normal);
    double cosine = -dot(s.normal, normal);
    if (cosine < .5)
        return;
    double d = -dot(sub(s.point, point(pose, t.point)), normal) / cosine;
    if (fabs(d) > reach)
        return;
    int slot = atomicAdd(count, 1);
    if (slot < capacity)
        out[slot] = {leaf.site, leaf.tile, d};
}
struct Counts {
    int traced, written;
};
__global__ void pressure_columns(View a, View b, const RuleBlock *jobs, int count, const int *targets,
                                 Pose forward, Pose reverse, double reach, Counts *stats, Depth *out,
                                 int capacity) {
    int thread = threadIdx.x + blockIdx.x * blockDim.x, j = thread / 8, local = thread % 8;
    if (j >= count)
        return;
    auto job = jobs[j];
    if (local >= job.count)
        return;
    auto site = job.sites[job.first + local];
    auto pose = job.side ? reverse : forward;
    auto target = job.side ? a : b;
    Ray ray{point(pose, site.point), direction(pose, scale(site.normal, -1))};
    bool possible = false;
    for (int i = job.targets; i < job.end && !possible; ++i)
        possible = intersects(target.charts[targets[i]].box, ray, reach, 1e-7);
    if (!possible)
        return;
    atomicAdd(&stats->traced, 1);
    auto hit = first(target, ray, reach);
    bool save = hit.face == -2;
    for (int i = job.targets; i < job.end && !save; ++i)
        save = hit.face == targets[i];
    if (save) {
        int k = atomicAdd(&stats->written, 1);
        if (k < capacity)
            out[k] = {hit.face == -2 ? -1. : hit.t, j, local};
    }
}
} // namespace
struct Columns::Storage {
    Buffer<Node> faces, spans;
    Buffer<FaceData> charts;
    Buffer<Span> patches;
    Buffer<SpanFrame> frames;
    Buffer<Cell> cells;
    Buffer<Ray> rays;
    Buffer<Hit> hits;
    Stream stream;
    View view{};
    Buffer<Tile> tiles;
    Buffer<Node> sites;
    Buffer<int> ids, count;
    Buffer<Leaf> staging;
    Buffer<Candidate> scores;
    Buffer<RuleBlock> blocks;
    Buffer<int> targets;
    Buffer<Depth> active;
    Buffer<Counts> stats;
};
struct Columns::Window {
    Buffer<Leaf> leaves;
    int count{};
};
Sites::Sites(const std::vector<RuleSite> &sites) {
    if (sites.empty())
        return;
    Buffer<RuleSite> storage;
    storage.upload(sites);
    ptr = storage.ptr;
    storage.ptr = nullptr;
}
Sites::~Sites() { cudaFree(ptr); }
Columns::Columns(const Tables &t) {
    auto s = std::make_unique<Storage>();
    s->faces.upload(t.faces);
    s->spans.upload(t.spans);
    s->charts.upload(t.charts);
    s->patches.upload(t.patches);
    s->frames.upload(t.frames);
    s->cells.upload(t.cells);
    s->tiles.upload(t.tiles);
    s->sites.upload(t.site_tree);
    s->ids.upload(t.target_ids);
    s->count.reserve(1);
    s->view = {s->faces.ptr,
               s->spans.ptr,
               s->charts.ptr,
               s->patches.ptr,
               s->frames.ptr,
               s->cells.ptr,
               static_cast<int>(t.faces.size())};
    data = s.release();
}
Columns::~Columns() { delete data; }
size_t Columns::integrate(const Columns &other, const std::vector<RuleBlock> &blocks,
                          const std::vector<int> &targets, Pose forward, Pose reverse, double reach,
                          std::vector<Depth> &out) {
    auto &s = *data;
    out.clear();
    if (blocks.empty())
        return 0;
    s.blocks.reserve(blocks.size());
    s.targets.reserve(targets.size());
    s.stats.reserve(1);
    s.active.reserve(128);
    auto stream = s.stream.id;
    check(cudaMemcpyAsync(s.blocks.ptr, blocks.data(), sizeof(RuleBlock) * blocks.size(),
                          cudaMemcpyHostToDevice, stream));
    check(cudaMemcpyAsync(s.targets.ptr, targets.data(), sizeof(int) * targets.size(), cudaMemcpyHostToDevice,
                          stream));
    for (;;) {
        out.resize(s.active.capacity);
        Counts counts{};
        check(cudaMemsetAsync(s.stats.ptr, 0, sizeof(Counts), stream));
        pressure_columns<<<(blocks.size() * 8 + 127) / 128, 128, 0, stream>>>(
            s.view, other.data->view, s.blocks.ptr, blocks.size(), s.targets.ptr, forward, reverse, reach,
            s.stats.ptr, s.active.ptr, s.active.capacity);
        check(cudaGetLastError());
        check(cudaMemcpyAsync(&counts, s.stats.ptr, sizeof(Counts), cudaMemcpyDeviceToHost, stream));
        check(cudaMemcpyAsync(out.data(), s.active.ptr, sizeof(Depth) * out.size(), cudaMemcpyDeviceToHost,
                              stream));
        check(cudaStreamSynchronize(stream));
        if (counts.written <= s.active.capacity) {
            out.resize(counts.written);
            return counts.traced;
        }
        s.active.reserve(counts.written);
    }
}
std::shared_ptr<Columns::Window> Columns::window(const Columns &target, Pose pose, double reach, double turn,
                                                 size_t &bytes) {
    auto &s = *data;
    const auto &t = *target.data;
    auto w = std::make_shared<Window>();
    if (t.ids.capacity && s.sites.capacity) {
        s.staging.reserve(2048);
        for (;;) {
            check(cudaMemsetAsync(s.count.ptr, 0, sizeof(int), s.stream.id));
            build_window<<<(t.ids.capacity + 127) / 128, 128, 0, s.stream.id>>>(
                s.tiles.ptr, s.sites.ptr, s.sites.capacity, t.tiles.ptr, t.ids.ptr, t.ids.capacity, pose,
                reach, turn, s.staging.ptr, s.staging.capacity, s.count.ptr);
            check(cudaGetLastError());
            check(cudaMemcpyAsync(&w->count, s.count.ptr, sizeof(int), cudaMemcpyDeviceToHost, s.stream.id));
            check(cudaStreamSynchronize(s.stream.id));
            if (w->count <= s.staging.capacity)
                break;
            s.staging.reserve(w->count);
        }
        if (w->count) {
            w->leaves.reserve(w->count);
            check(cudaMemcpyAsync(w->leaves.ptr, s.staging.ptr, w->count * sizeof(Leaf),
                                  cudaMemcpyDeviceToDevice, s.stream.id));
        }
    }
    bytes = w->count * sizeof(Leaf);
    return w;
}
void Columns::candidates(const Columns &target, const Window &w, Pose pose, double reach,
                         std::vector<Candidate> &out) {
    auto &s = *data;
    out.clear();
    if (!w.count)
        return;
    s.scores.reserve(128);
    for (;;) {
        out.resize(s.scores.capacity);
        check(cudaMemsetAsync(s.count.ptr, 0, sizeof(int), s.stream.id));
        score_window<<<(w.count + 127) / 128, 128, 0, s.stream.id>>>(
            s.tiles.ptr, target.data->tiles.ptr, w.leaves.ptr, w.count, pose, reach, s.scores.ptr,
            s.scores.capacity, s.count.ptr);
        check(cudaGetLastError());
        int n;
        check(cudaMemcpyAsync(&n, s.count.ptr, sizeof(n), cudaMemcpyDeviceToHost, s.stream.id));
        check(cudaMemcpyAsync(out.data(), s.scores.ptr, out.size() * sizeof(Candidate),
                              cudaMemcpyDeviceToHost, s.stream.id));
        check(cudaStreamSynchronize(s.stream.id));
        if (n <= s.scores.capacity) {
            out.resize(n);
            return;
        }
        s.scores.reserve(n);
    }
}
void Columns::trace(const std::vector<Ray> &rays, double reach, std::vector<Hit> &hits) {
    hits.resize(rays.size());
    if (rays.empty())
        return;
    auto &s = *data;
    s.rays.reserve(rays.size());
    s.hits.reserve(rays.size());
    check(cudaMemcpyAsync(s.rays.ptr, rays.data(), rays.size() * sizeof(Ray), cudaMemcpyHostToDevice,
                          s.stream.id));
    columns<<<(rays.size() + 63) / 64, 64, 0, s.stream.id>>>(s.view, s.rays.ptr, reach, s.hits.ptr,
                                                             static_cast<int>(rays.size()));
    check(cudaGetLastError());
    check(cudaMemcpyAsync(hits.data(), s.hits.ptr, hits.size() * sizeof(Hit), cudaMemcpyDeviceToHost,
                          s.stream.id));
    check(cudaStreamSynchronize(s.stream.id));
}
} // namespace nc::compiled::device
