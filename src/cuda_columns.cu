#include "cuda_columns.hpp"
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

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
struct StepGraph {
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    ~StepGraph() {
        if (executable)
            cudaGraphExecDestroy(executable);
        if (graph)
            cudaGraphDestroy(graph);
    }
};
struct View {
    const Node *faces, *spans;
    const FaceData *charts;
    const Span *patches;
    const SpanFrame *frames;
    const Cell *cells;
    int n;
    const Node *trim_tree;
    const TrimPiece *trims;
    const Pole *trim_poles;
    const int *grid;
    const double *knots;
    const WeightedSite *rules;
    const Bounds *trim_intervals;
    const RayGrid *ray_grid;
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
__device__ int isolate(const Span &s, const Cell &c, Ray ray, double tol, double end, double &t, V &normal,
                       double &u, double &v) {
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
            u = r.u + x * du;
            v = r.v + y * dv;
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
        if (x < -1e-10 || x > 1 + 1e-10 || y < -1e-10 || y > 1 + 1e-10)
            return -2;
    }
    return -2;
}
#include "cuda_cad_roots.cuh"
__global__ void columns(View view, const Ray *rays, double reach, Hit *hits, int n) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        hits[i] = first(view, rays[i], reach);
}
__global__ void chart_samples(View view, const ChartPoint *uv, ChartSample *result, int count) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= count)
        return;
    const auto x = uv[i];
    const auto &face = view.charts[x.face];
    Jet j{};
    ChartSample sample{};
    sample.material = face_contains(view, face, x.u, x.v);
    sample.defined = face_jet(view, x.face, x.u, x.v, j);
    if (sample.defined) {
        sample.point = j.p;
        V n = cross(j.u, j.v);
        double length = norm(n);
        if (length > 0)
            sample.normal = scale(n, face.orientation / length);
    }
    result[i] = sample;
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
#include "cuda_resident_geometry.cuh"
#include "cuda_resident_step.cuh"
} // namespace
struct Columns::Storage {
    Buffer<RayGrid> grid_descriptor;
    Buffer<int> short_offsets, short_ids;
    Buffer<RayPrimitive> short_primitives;
    bool resident_supported{};
    Buffer<WeightedSite> rule_sites;
    Buffer<Node> faces, spans;
    Buffer<FaceData> charts;
    Buffer<Span> patches;
    Buffer<SpanFrame> frames;
    Buffer<Cell> cells;
    Buffer<Node> trim_tree;
    Buffer<TrimPiece> trims;
    Buffer<Pole> trim_poles;
    Buffer<Bounds> trim_intervals;
    Buffer<double> knots;
    Buffer<int> grid;
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
    s->resident_supported =
        std::all_of(t.patches.begin(), t.patches.end(), [](const auto &p) { return p.cell >= 0; }) &&
        std::all_of(t.charts.begin(), t.charts.end(),
                    [](const auto &f) { return f.first >= 0 || (f.kind >= 1 && f.kind <= 3); });
    s->faces.upload(t.faces);
    s->rule_sites.upload(t.rule_sites);
    s->spans.upload(t.spans);
    s->charts.upload(t.charts);
    s->patches.upload(t.patches);
    s->frames.upload(t.frames);
    s->cells.upload(t.cells);
    s->trim_tree.upload(t.trim_tree);
    s->trims.upload(t.trims);
    s->trim_poles.upload(t.trim_poles);
    s->trim_intervals.upload(t.trim_intervals);
    if (t.ray_grid.reach > 0) {
        s->short_offsets.upload(t.ray_offsets);
        s->short_ids.upload(t.ray_ids);
        s->short_primitives.upload(t.ray_primitives);
        auto grid = t.ray_grid;
        grid.offsets = s->short_offsets.ptr;
        grid.ids = s->short_ids.ptr;
        grid.primitives = s->short_primitives.ptr;
        s->grid_descriptor.upload({grid});
    }
    s->grid.upload(t.grid);
    s->knots.upload(t.knots);
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
               static_cast<int>(t.faces.size()),
               s->trim_tree.ptr,
               s->trims.ptr,
               s->trim_poles.ptr,
               s->grid.ptr,
               s->knots.ptr,
               s->rule_sites.ptr,
               s->trim_intervals.ptr,
               s->grid_descriptor.ptr};
    data = s.release();
}
Columns::~Columns() { delete data; }
std::vector<ChartSample> Columns::sample_charts(const std::vector<ChartPoint> &queries) const {
    if (queries.empty())
        return {};
    if (queries.size() > INT_MAX)
        throw std::length_error("too many chart probes");
    for (const auto &p : queries)
        if (p.face < 0 || static_cast<size_t>(p.face) >= data->charts.capacity)
            throw std::out_of_range("invalid CAD face");
    Buffer<ChartPoint> uv;
    uv.upload(queries);
    Buffer<ChartSample> values;
    values.reserve(queries.size());
    chart_samples<<<(queries.size() + 127) / 128, 128>>>(data->view, uv.ptr, values.ptr,
                                                         static_cast<int>(queries.size()));
    check(cudaGetLastError());
    std::vector<ChartSample> out(queries.size());
    check(cudaMemcpy(out.data(), values.ptr, out.size() * sizeof(out[0]), cudaMemcpyDeviceToHost));
    return out;
}
ResidentResult advance_resident(const ResidentScene &scene, double step, int steps) {
    if (scene.coordinates.empty() || scene.coordinates.size() > coordinate_limit ||
        scene.geometry.size() != scene.bodies.size() || steps < 1 || steps == INT_MAX || !(step > 0) ||
        !std::isfinite(step))
        throw std::invalid_argument("unsupported resident scene dimensions");
    for (const auto &c : scene.coordinates) {
        if (!(c.mass > 0) || c.first < 0 || c.end < c.first || c.end > scene.loads.size() ||
            c.end - c.first == 1)
            throw std::invalid_argument("invalid resident coordinate or load table");
        for (int j = c.first + 1; j < c.end; ++j)
            if (!(scene.loads[j].t > scene.loads[j - 1].t))
                throw std::invalid_argument("resident load samples require increasing times");
    }
    for (const auto &b : scene.bodies)
        if (b.hinge < -1 || b.translation < -1 || b.hinge >= int(scene.coordinates.size()) ||
            b.translation >= int(scene.coordinates.size()))
            throw std::invalid_argument("resident joint coordinate is out of range");
    for (const auto &p : scene.pairs)
        if (p.a < 0 || p.b < 0 || p.a == p.b || p.a >= scene.bodies.size() || p.b >= scene.bodies.size() ||
            !(p.k > 0) || !(p.exponent >= 1) || !(p.rate >= 0) || !(p.max_depth > 0))
            throw std::invalid_argument("invalid resident contact law or body pair");
    int dev = 0;
    check(cudaGetDevice(&dev));
    cudaDeviceProp props{};
    check(cudaGetDeviceProperties(&props, dev));
    if (!props.cooperativeLaunch)
        throw std::runtime_error("resident CAD requires CUDA cooperative launch");
    size_t stack = 0;
    check(cudaDeviceGetLimit(&stack, cudaLimitStackSize));
    if (stack < 16384)
        check(cudaDeviceSetLimit(cudaLimitStackSize, 16384));
    Buffer<BodyView> bodies;
    std::vector<BodyView> views;
    for (auto &part : scene.geometry) {
        if (!part)
            throw std::invalid_argument("missing resident CAD solid");
        const auto &s = part->storage();
        if (!s.resident_supported || !s.view.n)
            throw std::invalid_argument(
                "resident CAD requires supported analytic or polynomial bicubic charts");
        views.push_back({s.view, s.tiles.ptr, s.sites.ptr, s.ids.ptr, static_cast<int>(s.sites.capacity),
                         static_cast<int>(s.ids.capacity)});
    }
    bodies.upload(views);
    Buffer<AxisCoordinate> coords;
    coords.upload(scene.coordinates);
    Buffer<JointBody> joints;
    joints.upload(scene.bodies);
    Buffer<ContactLaw> laws;
    laws.upload(scene.pairs);
    Buffer<LoadPoint> loads;
    loads.upload(scene.loads);
    const int n = static_cast<int>(scene.coordinates.size()), np = static_cast<int>(scene.pairs.size());
    constexpr int limit = 512, capacity = 65536, column_capacity = 96 * 96 * 32;
    Buffer<Suggestion> proposals;
    proposals.reserve(capacity);
    Buffer<Group> groups;
    groups.reserve(limit);
    Buffer<Witness> witnesses, confirmed;
    witnesses.reserve(limit);
    confirmed.reserve(limit);
    Buffer<int> flags;
    flags.reserve(6 * limit);
    Buffer<SampleDepth> columns;
    columns.reserve(column_capacity);
    Buffer<unsigned> masks;
    masks.reserve((column_capacity + 31) / 32);
    Buffer<Region> regions;
    regions.reserve(limit);
    Buffer<Anchor> anchors;
    anchors.reserve(np * limit);
    Buffer<SavedSeed> previous;
    previous.reserve(np * limit);
    std::vector<PairScratch> slots(np);
    Buffer<Connection> paths;
    paths.reserve(2 * static_cast<size_t>(np) * limit);
    size_t tile_count = 0;
    for (auto &p : scene.pairs)
        tile_count += views.at(p.a).nt + views.at(p.b).nt;
    Buffer<int> window_tiles;
    window_tiles.reserve(tile_count);
    size_t offset = 0;
    for (int i = 0; i < np; ++i) {
        slots[i].previous = previous.ptr + i * limit;
        slots[i].anchor = anchors.ptr + i * limit;
        slots[i].paths = paths.ptr + 2 * static_cast<size_t>(i) * limit;
        slots[i].next_paths = slots[i].paths + limit;
        for (int side = 0; side < 2; ++side) {
            slots[i].windows[side].tiles = window_tiles.ptr + offset;
            offset += views[side ? scene.pairs[i].a : scene.pairs[i].b].nt;
        }
    }
    Buffer<PairScratch> cache;
    cache.upload(slots);
    Buffer<StepControl> control;
    control.reserve(1);
    check(cudaMemset(control.ptr, 0, sizeof(StepControl)));
    Workspace w{};
    w.capacity = capacity;
    w.limit = limit;
    w.proposals = proposals.ptr;
    w.group = groups.ptr;
    w.witnesses = witnesses.ptr;
    w.confirmed = confirmed.ptr;
    w.valid = flags.ptr;
    w.labels = flags.ptr + limit;
    w.sides = flags.ptr + 2 * limit;
    w.faces = flags.ptr + 4 * limit;
    w.columns = columns.ptr;
    w.column_capacity = column_capacity;
    w.active_columns = masks.ptr;
    w.result = regions.ptr;
    w.status = &control.ptr->status;
    w.unresolved_face = -1;
    Buffer<Range> windows;
    windows.reserve(2 * limit);
    w.windows = windows.ptr;
    Buffer<ConnectionTrial> bridges;
    bridges.reserve(connection_batch);
    w.bridges = bridges.ptr;
    Buffer<Workspace> work;
    work.upload({w});
    Buffer<double> observed, frames;
    observed.reserve(3 * np);
    frames.reserve(static_cast<size_t>(steps + 1) * (1 + 4 * n + 3 * np));
    Buffer<unsigned long long> finished;
    finished.reserve(steps + 1);
    DeviceScene s{coords.ptr,
                  joints.ptr,
                  laws.ptr,
                  loads.ptr,
                  bodies.ptr,
                  n,
                  static_cast<int>(views.size()),
                  np,
                  cache.ptr,
                  work.ptr,
                  control.ptr,
                  observed.ptr,
                  frames.ptr,
                  finished.ptr};
    int block = 64, occupancy = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, resident_steps, block, 0));
    int grid = props.multiProcessorCount * std::min(occupancy, 1);
    if (grid < 1)
        throw std::runtime_error("resident CAD launch does not fit device");
    Stream stream;
    StepGraph schedule;
    check(cudaStreamBeginCapture(stream.id, cudaStreamCaptureModeThreadLocal));
    for (int begin = 0; begin < steps;) {
        int end = std::min(steps, begin + (np ? 1 : 64));
        if (np) {
            void *args[] = {&s, &begin, &end, &step};
            check(cudaLaunchCooperativeKernel(reinterpret_cast<const void *>(resident_steps), grid, block,
                                              args, 0, stream.id));
        } else {
            unconstrained_steps<<<1, 1, 0, stream.id>>>(s, begin, end, step);
            check(cudaGetLastError());
        }
        begin = end;
    }
    check(cudaStreamEndCapture(stream.id, &schedule.graph));
    check(cudaGraphInstantiate(&schedule.executable, schedule.graph, 0));
    check(cudaGraphUpload(schedule.executable, stream.id));
    check(cudaStreamSynchronize(stream.id));
    auto start = std::chrono::steady_clock::now();
    check(cudaGraphLaunch(schedule.executable, stream.id));
    check(cudaStreamSynchronize(stream.id));
    StepControl final{};
    check(cudaMemcpy(&final, control.ptr, sizeof(final), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&w, work.ptr, sizeof(w), cudaMemcpyDeviceToHost));
    if (final.status) {
        std::ostringstream ray;
        ray << std::setprecision(17) << " side=" << w.failed_side << " crossing=" << w.failed_any << " ray=";
        for (double x : w.failed_ray.p.x)
            ray << x << ' ';
        for (double x : w.failed_ray.d.x)
            ray << x << ' ';
        ray << w.failed_end;
        throw std::runtime_error(
            "device CAD unresolved: status=" + std::to_string(final.status) +
            " step=" + std::to_string(final.last_step + 1) + " pair=" + std::to_string(final.current_pair) +
            " face=" + std::to_string(w.unresolved_face) + " time_s=" + std::to_string(final.state.time) +
            " geometry_queries=" + std::to_string(w.queries) + ray.str());
    }
    ResidentResult result;
    result.queries = w.queries;
    result.forces = w.evaluations;
    result.splits = w.subdivisions;
    result.maximum_residual = w.maximum_residual;
    result.frames.resize(frames.capacity);
    check(cudaMemcpy(result.frames.data(), frames.ptr, frames.capacity * sizeof(double),
                     cudaMemcpyDeviceToHost));
    std::vector<unsigned long long> clocks(steps + 1);
    check(cudaMemcpy(clocks.data(), finished.ptr, clocks.size() * sizeof(clocks[0]), cudaMemcpyDeviceToHost));
    result.online_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const unsigned long long phases[]{w.broad_ns, w.sort_ns, w.search_ns, w.group_ns, w.integral_ns};
    for (int i = 0; i < 5; ++i)
        result.phase_s[i] = phases[i] * 1e-9;
    std::vector<double> durations(steps);
    for (int i = 0; i < steps; ++i) {
        durations[i] = (clocks[i + 1] - clocks[i]) * 1e-9;
        if (durations[i] > step)
            ++result.missed_deadlines;
    }
    std::sort(durations.begin(), durations.end());
    result.p50_s = durations[(steps - 1) / 2];
    result.p99_s = durations[static_cast<size_t>(std::ceil(.99 * steps)) - 1];
    result.maximum_step_s = durations.back();
    return result;
}
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
double Columns::time_ray_batches(const std::vector<Ray>& rays,double reach,int batch,int repeats,
                                 std::vector<Hit>& hits) {
    if(rays.empty() || rays.size()>INT_MAX || batch<=0 || repeats<=0 || !(reach>0))
        throw std::invalid_argument("invalid ray timing batch");
    auto& s=*data;
    s.rays.reserve(rays.size());
    check(cudaMemcpyAsync(s.rays.ptr,rays.data(),rays.size()*sizeof(Ray),cudaMemcpyHostToDevice,s.stream.id));
    check(cudaStreamSynchronize(s.stream.id));
    s.hits.reserve(rays.size());
    StepGraph schedule;
    const int n=static_cast<int>(rays.size());
    auto stream=s.stream.id;
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    for(size_t start=0;start<rays.size();start+=batch) {
        const int count=std::min<int>(batch,n-static_cast<int>(start));
        columns<<<(count+63)/64,64,0,stream>>>(s.view,s.rays.ptr+start,reach,s.hits.ptr+start,count);
    }
    check(cudaStreamEndCapture(stream,&schedule.graph));
    check(cudaGraphInstantiate(&schedule.executable,schedule.graph,0));
    const auto warm=std::chrono::steady_clock::now()+std::chrono::milliseconds(250);
    do {
        check(cudaGraphLaunch(schedule.executable,stream));
        check(cudaStreamSynchronize(stream));
    } while(std::chrono::steady_clock::now()<warm);
    struct Event {
        cudaEvent_t id{};
        Event(){check(cudaEventCreate(&id));}
        ~Event(){cudaEventDestroy(id);}
    } first,last;
    check(cudaEventRecord(first.id,stream));
    for(int i=0;i<repeats;++i) check(cudaGraphLaunch(schedule.executable,stream));
    check(cudaEventRecord(last.id,stream));
    check(cudaEventSynchronize(last.id));
    float ms;
    check(cudaEventElapsedTime(&ms,first.id,last.id));
    hits.resize(rays.size());
    check(cudaMemcpy(hits.data(),s.hits.ptr,rays.size()*sizeof(Hit),cudaMemcpyDeviceToHost));
    const size_t batches=(rays.size()+batch-1)/batch;
    return 1000.*ms/(repeats*double(batches));
}
} // namespace nc::compiled::device
