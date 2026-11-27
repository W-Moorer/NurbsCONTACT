// Runtime workspace for one trajectory. Work is indexed by CAD topology, independent of tooth counts.
struct BodyView {
    View g;
    const Tile *tiles;
    const Node *sites;
    const int *target_ids;
    int ns, nt;
};
struct MatPose {
    V origin, axis;
    double angle;
};
struct Site {
    V p, n;
    double u, v, area;
};
struct Seed {
    int side, tile;
    double u, v;
};
struct Faces {
    int a, b;
    __device__ int operator[](int i) const { return i ? b : a; }
};
__device__ bool equal(Faces a, Faces b) { return a.a == b.a && a.b == b.b; }
struct Suggestion {
    Faces faces;
    int side, source, tile;
    double score;
};
struct Witness {
    Faces faces;
    V p, n;
    double depth;
    Hit exits[2];
    Seed seed;
};
struct Region {
    V p, n, arm;
    double depth;
};
struct Anchor {
    V a, b, na, nb, ca, cb;
};
struct QueryFault {
    int code{}, face{-1}, side{}, any{};
    Ray ray{};
    double end{};
};
struct SavedSeed {
    Faces ids;
    Seed seed;
};
struct TileWindow {
    Pose pose;
    int *tiles;
    int count, valid, rebuild;
    double margin;
};
// A depth-five binary connection proof has at most 31 internal material points.
struct Connection {
    Faces left, right;
    int count;
    V points[31];
};
struct ConnectionTrial {
    Connection path;
    QueryFault fault;
    int left, right;
    bool joined;
};
constexpr int connection_batch = 32;
struct PairScratch {
    int previous_count, anchors, ready;
    int audited_faces, audited_regions;
    Faces *audit_previous;
    SavedSeed *previous;
    Anchor *anchor;
    TileWindow windows[2];
    Connection *paths, *next_paths;
    int path_count;
};
struct Group {
    Faces ids;
    int begin, end, prior;
};
struct SampleDepth {
    Site s;
    double depth;
    int side;
};
__device__ unsigned long long timer_ns() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}
struct Workspace {
    unsigned int count;
    int groups, found, regions, selected, active_faces, capacity, limit;
    Suggestion *proposals;
    Group *group;
    Witness *witnesses, *confirmed;
    int *valid, *labels, *sides, *faces;
    SampleDepth *columns;
    unsigned *active_columns;
    Range *windows;
    ConnectionTrial *bridges;
    int column_capacity;
    Region *result;
    Region *reference;
    RegionReduction reduction;
    RegionAudit *audit;
    unsigned long long reduction_ns, reductions;
    int *status;
    unsigned long long queries, subdivisions, evaluations;
    unsigned long long broad_ns, sort_ns, search_ns, group_ns, integral_ns, window_builds, window_reuses;
    double maximum_residual;
    int unresolved_face;
    Ray failed_ray;
    double failed_end;
    int failed_side, failed_any;
    unsigned long long failed_ray_types[4];
};
__device__ V rotate(MatPose p, V x, double sign = 1) {
    if (p.angle == 0)
        return x;
    double a = sign * p.angle, c = cos(a), s = sin(a);
    return plus(plus(scale(x, c), scale(cross(p.axis, x), s)), scale(p.axis, (1 - c) * dot(p.axis, x)));
}
__device__ V world(MatPose p, V x) { return plus(p.origin, rotate(p, x)); }
__device__ V local(MatPose p, V x) { return rotate(p, sub(x, p.origin), -1); }
__device__ Pose between(MatPose a, MatPose b) {
    Pose m{};
    m.offset = scale(rotate(b, sub(a.origin, b.origin), -1), 1000);
    for (int k = 0; k < 3; ++k) {
        V e{};
        e.x[k] = 1;
        m.cols[k] = rotate(b, rotate(a, e), -1);
    }
    return m;
}
struct Query {
    BodyView b[2];
    Pose rel[2];
    double reach;
    Workspace *work;
    QueryFault *speculative{};
    __device__ void record(QueryFault e) const {
        if (speculative) {
            if (!speculative->code)
                *speculative = e;
            return;
        }
        if (atomicCAS(work->status, 0, e.code) == 0) {
            work->unresolved_face = e.face;
            work->failed_ray = e.ray;
            work->failed_end = e.end;
            work->failed_side = e.side;
            work->failed_any = e.any;
        }
    }
    __device__ void fail(int code, int id = -1) const { record({code, id}); }
    __device__ Hit ray(int side, V p, V d, double end, bool any = false) const {
        Hit h = first(b[side].g, {p, d}, end, any);
        if (h.face == -2)
            record({30 + static_cast<int>(h.v), static_cast<int>(h.u), side, any, {p, d}, end});
        return h;
    }
    __device__ bool inside(int side, int id, double u, double v) const {
        auto &g = b[side].g;
        int r = face_contains(g, g.charts[id], u, v);
        if (r < 0)
            fail(4, id);
        return r == 1;
    }
    __device__ bool node(int side, int id, double u, double v, Site &s) const {
        Jet j;
        if (!face_jet(b[side].g, id, u, v, j)) {
            fail(5, id);
            return false;
        }
        V n = cross(j.u, j.v);
        double a = norm(n);
        s = {j.p, a ? scale(n, b[side].g.charts[id].orientation / a) : V{}, u, v, a};
        return a > 0;
    }
    __device__ bool chord(V p, V n, Witness &w) const {
        Hit a = ray(0, p, n, reach);
        if (a.face < 0)
            return false;
        Hit h = ray(1, point(rel[0], p), direction(rel[0], scale(n, -1)), reach);
        if (h.face < 0)
            return false;
        w = {{a.face, h.face}, plus(p, scale(n, .5 * (a.t - h.t))), n, a.t + h.t, {a, h}, {}};
        return true;
    }
    __device__ bool sample(Faces ids, Seed seed, Site site, double &best, Witness &out) const {
        const int side = seed.side, id = ids[1 - side];
        const auto &g = b[1 - side].g;
        const auto &tile = b[1 - side].tiles[seed.tile];
        double u = tile.u, v = tile.v;
        V p = point(rel[side], site.p), d = direction(rel[side], scale(site.n, -1));
        for (int iter = 0; iter < 8; ++iter) {
            Jet j;
            if (!face_jet(g, id, u, v, j)) {
                fail(5, id);
                return false;
            }
            V delta = sub(j.p, p);
            double t = dot(delta, d);
            V f = sub(delta, scale(d, t));
            if (dot(f, f) < 1e-14) {
                if (fabs(t) > reach || !inside(1 - side, id, u, v))
                    return false;
                best = fmax(best, t);
                if (t <= 2e-7)
                    return false;
                V mid = sub(site.p, scale(site.n, .5 * t)), n = site.n;
                if (side) {
                    mid = point(rel[1], mid);
                    n = direction(rel[1], scale(n, -1));
                }
                if (chord(mid, n, out) && equal(out.faces, ids)) {
                    out.seed = seed;
                    return true;
                }
                return false;
            }
            V cu = cross(j.v, d), cv = cross(d, j.u);
            double det = dot(j.u, cu);
            if (det == 0 || fabs(det) < 1e-11 * norm(j.u) * norm(j.v))
                return false;
            u -= dot(f, cu) / det;
            v -= dot(f, cv) / det;
            auto r = g.charts[id].domain;
            if (u < r.u || u > r.U || v < r.v || v > r.Vv)
                return false;
        }
        return false;
    }
    // All lanes work on consecutive proposals. Only the earliest accepted proposal or
    // error in the original deterministic order affects the group.
    __device__ bool collect_attempt(int lane, Seed trial, double depth, bool hit, const Witness &candidate,
                                    QueryFault fault, Seed &seed, double &best, Witness &out) const {
        constexpr unsigned mask = 0xffffffffu;
        unsigned stop = __ballot_sync(mask, hit || fault.code);
        if (stop) {
            if (lane == __ffs(stop) - 1) {
                if (fault.code)
                    record(fault);
                else
                    out = candidate;
            }
            __syncwarp();
            return true;
        }
        double maximum = depth;
        for (int distance = 16; distance; distance /= 2)
            maximum = fmax(maximum, __shfl_down_sync(mask, maximum, distance));
        maximum = __shfl_sync(mask, maximum, 0);
        if (maximum > best) {
            unsigned winners = __ballot_sync(mask, depth == maximum);
            int first = __ffs(winners) - 1;
            seed.side = __shfl_sync(mask, trial.side, first);
            seed.tile = __shfl_sync(mask, trial.tile, first);
            seed.u = __shfl_sync(mask, trial.u, first);
            seed.v = __shfl_sync(mask, trial.v, first);
            best = maximum;
        }
        return false;
    }
    __device__ __noinline__ bool find(Group group, const PairScratch &cache, Witness &out) const {
        constexpr unsigned mask = 0xffffffffu;
        const int lane = threadIdx.x % 32;
        double best = -HUGE_VAL;
        Seed seed{};
        Witness candidate{};
        if (group.prior >= 0) {
            seed = cache.previous[group.prior].seed;
            int hit = 0;
            if (!lane) {
                Site s;
                hit = node(seed.side, group.ids[seed.side], seed.u, seed.v, s) &&
                      sample(group.ids, seed, s, best, out);
            }
            if (__shfl_sync(mask, hit, 0))
                return true;
            best = __shfl_sync(mask, best, 0);
        }
        Query q = *this;
        QueryFault fault;
        q.speculative = &fault;
        for (int start = group.begin; start < group.end; start += 32) {
            int k = start + lane;
            Seed trial{};
            double depth = -HUGE_VAL;
            bool hit = false;
            fault = {};
            if (k < group.end) {
                const auto &c = work->proposals[k];
                bool repeat = false;
                for (int j = group.begin; j < k; ++j) {
                    const auto &a = work->proposals[j];
                    if (a.side == c.side && a.source == c.source) {
                        repeat = true;
                        break;
                    }
                }
                if (!repeat) {
                    const auto &t = b[c.side].tiles[c.source];
                    trial = {c.side, c.tile, t.u, t.v};
                    Site s{t.point, t.normal, t.u, t.v, 0};
                    hit = q.sample(group.ids, trial, s, depth, candidate);
                }
            }
            if (collect_attempt(lane, trial, depth, hit, candidate, fault, seed, best, out))
                return true;
        }
        if (!isfinite(best))
            return false;
        const int side = seed.side, id = group.ids[side];
        const auto r = b[side].g.charts[id].domain;
        double du = (r.U - r.u) / 12, dv = (r.Vv - r.v) / 12;
        const int dx[8] = {1, -1, 0, 0, 1, -1, 1, -1}, dy[8] = {0, 0, 1, -1, 1, 1, -1, -1};
        for (int k = 0; k < 10; ++k) {
            if (k >= 5 && best < -sqrt(1e-7 * reach))
                break;
            Seed trial = seed;
            double depth = -HUGE_VAL;
            bool hit = false;
            fault = {};
            if (lane < 8) {
                trial.u += dx[lane] * du;
                trial.v += dy[lane] * dv;
                if (trial.u > r.u + 1e-9 && trial.u < r.U - 1e-9 && trial.v > r.v + 1e-9 &&
                    trial.v < r.Vv - 1e-9 && q.inside(side, id, trial.u, trial.v)) {
                    Site s;
                    hit = q.node(side, id, trial.u, trial.v, s) &&
                          q.sample(group.ids, trial, s, depth, candidate);
                }
            }
            if (collect_attempt(lane, trial, depth, hit, candidate, fault, seed, best, out))
                return true;
            du *= .5;
            dv *= .5;
        }
        return false;
    }
    __device__ bool crosses(int side, V start, V end, const Hit &hint) const {
        auto d = sub(end, start);
        double len = norm(d);
        if (len < 1e-7)
            return false;
        d = scale(d, 1 / len);
        if (fabs(dot(hint.normal, d)) < 1e-5)
            return false;
        double u = hint.u, v = hint.v;
        const auto &g = b[side].g;
        for (int i = 0; i < 5; ++i) {
            Jet j;
            if (!face_jet(g, hint.face, u, v, j))
                return false;
            V diff = sub(j.p, start);
            double t = dot(diff, d);
            diff = sub(diff, scale(d, t));
            V c = cross(j.v, d);
            double det = dot(j.u, c);
            if (fabs(det) <= 1e-7 * norm(j.u) * norm(j.v))
                return false;
            if (dot(diff, diff) < 1e-16)
                return t > 4e-7 && t < len - 4e-7 && inside(side, hint.face, u, v);
            u -= dot(diff, c) / det;
            v -= dot(diff, cross(d, j.u)) / det;
            const auto &r = g.charts[hint.face].domain;
            if (u < r.u || u > r.U || v < r.v || v > r.Vv)
                return false;
        }
        return false;
    }
    __device__ bool straight(V a, V b0) const {
        for (int side = 0; side < 2; ++side) {
            V x = side ? point(rel[0], a) : a, y = side ? point(rel[0], b0) : b0;
            V d = sub(y, x);
            double len = norm(d);
            if (len < 1e-8)
                continue;
            if (ray(side, x, scale(d, 1 / len), len, true).face >= 0)
                return false;
        }
        return true;
    }
    __device__ __noinline__ bool connected(const Witness &a, const Witness &b0, int depth,
                                           Connection &path) const {
        auto c = point(rel[0], a.p), d = point(rel[0], b0.p);
        if (!crosses(0, a.p, b0.p, a.exits[0]) && !crosses(0, b0.p, a.p, b0.exits[0]) &&
            !crosses(1, c, d, a.exits[1]) && !crosses(1, d, c, b0.exits[1]) && straight(a.p, b0.p))
            return true;
        if (!depth)
            return false;
        for (int side = 0; side < 2; ++side) {
            if (a.faces[side] != b0.faces[side])
                continue;
            int id = a.faces[side];
            double u = .5 * (a.exits[side].u + b0.exits[side].u),
                   v = .5 * (a.exits[side].v + b0.exits[side].v);
            if (!inside(side, id, u, v))
                continue;
            Site s;
            if (!node(side, id, u, v, s))
                continue;
            V n = plus(a.exits[side].normal, b0.exits[side].normal);
            double len = norm(n);
            if (len < 1e-6)
                continue;
            n = scale(n, 1 / len);
            Hit h = ray(1 - side, point(rel[side], s.p), direction(rel[side], scale(n, -1)), reach);
            if (h.face < 0)
                continue;
            V p = sub(s.p, scale(n, .5 * h.t));
            if (side) {
                p = point(rel[1], p);
                n = direction(rel[1], scale(n, -1));
            }
            Witness mid;
            const int start = path.count;
            if (chord(p, n, mid) && mid.faces[side] == id && connected(a, mid, depth - 1, path)) {
                if (path.count >= 31) {
                    fail(1);
                    return false;
                }
                path.points[path.count++] = mid.p;
                if (connected(mid, b0, depth - 1, path))
                    return true;
            }
            path.count = start;
        }
        return false;
    }
};
__device__ bool before(const Suggestion &a, const Suggestion &b) {
    if (a.faces.a != b.faces.a)
        return a.faces.a < b.faces.a;
    if (a.faces.b != b.faces.b)
        return a.faces.b < b.faces.b;
    if (a.score != b.score)
        return a.score > b.score;
    if (a.side != b.side)
        return a.side < b.side;
    if (a.tile != b.tile)
        return a.tile < b.tile;
    return a.source < b.source;
}

__device__ bool check_connection(Query &q, const PairScratch &cache, const Witness &a, const Witness &b,
                                 Connection &path) {
    path.left = a.faces;
    path.right = b.faces;
    path.count = 0;
    for (int k = 0; k < cache.path_count; ++k) {
        const auto &old = cache.paths[k];
        if (!equal(old.left, a.faces) || !equal(old.right, b.faces))
            continue;
        bool valid = true;
        V last = a.p;
        for (int t = 0; t < old.count; ++t) {
            if (!q.straight(last, old.points[t])) {
                valid = false;
                break;
            }
            last = old.points[t];
        }
        if (valid && q.straight(last, b.p)) {
            path = old;
            return true;
        }
        break;
    }
    return q.connected(a, b, 5, path);
}
template <class Grid> __device__ void sort_proposals(Grid grid, Suggestion *items, int count) {
    if (count < 2)
        return;
    const int lane = grid.thread_rank(), threads = grid.size();
    int size = 1;
    while (size < count)
        size *= 2;
    for (int i = count + lane; i < size; i += threads)
        items[i] = {{INT_MAX, INT_MAX}, 0, 0, 0, 0};
    grid.sync();
    for (int span = 2; span <= size; span *= 2)
        for (int stride = span / 2; stride; stride /= 2) {
            for (int k = lane; k < size / 2; k += threads) {
                int i = 2 * (k / stride) * stride + k % stride, j = i + stride;
                auto a = items[i], b = items[j];
                if ((i & span) ? before(a, b) : before(b, a)) {
                    items[i] = b;
                    items[j] = a;
                }
            }
            grid.sync();
        }
}
template <class Grid> __device__ void detect(Grid grid, Query &q, PairScratch &cache) {
    auto &w = *q.work;
    int lane = grid.thread_rank(), threads = grid.size();
    unsigned long long stamp = lane ? 0 : timer_ns();
    if (!lane) {
        w.count = 0;
        w.groups = 0;
        w.found = 0;
        w.regions = 0;
        ++w.queries;
    }
    grid.sync();
    for (int side = 0; side < 2; ++side) {
        auto a = q.b[side], b = q.b[1 - side];
        Pose pose = q.rel[1 - side];
        auto &window = cache.windows[side];
        if (!lane) {
            const auto bounds = b.g.faces[0].box;
            V extreme{};
            for (int k = 0; k < 3; ++k)
                extreme.x[k] = fmax(fabs(bounds.lo.x[k]), fabs(bounds.hi.x[k]));
            double radius = norm(extreme), rotation = 0;
            for (int k = 0; k < 3; ++k) {
                auto d = sub(pose.cols[k], window.pose.cols[k]);
                rotation += dot(d, d);
            }
            double motion = norm(sub(pose.offset, window.pose.offset)) + radius * sqrt(rotation);
            window.rebuild = !window.valid || motion > window.margin;
            if (window.rebuild) {
                window.count = 0;
                window.pose = pose;
                window.valid = 1;
                window.margin = fmax(4 * q.reach, .02 * radius);
                ++w.window_builds;
            } else
                ++w.window_reuses;
        }
        grid.sync();
        if (window.rebuild) {
            for (int t = lane; t < b.nt; t += threads) {
                const int id = b.target_ids[t];
                const auto box = placed(pose, b.tiles[id].box);
                for (int i = 0; i < a.ns;) {
                    const auto &node = a.sites[i];
                    if (!overlaps(box, node.box, q.reach + window.margin + 2e-7)) {
                        i = node.next;
                        continue;
                    }
                    if (node.leaf >= 0) {
                        window.tiles[atomicAdd(&window.count, 1)] = id;
                        break;
                    }
                    ++i;
                }
            }
        }
        grid.sync();
        for (int t = lane; t < window.count; t += threads) {
            int tile = window.tiles[t];
            const auto &ref = b.tiles[tile];
            Bounds box = placed(pose, ref.box);
            V normal = direction(pose, ref.normal), center = point(pose, ref.point);
            for (int i = 0; i < a.ns;) {
                auto n = a.sites[i];
                if (!overlaps(box, n.box, q.reach + 1e-7)) {
                    i = n.next;
                    continue;
                }
                ++i;
                if (n.leaf < 0)
                    continue;
                const auto &s = a.tiles[n.leaf];
                double cosine = -dot(s.normal, normal);
                if (cosine < .5)
                    continue;
                double d = -dot(sub(s.point, center), normal) / cosine;
                if (fabs(d) > q.reach)
                    continue;
                unsigned slot = atomicAdd(&w.count, 1u);
                if (slot >= w.capacity) {
                    q.fail(1);
                    continue;
                }
                w.proposals[slot] = {side ? Faces{ref.face, s.face} : Faces{s.face, ref.face}, side, n.leaf,
                                     tile, d};
            }
        }
        grid.sync();
    }
    grid.sync();
    if (*w.status)
        return;
    if (!lane) {
        auto now = timer_ns();
        w.broad_ns += now - stamp;
        stamp = now;
    }
    sort_proposals(grid, w.proposals, w.count);
    if (!lane) {
        int k = 0, j = 0;
        while (k < w.count || j < cache.previous_count) {
            Faces p = k < w.count ? w.proposals[k].faces : Faces{INT_MAX, INT_MAX};
            Faces h = j < cache.previous_count ? cache.previous[j].ids : Faces{INT_MAX, INT_MAX};
            Faces ids = (h.a < p.a || (h.a == p.a && h.b < p.b)) ? h : p;
            int begin = k;
            while (k < w.count && equal(w.proposals[k].faces, ids))
                ++k;
            if (w.groups >= w.limit) {
                q.fail(1);
                break;
            }
            w.group[w.groups++] = {ids, begin, k, equal(h, ids) ? j++ : -1};
        }
        auto now = timer_ns();
        w.sort_ns += now - stamp;
        stamp = now;
    }
    grid.sync();
    if (*w.status)
        return;
    for (int i = lane / 32; i < w.groups; i += threads / 32) {
        bool found = q.find(w.group[i], cache, w.confirmed[i]);
        if (!(lane % 32))
            w.valid[i] = found;
    }
    grid.sync();
    if (*w.status)
        return;
    if (!lane) {
        auto now = timer_ns();
        w.search_ns += now - stamp;
        stamp = now;
        for (int i = 0; i < w.groups; ++i)
            if (w.valid[i]) {
                auto &v = w.confirmed[i];
                int k = w.found++;
                w.witnesses[k] = v;
                w.labels[k] = k;
                cache.previous[k] = {v.faces, v.seed};
            }
        cache.previous_count = w.found;
    }
    grid.sync();
    const int pairs = w.found * (w.found - 1) / 2;
    int saved = 0;
    for (int first = 0; first < pairs; first += connection_batch) {
        int count = min(connection_batch, pairs - first);
        // Use separate warps for independent, branch-heavy visibility proofs.
        for (int slot = lane / 32; slot < count; slot += threads / 32) {
            if (lane % 32)
                continue;
            int rank = first + slot, i = 0;
            while (rank >= w.found - 1 - i) {
                rank -= w.found - 1 - i;
                ++i;
            }
            int j = i + rank + 1;
            auto &trial = w.bridges[slot];
            trial.left = i;
            trial.right = j;
            trial.fault = {};
            trial.joined = false;
            if (w.labels[i] != w.labels[j]) {
                Query local = q;
                local.speculative = &trial.fault;
                trial.joined = check_connection(local, cache, w.witnesses[i], w.witnesses[j], trial.path);
            }
        }
        grid.sync();
        if (!lane) {
            for (int k = 0; k < count; ++k) {
                const auto &t = w.bridges[k];
                int left = w.labels[t.left], right = w.labels[t.right];
                if (left == right)
                    continue;
                if (t.fault.code) {
                    q.record(t.fault);
                    break;
                }
                if (!t.joined)
                    continue;
                for (int j = 0; j < w.found; ++j)
                    if (w.labels[j] == right)
                        w.labels[j] = left;
                if (t.path.count)
                    cache.next_paths[saved++] = t.path;
            }
        }
        grid.sync();
        if (*w.status)
            return;
    }
    if (!lane) {
        auto *scratch = cache.paths;
        cache.paths = cache.next_paths;
        cache.next_paths = scratch;
        cache.path_count = saved;
        w.group_ns += timer_ns() - stamp;
    }
    grid.sync();
}
__device__ bool accepts(const Workspace &w, int label, int side, int id, int target) {
    for (int i = 0; i < w.found; ++i)
        if (w.labels[i] == label) {
            auto f = w.witnesses[i].faces;
            if (f[side] == id && f[1 - side] == target)
                return true;
        }
    return false;
}
__device__ Range footprint(Query &q, int label, int side, int face) {
    const auto domain = q.b[side].g.charts[face].domain;
    Range out{1e300, -1e300, 1e300, -1e300};
    const auto &w = *q.work;
    for (int i = 0; i < w.found; ++i) {
        const auto &hit = w.witnesses[i];
        if (w.labels[i] != label || hit.faces[side] != face)
            continue;
        const auto &exit = hit.exits[side];
        for (int axis = 0; axis < 2; ++axis)
            for (int sign = -1; sign <= 1; sign += 2) {
                double low = axis ? domain.v : domain.u, high = axis ? domain.Vv : domain.U;
                double origin = axis ? exit.v : exit.u, h = (high - low) / 192., x = origin;
                for (int iteration = 0; iteration < 10; ++iteration) {
                    x = fmin(high, fmax(low, origin + sign * h));
                    double u = axis ? exit.u : x, v = axis ? x : exit.v;
                    Site s;
                    if (!q.inside(side, face, u, v) || !q.node(side, face, u, v, s))
                        break;
                    auto other = q.ray(1 - side, point(q.rel[side], s.p),
                                       direction(q.rel[side], scale(s.n, -1)), q.reach);
                    if (other.face < 0 || !accepts(w, label, side, face, other.face) || x == low || x == high)
                        break;
                    h *= 2;
                }
                if (axis) {
                    out.v = fmin(out.v, x);
                    out.Vv = fmax(out.Vv, x);
                } else {
                    out.u = fmin(out.u, x);
                    out.U = fmax(out.U, x);
                }
            }
    }
    return out;
}
struct IntegralSum {
    V area{}, moment{}, centroid{};
    double volume{};
};
__device__ IntegralSum column_term(const Query &q, const SampleDepth &s) {
    V n = s.side ? direction(q.rel[1], s.s.n) : s.s.n;
    V p = s.side ? point(q.rel[1], s.s.p) : s.s.p;
    V a = scale(n, (s.side ? -1 : 1) * s.s.area);
    double vol = s.depth * s.s.area;
    return {a, cross(p, a), scale(sub(p, scale(n, .5 * s.depth)), vol), vol};
}
__device__ __noinline__ IntegralSum sum_columns(const Query &q, int count, int first_word, int stride) {
    IntegralSum sum{};
    const auto &w = *q.work;
    for (int word = first_word; word < (count + 31) / 32; word += stride) {
        unsigned mask = w.active_columns[word];
        while (mask) {
            const auto &s = w.columns[32 * word + __ffs(mask) - 1];
            mask &= mask - 1;
            const auto term = column_term(q, s);
            sum.area = plus(sum.area, term.area);
            sum.moment = plus(sum.moment, term.moment);
            sum.centroid = plus(sum.centroid, term.centroid);
            sum.volume += term.volume;
        }
    }
    return sum;
}
__device__ __noinline__ IntegralSum reduce_columns(const Query &q, int count, int lane) {
    IntegralSum v = sum_columns(q, count, lane, 32);
    for (int distance = 16; distance; distance /= 2) {
        for (int k = 0; k < 3; ++k) {
            v.area.x[k] += __shfl_down_sync(0xffffffffu, v.area.x[k], distance);
            v.moment.x[k] += __shfl_down_sync(0xffffffffu, v.moment.x[k], distance);
            v.centroid.x[k] += __shfl_down_sync(0xffffffffu, v.centroid.x[k], distance);
        }
        v.volume += __shfl_down_sync(0xffffffffu, v.volume, distance);
    }
    return v;
}
__device__ bool make_region(Query &q, const IntegralSum &sum, Region &region) {
    const double a = norm(sum.area);
    if (!(sum.volume > 0 && a > 1e-18)) return false;
    V p = scale(sum.centroid, 1 / sum.volume), n = scale(sum.area, 1 / a);
    Witness hit;
    if (!q.chord(p, n, hit)) return false;
    region = {p, n, sub(scale(sum.moment, 1 / a), cross(p, n)), hit.depth};
    return true;
}
// Compact active column IDs in increasing order, form up to 32 independent
// contributions together, and assign one original FP64 sum to each component.
// Keeping each scalar addition in order preserves threshold decisions in HHT.
__device__ __noinline__ IntegralSum ordered_columns(const Query &q, int count, int lane) {
    __shared__ int ids[32 * 32];
    __shared__ double terms[32][10];
    constexpr unsigned full = 0xffffffffu;
    const auto &w = *q.work;
    const int words = (count + 31) / 32;
    double total = 0.;
    for (int base = 0; base < words; base += 32) {
        int word = base + lane;
        unsigned mask = word < words ? w.active_columns[word] : 0;
        if (!__any_sync(full, mask != 0)) continue;
        const int active = __popc(mask);
        int end = active;
        for (int shift = 1; shift < 32; shift *= 2) {
            int preceding = __shfl_up_sync(full, end, shift);
            if (lane >= shift) end += preceding;
        }
        const int size = __shfl_sync(full, end, 31);
        int out = end - active;
        while (mask) {
            ids[out++] = 32 * word + __ffs(mask) - 1;
            mask &= mask - 1;
        }
        __syncwarp();
        for (int start = 0; start < size; start += 32) {
            const int n = min(32, size - start);
            if (lane < n) {
                const auto term = column_term(q, w.columns[ids[start + lane]]);
                for (int k = 0; k < 3; ++k) {
                    terms[lane][k] = term.area.x[k];
                    terms[lane][3+k] = term.moment.x[k];
                    terms[lane][6+k] = term.centroid.x[k];
                }
                terms[lane][9] = term.volume;
            }
            __syncwarp();
            if (lane < 10)
                for (int k = 0; k < n; ++k) total += terms[k][lane];
            __syncwarp();
        }
    }
    IntegralSum sum{};
    for (int k = 0; k < 3; ++k) {
        sum.area.x[k] = __shfl_sync(full, total, k);
        sum.moment.x[k] = __shfl_sync(full, total, k+3);
        sum.centroid.x[k] = __shfl_sync(full, total, k+6);
    }
    sum.volume = __shfl_sync(full, total, 9);
    return sum;
}
template <class Grid> __device__ void integrate_regions(Grid grid, Query &q) {
    auto &w = *q.work;
    int lane = grid.thread_rank(), threads = grid.size();
    for (int label = 0; label < w.found; ++label) {
        if (w.labels[label] != label)
            continue;
        grid.sync();
        if (!lane) {
            w.active_faces = 0;
            w.selected = 0;
            for (int side = 0; side < 2; ++side) {
                const int first = w.active_faces;
                for (int i = 0; i < w.found; ++i)
                    if (w.labels[i] == label) {
                        const int id = w.witnesses[i].faces[side];
                        int at = first;
                        while (at < w.active_faces && w.faces[at] < id)
                            ++at;
                        if (at < w.active_faces && w.faces[at] == id)
                            continue;
                        for (int k = w.active_faces; k > at; --k)
                            w.faces[k] = w.faces[k - 1];
                        w.faces[at] = id;
                        w.sides[w.active_faces++] = side;
                    }
            }
        }
        grid.sync();
        for (int pass = 0; pass < 5; ++pass) {
            bool local_rule = pass == 4;
            int order = local_rule ? 24 : 12 << pass;
            if (local_rule) {
                for (int k = lane; k < w.active_faces; k += threads)
                    w.windows[k] = footprint(q, label, w.sides[k], w.faces[k]);
                grid.sync();
            }
            int count = w.active_faces * order * order;
            if (count > w.column_capacity) {
                if (!lane)
                    q.fail(1);
                grid.sync();
                return;
            }
            for (int i = lane; i < ((count + 31) / 32) * 32; i += threads) {
                SampleDepth s{};
                s.depth = -1;
                if (i < count) {
                    int face_index = i / (order * order), side = w.sides[face_index],
                        id = w.faces[face_index];
                    int ij = i % (order * order);
                    auto r = local_rule ? w.windows[face_index] : q.b[side].g.charts[id].domain;
                    double du = (r.U - r.u) / order, dv = (r.Vv - r.v) / order;
                    double u = r.u + (ij / order + .5) * du;
                    double v = r.v + (ij % order + .5) * dv;
                    const auto &g = q.b[side].g;
                    const int base = g.charts[id].rule_start;
                    bool valid = false;
                    if (!local_rule && order <= 24 && base >= 0) {
                        auto cached = g.rules[base + (order == 12 ? 0 : 144) + ij];
                        s.s = {cached.point, cached.normal, u, v, cached.area};
                        valid = cached.area > 0;
                    } else if (q.inside(side, id, u, v) && q.node(side, id, u, v, s.s)) {
                        s.s.area = local_rule ? s.s.area * ((r.U - r.u) * (r.Vv - r.v) / (order * order))
                                              : s.s.area * du * dv;
                        valid = true;
                    }
                    if (valid) {
                        V p = point(q.rel[side], s.s.p), d = direction(q.rel[side], scale(s.s.n, -1));
                        bool possible = false;
                        for (int j = 0; j < w.found && !possible; ++j)
                            if (w.labels[j] == label && w.witnesses[j].faces[side] == id)
                                possible =
                                    intersects(q.b[1 - side].g.charts[w.witnesses[j].faces[1 - side]].box,
                                               {p, d}, q.reach, 1e-7);
                        if (possible) {
                            auto h = q.ray(1 - side, p, d, q.reach);
                            if (h.face >= 0 && accepts(w, label, side, id, h.face)) {
                                s.depth = h.t;
                                s.side = side;
                            }
                        }
                    }
                }
                unsigned mask = __ballot_sync(0xffffffffu, s.depth >= 0);
                if (!(lane % 32))
                    w.active_columns[i / 32] = mask;
                if (s.depth >= 0)
                    w.columns[i] = s;
            }
            grid.sync();
            if (lane < 32) {
                auto start = timer_ns();
                IntegralSum sum{};
                if (w.reduction == RegionReduction::ordered || w.reduction == RegionReduction::audit)
                    sum = ordered_columns(q, count, lane);
                else if (w.reduction != RegionReduction::serial)
                    sum = reduce_columns(q, count, lane);
                else if (!lane)
                    sum = sum_columns(q, count, 0, 1);
                if (!lane) {
                    w.reduction_ns += timer_ns() - start;
                    ++w.reductions;
                    Region actual{};
                    const bool valid = make_region(q, sum, actual);
                    if (w.reference) {
                        Region ref{};
                        const auto original = sum_columns(q, count, 0, 1);
                        if (w.reduction == RegionReduction::audit) {
                            for (int k = 0; k < 3; ++k)
                                if (sum.area.x[k] != original.area.x[k] || sum.moment.x[k] != original.moment.x[k]
                                    || sum.centroid.x[k] != original.centroid.x[k]) q.fail(10);
                            if (sum.volume != original.volume) q.fail(10);
                        }
                        bool expected = make_region(q, original, ref);
                        if (valid != expected) q.fail(10);
                        if (expected) w.reference[w.regions] = ref;
                    }
                    if (valid) {
                        w.result[w.regions++] = actual;
                        w.selected = 1;
                    }
                }
            }
            grid.sync();
            bool done = w.selected || *w.status;
            grid.sync();
            if (done)
                break;
        }
        if (!w.selected && !lane)
            q.fail(6);
        grid.sync();
        if (*w.status)
            return;
    }
}
