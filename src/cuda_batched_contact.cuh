// Device-controlled phase graphs. Each pair owns its mutable workspace; heavy
// query kernels consume work from all pairs at the same trial pose. No host
// decision, geometry fallback, or trajectory replay occurs inside this loop.
struct BatchPair {
    Query query;
    MatPose a, b;
    int fresh, link_first, saved_paths, saved_blockers;
    int label, pass, columns, column_begin;
};
struct BatchRay {
    Ray ray;
    int pair, side, column;
};
struct BatchControl {
    DeviceScene scene;
    BatchPair *pairs;
    BatchRay *rays;
    int *deferred;
    Hit *audit_hits;
    int deferred_count;
    int *offsets;
    int ray_capacity, ray_count, total_columns;
    int step, end, level, cell, iter, evaluated, initial;
    double base_dt, dt, fraction;
    cudaGraphConditionalHandle force_graph, link_graph, integral_graph;
    unsigned long long stamp, accepted[9];
    unsigned long long ray_queries, ray_batches, precomputed_rays, deferred_rays, audited_rays, audited_groups;
};
__device__ Trial &batch_trial(BatchControl &g) {
    return g.evaluated ? g.scene.control->next : g.scene.control->working;
}
__device__ bool batch_enabled(const BatchControl &g, bool enabled = true) {
    // Conditional-graph workers cooperate within full warps. Publish one
    // decision per warp even if another block reports an error concurrently.
    const int run = (threadIdx.x % 32 == 0) ? !g.scene.control->status && enabled : 0;
    return __shfl_sync(0xffffffffu, run, 0);
}
// Only used by a 32-thread block. No cross-warp shared state is involved.
struct BatchWarp {
    __device__ int thread_rank() const { return threadIdx.x; }
    __device__ int size() const { return 32; }
    __device__ void sync() const { __syncwarp(); }
};
__device__ void batch_launch(BatchControl &g, cudaGraphConditionalHandle graph) {
    if (g.scene.control->status) return;
    cudaGraphSetConditional(graph, 1);
}
__device__ void batch_prepare_step(BatchControl &g) {
    auto &s = g.scene;
    auto &c = *s.control;
    g.dt = ldexp(g.base_dt, -g.level);
    g.iter = g.evaluated = 0;
    g.fraction = 0;
    c.last_step = g.step;
    for (int j = 0; j < s.np; ++j) s.cache[j].ready = 0;
    auto &trial = c.working;
    trial.state.time = c.state.time + g.dt;
    for (int i = 0; i < s.n; ++i) {
        c.qpred[i] = c.state.q[i] + g.dt * c.state.v[i] + g.dt * g.dt * (-.5) * c.state.a[i];
        c.vpred[i] = c.state.v[i] + g.dt * (-.5) * c.state.a[i];
        trial.state.a[i] = c.state.a[i];
        trial.state.q[i] = c.qpred[i] + g.dt * g.dt * trial.state.a[i];
        trial.state.v[i] = c.vpred[i] + 1.5 * g.dt * trial.state.a[i];
    }
}
__global__ void batch_start(BatchControl *control) {
    auto &g = *control;
    auto &s = g.scene;
    g.initial = 1;
    g.dt = g.base_dt;
    for (int i = 0; i < s.n; ++i) {
        s.control->working.state.q[i] = s.coordinates[i].q;
        s.control->working.state.v[i] = s.coordinates[i].v;
    }
    s.finished[0] = timer_ns();
    batch_launch(g, g.force_graph);
}
__global__ void batch_force_begin(BatchControl *control) {
    auto &g = *control;
    cudaGraphSetConditional(g.force_graph, 0);
    auto &s = g.scene;
    if (s.control->status) return;
    initialize_forces(s, batch_trial(g));
    const auto &state = batch_trial(g).state;
    g.stamp = timer_ns();
    for (int i = 0; i < s.np; ++i) {
        auto &p = g.pairs[i];
        auto &w = s.workspace[i + 1];
        const auto law = s.pairs[i];
        auto &cache = s.cache[i];
        p.a = body_pose(s.bodies[law.a], state.q);
        p.b = body_pose(s.bodies[law.b], state.q);
        p.query = {{{&s.views[law.a], &s.views[law.b]}},
                    {between(p.a, p.b), between(p.b, p.a)},
                    NC_DEVICE_QUERY_REACH * law.max_depth, &w};
        w.regions = 0;
        if (cache.ready)
            for (int k = 0; k < cache.anchors; ++k) {
                auto r = transport(cache.anchor[k], p.a, p.b, s.control->invalid);
                if (r.depth > 0) w.result[w.regions++] = r;
            }
        p.fresh = !w.regions || w.regions != cache.anchors || !cache.ready;
        if (!p.fresh) continue;
        w.count = w.groups = w.found = w.regions = 0;
        ++w.queries;
        for (int side = 0; side < 2; ++side) {
            auto &window = cache.windows[side];
            const auto &b = p.query.b[1 - side];
            const auto pose = p.query.rel[1 - side];
            const auto bounds = b.g.faces[0].box;
            V extreme{};
            for (int k = 0; k < 3; ++k)
                extreme.x[k] = fmax(fabs(bounds.lo.x[k]), fabs(bounds.hi.x[k]));
            double radius = norm(extreme), rotation = 0;
            for (int k = 0; k < 3; ++k) {
                const auto d = sub(pose.cols[k], window.pose.cols[k]);
                rotation += dot(d, d);
            }
            const double motion = norm(sub(pose.offset, window.pose.offset)) + radius * sqrt(rotation);
            window.rebuild = !window.valid || motion > window.margin;
            if (window.rebuild) {
                window.count = 0;
                window.pose = pose;
                window.valid = 1;
                window.margin = fmax(4 * p.query.reach, .02 * radius);
                ++w.window_builds;
            } else ++w.window_reuses;
        }
    }
}
__global__ void batch_windows(BatchControl *control) {
    auto &g = *control;
    if (g.scene.control->status) return;
    const int pair = blockIdx.y / 2, side = blockIdx.y % 2;
    auto &p = g.pairs[pair];
    auto &window = g.scene.cache[pair].windows[side];
    if (!p.fresh || !window.rebuild) return;
    const auto &q = p.query;
    const auto &a = q.b[side], &b = q.b[1 - side];
    const auto pose = q.rel[1 - side];
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < b.nt; t += gridDim.x * blockDim.x) {
        const int id = b.target_ids[t];
        const auto box = placed(pose, b.tiles[id].box);
        for (int i = 0; i < a.ns;) {
            const auto &node = a.sites[i];
            if (!overlaps(box, node.box, q.reach + window.margin + 2e-7)) { i = node.next; continue; }
            if (node.leaf >= 0) { window.tiles[atomicAdd(&window.count, 1)] = id; break; }
            ++i;
        }
    }
}
__global__ void batch_window_offsets(BatchControl *control) {
    auto &g = *control;
    g.offsets[0] = 0;
    for (int i = 0; i < 2 * g.scene.np; ++i)
        g.offsets[i + 1] = g.offsets[i] + (g.pairs[i / 2].fresh ? g.scene.cache[i / 2].windows[i % 2].count : 0);
}
__device__ int batch_owner(const int *offsets, int count, int index) {
    int lo = 0, hi = count;
    while (lo + 1 < hi) {
        int m = (lo + hi) / 2;
        if (offsets[m] <= index) lo = m; else hi = m;
    }
    return lo;
}
__global__ void batch_candidates(BatchControl *control) {
    auto &g = *control;
    if (g.scene.control->status) return;
    const int count = 2 * g.scene.np, total = g.offsets[count];
    for (int job = blockIdx.x * blockDim.x + threadIdx.x; job < total; job += blockDim.x * gridDim.x) {
        int owner = batch_owner(g.offsets, count, job), pair = owner / 2, side = owner % 2;
        const auto &q = g.pairs[pair].query;
        auto &w = *q.work;
        const auto &a = q.b[side], &b = q.b[1 - side];
        const auto pose = q.rel[1 - side];
        int tile = g.scene.cache[pair].windows[side].tiles[job - g.offsets[owner]];
        const auto &ref = b.tiles[tile];
        const Bounds box = placed(pose, ref.box);
        const V normal = direction(pose, ref.normal), center = point(pose, ref.point);
        for (int i = 0; i < a.ns;) {
            const auto &node = a.sites[i];
            if (!overlaps(box, node.box, q.reach + 1e-7)) { i = node.next; continue; }
            ++i;
            if (node.leaf < 0) continue;
            const auto &site = a.tiles[node.leaf];
            double cosine = -dot(site.normal, normal);
            if (cosine < .5) continue;
            double d = -dot(sub(site.point, center), normal) / cosine;
            if (fabs(d) > q.reach) continue;
            unsigned slot = atomicAdd(&w.count, 1u);
            if (slot >= w.capacity) { q.fail(1); continue; }
            w.proposals[slot] = {side ? Faces{ref.face, site.face} : Faces{site.face, ref.face}, side,
                                 node.leaf, tile, d};
        }
    }
}
template<int Phase> __global__ void batch_stamp(BatchControl *control) {
    auto &g = *control;
    const auto now = timer_ns(), elapsed = now - g.stamp;
    auto &w = *g.scene.workspace;
    if constexpr (Phase == 0) w.broad_ns += elapsed;
    if constexpr (Phase == 1) w.sort_ns += elapsed;
    g.stamp = now;
    if constexpr (Phase == 1) {
        g.offsets[0] = 0;
        for (int i = 0; i < g.scene.np; ++i)
            g.offsets[i + 1] = g.offsets[i] + (g.pairs[i].fresh ? g.pairs[i].query.work->groups : 0);
    }
}
constexpr int parallel_sort_limit = 1024;
__global__ void batch_sort_tiles(BatchControl *control) {
    auto &g = *control;
    auto &p = g.pairs[blockIdx.y];
    auto &w = *p.query.work;
    const int first = 32 * blockIdx.x, count = static_cast<int>(w.count);
    if (!batch_enabled(g, p.fresh && count > 32 && count <= parallel_sort_limit && first < count)) return;
    __shared__ Suggestion tile[32];
    const int lane = threadIdx.x, length = min(32, count - first);
    if (lane < length) tile[lane] = w.proposals[first + lane];
    __syncwarp();
    sort_proposals(BatchWarp{}, tile, length);
    if (lane < length) w.proposals[first + lane] = tile[lane];
}
__global__ void batch_sort_merge(BatchControl *control) {
    auto &g = *control;
    auto &p = g.pairs[blockIdx.y];
    auto &w = *p.query.work;
    const int count = static_cast<int>(w.count), index = blockIdx.x * blockDim.x + threadIdx.x;
    if (g.scene.control->status || !p.fresh || count <= 32 || count > parallel_sort_limit || index >= count) return;
    const auto candidate = w.proposals[index];
    int rank = index % 32;
    for (int first = 0; first < count; first += 32) {
        if (first == index - index % 32) continue;
        int lo = 0, hi = min(32, count - first);
        while (lo < hi) {
            const int middle = (lo + hi) / 2;
            const auto other = w.proposals[first + middle];
            // Original indices make equal entries distinct, so output ranks
            // are unique and require neither atomics nor block barriers.
            if (before(other, candidate) || (!before(candidate, other) && first + middle < index)) lo = middle + 1;
            else hi = middle;
        }
        rank += lo;
    }
    w.sorted_proposals[rank] = candidate;
}
__global__ void batch_sort(BatchControl *control) {
    auto &g = *control;
    const int pair = blockIdx.x;
    if (!batch_enabled(g, g.pairs[pair].fresh)) return;
    auto &q = g.pairs[pair].query;
    auto &w = *q.work;
    const bool merged = w.count > 32 && w.count <= parallel_sort_limit;
    if (merged && !threadIdx.x) {
        auto *previous = w.proposals; w.proposals = w.sorted_proposals; w.sorted_proposals = previous;
    }
    __syncwarp();
    // Group construction is ordered, but its keys can be loaded collectively.
    // Avoid a serial global-memory walk after the parallel merge scattered the
    // complete proposals. This copies integer IDs only, preserving group order.
    __shared__ Faces keys[parallel_sort_limit];
    if (merged) {
        for (int i = threadIdx.x; i < w.count; i += 32) keys[i] = w.proposals[i].faces;
        __syncwarp();
    }
    detect_sort(BatchWarp{}, q, g.scene.cache[pair], merged, merged ? keys : nullptr);
}
__global__ void batch_search(BatchControl *control) {
    auto &g = *control;
    if (!batch_enabled(g)) return;
    for (int job = blockIdx.x; job < g.offsets[g.scene.np]; job += gridDim.x) {
        int pair = batch_owner(g.offsets, g.scene.np, job), id = job - g.offsets[pair];
        auto &q = g.pairs[pair].query;
        auto &w = *q.work;
        const bool hit = q.find<32>(w.group[id], g.scene.cache[pair], w.confirmed[id]);
        if (!threadIdx.x) w.valid[id] = hit;
        __syncwarp();
    }
}
// Expand independent projections across the device, then consume their results
// in the original group/proposal order. Speculative faults only become visible
// when that candidate is selected by the same ordered search as the control.
__global__ void batch_search_seeds(BatchControl *control) {
    auto &g = *control;
    if (g.scene.control->status) return;
    for (int job = blockIdx.x; job < g.offsets[g.scene.np]; job += gridDim.x) {
        const int pair = batch_owner(g.offsets, g.scene.np, job), id = job - g.offsets[pair];
        const auto &q = g.pairs[pair].query;
        auto &w = *q.work;
        w.seed_attempts[id] = q.prior_attempt(w.group[id], g.scene.cache[pair]);
    }
}
__global__ void batch_search_proposals(BatchControl *control) {
    auto &g = *control;
    const int pair = blockIdx.y;
    if (g.scene.control->status || !g.pairs[pair].fresh) return;
    const auto &q = g.pairs[pair].query;
    auto &w = *q.work;
    const int width = w.count < gridDim.x * 32 ? 8 : 32;
    if (threadIdx.x >= width) return;
    for (int k = blockIdx.x * width + threadIdx.x; k < w.count; k += gridDim.x * width) {
        int lo = 0, hi = w.groups;
        while (lo + 1 < hi) {
            const int mid = (lo + hi) / 2;
            if (w.group[mid].begin <= k) lo = mid; else hi = mid;
        }
        const auto &seed = w.seed_attempts[lo];
        if (!seed.skip && !seed.hit && !seed.fault.code)
            w.attempts[k] = q.proposal_attempt(w.group[lo], k);
    }
}
__global__ void batch_search_ready(BatchControl *control) {
    auto &g = *control;
    if (!batch_enabled(g)) return;
    for (int job = blockIdx.x; job < g.offsets[g.scene.np]; job += gridDim.x) {
        const int pair = batch_owner(g.offsets, g.scene.np, job), id = job - g.offsets[pair];
        const auto &q = g.pairs[pair].query;
        auto &w = *q.work;
        const bool hit = q.find<32, true>(w.group[id], g.scene.cache[pair], w.confirmed[id], nullptr, id);
        if (!threadIdx.x) w.valid[id] = hit;
        __syncwarp();
    }
}
__device__ bool same_hit(const Hit &a, const Hit &b) {
    bool same = a.face == b.face && a.t == b.t && a.u == b.u && a.v == b.v;
    for (int k = 0; k < 3; ++k) same &= a.normal.x[k] == b.normal.x[k];
    return same;
}
__global__ void batch_search_audit(BatchControl *control) {
    auto &g = *control;
    if (!batch_enabled(g)) return;
    __shared__ Witness expected;
    for (int job = blockIdx.x; job < g.offsets[g.scene.np]; job += gridDim.x) {
        const int pair = batch_owner(g.offsets, g.scene.np, job), id = job - g.offsets[pair];
        const auto &q = g.pairs[pair].query;
        auto &w = *q.work;
        const bool hit = q.find<32>(w.group[id], g.scene.cache[pair], expected);
        if (!threadIdx.x) {
            const auto &a = w.confirmed[id];
            bool same = hit == bool(w.valid[id]);
            if (same && hit) {
                same = equal(a.faces, expected.faces) && a.depth == expected.depth &&
                    a.seed.side == expected.seed.side && a.seed.tile == expected.seed.tile &&
                    a.seed.u == expected.seed.u && a.seed.v == expected.seed.v &&
                    same_hit(a.exits[0], expected.exits[0]) && same_hit(a.exits[1], expected.exits[1]);
                for (int k = 0; k < 3; ++k)
                    same &= a.p.x[k] == expected.p.x[k] && a.n.x[k] == expected.n.x[k];
            }
            if (!same) q.fail(14);
            atomicAdd(&g.audited_groups, 1ULL);
        }
        __syncwarp();
    }
}
__device__ void batch_integral_start(BatchControl &g) {
    const auto now = timer_ns();
    g.scene.workspace->group_ns += now - g.stamp;
    g.stamp = now;
    for (int i = 0; i < g.scene.np; ++i) {
        auto &p = g.pairs[i];
        p.label = p.pass = 0;
        if (p.fresh) {
            auto &w = *p.query.work;
            while (p.label < w.found && w.labels[p.label] != p.label) ++p.label;
        }
    }
    batch_launch(g, g.integral_graph);
}
__global__ void batch_found(BatchControl *control) {
    auto &g = *control;
    auto &s = g.scene;
    const auto now = timer_ns();
    s.workspace->search_ns += now - g.stamp;
    g.stamp = now;
    if (s.control->status) return;
    bool links = false;
    for (int pair = 0; pair < s.np; ++pair) {
        auto &p = g.pairs[pair];
        if (!p.fresh) continue;
        auto &w = *p.query.work;
        auto &cache = s.cache[pair];
        for (int i = 0; i < w.groups; ++i) if (w.valid[i]) {
            const auto &v = w.confirmed[i];
            int k = w.found++;
            w.witnesses[k] = v;
            w.labels[k] = k;
            cache.previous[k] = {v.faces, v.seed};
        }
        cache.previous_count = w.found;
        w.confirmed_total += w.found;
        p.link_first = p.saved_paths = p.saved_blockers = 0;
        links |= w.found > 1;
        if (w.found <= 1) {
            cache.path_count = cache.blocker_count = 0;
            cache.paths_pose = p.query.rel[0];
        }
    }
    if (links) batch_launch(g, g.link_graph); else batch_integral_start(g);
}
__global__ void batch_links(BatchControl *control) {
    auto &g = *control;
    int pair = blockIdx.x / connection_batch, slot = blockIdx.x % connection_batch;
    auto &p = g.pairs[pair];
    auto &q = p.query;
    auto &w = *q.work;
    const int total = w.found * (w.found - 1) / 2;
    if (!batch_enabled(g, p.fresh && p.link_first + slot < total)) return;
    auto &cache = g.scene.cache[pair];
    int rank = p.link_first + slot, i = 0;
    while (rank >= w.found - 1 - i) { rank -= w.found - 1 - i; ++i; }
    const int j = i + rank + 1, lane = threadIdx.x;
    auto &trial = w.bridges[slot];
    if (!lane) {
        trial.left = i; trial.right = j;
        trial.fault = {}; trial.joined = false; trial.blockers = {};
        trial.blockers.left = w.witnesses[i].faces;
        trial.blockers.right = w.witnesses[j].faces;
        for (int k = 0; k < cache.blocker_count; ++k)
            if (equal(cache.blockers[k].left, trial.blockers.left) &&
                equal(cache.blockers[k].right, trial.blockers.right)) {
                trial.blockers = cache.blockers[k]; break;
            }
    }
    __syncwarp();
    if (w.labels[i] != w.labels[j]) {
        Query local = q;
        local.speculative = &trial.fault;
        local.blockers = &trial.blockers;
        bool joined = check_connection(local, cache, w.witnesses[i], w.witnesses[j], trial.path, lane);
        if (!lane) {
            trial.joined = joined;
            if (w.serial_links) {
                if (!agrees_with_serial_link(local, cache, w.witnesses[i], w.witnesses[j], trial,
                                             w.serial_links[slot])) q.fail(11);
                atomicAdd(&w.checked_links, 1ULL);
            }
        }
    }
}
__global__ void batch_links_finish(BatchControl *control) {
    auto &g = *control;
    cudaGraphSetConditional(g.link_graph, 0);
    if (g.scene.control->status) return;
    bool pending = false;
    for (int pair = 0; pair < g.scene.np; ++pair) {
        auto &p = g.pairs[pair];
        if (!p.fresh) continue;
        auto &q = p.query;
        auto &w = *q.work;
        auto &cache = g.scene.cache[pair];
        int total = w.found * (w.found - 1) / 2;
        if (p.link_first >= total) continue;
        int count = min(connection_batch, total - p.link_first);
        for (int k = 0; k < count; ++k) {
            const auto &t = w.bridges[k];
            int left = w.labels[t.left], right = w.labels[t.right];
            if (left == right) continue;
            if (t.fault.code) { q.record(t.fault); return; }
            if (t.blockers.count && p.saved_blockers < w.limit)
                cache.next_blockers[p.saved_blockers++] = t.blockers;
            if (!t.joined) continue;
            for (int j = 0; j < w.found; ++j) if (w.labels[j] == right) w.labels[j] = left;
            if (t.path.count) {
                if (p.saved_paths >= w.limit) { q.fail(1); return; }
                cache.next_paths[p.saved_paths++] = t.path;
            }
        }
        p.link_first += count;
        if (p.link_first < total) pending = true;
        else {
            auto *path = cache.paths; cache.paths = cache.next_paths; cache.next_paths = path;
            auto *blocker = cache.blockers; cache.blockers = cache.next_blockers; cache.next_blockers = blocker;
            cache.path_count = p.saved_paths; cache.blocker_count = p.saved_blockers;
            cache.paths_pose = q.rel[0];
        }
    }
    if (pending) batch_launch(g, g.link_graph); else batch_integral_start(g);
}
__global__ void batch_integral_prepare(BatchControl *control) {
    auto &g = *control;
    if (g.scene.control->status) return;
    g.ray_count = g.total_columns = g.deferred_count = 0;
    for (int pair = 0; pair < g.scene.np; ++pair) {
        auto &p = g.pairs[pair];
        auto &w = *p.query.work;
        p.columns = 0; p.column_begin = g.total_columns;
        if (!p.fresh || p.label >= w.found) continue;
        if (!p.pass) {
            w.active_faces = w.selected = 0;
            for (int side = 0; side < 2; ++side) {
                const int first = w.active_faces;
                for (int i = 0; i < w.found; ++i) if (w.labels[i] == p.label) {
                    int id = w.witnesses[i].faces[side], at = first;
                    while (at < w.active_faces && w.faces[at] < id) ++at;
                    if (at < w.active_faces && w.faces[at] == id) continue;
                    for (int k = w.active_faces; k > at; --k) w.faces[k] = w.faces[k - 1];
                    w.faces[at] = id;
                    w.sides[w.active_faces++] = side;
                }
            }
        }
        const int order = p.pass == 4 ? 24 : 12 << p.pass;
        p.columns = w.active_faces * order * order;
        if (p.columns > w.column_capacity) { p.query.fail(1); return; }
        g.total_columns += (p.columns + 31) / 32 * 32;
    }
}
__global__ void batch_footprints(BatchControl *control) {
    auto &g = *control;
    if (g.scene.control->status) return;
    auto &p = g.pairs[blockIdx.y];
    auto &w = *p.query.work;
    if (!p.columns || p.pass != 4) return;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < w.active_faces; k += blockDim.x * gridDim.x)
        w.windows[k] = footprint(p.query, p.label, w.sides[k], w.faces[k]);
}
__global__ void batch_column_tasks(BatchControl *control) {
    auto &g = *control;
    if (!batch_enabled(g)) return;
    const int lane = threadIdx.x % 32;
    for (int job = blockIdx.x * blockDim.x + threadIdx.x; job < g.total_columns;
         job += gridDim.x * blockDim.x) {
        int pair = 0;
        while (pair + 1 < g.scene.np && g.pairs[pair + 1].column_begin <= job) ++pair;
        auto &p = g.pairs[pair];
        auto &q = p.query;
        auto &w = *q.work;
        int index = job - p.column_begin;
        if (!lane) w.active_columns[index / 32] = 0;
        bool possible = false;
        BatchRay task{};
        if (index < p.columns) {
            const bool local_rule = p.pass == 4;
            const int order = local_rule ? 24 : 12 << p.pass;
            int face_index = index / (order * order), side = w.sides[face_index], id = w.faces[face_index];
            int ij = index % (order * order);
            const auto r = local_rule ? w.windows[face_index] : q.b[side].g.charts[id].domain;
            const double du = (r.U - r.u) / order, dv = (r.Vv - r.v) / order;
            const double u = r.u + (ij / order + .5) * du, v = r.v + (ij % order + .5) * dv;
            const auto &geometry = q.b[side].g;
            const int base = geometry.charts[id].rule_start;
            SampleDepth value{};
            value.depth = -1; value.side = side;
            bool valid = false;
            if (!local_rule && order <= 24 && base >= 0) {
                auto cached = geometry.rules[base + (order == 12 ? 0 : 144) + ij];
                value.s = {cached.point, cached.normal, u, v, cached.area}; valid = cached.area > 0;
            } else if (!local_rule && order >= 48 && q.b[side].refined_rules) {
                auto cached = q.b[side].refined_rules[id * refined_rule_sites + (order == 48 ? 0 : 48 * 48) + ij];
                value.s = {cached.point, cached.normal, u, v, cached.area}; valid = cached.area > 0;
            } else if (q.inside(side, id, u, v) && q.node(side, id, u, v, value.s)) {
                value.s.area = local_rule ? value.s.area * ((r.U - r.u) * (r.Vv - r.v) / (order * order))
                                         : value.s.area * du * dv;
                valid = true;
            }
            if (valid) {
                V origin = point(q.rel[side], value.s.p), direction0 = direction(q.rel[side], scale(value.s.n, -1));
                for (int j = 0; j < w.found && !possible; ++j)
                    if (w.labels[j] == p.label && w.witnesses[j].faces[side] == id)
                        possible = intersects(q.b[1 - side].g.charts[w.witnesses[j].faces[1 - side]].box,
                                              {origin, direction0}, q.reach, 1e-7);
                if (possible) {
                    w.columns[index] = value;
                    task = {{origin, direction0}, pair, side, index};
                }
            }
        }
        unsigned mask = __ballot_sync(0xffffffffu, possible);
        int begin = 0;
        if (!lane && mask) begin = atomicAdd(&g.ray_count, __popc(mask));
        begin = __shfl_sync(0xffffffffu, begin, 0);
        if (possible) {
            int at = begin + __popc(mask & ((1u << lane) - 1));
            if (at >= g.ray_capacity) q.fail(1); else g.rays[at] = task;
        }
    }
}
// One compact queue spans all active regions and rigid-body pairs. Scatter to
// original column indices preserves the deterministic integration ordering.
__global__ void batch_trace_columns(BatchControl *control) {
    auto &g = *control;
    if (g.scene.control->status) return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < g.ray_count; i += gridDim.x * blockDim.x) {
        const auto task = g.rays[i];
        auto &p = g.pairs[task.pair];
        auto &q = p.query;
        auto &w = *q.work;
        const auto hit = q.ray(1 - task.side, task.ray.p, task.ray.d, q.reach);
        const int order = p.pass == 4 ? 24 : 12 << p.pass;
        const int face = w.faces[task.column / (order * order)];
        if (hit.face >= 0 && accepts(w, p.label, task.side, face, hit.face)) {
            w.columns[task.column].depth = hit.t;
            atomicOr(&w.active_columns[task.column / 32], 1u << (task.column % 32));
        }
    }
}
// The common query does not reference recursive subdivision or trim isolation.
// An undecided query is compacted for the original complete GPU solver. Starting
// the deferred query from scratch retains all root/ambiguity/error ordering.
__device__ void batch_accept_hit(BatchControl &g, const BatchRay &task, const Hit &hit) {
    auto &p = g.pairs[task.pair];
    auto &w = *p.query.work;
    const int order = p.pass == 4 ? 24 : 12 << p.pass;
    const int face = w.faces[task.column / (order * order)];
    if (hit.face >= 0 && accepts(w, p.label, task.side, face, hit.face)) {
        w.columns[task.column].depth = hit.t;
        atomicOr(&w.active_columns[task.column / 32], 1u << (task.column % 32));
    }
}
__global__ void batch_trace_precomputed(BatchControl *control) {
    auto &g = *control;
    if (g.scene.control->status) return;
    // Small ray batches spread over more SMs, with independent warps hiding
    // the latency of dependent CAD loads. Large batches retain full warps.
    const int width = g.ray_count < gridDim.x * 32 ? 4 : 32;
    if (threadIdx.x >= width) return;
    for (int i = blockIdx.x * width + threadIdx.x; i < g.ray_count; i += gridDim.x * width) {
        const auto task = g.rays[i];
        const auto &q = g.pairs[task.pair].query;
        const Hit hit = first<true>(q.b[1 - task.side].g, task.ray, q.reach);
        if (g.audit_hits) g.audit_hits[i] = hit;
        if (hit.face == -3) g.deferred[atomicAdd(&g.deferred_count, 1)] = i;
        else batch_accept_hit(g, task, hit);
    }
}
__global__ void batch_trace_deferred(BatchControl *control) {
    auto &g = *control;
    if (g.scene.control->status) return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < g.deferred_count; i += gridDim.x * blockDim.x) {
        const auto task = g.rays[g.deferred[i]];
        const auto &q = g.pairs[task.pair].query;
        batch_accept_hit(g, task, q.ray(1 - task.side, task.ray.p, task.ray.d, q.reach));
    }
}
__global__ void batch_audit_precomputed(BatchControl *control) {
    auto &g = *control;
    if (g.scene.control->status) return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < g.ray_count; i += gridDim.x * blockDim.x) {
        const Hit actual = g.audit_hits[i];
        if (actual.face == -3) continue;
        const auto task = g.rays[i];
        const auto &q = g.pairs[task.pair].query;
        const Hit expected = q.ray(1 - task.side, task.ray.p, task.ray.d, q.reach);
        bool same = actual.face == expected.face;
        if (same && actual.face >= 0) {
            same = actual.t == expected.t && actual.u == expected.u && actual.v == expected.v;
            for (int k = 0; k < 3; ++k) same &= actual.normal.x[k] == expected.normal.x[k];
        }
        if (!same) q.fail(13);
        atomicAdd(&g.audited_rays, 1ULL);
    }
}
__global__ void batch_integral_reduce(BatchControl *control) {
    auto &g = *control;
    auto &p = g.pairs[blockIdx.x];
    if (!batch_enabled(g, p.columns > 0)) return;
    auto &q = p.query;
    auto &w = *q.work;
    const int lane = threadIdx.x;
    const auto start = timer_ns();
    IntegralSum sum{};
    if (w.reduction == RegionReduction::ordered || w.reduction == RegionReduction::audit)
        sum = ordered_columns(q, p.columns, lane);
    else if (w.reduction != RegionReduction::serial) sum = reduce_columns(q, p.columns, lane);
    else if (!lane) sum = sum_columns(q, p.columns, 0, 1);
    if (!lane) {
        w.reduction_ns += timer_ns() - start;
        ++w.reductions;
        Region actual{};
        bool valid = make_region(q, sum, actual);
        if (w.reference) {
            const auto original = sum_columns(q, p.columns, 0, 1);
            if (w.reduction == RegionReduction::audit) {
                for (int k = 0; k < 3; ++k)
                    if (sum.area.x[k] != original.area.x[k] || sum.moment.x[k] != original.moment.x[k] ||
                        sum.centroid.x[k] != original.centroid.x[k]) q.fail(10);
                if (sum.volume != original.volume) q.fail(10);
            }
            Region reference{};
            bool expected = make_region(q, original, reference);
            if (valid != expected) q.fail(10);
            if (expected) w.reference[w.regions] = reference;
        }
        if (valid) { w.result[w.regions++] = actual; w.selected = 1; }
    }
}
__device__ void batch_force_finish(BatchControl &g) {
    auto &s = g.scene;
    auto &c = *s.control;
    auto &trial = batch_trial(g);
    s.workspace->integral_ns += timer_ns() - g.stamp;
    for (int pair = 0; pair < s.np; ++pair) {
        auto &p = g.pairs[pair];
        auto &w = *p.query.work;
        auto &cache = s.cache[pair];
        if (p.fresh) {
            if (w.audit) audit_regions(s, trial.state, s.pairs[pair], p.a, p.b, pair, g.initial, &w);
            if (c.status) return;
            for (int k = 0; k < w.regions; ++k) {
                auto &r = w.result[k];
                r.p = world(p.a, scale(r.p, NC_DEVICE_GEOMETRY_SCALE));
                r.n = rotate(p.a, r.n);
                r.arm = rotate(p.a, scale(r.arm, NC_DEVICE_GEOMETRY_SCALE));
                r.depth *= NC_DEVICE_GEOMETRY_SCALE;
                if (!cache.ready) {
                    V half = scale(r.n, .5 * r.depth);
                    cache.anchor[k] = {local(p.a, plus(r.p, half)), local(p.b, sub(r.p, half)),
                                       rotate(p.a, r.n, -1), rotate(p.b, r.n, -1),
                                       rotate(p.a, r.arm, -1), rotate(p.b, r.arm, -1)};
                }
            }
            if (!cache.ready) { cache.anchors = w.regions; cache.ready = 1; }
        }
        accumulate_pair(s, trial, pair, &w);
    }
    linearize_forces(s, trial, g.dt, g.initial);
    if (g.initial) {
        if (c.invalid) { c.status = 8; return; }
        copy_state(c.state, c.working.state, s.n);
        for (int i = 0; i < s.n; ++i) c.state.a[i] = c.state.force[i] / s.coordinates[i].mass;
        save_frame(s, 0);
        g.initial = 0;
    } else {
        const auto dispatch = correct_step(s, g.dt, g.iter, g.fraction, g.evaluated);
        if (c.status) return;
        if (dispatch == refresh_contact) {
            g.evaluated = 1;
            batch_launch(g, g.force_graph); return;
        }
        if (dispatch == reject_step) {
            if (g.level == 8) { c.status = 7; return; }
            ++g.level; g.cell *= 2; ++s.workspace->subdivisions;
        } else {
            ++g.accepted[g.level];
            ++g.cell;
            while (g.level && !(g.cell & 1)) { g.cell /= 2; --g.level; }
            if (!g.level) {
                ++g.step; g.cell = 0;
                save_frame(s, g.step); s.finished[g.step] = timer_ns();
            }
        }
    }
    if (g.step < g.end) { batch_prepare_step(g); batch_launch(g, g.force_graph); }
}
__global__ void batch_integral_finish(BatchControl *control) {
    auto &g = *control;
    cudaGraphSetConditional(g.integral_graph, 0);
    if (g.scene.control->status) return;
    g.ray_queries += g.ray_count;
    if (g.deferred) { g.precomputed_rays += g.ray_count - g.deferred_count; g.deferred_rays += g.deferred_count; }
    g.ray_batches += g.ray_count > 0;
    bool pending = false;
    for (int pair = 0; pair < g.scene.np; ++pair) {
        auto &p = g.pairs[pair];
        auto &w = *p.query.work;
        if (!p.columns) continue;
        if (w.selected) {
            do { ++p.label; } while (p.label < w.found && w.labels[p.label] != p.label);
            p.pass = 0;
        } else if (++p.pass == 5) { p.query.fail(6); return; }
        pending |= p.label < w.found;
    }
    if (pending) batch_launch(g, g.integral_graph); else batch_force_finish(g);
}

struct BatchGraphBuilder {
    cudaGraph_t graph, root;
    cudaGraphNode_t last{};
    template<class F, class... A> void kernel(F function, dim3 grid, int block, A... values) {
        void *args[] = {static_cast<void *>(&values)...};
        cudaKernelNodeParams p{};
        p.func = reinterpret_cast<void *>(function); p.gridDim = grid; p.blockDim = dim3(block);
        p.kernelParams = args;
        cudaGraphNode_t node{};
        check(cudaGraphAddKernelNode(&node, graph, last ? &last : nullptr, last ? 1 : 0, &p));
        last = node;
    }
    BatchGraphBuilder loop(cudaGraphConditionalHandle &handle) {
        check(cudaGraphConditionalHandleCreate(&handle, root, 0, 0));
        cudaGraphNodeParams p{};
        p.type = cudaGraphNodeTypeConditional;
        p.conditional.handle = handle;
        p.conditional.type = cudaGraphCondTypeWhile;
        p.conditional.size = 1;
        cudaGraphNode_t node{};
        check(cudaGraphAddNode(&node, graph, last ? &last : nullptr, last ? 1 : 0, &p));
        last = node;
        return {p.conditional.phGraph_out[0], root};
    }
};
void build_batch_graphs(StepGraph &schedule, BatchControl *g, BatchControl &host, int pairs, int sms, bool wavefront, bool audit) {
    check(cudaGraphCreate(&schedule.graph, 0));
    BatchGraphBuilder root{schedule.graph, schedule.graph};
    root.kernel(batch_start, 1, 1, g);
    auto f = root.loop(host.force_graph);
    f.kernel(batch_force_begin, 1, 1, g);
    f.kernel(batch_windows, dim3(sms, 2 * pairs), 128, g);
    f.kernel(batch_window_offsets, 1, 1, g);
    f.kernel(batch_candidates, sms * 4, 128, g);
    f.kernel(batch_stamp<0>, 1, 1, g);
    f.kernel(batch_sort_tiles, dim3(parallel_sort_limit / 32, pairs), 32, g);
    f.kernel(batch_sort_merge, dim3(parallel_sort_limit / 128, pairs), 128, g);
    f.kernel(batch_sort, pairs, 32, g);
    f.kernel(batch_stamp<1>, 1, 1, g);
    if (wavefront) {
        f.kernel(batch_search_seeds, sms * 2, 1, g);
        f.kernel(batch_search_proposals, dim3(sms * 4, pairs), 32, g);
        f.kernel(batch_search_ready, sms * 2, 32, g);
        if (audit) f.kernel(batch_search_audit, sms * 2, 32, g);
    } else f.kernel(batch_search, sms * 2, 32, g);
    f.kernel(batch_found, 1, 1, g);
    auto l = f.loop(host.link_graph);
    l.kernel(batch_links, pairs * connection_batch, 32, g);
    l.kernel(batch_links_finish, 1, 1, g);
    auto i = f.loop(host.integral_graph);
    i.kernel(batch_integral_prepare, 1, 1, g);
    i.kernel(batch_footprints, dim3(1, pairs), 32, g);
    i.kernel(batch_column_tasks, sms * 4, 128, g);
    if (wavefront) {
        i.kernel(batch_trace_precomputed, sms * 4, 32, g);
        i.kernel(batch_trace_deferred, sms * 4, 32, g);
        if (audit) i.kernel(batch_audit_precomputed, sms * 4, 32, g);
    } else i.kernel(batch_trace_columns, sms * 4, 32, g);
    i.kernel(batch_integral_reduce, pairs, 32, g);
    i.kernel(batch_integral_finish, 1, 1, g);
    check(cudaGraphInstantiate(&schedule.executable, schedule.graph, 0));
}
