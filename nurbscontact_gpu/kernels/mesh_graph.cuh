// Each geometry phase has its own launch geometry and register allocation.
// All loop decisions, current poses, forces, and implicit states remain on device.
struct MeshGraphState {
    Query query;
    MatPose a, b;
    int step, end, level, cell, iter, evaluated, initial, support_blocks;
    double base_dt, dt, fraction;
    U64 stamp, step_condition, newton_condition, fresh_condition;
    U64 accepted_by_level[9], geometry_retries[2], newton_rejections;
    double first_geometry_failure[4]; // status, accepted time, trial time, dt
};
__device__ Trial &graph_trial(DeviceScene &s, MeshGraphState &g) {
    return g.evaluated ? s.control->next : s.control->working;
}
__device__ void mesh_graph_local_query(const MeshGraphState *g, Query &q, Workspace &w) {
    // Geometry metadata is uniform for the block. Keep its many dependent
    // pointer loads in shared memory; vertex/triangle arrays stay in global memory.
    for (int i = threadIdx.x; i < sizeof(Query); i += blockDim.x)
        reinterpret_cast<unsigned char *>(&q)[i] = reinterpret_cast<const unsigned char *>(&g->query)[i];
    for (int i = threadIdx.x; i < sizeof(Workspace); i += blockDim.x)
        reinterpret_cast<unsigned char *>(&w)[i] = reinterpret_cast<const unsigned char *>(g->query.work)[i];
    __syncthreads();
    if (!threadIdx.x)
        q.work = &w;
    __syncthreads();
}
extern "C" __global__ void mesh_graph_size(int *out) { *out = sizeof(MeshGraphState); }
extern "C" __global__ void mesh_graph_init(DeviceScene *s, MeshGraphState *g, int end, double dt, U64 steps,
                                           U64 newton, U64 fresh, int support_blocks) {
    *g = {};
    g->end = end;
    g->base_dt = g->dt = dt;
    g->initial = 1;
    g->support_blocks = support_blocks;
    g->step_condition = steps;
    g->newton_condition = newton;
    g->fresh_condition = fresh;
    auto &c = *s->control;
    auto &w = *s->workspace;
    c = {};
    w.queries = w.evaluations = w.subdivisions = w.broad_ns = w.narrow_ns = w.group_ns = w.support_ns = 0;
    w.maximum_residual = 0;
    for (int j = 0; j < s->np; ++j)
        s->cache[j].ready = 0;
    for (int i = 0; i < s->n; ++i) {
        c.working.state.q[i] = s->coordinates[i].q;
        c.working.state.v[i] = s->coordinates[i].v;
    }
    s->finished[0] = timer_ns();
}
extern "C" __global__ void mesh_graph_force_init(DeviceScene *s, MeshGraphState *g) {
    if (!s->control->status)
        initialize_forces(*s, graph_trial(*s, *g));
}
extern "C" __global__ void mesh_graph_pair(DeviceScene *s, MeshGraphState *g, int pair, U64 condition) {
    g->fresh_condition = condition;
    auto &c = *s->control;
    auto &w = *s->workspace;
    if (c.status) {
        cudaGraphSetConditional(g->fresh_condition, 0);
        return;
    }
    auto law = s->pairs[pair];
    auto &state = graph_trial(*s, *g).state;
    auto &cache = s->cache[pair];
    g->a = body_pose(s->bodies[law.a], state.q);
    g->b = body_pose(s->bodies[law.b], state.q);
    g->query = {{s->views[law.a], s->views[law.b]},
                {between(g->a, g->b), between(g->b, g->a)},
                4 * law.max_depth,
                &w};
    c.current_pair = w.pair = pair;
    w.regions = 0;
    if (cache.ready)
        for (int k = 0; k < cache.anchors; ++k) {
            Region r = transport(cache.anchor[k], g->a, g->b, c.invalid);
            if (r.depth > 0)
                w.result[w.regions++] = r;
        }
    w.selected = (!w.regions || w.regions != cache.anchors || !cache.ready);
    cudaGraphSetConditional(g->fresh_condition, w.selected);
    if (w.selected) {
        ++w.queries;
        w.counts[0] = w.counts[1] = w.filtered = w.refresh = 0;
        auto p = g->query.rel[1];
        for (int k = 0; k < 3; ++k) {
            for (int j = 0; j < 3; ++j)
                w.pose[4 * k + j] = component(p.cols[j], k);
            w.pose[4 * k + 3] = component(p.offset, k);
        }
        g->stamp = timer_ns();
    }
}
extern "C" __global__ void mesh_graph_broad(DeviceScene *s, MeshGraphState *g) {
    if (s->control->status)
        return;
    auto &w = *s->workspace;
    auto &p = w.pairs[w.pair];
    auto &q = g->query;
    mesh_detect::collect_leaves(p.leaves, p.leaves_n, p.bounds, p.traversal, p.nodes, w.pose, p.candidates,
                                p.candidate_n, w.counts, q.b[0].convex && q.b[1].convex ? q.reach : 0.);
}
extern "C" __global__ void mesh_graph_filter(DeviceScene *s, MeshGraphState *g) {
    if (s->control->status)
        return;
    auto &w = *s->workspace;
    auto &p = w.pairs[w.pair];
    if (w.counts[1] > p.candidate_n) {
        if (!blockIdx.x && !threadIdx.x)
            atomicCAS(w.status, 0, 23);
        return;
    }
    if (!blockIdx.x && !threadIdx.x) {
        auto now = timer_ns();
        w.broad_ns += now - g->stamp;
        g->stamp = now;
    }
    mesh_detect::batch_filter(&p, w.pose, w.counts, &w.filtered, &w.refresh, 0.);
}
extern "C" __global__ void mesh_graph_narrow(DeviceScene *s, MeshGraphState *g) {
    if (s->control->status)
        return;
    auto &w = *s->workspace;
    auto &p = w.pairs[w.pair];
    mesh_detect::batch_group_intersections(&p, w.pose, w.counts, &w.filtered, 0., 8);
}
extern "C" __global__ void mesh_graph_place(DeviceScene *s, MeshGraphState *g) {
    if (s->control->status)
        return;
    auto &w = *s->workspace;
    auto &q = g->query;
    auto &p = w.pairs[w.pair];
    int worker = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    if (w.counts[0] > p.hit_n) {
        if (!worker)
            atomicCAS(w.status, 0, 24);
        return;
    }
    if (!worker) {
        auto now = timer_ns();
        w.narrow_ns += now - g->stamp;
        g->stamp = now;
        w.regions = 0;
    }
    for (int i = worker; i < w.counts[0]; i += stride) {
        int2 hit = p.hits[i];
        p.hits[i] = make_int2(hit.y, hit.x);
    }
    if (!w.counts[0])
        return;
    for (int i = worker; i < q.b[1].nv; i += stride)
        w.placed_vertices[i] = point(q.rel[1], q.b[1].vertices[i]);
    for (int i = worker; i < q.b[1].nf; i += stride)
        w.placed_normals[i] = direction(q.rel[1], q.b[1].normals[i]);
}
extern "C" __global__ void mesh_graph_confirm(DeviceScene *s, MeshGraphState *g) {
    if (s->control->status)
        return;
    auto &w = *s->workspace;
    auto &p = w.pairs[w.pair];
    int worker = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = worker; i < w.counts[0]; i += gridDim.x * blockDim.x)
        if (!mesh_hit_plane_separated(g->query, p.hits[i]))
            p.filtered[atomicAdd(&w.refresh, 1)] = p.hits[i];
}
extern "C" __global__ void mesh_graph_group(DeviceScene *s, MeshGraphState *g) {
    if (s->control->status)
        return;
    auto &w = *s->workspace;
    __shared__ Query q;
    __shared__ Workspace local_workspace;
    mesh_graph_local_query(g, q, local_workspace);
    if (!threadIdx.x)
        w.counts[0] = local_workspace.counts[0] = w.refresh;
    __syncthreads();
    auto &pair = w.pairs[w.pair];
    for (int i = threadIdx.x; i < w.counts[0]; i += blockDim.x)
        pair.hits[i] = pair.filtered[i];
    __syncthreads();
    check_enclosure(q);
    __syncthreads();
    if (*w.status)
        return;
    if (w.counts[0] && !(q.b[0].convex && q.b[1].convex))
        sort_hits(q);
    if (!threadIdx.x) {
        w.groups = 0;
        if (w.counts[0] && !(q.b[0].convex && q.b[1].convex))
            for (int i = 0; i < w.counts[0]; ++i)
                w.groups += root(w.parents, i) == i;
        auto now = timer_ns();
        w.group_ns += now - g->stamp;
        g->stamp = now;
    }
}
extern "C" __global__ void mesh_graph_support(DeviceScene *s, MeshGraphState *g) {
    // Another support block may report failure while this block is starting.
    // Snapshot once per block so every thread reaches the same barriers.
    __shared__ int stopped;
    if (!threadIdx.x)
        stopped = s->control->status;
    __syncthreads();
    if (stopped)
        return;
    auto &w = *s->workspace;
    if (!w.counts[0])
        return;
    __shared__ Query q;
    __shared__ Workspace local_workspace;
    mesh_graph_local_query(g, q, local_workspace);
    if (q.b[0].convex && q.b[1].convex) {
        convex_partial(q);
        return;
    }
    for (int i = blockIdx.x; i < w.counts[0]; i += gridDim.x)
        if (root(w.parents, i) == i)
            component_support(q, i);
}
extern "C" __global__ void mesh_graph_collect(DeviceScene *s, MeshGraphState *g) {
    if (s->control->status)
        return;
    auto &w = *s->workspace;
    auto &q = g->query;
    auto &cache = s->cache[w.pair];
    if (w.counts[0]) {
        if (q.b[0].convex && q.b[1].convex)
            convex_finish(q, g->support_blocks);
        else {
            w.regions = 0;
            double eps = 1e-12 * (q.b[0].radius + q.b[1].radius + norm(q.rel[1].offset));
            for (int i = 0; i < w.counts[0]; ++i)
                if (root(w.parents, i) == i && w.region_valid[i]) {
                    auto r = w.unsorted[i];
                    bool same = false;
                    for (int j = 0; j < w.regions; ++j) {
                        auto old = w.result[j];
                        V p = sub(r.p, old.p), n = sub(r.n, old.n);
                        same |=
                            fabs(r.depth - old.depth) <= eps && dot(p, p) <= eps * eps && dot(n, n) <= 1e-18;
                    }
                    if (!same)
                        w.result[w.regions++] = r;
                }
        }
    }
    if (*w.status)
        return;
    for (int k = 0; k < w.regions; ++k) {
        auto &r = w.result[k];
        r.p = world(g->a, r.p);
        r.n = rotate(g->a, r.n);
        r.arm = rotate(g->a, r.arm);
        if (!cache.ready) {
            V half = scale(r.n, .5 * r.depth);
            cache.anchor[k] = {local(g->a, plus(r.p, half)), local(g->b, sub(r.p, half)),
                               rotate(g->a, r.n, -1),        rotate(g->b, r.n, -1),
                               rotate(g->a, r.arm, -1),      rotate(g->b, r.arm, -1)};
        }
    }
    if (!cache.ready) {
        cache.anchors = w.regions;
        cache.ready = 1;
    }
    w.support_ns += timer_ns() - g->stamp;
}
extern "C" __global__ void mesh_graph_accumulate(DeviceScene *s, MeshGraphState *g, int pair) {
    if (!s->control->status)
        accumulate_pair(*s, graph_trial(*s, *g), pair);
}
extern "C" __global__ void mesh_graph_linearize(DeviceScene *s, MeshGraphState *g) {
    if (!s->control->status)
        linearize_forces(*s, graph_trial(*s, *g), g->dt, g->initial);
}
extern "C" __global__ void mesh_graph_initial_finish(DeviceScene *s, MeshGraphState *g) {
    auto &c = *s->control;
    if (c.invalid && !c.status)
        c.status = 8;
    if (!c.status) {
        c.state = c.working.state;
        for (int i = 0; i < s->n; ++i)
            c.state.a[i] = c.state.force[i] / s->coordinates[i].mass;
        save_frame(*s, 0);
    }
    g->initial = 0;
    cudaGraphSetConditional(g->step_condition, !c.status && g->end > 0);
}
extern "C" __global__ void mesh_graph_prepare_step(DeviceScene *s, MeshGraphState *g) {
    auto &c = *s->control;
    if (c.status) {
        cudaGraphSetConditional(g->newton_condition, 0);
        return;
    }
    g->dt = ldexp(g->base_dt, -g->level);
    g->iter = 0;
    g->fraction = 0;
    g->evaluated = 0;
    c.last_step = g->step;
    for (int j = 0; j < s->np; ++j)
        s->cache[j].ready = 0;
    auto &trial = c.working;
    trial.state.time = c.state.time + g->dt;
    for (int i = 0; i < s->n; ++i) {
        c.qpred[i] = c.state.q[i] + g->dt * c.state.v[i] + g->dt * g->dt * (-.5) * c.state.a[i];
        c.vpred[i] = c.state.v[i] + g->dt * (-.5) * c.state.a[i];
        trial.state.a[i] = c.state.a[i];
        trial.state.q[i] = c.qpred[i] + g->dt * g->dt * trial.state.a[i];
        trial.state.v[i] = c.vpred[i] + 1.5 * g->dt * trial.state.a[i];
    }
    cudaGraphSetConditional(g->newton_condition, 1);
}
extern "C" __global__ void mesh_graph_newton(DeviceScene *s, MeshGraphState *g) {
    auto &c = *s->control;
    if (c.status) {
        cudaGraphSetConditional(g->newton_condition, 0);
        return;
    }
    c.dispatch = correct_step(*s, g->dt, g->iter, g->fraction, g->evaluated);
    g->evaluated = 1;
    cudaGraphSetConditional(g->newton_condition, !c.status && c.dispatch == refresh_contact);
}
extern "C" __global__ void mesh_graph_finish_step(DeviceScene *s, MeshGraphState *g) {
    auto &c = *s->control;
    if (retryable_mesh_contact(c.status) && g->level < 8) {
        if (!g->geometry_retries[0] && !g->geometry_retries[1]) {
            g->first_geometry_failure[0] = c.status;
            g->first_geometry_failure[1] = c.state.time;
            g->first_geometry_failure[2] = graph_trial(*s, *g).state.time;
            g->first_geometry_failure[3] = g->dt;
        }
        ++g->geometry_retries[c.status - 26];
        c.status = 0;
        // No trial with incomplete contacts has been committed to c.state.
        // prepare_step clears every contact cache before the retry.
        c.dispatch = reject_step;
    } else if (!c.status && c.dispatch == reject_step) {
        ++g->newton_rejections;
    }
    if (!c.status) {
        if (c.dispatch == reject_step) {
            if (g->level == 8)
                c.status = 7;
            else {
                ++g->level;
                g->cell *= 2;
                ++s->workspace->subdivisions;
            }
        } else {
            ++g->accepted_by_level[g->level];
            ++g->cell;
            while (g->level && !(g->cell & 1)) {
                g->cell /= 2;
                --g->level;
            }
            if (!g->level) {
                ++g->step;
                g->cell = 0;
                save_frame(*s, g->step);
                s->finished[g->step] = timer_ns();
            }
        }
    }
    cudaGraphSetConditional(g->step_condition, !c.status && g->step < g->end);
}

extern "C" __global__ void mesh_graph_telemetry(MeshGraphState *g, double *out) {
    for (int i = 0; i < 9; ++i)
        out[i] = g->accepted_by_level[i];
    out[9] = g->geometry_retries[0];
    out[10] = g->geometry_retries[1];
    out[11] = g->newton_rejections;
    for (int i = 0; i < 4; ++i)
        out[12 + i] = g->first_geometry_failure[i];
}

// Download only after the graph has stopped; never exchanges trial poses online.
extern "C" __global__ void mesh_graph_snapshot(DeviceScene *s, MeshGraphState *g, double *out,
                                                double *placements) {
    const auto &c = *s->control;
    const auto &trial = graph_trial(*s, *g).state;
    out[0] = c.status;
    out[1] = g->step;
    out[2] = g->level;
    out[3] = g->cell;
    out[4] = g->iter;
    out[5] = g->evaluated;
    out[6] = c.state.time;
    out[7] = trial.time;
    out[8] = g->dt;
    out[9] = c.current_pair;
    out[10] = s->workspace->counts[0];
    out[11] = s->workspace->groups;
    for (int i = 0; i < s->b; ++i) {
        auto p = body_pose(s->bodies[i], trial.q);
        double *v = placements + 7 * i;
        v[0] = p.origin.x;
        v[1] = p.origin.y;
        v[2] = p.origin.z;
        v[3] = p.axis.x;
        v[4] = p.axis.y;
        v[5] = p.axis.z;
        v[6] = p.angle;
    }
}
