// REJECTED EXPERIMENT: CUDA synchronization safety checks failed.
// This file is not compiled or exposed as a runtime execution mode.
// Keep the geometry and force definitions shared with the cooperative control.
// Only execution is split: geometry launches no longer carry the Newton solver.
struct CadGraphState {
    Query query;
    MatPose a, b;
    int step, end, level, cell, iter, evaluated, initial;
    double base_dt, dt, fraction;
    cudaGraphConditionalHandle steps, newton;
    unsigned long long accepted_by_level[9], newton_rejections;
};
__device__ Trial &cad_graph_trial(DeviceScene *s, CadGraphState *g) {
    return g->evaluated ? s->control->next : s->control->working;
}
__global__ void cad_graph_init(DeviceScene *s, CadGraphState *g, int end, double dt,
                              cudaGraphConditionalHandle steps, cudaGraphConditionalHandle newton) {
    *g = {};
    g->end = end;
    g->base_dt = g->dt = dt;
    g->initial = 1;
    g->steps = steps;
    g->newton = newton;
    auto &c = *s->control;
    c = {};
    for (int i = 0; i < s->n; ++i) {
        c.working.state.q[i] = s->coordinates[i].q;
        c.working.state.v[i] = s->coordinates[i].v;
    }
    s->finished[0] = timer_ns();
}
__global__ void cad_graph_force_init(DeviceScene *s, CadGraphState *g) {
    if (!s->control->status)
        initialize_forces(*s, cad_graph_trial(s, g));
}
__global__ void cad_graph_pair(DeviceScene *s, CadGraphState *g, int pair,
                              cudaGraphConditionalHandle fresh) {
    auto &c = *s->control;
    if (c.status) {
        cudaGraphSetConditional(fresh, 0);
        return;
    }
    auto &w = *s->workspace;
    auto &state = cad_graph_trial(s, g).state;
    auto law = s->pairs[pair];
    auto &cache = s->cache[pair];
    c.current_pair = pair;
    g->a = body_pose(s->bodies[law.a], state.q);
    g->b = body_pose(s->bodies[law.b], state.q);
    g->query = {{{&s->views[law.a], &s->views[law.b]}},
                {between(g->a, g->b), between(g->b, g->a)},
                NC_DEVICE_QUERY_REACH * law.max_depth, &w};
    w.regions = 0;
    if (cache.ready)
        for (int k = 0; k < cache.anchors; ++k) {
            auto r = transport(cache.anchor[k], g->a, g->b, c.invalid);
            if (r.depth > 0)
                w.result[w.regions++] = r;
        }
    w.selected = !w.regions || w.regions != cache.anchors || !cache.ready;
    cudaGraphSetConditional(fresh, w.selected);
}
template<int Stage> __global__ void cad_graph_detect_stage(DeviceScene *s, CadGraphState *g) {
    // The preceding graph node publishes this immutable descriptor. Keep
    // its geometry pointers in the uploaded address space throughout queries.
    Query &q = g->query;
    const auto grid = cooperative_groups::this_grid();
    const bool stopped = s->control->status;
    grid.sync(); // Every lane snapshots the old status before any stage can set it.
    if (stopped) return;
    auto &cache = s->cache[s->control->current_pair];
    if constexpr (Stage == 0) detect_broad(grid, q, cache);
    if constexpr (Stage == 1) detect_sort(grid, q, cache);
    if constexpr (Stage == 2) {
        __shared__ SearchBatch batch;
        auto &w = *q.work;
        const auto stamp = grid.thread_rank() ? 0 : timer_ns();
        for (int i = blockIdx.x; i < w.groups; i += gridDim.x) {
            const bool found = q.find<128>(w.group[i], cache, w.confirmed[i], &batch);
            if (!threadIdx.x) w.valid[i] = found;
            __syncthreads();
        }
        grid.sync();
        if (!grid.thread_rank()) w.search_ns += timer_ns() - stamp;
        grid.sync();
    }
    if constexpr (Stage == 3) detect_group(grid, q, cache);
}
__global__ void cad_graph_integrate(DeviceScene *s, CadGraphState *g) {
    // The preceding graph node publishes this immutable descriptor. Keep
    // its geometry pointers in the uploaded address space throughout queries.
    Query &q = g->query;
    const auto grid = cooperative_groups::this_grid();
    const bool stopped = s->control->status;
    grid.sync();
    if (stopped) return;
    const auto stamp = grid.thread_rank() ? 0 : timer_ns();
    integrate_regions(grid, q);
    if (!grid.thread_rank())
        s->workspace->integral_ns += timer_ns() - stamp;
}
__global__ void cad_graph_publish(DeviceScene *s, CadGraphState *g) {
    if (s->control->status)
        return;
    auto &w = *s->workspace;
    const int pair = s->control->current_pair;
    auto &cache = s->cache[pair];
    if (w.audit)
        audit_regions(*s, cad_graph_trial(s, g).state, s->pairs[pair], g->a, g->b, pair, g->initial);
    if (*w.status)
        return;
    for (int k = 0; k < w.regions; ++k) {
        auto &r = w.result[k];
        r.p = world(g->a, scale(r.p, NC_DEVICE_GEOMETRY_SCALE));
        r.n = rotate(g->a, r.n);
        r.arm = rotate(g->a, scale(r.arm, NC_DEVICE_GEOMETRY_SCALE));
        r.depth *= NC_DEVICE_GEOMETRY_SCALE;
        if (!cache.ready) {
            V half = scale(r.n, .5 * r.depth);
            cache.anchor[k] = {local(g->a, plus(r.p, half)), local(g->b, sub(r.p, half)),
                               rotate(g->a, r.n, -1), rotate(g->b, r.n, -1),
                               rotate(g->a, r.arm, -1), rotate(g->b, r.arm, -1)};
        }
    }
    if (!cache.ready) {
        cache.anchors = w.regions;
        cache.ready = 1;
    }
}
__global__ void cad_graph_accumulate(DeviceScene *s, CadGraphState *g, int pair) {
    if (!s->control->status)
        accumulate_pair(*s, cad_graph_trial(s, g), pair);
}
__global__ void cad_graph_linearize(DeviceScene *s, CadGraphState *g) {
    if (!s->control->status)
        linearize_forces(*s, cad_graph_trial(s, g), g->dt, g->initial);
}
__global__ void cad_graph_initial_finish(DeviceScene *s, CadGraphState *g) {
    auto &c = *s->control;
    if (c.invalid && !c.status)
        c.status = 8;
    if (!c.status) {
        copy_state(c.state, c.working.state, s->n);
        for (int i = 0; i < s->n; ++i)
            c.state.a[i] = c.state.force[i] / s->coordinates[i].mass;
        save_frame(*s, 0);
    }
    g->initial = 0;
    cudaGraphSetConditional(g->steps, !c.status && g->end > 0);
}
__global__ void cad_graph_prepare_step(DeviceScene *s, CadGraphState *g) {
    auto &c = *s->control;
    if (c.status) {
        cudaGraphSetConditional(g->newton, 0);
        return;
    }
    g->dt = ldexp(g->base_dt, -g->level);
    g->iter = g->evaluated = 0;
    g->fraction = 0;
    c.last_step = g->step;
    for (int j = 0; j < s->np; ++j)
        s->cache[j].ready = 0;
    auto &trial = c.working;
    trial.state.time = c.state.time + g->dt;
    for (int i = 0; i < s->n; ++i) {
        c.qpred[i] = c.state.q[i] + g->dt * c.state.v[i] - .5 * g->dt * g->dt * c.state.a[i];
        c.vpred[i] = c.state.v[i] - .5 * g->dt * c.state.a[i];
        trial.state.a[i] = c.state.a[i];
        trial.state.q[i] = c.qpred[i] + g->dt * g->dt * trial.state.a[i];
        trial.state.v[i] = c.vpred[i] + 1.5 * g->dt * trial.state.a[i];
    }
    cudaGraphSetConditional(g->newton, 1);
}
__global__ void cad_graph_newton(DeviceScene *s, CadGraphState *g) {
    auto &c = *s->control;
    if (c.status) {
        cudaGraphSetConditional(g->newton, 0);
        return;
    }
    c.dispatch = correct_step(*s, g->dt, g->iter, g->fraction, g->evaluated);
    g->evaluated = 1;
    cudaGraphSetConditional(g->newton, !c.status && c.dispatch == refresh_contact);
}
__global__ void cad_graph_finish_step(DeviceScene *s, CadGraphState *g) {
    auto &c = *s->control;
    if (!c.status) {
        if (c.dispatch == reject_step) {
            ++g->newton_rejections;
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
    cudaGraphSetConditional(g->steps, !c.status && g->step < g->end);
}

// Small host-side graph builder; every loop condition is written on device.
struct CadGraphBuilder {
    cudaGraph_t graph{}, root{};
    cudaGraphNode_t last{};
    cudaGraphConditionalHandle condition() const {
        cudaGraphConditionalHandle handle{};
        check(cudaGraphConditionalHandleCreate(&handle, root, 0, 0));
        return handle;
    }
    template<class F, class... A> void kernel(F function, int grid, int block, bool cooperative, A... values) {
        void *args[] = {static_cast<void *>(&values)...};
        cudaKernelNodeParams p{};
        p.func = reinterpret_cast<void *>(function);
        p.gridDim = dim3(grid);
        p.blockDim = dim3(block);
        p.kernelParams = args;
        cudaGraphNode_t node{};
        check(cudaGraphAddKernelNode(&node, graph, last ? &last : nullptr, last ? 1 : 0, &p));
        if (cooperative) {
            cudaKernelNodeAttrValue value{};
            value.cooperative = 1;
            check(cudaGraphKernelNodeSetAttribute(node, cudaKernelNodeAttributeCooperative, &value));
        }
        last = node;
    }
    CadGraphBuilder conditional(cudaGraphConditionalHandle handle, bool repeat) {
        cudaGraphNodeParams p{};
        p.type = cudaGraphNodeTypeConditional;
        p.conditional.handle = handle;
        p.conditional.type = repeat ? cudaGraphCondTypeWhile : cudaGraphCondTypeIf;
        p.conditional.size = 1;
        cudaGraphNode_t node{};
        check(cudaGraphAddNode(&node, graph, last ? &last : nullptr, last ? 1 : 0, &p));
        last = node;
        return {p.conditional.phGraph_out[0], root};
    }
};
void build_cad_graph(StepGraph &schedule, DeviceScene *s, CadGraphState *g, int steps, double dt,
                     int pairs, int sms) {
    check(cudaGraphCreate(&schedule.graph, 0));
    CadGraphBuilder root{schedule.graph, schedule.graph};
    auto stepping = root.condition(), newton = root.condition();
    auto grid_for = [&](auto function, int block, int per_sm) {
        int occupancy = 0;
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, function, block, 0));
        if (!occupancy) throw std::runtime_error("CAD graph geometry launch does not fit device");
        return sms * std::min(per_sm, occupancy);
    };
    const int broad_grid = grid_for(cad_graph_detect_stage<0>, 128, 2);
    const int search_grid = grid_for(cad_graph_detect_stage<2>, 128, 1);
    const int group_grid = std::min(16, grid_for(cad_graph_detect_stage<3>, 64, 1));
    const int integral_grid = grid_for(cad_graph_integrate, 64, 1);
    grid_for(cad_graph_detect_stage<1>, 256, 1);
    auto forces = [&](CadGraphBuilder &graph) {
        graph.kernel(cad_graph_force_init, 1, 1, false, s, g);
        for (int pair = 0; pair < pairs; ++pair) {
            auto fresh = graph.condition();
            graph.kernel(cad_graph_pair, 1, 1, false, s, g, pair, fresh);
            auto geometry = graph.conditional(fresh, false);
            geometry.kernel(cad_graph_detect_stage<0>, broad_grid, 128, true, s, g);
            geometry.kernel(cad_graph_detect_stage<1>, 1, 256, true, s, g);
            geometry.kernel(cad_graph_detect_stage<2>, search_grid, 128, true, s, g);
            geometry.kernel(cad_graph_detect_stage<3>, group_grid, 64, true, s, g);
            geometry.kernel(cad_graph_integrate, integral_grid, 64, true, s, g);
            geometry.kernel(cad_graph_publish, 1, 1, false, s, g);
            graph.kernel(cad_graph_accumulate, 1, 1, false, s, g, pair);
        }
        graph.kernel(cad_graph_linearize, 1, 1, false, s, g);
    };
    root.kernel(cad_graph_init, 1, 1, false, s, g, steps, dt, stepping, newton);
    forces(root);
    root.kernel(cad_graph_initial_finish, 1, 1, false, s, g);
    auto loop = root.conditional(stepping, true);
    loop.kernel(cad_graph_prepare_step, 1, 1, false, s, g);
    auto iteration = loop.conditional(newton, true);
    forces(iteration);
    iteration.kernel(cad_graph_newton, 1, 1, false, s, g);
    loop.kernel(cad_graph_finish_step, 1, 1, false, s, g);
}
