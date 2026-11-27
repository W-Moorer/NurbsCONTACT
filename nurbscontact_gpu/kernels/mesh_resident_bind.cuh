extern "C" __global__ void mesh_layout(int *out) {
    out[0] = sizeof(BodyView);
    out[1] = sizeof(Workspace);
    out[2] = sizeof(DeviceScene);
    out[3] = sizeof(StepControl);
    out[4] = sizeof(PairScratch);
    out[5] = sizeof(Anchor);
    out[6] = sizeof(Region);
    out[7] = sizeof(AxisCoordinate);
    out[8] = sizeof(JointBody);
    out[9] = sizeof(ContactLaw);
}
extern "C" __global__ void mesh_plane_sides(const V *triangles, const V *points, int *signs, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        signs[i] = mesh_plane_side(triangles + 3 * i, points[i]);
}
extern "C" __global__ void
bind_mesh_scene(DeviceScene *s, const AxisCoordinate *coords, const JointBody *bodies,
                const ContactLaw *pairs, const LoadPoint *loads, const BodyView *views, int n, int nb, int np,
                PairScratch *cache, Workspace *w, StepControl *control, double *observed, double *frames,
                U64 *finished, const mesh_detect::MeshPairView *mesh_pairs, int *parents, int *valid,
                U64 *patch_hash, U64 *feature_hash, U64 *patches, U64 *features, int capacity,
                int patch_capacity, int feature_capacity, Region *result, Region *unsorted, Anchor *anchors,
                int *visited, int *queue, int vertex_capacity, Region *contributions, V *placed_vertices,
                V *placed_normals) {
    *s = {coords, bodies, pairs, loads, views, n, nb, np, cache, w, control, observed, frames, finished};
    w->pairs = mesh_pairs;
    w->status = &control->status;
    w->parents = parents;
    w->region_valid = valid;
    w->patch_hash = patch_hash;
    w->feature_hash = feature_hash;
    w->patches = patches;
    w->features = features;
    w->capacity = capacity;
    w->patch_capacity = patch_capacity;
    w->feature_capacity = feature_capacity;
    w->result = result;
    w->unsorted = unsorted;
    w->visited = visited;
    w->queue = queue;
    w->vertex_capacity = vertex_capacity;
    w->contributions = contributions;
    w->placed_vertices = placed_vertices;
    w->placed_normals = placed_normals;
    for (int i = 0; i < np; ++i)
        cache[i] = {0, 0, anchors + i * capacity};
}
extern "C" __global__ void advance_mesh(DeviceScene *s, int begin, int end, double dt) {
    advance_steps(cooperative_groups::this_grid(), *s, begin, end, dt);
}
extern "C" __global__ void reset_mesh(DeviceScene *s) {
    *s->control = {};
    auto &w = *s->workspace;
    w.queries = w.evaluations = w.subdivisions = w.broad_ns = w.narrow_ns = w.group_ns = w.support_ns =
        w.integral_ns = 0;
    w.maximum_residual = 0;
    for (int i = 0; i < s->np; ++i)
        s->cache[i].ready = 0;
}
extern "C" __global__ void mesh_status(const DeviceScene *s, double *out) {
    auto &w = *s->workspace;
    auto &c = *s->control;
    out[0] = c.status;
    out[1] = c.state.time;
    out[2] = w.queries;
    out[3] = w.evaluations;
    out[4] = w.subdivisions;
    out[5] = w.broad_ns * 1e-9;
    out[6] = w.narrow_ns * 1e-9;
    out[7] = w.group_ns * 1e-9;
    out[8] = w.support_ns * 1e-9;
    out[9] = w.maximum_residual;
}
extern "C" __global__ void probe_mesh(DeviceScene *s, const double *poses, double *out, int *counts) {
    auto grid = cooperative_groups::this_grid();
    auto &w = *s->workspace;
    for (int pair = 0; pair < s->np; ++pair) {
        auto law = s->pairs[pair];
        Query q{};
        q.b[0] = s->views[law.a];
        q.b[1] = s->views[law.b];
        q.reach = 4 * law.max_depth;
        q.work = &w;
        const double *p = poses + 12 * pair;
        q.rel[0] = {
            {make_double3(p[0], p[4], p[8]), make_double3(p[1], p[5], p[9]), make_double3(p[2], p[6], p[10])},
            make_double3(p[3], p[7], p[11])};
        q.rel[1] = {
            {make_double3(p[0], p[1], p[2]), make_double3(p[4], p[5], p[6]), make_double3(p[8], p[9], p[10])},
            inverse_direction(q.rel[0], scale(q.rel[0].offset, -1))};
        if (!grid.thread_rank())
            w.pair = pair;
        grid.sync();
        detect(grid, q, s->cache[pair]);
        if (*w.status)
            return;
        if (!grid.thread_rank()) {
            counts[pair] = w.regions;
            for (int i = 0; i < w.regions; ++i) {
                auto r = w.result[i];
                double *row = out + (pair * w.capacity + i) * 7;
                row[0] = r.p.x;
                row[1] = r.p.y;
                row[2] = r.p.z;
                row[3] = r.n.x;
                row[4] = r.n.y;
                row[5] = r.n.z;
                row[6] = r.depth;
            }
        }
        grid.sync();
    }
}
extern "C" __global__ void probe_mesh_world(DeviceScene *s, const MatPose *poses, double *out, int *counts) {
    auto grid = cooperative_groups::this_grid();
    auto &w = *s->workspace;
    for (int pair = 0; pair < s->np; ++pair) {
        auto law = s->pairs[pair];
        auto a = poses[law.a], b = poses[law.b];
        Query q{{s->views[law.a], s->views[law.b]}, {between(a, b), between(b, a)}, 4 * law.max_depth, &w};
        if (!grid.thread_rank())
            w.pair = pair;
        grid.sync();
        detect(grid, q, s->cache[pair]);
        if (*w.status)
            return;
        if (!grid.thread_rank()) {
            counts[pair] = w.regions;
            for (int i = 0; i < w.regions; ++i) {
                auto r = w.result[i];
                r.p = world(a, r.p);
                r.n = rotate(a, r.n);
                double *row = out + (pair * w.capacity + i) * 7;
                row[0] = r.p.x;
                row[1] = r.p.y;
                row[2] = r.p.z;
                row[3] = r.n.x;
                row[4] = r.n.y;
                row[5] = r.n.z;
                row[6] = r.depth;
            }
        }
        grid.sync();
    }
}
