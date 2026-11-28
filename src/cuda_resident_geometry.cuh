// Runtime workspace for one trajectory. Work is indexed by CAD topology, independent of tooth counts.
struct TileJet {
    Jet jet;
    bool defined;
};
__global__ void prepare_tile_jets(View g, const Tile *tiles, int count, TileJet *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const auto &t = tiles[i];
        out[i].defined = face_jet(g, t.face, t.u, t.v, out[i].jet);
    }
}
struct RuleEnvelope { Bounds points, normals; int valid; };
constexpr int rule_envelopes = 9 + 36 + 144 + 576;
struct BodyView {
    View g;
    const Tile *tiles;
    const Node *sites;
    const int *target_ids;
    const TileJet *tile_jets;
    int ns, nt;
    const WeightedSite *refined_rules;
    const RuleEnvelope *envelopes;
    const unsigned char *neighbors;
    int face_count;
};
struct MatPose {
    V origin, axis;
    double angle;
};
struct Site {
    V p, n;
    double u, v, area;
};
// Fixed material quadrature on immutable rigid-body charts. Match the online
// expressions exactly; this is an evaluation cache, not a fitted surface.
constexpr int refined_rule_sites = 48 * 48 + 96 * 96;
__global__ void prepare_refined_rules(View g, WeightedSite *out, int count, int *status) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int face = index / refined_rule_sites, slot = index % refined_rule_sites;
    const int order = slot < 48 * 48 ? 48 : 96, ij = slot < 48 * 48 ? slot : slot - 48 * 48;
    const auto r = g.charts[face].domain;
    const double du = (r.U - r.u) / order, dv = (r.Vv - r.v) / order;
    const double u = r.u + (ij / order + .5) * du, v = r.v + (ij % order + .5) * dv;
    WeightedSite value{};
    const int material = face_contains(g, g.charts[face], u, v);
    if (material < 0) atomicCAS(status, 0, 4);
    else if (material) {
        Jet j;
        if (!face_jet(g, face, u, v, j)) atomicCAS(status, 0, 5);
        else {
            const V n = cross(j.u, j.v);
            const double area = norm(n);
            if (area > 0) value = {j.p, scale(n, g.charts[face].orientation / area), area * du * dv};
        }
    }
    out[index] = value;
}
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
struct SearchAttempt {
    Seed seed{};
    Witness witness{};
    QueryFault fault{};
    double depth;
    bool hit{}, skip{}, disjoint{}, projected{};
};
struct SearchProgress { int next, done; };
struct SavedSeed {
    Faces ids;
    Seed seed;
    Hit exits[2];
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
struct Blocker {
    int side;
    Hit root;
};
struct BlockerCache {
    Faces left, right;
    int count;
    Blocker items[4];
};
struct ConnectionTrial {
    Connection path;
    QueryFault fault;
    int left, right;
    bool joined;
    BlockerCache blockers;
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
    Pose paths_pose;
    BlockerCache *blockers, *next_blockers;
    int blocker_count;
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
// Only block zero uses this scratch space. Allocate one copy per trajectory
// and reuse it for every region and trial in the device graph.
struct OrderedScratch {
    int ids[32 * 32];
    double terms[32][10];
};
__device__ unsigned long long timer_ns() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}
struct Workspace {
    unsigned optimizations;
    SearchProgress *search_progress;
    unsigned long long projection_rounds, avoided_projections, unused_projections, projection_culls;
    unsigned long long topology_checks, topology_hits, warm_attempts, warm_hits, warm_fallbacks;
    int *unique_proposals, *link_order;
    unsigned char *rule_live;
    unsigned long long parameter_hint_calls, parameter_hint_hits, projection_attempts, rejected_rule_blocks, tested_rule_blocks, local_links, remote_links;
    unsigned int count;
    int groups, found, regions, selected, active_faces, capacity, limit;
    Suggestion *proposals, *sorted_proposals;
    SearchAttempt *seed_attempts, *attempts;
    Group *group;
    Witness *witnesses, *confirmed;
    int *valid, *labels, *sides, *faces;
    SampleDepth *columns;
    OrderedScratch *ordered;
    unsigned *active_columns;
    Range *windows;
    ConnectionTrial *bridges;
    Connection *serial_links;
    unsigned long long checked_links;
    int column_capacity;
    Region *result;
    Region *reference;
    RegionReduction reduction;
    RegionAudit *audit;
    unsigned long long reduction_ns, reductions;
    int *status;
    unsigned long long queries, subdivisions, evaluations;
    unsigned long long proposal_total, group_total, confirmed_total, seed_hits;
    unsigned int maximum_proposals, maximum_groups;
    bool reuse_blockers;
    bool transport_paths;
    bool profile_links;
    unsigned long long link_ns[4], link_calls[4];
    unsigned long long blocker_checks, blocker_hits;
    unsigned long long warp_segments, scalar_segments;
    unsigned long long separated_groups;
    unsigned long long path_checks, path_hits;
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
// Per-block reduction state for ordered parallel face proposals.
// The original 32-lane search uses no extra shared state.
struct SearchBatch {
    int stops[4], first;
    double depths[4], best;
    Seed seeds[4], seed;
};
struct BodyPair {
    const BodyView *parts[2];
    __device__ const BodyView &operator[](int side) const { return *parts[side]; }
};
struct Query {
    // Immutable uploaded geometry is shared, rather than copied into every
    // speculative per-lane query and recursive connection test.
    BodyPair b;
    Pose rel[2];
    double reach;
    Workspace *work;
    QueryFault *speculative;
    BlockerCache *blockers;
    bool cold_projection{};
    __device__ static Bound projection(Bounds box,V axis) {
        Bound value{0,0};
        for (int k=0;k<3;++k) {
            value.lo += axis.x[k]*(axis.x[k]>=0?box.lo.x[k]:box.hi.x[k]);
            value.hi += axis.x[k]*(axis.x[k]>=0?box.hi.x[k]:box.lo.x[k]);
        }
        return value;
    }
    __device__ bool separated_on(Faces ids,V axis) const {
        const auto &a=b[0].g.charts[ids.a],&other=b[1].g.charts[ids.b];
        const V target_axis=direction(rel[0],axis);
        const auto na=projection(a.normal_box,axis),nb=projection(other.normal_box,target_axis);
        const double roundoff=1e-10*(1+fabs(na.lo)+fabs(na.hi)+fabs(nb.lo)+fabs(nb.hi));
        if (!(na.lo>roundoff && nb.hi<-roundoff)) return false;
        V aa{},bb{};
        for (int k=0;k<3;++k) {
            aa.x[k]=dot(a.support_axes[k],axis);
            bb.x[k]=dot(other.support_axes[k],target_axis);
        }
        const double high=projection(a.support_box,aa).hi;
        const double low=projection(other.support_box,bb).lo+dot(rel[1].offset,axis);
        return low-high>1e-6+1e-10*(1+fabs(low)+fabs(high));
    }
    __device__ bool separated(Faces ids) const {
        const auto &a=b[0].g.charts[ids.a],&other=b[1].g.charts[ids.b];
        if (!a.support_ready || !other.support_ready) return false;
        return separated_on(ids,scale(a.support_axes[2],a.orientation)) ||
            separated_on(ids,direction(rel[1],scale(other.support_axes[2],-other.orientation)));
    }
    __device__ V path_point(const PairScratch &cache,V old) const {
        if (!work->transport_paths) return old;
        // Advect the old interior point with the mean of both rigid motions.
        // This is only a proposed path: every segment is rechecked on CAD.
        const V in_target=point(cache.paths_pose,old);
        return scale(plus(old,point(rel[1],in_target)),.5);
    }
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
    __device__ bool sample(Faces ids, Seed seed, Site site, double &best, Witness &out, const Hit *hint = nullptr) const {
        const int side = seed.side, id = ids[1 - side];
        const auto &g = b[1 - side].g;
        const auto &tile = b[1 - side].tiles[seed.tile];
        // The legacy mode preserves its canonical iteration exactly. The
        // experimental warm mode also fixes the accepted material normal ray;
        // its same-pose audit checks cold witness/partition compatibility.
        double u = tile.u, v = tile.v;
        const bool warm=hint && (work->optimizations&64) && !cold_projection;
        if (warm) { u=hint->u; v=hint->v; }
        int cached_u=-1,cached_v=-1;
        const auto &face=g.charts[id];
        if(hint && face.kind==0 && face.nu>=2 && face.nv>=2) {
            cached_u=knot_slot(g.knots+face.knots_u,face.nu,hint->u);
            cached_v=knot_slot(g.knots+face.knots_v,face.nv,hint->v);
            atomicAdd(&work->parameter_hint_calls,1ULL);
        }
        V p = point(rel[side], site.p), d = direction(rel[side], scale(site.n, -1));
        for (int iter = 0; iter < 8; ++iter) {
            Jet j;
            // Cold projections retain the precomputed material tile centre.
            if (!iter && !warm) {
                const auto &initial = b[1 - side].tile_jets[seed.tile];
                if (!initial.defined) {
                    fail(5, id);
                    return false;
                }
                j = initial.jet;
            } else {
                bool cached=false;
                if(cached_u>=0) {
                    const auto ku=g.knots+face.knots_u, kv=g.knots+face.knots_v;
                    // Adjacent knot intervals are cheap to check; all other
                    // locations retain the original binary grid lookup.
                    if(u<ku[cached_u] && cached_u>0) --cached_u;
                    else if(u>=ku[cached_u+1] && cached_u<face.nu-2) ++cached_u;
                    if(v<kv[cached_v] && cached_v>0) --cached_v;
                    else if(v>=kv[cached_v+1] && cached_v<face.nv-2) ++cached_v;
                    if(u>=ku[cached_u] && u<ku[cached_u+1] && v>=kv[cached_v] && v<kv[cached_v+1]) {
                        const int span=g.grid[face.grid+cached_u*(face.nv-1)+cached_v];
                        if(span>=0 && g.patches[span].cell>=0) {
                            j=at(g.patches[span],u,v); cached=true;
                            atomicAdd(&work->parameter_hint_hits,1ULL);
                        }
                    }
                }
                if(!cached && !face_jet(g,id,u,v,j)) { fail(5,id); return false; }
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
                if (work->optimizations&64) {
                    // Fix the accepted witness to the material normal ray.
                    // Its origin/direction do not depend on Newton's initial UV.
                    // The complete ray traversal checks earlier/occluding faces.
                    const auto exact=ray(1-side,p,d,reach);
                    if (exact.face!=id || exact.t<=2e-7) return false;
                    t=exact.t;
                }
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
    template<int Width>
    __device__ bool collect_attempt(SearchBatch *batch, int lane, Seed trial, double depth, bool hit, const Witness &candidate,
                                    QueryFault fault, Seed &seed, double &best, Witness &out) const {
        constexpr unsigned mask = 0xffffffffu;
        unsigned stop = __ballot_sync(mask, hit || fault.code);
        if constexpr (Width > 32) {
            const int warp = lane / 32, local = lane % 32;
            double maximum = depth;
            for (int distance = 16; distance; distance /= 2)
                maximum = fmax(maximum, __shfl_down_sync(mask, maximum, distance));
            maximum = __shfl_sync(mask, maximum, 0);
            const unsigned winners = __ballot_sync(mask, depth == maximum);
            const int winner = __ffs(winners) - 1;
            const Seed selected{__shfl_sync(mask, trial.side, winner), __shfl_sync(mask, trial.tile, winner),
                                __shfl_sync(mask, trial.u, winner), __shfl_sync(mask, trial.v, winner)};
            if (!local) {
                batch->stops[warp] = stop ? 32 * warp + __ffs(stop) - 1 : Width;
                batch->depths[warp] = maximum;
                batch->seeds[warp] = selected;
            }
            __syncthreads();
            if (!lane) {
                batch->first = Width;
                batch->best = best;
                batch->seed = seed;
                for (int w = 0; w < Width / 32; ++w) {
                    batch->first = min(batch->first, batch->stops[w]);
                    if (batch->depths[w] > batch->best) {
                        batch->best = batch->depths[w];
                        batch->seed = batch->seeds[w];
                    }
                }
            }
            __syncthreads();
            const bool stopped = batch->first < Width;
            if (lane == batch->first) {
                if (fault.code) record(fault);
                else out = candidate;
            }
            if (!stopped) {
                best = batch->best;
                seed = batch->seed;
            }
            __syncthreads();
            return stopped;
        }
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
    __device__ SearchAttempt prior_attempt(Group group, const PairScratch &cache) const {
        SearchAttempt a{};
        a.depth = -HUGE_VAL;
        a.disjoint = work->reuse_blockers && separated(group.ids);
        a.skip = a.disjoint && !work->serial_links;
        if (a.skip || group.prior < 0) return a;
        a.seed = cache.previous[group.prior].seed;
        Query q = *this;
        q.speculative = &a.fault;
        Site site;
        if (!q.node(a.seed.side, group.ids[a.seed.side], a.seed.u, a.seed.v, site)) return a;
        const auto &saved=cache.previous[group.prior];
        const auto *hint=(work->optimizations&1)?&saved.exits[1-a.seed.side]:nullptr;
        const bool warm=hint && (work->optimizations&64) && !q.cold_projection;
        if (warm) atomicAdd(&work->warm_attempts,1ULL);
        a.hit=q.sample(group.ids,a.seed,site,a.depth,a.witness,hint);
        if (warm && a.hit && !a.fault.code) atomicAdd(&work->warm_hits,1ULL);
        if (warm && (!a.hit || a.fault.code)) {
            // A failed speculative initial guess must not suppress discovery
            // or turn its private fault into a global solver failure.
            atomicAdd(&work->warm_fallbacks,1ULL);
            a.fault={}; a.depth=-HUGE_VAL; a.witness={};
            q.cold_projection=true;
            a.hit=q.sample(group.ids,a.seed,site,a.depth,a.witness,hint);
        }
        return a;
    }
    __device__ SearchAttempt proposal_attempt(Group group, int k) const {
        SearchAttempt a{};
        a.depth = -HUGE_VAL;
        const auto &c = work->proposals[k];
        if ((work->optimizations & 1) && !work->unique_proposals[k]) return a;
        for (int j = group.begin; !(work->optimizations & 1) && j < k; ++j) {
            const auto &previous = work->proposals[j];
            if (previous.side == c.side && previous.source == c.source) return a;
        }
        const auto &t = b[c.side].tiles[c.source];
        a.seed = {c.side, c.tile, t.u, t.v};
        Site site{t.point, t.normal, t.u, t.v, 0};
        if ((work->optimizations&16) && !cold_projection) {
            const auto &g=b[1-c.side].g;
            const auto &face=g.charts[group.ids[1-c.side]];
            const auto direction0=direction(rel[c.side],scale(site.n,-1));
            const auto origin=sub(point(rel[c.side],site.p),scale(direction0,reach));
            const Ray segment{origin,direction0};
            // Enclose BOTH signs of the projection depth. Negative depths
            // drive the existing local refinement and cannot be discarded.
            bool possible=intersects(face.box,segment,2*reach,4*face.tol+1e-6);
            if(possible && face.support_ready) {
                const auto oriented=Ray{basis(face.support_axes,origin),basis(face.support_axes,direction0)};
                possible=intersects(face.support_box,oriented,2*reach,4*face.tol+1e-6);
            }
            if(possible && face.first>=0 && (work->optimizations&128)) {
                possible=false;
                const auto local=slab_ray({basis(face.frame,origin),basis(face.frame,direction0)});
                for(int node=face.first;node<face.end;) {
                    const auto &span=g.spans[node];
                    if(!intersects_slabs(span.box,local,2*reach,4*face.tol+1e-6)) { node=span.next; continue; }
                    if(span.leaf>=0) { possible=true; break; }
                    ++node;
                }
            }
            if(!possible) { atomicAdd(&work->projection_culls,1ULL); return a; }
        }
        a.projected=true;
        if (work->optimizations & 1) atomicAdd(&work->projection_attempts, 1ULL);
        Query q = *this;
        q.speculative = &a.fault;
        a.hit = q.sample(group.ids, a.seed, site, a.depth, a.witness);
        return a;
    }
    template<int Width = 32, bool Queued = false>
    __device__ __noinline__ bool find(Group group, const PairScratch &cache, Witness &out,
                                     SearchBatch *batch = nullptr, int group_index = -1) const {
        static_assert(Width == 32 || Width == 128);
        const int lane = threadIdx.x % Width;
        // Positive separation along an axis with opposing normal cones makes
        // a positive-depth projection impossible from either chart. Audit mode
        // still runs the original search to check every rejected face pair.
        bool disjoint;
        if constexpr (Queued) disjoint = work->seed_attempts[group_index].disjoint;
        else disjoint = work->reuse_blockers && separated(group.ids);
        if (disjoint && !lane) atomicAdd(&work->separated_groups,1ULL);
        if (disjoint && !work->serial_links) return false;
        auto accepted = [&]() {
            if (disjoint && !lane) fail(12);
            return true;
        };
        double best = -HUGE_VAL;
        Seed seed{};
        Witness candidate{};
        if (group.prior >= 0) {
            seed = cache.previous[group.prior].seed;
            int hit = 0;
            if (!lane) {
                if constexpr (Queued) {
                    const auto &a = work->seed_attempts[group_index];
                    hit = a.hit; best = a.depth; out = a.witness;
                    if (a.fault.code) record(a.fault);
                } else {
                    const auto a = prior_attempt(group, cache);
                    hit = a.hit; best = a.depth; out = a.witness;
                    if (a.fault.code) record(a.fault);
                }
            }
            if constexpr (Width == 32) {
                // A failed prior attempt also copies its witness. Register
                // shuffles do not order those shared-memory stores against a
                // different lane writing the next accepted proposal.
                __syncwarp();
                if (__shfl_sync(0xffffffffu, hit, 0)) {
                    if (!lane) atomicAdd(&work->seed_hits, 1ULL);
                    return accepted();
                }
                best = __shfl_sync(0xffffffffu, best, 0);
            } else {
                if (!lane) { batch->first = hit; batch->best = best; }
                __syncthreads();
                if (batch->first) {
                    if (!lane) atomicAdd(&work->seed_hits, 1ULL);
                    return accepted();
                }
                best = batch->best;
                __syncthreads();
            }
            if constexpr (Queued) if (work->seed_attempts[group_index].fault.code) return false;
        }
        Query q = *this;
        QueryFault fault;
        q.speculative = &fault;
        for (int start = group.begin; start < group.end; start += Width) {
            int k = start + lane;
            Seed trial{};
            double depth = -HUGE_VAL;
            bool hit = false;
            fault = {};
            if (k < group.end) {
                if constexpr (Queued) {
                    if ((work->optimizations&16) && k>=work->search_progress[group_index].next) {
                        // Later proposals were never needed by this ordered search.
                    } else {
                    const auto &a = work->attempts[k];
                    trial = a.seed; depth = a.depth; hit = a.hit;
                    candidate = a.witness; fault = a.fault;
                    }
                } else {
                    const auto a=proposal_attempt(group,k);
                    trial=a.seed; depth=a.depth; hit=a.hit; candidate=a.witness; fault=a.fault;
                }
            }
            if (collect_attempt<Width>(batch, lane, trial, depth, hit, candidate, fault, seed, best, out))
                return accepted();
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
            if (collect_attempt<Width>(batch, lane, trial, depth, hit, candidate, fault, seed, best, out))
                return accepted();
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
    __device__ bool clear_segment(int side, V a, V b0) const {
        V x = side ? point(rel[0], a) : a, y = side ? point(rel[0], b0) : b0;
        V d = sub(y, x);
        double len = norm(d);
        return len < 1e-8 || ray(side, x, scale(d, 1 / len), len, true).face < 0;
    }
    __device__ bool straight(V a, V b0) const {
        if (!blockers) return clear_segment(0, a, b0) && clear_segment(1, a, b0);
        const V x[2]{a, point(rel[0], a)}, y[2]{b0, point(rel[0], b0)};
        for (int k = 0; k < blockers->count; ++k) {
            const auto &h = blockers->items[k];
            atomicAdd(&work->blocker_checks, 1ULL);
            if (crosses(h.side, x[h.side], y[h.side], h.root)) {
                atomicAdd(&work->blocker_hits, 1ULL);
                return false;
            }
        }
        for (int side = 0; side < 2; ++side) {
            const V d = sub(y[side], x[side]);
            const double length = norm(d);
            if (length < 1e-8) continue;
            const Hit h = ray(side, x[side], scale(d, 1 / length), length, true);
            if (h.face < 0) continue;
            int at = 0;
            while (at < blockers->count &&
                   !(blockers->items[at].side == side && blockers->items[at].root.face == h.face)) ++at;
            if (at == blockers->count && blockers->count < 4) ++blockers->count;
            at = min(at, 3);
            for (; at > 0; --at) blockers->items[at] = blockers->items[at - 1];
            blockers->items[0] = {side, h};
            return false;
        }
        return true;
    }
    __device__ bool straight_warp(V a, V b0) const {
        const int lane=threadIdx.x%32;
        const V x[2]{a,point(rel[0],a)},y[2]{b0,point(rel[0],b0)};
        bool blocked=false;
        if (!lane && blockers)
            for (int k=0;k<blockers->count;++k) {
                const auto &h=blockers->items[k];
                atomicAdd(&work->blocker_checks,1ULL);
                if (crosses(h.side,x[h.side],y[h.side],h.root)) {
                    atomicAdd(&work->blocker_hits,1ULL);
                    blocked=true;break;
                }
            }
        if (__any_sync(0xffffffffu,blocked)) return false;
        for (int side=0;side<2;++side) {
            const V d=sub(y[side],x[side]);
            const double length=norm(d);
            if (length<1e-8) continue;
            if (!lane) {
                if (b[side].g.ray_grid && length<=b[side].g.ray_grid->reach)
                    atomicAdd(&work->scalar_segments,1ULL);
                else atomicAdd(&work->warp_segments,1ULL);
            }
            const Hit h=crossing_warp(b[side].g,{x[side],scale(d,1/length)},length);
            if (h.face==-2 && !lane)
                record({30+static_cast<int>(h.v),static_cast<int>(h.u),side,1,
                        {x[side],scale(d,1/length)},length});
            if (h.face<0) continue;
            if (!lane && blockers) {
                int at=0;
                while (at<blockers->count && !(blockers->items[at].side==side &&
                       blockers->items[at].root.face==h.face)) ++at;
                if (at==blockers->count && blockers->count<4) ++blockers->count;
                at=min(at,3);
                for (;at>0;--at) blockers->items[at]=blockers->items[at-1];
                blockers->items[0]={side,h};
            }
            __syncwarp();
            return false;
        }
        return true;
    }
    __device__ __noinline__ bool connected(const Witness &a, const Witness &b0, int depth,
                                           Connection &path, bool try_direct = true) const {
        auto c = point(rel[0], a.p), d = point(rel[0], b0.p);
        if (try_direct && !crosses(0, a.p, b0.p, a.exits[0]) && !crosses(0, b0.p, a.p, b0.exits[0]) &&
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
    __device__ bool topology_connection(const Witness &a,const Witness &other,Connection &path) const {
        if (!(work->optimizations&32)) return false;
        for (int side=0;side<2;++side) {
            const auto &g=b[side].g;
            if (!g.boundary_offsets || a.faces[side]==other.faces[side]) continue;
            for (int i=g.boundary_offsets[a.faces[side]];i<g.boundary_offsets[a.faces[side]+1];++i) {
                const auto &link=g.boundary_links[i];
                if (link.other!=other.faces[side]) continue;
                for (int k=link.first;k<link.end;++k) {
                    atomicAdd(&work->topology_checks,1ULL);
                    Query q=*this; QueryFault fault{}; q.speculative=&fault; q.blockers=nullptr;
                    const auto &probe=g.boundary_probes[k];
                    const auto h=q.ray(1-side,point(rel[side],probe.point),
                                       direction(rel[side],scale(probe.normal,-1)),reach);
                    if (fault.code || h.face<0) continue;
                    V mid=sub(probe.point,scale(probe.normal,.5*h.t)),n=probe.normal;
                    if (side) { mid=point(rel[1],mid); n=direction(rel[1],scale(n,-1)); }
                    Witness bridge;
                    // A source edge is only a path proposal. Both solids must
                    // contain the bridge and both complete segments at this pose.
                    if (!q.chord(mid,n,bridge) || fault.code || !q.straight(a.p,bridge.p) ||
                        !q.straight(bridge.p,other.p) || fault.code) continue;
                    path={}; path.left=a.faces; path.right=other.faces;
                    path.count=1; path.points[0]=bridge.p;
                    atomicAdd(&work->topology_hits,1ULL);
                    return true;
                }
            }
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

__device__ bool previous_connection(Query &q, const PairScratch &cache, const Witness &a, const Witness &b,
                                    Connection &path) {
    path.left = a.faces;
    path.right = b.faces;
    path.count = 0;
    for (int k = 0; k < cache.path_count; ++k) {
        const auto &old = cache.paths[k];
        if (!equal(old.left, a.faces) || !equal(old.right, b.faces))
            continue;
        if (q.work->reuse_blockers) atomicAdd(&q.work->path_checks,1ULL);
        bool valid = true;
        V last = a.p;
        for (int t = 0; t < old.count; ++t) {
            const V next=q.path_point(cache,old.points[t]);
            if (!q.straight(last, next)) {
                valid = false;
                break;
            }
            last = next;
        }
        if (valid && q.straight(last, b.p)) {
            path = old;
            for (int t=0;t<old.count;++t) path.points[t]=q.path_point(cache,old.points[t]);
            if (q.work->reuse_blockers) atomicAdd(&q.work->path_hits,1ULL);
            return true;
        }
        break;
    }
    return false;
}
__device__ bool previous_connection_warp(Query &q,const PairScratch &cache,const Witness &a,const Witness &b,
                                         Connection &path) {
    const int lane=threadIdx.x%32;
    if (!lane) {path.left=a.faces;path.right=b.faces;path.count=0;}
    __syncwarp();
    for (int k=0;k<cache.path_count;++k) {
        const auto &old=cache.paths[k];
        if (!equal(old.left,a.faces) || !equal(old.right,b.faces)) continue;
        if (!lane) atomicAdd(&q.work->path_checks,1ULL);
        V last=a.p;
        for (int t=0;t<old.count;++t) {
            const V next=q.path_point(cache,old.points[t]);
            if (!q.straight_warp(last,next)) return false;
            last=next;
        }
        if (!q.straight_warp(last,b.p)) return false;
        if (!lane) {
            path=old;
            for (int t=0;t<old.count;++t) path.points[t]=q.path_point(cache,old.points[t]);
            atomicAdd(&q.work->path_hits,1ULL);
        }
        __syncwarp();
        return true;
    }
    return false;
}
// Speculative lanes may visit predicates beyond the first blocking one. Retain only
// the first fault the original left-to-right, short-circuit evaluation would see.
__device__ bool blocked_prefix(Query &q, bool blocked, const QueryFault &fault, int lane) {
    constexpr unsigned mask = 0xffffffffu;
    const unsigned stops = __ballot_sync(mask, blocked);
    const int last = stops ? __ffs(stops) - 1 : 31;
    const unsigned faults = __ballot_sync(mask, fault.code && lane <= last);
    if (faults && lane == __ffs(faults) - 1)
        q.record(fault);
    __syncwarp();
    return stops != 0;
}
__device__ bool check_connection(Query &q, const PairScratch &cache, const Witness &a, const Witness &b,
                                 Connection &path, int lane) {
    constexpr unsigned mask = 0xffffffffu;
    const bool profile = q.work->profile_links;
    auto stamp = profile ? timer_ns() : 0;
    auto record_time = [&](int stage) {
        if (profile && !lane) {
            atomicAdd(&q.work->link_ns[stage], timer_ns() - stamp);
            atomicAdd(&q.work->link_calls[stage], 1ULL);
        }
    };
    const bool reused = q.blockers ? previous_connection_warp(q,cache,a,b,path) :
        __any_sync(mask, lane == 0 && previous_connection(q, cache, a, b, path));
    record_time(0);
    if (reused) return true;
    __syncwarp(); // Publish any fault recorded by lane zero before the parallel predicates.
    stamp = profile ? timer_ns() : 0;
    Query local = q;
    QueryFault fault{};
    local.speculative = &fault;
    bool blocked = false;
    if (lane < 4) {
        const int side = lane / 2;
        const auto &from = (lane & 1) ? b : a;
        const auto &to = (lane & 1) ? a : b;
        V start = side ? point(q.rel[0], from.p) : from.p;
        V end = side ? point(q.rel[0], to.p) : to.p;
        blocked = local.crosses(side, start, end, from.exits[side]);
    }
    const bool endpoint_blocked = blocked_prefix(q, blocked, fault, lane);
    record_time(1);
    if (!endpoint_blocked) {
        stamp = profile ? timer_ns() : 0;
        bool clear = false;
        if (q.blockers) {
            clear = q.straight_warp(a.p, b.p);
        } else {
            fault = {};
            blocked = lane < 2 && !local.clear_segment(lane, a.p, b.p);
            clear = !blocked_prefix(q, blocked, fault, lane);
        }
        record_time(2);
        if (clear) return true;
    }
    bool joined = false;
    if (!lane) {
        stamp = profile ? timer_ns() : 0;
        joined = q.topology_connection(a,b,path) || q.connected(a, b, 5, path, false);
        record_time(3);
    }
    return joined;
}
__device__ __noinline__ bool agrees_with_serial_link(Query q, const PairScratch &cache, const Witness &a,
                                                     const Witness &b, const ConnectionTrial &trial,
                                                     Connection &reference) {
    QueryFault fault{};
    q.speculative = &fault;
    // Audit the cached decision against the independent uncached traversal.
    q.blockers = nullptr;
    bool joined = previous_connection(q, cache, a, b, reference) ||
        q.topology_connection(a,b,reference) || q.connected(a, b, 5, reference);
    if (joined != trial.joined || fault.code != trial.fault.code || fault.face != trial.fault.face ||
        fault.side != trial.fault.side || fault.any != trial.fault.any)
        return false;
    if (!joined)
        return true;
    if (q.work->optimizations&32) {
        // Different valid interior paths are equivalent. Independently verify
        // every segment instead of requiring the same arbitrary path vertices.
        V last=a.p;
        for (int i=0;i<trial.path.count;++i) {
            if (!q.straight(last,trial.path.points[i]) || fault.code) return false;
            last=trial.path.points[i];
        }
        return q.straight(last,b.p) && !fault.code;
    }
    if (reference.count != trial.path.count)
        return false;
    for (int i = 0; i < reference.count; ++i)
        for (int axis = 0; axis < 3; ++axis)
            if (reference.points[i].x[axis] != trial.path.points[i].x[axis])
                return false;
    return true;
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
template <class Grid> __device__ void detect_broad(Grid grid, Query &q, PairScratch &cache) {
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
}
template <class Grid> __device__ void detect_sort(Grid grid, Query &q, PairScratch &cache,
                                                bool presorted = false, const Faces *keys = nullptr) {
    auto &w = *q.work;
    int lane = grid.thread_rank();
    unsigned long long stamp = lane ? 0 : timer_ns();
    if (!presorted) sort_proposals(grid, w.proposals, w.count);
    if (!lane) {
        int k = 0, j = 0;
        while (k < w.count || j < cache.previous_count) {
            Faces p = k < w.count ? (keys ? keys[k] : w.proposals[k].faces) : Faces{INT_MAX, INT_MAX};
            Faces h = j < cache.previous_count ? cache.previous[j].ids : Faces{INT_MAX, INT_MAX};
            Faces ids = (h.a < p.a || (h.a == p.a && h.b < p.b)) ? h : p;
            int begin = k;
            while (k < w.count && equal(keys ? keys[k] : w.proposals[k].faces, ids))
                ++k;
            if (w.groups >= w.limit) {
                q.fail(1);
                break;
            }
            w.group[w.groups++] = {ids, begin, k, equal(h, ids) ? j++ : -1};
        }
        w.proposal_total += w.count;
        w.group_total += w.groups;
        w.maximum_proposals = max(w.maximum_proposals, w.count);
        w.maximum_groups = max(w.maximum_groups, static_cast<unsigned>(w.groups));
        auto now = timer_ns();
        w.sort_ns += now - stamp;
        stamp = now;
    }
    grid.sync();
    if (*w.status)
        return;
}
template <class Grid> __device__ void detect_search(Grid grid, Query &q, PairScratch &cache) {
    auto &w = *q.work;
    int lane = grid.thread_rank(), threads = grid.size();
    unsigned long long stamp = lane ? 0 : timer_ns();
    if (w.reuse_blockers) {
        __shared__ SearchBatch batch;
        for (int i = blockIdx.x; i < w.groups; i += gridDim.x) {
            bool found = q.find<128>(w.group[i], cache, w.confirmed[i], &batch);
            if (!threadIdx.x) w.valid[i] = found;
            __syncthreads();
        }
    } else for (int i = lane / 32; i < w.groups; i += threads / 32) {
        bool found = q.find(w.group[i], cache, w.confirmed[i]);
        if (!(lane % 32))
            w.valid[i] = found;
    }
    grid.sync();
    if (*w.status)
        return;
    if (!lane)
        w.search_ns += timer_ns() - stamp;
    grid.sync();
}
template <class Grid> __device__ void detect_group(Grid grid, Query &q, PairScratch &cache) {
    auto &w = *q.work;
    int lane = grid.thread_rank(), threads = grid.size();
    unsigned long long stamp = lane ? 0 : timer_ns();
    if (!lane) {
        for (int i = 0; i < w.groups; ++i)
            if (w.valid[i]) {
                auto &v = w.confirmed[i];
                int k = w.found++;
                w.witnesses[k] = v;
                w.labels[k] = k;
                cache.previous[k] = {v.faces, v.seed, {v.exits[0], v.exits[1]}};
            }
        cache.previous_count = w.found;
        w.confirmed_total += w.found;
    }
    grid.sync();
    const int pairs = w.found * (w.found - 1) / 2;
    int saved = 0;
    int saved_blockers = 0;
    for (int first = 0; first < pairs; first += connection_batch) {
        int count = min(connection_batch, pairs - first);
        // Use separate warps for independent, branch-heavy visibility proofs.
        const int worker = w.reuse_blockers ? blockIdx.x : lane / 32;
        const int workers = w.reuse_blockers ? gridDim.x : threads / 32;
        for (int slot = worker; slot < count; slot += workers) {
            // Few expensive links should occupy different SMs, rather than
            // packing four branch-heavy warps into the first 128-thread block.
            if (w.reuse_blockers && threadIdx.x >= 32) continue;
            int rank = first + slot, i = 0;
            while (rank >= w.found - 1 - i) {
                rank -= w.found - 1 - i;
                ++i;
            }
            int j = i + rank + 1;
            auto &trial = w.bridges[slot];
            int warp_lane = lane % 32;
            if (!warp_lane) {
                trial.left = i;
                trial.right = j;
                trial.fault = {};
                trial.joined = false;
                trial.blockers = {};
                if (w.reuse_blockers) {
                    const auto left = w.witnesses[i].faces, right = w.witnesses[j].faces;
                    trial.blockers.left = left;
                    trial.blockers.right = right;
                    for (int k = 0; k < cache.blocker_count; ++k)
                        if (equal(cache.blockers[k].left, left) && equal(cache.blockers[k].right, right)) {
                            trial.blockers = cache.blockers[k];
                            break;
                        }
                }
            }
            __syncwarp();
            if (w.labels[i] != w.labels[j]) {
                Query local = q;
                local.speculative = &trial.fault;
                if (w.reuse_blockers) local.blockers = &trial.blockers;
                bool joined = check_connection(local, cache, w.witnesses[i], w.witnesses[j], trial.path, warp_lane);
                if (!warp_lane) {
                    trial.joined = joined;
                    if (w.serial_links) {
                        if (!agrees_with_serial_link(local, cache, w.witnesses[i], w.witnesses[j],
                                                     trial, w.serial_links[slot]))
                            q.fail(11);
                        atomicAdd(&w.checked_links, 1ULL);
                    }
                }
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
                if (w.reuse_blockers && t.blockers.count && saved_blockers < w.limit)
                    cache.next_blockers[saved_blockers++] = t.blockers;
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
        cache.paths_pose = q.rel[0];
        auto *old_blockers = cache.blockers;
        cache.blockers = cache.next_blockers;
        cache.next_blockers = old_blockers;
        cache.blocker_count = saved_blockers;
        w.group_ns += timer_ns() - stamp;
    }
    grid.sync();
}
template <class Grid> __device__ void detect(Grid grid, Query &q, PairScratch &cache) {
    detect_broad(grid, q, cache);
    if (*q.work->status) return;
    detect_sort(grid, q, cache);
    if (*q.work->status) return;
    detect_search(grid, q, cache);
    if (*q.work->status) return;
    detect_group(grid, q, cache);
    if (*q.work->status) return;
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
// One shared device function keeps contribution formation identical in the
// scalar and parallel paths, even when multiply-add contraction is enabled.
__device__ __noinline__ IntegralSum column_term(const Query &q, const SampleDepth &s) {
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
            for (int k = 0; k < 3; ++k) {
                sum.area.x[k] = __dadd_rn(sum.area.x[k], term.area.x[k]);
                sum.moment.x[k] = __dadd_rn(sum.moment.x[k], term.moment.x[k]);
                sum.centroid.x[k] = __dadd_rn(sum.centroid.x[k], term.centroid.x[k]);
            }
            sum.volume = __dadd_rn(sum.volume, term.volume);
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
    auto &ids = q.work->ordered->ids;
    auto &terms = q.work->ordered->terms;
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
    // Keep each warp's column IDs consecutive for ballot masks, but distribute
    // the first warps across blocks. Low-order rules then use more SMs.
    const int column_lane = w.reuse_blockers ?
        (threadIdx.x / 32 * gridDim.x + blockIdx.x) * 32 + threadIdx.x % 32 : lane;
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
            for (int i = column_lane; i < ((count + 31) / 32) * 32; i += threads) {
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
                    } else if (!local_rule && order >= 48 && q.b[side].refined_rules) {
                        auto cached = q.b[side].refined_rules[id * refined_rule_sites +
                                                           (order == 48 ? 0 : 48 * 48) + ij];
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
