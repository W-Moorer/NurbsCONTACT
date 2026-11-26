#include "compiled_geometry.hpp"
#include "contact_tasks.hpp"
#ifdef NC_WITH_CUDA_COLUMNS
#include "cuda_columns.hpp"
#endif
#include <chrono>
#include <deque>
#include <unordered_map>
#include <set>
#include <utility>

namespace nc::compiled {
namespace {
using Clock = std::chrono::steady_clock;
using Pair = std::array<int, 2>;
struct BlockEntry {
    Box coverage;
    Solid::ColumnBlock columns;
};
struct RuleCache {
    Rule rule;
    std::vector<BlockEntry> blocks;
#ifdef NC_WITH_CUDA_COLUMNS
    std::shared_ptr<device::Sites> device_sites;
#endif
};
using Rules = std::map<std::pair<int, int>, RuleCache>;
struct Seed {
    int side;
    Param uv;
    int tile = -1;
};
struct Proposal {
    double score;
    int side, site, tile;
};
using Candidates = std::map<Pair, std::vector<Proposal>>;
#ifdef NC_WITH_CUDA_COLUMNS
device::Pose device_pose(const Placement &pose) {
    device::Pose transform{};
    for(int j=0;j<3;++j) {
        transform.offset.x[j]=pose.offset[j];
        for(int k=0;k<3;++k) transform.cols[j].x[k]=pose.cols[j][k];
    }
    return transform;
}
#endif
struct Window {
    Placement anchor;
    std::size_t key;
    std::vector<std::pair<int, int>> leaves;
#ifdef NC_WITH_CUDA_COLUMNS
    std::shared_ptr<device::Columns::Window> device_window;
    size_t device_bytes{};
#endif
};
struct WindowCache {
    struct PlacedNode {
        Box box;
        Vec normal{}, point{};
        std::size_t version{};
    };
    std::vector<PlacedNode> placed;
    std::size_t version{};
    std::deque<Window> windows;
    std::unordered_multimap<std::size_t, const Window *> bins;
    const Window *active{};
    std::size_t bytes{};
    void save(Window window) {
        constexpr std::size_t capacity = 32u << 20;
        const auto size = [](const Window &w) {
            return sizeof(Window) + sizeof(w.leaves.front()) * w.leaves.capacity() + 64
#ifdef NC_WITH_CUDA_COLUMNS
                   + w.device_bytes
#endif
                   ;
        };
        while (!windows.empty() && bytes + size(window) > capacity) {
            const auto &old = windows.front();
            auto [i, end] = bins.equal_range(old.key);
            while (i != end && i->second != &old)
                ++i;
            bins.erase(i);
            bytes -= size(old);
            windows.pop_front();
        }
        bytes += size(window);
        windows.push_back(std::move(window));
        active = &windows.back();
        bins.emplace(active->key, active);
    }
};
void collect(const Solid &source, const Solid &target, const Placement &pose, int side, double reach,
             WindowCache &cache, Candidates &pairs
#ifdef NC_WITH_CUDA_COLUMNS
             ,device::Columns *gpu_source, device::Columns *gpu_target
#endif
             ) {
    if (source.site_tree.nodes.empty() || target.tile_tree.nodes.empty())
        return;
    auto &tree = target.tile_tree.nodes;
    const auto root = tree[0].box;
    const Vec center = scale(add(root.lo, root.hi), .5), half = scale(sub(root.hi, root.lo), .5);
    const double turn = std::min(2., reach / std::max(reach, length(half)));
#ifdef NC_WITH_CUDA_COLUMNS
    const auto transform=device_pose(pose);
#endif
    auto valid = [&](const Window &w) {
        const auto delta = sub(w.anchor.point(center), pose.point(center));
        double norm2 = 0;
        for (int k = 0; k < 3; ++k) {
            double margin = std::abs(delta[k]);
            for (int j = 0; j < 3; ++j) {
                double v = w.anchor.cols[j][k] - pose.cols[j][k];
                margin += half[j] * std::abs(v);
                norm2 += v * v;
            }
            if (margin > reach)
                return false;
        }
        return norm2 <= 2 * turn * turn;
    };
    if (!cache.active || !valid(*cache.active)) {
        std::size_t key = 0;
        auto mix = [&](double x) {
            key ^= std::hash<double>{}(std::floor(x / (8 * reach))) + 0x9e3779b9u + (key << 6) + (key >> 2);
        };
        const auto c = pose.point(center);
        for (int i = 0; i < 3; ++i) {
            mix(c[i]);
            for (int j = 0; j < 3; ++j)
                mix(pose.cols[j][i] * half[j]);
        }
        cache.active = nullptr;
        auto [it, end] = cache.bins.equal_range(key);
        for (; it != end; ++it)
            if (valid(*it->second)) {
                cache.active = it->second;
                break;
            }
        if (!cache.active) {
            Window window{pose, key, {}};
#ifdef NC_WITH_CUDA_COLUMNS
            if(gpu_source) window.device_window=gpu_source->window(*gpu_target,transform,reach,turn,window.device_bytes);
            else
#endif
            {
            cache.placed.resize(tree.size());
            ++cache.version;
            auto visit = [&](auto &&self, int a, int b) -> void {
                const auto &node = tree[b];
                auto &geometry = cache.placed[b];
                if (geometry.version != cache.version) {
                    geometry.box = node.box.placed(pose);
                    if (node.leaf >= 0) {
                        const auto &tile = target.tiles[node.leaf].site;
                        geometry.normal = pose.direction(tile.normal);
                        geometry.point = pose.point(tile.point);
                    }
                    geometry.version = cache.version;
                }
                auto walk = [&](auto &&recurse, int i) -> void {
                    const auto &site = source.site_tree.nodes[i];
                    if (!geometry.box.overlaps(site.box, 2 * reach + 1e-7))
                        return;
                    if (node.leaf < 0 && (site.leaf >= 0 || squared(sub(node.box.hi, node.box.lo)) >
                                                                squared(sub(site.box.hi, site.box.lo)))) {
                        self(self, i, node.left);
                        self(self, i, node.right);
                    } else if (site.leaf < 0) {
                        recurse(recurse, site.left);
                        recurse(recurse, site.right);
                    } else {
                        const int sample = source.material_sites[site.leaf];
                        const auto &s = source.tiles[sample].site;
                        const auto &n = geometry.normal;
                        const auto v = sub(s.point, geometry.point);
                        if (-dot3(n, s.normal) + turn + 1e-12 >= .5 &&
                            std::abs(dot3(v, n)) <= reach * (1 + std::sqrt(3.)) + length(v) * turn + 1e-7)
                            window.leaves.emplace_back(node.leaf, sample);
                    }
                };
                walk(walk, a);
            };
            visit(visit, 0, 0);
            std::sort(window.leaves.begin(), window.leaves.end());
            }
            cache.save(std::move(window));
        }
    }
#ifdef NC_WITH_CUDA_COLUMNS
    if(gpu_source) {
        std::vector<device::Candidate> hits;
        gpu_source->candidates(*gpu_target,*cache.active->device_window,transform,reach,hits);
        std::sort(hits.begin(),hits.end(),[](const auto &a,const auto &b){
            return std::tie(a.tile,a.site)<std::tie(b.tile,b.site);
        });
        for(const auto &h:hits) {
            auto a=source.tiles[h.site].face,b=target.tiles[h.tile].face;
            pairs[side?Pair{b,a}:Pair{a,b}].push_back({h.score,side,h.site,h.tile});
        }
        return;
    }
#endif
    int last = -1;
    Box box;
    Vec point{}, normal{};
    for (auto [t, i] : cache.active->leaves) {
        const auto &ref = target.tiles[t];
        const auto &site = source.tiles[i];
        if (last != t) {
            last = t;
            box = ref.box.placed(pose);
            point = pose.point(ref.site.point);
            normal = pose.direction(ref.site.normal);
        }
        if (!box.overlaps({site.site.point, site.site.point}, reach + 1e-7))
            continue;
        double cosine = -dot3(site.site.normal, normal);
        if (cosine < .5)
            continue;
        double d = -dot3(sub(site.site.point, point), normal) / cosine;
        if (std::abs(d) > reach)
            continue;
        Pair key = side ? Pair{ref.face, site.face} : Pair{site.face, ref.face};
        pairs[key].push_back({d, side, i, t});
    }
}
struct Witness {
    Pair faces;
    Vec point, normal;
    double depth;
    std::array<RayRoot, 2> exits;
    Seed seed;
};
struct Blocker {
    int side;
    RayRoot root;
};
struct Detection {
    std::array<const Solid *, 2> bodies;
    std::array<Placement, 2> rel;
    std::array<Rules *, 2> rules;
    const double reach;
    std::vector<Blocker> *blockers{};
#ifdef NC_WITH_CUDA_COLUMNS
    const std::array<std::shared_ptr<device::Columns>,2> *gpu{};
    ContactWork *work{};
#endif
    const Face &face(int side, int id) const {
        return bodies[side]->faces[id];
    }
    RuleCache &rule(int side, int id, int order) const {
        auto [it, added] = rules[side]->try_emplace({id, order});
        if (added) {
            it->second.rule = face(side, id).quadrature(order);
            it->second.blocks.resize(it->second.rule.blocks.size());
#ifdef NC_WITH_CUDA_COLUMNS
            if(gpu) {
                std::vector<device::RuleSite> sites;
                for(const auto &s:it->second.rule.sites) sites.push_back({{{s.point[0],s.point[1],s.point[2]}},
                                                                          {{s.normal[0],s.normal[1],s.normal[2]}}});
                it->second.device_sites=std::make_shared<device::Sites>(sites);
            }
#endif
        }
        return it->second;
    }
    std::optional<Witness> chord(Vec p, Vec n) const {
        auto a = bodies[0]->first(p, n, reach);
        if (!a)
            return {};
        auto b = bodies[1]->first(rel[0].point(p), rel[0].direction(scale(n, -1)), reach);
        if (!b)
            return {};
        return Witness{{a->face, b->face}, add(p, scale(n, .5 * (a->t - b->t))), n, a->t + b->t, {*a, *b}, {}};
    }
    std::optional<Witness> sample(Pair ids, const Seed &seed, const NodeSample &site, double &best) const {
        const int side = seed.side;
        const auto &target = face(1 - side, ids[1 - side]);
        auto uv = bodies[1 - side]->tiles[seed.tile].site.uv;
        Vec p = rel[side].point(site.point), d = rel[side].direction(scale(site.normal, -1));
        for (int iter = 0; iter < 8; ++iter) {
            const auto j = target.at(uv);
            Vec delta = sub(j.p, p);
            double t = dot3(delta, d);
            Vec f = sub(delta, scale(d, t));
            if (squared(f) < 1e-14) {
                if (std::abs(t) > reach || !target.inside(uv))
                    return {};
                best = std::max(best, t);
                if (t <= 2e-7)
                    return {};
                Vec mid = sub(site.point, scale(site.normal, .5 * t)), n = site.normal;
                if (side) {
                    mid = rel[1].point(mid);
                    n = rel[1].direction(scale(n, -1));
                }
                auto w = chord(mid, n);
                if (w && w->faces == ids) {
                    w->seed = seed;
                    return w;
                }
                return {};
            }
            auto cu = cross3(j.v, d), cv = cross3(d, j.u);
            double det = dot3(j.u, cu);
            if (det == 0 || std::abs(det) < 1e-11 * length(j.u) * length(j.v))
                return {};
            uv[0] -= dot3(f, cu) / det;
            uv[1] -= dot3(f, cv) / det;
            if (!target.info.domain.u.contains(uv[0]) || !target.info.domain.v.contains(uv[1]) ||
                (!target.analytic && !target.patch(uv)))
                return {};
        }
        return {};
    }
    NodeSample node(int side, int id, Param uv) const {
        const auto &f = face(side, id);
        auto j = f.at(uv);
        Vec n = cross3(j.u, j.v);
        double mag = length(n);
        return {j.p, mag ? scale(n, f.info.orientation / mag) : Vec{}, uv, mag};
    }
    std::optional<Witness> find(Pair ids, std::vector<Proposal> &proposals, const Seed *prior) const {
        double best = -infinity;
        Seed seed{};
        if (prior) {
            auto s = node(prior->side, ids[prior->side], prior->uv);
            if (auto w = sample(ids, *prior, s, best))
                return w;
            seed = *prior;
        }
        std::sort(proposals.begin(), proposals.end(),
                  [](const auto &a, const auto &b) { return a.score > b.score; });
        std::vector<std::pair<int, int>> tried;
        for (auto &p : proposals) {
            std::pair k{p.side, p.site};
            if (std::find(tried.begin(), tried.end(), k) != tried.end())
                continue;
            tried.push_back(k);
            const auto &s = bodies[p.side]->tiles[p.site].site;
            Seed trial{p.side, s.uv, p.tile};
            const double old = best;
            if (auto w = sample(ids, trial, s, best))
                return w;
            if (best > old)
                seed = trial;
        }
        if (!std::isfinite(best))
            return {};
        const auto &f = face(seed.side, ids[seed.side]);
        Param span{f.info.domain.u.width() / 12, f.info.domain.v.width() / 12};
        for (int k = 0; k < 10; ++k) {
            Param base = seed.uv;
            if (k >= 5 && best < -std::sqrt(1e-7 * reach))
                break;
            constexpr std::array<Param, 8> directions{
                {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, 1}, {1, -1}, {-1, -1}}};
            for (auto step : directions) {
                Param uv{base[0] + step[0] * span[0], base[1] + step[1] * span[1]};
                if (!f.inside(uv) || uv[0] <= f.info.domain.u.lo + 1e-9 || uv[0] >= f.info.domain.u.hi - 1e-9 ||
                    uv[1] <= f.info.domain.v.lo + 1e-9 || uv[1] >= f.info.domain.v.hi - 1e-9)
                    continue;
                auto s = node(seed.side, ids[seed.side], uv);
                double old = best;
                auto trial = seed;
                trial.uv = uv;
                if (auto w = sample(ids, trial, s, best))
                    return w;
                if (best > old)
                    seed.uv = uv;
            }
            span[0] *= .5;
            span[1] *= .5;
        }
        return {};
    }
    bool crosses(int side, Vec start, Vec end, const RayRoot &hint) const {
        Vec d = sub(end, start);
        double len = length(d);
        if (len < 1e-7)
            return false;
        d = scale(d, 1 / len);
        if (std::abs(dot3(hint.normal, d)) < 1e-5)
            return false;
        const auto &f = face(side, hint.face);
        Param uv = hint.uv;
        for (int i = 0; i < 5; ++i) {
            auto s = f.at(uv);
            Vec v = sub(s.p, start);
            double t = dot3(v, d);
            v = sub(v, scale(d, t));
            Vec c = cross3(s.v, d);
            double det = dot3(s.u, c);
            if (std::abs(det) <= 1e-7 * length(s.u) * length(s.v))
                return false;
            if (squared(v) < 1e-16)
                return t > 4e-7 && t < len - 4e-7 && f.inside(uv);
            uv[0] -= dot3(v, c) / det;
            uv[1] -= dot3(v, cross3(d, s.u)) / det;
            if (!f.info.domain.u.contains(uv[0]) || !f.info.domain.v.contains(uv[1]) ||
                (!f.analytic && !f.patch(uv)))
                return false;
        }
        return false;
    }
    bool straight(Vec x, Vec y) const {
        const std::array<Vec, 2> a{x, rel[0].point(x)}, b{y, rel[0].point(y)};
        if (blockers)
            for (const auto &h : *blockers)
                if (crosses(h.side, a[h.side], b[h.side], h.root))
                    return false;
        for (int side = 0; side < 2; ++side)
            if (auto root = bodies[side]->crossing(a[side], b[side])) {
                if (blockers) {
                    auto i = std::find_if(blockers->begin(), blockers->end(), [&](const auto &h) {
                        return h.side == side && h.root.face == root->face;
                    });
                    if (i != blockers->end())
                        blockers->erase(i);
                    else if (blockers->size() == 4)
                        blockers->pop_back();
                    blockers->insert(blockers->begin(), {side, *root});
                }
                return false;
            }
        return true;
    }
    bool connected(const Witness &x, const Witness &y, int depth, std::vector<Vec> &path) const {
        Vec a = x.point, b = y.point, c = rel[0].point(a), d = rel[0].point(b);
        if (!crosses(0, a, b, x.exits[0]) && !crosses(0, b, a, y.exits[0]) && !crosses(1, c, d, x.exits[1]) &&
            !crosses(1, d, c, y.exits[1]) && straight(a, b))
            return true;
        if (!depth)
            return false;
        for (int side = 0; side < 2; ++side) {
            if (x.faces[side] != y.faces[side])
                continue;
            const auto a = x.exits[side].uv, b = y.exits[side].uv;
            const Param uv{(a[0] + b[0]) * .5, (a[1] + b[1]) * .5};
            const auto &f = face(side, x.faces[side]);
            if (!f.inside(uv))
                continue;
            auto s = node(side, x.faces[side], uv);
            Vec n = add(x.exits[side].normal, y.exits[side].normal);
            if (length(n) < 1e-6)
                continue;
            n = scale(n, 1 / length(n));
            const auto other =
                bodies[1 - side]->first(rel[side].point(s.point), rel[side].direction(scale(n, -1)), reach);
            if (!other)
                continue;
            auto p = sub(s.point, scale(n, .5 * other->t));
            if (side) {
                p = rel[1].point(p);
                n = rel[1].direction(scale(n, -1));
            }
            auto mid = chord(p, n);
            if (mid && mid->faces[side] == x.faces[side]) {
                auto start = path.size();
                if (connected(x, *mid, depth - 1, path)) {
                    path.push_back(mid->point);
                    if (connected(*mid, y, depth - 1, path))
                        return true;
                }
                path.resize(start);
            }
        }
        return false;
    }
    struct Integral {
        Vec area{}, moment{}, centroid{};
        double volume{};
    };
    void sum(Integral &q, int side, const NodeSample &s, double depth) const {
        Vec normal = side ? rel[1].direction(s.normal) : s.normal;
        Vec point = side ? rel[1].point(s.point) : s.point;
        Vec area = scale(normal, (side ? -1 : 1) * s.area);
        q.area = add(q.area, area);
        q.moment = add(q.moment, cross3(point, area));
        const double vol = depth * s.area;
        q.volume += vol;
        q.centroid = add(q.centroid, scale(sub(point, scale(normal, .5 * depth)), vol));
    }
    using Groups = std::array<std::map<int, std::set<int>>, 2>;
    std::optional<Witness> resultant(const Integral &q, Vec &arm) const {
        double area = length(q.area);
        if (!(q.volume > 0 && area > 1e-18))
            return {};
        auto p = scale(q.centroid, 1 / q.volume), n = scale(q.area, 1 / area);
        auto w = chord(p, n);
        if (w) {
            w->point = p;
            arm = sub(scale(q.moment, 1 / area), cross3(p, n));
        }
        return w;
    }
    std::optional<Witness> integrate(const Groups &groups, int order, Vec &arm) const {
        struct Job {
            const Rule *rule;
            const std::set<int> *targets;
            BlockEntry *table;
            std::size_t block_id, offset;
            int side;
        };
        std::vector<Job> jobs;
#ifdef NC_WITH_CUDA_COLUMNS
        std::vector<device::RuleBlock> device_jobs;
        std::vector<int> target_ids;
#endif
        std::size_t sites = 0;
        for (int side = 0; side < 2; ++side)
            for (auto &[id, targets] : groups[side]) {
                auto &cached = rule(side, id, order);
                const auto &r = cached.rule;
#ifdef NC_WITH_CUDA_COLUMNS
                int start=static_cast<int>(target_ids.size());
                if(gpu) target_ids.insert(target_ids.end(),targets.begin(),targets.end());
#endif
                Box around;
                for (int t : targets)
                    around.include(face(1 - side, t).box.placed(rel[1 - side]));
                for (std::size_t block_id = 0; block_id < r.blocks.size(); ++block_id) {
                    const auto &block = r.blocks[block_id];
                    if (!around.overlaps(block.box, reach))
                        continue;
                    jobs.push_back({&r, &targets, &cached.blocks[block_id], block_id, sites, side});
#ifdef NC_WITH_CUDA_COLUMNS
                    if(gpu) device_jobs.push_back({cached.device_sites->ptr,static_cast<int>(block.first),
                        static_cast<int>(block.end-block.first),static_cast<int>(sites),side,start,
                        static_cast<int>(target_ids.size())});
#endif
                    sites += block.end - block.first;
                }
            }
        // A block exclusively owns its column cache. Save depths so that load
        // accumulation follows the original face/block/site order on every CPU.
        std::vector<double> depths(sites, std::numeric_limits<double>::quiet_NaN());
#ifdef NC_WITH_CUDA_COLUMNS
        if(gpu) {
            std::vector<device::Depth> active;
            work->cuda_column_rays+=(*gpu)[0]->integrate(*(*gpu)[1],device_jobs,target_ids,device_pose(rel[0]),
                                                         device_pose(rel[1]),reach,active);
            for(const auto &hit:active) {
                const auto &job=jobs[hit.job];
                const auto &block=job.rule->blocks[job.block_id];
                double depth=hit.value;
                if(depth<0) {
                    ++work->cuda_column_subdivisions;
                    const auto &s=job.rule->sites[block.first+hit.local];
                    auto root=bodies[1-job.side]->first(rel[job.side].point(s.point),
                                                        rel[job.side].direction(scale(s.normal,-1)),reach);
                    if(!root || !job.targets->contains(root->face)) continue;
                    depth=root->t;
                }
                depths[job.offset+hit.local]=depth;
            }
        } else
#endif
        contact_tasks::ranges(jobs.size(), 2, [&](std::size_t first, std::size_t end) {
            std::vector<RayRoot> scratch;
            for (auto j = first; j < end; ++j) {
                const auto &[rule, target_set, entry, block_id, offset, side] = jobs[j];
                const auto &targets = *target_set;
                const auto &r = *rule;
                const auto &block = r.blocks[block_id];
                const auto &solid = *bodies[1 - side];
                Box swept = block.box;
                for (int k = 0; k < 3; ++k) {
                    swept.lo[k] -= reach * std::max(0., block.normal_hi[k]) + 2e-6;
                    swept.hi[k] -= reach * std::min(0., block.normal_lo[k]) - 2e-6;
                }
                const Box current = swept.placed(rel[side]);
                auto &table = *entry;
                bool contained = true;
                for (int k = 0; k < 3; ++k)
                    contained &=
                        current.lo[k] >= table.coverage.lo[k] && current.hi[k] <= table.coverage.hi[k];
                if (!contained) {
                    const Vec pad{2 * reach, 2 * reach, 2 * reach};
                    table.coverage = {sub(current.lo, pad), add(current.hi, pad)};
                    table.columns = solid.column_block(table.coverage);
                }
                const auto &candidates = table.columns;
                if (std::none_of(candidates.candidates.begin(), candidates.candidates.end(),
                                 [&](const auto &c) { return targets.contains(c.face); }))
                    continue;
                for (auto i = block.first; i < block.end; ++i) {
                    const auto &s = r.sites[i];
                    Vec p = rel[side].point(s.point), d = rel[side].direction(scale(s.normal, -1));
                    if (std::none_of(targets.begin(), targets.end(),
                                     [&](int t) { return face(1 - side, t).box.ray(p, d, reach, 1e-7); }))
                        continue;
                    auto root = solid.first(p, d, reach, candidates, scratch);
                    if (root && targets.contains(root->face))
                        depths[offset + (i - block.first)] = root->t;
                }
            }
        });
        Integral q;
        for (const auto &job : jobs) {
            const auto &block = job.rule->blocks[job.block_id];
            for (auto i = block.first; i < block.end; ++i) {
                const double depth = depths[job.offset + (i - block.first)];
                if (!std::isnan(depth)) sum(q, job.side, job.rule->sites[i], depth);
            }
        }
        return resultant(q, arm);
    }
    using Windows = std::array<std::map<int, Rect>, 2>;
    Rect footprint(int side, int id, const std::set<int> &targets, std::span<const Witness> witnesses) const {
        const auto &f = face(side, id);
        Rect r{{infinity, -infinity}, {infinity, -infinity}};
        for (const auto &w : witnesses)
            if (w.faces[side] == id) {
                Param uv = w.exits[side].uv;
                for (int k = 0; k < 2; ++k) {
                    const Interval range = k ? f.info.domain.v : f.info.domain.u;
                    for (int sign : {-1, 1}) {
                        double h = range.width() / 192, delta = 0;
                        for (int j = 0; j < 10; ++j) {
                            Param trial = uv;
                            trial[k] = std::clamp(uv[k] + sign * h, range.lo, range.hi);
                            delta = trial[k] - uv[k];
                            if (!f.inside(trial))
                                break;
                            auto s = node(side, id, trial);
                            auto t = bodies[1 - side]->first(rel[side].point(s.point),
                                                             rel[side].direction(scale(s.normal, -1)), reach);
                            if (!t || !targets.contains(t->face) || trial[k] == range.lo ||
                                trial[k] == range.hi)
                                break;
                            h *= 2;
                        }
                        auto &a = k ? r.v : r.u;
                        a.lo = std::min(a.lo, uv[k] + delta);
                        a.hi = std::max(a.hi, uv[k] + delta);
                    }
                }
            }
        return r;
    }
    // Clip material intervals on each integration line. Plain midpoint rules
    // otherwise change an entire cell's weight when a narrow strip crosses it.
    void strip_integral(Integral &q, int side, int id, const std::set<int> &targets, Rect r,
                        int order) const {
        const auto &f = face(side, id);
        const auto &solid = *bodies[1 - side];
        const bool transpose = r.u.width() / f.info.domain.u.width() < r.v.width() / f.info.domain.v.width();
        const Interval outer = transpose ? r.v : r.u, inner = transpose ? r.u : r.v;
        auto column = [&](double u, double v, double weight) {
            const Param uv = transpose ? Param{v, u} : Param{u, v};
            if (!f.inside(uv))
                return false;
            auto s = node(side, id, uv);
            auto hit = solid.first(rel[side].point(s.point), rel[side].direction(scale(s.normal, -1)), reach);
            if (!hit || !targets.contains(hit->face))
                return false;
            if (weight > 0) {
                s.area *= weight;
                sum(q, side, s, hit->t);
            }
            return true;
        };
        const double du = outer.width() / order;
        const int scans = std::max(6, order);
        for (int i = 0; i < order; ++i) {
            const double u = outer.lo + (i + .5) * du;
            auto integrate_span = [&](double lo, double hi) {
                const double half = (hi - lo) * .5, mid = std::midpoint(lo, hi);
                for (double sign : {-1., 1.})
                    column(u, mid + sign * half / std::sqrt(3.), du * half);
            };
            double last_v = inner.lo, start = last_v;
            bool loaded = column(u, last_v, 0);
            for (int j = 1; j <= scans; ++j) {
                const double v = inner.lo + j * inner.width() / scans;
                const bool next = column(u, v, 0);
                if (next != loaded) {
                    double lo = last_v, hi = v;
                    for (int k = 0; k < 10; ++k) {
                        const double mid = std::midpoint(lo, hi);
                        if (column(u, mid, 0) == loaded)
                            lo = mid;
                        else
                            hi = mid;
                    }
                    const double cut = std::midpoint(lo, hi);
                    if (next)
                        start = cut;
                    else
                        integrate_span(start, cut);
                }
                if (j == scans && next)
                    integrate_span(start, v);
                loaded = next;
                last_v = v;
            }
        }
    }
    std::optional<Witness> local_integral(std::span<const Witness> witnesses, const Groups &groups, Vec &arm,
                                          int order = 24, Windows *windows = nullptr) const {
        Integral q;
        for (int side = 0; side < 2; ++side)
            for (const auto &[id, targets] : groups[side]) {
                const auto &f = face(side, id);
                Rect r;
                if (windows) {
                    auto [it, inserted] = (*windows)[side].try_emplace(id);
                    if (inserted)
                        it->second = footprint(side, id, targets, witnesses);
                    r = it->second;
                    strip_integral(q, side, id, targets, r, order);
                    continue;
                }
                r = footprint(side, id, targets, witnesses);
                for (int i = 0; i < order; ++i)
                    for (int j = 0; j < order; ++j) {
                        Param uv{r.u.lo + (i + .5) * r.u.width() / order,
                                 r.v.lo + (j + .5) * r.v.width() / order};
                        if (!f.inside(uv))
                            continue;
                        auto s = node(side, id, uv);
                        s.area *= r.u.width() * r.v.width() / (order * order);
                        auto h = bodies[1 - side]->first(rel[side].point(s.point),
                                                         rel[side].direction(scale(s.normal, -1)), reach);
                        if (h && targets.contains(h->face))
                            sum(q, side, s, h->t);
                    }
            }
        return resultant(q, arm);
    }
};
class CompiledPair final : public RegionQuery {
    std::array<std::shared_ptr<const Solid>, 2> bodies_;
    double reach_;
    std::optional<LoadQuadrature> integration_;
#ifdef NC_WITH_CUDA_COLUMNS
    std::array<std::shared_ptr<device::Columns>,2> gpu_{};
#endif
    mutable std::array<Rules, 2> rules_;
    mutable std::map<Pair, Seed> previous_;
    mutable std::array<WindowCache, 2> windows_{};
    mutable std::map<std::pair<Pair, Pair>, std::vector<Vec>> paths_;
    mutable std::map<std::pair<Pair, Pair>, std::vector<Blocker>> blockers_;

  public:
    CompiledPair(std::shared_ptr<const Solid> a, std::shared_ptr<const Solid> b, double reach,
                 std::optional<LoadQuadrature> integration,bool use_cuda)
        : bodies_{std::move(a), std::move(b)}, reach_(reach), integration_(integration) {
        if(use_cuda) {
#ifdef NC_WITH_CUDA_COLUMNS
            if(integration && !integration->reference_order) throw std::invalid_argument("cuda-columns requires fixed cubature");
            for(int i=0;i<2;++i) gpu_[i]=cuda_columns(*bodies_[i]);
#else
            throw std::invalid_argument("configure NC_WITH_CUDA_COLUMNS=ON to use cuda-columns");
#endif
        }
    }
    bool independent_cad_patch() const override {
        return true;
    }
    std::vector<ContactRegion> regions(ContactPose a, ContactPose b, ContactWork &work) const override {
        return evaluate(a, b, work, nullptr);
    }
    std::vector<ContactRegion> with_faces(ContactPose a, ContactPose b, ContactWork &work,
                                          GroupFaces &groups) const override {
        groups.clear();
        return evaluate(a, b, work, &groups);
    }
    std::vector<ContactRegion> evaluate(ContactPose a, ContactPose b, ContactWork &work,
                                        GroupFaces *groups) const {
        ++work.cad_patch_queries;
        Detection query{{bodies_[0].get(), bodies_[1].get()},
                        {Placement::between(a, b), Placement::between(b, a)},
                        {&rules_[0], &rules_[1]},
                        reach_};
#ifdef NC_WITH_CUDA_COLUMNS
        if(gpu_[0]) {query.gpu=&gpu_;query.work=&work;}
#endif
        const auto tic = Clock::now();
        Candidates pairs;
        Candidates reverse;
        contact_tasks::ranges(2, 1, [&](auto first, auto end) {
            for (int side = static_cast<int>(first); side < static_cast<int>(end); ++side)
                collect(*bodies_[side], *bodies_[1 - side], query.rel[1 - side], side, reach_,
                        windows_[side], side ? reverse : pairs
#ifdef NC_WITH_CUDA_COLUMNS
                        ,gpu_[side].get(),gpu_[1-side].get()
#endif
                        );
        });
        pairs.merge(reverse);
        for (auto &[ids, more] : reverse) {
            auto &ordered = pairs.at(ids);
            ordered.insert(ordered.end(), more.begin(), more.end());
        }
        for (auto &[ids, seed] : previous_)
            pairs.try_emplace(ids);
        auto now = Clock::now();
        work.cad_patch_broad_s += std::chrono::duration<double>(now - tic).count();
        std::vector<Witness> found;
        std::map<Pair, Seed> next;
        std::vector<Candidates::iterator> candidates;
        candidates.reserve(pairs.size());
        for (auto it = pairs.begin(); it != pairs.end(); ++it) candidates.push_back(it);
        std::vector<std::optional<Witness>> confirmed(candidates.size());
        const auto &previous = std::as_const(previous_);
        contact_tasks::ranges(candidates.size(), 2, [&](auto first, auto end) {
            for (auto i = first; i < end; ++i) {
                auto &[ids, proposals] = *candidates[i];
                const auto seed = previous.find(ids);
                confirmed[i] = query.find(ids, proposals, seed == previous.end() ? nullptr : &seed->second);
            }
        });
        for (std::size_t i = 0; i < confirmed.size(); ++i) {
            if (!confirmed[i]) continue;
            found.push_back(*confirmed[i]);
            next.emplace_hint(next.end(), candidates[i]->first, confirmed[i]->seed);
        }
        previous_ = std::move(next);
        const auto search = Clock::now();
        work.cad_patch_search_s += std::chrono::duration<double>(search - now).count();
        std::vector<int> labels(found.size());
        std::iota(labels.begin(), labels.end(), 0);
        decltype(paths_) next_paths;
        decltype(blockers_) next_blockers;
        for (int i = 0; i < int(found.size()); ++i)
            for (int j = i + 1; j < int(found.size()); ++j) {
                if (labels[i] == labels[j])
                    continue;
                auto key = std::pair{found[i].faces, found[j].faces};
                query.blockers = &next_blockers[key];
                if (auto b = blockers_.find(key); b != blockers_.end())
                    *query.blockers = std::move(b->second);
                bool joined = false;
                std::vector<Vec> path;
                if (auto it = paths_.find(key); it != paths_.end()) {
                    Vec last = found[i].point;
                    joined = true;
                    for (Vec p : it->second) {
                        if (!query.straight(last, p)) {
                            joined = false;
                            break;
                        }
                        last = p;
                    }
                    joined = joined && query.straight(last, found[j].point);
                    if (joined)
                        path = std::move(it->second);
                }
                if (joined || query.connected(found[i], found[j], 5, path)) {
                    int old = labels[j];
                    std::replace(labels.begin(), labels.end(), old, labels[i]);
                    if (!path.empty())
                        next_paths.emplace(key, std::move(path));
                }
            }
        paths_ = std::move(next_paths);
        blockers_ = std::move(next_blockers);
        query.blockers = nullptr;
        std::map<int, std::vector<Witness>> components;
        for (std::size_t k = 0; k < found.size(); ++k)
            components[labels[k]].push_back(found[k]);
        const auto grouped = Clock::now();
        work.cad_patch_group_s += std::chrono::duration<double>(grouped - search).count();
        std::vector<ContactRegion> result;
        for (auto &[label, witnesses] : components) {
            if (groups) {
                auto &faces = groups->emplace_back();
                for (const auto &w : witnesses)
                    faces.push_back({w.faces[0] + 1, w.faces[1] + 1});
            }
            Detection::Groups faces;
            for (auto &w : witnesses)
                for (int i = 0; i < 2; ++i)
                    faces[i][w.faces[i]].insert(w.faces[1 - i]);
            Vec arm{};
            std::optional<Witness> w;
            if (integration_) {
                const auto &law = *integration_;
                // At this geometry pose the strictly positive elastic loads
                // are averaged by the shared law and summed by the other law.
                const double count = law.shared_stiffness ? 1. : double(components.size());
                std::optional<std::array<Vec, 3>> last;
                Detection::Windows windows;
                for (int order : {3, 6, 12, 24, 48, 96}) {
                    if (law.reference_order && order != law.reference_order)
                        continue;
                    w = law.reference_order ? query.integrate(faces, order, arm)
                                            : query.local_integral(witnesses, faces, arm, order, &windows);
                    ++work.cad_quadrature_rules;
                    if (!w) {
                        last.reset();
                        continue;
                    }
                    if (law.reference_order)
                        break;
                    const double load = law.stiffness * std::pow(.001 * w->depth, law.exponent);
                    const Vec force = scale(w->normal, load);
                    const Vec moment = scale(add(cross3(w->point, force), scale(arm, load)), .001);
                    const Vec other = sub(moment, scale(cross3(query.rel[1].offset, force), .001));
                    const std::array current{force, moment, other};
                    if (last && length(sub(force, (*last)[0])) * count <= law.force_tolerance &&
                        std::max(length(sub(moment, (*last)[1])), length(sub(other, (*last)[2]))) * count <=
                            law.moment_tolerance) {
                        ++work.cad_quadrature_converged;
                        break;
                    }
                    last = current;
                    w.reset();
                }
                if (!w && law.reference_order) {
                    // A known contact can be narrower than a whole-face cell.
                    // Apply the same fixed order on its local material footprint.
                    w = query.local_integral(witnesses, faces, arm, law.reference_order, &windows);
                    ++work.cad_quadrature_rules;
                }
                if (!w)
                    throw UnresolvedContact(law.reference_order
                                                ? "fixed CAD quadrature missed the contact region"
                                                : "compiled CAD force/moment quadrature did not converge");
            } else {
                for (int order : {12, 24, 48, 96})
                    if ((w = query.integrate(faces, order, arm)))
                        break;
            }
            if (!w)
                w = query.local_integral(witnesses, faces, arm);
            if (!w)
                throw UnresolvedContact("compiled CAD pressure centroid has no interior chord");
            result.push_back({a.point(scale(w->point, .001)), a.rotate(w->normal), .001 * w->depth,
                              a.rotate(scale(arm, .001))});
        }
        work.cad_patch_area_s += std::chrono::duration<double>(Clock::now() - grouped).count();
        return result;
    }
    SurfaceTransition transition(ContactPose, ContactPose, ContactPose, ContactPose, double) const override {
        throw std::invalid_argument(
            "compiled contact supports discrete integration; swept event queries are unavailable");
    }
};
} // namespace
std::shared_ptr<const RegionQuery> region_query(std::shared_ptr<const Solid> a,
                                                std::shared_ptr<const Solid> b, double max_depth,
                                                std::optional<LoadQuadrature> integration,bool use_cuda) {
    if (!(max_depth > 0))
        throw std::invalid_argument("compiled CAD requires a positive contact depth range");
    if (integration) {
        const auto &p = *integration;
        const auto positive=[](double x) { return std::isfinite(x) && x>0; };
        if (!positive(p.stiffness) || !positive(p.exponent) ||
            (p.reference_order != 0 && p.reference_order != 96) ||
            (!p.reference_order && (!positive(p.force_tolerance) || !positive(p.moment_tolerance))))
            throw std::invalid_argument("invalid compiled CAD quadrature settings");
    }
    return std::make_shared<CompiledPair>(std::move(a), std::move(b), 2000 * max_depth, integration,use_cuda);
}
} // namespace nc::compiled
