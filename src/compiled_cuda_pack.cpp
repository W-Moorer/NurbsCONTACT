#include "compiled_geometry.hpp"
#include "cuda_columns.hpp"

namespace nc::compiled {
namespace {
device::V pack(Vec a) { return {{a[0], a[1], a[2]}}; }
device::V pack_midpoint(const IVec &a) { return {{a[0].center(), a[1].center(), a[2].center()}}; }
device::Bounds pack(Box a) { return {pack(a.lo), pack(a.hi)}; }
device::Range pack(Rect a) { return {a.u.lo, a.u.hi, a.v.lo, a.v.hi}; }
void prepare_support(const Face &f, device::FaceData &out) {
    std::array<Vec,3> axes;
    std::vector<Vec> points;
    Box normals;
    double evaluation_pad=0;
    auto add_normal = [&](IVec u, IVec v) {
        auto n=cross(u,v);
        for (int k=0;k<3;++k) n[k]=n[k]*Interval{double(f.info.orientation)};
        normals.include(Vec{n[0].lo,n[1].lo,n[2].lo});
        normals.include(Vec{n[0].hi,n[1].hi,n[2].hi});
    };
    if (f.analytic) {
        const auto *plane=std::get_if<Plane>(f.analytic);
        if (!plane) return;
        Vec u{},v{};
        for (int k=0;k<3;++k) {u[k]=plane->x[k].center();v[k]=plane->y[k].center();}
        const auto normal=cross3(u,v);
        if (length(u)<1e-12 || length(normal)<1e-12) return;
        axes[0]=scale(u,1/length(u));axes[2]=scale(normal,1/length(normal));
        axes[1]=cross3(axes[2],axes[0]);
        add_normal(plane->x,plane->y);
        for (double a:{f.info.domain.u.lo,f.info.domain.u.hi})
            for (double b:{f.info.domain.v.lo,f.info.domain.v.hi}) {
                const auto p=plane->eval(UV{Interval{a},Interval{b}});
                for (int bits=0;bits<8;++bits)
                    points.push_back({bits&1?p[0].hi:p[0].lo,bits&2?p[1].hi:p[1].lo,bits&4?p[2].hi:p[2].lo});
            }
    } else {
        if (f.patches.empty()) return;
        axes=f.span_frame;
        for (const auto &patch:f.patches) {
            if (!patch.cubic || patch.cells.empty()) return;
            double magnitude=0;
            for (auto coefficient:patch.power) for (double x:coefficient) magnitude+=std::abs(x);
            const double polynomial_pad=4096*std::numeric_limits<double>::epsilon()*(1+magnitude);
            evaluation_pad=std::max(evaluation_pad,polynomial_pad);
            for (const auto &h:patch.net) {
                if (!(h[3]>0)) return;
                points.push_back({h[0]/h[3],h[1]/h[3],h[2]/h[3]});
            }
            // Terminal derivative hulls cover the complete parameter rectangle.
            for (size_t i=0;i<patch.cells.size();++i) {
                if (2*i+1<patch.cells.size()) continue;
                const auto &c=patch.cells[i];
                IVec d[2];
                for (int a=0;a<2;++a) for (int k=0;k<3;++k) {
                    const double pad=polynomial_pad+1e-10*(1+std::abs(c.slope[a][k])+c.radius[a][k]);
                    d[a][k]={down(c.slope[a][k]-c.radius[a][k]-pad),up(c.slope[a][k]+c.radius[a][k]+pad)};
                }
                add_normal(d[0],d[1]);
            }
        }
    }
    Box support;
    for (auto p:points) support.include(Vec{dot3(axes[0],p),dot3(axes[1],p),dot3(axes[2],p)});
    const double pad=std::max({1e-5,16*out.tol,16*evaluation_pad})+1e-10*(1+length(f.box.lo)+length(f.box.hi));
    for (int k=0;k<3;++k) {
        if (!std::isfinite(support.lo[k]) || !std::isfinite(support.hi[k]) ||
            !std::isfinite(normals.lo[k]) || !std::isfinite(normals.hi[k])) return;
        if (std::abs(length(axes[k])-1)>1e-10) return;
        out.support_axes[k]=pack(axes[k]);
        support.lo[k]=down(support.lo[k]-pad);support.hi[k]=up(support.hi[k]+pad);
    }
    out.support_box=pack(support);out.normal_box=pack(normals);out.support_ready=true;
}
device::TrimPiece trim_piece(const Face::TrimSpan &edge, device::Tables &t) {
    device::TrimPiece r{};
    r.box = pack(edge.box);
    r.first_y = edge.first_y.center();
    r.last_y = edge.last_y.center();
    r.start = static_cast<int>(t.trim_poles.size());
    if (const auto *b = std::get_if<BezierCurve>(edge.curve)) {
        r.count = static_cast<int>(b->poles.size());
        double min_weight = infinity, max_x = 0, max_y = 0, dx = 0, dy = 0, dw = 0;
        for (size_t i = 0; i < b->poles.size(); ++i) {
            const auto &h = b->poles[i];
            min_weight = std::min(min_weight, h[3].lo);
            max_x = std::max(max_x, std::max(std::abs(h[0].lo), std::abs(h[0].hi)) / h[3].lo);
            max_y = std::max(max_y, std::max(std::abs(h[1].lo), std::abs(h[1].hi)) / h[3].lo);
            const auto xy = cartesian(h);
            if (i == 0 || i + 1 == b->poles.size())
                r.endpoints[i == 0 ? 0 : 1] = {{{xy[0].lo, xy[1].lo, 0}}, {{xy[0].hi, xy[1].hi, 0}}};
            r.roundoff = std::max({r.roundoff, xy[0].width(), xy[1].width()});
            t.trim_poles.push_back({h[0].center(), h[1].center(), h[3].center()});
            t.trim_intervals.push_back({{{h[0].lo, h[1].lo, h[3].lo}}, {{h[0].hi, h[1].hi, h[3].hi}}});
            if (i) {
                const auto bound = [&](int k) {
                    auto d = h[k] - b->poles[i - 1][k];
                    return std::max(std::abs(d.lo), std::abs(d.hi));
                };
                dx = std::max(dx, bound(0));
                dy = std::max(dy, bound(1));
                dw = std::max(dw, bound(3));
            }
        }
        r.du = (r.count - 1) * (dx + max_x * dw) / min_weight;
        r.dv = (r.count - 1) * (dy + max_y * dw) / min_weight;
    } else {
        const auto &arc = std::get<Arc>(*edge.curve);
        for (const auto &a : {arc.center, arc.x, arc.y}) {
            t.trim_poles.push_back({a[0].center(), a[1].center(), 1});
            t.trim_intervals.push_back({{{a[0].lo, a[1].lo, 1}}, {{a[0].hi, a[1].hi, 1}}});
        }
        r.angle = arc.reversed ? arc.angles.hi : arc.angles.lo;
        r.last_angle = arc.reversed ? arc.angles.lo : arc.angles.hi;
        r.turn = (arc.reversed ? -1 : 1) * arc.angles.width();
        r.du = std::abs(r.turn) * (magnitude(arc.x[0]) + magnitude(arc.y[0]));
        r.dv = std::abs(r.turn) * (magnitude(arc.x[1]) + magnitude(arc.y[1]));
    }
    r.roundoff += 1e-13 * (1 + r.du + r.dv + length(edge.box.lo) + length(edge.box.hi));
    return r;
}
void append_tree(const Tree &tree, std::vector<device::Node> &out, int leaf_offset) {
    if (tree.nodes.empty())
        return;
    const int start = static_cast<int>(out.size());
    out.resize(out.size() + tree.nodes.size());
    auto visit = [&](auto &&self, int index, int next) -> void {
        const auto &n = tree.nodes[index];
        out[start + index] = {pack(n.box), n.leaf < 0 ? -1 : n.leaf + leaf_offset, start + next};
        if (n.leaf < 0) {
            self(self, n.left, n.right);
            self(self, n.right, next);
        }
    };
    visit(visit, 0, static_cast<int>(tree.nodes.size()));
}
void build_ray_grid(device::Tables &t, const std::vector<device::Bounds> &spans, double reach) {
    if (!(reach > 0) || t.faces.empty())
        return;
    const auto root = t.faces.front().box;
    double pad = 1e-6;
    for (const auto &face : t.charts)
        pad = std::max(pad, 4 * face.tol);
    double largest = 0;
    for (int k = 0; k < 3; ++k)
        largest = std::max(largest, root.hi.x[k] - root.lo.x[k]);
    const double width = std::max({reach, largest / 160., 8 * pad});
    auto &grid = t.ray_grid;
    const double halo = reach + pad;
    size_t cells = 1;
    for (int k = 0; k < 3; ++k) {
        grid.low.x[k] = root.lo.x[k] - halo;
        grid.inverse_width.x[k] = 1 / width;
        grid.size[k] =
            std::max(1, static_cast<int>(std::ceil((root.hi.x[k] + halo - grid.low.x[k]) / width)));
        cells *= grid.size[k];
    }
    // Large spatial incidences remain on the device BVH. Both indices use the same surface tests.
    if (cells > 3000000)
        return;
    std::vector<device::Bounds> boxes;
    auto insert = [&](int face, int node, device::Bounds box) {
        t.ray_primitives.push_back({face, node});
        boxes.push_back(box);
    };
    for (const auto &node : t.faces)
        if (node.leaf >= 0) {
            const int id = node.leaf;
            const auto &f = t.charts[id];
            if (f.first < 0)
                insert(id, -1, f.box);
            else
                for (int j = f.first; j < f.end; ++j)
                    if (t.spans[j].leaf >= 0)
                        insert(id, j, spans.at(t.spans[j].leaf));
        }
    auto visit = [&](const device::Bounds &box, auto emit) {
        int lo[3], hi[3];
        for (int k = 0; k < 3; ++k) {
            auto cell = [&](double v) {
                return static_cast<int>(std::clamp(std::floor((v - grid.low.x[k]) * grid.inverse_width.x[k]),
                                                   0., double(grid.size[k] - 1)));
            };
            lo[k] = cell(box.lo.x[k] - halo);
            hi[k] = cell(box.hi.x[k] + halo);
        }
        for (int x = lo[0]; x <= hi[0]; ++x)
            for (int y = lo[1]; y <= hi[1]; ++y)
                for (int z = lo[2]; z <= hi[2]; ++z)
                    emit((x * grid.size[1] + y) * grid.size[2] + z);
    };
    std::vector<int> counts(cells);
    size_t total = 0;
    for (const auto &box : boxes) {
        visit(box, [&](int i) {
            ++counts[i];
            ++total;
        });
        if (total > 16000000) {
            t.ray_primitives.clear();
            return;
        }
    }
    t.ray_offsets.resize(cells + 1);
    for (size_t i = 0; i < cells; ++i)
        t.ray_offsets[i + 1] = t.ray_offsets[i] + counts[i];
    t.ray_ids.resize(total);
    auto cursor = t.ray_offsets;
    for (size_t id = 0; id < boxes.size(); ++id)
        visit(boxes[id], [&](int i) { t.ray_ids[cursor[i]++] = static_cast<int>(id); });
    grid.reach = reach;
}
} // namespace
std::shared_ptr<device::Columns> cuda_columns(const Solid &solid, bool prepare_rules, double ray_reach_mm) {
    device::Tables t;
    std::vector<device::Bounds> span_bounds;
    append_tree(solid.tree, t.faces, 0);
    append_tree(solid.site_tree, t.site_tree, 0);
    for (auto &n : t.site_tree)
        if (n.leaf >= 0)
            n.leaf = solid.material_sites[n.leaf];
    for (const auto &node : solid.tile_tree.nodes)
        if (node.leaf >= 0)
            t.target_ids.push_back(node.leaf);
    for (const auto &tile : solid.tiles)
        t.tiles.push_back({pack(tile.box), pack(tile.site.point), pack(tile.site.normal), tile.site.uv[0],
                           tile.site.uv[1], tile.face});
    for (const auto &f : solid.faces) {
        device::FaceData face{};
        face.box = pack(f.box);
        face.first = -1;
        face.tol = std::max(1e-7, f.info.tolerance);
        face.orientation = f.info.orientation;
        face.domain = pack(f.info.domain);
        face.rectangle = f.info.rectangle;
        if (prepare_rules) {
            if (t.rule_sites.size() > static_cast<std::size_t>(std::numeric_limits<int>::max() - 720))
                throw std::length_error("resident CAD quadrature table is too large");
            face.rule_start = static_cast<int>(t.rule_sites.size());
            const auto &r = f.info.domain;
            for (int order : {12, 24}) {
                const auto base = t.rule_sites.size();
                t.rule_sites.resize(base + order * order);
                const auto rule = f.quadrature(order);
                for (const auto &site : rule.sites) {
                    auto i = std::lround((site.uv[0] - r.u.lo) / (r.u.width() / order) - .5);
                    auto j = std::lround((site.uv[1] - r.v.lo) / (r.v.width() / order) - .5);
                    if (i < 0 || i >= order || j < 0 || j >= order)
                        throw std::logic_error("invalid quadrature grid index");
                    t.rule_sites[base + i * order + j] = {pack(site.point), pack(site.normal), site.area};
                }
            }
        }
        face.trim_first = static_cast<int>(t.trim_tree.size());
        append_tree(f.trim_tree, t.trim_tree, static_cast<int>(t.trims.size()));
        for (const auto &edge : f.boundary)
            t.trims.push_back(trim_piece(edge, t));
        face.trim_end = static_cast<int>(t.trim_tree.size());
        if (f.analytic) {
            if (const auto *p = std::get_if<Plane>(f.analytic)) {
                face.kind = 1;
                face.origin = pack_midpoint(p->origin);
                face.frame[0] = pack_midpoint(p->x);
                face.frame[1] = pack_midpoint(p->y);
            } else if (const auto *p = std::get_if<Cylinder>(f.analytic)) {
                face.kind = 2;
                face.origin = pack_midpoint(p->origin);
                face.radius = p->radius.center();
                face.frame[0] = pack_midpoint(p->x);
                face.frame[1] = pack_midpoint(p->y);
                face.frame[2] = pack_midpoint(p->z);
            } else if (const auto *p = std::get_if<Revolved>(f.analytic)) {
                face.kind = 3;
                face.origin = pack_midpoint(p->origin);
                face.frame[0] = pack_midpoint(p->x);
                face.frame[1] = pack_midpoint(p->y);
                face.frame[2] = pack_midpoint(p->z);
                face.radius = p->radius.center();
                face.meridian_radius = p->meridian_radius.center();
                face.radial_slope = p->radial_slope.center();
                face.height_slope = p->height_slope.center();
            }
        }
        if (!f.analytic) {
            for (int k = 0; k < 3; ++k)
                face.frame[k] = pack(f.span_frame[k]);
            face.first = static_cast<int>(t.spans.size());
            append_tree(f.tree, t.spans, static_cast<int>(t.patches.size()));
            face.end = static_cast<int>(t.spans.size());
            face.knots_u = static_cast<int>(t.knots.size());
            face.nu = static_cast<int>(f.u.size());
            t.knots.insert(t.knots.end(), f.u.begin(), f.u.end());
            face.knots_v = static_cast<int>(t.knots.size());
            face.nv = static_cast<int>(f.v.size());
            t.knots.insert(t.knots.end(), f.v.begin(), f.v.end());
            face.grid = static_cast<int>(t.grid.size());
            for (auto span : f.grid)
                t.grid.push_back(span < 0 ? -1 : static_cast<int>(t.patches.size()) + span);
            for (const auto &p : f.patches) {
                if (ray_reach_mm > 0)
                    span_bounds.push_back(pack(p.box));
                device::Span s{};
                s.cell = -1;
                device::SpanFrame frame{};
                if (p.cubic && !p.cells.empty()) {
                    s.axis = pack(p.axis);
                    s.plane_width = p.far_plane - p.near_plane;
                    s.range = pack(p.full);
                    for (int i = 0; i < 16; ++i)
                        s.power[i] = pack(p.power[i]);
                    s.cell = static_cast<int>(t.cells.size());
                    s.count = static_cast<int>(p.cells.size());
                    for (const auto &c : p.cells)
                        t.cells.push_back({pack(c.bounds), pack(c.rect), pack(c.slope[0]), pack(c.slope[1]),
                                           pack(c.radius[0]), pack(c.radius[1]), pack(c.center)});
                    frame.origin = pack(p.anchor);
                    for (int k = 0; k < 3; ++k)
                        frame.axes[k] = pack(p.frame[k]);
                }
                t.frames.push_back(frame);
                t.patches.push_back(s);
            }
        }
        prepare_support(f,face);
        t.charts.push_back(face);
    }
    build_ray_grid(t, span_bounds, ray_reach_mm);
    return std::make_shared<device::Columns>(t);
}
} // namespace nc::compiled
