#include "compiled_geometry.hpp"
#include "cuda_columns.hpp"

namespace nc::compiled {
namespace {
device::V pack(Vec a) { return {{a[0], a[1], a[2]}}; }
device::V pack_midpoint(const IVec &a) { return {{a[0].center(), a[1].center(), a[2].center()}}; }
device::Bounds pack(Box a) { return {pack(a.lo), pack(a.hi)}; }
device::Range pack(Rect a) { return {a.u.lo, a.u.hi, a.v.lo, a.v.hi}; }
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
} // namespace
std::shared_ptr<device::Columns> cuda_columns(const Solid &solid) {
    device::Tables t;
    append_tree(solid.tree, t.faces, 0);
    append_tree(solid.site_tree, t.site_tree, 0);
    for (auto &n : t.site_tree)
        if (n.leaf >= 0)
            n.leaf = solid.material_sites[n.leaf];
    for (const auto &node : solid.tile_tree.nodes)
        if (node.leaf >= 0)
            t.target_ids.push_back(node.leaf);
    for (const auto &tile : solid.tiles)
        t.tiles.push_back({pack(tile.box), pack(tile.site.point), pack(tile.site.normal)});
    for (const auto &f : solid.faces) {
        device::FaceData face{};
        face.box = pack(f.box);
        face.first = -1;
        face.tol = std::max(1e-7, f.info.tolerance);
        face.orientation = f.info.orientation;
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
            }
        }
        if (!f.analytic && f.info.rectangle) {
            for (int k = 0; k < 3; ++k)
                face.frame[k] = pack(f.span_frame[k]);
            face.first = static_cast<int>(t.spans.size());
            append_tree(f.tree, t.spans, static_cast<int>(t.patches.size()));
            face.end = static_cast<int>(t.spans.size());
            for (const auto &p : f.patches) {
                device::Span s{};
                s.cell = -1;
                device::SpanFrame frame{};
                if (p.cubic && !p.cells.empty() && p.full.u.lo >= f.info.domain.u.lo - 1e-10 &&
                    p.full.u.hi <= f.info.domain.u.hi + 1e-10 && p.full.v.lo >= f.info.domain.v.lo - 1e-10 &&
                    p.full.v.hi <= f.info.domain.v.hi + 1e-10) {
                    s.range = pack(p.full);
                    for (int i = 0; i < 16; ++i)
                        s.power[i] = pack(p.power[i]);
                    s.cell = static_cast<int>(t.cells.size());
                    const auto &c = p.cells.front();
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
        t.charts.push_back(face);
    }
    return std::make_shared<device::Columns>(t);
}
} // namespace nc::compiled
