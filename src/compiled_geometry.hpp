#pragma once
#include "nurbscontact/compiled_part.hpp"
#include <map>
#include <numeric>

namespace nc::compiled {
using namespace dynamics;
using Param = std::array<double, 2>;
struct Placement {
    std::array<Vec, 3> cols;
    Vec offset{};
    static Placement between(const ContactPose &from, const ContactPose &to);
    Vec direction(Vec x) const {
        return {cols[0][0] * x[0] + cols[1][0] * x[1] + cols[2][0] * x[2],
                cols[0][1] * x[0] + cols[1][1] * x[1] + cols[2][1] * x[2],
                cols[0][2] * x[0] + cols[1][2] * x[1] + cols[2][2] * x[2]};
    }
    Vec point(Vec x) const {
        return add(offset, direction(x));
    }
};
struct Box {
    Vec lo{infinity, infinity, infinity}, hi{-infinity, -infinity, -infinity};
    void include(Vec p);
    void include(const Box &b) {
        include(b.lo);
        include(b.hi);
    }
    bool overlaps(const Box &b, double pad = 0) const;
    bool ray(Vec p, Vec d, double end, double pad) const;
    Box placed(const Placement &t) const;
};
struct SlabRay {
    Vec origin, inverse{};
    std::array<bool, 3> parallel{}, negative{};
    SlabRay(Vec p, Vec d) : origin(p) {
        for (int k = 0; k < 3; ++k) {
            parallel[k] = std::abs(d[k]) < 1e-18;
            if (!parallel[k])
                inverse[k] = 1 / d[k];
            negative[k] = d[k] < 0;
        }
    }
    bool intersects(const Box &box, double end, double pad) const {
        double near = -pad, far = end + pad;
        for (int k = 0; k < 3; ++k) {
            const double lo = box.lo[k] - origin[k] - pad, hi = box.hi[k] - origin[k] + pad;
            if (parallel[k]) {
                if (lo > 0 || hi < 0)
                    return false;
                continue;
            }
            near = std::max(near, (negative[k] ? hi : lo) * inverse[k]);
            far = std::min(far, (negative[k] ? lo : hi) * inverse[k]);
            if (near > far)
                return false;
        }
        return true;
    }
};
struct Tree {
    struct Node {
        Box box;
        int left = -1, right = -1, leaf = -1;
    };
    std::vector<Node> nodes;
    void build(const std::vector<Box> &boxes);
    template <class Visit> void visit(int k, const Box &box, Visit &f) const {
        if (!nodes[k].box.overlaps(box))
            return;
        if (nodes[k].leaf >= 0)
            f(nodes[k].leaf);
        else {
            visit(nodes[k].left, box, f);
            visit(nodes[k].right, box, f);
        }
    }
    void shortlist(int k, const Box &box, std::vector<int> &found, std::size_t limit) const {
        if (found.size() >= limit || !nodes[k].box.overlaps(box))
            return;
        const auto &node = nodes[k];
        if (node.leaf >= 0)
            found.push_back(node.leaf);
        else {
            shortlist(node.left, box, found, limit);
            shortlist(node.right, box, found, limit);
        }
    }
    template <class Visit> void ray(int k, const SlabRay &ray, double &end, double pad, Visit &f) const {
        auto &n = nodes[k];
        if (end < 0 || !ray.intersects(n.box, end, pad))
            return;
        if (n.leaf >= 0)
            f(n.leaf);
        else {
            this->ray(n.left, ray, end, pad, f);
            this->ray(n.right, ray, end, pad, f);
        }
    }
    template <class Visit> void ray(int k, Vec p, Vec d, double &end, double pad, Visit &f) const {
        ray(k, SlabRay{p, d}, end, pad, f);
    }
};
struct Jet {
    Vec p, u, v;
};
struct RayRoot {
    double t;
    Vec normal;
    Param uv;
    int face;
};
struct NodeSample {
    Vec point, normal;
    Param uv;
    double area;
};
struct Tile {
    NodeSample site;
    Box box;
    int face;
    bool interior;
};
struct Patch {
    Rect domain, full;
    int face{};
    int nu{}, nv{};
    std::vector<std::array<double, 4>> net;
    std::array<Vec, 16> power{};
    bool cubic{};
    Box box;
    Vec axis{}, anchor{};
    std::array<Vec, 3> frame{};
    Box frame_box;
    double near_plane{}, far_plane{};
    struct Cell {
        Rect rect;
        Box bounds;
        Vec slope[2], radius[2], center;
    };
    std::vector<Cell> cells;
    void prepare_cells();
    Jet at(Param uv) const;
    std::vector<RayRoot> roots(Vec p, Vec d, double end, double tol) const;
};
struct Rule {
    std::vector<NodeSample> sites;
    struct Block {
        Box box;
        Vec normal_lo, normal_hi;
        std::size_t first, end;
    };
    std::vector<Block> blocks;
};
struct Face {
    FaceInfo info;
    const Surface *analytic{};
    const TrimDomain *trim{};
    struct TrimSpan {
        const Curve *curve;
        Box box;
        Interval first_y, last_y;
    };
    std::vector<TrimSpan> boundary;
    Tree trim_tree;
    std::vector<Patch> patches;
    std::vector<double> u, v;
    std::vector<int> grid;
    Tree tree;
    std::array<Vec, 3> span_frame{};
    Vec in_frame(Vec x) const {
        return {dot3(span_frame[0], x), dot3(span_frame[1], x), dot3(span_frame[2], x)};
    }
    Box box;
    const Patch *patch(Param uv) const;
    Jet at(Param uv) const;
    bool inside(Param uv) const;
    void roots(Vec p, Vec d, double end, double tol, int id, std::vector<RayRoot> &out) const;
    void span_roots(int span, Vec p, Vec d, double end, double tol, std::vector<RayRoot> &out) const;
    Rule quadrature(int order) const;
};
class Solid {
    PartFile file_;

  public:
    std::vector<Face> faces;
    Tree tree;
    std::vector<Tile> tiles;
    std::vector<int> material_sites;
    Tree site_tree, tile_tree;
    explicit Solid(PartFile, DiscoverySites = DiscoverySites::face_grid);
    struct BlockFace {
        int face;
        std::vector<int> spans;
    };
    struct ColumnBlock {
        std::vector<BlockFace> candidates;
    };
    ColumnBlock column_block(const Box &swept) const;
    std::optional<RayRoot> first(Vec p, Vec d, double end, const ColumnBlock &block) const;
    std::optional<RayRoot> first(Vec p, Vec d, double end) const;
    std::optional<RayRoot> crossing(Vec a, Vec b) const;
};
} // namespace nc::compiled
