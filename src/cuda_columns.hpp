#pragma once
#include <cstddef>
#include <memory>
#include <vector>

namespace nc::compiled::device {
struct V {
    double x[3];
};
struct Bounds {
    V lo, hi;
};
struct Range {
    double u, U, v, Vv;
};
struct Ray {
    V p, d;
};
struct Hit {
    double t;
    int face;
}; // -1 is clear, -2 requests the general compiled solver.
struct Node {
    Bounds box;
    int leaf, next;
};
struct FaceData {
    Bounds box;
    V frame[3], origin;
    int first, end, orientation, kind;
    double tol, radius;
};
struct Span {
    V power[16];
    Range range;
    int cell;
};
struct Cell {
    Bounds box;
    Range range;
    V su, sv, ru, rv, center;
};
struct SpanFrame {
    V origin, axes[3];
};
struct Tile {
    Bounds box;
    V point, normal;
};
struct Pose {
    V cols[3], offset;
};
struct Candidate {
    int site, tile;
    double score;
};
struct RuleSite {
    V point, normal;
};
struct RuleBlock {
    const RuleSite *sites;
    int first, count, offset, side, targets, end;
};
struct Depth {
    double value;
    int job, local;
};
struct Sites {
    explicit Sites(const std::vector<RuleSite> &);
    ~Sites();
    Sites(const Sites &) = delete;
    Sites &operator=(const Sites &) = delete;
    RuleSite *ptr{};
};
struct Tables {
    std::vector<Node> faces, spans;
    std::vector<FaceData> charts;
    std::vector<Span> patches;
    std::vector<Cell> cells;
    std::vector<SpanFrame> frames;
    std::vector<Node> site_tree;
    std::vector<Tile> tiles;
    std::vector<int> target_ids;
};
class Columns {
  public:
    struct Window;
    std::shared_ptr<Window> window(const Columns &target, Pose, double reach, double turn, size_t &bytes);
    void candidates(const Columns &target, const Window &, Pose, double reach, std::vector<Candidate> &out);
    size_t integrate(const Columns &other, const std::vector<RuleBlock> &blocks,
                     const std::vector<int> &targets, Pose forward, Pose reverse, double reach,
                     std::vector<Depth> &out);
    explicit Columns(const Tables &);
    ~Columns();
    Columns(const Columns &) = delete;
    Columns &operator=(const Columns &) = delete;
    void trace(const std::vector<Ray> &, double reach, std::vector<Hit> &);

  private:
    struct Storage;
    Storage *data;
};
} // namespace nc::compiled::device
