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
struct ChartPoint {
    int face;
    double u, v;
};
struct ChartSample {
    V point, normal;
    int material, defined;
};
struct Hit {
    double t;
    int face;
    double u{}, v{};
    V normal{};
}; // -1 is clear, -2 requests the general compiled solver.
struct Node {
    Bounds box;
    int leaf, next;
};
struct FaceData {
    Bounds box;
    V frame[3], origin;
    int first, end, orientation, kind;
    double tol, radius, meridian_radius{}, radial_slope{}, height_slope{1};
    Range domain{};
    int trim_first{}, trim_end{}, grid{}, knots_u{}, knots_v{}, nu{}, nv{};
    bool rectangle{};
    int rule_start{-1};
};
struct Span {
    V axis;
    double plane_width;
    V power[16];
    Range range;
    int cell;
    int count;
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
    double u, v;
    int face;
};
struct TrimPiece {
    Bounds box;
    Bounds endpoints[2];
    double first_y, last_y, du, dv, roundoff;
    int start, count;
    double angle, turn, last_angle;
};
struct Pole {
    double x, y, w;
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
struct WeightedSite {
    V point, normal;
    double area;
};
struct RayPrimitive {
    int face, span_node;
};
struct RayGrid {
    V low, inverse_width;
    int size[3]{};
    double reach{};
    const int *offsets, *ids;
    const RayPrimitive *primitives;
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
    RayGrid ray_grid{};
    std::vector<int> ray_offsets, ray_ids;
    std::vector<RayPrimitive> ray_primitives;
    std::vector<Node> faces, spans;
    std::vector<FaceData> charts;
    std::vector<Span> patches;
    std::vector<Cell> cells;
    std::vector<SpanFrame> frames;
    std::vector<Node> site_tree;
    std::vector<Tile> tiles;
    std::vector<int> target_ids;
    std::vector<Node> trim_tree;
    std::vector<TrimPiece> trims;
    std::vector<Pole> trim_poles;
    std::vector<Bounds> trim_intervals;
    std::vector<double> knots;
    std::vector<int> grid;
    std::vector<WeightedSite> rule_sites;
};
class Columns {
  public:
    struct Storage;
    const Storage &storage() const { return *data; }
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
    // Profile fixed-size resident batches using the same material-exit kernel as trace().
    double time_ray_batches(const std::vector<Ray> &, double reach, int batch, int repeats, std::vector<Hit> &);
    std::vector<ChartSample> sample_charts(const std::vector<ChartPoint> &) const;

  private:
    Storage *data;
};
struct AxisCoordinate {
    double mass, q, v, k, c, rest, load;
    int first, end;
};
struct LoadPoint {
    double t, value;
};
struct JointBody {
    V origin, axis, pivot;
    double angle;
    int hinge, translation;
    V direction;
};
struct ContactLaw {
    int a, b;
    double k, exponent, rate, max_depth;
    bool shared;
};
struct ResidentScene {
    std::vector<AxisCoordinate> coordinates;
    std::vector<LoadPoint> loads;
    std::vector<JointBody> bodies;
    std::vector<ContactLaw> pairs;
    std::vector<std::shared_ptr<Columns>> geometry;
};
enum class RegionReduction { serial, ordered, warp, audit, audit_warp };
struct RegionAudit {
    unsigned queries{}, regions{}, entered_faces{}, exited_faces{}, count_changes{};
    double force_error[3]{}, torque_a_error[3]{}, torque_b_error[3]{};
};
struct ResidentResult {
    std::vector<double> frames;
    std::vector<RegionAudit> region_audit;
    double reduction_s{};
    unsigned long long reductions{};
    double online_s{}, maximum_residual{};
    unsigned long long queries{}, forces{}, splits{}, missed_deadlines{};
    double phase_s[5]{}, p50_s{}, p99_s{}, maximum_step_s{};
};
ResidentResult advance_resident(const ResidentScene &, double step, int steps, RegionReduction = RegionReduction::ordered);
} // namespace nc::compiled::device
