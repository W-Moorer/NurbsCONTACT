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
    // Conservative optional support data for rejecting separated, opposing charts.
    V support_axes[3]{};
    Bounds support_box{}, normal_box{};
    bool support_ready{};
};
struct Span {
    V axis;
    double plane_width;
    V power[16];
    V coefficient_magnitude;
    Range range;
    int cell;
    int count;
    int filter_first{-1}, filter_end{};
};
struct Cell {
    Bounds box;
    Range range;
    V su, sv, ru, rv, center;
    double tangent_scale;
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
// A preorder of the face tree with each polynomial face's span tree embedded.
// frame < 0 uses part coordinates; otherwise box is in that face's OBB frame.
struct SegmentNode {
    Bounds box;
    int face, span, frame, next; // span: -2 analytic, -1 branch, >= 0 polynomial
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
// Incidence comes from the saved CAD topology, never from spatial proximity.
struct BoundaryLink {
    int other, first, end;
    std::size_t edge, start_vertex, end_vertex;
};
struct BoundaryProbe { V point, normal; };
struct Sites {
    explicit Sites(const std::vector<RuleSite> &);
    ~Sites();
    Sites(const Sites &) = delete;
    Sites &operator=(const Sites &) = delete;
    RuleSite *ptr{};
};
struct Tables {
    std::vector<int> boundary_offsets;
    std::vector<BoundaryLink> boundary_links;
    std::vector<BoundaryProbe> boundary_probes;
    RayGrid ray_grid{};
    std::vector<int> ray_offsets, ray_ids;
    std::vector<RayPrimitive> ray_primitives;
    std::vector<SegmentNode> segment_nodes;
    std::vector<Node> faces, spans;
    std::vector<Node> curvature_nodes;
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
    // Precomputed-only queries return face == -3 when complete GPU solving is required.
    void trace(const std::vector<Ray> &, double reach, std::vector<Hit> &, bool precomputed_only = false);
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
enum class ResidentExecution { cooperative, cached, batched, wavefront, coherent };
struct RegionAudit {
    unsigned queries{}, regions{}, entered_faces{}, exited_faces{}, count_changes{};
    double force_error[3]{}, torque_a_error[3]{}, torque_b_error[3]{};
};
struct ResidentResult {
    std::vector<double> frames;
    std::vector<RegionAudit> region_audit;
    double reduction_s{};
    unsigned long long reductions{}, checked_links{};
    double online_s{}, device_s{}, download_s{}, maximum_residual{};
    unsigned long long queries{}, forces{}, splits{}, missed_deadlines{};
    double phase_s[5]{}, p50_s{}, p99_s{}, maximum_step_s{};
    unsigned long long accepted_by_level[9]{};
    unsigned long long proposals{}, face_groups{}, confirmed_groups{}, seed_hits{};
    unsigned maximum_proposals{}, maximum_groups{};
    unsigned long long blocker_checks{}, blocker_hits{};
    unsigned long long warp_segments{}, scalar_segments{};
    unsigned long long separated_groups{};
    unsigned long long path_checks{}, path_hits{};
    unsigned long long batched_rays{}, ray_batches{}, precomputed_rays{}, deferred_rays{}, audited_rays{}, audited_groups{};
    unsigned optimizations{};
    unsigned long long projection_rounds{}, avoided_projections{}, unused_projections{}, projection_culls{};
    unsigned long long topology_checks{}, topology_hits{}, warm_attempts{}, warm_hits{}, warm_fallbacks{};
    unsigned long long parameter_hint_calls{}, parameter_hint_hits{}, projection_attempts{}, rejected_rule_blocks{}, tested_rule_blocks{}, local_links{}, remote_links{};
    bool fused_math{}, profiled_links{};
    double link_work_s[4]{};
    unsigned long long link_calls[4]{};
};
ResidentResult advance_resident(const ResidentScene &, double step, int steps,
                                RegionReduction = RegionReduction::ordered,
                                ResidentExecution = ResidentExecution::cooperative,
                                bool profile_links = false, unsigned optimizations = 15);
} // namespace nc::compiled::device
