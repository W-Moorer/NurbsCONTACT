#pragma once

#include "nurbscontact/contact_dynamics.hpp"
#include "nurbscontact/occt.hpp"

namespace nc {
namespace cad { class BezierRayAtlas; }

struct CadFaceWitness {
    std::size_t face{}; // One-based face index from TopExp::MapShapes.
    FeatureKind support{};
    std::optional<std::array<double, 2>> uv;
};

struct CadContactCandidate {
    CadFaceWitness a, b;
    dynamics::PointPair points; // Coordinates in A's frame, in meters.
    bool facing_interiors{};
};

struct CadContactRegions {
    std::vector<CadContactCandidate> candidates;
    std::size_t tested_face_pairs{};
    std::size_t unresolved_face_pairs{};
    std::size_t reused_face_exclusions{};
    std::size_t attempted_face_exclusions{};
};

struct CadContactSeed {
    std::size_t face_a{}, face_b{};
    std::array<double, 2> uv_a{}, uv_b{};
};

struct CadContactLine {
    std::size_t face_a{}, face_b{};
    std::vector<dynamics::ContactLineSample> samples;
};

struct CadContactLines {
    std::vector<CadContactLine> lines;
    std::size_t unresolved_face_pairs{};
};

// Owns the geometric separation records for one pair of prepared shapes.
class CadContactCache {
public:
    CadContactCache();
    ~CadContactCache();

private:
    struct Data;
    std::unique_ptr<Data> data_;
    friend class CadContactIndex;
};

class CadContactIndex {
public:
    explicit CadContactIndex(const TopoDS_Shape& shape);
    std::size_t face_count() const;
    // Lower and upper corners in the body's local frame, in meters and face-ID order.
    std::vector<std::array<Vec,2>> face_bounds() const;

    // Enumerates numerical minima of candidate face pairs. Boundary witnesses
    // and ties are retained for later contact-region assembly.
    // With a cache, skip only the face pairs whose projected boxes still prove separation.
    CadContactRegions query(const CadContactIndex& b, dynamics::ContactPose b_in_a, double cutoff_m,
                            CadContactCache* cache = nullptr) const;
    // An external broad phase can supply a conservative row-major mask [B face][A face].
    CadContactRegions query(const CadContactIndex& b, dynamics::ContactPose b_in_a, double cutoff_m,
                            CadContactCache* cache, std::span<const unsigned char> candidates) const;

    // Updates known face-interior candidates. Each seed is reclassified on
    // its trimmed face. A seed that cannot be recovered is counted unresolved.
    CadContactRegions track_interiors(const CadContactIndex& b, dynamics::ContactPose b_in_a, double cutoff_m,
                                     std::span<const CadContactSeed> seeds) const;

    // Reconstructs sampled straight contact lines from facing interior and
    // end witnesses. The tolerance applies to the numeric CAD evaluation.
    CadContactLines straight_lines(const CadContactIndex& b, dynamics::ContactPose b_in_a, double cutoff_m,
                                   double tolerance_m, const CadContactRegions& regions) const;

private:
    struct Data;
    std::shared_ptr<const Data> data_;
    friend class CadExitQuery;
};

struct CadExit {
    std::size_t face{}; // One-based face ID. Zero denotes an unresolved exit.
    double distance_m{};
    Vec outward{};
    std::array<double,2> uv{};
};

// Query-local mutable intersectors over the immutable trimmed-face tree.
// Coordinates and unit direction are in the body's local frame.
class CadExitQuery {
public:
    explicit CadExitQuery(const CadContactIndex& index, std::shared_ptr<const cad::BezierRayAtlas> bezier = {});
    ~CadExitQuery();
    CadExit first_exit(Vec origin, Vec direction, double reach_m, std::size_t hint,
                       std::size_t& face_tests, std::span<const std::size_t> required_faces = {});
    // Both endpoints must already be strictly interior to this solid.
    bool interior_segment(Vec a,Vec b,const CadExit* near_a=nullptr,const CadExit* near_b=nullptr);
    bool crosses_from_hints(Vec a,Vec b,const CadExit& near_a,const CadExit& near_b);
    Vec surface_point(std::size_t face,std::array<double,2> uv) const;
    std::array<std::size_t,4> ray_counts() const;
private:
    struct Data;
    std::unique_ptr<Data> data_;
};

struct CadChord {
    std::array<CadExit,2> exits;
    dynamics::ContactRegion region;
};

// One query instance per worker. Hints accelerate traversal but do not decide
// which faces can intersect. The two solids must have oriented closed boundaries.
std::optional<CadChord> local_chord(CadExitQuery& a, CadExitQuery& b,
    const std::array<dynamics::ContactPose,2>& poses, Vec point, Vec normal,
    double reach_m, std::array<std::size_t,2> hints, std::size_t& face_tests);

// A positive answer joins two material witnesses by an interior polyline in
// both original solids. Failure to find a path leaves the witnesses separate.
struct ChordPath {
    std::vector<Vec> intermediate_in_a;
    bool reused{};
};
bool connected_chords(CadExitQuery& a,CadExitQuery& b,const std::array<dynamics::ContactPose,2>& poses,
                      const CadChord& x,const CadChord& y,double reach_m,ChordPath* previous=nullptr);
std::vector<int> chord_groups(CadExitQuery& a,CadExitQuery& b,const std::array<dynamics::ContactPose,2>& poses,
                              std::span<const std::optional<CadChord>> witnesses,double reach_m);

} // namespace nc
