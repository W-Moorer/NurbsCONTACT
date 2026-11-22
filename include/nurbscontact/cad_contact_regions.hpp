#pragma once

#include "nurbscontact/contact_dynamics.hpp"
#include "nurbscontact/occt.hpp"

namespace nc {

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
};

} // namespace nc
