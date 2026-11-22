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
};

struct CadContactSeed {
    std::size_t face_a{}, face_b{};
    std::array<double, 2> uv_a{}, uv_b{};
};

class CadContactIndex {
public:
    explicit CadContactIndex(const TopoDS_Shape& shape);
    std::size_t face_count() const;

    // Enumerates numerical minima of candidate face pairs. Boundary witnesses
    // and ties are retained for later contact-region assembly.
    CadContactRegions query(const CadContactIndex& b, dynamics::ContactPose b_in_a, double cutoff_m) const;

    // Updates known face-interior candidates. Each seed is reclassified on
    // its trimmed face. A seed that cannot be recovered is counted unresolved.
    CadContactRegions track_interiors(const CadContactIndex& b, dynamics::ContactPose b_in_a, double cutoff_m,
                                     std::span<const CadContactSeed> seeds) const;

private:
    struct Data;
    std::shared_ptr<const Data> data_;
};

} // namespace nc
