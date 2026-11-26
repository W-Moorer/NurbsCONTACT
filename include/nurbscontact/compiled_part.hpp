#pragma once
#include "nurbscontact/geometry.hpp"
#include "nurbscontact/contact_dynamics.hpp"
#include <filesystem>

namespace nc::compiled {
// Model coordinates and tolerances are millimetres; the dynamics API uses metres.
struct FaceInfo {
    Rect domain;
    Vec lo, hi;
    double tolerance{};
    int orientation{};
    bool rectangle{};
};
struct PartFile {
    Model geometry;
    std::vector<FaceInfo> faces;
};
void save(const PartFile &, const std::filesystem::path &);
PartFile load(const std::filesystem::path &);
class Solid;
enum class DiscoverySites { face_grid, audit_chart_edges };
std::shared_ptr<const Solid> load_solid(const std::filesystem::path &,
                                        DiscoverySites = DiscoverySites::face_grid);
using GroupFaces = std::vector<std::vector<std::array<int, 2>>>;
class RegionQuery : public dynamics::PairRegionQuery {
  public:
    // Diagnostic output uses the one-based STEP face IDs of each loaded region.
    virtual std::vector<dynamics::ContactRegion> with_faces(dynamics::ContactPose a, dynamics::ContactPose b,
                                                            dynamics::ContactWork &work,
                                                            GroupFaces &groups) const = 0;
};
std::shared_ptr<const RegionQuery> region_query(std::shared_ptr<const Solid>, std::shared_ptr<const Solid>,
                                                double maximum_depth_m);
} // namespace nc::compiled
