#pragma once

#include "cad_patch_assets.hpp"
#include "nurbscontact/cad_contact_regions.hpp"
#include <gp_XYZ.hxx>

namespace nc::cad {
// All ray coordinates are in the rigid body's local frame, in millimetres.
// A resolved empty result excludes the entire supporting surface on this ray.
struct BezierRayHit {
    double t, u, v;
    gp_XYZ normal;
};
class BezierRayAtlas {
  public:
    BezierRayAtlas(const TopoDS_Shape &, PatchAssets &, bool relocate=false);
    bool relocates() const { return relocate_; }
    bool query(std::size_t face, const gp_XYZ &point, const gp_XYZ &unit_direction, double end,
               double tolerance, std::optional<BezierRayHit> &hit,
               std::size_t *candidate_spans = nullptr,const std::array<double,2>* previous_uv=nullptr) const;

  private:
    struct Span;
    struct Part;
    std::shared_ptr<const Part> part_;
    bool relocate_;
};
} // namespace nc::cad
