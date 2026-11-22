#pragma once

#include "nurbscontact/contact_dynamics.hpp"
#include "nurbscontact/occt.hpp"
#include "nurbscontact/query.hpp"

namespace nc {

// The source shape is in millimeters. The contact surface is queried in meters.
// A positive deflection selects an OCCT tessellation and the triangle BVH.
std::shared_ptr<const dynamics::ContactSurface> contact_surface(
    const TopoDS_Shape& shape, double deflection_m = 0);

enum class PairBackend { occt, mesh, brep };

struct PairSearchOptions {
    double tolerance_m{2e-6};
    std::size_t max_nodes{30'000};
};

class PairGeometry;
std::shared_ptr<const PairGeometry> pair_geometry(const TopoDS_Shape& shape, PairBackend backend,
                                                  double deflection_m);
std::shared_ptr<const dynamics::PairContactQuery> pair_contact_query(
    std::shared_ptr<const PairGeometry> a, std::shared_ptr<const PairGeometry> b,
    PairSearchOptions options = {});

// A's rotation is constant. The origins interpolate affinely in world space.
// B may rotate about its fixed axis. Time is normalized to [0, 1].
SweepResult fixed_axis_pair_sweep(const PairGeometry& a, const PairGeometry& b,
                                 dynamics::ContactPose a0, dynamics::ContactPose a1,
                                 dynamics::ContactPose b0, dynamics::ContactPose b1,
                                 double clearance_m, double time_tolerance,
                                 PairSearchOptions options = {});

std::vector<dynamics::Triangle> tessellate_occt(const TopoDS_Shape& shape, double deflection_mm);

} // namespace nc
