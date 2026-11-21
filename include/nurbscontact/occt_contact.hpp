#pragma once

#include "nurbscontact/contact_dynamics.hpp"
#include "nurbscontact/occt.hpp"

namespace nc {

// The source shape is in millimeters. The contact surface is queried in meters.
// A positive deflection selects an OCCT tessellation and the triangle BVH.
std::shared_ptr<const dynamics::ContactSurface> contact_surface(
    const TopoDS_Shape& shape, double deflection_m = 0);

std::vector<dynamics::Triangle> tessellate_occt(const TopoDS_Shape& shape, double deflection_mm);

} // namespace nc
