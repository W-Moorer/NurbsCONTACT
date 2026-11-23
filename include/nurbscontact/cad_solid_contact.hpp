#pragma once
#include "nurbscontact/occt_contact.hpp"

namespace nc {
// Input solids use STEP millimetres. Poses and local rotation centers use metres.
std::shared_ptr<const dynamics::PairRegionQuery> cad_solid_region_query(
    const TopoDS_Shape& a, const TopoDS_Shape& b, std::array<Vec,2> centers, bool chen_events);
}
