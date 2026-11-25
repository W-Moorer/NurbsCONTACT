#pragma once
#include "nurbscontact/occt_contact.hpp"

namespace nc {
enum class CadEventMethod { conservative_advancement, chen_2024, chen_2026 };
// Input solids use STEP millimetres. Poses and local rotation centers use metres.
std::shared_ptr<const dynamics::PairRegionQuery> cad_solid_region_query(
    const TopoDS_Shape& a, const TopoDS_Shape& b, std::array<Vec,2> centers, CadEventMethod events);
// Native connected overlap components with pointwise trimmed-STEP depth.
std::shared_ptr<const dynamics::PairRegionQuery> cad_local_region_query(
    const TopoDS_Shape& a, const TopoDS_Shape& b, std::array<Vec,2> centers, double maximum_depth_m);
std::shared_ptr<const dynamics::PairRegionQuery> cad_patch_region_query(
    const TopoDS_Shape& a,const TopoDS_Shape& b,std::array<Vec,2> centers,double maximum_depth_m);
inline std::shared_ptr<const dynamics::PairRegionQuery> cad_solid_region_query(
    const TopoDS_Shape& a, const TopoDS_Shape& b, std::array<Vec,2> centers, bool chen_events) {
    return cad_solid_region_query(a,b,centers,chen_events ? CadEventMethod::chen_2024 : CadEventMethod::conservative_advancement);
}
}
