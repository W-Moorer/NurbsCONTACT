#pragma once

#include "cad_boundary_distance.hpp"

namespace nc::cad {
struct RigidPath;
// Apply the original OCCT force-region construction to two already placed solids.
std::vector<dynamics::ContactRegion> solid_regions(const TopoDS_Shape& first,const TopoDS_Shape& second,
                                                  dynamics::ContactWork& work,bool measure_span=true,
                                                  std::vector<Vec>* pressure_moment_arms=nullptr,
                                                  bool parallel_oriented_boxes=false);

struct LocalSolidGeometry {
    TopoDS_Shape shape;
    std::shared_ptr<const BoundaryIndex> boundary;
};

struct LocalSolidPair {
    LocalSolidGeometry a,b;
    bool separated{};
};

class LocalSolidSearch {
public:
    LocalSolidSearch(const TopoDS_Shape& a,const TopoDS_Shape& b);
    // Both closest points within reach, and all common volume, are retained.
    LocalSolidPair select(const gp_Trsf& a_in_b,double reach) const;
    double motion_bound(const RigidPath& a,const RigidPath& b,double global_speed,double cap) const;
private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

}
