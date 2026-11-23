#pragma once

#include "nurbscontact/contact_dynamics.hpp"
#include <TopoDS_Shape.hxx>
#include <gp_Trsf.hxx>

namespace nc::cad {

class BoundaryIndex;
std::shared_ptr<const BoundaryIndex> boundary_index(const TopoDS_Shape& solid);

// Numerical minimum solid distance in STEP units, capped at the requested
// horizon. Distances below contact_tol may be returned as zero.
double bounded_distance(const BoundaryIndex& a,const BoundaryIndex& b,const gp_Trsf& a_in_b,
                        double cap,double contact_tol,dynamics::SurfaceTransition& stats);
}
