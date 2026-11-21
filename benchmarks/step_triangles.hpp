#pragma once

#include "nurbscontact/occt_contact.hpp"

namespace nc::bench {

inline std::vector<dynamics::Triangle> step_triangles(const TopoDS_Shape& shape, double deflection) {
    return tessellate_occt(shape,deflection);
}

}  // namespace nc::bench
