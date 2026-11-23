#pragma once

#include "nurbscontact/geometry.hpp"

namespace nc {

enum class BezierLayout { tensor, quadratic_triangle };
// Affine trajectories of Cartesian poles with constant positive weights.
// Triangle order is W², WU, U², WV, UV, V² in barycentric coordinates.
struct MovingBezier {
    std::size_t nu{}, nv{};
    std::vector<HPoint> first, last;
    BezierLayout layout=BezierLayout::tensor;
};

enum class TdibStatus { clear, candidate, unresolved };
struct TdibResult {
    TdibStatus status;
    Interval time;
    std::size_t nodes;
};

// margin bounds the combined spatial departure from the two affine patches.
// A terminal candidate is an inclusion cell, not an intersection witness.
TdibResult tdib_envelopes(const MovingBezier& a,const MovingBezier& b,
                          double width,double horizon,double margin,std::size_t node_limit);

} // namespace nc
