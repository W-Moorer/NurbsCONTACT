#pragma once

#include "nurbscontact/cpu_mesh.hpp"

namespace nc::cpu {
struct WindowTriangle {
    dynamics::Triangle triangle;
    Vec lo, hi;
    std::size_t face;
    int side;
};
std::size_t window_pairs(std::span<const WindowTriangle> candidates, double pad,
                         std::vector<std::array<std::size_t,2>>& pairs,const TrianglePairFilter& filter);
}
