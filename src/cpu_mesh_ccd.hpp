#pragma once

#include "nurbscontact/cpu_mesh.hpp"

namespace nc::cpu {

struct TriangleTime {
    double time{nc::infinity};
    double output_tolerance{};
};

// Pure translations allow exact time clipping on a fixed set of separating axes.
TriangleTime translated_triangle_ccd(const Triangle& a, const Triangle& b,
                                     Vec a0, Vec a1, Vec b0, Vec b1);
std::pair<double,double> squared_distance_bounds(const Triangle& a, const Triangle& b);

// Earliest conservative candidate for two linearly moving triangles on [0,1].
TriangleTime linear_triangle_ccd(const Triangle& a0, const Triangle& a1,
                                 const Triangle& b0, const Triangle& b1,
                                 double margin, double distance_tolerance);
}
