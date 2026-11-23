#pragma once

#include "nurbscontact/cpu_mesh.hpp"

namespace nc::cpu {
bool triangle_overlap(const Triangle& a,const Triangle& b);

// One instance per body pair. Exact witness coordinates join triangle hits
// across mesh edges without a user-selected spatial merging distance.
class TriangleContours {
public:
    TriangleContours();
    ~TriangleContours();
    std::vector<Vec> intersect(const Triangle& a, const Triangle& b);
    std::vector<IntersectionResult::Contour> finish(std::span<const Intersection> hits);
private:
    struct Data;
    std::unique_ptr<Data> data_;
};

}
