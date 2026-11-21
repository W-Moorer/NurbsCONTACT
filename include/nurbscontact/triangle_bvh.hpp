#pragma once

#include "nurbscontact/interval.hpp"

#include <optional>
#include <vector>

namespace nc::dynamics {

inline Vec add(Vec a, Vec b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
inline Vec sub(Vec a, Vec b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
inline Vec scale(Vec a, double x) { return {a[0] * x, a[1] * x, a[2] * x}; }
inline double dot3(Vec a, Vec b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
inline Vec cross3(Vec a, Vec b) {
    return {a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]};
}
inline double squared(Vec a) { return dot3(a, a); }
inline double length(Vec a) { return std::sqrt(squared(a)); }

struct Pose {
    Vec origin{};
    double cosine{1};
    double sine{};
    Vec rotate(Vec p) const { return {cosine*p[0] - sine*p[1], sine*p[0] + cosine*p[1], p[2]}; }
    Vec inverse_rotate(Vec p) const { return {cosine*p[0] + sine*p[1], -sine*p[0] + cosine*p[1], p[2]}; }
    Vec point(Vec p) const { return add(origin, rotate(p)); }
};

Pose relative_pose(Pose a, Pose b);

using Triangle = std::array<Vec, 3>;

struct TriangleHint {
    std::size_t a{};
    std::size_t b{};
    bool valid{};
};

struct TriangleStatistics {
    std::size_t nodes{};
    std::size_t triangles{};
};

struct PointPair {
    Vec a;
    Vec b;
    double distance{};
};

PointPair closest_triangles(const Triangle& a, const Triangle& b);

class TriangleBvh {
public:
    explicit TriangleBvh(std::vector<Triangle> triangles, std::size_t leaf_size = 4);
    const std::vector<Triangle>& triangles() const { return triangles_; }

    // Output points are expressed in A's frame. The cutoff merely limits search.
    std::optional<PointPair> closest_within(const TriangleBvh& b, Pose b_in_a, double cutoff,
                                            TriangleHint* hint, TriangleStatistics& statistics) const;
    // Parameter of the first two-sided hit and, if requested, its unnormalized face normal.
    std::optional<double> ray_first(Vec origin, Vec direction, double maximum,
                                    TriangleStatistics& statistics, Vec* hit_normal = nullptr) const;

    struct Box { Vec lo; Vec hi; };

private:
    struct Node {
        Box box;
        std::size_t begin{};
        std::size_t end{};
        std::size_t first{};
        std::size_t second{};
        bool leaf() const { return first == second; }
    };
    std::size_t build(std::size_t begin, std::size_t end, std::size_t leaf_size);

    std::vector<Triangle> triangles_;
    std::vector<Box> triangle_boxes_;
    std::vector<std::size_t> ids_;
    std::vector<Node> nodes_;
};

}  // namespace nc::dynamics
