#pragma once

#include "nurbscontact/triangle_bvh.hpp"

#include <filesystem>
#include <memory>
#include <span>
#include <string_view>

namespace nc::cpu {

using dynamics::RigidPlacement;
using dynamics::Triangle;
using nc::Vec;

enum class MeshMethod { aabb, obb, sap };
MeshMethod mesh_method(std::string_view name);

struct MeshData {
    std::vector<Vec> vertices;
    std::vector<std::array<std::size_t, 3>> faces;
    std::vector<int> cad_faces;
};

MeshData read_gmsh(const std::filesystem::path& file);

struct Intersection {
    std::size_t body_a, body_b, triangle_a, triangle_b;
    // One point, a segment, or an ordered coplanar polygon in world coordinates.
    std::vector<Vec> points;
};

struct IntersectionResult {
    std::vector<Intersection> intersections;
    // Each entry identifies an enclosed boundary component with no crossing of its own.
    struct Enclosed { std::size_t body, enclosing_body, vertex; };
    std::vector<Enclosed> enclosed;
    std::size_t body_pairs{}, node_pairs{}, triangle_tests{};
};

// Surface intersections and complete containment for closed, consistently wound solid boundaries.
class MeshScene {
public:
    MeshScene(std::span<const std::shared_ptr<const MeshData>> bodies, MeshMethod method);
    ~MeshScene();
    MeshScene(MeshScene&&) noexcept;
    MeshScene& operator=(MeshScene&&) noexcept;
    IntersectionResult intersect(std::span<const RigidPlacement> poses);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::vector<Vec> intersect_triangles(const Triangle& a, const Triangle& b);

} // namespace nc::cpu
