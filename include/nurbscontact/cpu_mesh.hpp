#pragma once

#include "nurbscontact/triangle_bvh.hpp"
#include "nurbscontact/contact_dynamics.hpp"

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
    struct Contour {
        std::size_t body_a, body_b;
        std::vector<std::array<std::size_t,2>> triangles;
        // Oriented vector area of one closed transverse loop, following body A's
        // boundary winding. Physical outward normals require an outward solid
        // boundary (cavity faces point into their cavities).
        std::optional<Vec> vector_area_a_m2;
    };
    // Connected sets of boundary intersections (points, curves or coplanar areas).
    std::vector<Contour> contours;
    std::size_t body_pairs{}, node_pairs{}, triangle_tests{};
};

#ifdef NC_WITH_MESH_CCD
struct ImpactCandidate {
    std::size_t body_a, body_b, triangle_a, triangle_b;
    double fraction; // Earliest retained candidate on the normalized motion.
    double geometric_tolerance_m; // Includes both bodies' chord error.
};

struct MeshSweepResult {
    std::optional<ImpactCandidate> first;
    std::size_t body_pairs{}, node_pairs{}, triangle_pairs{}, linear_tests{}, translation_tests{};
};
#endif

struct OverlapResult {
    bool overlap{}, reused_triangle_pair{};
    std::size_t body_pairs{}, node_pairs{}, triangle_tests{};
    bool adjacent_triangle_pair{};
};

// Surface intersections and complete containment for closed, consistently wound solid boundaries.
class MeshScene {
public:
    MeshScene(std::span<const std::shared_ptr<const MeshData>> bodies, MeshMethod method, bool candidate_window=false);
    ~MeshScene();
    MeshScene(MeshScene&&) noexcept;
    MeshScene& operator=(MeshScene&&) noexcept;
    IntersectionResult intersect(std::span<const RigidPlacement> poses, bool collect_contours = false);
    OverlapResult overlaps(std::span<const RigidPlacement> poses, bool follow_adjacent=true);
#ifdef NC_WITH_MESH_CCD
    // Straight translation and shortest-arc orientation per body.
    MeshSweepResult sweep(std::span<const RigidPlacement> start, std::span<const RigidPlacement> end,
                          double distance_tolerance_m);
#endif

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::vector<Vec> intersect_triangles(const Triangle& a, const Triangle& b);

// Each connected overlap volume supplies one unilateral penalty region.
std::shared_ptr<const dynamics::PairRegionQuery> mesh_region_query(
    std::shared_ptr<const MeshData> a, std::shared_ptr<const MeshData> b, MeshMethod method,
    std::array<Vec,2> local_motion_centers = {}, bool localize = true);

} // namespace nc::cpu
