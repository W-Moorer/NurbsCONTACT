#pragma once

#include "nurbscontact/query.hpp"
#include "nurbscontact/triangle_bvh.hpp"

#include <filesystem>
#include <memory>
#include <string_view>

namespace nc::hht_bench {

using dynamics::PointPair;
using dynamics::Pose;
using dynamics::TriangleBvh;
using dynamics::TriangleHint;

enum class Shape { plate, sphere, cylinder };
enum class Backend { cad, general_cad, mesh };

struct Statistics {
    std::size_t queries{};
    std::size_t analytic{};
    std::size_t cad_fallback{};
    dynamics::TriangleStatistics mesh;
};

struct Geometry {
    Shape kind{};
    Vec center{};
    Vec u{};
    Vec v{};
    Vec axis{};
    double radius{};
    double half{};
    double half_u{};
    double half_v{};
    double motion_radius{};
    std::unique_ptr<PreparedModel> native;
    std::unique_ptr<TriangleBvh> mesh;
};

struct BoundedPose {
    IVec origin;
    Interval angle;
    IVec rotate(IVec p) const;
    IVec point(IVec p) const { return origin + rotate(p); }
};

struct BoundedPair {
    IVec a, b, normal;
    Interval gap;
};

void make_step_models(const std::filesystem::path& folder);
Geometry load_geometry(const std::filesystem::path& path, unsigned facets);
std::optional<PointPair> closest(
    const Geometry& a, Pose pa, const Geometry& b, Pose pb, Backend backend, double skin,
    TriangleHint* hint, Statistics& statistics
);
double safe_gap(const Geometry& a, BoundedPose pa, const Geometry& b, BoundedPose pb);
std::optional<BoundedPair> bounded_pair(const Geometry& a, BoundedPose pa, const Geometry& b, BoundedPose pb);
std::string_view name(Backend backend);

}  // namespace nc::hht_bench
