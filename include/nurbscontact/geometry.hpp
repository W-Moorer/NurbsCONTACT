#pragma once

#include "nurbscontact/interval.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <variant>
#include <vector>

namespace nc {

using UV = std::array<Interval, 2>;

struct Rect {
    Interval u;
    Interval v;
};

using HPoint = std::array<Interval, 4>;
HPoint homogeneous(IVec point, Interval weight = 1);
IVec cartesian(HPoint point);

struct BezierCurve {
    std::vector<HPoint> poles;
    IVec eval(Interval parameter) const;
    std::pair<BezierCurve, BezierCurve> split(Interval parameter = 0.5) const;
    BezierCurve restrict(Interval parameters) const;
    BezierCurve restrict(Interval first, Interval last) const;
    BezierCurve reparameterize(Interval first, Interval last) const;
    IVec box() const;
};

struct SurfaceMode {
    // Cartesian displacements in the order of the surface's control points.
    std::vector<IVec> displacements;
};

struct BezierSurface {
    // Consecutive entries vary in v. The source chart uses the supplied domain.
    std::size_t nu{};
    std::size_t nv{};
    std::vector<HPoint> poles;
    Rect domain;
    IVec eval(UV uv) const;
    std::vector<IVec> support(Rect rect) const;
    // Keeps the material chart and weights. Interval amplitudes enclose a
    // family of patches for support bounds.
    BezierSurface deformed(std::span<const SurfaceMode> modes,
                           std::span<const Interval> amplitudes) const;
};

struct Plane {
    IVec origin;
    IVec x;
    IVec y;
    IVec eval(UV uv) const;
    std::vector<IVec> support(Rect rect) const;
};

struct Cylinder {
    IVec origin;
    IVec x;
    IVec y;
    IVec z;
    Interval radius;
    IVec eval(UV uv) const;
    std::vector<IVec> support(Rect rect) const;
};

// Native cone, sphere and torus parameters. A meridian is a linear term
// plus a circular term in the radial and height coordinates.
struct Revolved {
    IVec origin;
    IVec x;
    IVec y;
    IVec z;
    Interval radius;
    Interval radial_slope;
    Interval height_slope;
    Interval meridian_radius;
    IVec eval(UV uv) const;
    std::vector<IVec> support(Rect rect) const;
};

using Surface = std::variant<Plane, Cylinder, Revolved, BezierSurface>;

// Elliptical arc in a face's UV chart. Its z component is zero.
struct Arc {
    IVec center;
    IVec x;
    IVec y;
    Interval angles;
    bool reversed{};
    IVec eval(Interval parameter) const;
    IVec box(Interval parameters) const;
};

using Curve = std::variant<BezierCurve, Arc>;

enum class FeatureKind { face, edge, vertex };
struct Feature {
    FeatureKind kind{FeatureKind::face};
    std::size_t id{};
};

struct TrimCurve {
    Curve geometry;
    Feature feature;
    Feature start;
    Feature end;
};

enum class TrimClass { inside, outside, mixed };

struct TrimDomain {
    // Each loop must be closed. Even-odd membership allows holes independent
    // of the order and orientation in which the importer encounters the loops.
    std::vector<std::vector<TrimCurve>> loops;
    TrimClass classify(Vec uv) const;
    TrimClass classify(Rect rect) const;
};

struct FacePatch {
    Surface surface;
    Rect domain;
    std::size_t face_id{};
    std::size_t trim_id{};
};

struct Model {
    std::vector<FacePatch> patches;
    std::vector<TrimDomain> trims;
    // Maximum UV connector enclosure reported by the STEP adapter.
    double max_trim_join{};
};

struct Sample {
    UV uv;
    IVec position;
    Feature feature;
};

IVec evaluate(const Surface& surface, UV uv);
IVec evaluate(const Curve& curve, Interval parameter);
IVec bounds(const Curve& curve, Interval parameters = {0, 1});
std::vector<IVec> support(const Surface& surface, Rect rect);
IVec bounding_box(const std::vector<IVec>& support_points);
std::vector<IVec> corners(IVec box);
std::pair<Rect, Rect> split(Rect rect, bool along_u);
std::vector<Sample> samples(const FacePatch& patch, const TrimDomain& trim, Rect rect, TrimClass kind);

}  // namespace nc
