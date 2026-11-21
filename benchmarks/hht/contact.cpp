#include "contact.hpp"

#include "nurbscontact/occt.hpp"

#include <BRepAdaptor_Curve.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRepGProp.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakeSphere.hxx>
#include <STEPControl_Writer.hxx>
#include <GProp_GProps.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <gp_Pln.hxx>

#include <numbers>
#include <string>

namespace nc::hht_bench {
namespace {

using namespace dynamics;

template<class P> Vec vector(P p) { return {p.X(), p.Y(), p.Z()}; }

bool rectangle(const TopoDS_Face& face, const Geometry& g) {
    unsigned wires = 0;
    for (TopExp_Explorer it(face,TopAbs_WIRE); it.More(); it.Next()) ++wires;
    if (wires != 1) return false;
    unsigned edges = 0;
    unsigned corners = 0;
    for (TopExp_Explorer it(face,TopAbs_EDGE); it.More(); it.Next()) {
        ++edges;
        const BRepAdaptor_Curve edge(TopoDS::Edge(it.Current()));
        if (edge.GetType() != GeomAbs_Line) return false;
        std::array<unsigned,2> code{};
        unsigned i = 0;
        for (double p : {edge.FirstParameter(),edge.LastParameter()}) {
            const Vec v = sub(vector(edge.Value(p)),g.center);
            const double x = dot3(v,g.u), y = dot3(v,g.v);
            if (std::abs(std::abs(x)-g.half_u)>1e-10 || std::abs(std::abs(y)-g.half_v)>1e-10) return false;
            code[i++] = static_cast<unsigned>(x>0) + 2*static_cast<unsigned>(y>0);
        }
        if ((code[0]^code[1])!=1 && (code[0]^code[1])!=2) return false;
        corners |= (1u << code[0]) | (1u << code[1]);
    }
    return edges == 4 && corners == 15;
}

void write_step(const TopoDS_Shape& shape, const std::filesystem::path& file) {
    STEPControl_Writer writer;
    if (writer.Transfer(shape, STEPControl_AsIs) != IFSelect_RetDone ||
        writer.Write(file.string().c_str()) != IFSelect_RetDone) {
        throw std::runtime_error("STEP fixture export failed: " + file.string());
    }
}

void plate_step(const std::filesystem::path& file, gp_Dir normal, gp_Dir u, double hu, double hv) {
    const gp_Pln plane{gp_Ax3{gp_Pnt{}, normal, u}};
    write_step(BRepBuilderAPI_MakeFace(plane, -hu, hu, -hv, hv).Shape(), file);
}

Vec circle(const Geometry& g, double angle) {
    return add(scale(g.u, std::cos(angle)), scale(g.v, std::sin(angle)));
}

std::vector<Triangle> triangulate(const Geometry& g, unsigned facets) {
    std::vector<Triangle> result;
    if (g.kind == Shape::plate) {
        const Vec u = scale(g.u, g.half_u);
        const Vec v = scale(g.v, g.half_v);
        const Vec p = sub(sub(g.center, u), v);
        const Vec q = add(sub(g.center, v), u);
        const Vec r = add(add(g.center, u), v);
        const Vec s = add(sub(g.center, u), v);
        return {{p, q, r}, {p, r, s}};
    }
    constexpr double tau = 2 * std::numbers::pi;
    if (g.kind == Shape::cylinder) {
        const Vec top = add(g.center, scale(g.axis, g.half));
        const Vec bottom = sub(g.center, scale(g.axis, g.half));
        for (unsigned i = 0; i < facets; ++i) {
            const Vec a = scale(circle(g, tau*i/facets), g.radius);
            const Vec b = scale(circle(g, tau*((i+1)%facets)/facets), g.radius);
            const Vec a0 = add(bottom, a), b0 = add(bottom, b);
            const Vec a1 = add(top, a), b1 = add(top, b);
            result.insert(result.end(), {{a0, b0, a1}, {a1, b0, b1}, {bottom, b0, a0}, {top, a1, b1}});
        }
        return result;
    }

    const unsigned latitude = facets / 2;
    const auto vertex = [&](unsigned i, unsigned j) {
        if (i == 0) return sub(g.center, scale(g.axis, g.radius));
        if (i == latitude) return add(g.center, scale(g.axis, g.radius));
        const double azimuth = tau * (j % facets) / facets;
        const double elevation = -std::numbers::pi / 2 + std::numbers::pi * i / latitude;
        return add(g.center, scale(add(scale(circle(g, azimuth), std::cos(elevation)),
                                      scale(g.axis, std::sin(elevation))), g.radius));
    };
    for (unsigned i = 0; i < latitude; ++i) {
        for (unsigned j = 0; j < facets; ++j) {
            const Vec a = vertex(i, j), b = vertex(i, j + 1);
            const Vec c = vertex(i + 1, j), d = vertex(i + 1, j + 1);
            if (i != 0) result.push_back({a, b, c});
            if (i + 1 != latitude) result.push_back({b, d, c});
        }
    }
    return result;
}

std::optional<PointPair> analytic_pair(const Geometry& a, Pose pa, const Geometry& b, Pose pb) {
    const Vec center_a = pa.point(a.center);
    const Vec center_b = pb.point(b.center);
    if (a.kind == Shape::sphere && b.kind == Shape::sphere) {
        const Vec offset = sub(center_b, center_a);
        const double d = length(offset);
        if (d <= a.radius + b.radius) return std::nullopt;
        const Vec n = scale(offset, 1/d);
        return PointPair{add(center_a, scale(n, a.radius)), sub(center_b, scale(n, b.radius)),
                         d - a.radius - b.radius};
    }
    if (a.kind != Shape::plate || b.kind == Shape::plate) return std::nullopt;

    const Vec n = pa.rotate(a.axis);
    Vec on_b;
    if (b.kind == Shape::sphere) {
        on_b = sub(center_b, scale(n, b.radius));
    } else {
        const Vec axis = pb.rotate(b.axis);
        const double component = dot3(n, axis);
        const Vec radial = sub(n, scale(axis, component));
        const double radial_length = length(radial);
        const double axial = component == 0 ? 0 : std::copysign(b.half, component);
        on_b = sub(center_b, scale(axis, axial));
        if (radial_length > 0) on_b = sub(on_b, scale(radial, b.radius / radial_length));
    }
    const double d = dot3(n, sub(on_b, center_a));
    const Vec on_a = sub(on_b, scale(n, d));
    const Vec local = sub(on_a, center_a);
    if (d <= 0 || std::abs(dot3(local, pa.rotate(a.u))) > a.half_u ||
        std::abs(dot3(local, pa.rotate(a.v))) > a.half_v) return std::nullopt;
    return PointPair{on_a, on_b, d};
}

}  // namespace

void make_step_models(const std::filesystem::path& folder) {
    std::filesystem::create_directories(folder);
    plate_step(folder/"floor.step", gp_Dir(0,0,1), gp_Dir(1,0,0), 380, 270);
    plate_step(folder/"follower.step", gp_Dir(0,-1,0), gp_Dir(1,0,0), 172, 50);
    plate_step(folder/"left-stop.step", gp_Dir(1,0,0), gp_Dir(0,1,0), 95, 95);
    plate_step(folder/"right-stop.step", gp_Dir(-1,0,0), gp_Dir(0,-1,0), 95, 95);
    write_step(BRepPrimAPI_MakeSphere(55).Shape(), folder/"ball.step");
    const gp_Ax2 axis{gp_Pnt(24, 0, -30), gp_Dir(0,0,1)};
    write_step(BRepPrimAPI_MakeCylinder(axis, 105, 60).Shape(), folder/"cam.step");
}

Geometry load_geometry(const std::filesystem::path& path, unsigned facets) {
    if (facets < 8 || facets > 256 || facets % 2 != 0) throw std::invalid_argument("even facet count must be 8 to 256");
    gp_Trsf scaling;
    scaling.SetScale(gp_Pnt{}, 0.001);
    const TopoDS_Shape original = read_step_shape(path);
    const auto model = BRepBuilderAPI_Transform(original, scaling, true).Shape();
    Geometry g;
    unsigned planes = 0, balls = 0, cylinders = 0, total = 0;
    bool rectangular = false;
    bool full_round = true;
    for (TopExp_Explorer it(model, TopAbs_FACE); it.More(); it.Next()) {
        ++total;
        const BRepAdaptor_Surface face{TopoDS::Face(it.Current())};
        if (face.GetType() == GeomAbs_Sphere) {
            ++balls;
            const auto sphere = face.Sphere();
            g.kind = Shape::sphere;
            g.center = vector(sphere.Location());
            g.axis = vector(sphere.Position().Direction());
            g.u = vector(sphere.Position().XDirection());
            g.v = vector(sphere.Position().YDirection());
            g.radius = sphere.Radius();
            full_round = std::abs(face.LastUParameter()-face.FirstUParameter()-2*std::numbers::pi)<1e-9 &&
                         std::abs(face.LastVParameter()-face.FirstVParameter()-std::numbers::pi)<1e-9;
        } else if (face.GetType() == GeomAbs_Cylinder) {
            ++cylinders;
            const auto cylinder = face.Cylinder();
            g.kind = Shape::cylinder;
            g.axis = vector(cylinder.Axis().Direction());
            g.u = vector(cylinder.Position().XDirection());
            g.v = vector(cylinder.Position().YDirection());
            const double v0 = face.FirstVParameter(), v1 = face.LastVParameter();
            g.center = add(vector(cylinder.Location()), scale(g.axis, std::midpoint(v0, v1)));
            g.half = (v1-v0)/2;
            g.radius = cylinder.Radius();
            full_round = std::abs(face.LastUParameter()-face.FirstUParameter()-2*std::numbers::pi)<1e-9;
        } else if (face.GetType() == GeomAbs_Plane) {
            ++planes;
            if (cylinders || balls) continue;
            const auto plane = face.Plane();
            g.kind = Shape::plate;
            g.axis = vector(plane.Axis().Direction());
            g.u = vector(plane.Position().XDirection());
            g.v = vector(plane.Position().YDirection());
            const double u0 = face.FirstUParameter(), u1 = face.LastUParameter();
            const double v0 = face.FirstVParameter(), v1 = face.LastVParameter();
            g.center = vector(face.Value(std::midpoint(u0,u1), std::midpoint(v0,v1)));
            g.half_u = (u1-u0)/2;
            g.half_v = (v1-v0)/2;
            rectangular = rectangle(TopoDS::Face(it.Current()),g);
        }
    }
    if (!((planes==1 && total==1 && rectangular) || (balls==1 && total==1 && full_round) ||
          (cylinders==1 && planes==2 && total==3 && full_round))) {
        throw std::runtime_error("unsupported analytic dynamics fixture: " + path.string());
    }
    if (g.kind != Shape::plate) {
        GProp_GProps properties;
        BRepGProp::VolumeProperties(model,properties);
        const double expected = g.kind==Shape::sphere
            ? (4.0/3)*std::numbers::pi*g.radius*g.radius*g.radius
            : std::numbers::pi*g.radius*g.radius*(2*g.half);
        if (!(expected>0) || std::abs(properties.Mass()-expected)>1e-8*expected) {
            throw std::runtime_error("STEP fixture does not enclose the expected round solid");
        }
    }
    g.motion_radius = (norm(exact(g.center)) + g.radius + g.half + g.half_u + g.half_v).hi;
    g.native = std::make_unique<PreparedModel>(adapt_occt(model));
    g.mesh = std::make_unique<TriangleBvh>(triangulate(g, facets), 2);
    return g;
}

IVec BoundedPose::rotate(IVec p) const {
    const auto sc = sin_cos(angle);
    return {p[0]*sc.cos - p[1]*sc.sin, p[0]*sc.sin + p[1]*sc.cos, p[2]};
}

double safe_gap(const Geometry& a, BoundedPose pa, const Geometry& b, BoundedPose pb) {
    const IVec offset = pb.point(exact(b.center)) - pa.point(exact(a.center));
    if (a.kind == Shape::sphere && b.kind == Shape::sphere) {
        return (norm(offset) - a.radius - b.radius).lo;
    }
    if (a.kind != Shape::plate || b.kind == Shape::plate) {
        throw std::invalid_argument("no analytic clearance guard for fixture pair");
    }
    const IVec axis = pa.rotate(exact(a.axis));
    const IVec n = axis / norm(axis);
    Interval support_radius{b.radius};
    if (b.kind == Shape::cylinder) {
        const IVec z = pb.rotate(exact(b.axis));
        support_radius = norm(cross(n,z)) * b.radius + Interval{magnitude(dot(n,z))} * b.half;
    }
    return (dot(n, offset) - support_radius).lo;
}

std::optional<PointPair> closest(
    const Geometry& a, Pose pa, const Geometry& b, Pose pb, Backend backend, double skin,
    TriangleHint* hint, Statistics& statistics
) {
    ++statistics.queries;
    if (backend == Backend::cad) {
        if (const auto pair = analytic_pair(a, pa, b, pb)) {
            ++statistics.analytic;
            return pair->distance <= skin ? pair : std::nullopt;
        }
    }
    const Pose placement = relative_pose(pa, pb);
    if (backend == Backend::mesh) {
        auto result = a.mesh->closest_within(*b.mesh, placement, skin, hint, statistics.mesh);
        if (result) {
            result->a = pa.point(result->a);
            result->b = pa.point(result->b);
        }
        return result;
    }
    ++statistics.cad_fallback;
    Motion pose;
    pose.translation = placement.origin;
    pose.angle = std::atan2(placement.sine, placement.cosine);
    Options options;
    options.distance_tolerance = 2e-6;
    options.max_nodes = 30'000;
    const auto answer = nc::distance(*a.native, *b.native, pose, 0, options);
    if (answer.lower > skin) return std::nullopt;
    if (!answer.witness || answer.upper - answer.lower > options.distance_tolerance) {
        throw std::runtime_error("general B-rep query did not reach 2 micrometers");
    }
    return PointPair{pa.point(midpoint(answer.witness->a.position)),
                     pa.point(midpoint(answer.witness->b.position)), answer.upper};
}

std::string_view name(Backend backend) {
    if (backend == Backend::cad) return "brep";
    if (backend == Backend::mesh) return "mesh";
    return "general-brep";
}

}  // namespace nc::hht_bench
