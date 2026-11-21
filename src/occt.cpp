#include "nurbscontact/occt.hpp"

#include <BRepTools_WireExplorer.hxx>
#include <BRep_Tool.hxx>
#include <Geom2d_BSplineCurve.hxx>
#include <Geom2d_BezierCurve.hxx>
#include <Geom2d_Circle.hxx>
#include <Geom2d_Ellipse.hxx>
#include <Geom2d_Line.hxx>
#include <Geom2d_TrimmedCurve.hxx>
#include <Geom_BSplineSurface.hxx>
#include <Geom_BezierSurface.hxx>
#include <Geom_ConicalSurface.hxx>
#include <Geom_CylindricalSurface.hxx>
#include <Geom_Plane.hxx>
#include <Geom_RectangularTrimmedSurface.hxx>
#include <Geom_SphericalSurface.hxx>
#include <Geom_ToroidalSurface.hxx>
#include <STEPControl_Reader.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopLoc_Location.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Iterator.hxx>
#include <TopoDS_Vertex.hxx>
#include <TopoDS_Wire.hxx>

#include <algorithm>
#include <iterator>
#include <string>
#include <utility>

namespace nc {
namespace {

template<class Point>
IVec xy(const Point& point) {
    return {point.X(), point.Y(), 0};
}

template<class Point>
IVec placed(const Point& point, const gp_Trsf& transform, bool translate) {
    IVec out;
    const IVec source{point.X(), point.Y(), point.Z()};
    for (int i = 1; i <= 3; ++i) {
        const IVec row{transform.Value(i, 1), transform.Value(i, 2), transform.Value(i, 3)};
        out[static_cast<std::size_t>(i - 1)] = dot(row, source);
        if (translate) out[static_cast<std::size_t>(i - 1)] = out[static_cast<std::size_t>(i - 1)] + transform.Value(i, 4);
    }
    return out;
}

struct Knots {
    int degree;
    std::vector<double> distinct;
    std::vector<int> count;
};

template<class Value, class Count>
Knots knots(int degree, int count, Value value, Count multiplicity) {
    Knots result{degree, {}, {}};
    for (int i = 1; i <= count; ++i) {
        result.distinct.push_back(value(i));
        result.count.push_back(multiplicity(i));
    }
    return result;
}

// Insert every interior knot to multiplicity degree in homogeneous coordinates.
// Input poles, weights and knots are the binary64 values read from the CAD model.
std::vector<HPoint> bezier_poles(const Knots& knots, std::vector<HPoint> poles) {
    const int degree = knots.degree;
    if (degree < 1 || knots.distinct.size() < 2 ||
        knots.count.front() != degree + 1 || knots.count.back() != degree + 1) {
        throw std::runtime_error("expected a clamped nonperiodic B-spline");
    }
    std::vector<double> expanded;
    for (std::size_t i = 0; i < knots.count.size(); ++i) {
        if (knots.count[i] < 1 || knots.count[i] > degree + 1 ||
            !std::isfinite(knots.distinct[i]) ||
            (i > 0 && knots.distinct[i - 1] >= knots.distinct[i])) {
            throw std::runtime_error("invalid B-spline knot sequence");
        }
        expanded.insert(expanded.end(), static_cast<std::size_t>(knots.count[i]), knots.distinct[i]);
    }
    if (expanded.size() != poles.size() + static_cast<std::size_t>(degree) + 1) {
        throw std::runtime_error("inconsistent B-spline pole count");
    }
    for (std::size_t j = 1; j + 1 < knots.distinct.size(); ++j) {
        const double u = knots.distinct[j];
        if (knots.count[j] > degree) throw std::runtime_error("discontinuous interior spline knot");
        for (int s = knots.count[j]; s < degree; ++s) {
            const auto upper = std::upper_bound(expanded.begin(), expanded.end(), u);
            const auto k = static_cast<int>(upper - expanded.begin()) - 1;
            std::vector<HPoint> next(poles.size() + 1);
            for (int i = 0; i <= k - degree; ++i) next[i] = poles[i];
            for (int i = k - s + 1; i < static_cast<int>(next.size()); ++i) next[i] = poles[i - 1];
            for (int i = k - degree + 1; i <= k - s; ++i) {
                const Interval alpha = (Interval{u} - expanded[i]) /
                    (Interval{expanded[i + degree]} - expanded[i]);
                HPoint point;
                for (std::size_t dim = 0; dim < point.size(); ++dim) {
                    point[dim] = (Interval{1} - alpha) * poles[i - 1][dim] + alpha * poles[i][dim];
                }
                next[i] = point;
            }
            expanded.insert(upper, u);
            poles = std::move(next);
        }
    }
    return poles;
}

Interval normalized(double parameter, double first, double last) {
    if (parameter == first) return 0;
    if (parameter == last) return 1;
    return (Interval{parameter} - first) / (Interval{last} - first);
}

std::vector<Curve> curves(Handle(Geom2d_Curve) geometry, double first, double last, bool reverse) {
    if (geometry.IsNull() || !std::isfinite(first) || !std::isfinite(last) || first > last) {
        throw std::runtime_error("missing or invalid p-curve");
    }
    while (const auto trimmed = Handle(Geom2d_TrimmedCurve)::DownCast(geometry)) {
        geometry = trimmed->BasisCurve();
    }
    if (const auto line = Handle(Geom2d_Line)::DownCast(geometry)) {
        const IVec position = xy(line->Position().Location());
        const IVec direction = xy(line->Position().Direction());
        const IVec a = position + direction * first;
        const IVec b = position + direction * last;
        return {BezierCurve{{homogeneous(reverse ? b : a), homogeneous(reverse ? a : b)}}};
    }
    if (const auto circle = Handle(Geom2d_Circle)::DownCast(geometry)) {
        const auto& frame = circle->Position();
        return {Arc{xy(frame.Location()), xy(frame.XDirection()) * circle->Radius(),
                    xy(frame.YDirection()) * circle->Radius(), {first, last}, reverse}};
    }
    if (const auto ellipse = Handle(Geom2d_Ellipse)::DownCast(geometry)) {
        const auto& frame = ellipse->Position();
        return {Arc{xy(frame.Location()), xy(frame.XDirection()) * ellipse->MajorRadius(),
                    xy(frame.YDirection()) * ellipse->MinorRadius(), {first, last}, reverse}};
    }
    if (const auto bezier = Handle(Geom2d_BezierCurve)::DownCast(geometry)) {
        BezierCurve curve;
        for (int i = 1; i <= bezier->NbPoles(); ++i) {
            curve.poles.push_back(homogeneous(xy(bezier->Pole(i)), bezier->Weight(i)));
        }
        curve = first >= 0 && last <= 1 ? curve.restrict({first, last}) : curve.reparameterize(first, last);
        if (reverse) std::reverse(curve.poles.begin(), curve.poles.end());
        return {std::move(curve)};
    }
    if (const auto spline = Handle(Geom2d_BSplineCurve)::DownCast(geometry)) {
        if (spline->IsPeriodic()) throw std::runtime_error("periodic B-spline p-curve is unsupported");
        const Knots k = knots(spline->Degree(), spline->NbKnots(), [&](int i) { return spline->Knot(i); },
                              [&](int i) { return spline->Multiplicity(i); });
        std::vector<HPoint> poles;
        for (int i = 1; i <= spline->NbPoles(); ++i) {
            poles.push_back(homogeneous(xy(spline->Pole(i)), spline->Weight(i)));
        }
        poles = bezier_poles(k, std::move(poles));
        std::vector<Curve> result;
        const auto degree = static_cast<std::ptrdiff_t>(k.degree);
        for (std::size_t i = 0; i + 1 < k.distinct.size(); ++i) {
            const double lo = k.distinct[i];
            const double hi = k.distinct[i + 1];
            const double from = i == 0 ? first : std::max(lo, first);
            const double to = i + 2 == k.distinct.size() ? last : std::min(hi, last);
            if (from >= to) continue;
            const auto start = poles.begin() + degree * static_cast<std::ptrdiff_t>(i);
            BezierCurve piece{{start, start + degree + 1}};
            const Interval a = normalized(from, lo, hi);
            const Interval b = normalized(to, lo, hi);
            piece = from >= lo && to <= hi ? piece.restrict(a, b) : piece.reparameterize(a, b);
            if (reverse) std::reverse(piece.poles.begin(), piece.poles.end());
            result.push_back(std::move(piece));
        }
        if (result.empty()) throw std::runtime_error("empty B-spline p-curve span");
        if (reverse) std::reverse(result.begin(), result.end());
        return result;
    }
    throw std::runtime_error(std::string("unsupported p-curve type: ") + geometry->DynamicType()->Name());
}

bool same_exact(IVec a, IVec b) {
    for (std::size_t i = 0; i < 3; ++i) {
        if (a[i].lo != a[i].hi || b[i].lo != b[i].hi || a[i].lo != b[i].lo) return false;
    }
    return true;
}

void join(std::vector<TrimCurve>& loop, IVec a, IVec b, Feature feature, double allowed, double& max_join) {
    if (same_exact(a, b)) return;
    const double gap = norm(a - b).hi;
    if (gap > allowed) throw std::runtime_error("p-curve loop has a UV gap above the import tolerance");
    max_join = std::max(max_join, gap);
    loop.push_back({BezierCurve{{homogeneous(a), homogeneous(b)}}, feature, feature, feature});
}

TrimDomain extract_trim(const TopoDS_Face& face, std::size_t face_id, const TopTools_IndexedMapOfShape& edges,
                        const TopTools_IndexedMapOfShape& vertices, double allowed, double& max_join) {
    TrimDomain domain;
    for (TopExp_Explorer explorer(face, TopAbs_WIRE); explorer.More(); explorer.Next()) {
        const auto& wire = TopoDS::Wire(explorer.Current());
        std::vector<TrimCurve> loop;
        std::size_t edge_count = 0;
        for (TopoDS_Iterator it(wire); it.More(); it.Next()) ++edge_count;
        std::size_t visited = 0;
        for (BRepTools_WireExplorer it(wire, face); it.More(); it.Next()) {
            ++visited;
            const TopoDS_Edge& edge = it.Current();
            double first = 0;
            double last = 0;
            const auto pcurve = BRep_Tool::CurveOnSurface(edge, face, first, last);
            const auto edge_id = static_cast<std::size_t>(edges.FindIndex(edge));
            const bool seam = BRep_Tool::IsClosed(edge, face);
            const Feature on_curve = seam ? Feature{FeatureKind::face, face_id} : Feature{FeatureKind::edge, edge_id};
            TopoDS_Vertex vertex_a;
            TopoDS_Vertex vertex_b;
            TopExp::Vertices(edge, vertex_a, vertex_b, true);
            const Feature start = vertex_a.IsNull() ? on_curve
                : Feature{FeatureKind::vertex, static_cast<std::size_t>(vertices.FindIndex(vertex_a))};
            const Feature end = vertex_b.IsNull() ? on_curve
                : Feature{FeatureKind::vertex, static_cast<std::size_t>(vertices.FindIndex(vertex_b))};
            auto pieces = curves(pcurve, first, last, edge.Orientation() == TopAbs_REVERSED);
            for (std::size_t i = 0; i < pieces.size(); ++i) {
                Curve& piece = pieces[i];
                const Feature piece_start = i == 0 ? start : on_curve;
                const Feature piece_end = i + 1 == pieces.size() ? end : on_curve;
                if (!loop.empty()) {
                    join(loop, evaluate(loop.back().geometry, 1), evaluate(piece, 0),
                         piece_start, allowed, max_join);
                }
                loop.push_back({std::move(piece), on_curve, piece_start, piece_end});
            }
        }
        if (loop.empty() || visited != edge_count) throw std::runtime_error("face contains an unordered or empty trim wire");
        join(loop, evaluate(loop.back().geometry, 1), evaluate(loop.front().geometry, 0),
             loop.front().start, allowed, max_join);
        domain.loops.push_back(std::move(loop));
    }
    if (domain.loops.empty()) throw std::runtime_error("face has no bounded trim loop");
    return domain;
}

Rect chart_bounds(const TrimDomain& trim) {
    IVec box = bounds(trim.loops.front().front().geometry);
    for (const auto& loop : trim.loops) {
        for (const TrimCurve& curve : loop) box = hull(box, bounds(curve.geometry));
    }
    return {box[0], box[1]};
}

void append_spline(Model& out, const Geom_BSplineSurface& surface, const gp_Trsf& loc,
                   std::size_t face, std::size_t trim, Rect limits) {
    if (surface.IsUPeriodic() || surface.IsVPeriodic()) {
        throw std::runtime_error("periodic B-spline surface is unsupported");
    }
    const Knots u = knots(surface.UDegree(), surface.NbUKnots(), [&](int i) { return surface.UKnot(i); },
                          [&](int i) { return surface.UMultiplicity(i); });
    const Knots v = knots(surface.VDegree(), surface.NbVKnots(), [&](int i) { return surface.VKnot(i); },
                          [&](int i) { return surface.VMultiplicity(i); });
    std::vector<std::vector<HPoint>> rows;
    for (int i = 1; i <= surface.NbUPoles(); ++i) {
        std::vector<HPoint> row;
        for (int j = 1; j <= surface.NbVPoles(); ++j) {
            row.push_back(homogeneous(placed(surface.Pole(i, j), loc, true), surface.Weight(i, j)));
        }
        rows.push_back(bezier_poles(v, std::move(row)));
    }
    std::vector<std::vector<HPoint>> columns;
    for (std::size_t j = 0; j < rows.front().size(); ++j) {
        std::vector<HPoint> column;
        for (const auto& row : rows) column.push_back(row[j]);
        columns.push_back(bezier_poles(u, std::move(column)));
    }
    const auto du = static_cast<std::size_t>(u.degree);
    const auto dv = static_cast<std::size_t>(v.degree);
    for (std::size_t a = 0; a + 1 < u.distinct.size(); ++a) {
        for (std::size_t b = 0; b + 1 < v.distinct.size(); ++b) {
            const Rect chart{{u.distinct[a], u.distinct[a + 1]}, {v.distinct[b], v.distinct[b + 1]}};
            const Rect clipped{{std::max(chart.u.lo, limits.u.lo), std::min(chart.u.hi, limits.u.hi)},
                               {std::max(chart.v.lo, limits.v.lo), std::min(chart.v.hi, limits.v.hi)}};
            if (clipped.u.lo >= clipped.u.hi || clipped.v.lo >= clipped.v.hi) continue;
            BezierSurface patch{du + 1, dv + 1, {}, chart};
            for (std::size_t i = 0; i <= du; ++i) {
                for (std::size_t j = 0; j <= dv; ++j) patch.poles.push_back(columns[b * dv + j][a * du + i]);
            }
            out.patches.push_back({std::move(patch), clipped, face, trim});
        }
    }
}

void append_face(Model& out, const TopoDS_Face& face, std::size_t face_id, std::size_t trim_id) {
    TopLoc_Location location;
    Handle(Geom_Surface) surface = BRep_Tool::Surface(face, location);
    while (const auto trimmed = Handle(Geom_RectangularTrimmedSurface)::DownCast(surface)) {
        surface = trimmed->BasisSurface();
    }
    if (surface.IsNull()) throw std::runtime_error("face has no surface");
    const gp_Trsf& loc = location.Transformation();
    const Rect rect = chart_bounds(out.trims[trim_id]);
    const auto axis = [&](const gp_Ax3& frame) {
        return std::array{placed(frame.Location(), loc, true), placed(frame.XDirection(), loc, false),
                          placed(frame.YDirection(), loc, false), placed(frame.Direction(), loc, false)};
    };
    const auto add = [&](Surface patch, Rect bounds) {
        out.patches.push_back({std::move(patch), bounds, face_id, trim_id});
    };
    if (const auto plane = Handle(Geom_Plane)::DownCast(surface)) {
        const auto [o, x, y, z] = axis(plane->Position());
        add(Plane{o, x, y}, rect);
    } else if (const auto cylinder = Handle(Geom_CylindricalSurface)::DownCast(surface)) {
        const auto [o, x, y, z] = axis(cylinder->Position());
        add(Cylinder{o, x, y, z, cylinder->Radius()}, rect);
    } else if (const auto cone = Handle(Geom_ConicalSurface)::DownCast(surface)) {
        const auto [o, x, y, z] = axis(cone->Position());
        const auto sc = sin_cos(cone->SemiAngle());
        add(Revolved{o, x, y, z, cone->RefRadius(), sc.sin, sc.cos, 0}, rect);
    } else if (const auto sphere = Handle(Geom_SphericalSurface)::DownCast(surface)) {
        const auto [o, x, y, z] = axis(sphere->Position());
        add(Revolved{o, x, y, z, 0, 0, 0, sphere->Radius()}, rect);
    } else if (const auto torus = Handle(Geom_ToroidalSurface)::DownCast(surface)) {
        const auto [o, x, y, z] = axis(torus->Position());
        add(Revolved{o, x, y, z, torus->MajorRadius(), 0, 0, torus->MinorRadius()}, rect);
    } else if (const auto bezier = Handle(Geom_BezierSurface)::DownCast(surface)) {
        const Rect native{{0, 1}, {0, 1}};
        BezierSurface patch{static_cast<std::size_t>(bezier->NbUPoles()),
                            static_cast<std::size_t>(bezier->NbVPoles()), {}, native};
        for (int i = 1; i <= bezier->NbUPoles(); ++i) {
            for (int j = 1; j <= bezier->NbVPoles(); ++j) {
                patch.poles.push_back(homogeneous(placed(bezier->Pole(i, j), loc, true), bezier->Weight(i, j)));
            }
        }
        add(std::move(patch), {{std::max(0.0, rect.u.lo), std::min(1.0, rect.u.hi)},
                               {std::max(0.0, rect.v.lo), std::min(1.0, rect.v.hi)}});
    } else if (const auto spline = Handle(Geom_BSplineSurface)::DownCast(surface)) {
        append_spline(out, *spline, loc, face_id, trim_id, rect);
    } else {
        throw std::runtime_error(std::string("unsupported surface type: ") + surface->DynamicType()->Name());
    }
}

}  // namespace

Model adapt_occt(const TopoDS_Shape& shape, const ImportOptions& options) {
    if (shape.IsNull()) throw std::invalid_argument("empty CAD shape");
    if (!std::isfinite(options.uv_join_tolerance) || options.uv_join_tolerance < 0) {
        throw std::invalid_argument("invalid import UV join tolerance");
    }
    TopTools_IndexedMapOfShape faces;
    TopTools_IndexedMapOfShape edges;
    TopTools_IndexedMapOfShape vertices;
    TopExp::MapShapes(shape, TopAbs_FACE, faces);
    TopExp::MapShapes(shape, TopAbs_EDGE, edges);
    TopExp::MapShapes(shape, TopAbs_VERTEX, vertices);
    Model result;
    for (int i = 1; i <= faces.Extent(); ++i) {
        const auto face = TopoDS::Face(faces(i).Oriented(TopAbs_FORWARD));
        const auto id = static_cast<std::size_t>(i);
        const std::size_t trim_id = result.trims.size();
        const std::size_t before = result.patches.size();
        try {
            result.trims.push_back(extract_trim(face, id, edges, vertices, options.uv_join_tolerance,
                                                result.max_trim_join));
            append_face(result, face, id, trim_id);
            if (result.patches.size() == before) throw std::runtime_error("face has no supported nonempty chart");
        } catch (const Standard_Failure& e) {
            throw std::runtime_error("OCCT face " + std::to_string(i) + ": " + e.GetMessageString());
        } catch (const std::exception& e) {
            throw std::runtime_error("face " + std::to_string(i) + ": " + e.what());
        }
    }
    if (result.patches.empty()) throw std::runtime_error("CAD shape has no faces");
    return result;
}

TopoDS_Shape read_step_shape(const std::filesystem::path& file) {
    const auto utf8 = file.u8string();
    const std::string name(utf8.begin(), utf8.end());
    STEPControl_Reader reader;
    if (reader.ReadFile(name.c_str()) != IFSelect_RetDone || reader.TransferRoots() <= 0) {
        throw std::runtime_error("cannot read STEP roots from " + name);
    }
    return reader.OneShape();
}

Model read_step(const std::filesystem::path& file, const ImportOptions& options) {
    return adapt_occt(read_step_shape(file), options);
}

}  // namespace nc
