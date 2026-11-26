#include "../src/compiled_geometry.hpp"
#include "nurbscontact/occt.hpp"
#include <BRepAdaptor_Surface.hxx>
#include <BRepTools.hxx>
#include <BRepTools_WireExplorer.hxx>
#include <BRepTopAdaptor_FClass2d.hxx>
#include <BRep_Tool.hxx>
#include <Geom2dAPI_ProjectPointOnCurve.hxx>
#include <Geom2dAdaptor_Curve.hxx>
#include <Geom2dConvert_BSplineCurveToBezierCurve.hxx>
#include <Geom2d_BSplineCurve.hxx>
#include <Geom2d_BezierCurve.hxx>
#include <GeomConvert_BSplineSurfaceToBezierSurface.hxx>
#include <Geom_BezierSurface.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <iomanip>
#include <iostream>
#include <set>

namespace {
using namespace nc;
using namespace nc::compiled;
struct Report {
    int faces{}, spans{}, analytic{}, poles{}, samples{}, singular{}, membership{}, wires{}, trim_samples{};
    double position{}, normal{}, boundary{}, net_bound{}, max_tolerance_ratio{};
    int planes{}, cylinders{};
    double primitive_bound{};
    int trim_spans{}, connectors{};
    double trim_bound{}, connector_extent{};
};
void require(bool ok, const char *what) {
    if (!ok)
        throw std::runtime_error(what);
}
Vec xyz(const gp_XYZ &p) { return {p.X(), p.Y(), p.Z()}; }
double mix(Interval r, double a) { return std::lerp(r.lo, r.hi, a); }

// Both Bernstein bases are nonnegative and sum to one. With positive weights,
// |A/W - B/V| <= (max |Ai-Bi| + max |Bi/Vi| max |Wi-Vi|)/min Wi.
// Common weight scaling is immaterial. Compare the net actually used at runtime.
void compare_net(const Patch &q, const Handle(Geom_BezierSurface) & other, Report &r) {
    require(q.nu == other->NbUPoles() && q.nv == other->NbVPoles(), "degree mismatch");
    double da = 0, dw = 0, radius = 0, minw = infinity;
    for (int i = 0; i < q.nu; ++i)
        for (int j = 0; j < q.nv; ++j) {
            const auto &h = q.net[i * q.nv + j];
            const double w = q.cubic ? 1. : h[3];
            const double v = other->Weight(i + 1, j + 1) / other->Weight(1, 1);
            const auto p = xyz(other->Pole(i + 1, j + 1).XYZ());
            const Vec a = q.cubic ? Vec{h[0] / h[3], h[1] / h[3], h[2] / h[3]} : Vec{h[0], h[1], h[2]};
            da = std::max(da, length(sub(a, scale(p, v))));
            dw = std::max(dw, std::abs(w - v));
            minw = std::min(minw, w);
            radius = std::max(radius, length(p));
            ++r.poles;
        }
    require(minw > 0, "nonpositive weight");
    r.net_bound = std::max(r.net_bound, (da + radius * dw) / minw);
}
void check_sample(const Face &face, const BRepAdaptor_Surface &surface, Rect domain, Report &r) {
    for (auto uv0 : {Param{.07, .81}, Param{.5, .5}, Param{.91, .19}}) {
        const Param uv{mix(domain.u, uv0[0]), mix(domain.v, uv0[1])};
        const auto jet = face.at(uv);
        gp_Pnt p;
        gp_Vec u, v;
        surface.D1(uv[0], uv[1], p, u, v);
        const double error = length(sub(jet.p, xyz(p.XYZ())));
        r.position = std::max(r.position, error);
        r.max_tolerance_ratio = std::max(r.max_tolerance_ratio, error / face.info.tolerance);
        Vec a = cross3(jet.u, jet.v), b = xyz(u.Crossed(v).XYZ());
        if (length(a) > 1e-14 && length(b) > 1e-14)
            r.normal = std::max(r.normal, std::atan2(length(cross3(a, b)), dot3(a, b)));
        else
            ++r.singular;
        ++r.samples;
    }
}
void check_spline(const Face &f, const BRepAdaptor_Surface &source, Report &r) {
    if (source.GetType() == GeomAbs_BezierSurface) {
        require(f.patches.size() == 1, "missing Bezier chart");
        compare_net(f.patches.front(), source.Bezier(), r);
    } else {
        require(source.GetType() == GeomAbs_BSplineSurface, "surface type mismatch");
        GeomConvert_BSplineSurfaceToBezierSurface ref(source.BSpline());
        TColStd_Array1OfReal u(1, ref.NbUPatches() + 1), v(1, ref.NbVPatches() + 1);
        ref.UKnots(u);
        ref.VKnots(v);
        std::set<std::pair<int, int>> present;
        for (const auto &q : f.patches) {
            int i = 1, j = 1;
            while (i < u.Upper() && u(i) != q.full.u.lo)
                ++i;
            while (j < v.Upper() && v(j) != q.full.v.lo)
                ++j;
            require(i < u.Upper() && j < v.Upper() && u(i + 1) == q.full.u.hi && v(j + 1) == q.full.v.hi,
                    "compiled span missing in source knot grid");
            require(present.emplace(i, j).second, "duplicate span");
            compare_net(q, ref.Patch(i, j), r);
        }
        for (int i = 1; i < u.Upper(); ++i)
            for (int j = 1; j < v.Upper(); ++j) {
                auto overlaps = [](double a, double b, Interval s) {
                    return std::min(b, s.hi) > std::max(a, s.lo) + 1e-12;
                };
                if (overlaps(u(i), u(i + 1), f.info.domain.u) && overlaps(v(j), v(j + 1), f.info.domain.v))
                    require(present.contains({i, j}), "source knot cell missing in compiled chart");
            }
    }
    for (const auto &q : f.patches) {
        check_sample(f, source, q.domain, r);
        ++r.spans;
    }
}
void check_primitive(const Face &f, const BRepAdaptor_Surface &source, Report &r) {
    const auto error = [](const IVec &a, const gp_XYZ &b) {
        return length(sub(Vec{a[0].center(), a[1].center(), a[2].center()}, xyz(b)));
    };
    const auto magnitude = [](Interval i) { return std::max(std::abs(i.lo), std::abs(i.hi)); };
    double bound{};
    if (const auto *p = std::get_if<Plane>(f.analytic)) {
        require(source.GetType() == GeomAbs_Plane, "plane type differs");
        auto a = source.Plane().Position();
        bound = error(p->origin, a.Location().XYZ()) +
                magnitude(f.info.domain.u) * error(p->x, a.XDirection().XYZ()) +
                magnitude(f.info.domain.v) * error(p->y, a.YDirection().XYZ());
        ++r.planes;
    } else if (const auto *p = std::get_if<Cylinder>(f.analytic)) {
        require(source.GetType() == GeomAbs_Cylinder, "cylinder type differs");
        auto cylinder = source.Cylinder();
        auto a = cylinder.Position();
        bound = error(p->origin, a.Location().XYZ()) + std::abs(cylinder.Radius() - p->radius.center()) +
                std::abs(p->radius.center()) *
                    (error(p->x, a.XDirection().XYZ()) + error(p->y, a.YDirection().XYZ())) +
                magnitude(f.info.domain.v) * error(p->z, a.Direction().XYZ());
        ++r.cylinders;
    } else
        throw std::runtime_error("whole-chart primitive verification unavailable for this chart");
    r.primitive_bound = std::max(r.primitive_bound, bound);
    r.max_tolerance_ratio = std::max(r.max_tolerance_ratio, bound / f.info.tolerance);
}
Vec uv_vec(gp_Pnt2d p) { return {p.X(), p.Y(), 0}; }
Vec mid(IVec p) { return {p[0].center(), p[1].center(), p[2].center()}; }
struct WireCheck {
    const std::vector<TrimCurve> &loop;
    Report &report;
    size_t next{};
    void skip_connectors() {
        while (next < loop.size()) {
            const auto &c = loop[next];
            const auto *b = std::get_if<BezierCurve>(&c.geometry);
            auto same = [&](Feature x) { return c.feature.id == x.id && c.feature.kind == x.kind; };
            if (!b || b->poles.size() != 2 || !same(c.start) || !same(c.end))
                break;
            auto box = b->box();
            const double extent = std::hypot(box[0].width(), box[1].width());
            if (extent > 1e-5)
                break;
            report.connector_extent = std::max(report.connector_extent, extent);
            ++report.connectors;
            ++next;
        }
    }
    const Curve &take(Feature id) {
        skip_connectors();
        require(next < loop.size(), "missing source trim span");
        const auto &c = loop[next++];
        require(id.kind == c.feature.kind && id.id == c.feature.id, "trim traversal/feature differs");
        ++report.trim_spans;
        return c.geometry;
    }
    void bezier(Feature id, const Handle(Geom2d_BezierCurve) & ref, bool reverse) {
        const auto *b = std::get_if<BezierCurve>(&take(id));
        require(b && b->poles.size() == static_cast<size_t>(ref->NbPoles()), "trim degree differs");
        auto index = [&](int i) { return reverse ? ref->NbPoles() - i : i + 1; };
        double da = 0, dw = 0, minw = infinity, radius = 0;
        // Interval hulls also enclose roundoff from the serialized trim net.
        for (int i = 0; i < ref->NbPoles(); ++i) {
            const auto &h = b->poles[i];
            const auto w = h[3] / b->poles.front()[3];
            const double v = ref->Weight(index(i)) / ref->Weight(index(0));
            auto p = uv_vec(ref->Pole(index(i)));
            Vec component{};
            for (int k = 0; k < 3; ++k) {
                auto d = h[k] / b->poles.front()[3] - Interval{p[k]} * v;
                component[k] = std::max(std::abs(d.lo), std::abs(d.hi));
            }
            da = std::max(da, length(component));
            dw = std::max({dw, std::abs(w.lo - v), std::abs(w.hi - v)});
            minw = std::min(minw, w.lo);
            radius = std::max(radius, length(p));
        }
        require(minw > 0, "nonpositive trim weight");
        report.trim_bound = std::max(report.trim_bound, (da + dw * radius) / minw);
    }
    void edge(Feature id, const Handle(Geom2d_Curve) & curve, double lo, double hi, bool reverse) {
        Geom2dAdaptor_Curve a(curve, lo, hi);
        if (a.GetType() == GeomAbs_Line) {
            TColgp_Array1OfPnt2d p(1, 2);
            p(1) = curve->Value(lo);
            p(2) = curve->Value(hi);
            bezier(id, new Geom2d_BezierCurve(p), reverse);
        } else if (a.GetType() == GeomAbs_BSplineCurve) {
            Geom2dConvert_BSplineCurveToBezierCurve ref(a.BSpline());
            const int n = ref.NbArcs();
            TColStd_Array1OfReal knots(1, n + 1);
            ref.Knots(knots);
            for (int k = 1; k <= n; ++k) {
                const int i = reverse ? n + 1 - k : k;
                double low = i == 1 ? lo : std::max(lo, knots(i)),
                       high = i == n ? hi : std::min(hi, knots(i + 1));
                if (low >= high)
                    continue;
                auto piece = ref.Arc(i);
                if (low != knots(i) || high != knots(i + 1))
                    piece->Segment((low - knots(i)) / (knots(i + 1) - knots(i)),
                                   (high - knots(i)) / (knots(i + 1) - knots(i)));
                bezier(id, piece, reverse);
            }
        } else if (a.GetType() == GeomAbs_BezierCurve) {
            auto ref = Handle(Geom2d_BezierCurve)::DownCast(a.Bezier()->Copy());
            ref->Segment(lo, hi);
            bezier(id, ref, reverse);
        } else {
            const auto *arc = std::get_if<Arc>(&take(id));
            require(arc && arc->angles.lo == lo && arc->angles.hi == hi && arc->reversed == reverse,
                    "conic trim range differs");
            require(a.GetType() == GeomAbs_Circle || a.GetType() == GeomAbs_Ellipse,
                    "unsupported trim verification");
            auto frame = a.GetType() == GeomAbs_Circle ? a.Circle().Axis() : a.Ellipse().Axis();
            double rx = a.GetType() == GeomAbs_Circle ? a.Circle().Radius() : a.Ellipse().MajorRadius();
            double ry = a.GetType() == GeomAbs_Circle ? rx : a.Ellipse().MinorRadius();
            auto x = frame.XDirection(), y = frame.YDirection();
            double bound = length(sub(mid(arc->center), uv_vec(frame.Location()))) +
                           length(sub(mid(arc->x), Vec{rx * x.X(), rx * x.Y(), 0})) +
                           length(sub(mid(arc->y), Vec{ry * y.X(), ry * y.Y(), 0}));
            report.trim_bound = std::max(report.trim_bound, bound);
        }
    }
};
void check_trims(const Face &f, TopoDS_Face source, const TopTools_IndexedMapOfShape &edges,
                 const BRepAdaptor_Surface &surface, size_t face_id, Report &r) {
    source.Orientation(TopAbs_FORWARD);
    BRepTopAdaptor_FClass2d classifier(source, 1e-9);
    for (int i = 0; i < 11; ++i)
        for (int j = 0; j < 11; ++j) {
            Param uv{mix(f.info.domain.u, (i + .37) / 11), mix(f.info.domain.v, (j + .63) / 11)};
            auto state = classifier.Perform(gp_Pnt2d(uv[0], uv[1]), false);
            if (state == TopAbs_IN || state == TopAbs_OUT) {
                require(f.inside(uv) == (state == TopAbs_IN), "material/void classification differs");
                ++r.membership;
            }
        }
    size_t wi = 0;
    for (TopExp_Explorer w(source, TopAbs_WIRE); w.More(); w.Next(), ++wi) {
        require(wi < f.trim->loops.size(), "missing compiled wire");
        struct Edge {
            Handle(Geom2d_Curve) curve;
            double lo, hi;
            size_t id;
            bool seam;
        };
        std::vector<Edge> native;
        std::set<size_t> ids, seen;
        WireCheck check{f.trim->loops[wi], r};
        for (BRepTools_WireExplorer it(TopoDS::Wire(w.Current()), source); it.More(); it.Next()) {
            auto edge = it.Current();
            double lo, hi;
            auto curve = BRep_Tool::CurveOnSurface(edge, source, lo, hi);
            size_t id = edges.FindIndex(edge);
            bool seam = BRep_Tool::IsClosed(edge, source);
            check.edge(seam ? Feature{FeatureKind::face, face_id} : Feature{FeatureKind::edge, id}, curve, lo,
                       hi, edge.Orientation() == TopAbs_REVERSED);
            native.push_back({curve, lo, hi, id, seam});
            if (!seam)
                ids.insert(id);
        }
        check.skip_connectors();
        require(check.next == check.loop.size(), "extra compiled trim span");
        for (const auto &c : f.trim->loops[wi]) {
            if (c.feature.kind == FeatureKind::edge)
                seen.insert(c.feature.id);
            for (double s : {0., .5, 1.}) {
                auto uv = evaluate(c.geometry, s);
                gp_Pnt2d x(uv[0].center(), uv[1].center()), closest;
                double best = infinity;
                auto consider = [&](gp_Pnt2d y) {
                    if (double d = x.SquareDistance(y); d < best) {
                        best = d;
                        closest = y;
                    }
                };
                for (const auto &e : native) {
                    if (c.feature.kind == FeatureKind::edge && e.id != c.feature.id)
                        continue;
                    consider(e.curve->Value(e.lo));
                    consider(e.curve->Value(e.hi));
                    if (best == 0)
                        break;
                    Geom2dAPI_ProjectPointOnCurve proj(x, e.curve, e.lo, e.hi);
                    if (proj.NbPoints())
                        consider(proj.NearestPoint());
                }
                require(std::isfinite(best), "unmatched trim edge");
                const auto a = f.at({x.X(), x.Y()}).p;
                auto b = surface.Value(closest.X(), closest.Y());
                const double distance = length(sub(a, xyz(b.XYZ())));
                r.boundary = std::max(r.boundary, distance);
                r.max_tolerance_ratio = std::max(r.max_tolerance_ratio, distance / f.info.tolerance);
                ++r.trim_samples;
            }
        }
        require(ids == seen, "wire edge incidence differs");
        ++r.wires;
    }
    require(wi == f.trim->loops.size(), "extra compiled wire");
}
} // namespace
int main(int argc, char **argv) {
    try {
        require(argc == 3, "nc_check_part SOURCE.step COMPILED.ncpart");
        const auto shape = nc::read_step_shape(argv[1]);
        TopTools_IndexedMapOfShape faces, edges;
        TopExp::MapShapes(shape, TopAbs_FACE, faces);
        TopExp::MapShapes(shape, TopAbs_EDGE, edges);
        const auto solid = nc::compiled::load_solid(argv[2]);
        require(solid->faces.size() == static_cast<size_t>(faces.Extent()), "face count differs");
        Report r;
        for (int i = 0; i < faces.Extent(); ++i) {
            const auto &f = solid->faces[i];
            auto face = TopoDS::Face(faces(i + 1));
            require(f.info.orientation == (face.Orientation() == TopAbs_REVERSED ? -1 : 1),
                    "face orientation differs");
            require(f.info.tolerance == BRep_Tool::Tolerance(face), "face tolerance differs");
            double u, U, v, V;
            BRepTools::UVBounds(face, u, U, v, V);
            require(std::abs(u - f.info.domain.u.lo) < 1e-9 && std::abs(U - f.info.domain.u.hi) < 1e-9 &&
                        std::abs(v - f.info.domain.v.lo) < 1e-9 && std::abs(V - f.info.domain.v.hi) < 1e-9,
                    "UV range differs");
            BRepAdaptor_Surface surface(face);
            if (f.analytic) {
                check_primitive(f, surface, r);
                check_sample(f, surface, f.info.domain, r);
                ++r.analytic;
            } else
                check_spline(f, surface, r);
            check_trims(f, face, edges, surface, i + 1, r);
            ++r.faces;
        }
        std::cout << std::setprecision(17) << "{\"faces\":" << r.faces << ",\"analytic_faces\":" << r.analytic
                  << ",\"planes\":" << r.planes << ",\"cylinders\":" << r.cylinders
                  << ",\"max_primitive_bound_mm\":" << r.primitive_bound << ",\"spline_spans\":" << r.spans
                  << ",\"poles\":" << r.poles << ",\"surface_samples\":" << r.samples
                  << ",\"singular_normals\":" << r.singular << ",\"wires\":" << r.wires
                  << ",\"trim_samples\":" << r.trim_samples << ",\"source_trim_spans\":" << r.trim_spans
                  << ",\"trim_connectors\":" << r.connectors << ",\"max_trim_net_bound_uv\":" << r.trim_bound
                  << ",\"max_connector_extent_uv\":" << r.connector_extent
                  << ",\"membership_tests\":" << r.membership << ",\"max_position_mm\":" << r.position
                  << ",\"max_normal_rad\":" << r.normal << ",\"max_boundary_mm\":" << r.boundary
                  << ",\"max_control_net_bound_mm\":" << r.net_bound
                  << ",\"max_source_tolerance_fraction\":" << r.max_tolerance_ratio << "}\n";
        require(r.max_tolerance_ratio <= 1 && r.net_bound < 1e-7 && r.normal < 1e-6 && r.trim_bound < 1e-7,
                "exceeds source geometry, trim or normal limit");
    } catch (const Standard_Failure &e) {
        std::cerr << e.GetMessageString() << '\n';
        return 1;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
