#pragma once

#include <BRepTools_WireExplorer.hxx>
#include <BRepTopAdaptor_FClass2d.hxx>
#include <BRep_Tool.hxx>
#include <BndLib_Add2dCurve.hxx>
#include <Bnd_Box2d.hxx>
#include <Geom2dAdaptor_Curve.hxx>
#include <TopExp_Explorer.hxx>
#include <Precision.hxx>
#include <Standard_Failure.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Wire.hxx>
#include <gp_Pln.hxx>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace nc::cad {
// The curved trim segment and its chord occupy the same conservative box.
// Winding parity away from all such boxes is therefore unchanged by replacing
// each trim segment with its chord. The planar chart uses physical STEP units.
class PlanarTrim {
    struct Fragment {
        gp_Pnt2d a, b;
        double u_min, v_min, u_max, v_max;
    };
    std::vector<Fragment> fragments_;
    gp_Trsf frame_;
    double tolerance_{};
    void add(const gp_Pnt2d& a, const gp_Pnt2d& b, const Bnd_Box2d& box) {
        Fragment f{a, b};
        box.Get(f.u_min, f.v_min, f.u_max, f.v_max);
        fragments_.push_back(f);
    }
    bool read(const TopoDS_Face& face) {
        if (BRepTopAdaptor_FClass2d(face, Precision::PConfusion()).PerformInfinitePoint() != TopAbs_OUT)
            return false;
        for (TopExp_Explorer it(face, TopAbs_WIRE); it.More(); it.Next()) {
            const auto wire = TopoDS::Wire(it.Current());
            int expected = 0, visited = 0;
            for (TopExp_Explorer e(wire, TopAbs_EDGE); e.More(); e.Next())
                ++expected;
            gp_Pnt2d first, last;
            for (BRepTools_WireExplorer e(wire, face); e.More(); e.Next()) {
                const auto& edge = e.Current();
                if (edge.Orientation() != TopAbs_FORWARD && edge.Orientation() != TopAbs_REVERSED)
                    return false;
                tolerance_ = std::max(tolerance_, BRep_Tool::Tolerance(edge));
                double u, v;
                const auto curve = BRep_Tool::CurveOnSurface(edge, face, u, v);
                if (curve.IsNull() || !std::isfinite(u) || !std::isfinite(v) || u >= v)
                    return false;
                Geom2dAdaptor_Curve adaptor(curve, u, v);
                const auto kind = adaptor.GetType();
                if (kind != GeomAbs_Line && kind != GeomAbs_Circle && kind != GeomAbs_Ellipse &&
                    kind != GeomAbs_BSplineCurve && kind != GeomAbs_BezierCurve)
                    return false;
                if (edge.Orientation() == TopAbs_REVERSED)
                    std::swap(u, v);
                auto a = curve->Value(u);
                if (!visited)
                    first = a;
                else if (!join(last, a))
                    return false;
                const int count = kind == GeomAbs_Line ? 1 : 16;
                for (int j = 0; j < count; ++j) {
                    const double lo = u + (v - u) * j / count,
                                 hi = (j + 1 == count ? v : u + (v - u) * (j + 1) / count);
                    const auto b = curve->Value(hi);
                    Bnd_Box2d box;
                    BndLib_Add2dCurve::Add(adaptor, std::min(lo, hi), std::max(lo, hi), tolerance_, box);
                    if (box.IsVoid() || box.IsOpenXmin() || box.IsOpenXmax() || box.IsOpenYmin() ||
                        box.IsOpenYmax())
                        return false;
                    box.Add(a);
                    box.Add(b);
                    add(a, b, box);
                    a = b;
                }
                last = a;
                ++visited;
            }
            if (!visited || visited != expected || !join(last, first))
                return false;
        }
        return !fragments_.empty();
    }
    bool join(const gp_Pnt2d& a, const gp_Pnt2d& b) {
        if (a.Distance(b) > 2 * tolerance_)
            return false;
        if (a.X() != b.X() || a.Y() != b.Y()) {
            Bnd_Box2d box;
            box.Add(a);
            box.Add(b);
            box.Enlarge(tolerance_);
            add(a, b, box);
        }
        return true;
    }

  public:
    explicit PlanarTrim(const TopoDS_Face& face, const gp_Pln& plane) {
        frame_.SetTransformation(plane.Position());
        tolerance_ = std::max(Precision::Confusion(), BRep_Tool::Tolerance(face));
        try {
            if (!read(TopoDS::Face(face.Oriented(TopAbs_FORWARD))))
                fragments_.clear();
        } catch (const Standard_Failure&) {
            fragments_.clear();
        }
    }
    bool separates(const std::vector<gp_XYZ>& poles, const gp_Trsf& in_plane_body, double limit) const {
        if (fragments_.empty() || poles.size() > 64)
            return false;
        const auto relative = frame_ * in_plane_body;
        std::vector<gp_XYZ> points;
        points.reserve(poles.size());
        double scale = 1 + relative.TranslationPart().Modulus() + limit;
        for (auto p : poles) {
            scale = std::max(scale, p.Modulus() + relative.TranslationPart().Modulus());
            relative.Transforms(p);
            points.push_back(p);
        }
        const double pad = limit + tolerance_ + 128 * std::numeric_limits<double>::epsilon() * scale;
        // Enclose the projection of the hull clipped by |z| <= pad. Every
        // vertex of this clip is an original vertex or an edge/plane crossing.
        // All vertex pairs include the edges without constructing a hull.
        std::vector<gp_XYZ> footprint;
        for (const auto& p : points)
            if (std::abs(p.Z()) <= pad)
                footprint.push_back(p);
        for (std::size_t i = 0; i < points.size(); ++i)
            for (std::size_t j = 0; j < i; ++j) {
                const auto &p = points[i], &q = points[j];
                for (const double h : {-pad, pad})
                    if ((p.Z() < h && q.Z() > h) || (q.Z() < h && p.Z() > h)) {
                        footprint.push_back(p + (q - p) * ((h - p.Z()) / (q.Z() - p.Z())));
                    }
            }
        if (footprint.empty())
            return true;
        const auto p = footprint.front();
        double u_min = p.X(), u_max = u_min, v_min = p.Y(), v_max = v_min;
        for (const auto& v : footprint) {
            u_min = std::min(u_min, v.X());
            u_max = std::max(u_max, v.X());
            v_min = std::min(v_min, v.Y());
            v_max = std::max(v_max, v.Y());
        }
        bool inside = false;
        for (const auto& f : fragments_) {
            if (f.u_max >= u_min - pad && f.u_min <= u_max + pad && f.v_max >= v_min - pad &&
                f.v_min <= v_max + pad) {
                const double nx = f.a.Y() - f.b.Y(), ny = f.b.X() - f.a.X();
                const double lo = nx * (nx >= 0 ? f.u_min : f.u_max) + ny * (ny >= 0 ? f.v_min : f.v_max);
                const double hi = nx * (nx >= 0 ? f.u_max : f.u_min) + ny * (ny >= 0 ? f.v_max : f.v_min);
                const double gap = pad * std::hypot(nx, ny);
                bool below = true, above = true;
                for (const auto& v : footprint) {
                    const double t = nx * v.X() + ny * v.Y();
                    below &= t < lo - gap;
                    above &= t > hi + gap;
                }
                if (!below && !above)
                    return false;
            }
            if ((f.a.Y() > p.Y()) != (f.b.Y() > p.Y())) {
                const double t = (p.Y() - f.a.Y()) / (f.b.Y() - f.a.Y());
                if (f.a.X() + t * (f.b.X() - f.a.X()) > p.X())
                    inside = !inside;
            }
        }
        return !inside;
    }
};
} // namespace nc::cad
