#include "flank.hpp"
#include "../step_triangles.hpp"
#include "nurbscontact/occt.hpp"

#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakeSolid.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepBuilderAPI_Sewing.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepLib.hxx>
#include <BRep_Tool.hxx>
#include <BRepTools.hxx>
#include <GeomAPI_PointsToBSplineSurface.hxx>
#include <GeomConvert_BSplineCurveToBezierCurve.hxx>
#include <Geom_BSplineCurve.hxx>
#include <Geom_BSplineSurface.hxx>
#include <Geom_BezierCurve.hxx>
#include <STEPControl_Writer.hxx>
#include <TColgp_Array2OfPnt.hxx>
#include <TColStd_Array1OfReal.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Solid.hxx>
#include <TopoDS_Wire.hxx>
#include <gp_Circ.hxx>
#include <gp_Cylinder.hxx>
#include <gp_Pln.hxx>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>

namespace nc::gears {
namespace {

using namespace dynamics;
using Path = std::filesystem::path;
constexpr double tolerance = 1e-6;
constexpr double ray_half_length = 2.0;

gp_Pnt point(Vec a) { return {a[0], a[1], a[2]}; }
Vec vector(const gp_XYZ& a) { return {a.X(), a.Y(), a.Z()}; }
double involute(double t) { return t - std::atan(t); }

Vec polar(double radius, double angle, double z) {
    return {radius*std::cos(angle), radius*std::sin(angle), z};
}

enum class Strip { rising_root, negative, tip, positive, falling_root, root };
constexpr std::array strips{Strip::rising_root, Strip::negative, Strip::tip, Strip::positive,
                            Strip::falling_root, Strip::root};

Vec strip_point(const Gear& g, Strip strip, double u, double v) {
    const double z = g.width*(v - 0.5);
    const double phase = g.twist()*z;
    const double base_half = g.half_angle(g.roll_min());
    const double tip_half = g.half_angle(g.roll_max());
    const double flank_start = g.base_radius()*std::hypot(1., g.roll_min());
    switch (strip) {
    case Strip::positive: return g.flank(u, v);
    case Strip::negative: {
        const double roll = std::lerp(g.roll_min(), g.roll_max(), u);
        return polar(g.base_radius()*std::hypot(1., roll), phase-g.half_angle(roll), z);
    }
    case Strip::rising_root: return polar(std::lerp(g.root_radius(), flank_start, u), phase-base_half, z);
    case Strip::tip: return polar(g.tip_radius(), phase+std::lerp(-tip_half,tip_half,u), z);
    case Strip::falling_root: return polar(std::lerp(flank_start, g.root_radius(), u), phase+base_half, z);
    case Strip::root: return polar(g.root_radius(), phase+std::lerp(base_half,2*pi/g.teeth-base_half,u), z);
    }
    throw std::logic_error("unhandled gear strip");
}

Handle(Geom_BSplineSurface) surface(const Gear& g, Strip strip) {
    const int profile_sites = (strip == Strip::positive || strip == Strip::negative)
        ? (g.measured_relief.empty() ? 23 : 81) : 7;
    const int axial_sites = 7;
    TColgp_Array2OfPnt samples(1, profile_sites, 1, axial_sites);
    for (int i = 1; i <= profile_sites; ++i) {
        for (int j = 1; j <= axial_sites; ++j) {
            const double u = double(i-1)/(profile_sites-1);
            const double v = double(j-1)/(axial_sites-1);
            samples.SetValue(i,j,point(strip_point(g,strip,u,v)));
        }
    }
    GeomAPI_PointsToBSplineSurface interpolator;
    interpolator.Interpolate(samples, Approx_IsoParametric);
    if (!interpolator.IsDone()) throw std::runtime_error("gear CAD surface interpolation failed");
    return interpolator.Surface();
}

void step(const TopoDS_Shape& model, const Path& path) {
    STEPControl_Writer writer;
    if (writer.Transfer(model, STEPControl_AsIs) != IFSelect_RetDone ||
        writer.Write(path.string().c_str()) != IFSelect_RetDone) {
        throw std::runtime_error("cannot export STEP " + path.string());
    }
}

}  // namespace

double Gear::transverse_module() const { return module / std::cos(helix); }
double Gear::transverse_pressure() const { return std::atan(std::tan(pressure) / std::cos(helix)); }
double Gear::pitch_radius() const { return transverse_module()*teeth/2; }
double Gear::base_radius() const { return pitch_radius()*std::cos(transverse_pressure()); }
double Gear::tip_radius() const { return pitch_radius() + module; }
double Gear::root_radius() const { return pitch_radius() - 1.25*module; }
double Gear::twist() const { return std::tan(helix) / pitch_radius(); }
double Gear::roll_min() const {
    const double ratio = std::max(1., root_radius()/base_radius());
    return std::sqrt(ratio*ratio-1) + 0.001;
}
double Gear::roll_max() const {
    const double ratio = tip_radius()/base_radius();
    return std::sqrt(ratio*ratio-1);
}
double Gear::bore() const { return bore_radius > 0 ? bore_radius : 0.32*pitch_radius(); }
double Gear::half_angle(double roll) const {
    double relief = 0;
    if (!measured_relief.empty()) {
        const auto next=std::upper_bound(measured_relief.begin(),measured_relief.end(),roll,
            [](double value,const auto& point) { return value<point[0]; });
        if (next==measured_relief.begin()) {
            relief=next->at(1);
        } else if (next==measured_relief.end()) {
            relief=measured_relief.back()[1];
        } else {
            const auto& previous=*(next-1);
            relief=std::lerp(previous[1],(*next)[1],(roll-previous[0])/((*next)[0]-previous[0]));
        }
    } else if (tip_relief>0) {
        constexpr double start = radians(24);
        relief=tip_relief*std::max(0.,(roll-start)/(roll_max()-start));
    }
    return pi/(2*teeth)+involute(std::tan(transverse_pressure()))-involute(roll)
        -(backlash/4+relief)/base_radius();
}

Vec Gear::flank(double u, double v) const {
    const double roll = std::lerp(roll_min(), roll_max(), u);
    const double z = width*(v - 0.5);
    return polar(base_radius()*std::hypot(1., roll), half_angle(roll)+twist()*z,z);
}

void export_gear(const Gear& gear, const Path& solid, const Path& flank, bool mirror) {
    BRepBuilderAPI_Sewing sewing(tolerance);
    std::array<BRepBuilderAPI_MakeWire, 2> end_wires;
    std::array<Handle(Geom_BSplineSurface), strips.size()> basis;
    for (std::size_t i = 0; i < strips.size(); ++i) basis[i] = surface(gear,strips[i]);
    step(BRepBuilderAPI_MakeFace(basis[3], tolerance).Face(),flank);
    for (int tooth = 0; tooth < gear.teeth; ++tooth) {
        gp_Trsf spin;
        spin.SetRotation(gp_Ax1(gp_Pnt(0,0,0),gp_Dir(0,0,1)), 2*pi*tooth/gear.teeth);
        for (std::size_t i = 0; i < strips.size(); ++i) {
            const Handle(Geom_Surface) sheet = Handle(Geom_Surface)::DownCast(basis[i]->Transformed(spin));
            sewing.Add(BRepBuilderAPI_MakeFace(sheet,tolerance).Face());
            double u0,u1,v0,v1;
            sheet->Bounds(u0,u1,v0,v1);
            for (std::size_t end = 0; end < end_wires.size(); ++end) {
                auto edge = BRepBuilderAPI_MakeEdge(sheet->VIso(end == 0 ? v0 : v1),u0,u1).Edge();
                if (strips[i] == Strip::positive) edge.Reverse();
                end_wires[end].Add(edge);
            }
        }
    }
    for (std::size_t end = 0; end < end_wires.size(); ++end) {
        if (!end_wires[end].IsDone()) throw std::runtime_error("gear end profile is open");
        const double z = (double(end)-0.5)*gear.width;
        const gp_Ax2 axis(gp_Pnt(0,0,z),gp_Dir(0,0,1));
        const auto circle = BRepBuilderAPI_MakeEdge(gp_Circ(axis,gear.bore())).Edge();
        auto hole = BRepBuilderAPI_MakeWire(circle).Wire();
        hole.Reverse();
        BRepBuilderAPI_MakeFace face(gp_Pln(axis.Location(), axis.Direction()),end_wires[end].Wire(),true);
        face.Add(hole);
        sewing.Add(face.Face());
    }
    const gp_Cylinder bore(gp_Ax3(gp_Pnt(0,0,0),gp_Dir(0,0,1)),gear.bore());
    sewing.Add(BRepBuilderAPI_MakeFace(bore,0,2*pi,-gear.width/2,gear.width/2).Face());
    sewing.Perform();
    const auto shell = sewing.SewedShape();
    if (shell.IsNull() || shell.ShapeType() != TopAbs_SHELL) {
        throw std::runtime_error("gear patches did not sew to one shell");
    }
    auto gear_solid = BRepBuilderAPI_MakeSolid(TopoDS::Shell(shell)).Solid();
    if (!BRepLib::OrientClosedSolid(gear_solid) || !BRepCheck_Analyzer(gear_solid).IsValid()) {
        throw std::runtime_error("sewn gear solid is invalid: " + solid.string());
    }
    if (mirror) {
        gp_Trsf reflection;
        reflection.SetMirror(gp_Ax2(gp_Pnt(0,0,0),gp_Dir(0,1,0)));
        gear_solid = TopoDS::Solid(BRepBuilderAPI_Transform(gear_solid,reflection,true).Shape());
    }
    step(gear_solid,solid);
}

Flank::Flank(const Path& path, double deflection, unsigned stations) {
    const auto shape = read_step_shape(path);
    TopExp_Explorer faces(shape, TopAbs_FACE);
    if (!faces.More()) throw std::runtime_error("missing STEP flank");
    const auto topo_face = TopoDS::Face(faces.Current());
    face = BRep_Tool::Surface(topo_face);
    BRepTools::UVBounds(topo_face,u0,u1,v0,v1);
    if (deflection > 0) mesh.emplace(bench::step_triangles(shape,deflection),4);
    if (deflection <= -16 && deflection >= -16384 && std::floor(deflection) == deflection) {
        const auto segments = static_cast<unsigned>(-deflection);
        for (unsigned i=0; i<stations; ++i) {
            const double v = std::lerp(v0,v1,(i+.5)/stations);
            auto& points = polylines_.emplace_back();
            points.reserve(segments+1);
            for (unsigned j=0; j<=segments; ++j) {
                const double u = std::lerp(u0,u1,static_cast<double>(j)/segments);
                points.push_back(vector(face->Value(u,v).XYZ()));
            }
        }
    }
    if (deflection != 0) return;
    for (unsigned i = 0; i < stations; ++i) {
        const double v = std::lerp(v0,v1,(i+.5)/stations);
        const auto curve = Handle(Geom_BSplineCurve)::DownCast(face->VIso(v));
        if (curve.IsNull() || curve->IsRational() || curve->Degree()!=3) {
            throw std::runtime_error("gear section requires a nonrational cubic STEP isocurve");
        }
        GeomConvert_BSplineCurveToBezierCurve decomposed(curve,u0,u1,1e-12);
        TColStd_Array1OfReal knots(1,decomposed.NbArcs()+1);
        decomposed.Knots(knots);
        auto& spans = sections_.emplace_back();
        SectionBound section{};
        for (int j=1; j<=decomposed.NbArcs(); ++j) {
            const auto arc = decomposed.Arc(j);
            const Vec p0=vector(arc->Pole(1).XYZ()), p1=vector(arc->Pole(2).XYZ());
            const Vec p2=vector(arc->Pole(3).XYZ()), p3=vector(arc->Pole(4).XYZ());
            const Vec d0=sub(p1,p0), d1=sub(p2,p1), d2=sub(p3,p2);
            const double width=knots(j+1)-knots(j);
            const std::array coefficients{p0,scale(d0,3),scale(sub(d1,d0),3),add(sub(d2,scale(d1,2)),d0)};
            const IVec h0 = exact(p1) - exact(p0);
            const IVec h1 = exact(p2) - exact(p1);
            const IVec h2 = exact(p3) - exact(p2);
            const std::array enclosed{exact(p0), h0*3, (h1-h0)*3, h2-h1*2+h0};
            spans.push_back({knots(j),knots(j+1),1/width,coefficients,enclosed,
                             {exact(p0),exact(p1),exact(p2),exact(p3)}});
            const Interval span_width = Interval{knots(j+1)} - knots(j);
            if (j == 1) {
                section.derivative = (h0*3)/span_width;
            } else if (midpoint(spans[spans.size()-2].control.back()) != p0) {
                section.continuous = false;
            }
            for (const IVec& d : {h0, h1, h2}) section.derivative = hull(section.derivative, (d*3)/span_width);
        }
        section_bounds_.push_back(section);
    }
}

double flank_fit(const Gear& g, const Flank& f) {
    double error = 0;
    for (int i = 0; i <= 80; ++i) {
        for (int j = 0; j <= 22; ++j) {
            const double u = i/80., v = j/22.;
            const auto p = f.face->Value(std::lerp(f.u0,f.u1,u),std::lerp(f.v0,f.v1,v));
            error = std::max(error, length(sub(g.flank(u,v),vector(p.XYZ()))));
        }
    }
    return error;
}

std::optional<double> Flank::probe(Vec p, Vec n, double seed, double axial, unsigned station,
                                   QueryWork& work, double* slope, double* parameter) const {
    ++work.rays;
    const Vec tangent{n[1],-n[0],0};
    if (!polylines_.empty()) {
        const auto& points = polylines_[station];
        const auto project = [&](std::size_t index) {
            ++work.evaluations;
            return dot3(sub(points[index],p),tangent);
        };
        // Bracket a crossing near the prescribed line-of-action seed.
        // Select successive segments along the initial sign-change direction.
        const auto last = points.size()-1;
        const auto guess = static_cast<std::size_t>(std::clamp(seed,0.,1.) * last);
        std::size_t lo = 0, hi = last;
        double a = project(lo), b = project(hi);
        if ((a <= 0 && b >= 0) || (b <= 0 && a >= 0)) {
            bool use_guess = guess != lo && guess != hi;
            while (hi-lo > 1) {
                const std::size_t mid = use_guess ? guess : std::midpoint(lo,hi);
                use_guess = false;
                const double f = project(mid);
                if ((a <= 0 && f >= 0) || (a >= 0 && f <= 0)) {
                    hi = mid;
                    b = f;
                } else {
                    lo = mid;
                    a = f;
                }
            }
            const Vec direction = sub(points[hi],points[lo]);
            const double s = a == b ? 0 : -a/(b-a);
            const double distance = dot3(sub(add(points[lo],scale(direction,s)),p),n);
            if (std::abs(distance) <= ray_half_length) {
                if (slope) {
                    if (a == b) return std::nullopt;
                    *slope = dot3(direction,n)/(b-a);
                }
                return distance;
            }
        }
        ++work.misses;
        return std::nullopt;
    }
    if (mesh) {
        Vec normal;
        const auto hit = mesh->ray_first(sub(p,scale(n,ray_half_length)),n,2*ray_half_length,work.mesh,
                                        slope ? &normal : nullptr);
        if (hit) {
            if (slope) *slope=-dot3(normal,tangent)/dot3(normal,n);
            return *hit-ray_half_length;
        }
        ++work.misses;
        return std::nullopt;
    }
    double u = std::lerp(u0,u1,std::clamp(seed,0.,1.));
    const double v = std::lerp(v0,v1,std::clamp(axial,0.,1.));
    double best_u = u;
    double best_residual = std::numeric_limits<double>::infinity();
    std::optional<double> best_distance;
    for (unsigned iteration = 0; iteration < 8; ++iteration) {
        Vec value,du;
        if (sections_.empty()) {
            gp_Pnt native;
            gp_Vec derivative,dv;
            face->D1(u,v,native,derivative,dv);
            value=vector(native.XYZ());
            du=vector(derivative.XYZ());
        } else {
            const auto& spans=sections_[station];
            const auto iter=std::upper_bound(spans.begin(),spans.end(),u,
                                             [](double x,const Span& s){return x<s.high;});
            const Span& span = iter==spans.end() ? spans.back() : *iter;
            const double t=(u-span.low)*span.inverse_width;
            const auto& c=span.coefficients;
            for (unsigned i=0; i<3; ++i) {
                value[i]=((c[3][i]*t+c[2][i])*t+c[1][i])*t+c[0][i];
                du[i]=((3*c[3][i]*t+2*c[2][i])*t+c[1][i])*span.inverse_width;
            }
        }
        ++work.evaluations;
        const Vec displacement = sub(value,p);
        const double residual = dot3(displacement,tangent);
        if (std::abs(residual) < 2e-10) {
            const double distance = dot3(displacement,n);
            if (std::abs(distance) <= ray_half_length) {
                if (slope) {
                    const double derivative=dot3(du,tangent);
                    if (std::abs(derivative)<1e-10) break;
                    *slope=dot3(du,n)/derivative;
                }
                if (!parameter) return distance;
                if (std::abs(residual) < best_residual) {
                    best_residual = std::abs(residual);
                    best_u = u;
                    best_distance = distance;
                }
                if (best_residual <= 4e-14) break;
            } else {
                break;
            }
        }
        const double derivative = dot3(du,tangent);
        if (std::abs(derivative) < 1e-10) break;
        const double next = std::clamp(u-residual/derivative,u0,u1);
        if (next == u) break;
        u=next;
    }
    if (best_distance) {
        *parameter = best_u;
        return best_distance;
    }
    ++work.misses;
    return std::nullopt;
}

std::optional<Interval> Flank::probe_bound(Vec p, Vec n, double seed, double axial, unsigned station,
                                           double target, QueryWork& work, std::optional<Hint>* cached_point) const {
    if (mesh || station >= sections_.size()) throw std::invalid_argument("bounded probe requires a cubic CAD section");
    const auto& spans = sections_[station];
    const auto& section = section_bounds_[station];
    const IVec origin = exact(p), normal = exact(n), tangent = exact({n[1], -n[0], 0});
    if (!section.continuous) return std::nullopt;
    Interval derivative = dot(section.derivative, tangent);
    double along_bound = magnitude(dot(section.derivative, normal));
    auto begin = spans.begin();
    auto end = spans.end();
    if (derivative.contains(0)) {
        // Bernstein hulls discard spans whose transverse projection cannot
        // cross the ray. Prove monotonicity over the remaining contiguous range.
        const auto may_intersect = [&](const Span& span) {
            Interval projected = dot(span.control[0]-origin,tangent);
            for (unsigned i=1; i<4; ++i) projected = hull(projected,dot(span.control[i]-origin,tangent));
            return projected.contains(0);
        };
        begin = std::find_if(begin,end,may_intersect);
        if (begin == end) return std::nullopt;
        while (!may_intersect(*(end-1))) --end;
        derivative = {infinity,-infinity};
        along_bound = 0;
        for (auto it=begin; it!=end; ++it) {
            const Interval width = Interval{it->high} - it->low;
            for (unsigned i=1; i<4; ++i) {
                const IVec edge = (it->control[i]-it->control[i-1])*3;
                derivative = hull(derivative,dot(edge,tangent)/width);
                along_bound = std::max(along_bound,magnitude(dot(edge,normal)/width));
            }
        }
        if (derivative.contains(0)) return std::nullopt;
    }
    const Interval first = dot(begin->control.front() - origin, tangent);
    const Interval last = dot((end-1)->control.back() - origin, tangent);
    if (!(first.hi <= 0 && last.lo >= 0) && !(last.hi <= 0 && first.lo >= 0)) return std::nullopt;
    const double min_derivative = std::min(std::abs(derivative.lo), std::abs(derivative.hi));
    const Interval sensitivity = Interval{along_bound} / min_derivative;
    // The transverse residual bounds the parameter error. The longitudinal
    // derivative then transports that error to the returned ray coordinate.
    const auto ray_range = [&](const IVec& point, Interval slope_bound) {
        const IVec offset = point - origin;
        const double error = (magnitude(dot(offset, tangent)) * slope_bound).hi;
        return dot(offset, normal) + Interval{-error, error};
    };
    if (cached_point && *cached_point && target > 0
        && (*cached_point)->parameter >= begin->low && (*cached_point)->parameter <= (end-1)->high) {
        const Interval bracket = ray_range((*cached_point)->point,sensitivity);
        if ((Interval{bracket.hi} - bracket.lo).hi <= target && magnitude(bracket) <= ray_half_length) {
            ++work.certificate_hits;
            return bracket;
        }
    }
    double u;
    if (!probe(p, n, seed, axial, station, work, nullptr, &u)) return std::nullopt;
    if (u < begin->low || u > (end-1)->high) return std::nullopt;
    const auto iter = std::upper_bound(begin, end, u,
        [](double x, const Span& span) { return x < span.high; });
    const Span& span = iter == end ? *(end-1) : *iter;
    const Interval t = (Interval{u} - span.low) / (Interval{span.high} - span.low);
    const auto& c = span.enclosed;
    const IVec point = ((c[3]*t + c[2])*t + c[1])*t + c[0];
    if (cached_point) *cached_point = Hint{u,point};
    Interval bracket = ray_range(point,sensitivity);
    if (bracket.width() > 2e-13) {
        const Interval f0 = dot(span.control.front()-origin,tangent);
        const Interval f1 = dot(span.control.back()-origin,tangent);
        if ((f0.hi <= 0 && f1.lo >= 0) || (f1.hi <= 0 && f0.lo >= 0)) {
            Interval local_derivative{infinity,-infinity};
            double local_along = 0;
            for (unsigned i=1; i<4; ++i) {
                // The common span width cancels from the slope ratio.
                const IVec edge = span.control[i]-span.control[i-1];
                local_derivative = hull(local_derivative,dot(edge,tangent));
                local_along = std::max(local_along,magnitude(dot(edge,normal)));
            }
            if (!local_derivative.contains(0)) {
                const Interval slope = Interval{local_along} /
                    std::min(std::abs(local_derivative.lo),std::abs(local_derivative.hi));
                const Interval tight = ray_range(point,slope);
                bracket = {std::max(bracket.lo,tight.lo),std::min(bracket.hi,tight.hi)};
            }
        }
    }
    if (magnitude(bracket) > ray_half_length) return std::nullopt;
    return bracket;
}

}  // namespace nc::gears
