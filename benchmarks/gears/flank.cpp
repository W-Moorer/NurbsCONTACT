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
#include <numbers>
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
        for (int j=1; j<=decomposed.NbArcs(); ++j) {
            const auto arc = decomposed.Arc(j);
            const Vec p0=vector(arc->Pole(1).XYZ()), p1=vector(arc->Pole(2).XYZ());
            const Vec p2=vector(arc->Pole(3).XYZ()), p3=vector(arc->Pole(4).XYZ());
            const Vec d0=sub(p1,p0), d1=sub(p2,p1), d2=sub(p3,p2);
            const double width=knots(j+1)-knots(j);
            const std::array coefficients{p0,scale(d0,3),scale(sub(d1,d0),3),add(sub(d2,scale(d1,2)),d0)};
            spans.push_back({knots(j),knots(j+1),1/width,coefficients});
        }
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
                                   QueryWork& work, double* slope) const {
    ++work.rays;
    const Vec tangent{n[1],-n[0],0};
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
                return distance;
            }
            break;
        }
        const double derivative = dot3(du,tangent);
        if (std::abs(derivative) < 1e-10) break;
        const double next = std::clamp(u-residual/derivative,u0,u1);
        if (next == u) break;
        u=next;
    }
    ++work.misses;
    return std::nullopt;
}

}  // namespace nc::gears
