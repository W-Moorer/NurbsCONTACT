#pragma once
#include <BRepAdaptor_Surface.hxx>
#include <BRepTopAdaptor_FClass2d.hxx>
#include <BRep_Tool.hxx>
#include <BndLib_Add2dCurve.hxx>
#include <Bnd_Box2d.hxx>
#include <Geom_BSplineSurface.hxx>
#include <Geom2dAdaptor_Curve.hxx>
#include <GeomConvert_BSplineSurfaceToBezierSurface.hxx>
#include <Geom_BezierSurface.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <algorithm>
#include <array>
#include <mutex>
#include <vector>

namespace nc::cad {
// Polynomial bicubic knot spans admit direct evaluation in their power basis.
// Other surface types keep the native adaptor, including rational splines.
struct CompiledChart {
    BRepAdaptor_Surface surface;
    BRepTopAdaptor_FClass2d trims;
    double u0,u1,v0,v1,eu,ev;
    bool rectangular=false;
    using Row=std::array<gp_XYZ,4>;
    using Patch=std::array<Row,4>;
    struct Hull {gp_XYZ lo,hi,n;double low,high;};
    std::vector<Patch> patches;
    std::vector<Hull> hulls;
    std::vector<double> ku,kv;
    mutable std::mutex adaptor_mutex;
    static Row polynomial(const Row& p) {
        return {p[0],(p[1]-p[0])*3.,((p[2]-p[1])-(p[1]-p[0]))*3.,
                (p[3]-p[0])+(p[1]-p[2])*3.};
    }
    explicit CompiledChart(const TopoDS_Face& f):surface(f),trims(f,Precision::PConfusion()),
        u0(surface.FirstUParameter()),u1(surface.LastUParameter()),
        v0(surface.FirstVParameter()),v1(surface.LastVParameter()),
        eu(std::max(32*Precision::PConfusion(),1e-7*(u1-u0))),
        ev(std::max(32*Precision::PConfusion(),1e-7*(v1-v0))) {
        try { prepare(f); }
        catch (const Standard_Failure&) {rectangular=false;patches.clear();hulls.clear();}
    }
    void prepare(const TopoDS_Face& f) {
        rectangular=trims.Perform(gp_Pnt2d(.5*(u0+u1),.5*(v0+v1)))==TopAbs_IN;
        for (TopExp_Explorer it(f.Oriented(TopAbs_FORWARD),TopAbs_EDGE);rectangular && it.More();it.Next()) {
            double lo,hi;
            auto curve=BRep_Tool::CurveOnSurface(TopoDS::Edge(it.Current()),f,lo,hi);
            if (curve.IsNull()) {rectangular=false;break;}
            Geom2dAdaptor_Curve adaptor(curve,lo,hi);
            const auto type=adaptor.GetType();
            if (type!=GeomAbs_Line && type!=GeomAbs_BSplineCurve && type!=GeomAbs_BezierCurve) {
                rectangular=false;break;
            }
            Bnd_Box2d box;
            BndLib_Add2dCurve::Add(adaptor,lo,hi,Precision::PConfusion(),box);
            if (box.IsVoid() || box.IsOpenXmin() || box.IsOpenXmax() ||
                box.IsOpenYmin() || box.IsOpenYmax()) {rectangular=false;break;}
            double x,y,X,Y; box.Get(x,y,X,Y);
            rectangular=X<=u0+eu || x>=u1-eu || Y<=v0+ev || y>=v1-ev;
        }
        if (surface.GetType()!=GeomAbs_BSplineSurface) return;
        const auto bs=surface.BSpline();
        if (bs->UDegree()!=3 || bs->VDegree()!=3 || bs->IsURational() || bs->IsVRational()) return;
        GeomConvert_BSplineSurfaceToBezierSurface net(bs);
        const int nu=net.NbUPatches(),nv=net.NbVPatches();
        TColStd_Array1OfReal u(1,nu+1),v(1,nv+1);
        net.UKnots(u); net.VKnots(v);
        ku.assign(u.begin(),u.end());kv.assign(v.begin(),v.end());
        for (int i=1;i<=nu;++i) for (int j=1;j<=nv;++j) {
            const auto bez=net.Patch(i,j);
            gp_Pnt center;gp_Vec tu,tv;
            bez->D1(.5,.5,center,tu,tv);
            auto normal=tu.Crossed(tv);
            if (normal.SquareMagnitude()>0) normal.Normalize();
            const auto first=bez->Pole(1,1).XYZ();
            Hull hull{first,first,normal.XYZ(),normal.XYZ().Dot(first),normal.XYZ().Dot(first)};
            Patch rows,coeff;
            for (int a=0;a<4;++a) {
                Row values;
                for (int b=0;b<4;++b) {
                    values[b]=bez->Pole(a+1,b+1).XYZ();
                    for (int d=1;d<=3;++d) {
                        hull.lo.SetCoord(d,std::min(hull.lo.Coord(d),values[b].Coord(d)));
                        hull.hi.SetCoord(d,std::max(hull.hi.Coord(d),values[b].Coord(d)));
                    }
                    const double h=values[b].Dot(hull.n);
                    hull.low=std::min(hull.low,h); hull.high=std::max(hull.high,h);
                }
                rows[a]=polynomial(values);
            }
            for (int b=0;b<4;++b) {
                Row values;
                for (int a=0;a<4;++a) values[a]=rows[a][b];
                const auto col=polynomial(values);
                for (int a=0;a<4;++a) coeff[a][b]=col[a];
            }
            patches.push_back(coeff);
            hulls.push_back(hull);
        }
    }
    bool block_possible(const Bnd_Box& swept) const {
        if (hulls.empty()) return true;
        const auto lo = swept.CornerMin().XYZ(), hi = swept.CornerMax().XYZ();
        const auto center = (lo + hi) * .5, half = (hi - lo) * .5;
        for (const auto& h : hulls) {
            if (hi.X() < h.lo.X() || lo.X() > h.hi.X() || hi.Y() < h.lo.Y() || lo.Y() > h.hi.Y() ||
                hi.Z() < h.lo.Z() || lo.Z() > h.hi.Z())
                continue;
            const double mid = center.Dot(h.n);
            const double radius =
                std::abs(h.n.X()) * half.X() + std::abs(h.n.Y()) * half.Y() + std::abs(h.n.Z()) * half.Z();
            if (mid + radius >= h.low - 2 * Precision::Confusion() &&
                mid - radius <= h.high + 2 * Precision::Confusion())
                return true;
        }
        return false;
    }
    bool column_possible(const gp_XYZ& p,const gp_XYZ& segment) const {
        if (hulls.empty()) return true;
        const double tol=2*Precision::Confusion();
        const auto q=p+segment;
        const gp_XYZ low(std::min(p.X(),q.X())-tol,std::min(p.Y(),q.Y())-tol,std::min(p.Z(),q.Z())-tol);
        const gp_XYZ high(std::max(p.X(),q.X())+tol,std::max(p.Y(),q.Y())+tol,std::max(p.Z(),q.Z())+tol);
        for (const auto& h:hulls) {
            if (high.X()<h.lo.X() || low.X()>h.hi.X() || high.Y()<h.lo.Y() || low.Y()>h.hi.Y() ||
                high.Z()<h.lo.Z() || low.Z()>h.hi.Z()) continue;
            const double x=p.Dot(h.n),y=q.Dot(h.n);
            if (std::max(x,y)>=h.low-tol && std::min(x,y)<=h.high+tol) return true;
        }
        return false;
    }
    bool inside(double u,double v) const {
        if (rectangular && u>u0+eu && u<u1-eu && v>v0+ev && v<v1-ev) return true;
        const std::lock_guard lock(adaptor_mutex);
        return trims.Perform(gp_Pnt2d(u,v))==TopAbs_IN;
    }
    void d1(double u,double v,gp_Pnt& p,gp_Vec& du,gp_Vec& dv) const {
        if (patches.empty() || u<ku.front() || u>ku.back() || v<kv.front() || v>kv.back()) {
            const std::lock_guard lock(adaptor_mutex);
            surface.D1(u,v,p,du,dv);return;
        }
        const auto span=[](const std::vector<double>& knots,double t) {
            return std::upper_bound(knots.begin()+1,knots.end()-1,t)-knots.begin()-1;
        };
        const auto i=span(ku,u),j=span(kv,v);
        const auto& c=patches[i*(kv.size()-1)+j];
        const double h=ku[i+1]-ku[i],k=kv[j+1]-kv[j],x=(u-ku[i])/h,y=(v-kv[j])/k;
        Row a,b;
        for (int n=0;n<4;++n) {
            a[n]=((c[n][3]*y+c[n][2])*y+c[n][1])*y+c[n][0];
            b[n]=(c[n][3]*(3*y)+c[n][2]*2)*y+c[n][1];
        }
        p.SetXYZ(((a[3]*x+a[2])*x+a[1])*x+a[0]);
        du.SetXYZ(((a[3]*(3*x)+a[2]*2)*x+a[1])/h);
        dv.SetXYZ((((b[3]*x+b[2])*x+b[1])*x+b[0])/k);
    }
};
}
