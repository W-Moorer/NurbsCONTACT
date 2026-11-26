#pragma once
#include "cad_span_index.hpp"
#include "cad_column_depth.hpp"
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
    std::vector<std::array<Hull,4>> column_hulls;
    std::vector<ColumnDepthBound> depth_bounds;
    const double orientation;
    const double face_tolerance;
    SpanIndex column_index;
    std::vector<double> ku,kv;
    struct TranslationBand { gp_XYZ axis; double lo,hi; };
    std::vector<TranslationBand> bands;
    mutable std::mutex adaptor_mutex;
    static Row polynomial(const Row& p) {
        return {p[0],(p[1]-p[0])*3.,((p[2]-p[1])-(p[1]-p[0]))*3.,
                (p[3]-p[0])+(p[1]-p[2])*3.};
    }
    explicit CompiledChart(const TopoDS_Face& f,bool refine_columns=false,bool depth=false):surface(f),trims(f,Precision::PConfusion()),
        u0(surface.FirstUParameter()),u1(surface.LastUParameter()),
        v0(surface.FirstVParameter()),v1(surface.LastVParameter()),
        eu(std::max(32*Precision::PConfusion(),1e-7*(u1-u0))),
        ev(std::max(32*Precision::PConfusion(),1e-7*(v1-v0))),orientation(f.Orientation()==TopAbs_REVERSED ? -1. : 1.),
        face_tolerance(std::max(Precision::Confusion(),BRep_Tool::Tolerance(f))) {
        try { prepare(f,refine_columns,depth); }
        catch (const Standard_Failure&) {rectangular=false;patches.clear();hulls.clear();column_hulls.clear();depth_bounds.clear();}
    }
    // Subdivide a tensor-product control net at its parameter midpoint. The
    // convex hull of each child encloses its entire subpatch.
    static std::array<Patch,2> halves(const Patch& poles,bool along_u) {
        std::array<Patch,2> result;
        for (int i=0;i<4;++i) {
            Row p;
            for (int j=0;j<4;++j) p[j]=along_u ? poles[j][i] : poles[i][j];
            const auto a=(p[0]+p[1])*.5,b=(p[1]+p[2])*.5,c=(p[2]+p[3])*.5;
            const auto l=(a+b)*.5,r=(b+c)*.5,m=(l+r)*.5;
            const std::array<Row,2> rows{{{p[0],a,l,m},{m,r,c,p[3]}}};
            for (int side=0;side<2;++side) for (int j=0;j<4;++j)
                (along_u ? result[side][j][i] : result[side][i][j])=rows[side][j];
        }
        return result;
    }
    static Hull enclosure(const Patch& poles) {
        const auto& p=poles[0][0];
        auto n=(poles[3][3]-p).Crossed(poles[0][3]-poles[3][0]);
        if (n.SquareModulus()>0) n.Normalize();
        Hull h{p,p,n,n.Dot(p),n.Dot(p)};
        for (const auto& row:poles) for (const auto& q:row) {
            for (int d=1;d<=3;++d) {
                h.lo.SetCoord(d,std::min(h.lo.Coord(d),q.Coord(d)));
                h.hi.SetCoord(d,std::max(h.hi.Coord(d),q.Coord(d)));
            }
            h.low=std::min(h.low,n.Dot(q));h.high=std::max(h.high,n.Dot(q));
        }
        return h;
    }
    void prepare(const TopoDS_Face& f,bool refine_columns,bool depth) {
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
        if (refine_columns && rectangular) for (int along=0;along<2;++along) {
            // A tensor-product net is invariant under translations along a
            // generator when its rows are translations of a transverse curve.
            // Monotone axial poles ensure the entire trimmed band is covered.
            const int n=along ? bs->NbVPoles():bs->NbUPoles(), m=along ? bs->NbUPoles():bs->NbVPoles();
            const auto pole=[&](int i,int j) {return (along ? bs->Pole(j,i):bs->Pole(i,j)).XYZ();};
            const auto p=pole(1,1);
            auto axis=pole(n,1)-p;
            if (axis.Modulus()<Precision::Confusion()) continue;
            axis.Normalize();
            const double guard=1e-10; // mm, well below the native ray tolerance
            bool valid=true;
            double previous=0;
            for (int i=1;valid && i<=n;++i) {
                const auto shift=pole(i,1)-p;
                const double axial=shift.Dot(axis);
                valid=axial>=previous && (shift-axis*axial).Modulus()<=guard;
                previous=axial;
                for (int j=1;valid && j<=m;++j)
                    valid=(pole(i,j)-pole(1,j)-shift).Modulus()<=guard &&
                          std::abs((pole(1,j)-p).Dot(axis))<=guard;
            }
            if (valid) {
                const double lo=surface.Value(u0,v0).XYZ().Dot(axis);
                const double hi=surface.Value(along ? u0:u1,along ? v1:v0).XYZ().Dot(axis);
                if (hi>lo) bands.push_back({axis,lo,hi});
            }
        }
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
            Patch rows,coeff,poles;
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
                poles[a]=values;
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
            if (depth) depth_bounds.emplace_back(poles,face_tolerance);
            if (refine_columns) {
                std::array<Hull,4> sub;
                int k=0;
                for (const auto& u:halves(poles,true)) for (const auto& v:halves(u,false))
                    sub[k++]=enclosure(v);
                column_hulls.push_back(sub);
            }
        }
        if (refine_columns)
            column_index.build(hulls.size(),[&](auto i)->const Hull& {return hulls[i];});
    }
    // A rigidly placed box encloses both ends of every finite pressure column
    // in one quadrature block. Test its projection in the hull's own slab.
    struct ColumnBox {
        gp_XYZ center;
        std::array<gp_XYZ,3> axes;
        gp_XYZ lo,hi;
    };
    void block_spans(const ColumnBox& box,std::vector<std::size_t>& spans) const {
        spans.clear();
        const auto& lo=box.lo;
        const auto& hi=box.hi;
        const auto possible=[&](const Hull& h) {
            if (hi.X() < h.lo.X() || lo.X() > h.hi.X() || hi.Y() < h.lo.Y() || lo.Y() > h.hi.Y() ||
                hi.Z() < h.lo.Z() || lo.Z() > h.hi.Z())
                return false;
            const double mid=box.center.Dot(h.n);
            const double radius=std::abs(box.axes[0].Dot(h.n))+std::abs(box.axes[1].Dot(h.n))+
                                std::abs(box.axes[2].Dot(h.n))+4*Precision::Confusion();
            return mid+radius>=h.low && mid-radius<=h.high;
        };
        column_index.query(lo,hi,[&](auto i) {
            if (possible(hulls[i]) && (column_hulls.empty() ||
                std::any_of(column_hulls[i].begin(),column_hulls[i].end(),possible))) spans.push_back(i);
            return true;
        });
    }
    bool column_possible(const gp_XYZ& p,const gp_XYZ& segment,const std::vector<std::size_t>* spans=nullptr,
                         double tol=2*Precision::Confusion()) const {
        if (hulls.empty()) return true;
        const auto q=p+segment;
        const gp_XYZ low(std::min(p.X(),q.X())-tol,std::min(p.Y(),q.Y())-tol,std::min(p.Z(),q.Z())-tol);
        const gp_XYZ high(std::max(p.X(),q.X())+tol,std::max(p.Y(),q.Y())+tol,std::max(p.Z(),q.Z())+tol);
        const auto possible=[&](const Hull& h) {
            if (high.X()<h.lo.X() || low.X()>h.hi.X() || high.Y()<h.lo.Y() || low.Y()>h.hi.Y() ||
                high.Z()<h.lo.Z() || low.Z()>h.hi.Z()) return false;
            const double x=p.Dot(h.n),y=q.Dot(h.n);
            return std::max(x,y)>=h.low-tol && std::min(x,y)<=h.high+tol;
        };
        const auto accepts=[&](auto i) {
            return possible(hulls[i]) && (column_hulls.empty() ||
                std::any_of(column_hulls[i].begin(),column_hulls[i].end(),possible));
        };
        if (spans) return std::any_of(spans->begin(),spans->end(),accepts);
        return !column_index.query(low,high,[&](auto i){return !accepts(i);});
    }
    bool inside(double u,double v) const {
        if (rectangular && u>u0+eu && u<u1-eu && v>v0+ev && v<v1-ev) return true;
        const std::lock_guard lock(adaptor_mutex);
        return trims.Perform(gp_Pnt2d(u,v))==TopAbs_IN;
    }
    // Refine an already isolated interior root on the original polynomial.
    // Distances in this routine are in mm. Reject every unverified iterate.
    bool interior_depth(std::size_t index,const MaterialRay& ray,double reach,double& depth) const {
        const auto& cell=depth_bounds[index];
        const auto dir=ray.direction, offset=ray.point-cell.origin;
        const double nd=cell.row[2].Dot(dir);
        if (!rectangular || std::abs(nd)<.02) return false;
        const auto plane=offset-dir*(cell.row[2].Dot(offset)/nd);
        const auto i=index/(kv.size()-1),j=index%(kv.size()-1);
        const double du=ku[i+1]-ku[i],dv=kv[j+1]-kv[j];
        double u=ku[i]+cell.row[0].Dot(plane)*du,v=kv[j]+cell.row[1].Dot(plane)*dv;
        for (int it=0;it<8;++it) {
            if (!(u>ku[i] && u<ku[i+1] && v>kv[j] && v<kv[j+1])) return false;
            gp_Pnt point;
            gp_Vec a,b;
            d1(u,v,point,a,b);
            const auto delta=point.XYZ()-ray.point;
            const double t=dir.Dot(delta);
            const auto perp=delta-dir*t;
            const auto r=b.XYZ().Crossed(dir),s=dir.Crossed(a.XYZ());
            const double det=a.XYZ().Dot(r);
            if (!(orientation*det>1e-8*a.Magnitude()*b.Magnitude())) return false;
            // Banach's a posteriori bound along the chart normal. classify()
            // establishes a contraction below .8 over the entire parameter cell.
            const double uv_error=5*std::max(std::abs(cell.row[0].Dot(perp)-cell.row[0].Dot(dir)*cell.row[2].Dot(perp)/nd),
                                            std::abs(cell.row[1].Dot(perp)-cell.row[1].Dot(dir)*cell.row[2].Dot(perp)/nd))+1e-13;
            const double margin_u=du*uv_error+eu,margin_v=dv*uv_error+ev;
            if (perp.SquareModulus()<1e-20 && (std::abs(cell.row[2].Dot(perp))+cell.gradient[2]*uv_error)/std::abs(nd)<1e-9 &&
                u>u0+margin_u && u<u1-margin_u && v>v0+margin_v && v<v1-margin_v &&
                t>8*Precision::Confusion() && t<reach-8*Precision::Confusion()) {
                depth=t;
                return true;
            }
            u-=perp.Dot(r)/det;
            v-=perp.Dot(s)/det;
        }
        return false;
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
