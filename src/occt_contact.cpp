#include "nurbscontact/occt_contact.hpp"

#include <BRepAdaptor_Surface.hxx>
#include <BRepBndLib.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRepAdaptor_Curve.hxx>
#include <BRepTools_WireExplorer.hxx>
#include <BRep_Builder.hxx>
#include <BRep_Tool.hxx>
#include <Bnd_Box.hxx>
#include <IntCurvesFace_ShapeIntersector.hxx>
#include <Poly_Triangulation.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Compound.hxx>
#include <gp_Lin.hxx>

#include <algorithm>
#include <stdexcept>

namespace nc {
namespace {

using namespace dynamics;
constexpr double mm = .001;

template<class T> Vec vector(const T& p) { return {p.X(),p.Y(),p.Z()}; }

using Point2 = std::array<double,2>;

struct PlanarFace {
    Vec origin,u,v,normal;
    Point2 low{infinity,infinity},high{-infinity,-infinity};
    double tolerance;
    std::vector<std::vector<Point2>> loops;
    bool rectangle{};

    Point2 project(Vec point) const {
        const auto relative=sub(point,origin);
        return {dot3(relative,u),dot3(relative,v)};
    }

    bool contains(Point2 p) const {
        for (std::size_t i=0;i<2;++i) {
            if (p[i]<low[i]-tolerance || p[i]>high[i]+tolerance) return false;
        }
        if (rectangle) return true;
        bool inside=false;
        for (const auto& loop:loops) {
            for (std::size_t i=0;i<loop.size();++i) {
                const auto a=loop[i], b=loop[(i+1)%loop.size()];
                const double dx=b[0]-a[0],dy=b[1]-a[1];
                const double x=p[0]-a[0],y=p[1]-a[1],edge=std::hypot(dx,dy);
                if (std::abs(dx*y-dy*x)<=tolerance*edge &&
                    dx*x+dy*y>=-tolerance*edge && dx*x+dy*y<=edge*(edge+tolerance)) return true;
                if ((a[1]>p[1])!=(b[1]>p[1]) && x<dx*y/dy) inside=!inside;
            }
        }
        return inside;
    }
};

std::optional<PlanarFace> prepare_planar(const TopoDS_Face& face) {
    const BRepAdaptor_Surface surface(face);
    if (surface.GetType()!=GeomAbs_Plane) return std::nullopt;
    const auto plane=surface.Plane().Position();
    PlanarFace result;
    result.origin=scale(vector(plane.Location()),mm);
    result.u=vector(plane.XDirection());
    result.v=vector(plane.YDirection());
    result.normal=vector(plane.Direction());
    result.tolerance=std::max(1e-10,BRep_Tool::Tolerance(face)*mm);
    for (TopExp_Explorer wire_it(face,TopAbs_WIRE);wire_it.More();wire_it.Next()) {
        const auto wire=TopoDS::Wire(wire_it.Current());
        std::size_t edge_count=0;
        for (TopExp_Explorer edge(wire,TopAbs_EDGE);edge.More();edge.Next()) ++edge_count;
        std::vector<Point2> points;
        for (BRepTools_WireExplorer edge(wire,face);edge.More();edge.Next()) {
            if (edge.CurrentVertex().IsNull() || BRepAdaptor_Curve(edge.Current()).GetType()!=GeomAbs_Line)
                return std::nullopt;
            const auto point=scale(vector(BRep_Tool::Pnt(edge.CurrentVertex())),mm);
            if (std::abs(dot3(sub(point,result.origin),result.normal))>result.tolerance) return std::nullopt;
            const auto p=result.project(point);
            if (!points.empty() && std::hypot(p[0]-points.back()[0],p[1]-points.back()[1])<=result.tolerance)
                return std::nullopt;
            for (std::size_t i=0;i<2;++i) {
                result.low[i]=std::min(result.low[i],p[i]);
                result.high[i]=std::max(result.high[i],p[i]);
            }
            points.push_back(p);
        }
        if (points.size()<3 || points.size()!=edge_count ||
            std::hypot(points.front()[0]-points.back()[0],points.front()[1]-points.back()[1])<=result.tolerance)
            return std::nullopt;
        result.loops.push_back(std::move(points));
    }
    if (result.loops.empty()) return std::nullopt;
    if (result.loops.size()==1 && result.loops.front().size()==4) {
        unsigned corners=0;
        for (const auto p:result.loops.front()) {
            unsigned flag=0;
            for (std::size_t i=0;i<2;++i) {
                if (std::abs(p[i]-result.high[i])<=result.tolerance) flag|=1u<<i;
                else if (std::abs(p[i]-result.low[i])>result.tolerance) return result;
            }
            corners|=1u<<flag;
        }
        // Every edge must follow one side. The bounds then give exact trim.
        bool axial=true;
        const auto& p=result.loops.front();
        for (std::size_t i=0;i<p.size();++i) {
            const auto a=p[i],b=p[(i+1)%p.size()];
            axial&=std::abs(a[0]-b[0])<=result.tolerance || std::abs(a[1]-b[1])<=result.tolerance;
        }
        result.rectangle=corners==15 && axial;
    }
    return result;
}

class StepSurface final : public ContactSurface {
public:
    StepSurface(const TopoDS_Shape& shape, double deflection) {
        Bnd_Box box;
        BRepBndLib::AddOptimal(shape,box,false,false);
        if (box.IsVoid() || box.IsOpen()) throw std::invalid_argument("contact STEP needs finite bounds");
        bounds_ = {scale(vector(box.CornerMin()),mm),scale(vector(box.CornerMax()),mm)};
        if (deflection>0) {
            auto triangles=tessellate_occt(shape,deflection/mm);
            for (auto& triangle:triangles) for (auto& point:triangle) point=scale(point,mm);
            mesh_.emplace(std::move(triangles),4);
        } else {
            BRep_Builder builder;
            TopoDS_Compound remaining;
            builder.MakeCompound(remaining);
            for (TopExp_Explorer it(shape,TopAbs_FACE);it.More();it.Next()) {
                const auto face=TopoDS::Face(it.Current());
                auto planar=prepare_planar(face);
                if (planar) planar_.push_back(std::move(*planar));
                else {
                    has_native_=true;
                    builder.Add(remaining,face);
                }
            }
            if (has_native_) native_.Load(remaining,1e-7);
        }
    }

    TriangleBvh::Box bounds() const override { return bounds_; }

    std::optional<RayHit> first_hit(Vec origin, Vec direction, double maximum, ContactWork& work) const override {
        if (mesh_) {
            Vec normal{};
            const auto hit=mesh_->ray_first(origin,direction,maximum,work.mesh,&normal);
            if (!hit) return std::nullopt;
            return RayHit{*hit,scale(normal,1/length(normal))};
        }
        std::optional<RayHit> result;
        for (const auto& plane:planar_) {
            const double cosine=dot3(plane.normal,direction);
            if (std::abs(cosine)<1e-12) continue;
            const double hit=dot3(plane.normal,sub(plane.origin,origin))/cosine;
            if (hit<0 || hit>maximum || !plane.contains(plane.project(add(origin,scale(direction,hit)))))
                continue;
            maximum=hit;
            result=RayHit{hit,plane.normal};
        }
        if (!has_native_) return result;
        const gp_Lin ray(gp_Pnt(origin[0]/mm,origin[1]/mm,origin[2]/mm),
                         gp_Dir(direction[0],direction[1],direction[2]));
        native_.PerformNearest(ray,0,maximum/mm);
        if (!native_.IsDone()) throw std::runtime_error("unresolved STEP surface intersection");
        if (!native_.NbPnt()) return result;
        const BRepAdaptor_Surface face(native_.Face(1));
        gp_Pnt point;
        gp_Vec du,dv;
        face.D1(native_.UParameter(1),native_.VParameter(1),point,du,dv);
        auto normal=vector(du.Crossed(dv));
        if (length(normal)<1e-12 && face.GetType()==GeomAbs_Sphere) {
            normal=vector(gp_Vec(face.Sphere().Location(),point));
        }
        if (length(normal)<1e-12) {
            return RayHit{native_.WParameter(1)*mm,{}};
        }
        return RayHit{native_.WParameter(1)*mm,scale(normal,1/length(normal))};
    }

private:
    TriangleBvh::Box bounds_;
    std::optional<TriangleBvh> mesh_;
    std::vector<PlanarFace> planar_;
    bool has_native_{};
    // OCCT's intersector keeps scratch state. Each scene uses it serially.
    mutable IntCurvesFace_ShapeIntersector native_;
};

} // namespace

std::shared_ptr<const dynamics::ContactSurface> contact_surface(const TopoDS_Shape& shape, double deflection_m) {
    if (!(deflection_m>=0) || !std::isfinite(deflection_m)) throw std::invalid_argument("invalid mesh deflection");
    return std::make_shared<StepSurface>(shape,deflection_m);
}

std::vector<dynamics::Triangle> tessellate_occt(const TopoDS_Shape& shape, double deflection_mm) {
    const BRepMesh_IncrementalMesh mesh(shape,deflection_mm,false,.35,false);
    if (!mesh.IsDone()) throw std::runtime_error("STEP tessellation failed");
    std::vector<dynamics::Triangle> triangles;
    for (TopExp_Explorer i(shape,TopAbs_FACE);i.More();i.Next()) {
        TopLoc_Location location;
        const auto data=BRep_Tool::Triangulation(TopoDS::Face(i.Current()),location);
        if (data.IsNull()) throw std::runtime_error("STEP face has no triangles");
        for (int j=1;j<=data->NbTriangles();++j) {
            int a,b,c;
            data->Triangle(j).Get(a,b,c);
            dynamics::Triangle triangle;
            std::size_t vertex=0;
            for (int index:{a,b,c}) {
                triangle[vertex++]=vector(data->Node(index).Transformed(location.Transformation()));
            }
            triangles.push_back(triangle);
        }
    }
    return triangles;
}

} // namespace nc
