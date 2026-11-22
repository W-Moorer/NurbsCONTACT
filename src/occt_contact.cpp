#include "nurbscontact/occt_contact.hpp"

#include <BRepExtrema_DistShapeShape.hxx>
#include <gp_Ax1.hxx>

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

class PairGeometry {
public:
    PairBackend backend;
    TopoDS_Shape shape;
    std::optional<dynamics::TriangleBvh> mesh;
    std::optional<PreparedModel> native;
};
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

gp_Trsf placement(ContactPose pose) {
    gp_Trsf transform;
    if (pose.angle != 0) {
        transform.SetRotation(gp_Ax1{gp_Pnt{}, gp_Dir{pose.axis[0], pose.axis[1], pose.axis[2]}}, pose.angle);
    }
    transform.SetTranslationPart(gp_Vec{pose.origin[0]/mm, pose.origin[1]/mm, pose.origin[2]/mm});
    return transform;
}

Motion static_motion(const gp_Trsf& relative) {
    Motion motion;
    const auto& offset = relative.TranslationPart();
    motion.translation = {offset.X(), offset.Y(), offset.Z()};
    gp_XYZ axis;
    if (relative.GetRotation(axis, motion.angle)) motion.axis = {axis.X(), axis.Y(), axis.Z()};
    else motion.angle = 0;
    return motion;
}

class StepPair final : public PairContactQuery {
public:
    StepPair(std::shared_ptr<const PairGeometry> a, std::shared_ptr<const PairGeometry> b, PairSearchOptions options)
        : a_(std::move(a)), b_(std::move(b)), options_(options) {
        if (a_->backend == PairBackend::occt) {
            extrema_.emplace();
            extrema_->SetDeflection(1e-7);
            extrema_->LoadS1(a_->shape);
        }
    }

    PairProximity closest(ContactPose a, ContactPose b, double cutoff, ContactWork& work) const override {
        if (a_->backend == PairBackend::mesh) {
            RigidPlacement relative;
            relative.origin = a.inverse_rotate(sub(b.origin, a.origin));
            for (auto& basis : relative.basis) basis = a.inverse_rotate(b.rotate(basis));
            const auto result = a_->mesh->closest_rigid_within(*b_->mesh, relative, cutoff, &hint_, work.mesh);
            if (!result) return {PairStatus::separated};
            return {PairStatus::within_cutoff, {a.point(result->a), a.point(result->b), result->distance}};
        }
        const gp_Trsf relative = placement(a).Inverted() * placement(b);
        if (extrema_) {
            extrema_->LoadS2(b_->shape.Moved(TopLoc_Location{relative}));
            if (!extrema_->Perform() || extrema_->NbSolution() == 0 || extrema_->InnerSolution()) return {};
            const double distance_m = extrema_->Value() * mm;
            if (distance_m > cutoff) return {PairStatus::separated};
            return {PairStatus::within_cutoff, {a.point(scale(vector(extrema_->PointOnShape1(1)), mm)),
                                      a.point(scale(vector(extrema_->PointOnShape2(1)), mm)), distance_m}};
        }
        Options options;
        options.distance_tolerance = options_.tolerance_m / mm;
        options.max_nodes = options_.max_nodes;
        const auto result = distance(*a_->native, *b_->native, static_motion(relative), 0, options);
        work.native_nodes += result.statistics.visited;
        const double lower = (Interval{result.lower} * mm).lo;
        const double upper = (Interval{result.upper} * mm).hi;
        const Interval bound{lower, upper};
        if (lower > cutoff) return {PairStatus::separated, {}, bound};
        if (!result.witness || upper > cutoff || up(upper - lower) > options_.tolerance_m) {
            return {PairStatus::unresolved, {}, bound};
        }
        const auto& witness = *result.witness;
        return {PairStatus::within_cutoff, {a.point(scale(midpoint(witness.a.position), mm)),
                                  a.point(scale(midpoint(witness.b.position), mm)), upper}, bound};
    }

private:
    std::shared_ptr<const PairGeometry> a_, b_;
    PairSearchOptions options_;
    mutable TriangleHint hint_;
    mutable std::optional<BRepExtrema_DistShapeShape> extrema_;
};

} // namespace

std::shared_ptr<const dynamics::ContactSurface> contact_surface(const TopoDS_Shape& shape, double deflection_m) {
    if (!(deflection_m>=0) || !std::isfinite(deflection_m)) throw std::invalid_argument("invalid mesh deflection");
    return std::make_shared<StepSurface>(shape,deflection_m);
}

std::shared_ptr<const PairGeometry> pair_geometry(const TopoDS_Shape& shape, PairBackend backend,
                                                  double deflection_m) {
    auto result = std::make_shared<PairGeometry>();
    result->backend = backend;
    if (backend == PairBackend::mesh) {
        if (!(deflection_m > 0) || !std::isfinite(deflection_m))
            throw std::invalid_argument("pair-mesh requires a positive deflection in meters");
        auto triangles = tessellate_occt(shape, deflection_m / mm);
        for (auto& triangle : triangles) for (auto& point : triangle) point = scale(point, mm);
        result->mesh.emplace(std::move(triangles));
    } else if (backend == PairBackend::brep) {
        result->native.emplace(adapt_occt(shape));
    } else {
        result->shape = shape;
    }
    return result;
}

std::shared_ptr<const dynamics::PairContactQuery> pair_contact_query(
    std::shared_ptr<const PairGeometry> a, std::shared_ptr<const PairGeometry> b, PairSearchOptions options) {
    if (!a || !b || a->backend != b->backend || !(options.tolerance_m > 0) ||
        !std::isfinite(options.tolerance_m) || options.max_nodes == 0) {
        throw std::invalid_argument("invalid pair contact backends or search options");
    }
    return std::make_shared<StepPair>(std::move(a), std::move(b), options);
}

SweepResult fixed_axis_pair_sweep(const PairGeometry& a, const PairGeometry& b,
                                 ContactPose a0, ContactPose a1, ContactPose b0, ContactPose b1,
                                 double clearance_m, double time_tolerance, PairSearchOptions settings) {
    if (!a.native || !b.native || a0.angle != a1.angle || a0.axis != a1.axis || b0.axis != b1.axis) {
        throw std::invalid_argument("CCD requires native STEP shapes and fixed axes with A's orientation constant");
    }
    Motion path = static_motion(placement(a0).Inverted() * placement(b0));
    path.velocity = scale(a0.inverse_rotate(sub(sub(b1.origin, b0.origin), sub(a1.origin, a0.origin))), 1/mm);
    if (b1.angle != b0.angle) {
        if (a0.angle != 0 && a0.axis != b0.axis && a0.axis != scale(b0.axis, -1)) {
            throw std::invalid_argument("CCD relative rotation requires aligned axes or an unrotated body A");
        }
        path.axis = b0.axis;
        path.angle = b0.angle - (a0.axis == b0.axis ? a0.angle : -a0.angle);
        path.angular_velocity = b1.angle - b0.angle;
    }
    Options options;
    options.distance_tolerance = settings.tolerance_m / mm;
    options.time_tolerance = time_tolerance;
    options.max_nodes = settings.max_nodes;
    options.axial_time = false;
    const Interval clearance = Interval{clearance_m}/mm;
    auto result = sweep(*a.native, *b.native, path, {0, 1}, clearance.hi, options);
    if (result.witness && norm(result.witness->b.position - result.witness->a.position).hi > clearance.lo) {
        result.status = SweepStatus::unresolved;
        result.witness_time.reset();
        result.witness.reset();
    }
    return result;
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
