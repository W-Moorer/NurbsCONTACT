#include "cad_rigid_path.hpp"
#include <BRepAdaptor_Curve.hxx>
#include <BRepAlgoAPI_Common.hxx>
#include <BRepBndLib.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepGProp.hxx>
#include <BRep_Tool.hxx>
#include <GProp_GProps.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <math_Function.hxx>
#include <math_KronrodSingleIntegration.hxx>

namespace nc {
namespace {
using namespace dynamics;
using namespace cad;

Bnd_Box bounds(const TopoDS_Shape& shape,bool tight=false) {
    Bnd_Box box;
    if (tight) BRepBndLib::AddOptimal(shape,box,false,false);
    else BRepBndLib::Add(shape,box,false);
    if (box.IsVoid() || box.IsOpen()) throw std::invalid_argument("expected a bounded CAD solid");
    return box;
}

struct Solid {
    TopoDS_Shape shape;
    Bnd_Box box;
    double radius{};
    Solid(const TopoDS_Shape& s,Vec center) : shape(s),box(bounds(s)) {
        if (!TopExp_Explorer(s,TopAbs_SOLID).More()) throw std::invalid_argument("CAD penalty contact requires solids");
        const auto low=coordinates(box.CornerMin()),high=coordinates(box.CornerMax());
        for (int i=0;i<3;++i) {
            const double extent=std::max(std::abs(low[i]-1000*center[i]),std::abs(high[i]-1000*center[i]));
            radius=std::hypot(radius,extent);
        }
    }
    Bnd_Box swept(const RigidPath& path) const {
        auto envelope=box.Transformed(path.start);
        envelope.Add(box.Transformed(path.end));
        const double s=std::sin(.25*path.angle);
        envelope.Enlarge(envelope.GetGap()+2*radius*s*s+Precision::Confusion());
        return envelope;
    }
};

// Stokes' theorem gives the integrated outward normal of a trimmed face as
// half the oriented boundary integral of (p - origin) cross dp.
class AreaIntegrand final : public math_Function {
public:
    AreaIntegrand(const BRepAdaptor_Curve& edge,const gp_Pnt& origin,int coordinate)
        : edge_(edge),origin_(origin),coordinate_(coordinate) {}
    Standard_Boolean Value(Standard_Real t,Standard_Real& result) override {
        gp_Pnt p;
        gp_Vec derivative;
        edge_.D1(t,p,derivative);
        result=gp_Vec(origin_,p).Crossed(derivative).Coord(coordinate_);
        return std::isfinite(result);
    }
private:
    const BRepAdaptor_Curve& edge_;
    gp_Pnt origin_;
    int coordinate_;
};

Vec vector_area(const TopoDS_Face& face,const gp_Pnt& origin) {
    Vec area{};
    for (TopExp_Explorer it(face,TopAbs_EDGE);it.More();it.Next()) {
        const auto edge=TopoDS::Edge(it.Current());
        if (BRep_Tool::Degenerated(edge)) continue;
        const double sign=edge.Orientation()==TopAbs_REVERSED ? -.5 : .5;
        BRepAdaptor_Curve curve(edge);
        TColStd_Array1OfReal knots(1,1+curve.NbIntervals(GeomAbs_C1));
        curve.Intervals(knots,GeomAbs_C1);
        for (int i=1;i<knots.Upper();++i) {
            for (int k=0;k<3;++k) {
                AreaIntegrand function(curve,origin,k+1);
                math_KronrodSingleIntegration integral(function,knots(i),knots(i+1),15,1e-8,8);
                if (!integral.IsDone()) throw UnresolvedContact("CAD contact boundary integration failed");
                area[k]+=sign*integral.Value();
            }
        }
    }
    return area;
}

TopTools_IndexedMapOfShape source_faces(const TopoDS_Shape& shape,BRepAlgoAPI_Common& op) {
    TopTools_IndexedMapOfShape faces;
    for (TopExp_Explorer it(shape,TopAbs_FACE);it.More();it.Next()) {
        faces.Add(it.Current());
        for (const auto& subface:op.Modified(it.Current())) faces.Add(subface);
    }
    return faces;
}

class CadSolidPair final : public PairRegionQuery {
public:
    CadSolidPair(const TopoDS_Shape& a,const TopoDS_Shape& b,std::array<Vec,2> centers,bool chen)
        : a_(a,centers[0]),b_(b,centers[1]),centers_(centers) {
        if (chen) {
#ifdef NC_WITH_CHEN
            candidate_=chen_candidates(a,b);
#else
            throw std::invalid_argument("configure NC_CHEN_SOURCE_DIR to enable Chen 2024 CCD");
#endif
        }
    }
    std::vector<ContactRegion> regions(ContactPose a,ContactPose b,ContactWork& work) const override try {
        const auto ta=placement(a),tb=placement(b);
        if (a_.box.Transformed(ta).IsOut(b_.box.Transformed(tb))) return {};
        const auto first=a_.shape.Moved(TopLoc_Location(ta)),second=b_.shape.Moved(TopLoc_Location(tb));
        ++work.cad_boolean_queries;
        BRepAlgoAPI_Common op;
        TopTools_ListOfShape args,tools;
        args.Append(first); tools.Append(second);
        op.SetArguments(args);
        op.SetTools(tools);
        op.SetNonDestructive(true);
        op.Build();
        if (!op.IsDone() || op.HasErrors() || op.HasWarnings())
            throw UnresolvedContact("CAD solid intersection failed or reported incomplete geometry");
        TopExp_Explorer solids(op.Shape(),TopAbs_SOLID);
        if (!solids.More()) return {};
        const auto one=source_faces(first,op),two=source_faces(second,op);
        std::vector<ContactRegion> regions;
        for (;solids.More();solids.Next()) {
            const auto& solid=solids.Current();
            GProp_GProps props;
            BRepGProp::VolumeProperties(solid,props,1e-8,true);
            if (!(props.Mass()>0)) throw UnresolvedContact("invalid CAD intersection volume");
            const auto origin=props.CentreOfMass();
            Vec direction{};
            double total=0;
            bool source_a=false,source_b=false;
            for (TopExp_Explorer it(solid,TopAbs_FACE);it.More();it.Next()) {
                const auto face=TopoDS::Face(it.Current());
                const bool on_a=one.Contains(face),on_b=two.Contains(face);
                if (!on_a && !on_b) throw UnresolvedContact("CAD intersection face has no source");
                if (on_a==on_b) continue;
                source_a|=on_a; source_b|=on_b;
                const Vec area=vector_area(face,origin);
                direction=add(direction,scale(area,on_a ? 1. : -1.));
                total+=length(area);
            }
            const auto magnitude=length(direction);
            if (!source_a || !source_b || !(magnitude>1e-10*total))
                throw UnresolvedContact("CAD overlap has no resolved separation normal");
            const auto n=scale(direction,1/magnitude);
            gp_Trsf frame;
            frame.SetTransformation(gp_Ax3(origin,gp_Dir(n[0],n[1],n[2])));
            const auto extent=bounds(solid.Moved(TopLoc_Location(frame)),true);
            regions.push_back({scale(coordinates(origin),.001),n,
                               .001*(extent.CornerMax().Z()-extent.CornerMin().Z())});
        }
        return regions;
    } catch (const Standard_Failure& e) {
        throw UnresolvedContact(std::string("CAD overlap failed: ")+e.GetMessageString());
    }
    SurfaceTransition transition(ContactPose a0,ContactPose b0,ContactPose a1,ContactPose b1,
                                 double tolerance_m) const override try {
        if (!(tolerance_m>0 && std::isfinite(tolerance_m))) throw std::invalid_argument("invalid CAD event tolerance");
        SurfaceTransition result;
        const double tol=std::max(1000*tolerance_m,2.*Precision::Confusion());
        RigidPath a(a0,a1,centers_[0]),b(b0,b1,centers_[1]);
        if (a_.swept(a).Distance(b_.swept(b))>tol) return result;
        const auto distance=[&](const gp_Trsf& a,const gp_Trsf& b) {
            ++result.nodes;
            ++result.cad_distances;
            BRepExtrema_DistShapeShape query(a_.shape.Moved(TopLoc_Location(a)),b_.shape.Moved(TopLoc_Location(b)),
                                            std::min(.05*tol,Precision::Confusion()));
            if (!query.IsDone() || !std::isfinite(query.Value())) throw UnresolvedContact("CAD distance query failed");
            return query.InnerSolution() ? 0. : query.Value();
        };
        double d0=distance(a.start,b.start),d1=distance(a.end,b.end);
        result.start_in_contact=d0<=tol;
        result.end_in_contact=d1<=tol;
        if (result.start_in_contact && result.end_in_contact) return result;
        if (result.start_in_contact) {
            a=RigidPath(a1,a0,centers_[0]); b=RigidPath(b1,b0,centers_[1]); std::swap(d0,d1);
        }
        const double speed=(gp_Vec(a.from,a.to)-gp_Vec(b.from,b.to)).Magnitude()+
                           std::abs(a.angle)*a_.radius+std::abs(b.angle)*b_.radius;
        if (speed==0 || d0>speed+tol) return result;
        double horizon=1.,safe_left=1.,safe_right=1.;
        bool known_hit=d1<=tol;
        // Validate a polynomial-patch proposal on both original trimmed solids.
        // The distance at a rejected proposal can also exclude a time interval.
        if (candidate_) {
            const auto hint=candidate_(a,b,result);
            if (hint) {
                const double d=distance(a.at(*hint),b.at(*hint));
                if (d<=tol) {
                    horizon=*hint;
                    known_hit=true;
                    ++result.chen_confirmed;
                } else {
                    const double radius=(d-tol)/speed;
                    safe_left=*hint-radius;
                    safe_right=*hint+radius;
                }
            }
        }
        double t=0.,d=d0;
        for (int i=0;i<4096;++i) {
            if (d<=tol) {
                result.fraction=result.start_in_contact ? 1-t : t;
                return result;
            }
            double next=t+(d-.25*tol)/speed;
            if (next>=safe_left && next<safe_right) next=safe_right;
            if (next>=horizon) {
                if (known_hit) result.fraction=result.start_in_contact ? 1-horizon : horizon;
                return result;
            }
            if (!(next>t)) break;
            t=next;
            d=distance(a.at(t),b.at(t));
        }
        throw UnresolvedContact("CAD conservative advancement exhausted its interval budget");
    } catch (const Standard_Failure& e) {
        throw UnresolvedContact(std::string("CAD sweep failed: ")+e.GetMessageString());
    }
private:
    Solid a_,b_;
    std::array<Vec,2> centers_;
    CandidateQuery candidate_;
};
}

std::shared_ptr<const dynamics::PairRegionQuery> cad_solid_region_query(
    const TopoDS_Shape& a,const TopoDS_Shape& b,std::array<Vec,2> centers,bool chen_events) {
    return std::make_shared<CadSolidPair>(a,b,centers,chen_events);
}
}
