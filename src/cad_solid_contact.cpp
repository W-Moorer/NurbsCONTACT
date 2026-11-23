#include "cad_rigid_path.hpp"
#include "cad_boundary_distance.hpp"
#include "cad_local_solid.hpp"
#include <mutex>
#include <deque>
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

using TransformKey=std::array<double,12>;
TransformKey key(const gp_Trsf& pose) {
    TransformKey value;
    for (int i=0;i<3;++i) for (int j=0;j<4;++j) value[4*i+j]=pose.Value(i+1,j+1);
    return value;
}

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

}
namespace cad {
std::vector<dynamics::ContactRegion> solid_regions(const TopoDS_Shape& first,const TopoDS_Shape& second,
                                                  dynamics::ContactWork& work) try {
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
    throw dynamics::UnresolvedContact(std::string("CAD intersection failed: ")+e.GetMessageString());
}
}

namespace {
class CadSolidPair final : public PairRegionQuery {
public:
    CadSolidPair(const TopoDS_Shape& a,const TopoDS_Shape& b,std::array<Vec,2> centers,CadEventMethod method)
        : a_(a,centers[0]),b_(b,centers[1]),centers_(centers) {
        if (method==CadEventMethod::chen_2026) candidate_=chen_2026_candidates(a,b);
        if (method==CadEventMethod::chen_2024) {
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
        const std::array poses{key(ta),key(tb)};
        {
            std::lock_guard lock(cache_mutex_);
            for (const auto& entry:regions_) if (entry.poses==poses) {
                ++work.cad_region_cache_hits;
                return entry.regions;
            }
        }
        initialize_search();
        LocalSolidPair pair{{a_.shape,{}},{b_.shape,{}},false};
        if (local_) pair=local_->select(tb.Inverted()*ta,100*Precision::Confusion());
        std::vector<ContactRegion> answer;
        if (!pair.separated) {
            try {
                answer=solid_regions(pair.a.shape.Moved(TopLoc_Location(ta)),pair.b.shape.Moved(TopLoc_Location(tb)),work);
            } catch (const UnresolvedContact&) {
                if (pair.a.shape.IsSame(a_.shape) && pair.b.shape.IsSame(b_.shape)) throw;
                answer=solid_regions(a_.shape.Moved(TopLoc_Location(ta)),b_.shape.Moved(TopLoc_Location(tb)),work);
            }
        }
        {
            std::lock_guard lock(cache_mutex_);
            if (regions_.size()==8) regions_.pop_back();
            regions_.push_front({poses,answer});
        }
        return answer;
    } catch (const Standard_Failure& e) {
        throw UnresolvedContact(std::string("CAD overlap failed: ")+e.GetMessageString());
    }
    SurfaceTransition transition(ContactPose a0,ContactPose b0,ContactPose a1,ContactPose b1,
                                 double tolerance_m) const override {
        return transition(a0,b0,a1,b1,tolerance_m,0.);
    }
    SurfaceTransition transition(ContactPose a0,ContactPose b0,ContactPose a1,ContactPose b1,
                                 double tolerance_m,double fraction_tolerance) const override try {
        if (!(tolerance_m>0 && std::isfinite(tolerance_m))) throw std::invalid_argument("invalid CAD event tolerance");
        SurfaceTransition result;
        const double tol=std::max(1000*tolerance_m,2.*Precision::Confusion());
        RigidPath a(a0,a1,centers_[0]),b(b0,b1,centers_[1]);
        if (a_.swept(a).Distance(b_.swept(b))>tol) return result;
        const double global_speed=(gp_Vec(a.from,a.to)-gp_Vec(b.from,b.to)).Magnitude()+
                           std::abs(a.angle)*a_.radius+std::abs(b.angle)*b_.radius;
        initialize_search();
        const double speed=local_ ? local_->motion_bound(a,b,global_speed,global_speed+2*tol) : global_speed;
        if (!boundaries_[0] && !local_) fraction_tolerance=0.;
        const auto distance=[&](const gp_Trsf& a,const gp_Trsf& b) {
            ++result.nodes;
            if (boundaries_[0] || local_) {
                const auto transform=b.Inverted()*a;
                const auto pose=key(transform);
                const double cap=speed+2*tol;
                BoundaryWitness witness;
                {
                    std::lock_guard lock(cache_mutex_);
                    for (const auto& entry:distances_)
                        if (entry.pose==pose && tol==entry.tol && (cap<=entry.cap || entry.distance<entry.cap)) {
                            ++result.cad_distance_cache_hits;
                            return std::min(cap,entry.distance);
                        }
                    witness=witness_;
                }
                ++result.cad_distances;
                double value;
                if (local_) {
                    const auto parts=local_->select(transform,cap);
                    value=parts.separated ? cap : bounded_distance(parts.a.boundary,parts.b.boundary,
                                                                   transform,cap,tol,result,&witness);
                } else value=bounded_distance(boundaries_[0],boundaries_[1],transform,cap,tol,result,&witness);
                std::lock_guard lock(cache_mutex_);
                witness_=std::move(witness);
                if (distances_.size()==32) distances_.pop_back();
                distances_.push_front({pose,cap,tol,value});
                return value;
            }
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
        if (speed==0 || d0>speed+tol) return result;
        double horizon=1.,safe_left=1.,safe_right=1.;
        bool known_hit=d1<=tol;
        // A separated far endpoint also excludes the final part of the step.
        const double end_clear_from=known_hit ? 1. : 1.-(d1-tol)/speed;
        // Validate a polynomial-patch proposal on both original trimmed solids.
        // The distance at a rejected proposal can also exclude a time interval.
        if (candidate_) {
            const auto hint=candidate_(a,b,result);
            const auto probe=[&](double t) {
                const double d=t==0. ? d0 : t==1. ? d1 : distance(a.at(t),b.at(t));
                if (d<=tol) {
                    horizon=t;
                    known_hit=true;
                    ++result.chen_confirmed;
                } else {
                    const double radius=(d-tol)/speed;
                    safe_left=t-radius;
                    safe_right=t+radius;
                }
            };
            if (hint) {
                // The inclusion cell can begin before physical contact. A
                // loaded right end bounds the subsequent trimmed-solid sweep.
                probe(hint->hi);
                if (hint->lo<hint->hi && hint->lo<horizon && hint->lo<safe_left) probe(hint->lo);
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
            if (!known_hit && next>end_clear_from) return result;
            if (next>=horizon) {
                if (known_hit) result.fraction=result.start_in_contact ? 1-horizon : horizon;
                return result;
            }
            // A verified upper endpoint and a collision-free prefix bracket the
            // earliest physical touch. Respect the integrator's event resolution.
            if (known_hit && fraction_tolerance>0. && horizon-next<=fraction_tolerance) {
                result.fraction=result.start_in_contact ? 1-horizon : horizon;
                return result;
            }
            if (known_hit && fraction_tolerance>0. && i%4==0) {
                const double mid=std::midpoint(next,horizon);
                const double dmid=distance(a.at(mid),b.at(mid));
                if (dmid<=tol) horizon=mid;
                else {
                    const double radius=(dmid-tol)/speed;
                    safe_left=mid-radius;
                    safe_right=mid+radius;
                    if (next>=safe_left && next<safe_right) next=std::min(safe_right,horizon);
                }
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
    void initialize_search() const {
        std::call_once(index_once_,[&] {
            TopTools_IndexedMapOfShape fa,fb;
            TopExp::MapShapes(a_.shape,TopAbs_FACE,fa);
            TopExp::MapShapes(b_.shape,TopAbs_FACE,fb);
            const int faces=fa.Extent()+fb.Extent();
            if (faces<=16) return;
            if (faces>=64) {
                try {
                    local_=std::make_unique<LocalSolidSearch>(a_.shape,b_.shape);
                    return;
                } catch (const UnresolvedContact&) {
                    // Try the whole-body boundary index if local envelopes are unavailable.
                } catch (const Standard_Failure&) {
                    // OCCT can reject optional envelope geometry.
                }
            }
            try {
                auto first=boundary_index(a_.shape),second=boundary_index(b_.shape);
                boundaries_={std::move(first),std::move(second)};
            } catch (const UnresolvedContact&) {
                // Unsupported topology still uses the whole-solid kernel.
            } catch (const Standard_Failure&) {
                // Use the same whole-solid kernel if index construction fails inside OCCT.
            }
        });
    }
    struct RegionEntry { std::array<TransformKey,2> poses; std::vector<ContactRegion> regions; };
    struct DistanceEntry { TransformKey pose; double cap,tol,distance; };
    Solid a_,b_;
    mutable std::mutex cache_mutex_;
    mutable std::deque<RegionEntry> regions_;
    mutable std::deque<DistanceEntry> distances_;
    mutable BoundaryWitness witness_;
    mutable std::once_flag index_once_;
    mutable std::unique_ptr<LocalSolidSearch> local_;
    mutable std::array<std::shared_ptr<const BoundaryIndex>,2> boundaries_;
    std::array<Vec,2> centers_;
    CandidateQuery candidate_;
};
}

std::shared_ptr<const dynamics::PairRegionQuery> cad_solid_region_query(
    const TopoDS_Shape& a,const TopoDS_Shape& b,std::array<Vec,2> centers,CadEventMethod chen_events) {
    return std::make_shared<CadSolidPair>(a,b,centers,chen_events);
}
}
