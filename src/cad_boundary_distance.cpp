#include "cad_boundary_distance.hpp"
#include "cad_planar_trim.hpp"
#include <BRepAdaptor_Curve.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <BRepBndLib.hxx>
#include <BRepClass3d_SolidClassifier.hxx>
#include <BRepExtrema_DistanceSS.hxx>
#include <BRep_Tool.hxx>
#include <Bnd_Box.hxx>
#include <Geom_BSplineCurve.hxx>
#include <Geom_BSplineSurface.hxx>
#include <Geom_BezierCurve.hxx>
#include <Geom_BezierSurface.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Vertex.hxx>
#include <TopoDS_Face.hxx>
#include <gp_Ax3.hxx>
#include <limits>
#include <mutex>
#include <queue>

namespace nc::cad {
namespace {
Bnd_Box box_of(const TopoDS_Shape& shape) {
    Bnd_Box box;
    BRepBndLib::Add(shape,box,false);
    if (box.IsVoid() || box.IsOpen()) throw dynamics::UnresolvedContact("unbounded STEP feature");
    return box;
}

// A bit denotes a Cartesian coordinate that is constant along a curve or an
// entire isoparametric family. Retain the two parameter directions separately.
unsigned constant_components(const gp_XYZ& displacement) {
    return (displacement.X()==0. ? 1u : 0u) | (displacement.Y()==0. ? 2u : 0u) |
           (displacement.Z()==0. ? 4u : 0u);
}
struct FlatDirections {
    unsigned u{},v{};
    bool preserves_height(const gp_XYZ& plane_normal) const {
        const unsigned active=7u^constant_components(plane_normal);
        return active && ((u&active)==active || (v&active)==active);
    }
};
// A trim is contained in the convex hull of its positive-weight spline.
// Unsupported geometry retains the original OCCT bounding box.
struct SupportHull {
    std::vector<gp_XYZ> points;
    gp_XYZ direction;
    double tolerance{};
    double magnitude{};
    gp_XYZ support(const gp_XYZ& axis) const {
        auto best=points.front();
        auto high=best.Dot(axis);
        for (std::size_t i=1;i<points.size();++i) {
            const double value=points[i].Dot(axis);
            if (value>high) {high=value;best=points[i];}
        }
        return best;
    }
    template<class Curve> void curve(const Curve& c) {
        if (c->NbPoles()>256) return;
        for (int i=1;i<=c->NbPoles();++i) {
            if (!(c->Weight(i)>0)) {points.clear();return;}
            points.push_back(c->Pole(i).XYZ());
        }
    }
    template<class Surface> void surface(const Surface& s) {
        if (s->NbUPoles()>256/s->NbVPoles()) return;
        for (int i=1;i<=s->NbUPoles();++i) for (int j=1;j<=s->NbVPoles();++j) {
            if (!(s->Weight(i,j)>0)) {points.clear();return;}
            points.push_back(s->Pole(i,j).XYZ());
        }
    }
    void box(const Bnd_Box& bounds) {
        const auto lo=bounds.CornerMin().XYZ(),hi=bounds.CornerMax().XYZ();
        for (int i=0;i<8;++i) points.emplace_back(i&1 ? hi.X() : lo.X(),
            i&2 ? hi.Y() : lo.Y(),i&4 ? hi.Z() : lo.Z());
    }
};
template<class Curve>
unsigned curve_flat(const Curve& c) {
    unsigned flat=7;
    for (int i=2;i<=c->NbPoles() && flat;++i)
        flat&=constant_components(c->Pole(i).XYZ()-c->Pole(1).XYZ());
    return flat;
}
template<class Surface>
FlatDirections surface_flat(const Surface& s,bool periodic_u,bool periodic_v) {
    FlatDirections flat{periodic_u ? 0u : 7u,periodic_v ? 0u : 7u};
    for (int i=1;i<=s->NbUPoles();++i) for (int j=1;j<=s->NbVPoles();++j) {
        const auto p=s->Pole(i,j).XYZ();
        // Uniform weights along the varying direction keep the other
        // direction's rational coefficients independent of that parameter.
        if (s->Weight(i,j)!=s->Weight(1,j)) flat.u=0;
        if (s->Weight(i,j)!=s->Weight(i,1)) flat.v=0;
        flat.u&=constant_components(p-s->Pole(1,j).XYZ());
        flat.v&=constant_components(p-s->Pole(i,1).XYZ());
    }
    return flat;
}
struct FeatureBox {
    struct TrimCache {
        std::once_flag once;
        std::optional<PlanarTrim> trim;
    };
    TopoDS_Shape shape;
    Bnd_Box box;
    gp_Pnt center;
    bool plane{};
    gp_XYZ normal;
    FlatDirections flat;
    SupportHull hull;
    std::shared_ptr<TrimCache> trim;
    bool excludes_projection(const SupportHull& other,const gp_Trsf& relative,double limit) const {
        if (!trim) return false;
        std::call_once(trim->once,[&] {
            const auto face=TopoDS::Face(shape);
            trim->trim.emplace(face,BRepAdaptor_Surface(face).Plane());
        });
        return trim->trim->separates(other.points,relative,limit+other.tolerance);
    }
    explicit FeatureBox(const TopoDS_Shape& s) : shape(s),box(box_of(s)),
        center((box.CornerMin().XYZ()+box.CornerMax().XYZ())*.5) {
        if (s.ShapeType()==TopAbs_VERTEX) {
            const auto v=TopoDS::Vertex(s);
            hull.points.push_back(BRep_Tool::Pnt(v).XYZ());
            hull.tolerance=BRep_Tool::Tolerance(v);
        } else if (s.ShapeType()==TopAbs_EDGE && TopExp_Explorer(s,TopAbs_VERTEX).More()) {
            const BRepAdaptor_Curve c(TopoDS::Edge(s));
            hull.tolerance=BRep_Tool::Tolerance(TopoDS::Edge(s));
            switch (c.GetType()) {
                case GeomAbs_Line:
                    flat.u=constant_components(c.Line().Direction().XYZ());
                    hull.points={c.Value(c.FirstParameter()).XYZ(),c.Value(c.LastParameter()).XYZ()};
                    break;
                case GeomAbs_BSplineCurve: {
                    auto spline=c.BSpline(); flat.u=curve_flat(spline);
                    if (spline->IsPeriodic() || (c.FirstParameter()>=spline->FirstParameter() &&
                        c.LastParameter()<=spline->LastParameter())) hull.curve(spline);
                    break;
                }
                case GeomAbs_BezierCurve: {
                    auto bez=c.Bezier(); flat.u=curve_flat(bez);
                    if (c.FirstParameter()>=0 && c.LastParameter()<=1) hull.curve(bez);
                    break;
                }
                default: break;
            }
        } else if (s.ShapeType()==TopAbs_FACE) {
            const BRepAdaptor_Surface surf(TopoDS::Face(s));
            hull.tolerance=BRep_Tool::Tolerance(TopoDS::Face(s));
            switch (surf.GetType()) {
                case GeomAbs_Plane: plane=true; normal=surf.Plane().Axis().Direction().XYZ(); break;
                case GeomAbs_BSplineSurface: {
                    auto spline=surf.BSpline(); flat=surface_flat(spline,surf.IsUPeriodic(),surf.IsVPeriodic());
                    double u,v,U,V; spline->Bounds(u,U,v,V);
                    if ((spline->IsUPeriodic() || (surf.FirstUParameter()>=u && surf.LastUParameter()<=U)) &&
                        (spline->IsVPeriodic() || (surf.FirstVParameter()>=v && surf.LastVParameter()<=V)))
                        hull.surface(spline);
                    break;
                }
                case GeomAbs_BezierSurface: {
                    const auto bez=surf.Bezier(); flat=surface_flat(bez,false,false);
                    if (surf.FirstUParameter()>=0 && surf.LastUParameter()<=1 &&
                        surf.FirstVParameter()>=0 && surf.LastVParameter()<=1) hull.surface(bez);
                    break;
                }
                default: break;
            }
            if (plane) {hull.direction=normal;trim=std::make_shared<TrimCache>();}
            else if (!hull.points.empty()) {
                gp_Pnt p;gp_Vec u,v;
                try {
                    surf.D1(.5*(surf.FirstUParameter()+surf.LastUParameter()),
                            .5*(surf.FirstVParameter()+surf.LastVParameter()),p,u,v);
                    hull.direction=u.Crossed(v).XYZ();
                } catch (const Standard_Failure&) {} // The support test can choose another axis.
            }
        }
        if (hull.points.empty()) hull.box(box);
        for (const auto& p:hull.points) hull.magnitude=std::max(hull.magnitude,p.Modulus());
    }
};

// Each rejection is a direct separating-axis test on both complete hulls.
// The iterative search only chooses directions. Stagnation or exhaustion
// falls through to the existing trimmed-feature distance kernel.
bool hulls_separate(const FeatureBox& a,const FeatureBox& b,const gp_Trsf& pose,
                    const gp_Mat& inverse,double limit) {
    const auto rotation=pose.VectorialPart();
    const auto translation=pose.TranslationPart();
    const auto closest_support=[&](const gp_XYZ& axis) {
        return rotation*a.hull.support(inverse*axis.Reversed())+translation-b.hull.support(axis);
    };
    limit+=a.hull.tolerance+b.hull.tolerance+128*std::numeric_limits<double>::epsilon()*
           (a.hull.magnitude+b.hull.magnitude+translation.Modulus()+1.);
    gp_XYZ p=rotation*a.center.XYZ()+translation-b.center.XYZ();
    for (auto axis:{rotation*a.hull.direction,b.hull.direction}) {
        if (axis.Dot(p)<0) axis.Reverse();
        if (axis.SquareModulus()>0 && closest_support(axis).Dot(axis)>limit*axis.Modulus()) return true;
    }
    // Start inside A-B to keep subsequent segment updates bounded.
    p=closest_support(p);
    for (int i=0;i<24;++i) {
        const double norm=p.Modulus();
        if (norm<=limit) break;
        const auto q=closest_support(p);
        if (p.Dot(q)>limit*norm) return true;
        const auto delta=q-p;
        const auto length2=delta.SquareModulus();
        if (!(length2>0)) break;
        const auto t=std::clamp(-p.Dot(delta)/length2,0.,1.);
        if (!(t>0)) break;
        p+=delta*t;
    }
    return false;
}
struct Node {
    Bnd_Box box;
    std::size_t begin{},end{},left{},right{};
    bool leaf() const {return !right;}
};
struct Component { TopoDS_Shape solid; Bnd_Box box; };
struct DirectionalBoxes {
    gp_Trsf frame;
    std::vector<Bnd_Box> features,nodes;
};
}

class BoundaryIndex {
public:
    std::vector<FeatureBox> features;
    std::vector<Node> nodes;
    std::vector<Component> solids;
    std::vector<std::array<gp_Pnt,6>> shell_points;

    // A second set of enclosures in a body-fixed frame aligned with the
    // separating direction at the first successful reuse of a near feature.
    const DirectionalBoxes* directional(const gp_Trsf& frame) const {
        std::call_once(frame_once_,[&] {
            try {
                DirectionalBoxes result{frame};
                const TopLoc_Location location(frame);
                result.features.reserve(features.size());
                for (const auto& feature:features)
                    result.features.push_back(box_of(feature.shape.Moved(location)));
                result.nodes.resize(nodes.size());
                for (auto i=nodes.size();i-- > 0;) {
                    const auto& node=nodes[i];
                    if (node.leaf()) {
                        for (auto f=node.begin;f<node.end;++f) result.nodes[i].Add(result.features[f]);
                    } else {
                        result.nodes[i].Add(result.nodes[node.left]);
                        result.nodes[i].Add(result.nodes[node.right]);
                    }
                }
                directional_.emplace(std::move(result));
            } catch (const Standard_Failure&) {
                // The ordinary BVH remains available if OCCT rejects this frame.
            } catch (const dynamics::UnresolvedContact&) {
                // A valid bound is required on every child of the second tree.
            }
        });
        return directional_ ? &*directional_ : nullptr;
    }

    explicit BoundaryIndex(const TopoDS_Shape& shape) {
        // The extrema kernel tests interiors of individual features. All
        // vertices and edges are included to cover their trimmed boundaries.
        for (auto kind:{TopAbs_VERTEX,TopAbs_EDGE,TopAbs_FACE}) {
            TopTools_IndexedMapOfShape items;
            TopExp::MapShapes(shape,kind,items);
            for (int i=1;i<=items.Extent();++i) {
                const auto& item=items(i);
                if (kind==TopAbs_EDGE && BRep_Tool::Degenerated(TopoDS::Edge(item))) continue;
                features.emplace_back(item);
            }
        }
        if (features.empty()) throw dynamics::UnresolvedContact("empty STEP boundary");
        build(0,features.size());
        for (TopExp_Explorer s(shape,TopAbs_SOLID);s.More();s.Next()) {
            solids.push_back({s.Current(),box_of(s.Current())});
            for (TopExp_Explorer shell(s.Current(),TopAbs_SHELL);shell.More();shell.Next()) {
                TopTools_IndexedMapOfShape vertices;
                TopExp::MapShapes(shell.Current(),TopAbs_VERTEX,vertices);
                if (vertices.IsEmpty()) throw dynamics::UnresolvedContact("STEP shell has no representative vertex");
                std::array<gp_Pnt,6> sample;
                sample.fill(BRep_Tool::Pnt(TopoDS::Vertex(vertices(1))));
                for (int i=2;i<=vertices.Extent();++i) {
                    const auto p=BRep_Tool::Pnt(TopoDS::Vertex(vertices(i)));
                    for (int k=0;k<3;++k) {
                        if (p.Coord(k+1)<sample[2*k].Coord(k+1)) sample[2*k]=p;
                        if (p.Coord(k+1)>sample[2*k+1].Coord(k+1)) sample[2*k+1]=p;
                    }
                }
                shell_points.push_back(sample);
            }
        }
    }
private:
    mutable std::once_flag frame_once_;
    mutable std::optional<DirectionalBoxes> directional_;
    std::size_t build(std::size_t begin,std::size_t end) {
        const auto id=nodes.size();
        nodes.push_back({});
        Bnd_Box centers,box;
        for (auto i=begin;i<end;++i) { box.Add(features[i].box); centers.Add(features[i].center); }
        nodes[id]={box,begin,end};
        if (end-begin>4) {
            const auto span=centers.CornerMax().XYZ()-centers.CornerMin().XYZ();
            int axis=1;
            for (int k=2;k<=3;++k) if (span.Coord(k)>span.Coord(axis)) axis=k;
            const auto mid=begin+(end-begin)/2;
            std::nth_element(features.begin()+begin,features.begin()+mid,features.begin()+end,
                [axis](const auto& a,const auto& b){return a.center.Coord(axis)<b.center.Coord(axis);});
            const auto left=build(begin,mid),right=build(mid,end);
            nodes[id].left=left; nodes[id].right=right;
        }
        return id;
    }
};

std::shared_ptr<const BoundaryIndex> boundary_index(const TopoDS_Shape& shape) {
    return std::make_shared<BoundaryIndex>(shape);
}

namespace {
// After excluding crossings, a connected shell is entirely inside or
// outside each opposing solid. An actual exterior vertex rules out enclosure.
bool enclosed(const BoundaryIndex& a,const BoundaryIndex& b,const gp_Trsf& placement,
              double tol,dynamics::SurfaceTransition& stats) {
    for (const auto& points:a.shell_points) for (const auto& other:b.solids) {
        if (std::any_of(points.begin(),points.end(),[&](const gp_Pnt& p){
                return other.box.IsOut(p.Transformed(placement));
            })) continue;
        ++stats.cad_classifications;
        BRepClass3d_SolidClassifier classify(other.solid,points[0].Transformed(placement),tol);
        const auto state=classify.State();
        if (state==TopAbs_IN || state==TopAbs_ON) return true;
        if (state!=TopAbs_OUT) throw dynamics::UnresolvedContact("indeterminate STEP solid containment");
    }
    return false;
}
}

double bounded_distance(const std::shared_ptr<const BoundaryIndex>& first,const std::shared_ptr<const BoundaryIndex>& second,
                        const gp_Trsf& placement,double cap,double tol,dynamics::SurfaceTransition& stats,
                        BoundaryWitness* witness) {
    const auto& a=*first;
    const auto& b=*second;
    struct Pair {
        double lower;
        std::size_t i,j;
        bool operator<(const Pair& rhs) const { return lower>rhs.lower; }
    };
    const TopLoc_Location location(placement);
    const auto reverse_placement=placement.Inverted();
    const auto rotation=placement.VectorialPart(),inverse=rotation.Transposed();
    std::priority_queue<Pair> queue;
    const double eps=std::min(.05*tol,Precision::Confusion());
    double best=cap;
    const bool reusable=witness && witness->first.lock()==first && witness->second.lock()==second;
    gp_XYZ normal=reusable ? witness->direction : gp_XYZ{};
    const auto evaluate=[&](std::size_t x,std::size_t y,const TopoDS_Shape& shifted,const Bnd_Box& box) {
        const auto& g=b.features[y];
        ++stats.cad_extrema;
        BRepExtrema_DistanceSS test(shifted,g.shape,box,g.box,best,eps);
        if (!test.IsDone()) return false;
        const double d=test.DistValue();
        if (!std::isfinite(d)) throw dynamics::UnresolvedContact("nonfinite STEP feature distance");
        if (d>=best) return false;
        best=d;
        if (witness && d>tol && !test.Seq1Value().IsEmpty() && !test.Seq2Value().IsEmpty())
            normal=test.Seq2Value().First().Point().XYZ()-test.Seq1Value().First().Point().XYZ();
        if (witness) *witness={first,second,x,y,normal};
        return true;
    };
    std::optional<std::pair<std::size_t,std::size_t>> seed;
    if (reusable) {
        const auto x=witness->x,y=witness->y;
        const auto& f=a.features[x];
        const auto box=f.box.Transformed(placement);
        if (box.Distance(b.features[y].box)<=cap+eps) {
            seed.emplace(x,y);
            ++stats.cad_witness_tests;
            if (evaluate(x,y,f.shape.Moved(location),box)) {
                ++stats.cad_witness_improvements;
                if (best<=tol) return 0.;
            }
        }
    }
    const DirectionalBoxes *da=nullptr,*db=nullptr;
    gp_Trsf relative_frame;
    if (reusable && normal.Modulus()>tol) {
        // Fit boxes to the CAD geometry in the new frames. Rotating the old
        // AABBs would only enclose them again and lose the tighter bounds.
        gp_Trsf frame_b;
        frame_b.SetTransformation(gp_Ax3({},gp_Dir(normal)));
        auto frame_a=frame_b*placement;
        frame_a.SetTranslationPart(gp_Vec());
        da=a.directional(frame_a);
        db=b.directional(frame_b);
        // A concurrent query may have initialized either index in another
        // frame. Use the frames of the boxes actually returned by the indices.
        if (da && db) relative_frame=db->frame*placement*da->frame.Inverted();
    }
    const double frame_pad=8*Precision::Confusion()+128*std::numeric_limits<double>::epsilon()*
        (a.nodes.front().box.CornerMin().XYZ().Modulus()+a.nodes.front().box.CornerMax().XYZ().Modulus()+
         b.nodes.front().box.CornerMin().XYZ().Modulus()+b.nodes.front().box.CornerMax().XYZ().Modulus()+
         placement.TranslationPart().Modulus());
    const auto push=[&](std::size_t i,std::size_t j) {
        double lower=a.nodes[i].box.Transformed(placement).Distance(b.nodes[j].box);
        ++stats.cad_box_tests;
        if (da && db && lower<=best+eps) {
            lower=std::max(lower,da->nodes[i].Transformed(relative_frame).Distance(db->nodes[j])-frame_pad);
            ++stats.cad_box_tests;
        }
        if (lower<=best+eps) queue.push({lower,i,j});
    };
    push(0,0);
    while (!queue.empty()) {
        const auto [lower,i,j]=queue.top(); queue.pop();
        if (lower>best+eps) break;
        const auto& left=a.nodes[i]; const auto& right=b.nodes[j];
        if (left.leaf() && right.leaf()) {
            for (auto x=left.begin;x<left.end;++x) {
                const auto& f=a.features[x];
                const auto box=f.box.Transformed(placement);
                const auto shifted=f.shape.Moved(location);
                const auto aligned=da && db ? da->features[x].Transformed(relative_frame) : Bnd_Box{};
                for (auto y=right.begin;y<right.end;++y) {
                    const auto& g=b.features[y];
                    if (seed && seed->first==x && seed->second==y) continue;
                    // A minimizing pair in two bounded plane interiors can slide
                    // along a common tangent until a trim is reached. Its distance
                    // is covered by the edge/face or lower-dimensional queries.
                    if (f.plane && g.plane) continue;
                    // If the projection is inside the plane trim, follow a
                    // constant-height parameter line to either feature's trim.
                    // The edge/face, edge/edge or vertex tests cover that pair
                    // at the same distance, including a zero-distance touch.
                    if ((f.plane && g.flat.preserves_height(rotation*f.normal)) ||
                        (g.plane && f.flat.preserves_height(inverse*g.normal))) continue;
                    ++stats.cad_box_tests;
                    if (box.Distance(g.box)>best+eps) continue;
                    if (da && db) {
                        ++stats.cad_box_tests;
                        if (aligned.Distance(db->features[y])>best+eps+frame_pad) continue;
                    }
                    if (hulls_separate(f,g,placement,inverse,best+eps+frame_pad)) continue;
                    if (g.excludes_projection(f.hull,placement,best+eps+frame_pad) ||
                        f.excludes_projection(g.hull,reverse_placement,best+eps+frame_pad)) continue;
                    if (evaluate(x,y,shifted,box) && best<=tol) return 0.;
                }
            }
        } else if (!left.leaf() && (right.leaf() || left.end-left.begin>=right.end-right.begin)) {
            push(left.left,j); push(left.right,j);
        } else {
            push(i,right.left); push(i,right.right);
        }
    }
    if (enclosed(a,b,placement,eps,stats) || enclosed(b,a,placement.Inverted(),eps,stats)) return 0.;
    return best;
}
}
