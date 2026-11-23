#include "cad_boundary_distance.hpp"
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
    TopoDS_Shape shape;
    Bnd_Box box;
    gp_Pnt center;
    bool plane{};
    gp_XYZ normal;
    FlatDirections flat;
    explicit FeatureBox(const TopoDS_Shape& s) : shape(s),box(box_of(s)),
        center((box.CornerMin().XYZ()+box.CornerMax().XYZ())*.5) {
        if (s.ShapeType()==TopAbs_EDGE && TopExp_Explorer(s,TopAbs_VERTEX).More()) {
            const BRepAdaptor_Curve c(TopoDS::Edge(s));
            switch (c.GetType()) {
                case GeomAbs_Line: flat.u=constant_components(c.Line().Direction().XYZ()); break;
                case GeomAbs_BSplineCurve: flat.u=curve_flat(c.BSpline()); break;
                case GeomAbs_BezierCurve: flat.u=curve_flat(c.Bezier()); break;
                default: break;
            }
        } else if (s.ShapeType()==TopAbs_FACE) {
            const BRepAdaptor_Surface surf(TopoDS::Face(s));
            switch (surf.GetType()) {
                case GeomAbs_Plane: plane=true; normal=surf.Plane().Axis().Direction().XYZ(); break;
                case GeomAbs_BSplineSurface: flat=surface_flat(surf.BSpline(),surf.IsUPeriodic(),surf.IsVPeriodic()); break;
                case GeomAbs_BezierSurface: flat=surface_flat(surf.Bezier(),false,false); break;
                default: break;
            }
        }
    }
};
struct Node {
    Bnd_Box box;
    std::size_t begin{},end{},left{},right{};
    bool leaf() const {return !right;}
};
struct Component { TopoDS_Shape solid; Bnd_Box box; };
}

class BoundaryIndex {
public:
    std::vector<FeatureBox> features;
    std::vector<Node> nodes;
    std::vector<Component> solids;
    std::vector<std::array<gp_Pnt,6>> shell_points;

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
    const auto rotation=placement.VectorialPart(),inverse=rotation.Transposed();
    std::priority_queue<Pair> queue;
    const double eps=std::min(.05*tol,Precision::Confusion());
    double best=cap;
    const auto evaluate=[&](std::size_t x,std::size_t y,const TopoDS_Shape& shifted,const Bnd_Box& box) {
        const auto& g=b.features[y];
        ++stats.cad_extrema;
        BRepExtrema_DistanceSS test(shifted,g.shape,box,g.box,best,eps);
        if (!test.IsDone()) return false;
        const double d=test.DistValue();
        if (!std::isfinite(d)) throw dynamics::UnresolvedContact("nonfinite STEP feature distance");
        if (d>=best) return false;
        best=d;
        if (witness) *witness={first,second,x,y};
        return true;
    };
    std::optional<std::pair<std::size_t,std::size_t>> seed;
    if (witness && witness->first.lock()==first && witness->second.lock()==second) {
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
    const auto push=[&](std::size_t i,std::size_t j) {
        const double lower=a.nodes[i].box.Transformed(placement).Distance(b.nodes[j].box);
        ++stats.cad_box_tests;
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
