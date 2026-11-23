#include "cad_local_solid.hpp"
#include "cad_rigid_path.hpp"
#include <BRepAlgoAPI_Common.hxx>
#include <BRepBndLib.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <TopExp_Explorer.hxx>
#include <gp_Ax1.hxx>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <mutex>
#include <numbers>
#include <optional>

namespace nc::cad {
namespace {
using Range=std::array<double,3>;
struct Box {
    Range lo,hi;
    bool contains(const Box& other) const {
        for (int k=0;k<3;++k) if (other.lo[k]<lo[k] || other.hi[k]>hi[k]) return false;
        return true;
    }
    bool empty() const { return lo[0]>hi[0] || lo[1]>hi[1] || lo[2]>hi[2]; }
};
Box bounds(const TopoDS_Shape& shape) {
    Bnd_Box b;
    BRepBndLib::Add(shape,b,false);
    if (b.IsVoid() || b.IsOpen()) throw dynamics::UnresolvedContact("unbounded local CAD envelope");
    Box result;
    b.Get(result.lo[0],result.lo[1],result.lo[2],result.hi[0],result.hi[1],result.hi[2]);
    return result;
}
struct Slab { gp_XYZ normal; double lo,hi; };
struct Envelope {
    Box box;
    std::vector<Slab> slabs;
    double scale=1.;
    explicit Envelope(const TopoDS_Shape& shape) : box(bounds(shape)) {
        for (int k=0;k<3;++k) {
            gp_XYZ axis; axis.SetCoord(k+1,1.);
            slabs.push_back({axis,box.lo[k],box.hi[k]});
            scale=std::max({scale,std::abs(box.lo[k]),std::abs(box.hi[k])});
            for (int j=1;j<=3;++j) {
                gp_Trsf turn;
                turn.SetRotation(gp_Ax1({},gp_Dir(axis)),std::numbers::pi*j/8.);
                const auto projected=bounds(shape.Moved(TopLoc_Location(turn)));
                for (int d=0;d<3;++d) if (d!=k)
                    slabs.push_back({{turn.Value(d+1,1),turn.Value(d+1,2),turn.Value(d+1,3)},
                                     projected.lo[d],projected.hi[d]});
            }
        }
    }
};

// Intersect an enclosing box with u.x <= limit. Minimum contributions from
// the other two coordinates give a conservative bound on each coordinate.
void tighten(Box& box,const gp_XYZ& u,double limit,double padding) {
    for (int k=0;k<3;++k) {
        const double c=u.Coord(k+1);
        if (std::abs(c)<1e-8) continue;
        double rhs=limit+padding;
        for (int j=0;j<3;++j) if (j!=k) {
            const double n=u.Coord(j+1);
            rhs-=n*(n>=0. ? box.lo[j] : box.hi[j]);
        }
        if (c>0.) box.hi[k]=std::min(box.hi[k],rhs/c+padding);
        else box.lo[k]=std::max(box.lo[k],rhs/c-padding);
    }
}
Box interaction(const Envelope& own,const Envelope& other,const gp_Trsf& other_in_own,double reach) {
    Box box=own.box;
    const auto shift=other_in_own.TranslationPart();
    const double pad=8*Precision::Confusion()+128*std::numeric_limits<double>::epsilon()*
                     (own.scale+other.scale+shift.Modulus()+reach);
    for (int pass=0;pass<3;++pass) {
        for (const auto& s:other.slabs) {
            const auto n=other_in_own.VectorialPart()*s.normal;
            const double offset=n.Dot(shift);
            tighten(box,n,s.hi+offset+reach,pad);
            tighten(box,n.Reversed(),-s.lo-offset+reach,pad);
        }
        for (const auto& s:own.slabs) {
            tighten(box,s.normal,s.hi,pad);
            tighten(box,s.normal.Reversed(),-s.lo,pad);
        }
        if (box.empty()) break;
    }
    return box;
}

class LocalBody {
public:
    explicit LocalBody(const TopoDS_Shape& s) : envelope(s),full{s,boundary_index(s)} {}
    Envelope envelope;
    LocalSolidGeometry full;
    LocalSolidGeometry select(const Box& required,double reach) {
        if (clip && clip->bounds.contains(required)) return clip->piece;
        Box window=envelope.box;
        bool reduced=false;
        for (int k=0;k<3;++k) {
            const double span=envelope.box.hi[k]-envelope.box.lo[k];
            const double margin=std::max({.05*span,2*reach,100*Precision::Confusion()});
            window.lo[k]=std::max(envelope.box.lo[k]-margin,required.lo[k]-margin);
            window.hi[k]=std::min(envelope.box.hi[k]+margin,required.hi[k]+margin);
            if (window.hi[k]-window.lo[k]<.65*span) reduced=true;
        }
        if (!reduced) return full;
        // Cutter faces beyond the input box must stay clear of coplanar trims.
        for (int k=0;k<3;++k) {
            const double guard=std::max(100*Precision::Confusion(),1e-5*(envelope.box.hi[k]-envelope.box.lo[k]));
            if (window.lo[k]<=envelope.box.lo[k]+guard) window.lo[k]=envelope.box.lo[k]-guard;
            if (window.hi[k]>=envelope.box.hi[k]-guard) window.hi[k]=envelope.box.hi[k]+guard;
        }
        auto piece=full;
        try {
            const auto box=BRepPrimAPI_MakeBox(gp_Pnt(window.lo[0],window.lo[1],window.lo[2]),
                                               gp_Pnt(window.hi[0],window.hi[1],window.hi[2])).Shape();
            TopTools_ListOfShape args,tools;
            args.Append(full.shape); tools.Append(box);
            BRepAlgoAPI_Common op;
            op.SetArguments(args); op.SetTools(tools); op.SetNonDestructive(true); op.SetUseOBB(true);
            op.Build();
            if (op.IsDone() && !op.HasErrors() && !op.HasWarnings() &&
                TopExp_Explorer(op.Shape(),TopAbs_SOLID).More() && BRepCheck_Analyzer(op.Shape()).IsValid())
                piece={op.Shape(),boundary_index(op.Shape())};
        } catch (const Standard_Failure&) {
            // A failed optional crop uses the original topology in this window.
        } catch (const dynamics::UnresolvedContact&) {
        }
        clip=Clip{window,piece};
        return piece;
    }
private:
    struct Clip { Box bounds; LocalSolidGeometry piece; };
    std::optional<Clip> clip;
};
}

class LocalSolidSearch::Impl {
public:
    Impl(const TopoDS_Shape& a,const TopoDS_Shape& b) : a(a),b(b) {}
    LocalBody a,b;
    std::mutex mutex;
};
LocalSolidSearch::LocalSolidSearch(const TopoDS_Shape& a,const TopoDS_Shape& b)
    : impl_(std::make_shared<Impl>(a,b)) {}
LocalSolidPair LocalSolidSearch::select(const gp_Trsf& a_in_b,double reach) const {
    const auto ar=interaction(impl_->a.envelope,impl_->b.envelope,a_in_b.Inverted(),reach);
    if (ar.empty()) return {{},{},true};
    const auto br=interaction(impl_->b.envelope,impl_->a.envelope,a_in_b,reach);
    if (br.empty()) return {{},{},true};
    std::lock_guard lock(impl_->mutex);
    return {impl_->a.select(ar,reach),impl_->b.select(br,reach),false};
}

double LocalSolidSearch::motion_bound(const RigidPath& a,const RigidPath& b,double global_speed,double cap) const {
    const auto a_in_b=b.start.Inverted()*a.start;
    // Any pair closer than cap during the step was within cap + global_speed
    // at its start. Its local material points therefore lie in these boxes.
    const auto ar=interaction(impl_->a.envelope,impl_->b.envelope,a_in_b.Inverted(),cap+global_speed);
    const auto br=interaction(impl_->b.envelope,impl_->a.envelope,a_in_b,cap+global_speed);
    if (ar.empty() || br.empty()) return global_speed;
    const auto wa=a.axis.Transformed(a.start)*a.angle,wb=b.axis.Transformed(b.start)*b.angle;
    const auto translation=gp_Vec(a.from,a.to)-gp_Vec(b.from,b.to);
    const auto velocity=[&](const gp_Pnt& p,const gp_Pnt& pa,const gp_Pnt& pb) {
        return (translation+wa.Crossed(gp_Vec(pa,p))-wb.Crossed(gp_Vec(pb,p))).Magnitude();
    };
    // At closest points x and y, (x-y).(omega x (x-y)) is zero.
    // The normal distance rate is thus bounded by relative rigid velocity
    // evaluated at either point. Enclose both swept boxes by corner chords
    // plus their rotation sagitta; pivot translations are linear.
    const auto bound=[&](const Box& box,const RigidPath& path) {
        double fastest=0.,radius=0.;
        for (unsigned i=0;i<8;++i) {
            const gp_Pnt p(i&1 ? box.hi[0] : box.lo[0],i&2 ? box.hi[1] : box.lo[1],
                           i&4 ? box.hi[2] : box.lo[2]);
            radius=std::max(radius,p.Distance(path.pivot));
            fastest=std::max({fastest,velocity(p.Transformed(path.start),a.from,b.from),
                                      velocity(p.Transformed(path.end),a.to,b.to)});
        }
        return fastest+(wa-wb).Magnitude()*radius*path.angle*path.angle/8.;
    };
    return std::min(global_speed,std::min(bound(ar,a),bound(br,b))+Precision::Confusion());
}
}
