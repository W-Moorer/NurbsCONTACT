#include "cad_rigid_path.hpp"
#include "chen_patches.hpp"
#include "nurbscontact/tdib.hpp"
#include <BRepTools.hxx>
#include <BRep_Tool.hxx>
#include <GeomConvert.hxx>
#include <GeomConvert_BSplineSurfaceToBezierSurface.hxx>
#include <Geom_RectangularTrimmedSurface.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Face.hxx>

namespace nc::cad {
std::vector<Handle(Geom_BezierSurface)> bezier_charts(const TopoDS_Shape& shape) {
    std::vector<Handle(Geom_BezierSurface)> result;
    for (TopExp_Explorer it(shape,TopAbs_FACE);it.More();it.Next()) {
        const auto face=TopoDS::Face(it.Current());
        double u0,u1,v0,v1;
        BRepTools::UVBounds(face,u0,u1,v0,v1);
        try {
            Handle(Geom_Surface) bounded=new Geom_RectangularTrimmedSurface(BRep_Tool::Surface(face),u0,u1,v0,v1);
            GeomConvert_BSplineSurfaceToBezierSurface pieces(GeomConvert::SurfaceToBSplineSurface(bounded));
            for (int u=1;u<=pieces.NbUPatches();++u) for (int v=1;v<=pieces.NbVPatches();++v)
                result.push_back(pieces.Patch(u,v));
        } catch (const Standard_Failure&) {
            // The ordinary trimmed-solid search still includes this face.
        }
    }
    return result;
}

namespace {
struct SweptChart {
    MovingBezier moving;
    IVec box{};
    double margin{};
};

IVec placed(const gp_Trsf& transform,IVec point) {
    IVec result{};
    for (int k=0;k<3;++k) {
        const auto row=exact({transform.Value(k+1,1),transform.Value(k+1,2),transform.Value(k+1,3)});
        result[k]=dot(row,point)+transform.Value(k+1,4);
    }
    return result;
}

std::vector<SweptChart> place(const std::vector<Handle(Geom_BezierSurface)>& charts,const RigidPath& path) {
    std::vector<SweptChart> result;
    for (const auto& chart:charts) {
        SweptChart piece;
        auto& net=piece.moving;
        net.nu=chart->NbUPoles(); net.nv=chart->NbVPoles();
        double radius=0.;
        bool usable=true;
        for (int i=1;i<=chart->NbUPoles();++i) for (int j=1;j<=chart->NbVPoles();++j) {
            const auto p=exact(coordinates(chart->Pole(i,j)));
            const double w=chart->Weight(i,j);
            if (!(w>0 && std::isfinite(w))) usable=false;
            const auto a=placed(path.start,p),b=placed(path.end,p);
            const auto bounds=hull(a,b);
            piece.box=net.first.empty() ? bounds : hull(piece.box,bounds);
            net.first.push_back(homogeneous(a,w)); net.last.push_back(homogeneous(b,w));
            radius=std::max(radius,norm(p-exact(coordinates(path.pivot))).hi);
        }
        if (!usable) continue;
        // The chord error of a rotating pole is at most r * angle^2 / 8.
        piece.margin=(Interval{radius}*square(path.angle)*.125).hi;
        for (auto& coordinate:piece.box) coordinate=coordinate+Interval{-piece.margin,piece.margin};
        result.push_back(std::move(piece));
    }
    return result;
}
}

CandidateQuery chen_2026_candidates(const TopoDS_Shape& one,const TopoDS_Shape& two) {
    return [a=bezier_charts(one),b=bezier_charts(two)](const RigidPath& pa,const RigidPath& pb,
                                                      dynamics::SurfaceTransition& stats) {
        const auto first=place(a,pa),second=place(b,pb);
        std::optional<Interval> earliest;
        std::size_t budget=4096, comparisons=0;
        for (const auto& p:first) for (const auto& q:second) {
            if (++comparisons>131072 || !budget || (earliest && earliest->hi==0.)) return earliest;
            if (disjoint(p.box[0],q.box[0]) || disjoint(p.box[1],q.box[1]) || disjoint(p.box[2],q.box[2])) continue;
            ++stats.chen_queries;
            const auto hit=tdib_envelopes(p.moving,q.moving,.01,earliest ? earliest->hi : 1.,
                                          (Interval{p.margin}+q.margin).hi,std::min<std::size_t>(budget,1024));
            budget-=hit.nodes;
            stats.nodes+=hit.nodes;
            stats.tdib_cells+=hit.nodes;
            if (hit.status==TdibStatus::unresolved) ++stats.tdib_unresolved;
            if (hit.status==TdibStatus::candidate) {
                ++stats.tdib_candidates;
                if (!earliest || hit.time.lo<earliest->lo ||
                    (hit.time.lo==earliest->lo && hit.time.hi<earliest->hi)) earliest=hit.time;
            }
        }
        return earliest;
    };
}
}
