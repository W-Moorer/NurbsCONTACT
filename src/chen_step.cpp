#include "cad_rigid_path.hpp"
#include "chen_patches.hpp"
#include <Bnd_Box.hxx>
#include "config.h"
#include "recRatBezier.h"
#include "solverTD_margin.h"

namespace nc::cad {
namespace {
using Patch=RecCubicRatBezier;
using Solver=SolverTD<Patch,Patch,RecParamBound,RecParamBound>;
struct Net {
    std::array<gp_Pnt,16> points;
    std::array<double,16> weights;
};

std::vector<Net> nets(const TopoDS_Shape& shape) {
    std::vector<Net> result;
    for (const auto& patch:bezier_charts(shape)) {
        try {
            if (patch->UDegree()>3 || patch->VDegree()>3) continue;
            patch->Increase(3,3);
            Net net;
            for (int i=0;i<4;++i) for (int j=0;j<4;++j) {
                net.points[4*i+j]=patch->Pole(i+1,j+1);
                net.weights[4*i+j]=patch->Weight(i+1,j+1);
            }
            if (std::all_of(net.weights.begin(),net.weights.end(),[](double w){return w>0 && std::isfinite(w);}))
                result.push_back(std::move(net));
        } catch (const Standard_Failure&) {
            // Every trimmed face, including unsupported charts, remains in the CAD sweep.
        }
    }
    return result;
}

struct MovingNet {
    Patch start,velocity;
    Bnd_Box swept;
    double margin;
    MovingNet(const Net& net,const RigidPath& path) {
        double radius=0.;
        for (int i=0;i<16;++i) {
            const auto a=net.points[i].Transformed(path.start),b=net.points[i].Transformed(path.end);
            const double w=net.weights[i];
            start.ctrlp[i]={w*a.X(),w*a.Y(),w*a.Z(),w};
            velocity.ctrlp[i]={w*(b.X()-a.X()),w*(b.Y()-a.Y()),w*(b.Z()-a.Z()),w};
            swept.Add(a); swept.Add(b);
            radius=std::max(radius,net.points[i].Distance(path.pivot));
        }
        const double s=std::sin(.25*path.angle);
        margin=2*radius*s*s;
        swept.Enlarge(margin);
    }
};
std::vector<MovingNet> place(const std::vector<Net>& nets,const RigidPath& path) {
    std::vector<MovingNet> result;
    result.reserve(nets.size());
    for (const auto& net:nets) result.emplace_back(net,path);
    return result;
}
}

CandidateQuery chen_candidates(const TopoDS_Shape& a,const TopoDS_Shape& b) {
    return [one=nets(a),two=nets(b)](const RigidPath& a,const RigidPath& b,dynamics::SurfaceTransition& stats) {
        std::optional<double> best;
        const auto first=place(one,a),second=place(two,b);
        const std::size_t node_limit=stats.nodes+1024;
        std::size_t boxes=0;
        for (const auto& p:first) for (const auto& q:second) {
            if (++boxes>131072 || stats.nodes>=node_limit || best==0.) return best;
            if (p.swept.IsOut(q.swept)) continue;
            Array2d u,v;
            ++stats.chen_queries;
            double t;
            try {
                t=Solver::solveCCD(p.start,p.velocity,q.start,q.velocity,u,v,BoundingBoxType::OBB,
                                   .01,best.value_or(1.),p.margin+q.margin,&stats.nodes,node_limit);
            } catch (const std::runtime_error&) {
                return best;
            }
            if (t>=0 && t<=best.value_or(1.)) best=t;
        }
        return best;
    };
}
}
