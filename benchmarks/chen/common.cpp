#include "nurbscontact/query.hpp"

#include "config.h"
#include "recBezier.h"
#include "solverTD.h"

#include <chrono>
#include <iomanip>
#include <string_view>

namespace {
using Time = std::chrono::steady_clock;
using Solver = SolverTD<RecQuadBezier,RecQuadBezier,RecParamBound,RecParamBound>;

struct Case {
    const char* name;
    double slope, curvature, speed;
    bool event;
};

RecQuadBezier patch(double height, double slope, double curvature) {
    RecQuadBezier result;
    for (std::size_t i=0;i<3;++i) {
        for (std::size_t j=0;j<3;++j) {
            const double u=.5*static_cast<double>(i), v=.5*static_cast<double>(j);
            const double uu=i==1 ? -.25 : .25;
            const double vv=j==1 ? -.25 : .25;
            result.ctrlp[3*i+j]={u,v,height+slope*u+curvature*(uu+vv)};
        }
    }
    return result;
}

nc::PreparedModel prepared(const RecQuadBezier& patch) {
    const nc::Rect chart{{0,1},{0,1}};
    nc::BezierSurface surface{3,3,{},chart};
    for (const auto& p:patch.ctrlp) surface.poles.push_back(nc::homogeneous({p.x(),p.y(),p.z()}));
    return nc::PreparedModel({{{std::move(surface),chart,0,0}},{{}},0});
}
}

int main(int argc,char** argv) {
    const bool positive=argc==2 && std::string_view(argv[1])=="--positive-only";
    if (argc>1 && !positive) return 2;
    const double clearance=positive ? .025 : 0.;
    std::cout<<std::setprecision(17)
             <<"case,method,repeat,analytic_entry,outcome,lower_or_candidate,witness_time,point_residual,query_us\n";
    const auto a=patch(0,0,0);
    RecQuadBezier stationary;
    for (auto& p:stationary.ctrlp) p.setZero();
    const auto native_a=prepared(a);
    const nc::Options options{.0005,.0005,25'000};
    const double missing=std::numeric_limits<double>::quiet_NaN();
    for (const auto& c:{Case{"planar-entry",.25,0,1,true},
                         Case{"curved-entry",0,.5,1,true},
                         Case{"curved-clear",0,.5,.125,false}}) {
        const auto b=patch(.375,c.slope,c.curvature);
        auto velocity=stationary;
        for (auto& p:velocity.ctrlp) p.z()=-c.speed;
        const auto native_b=prepared(b);
        const nc::Motion motion{{},{0,0,-c.speed}};
        const double reference=c.event ? (.375-clearance)/c.speed : missing;
        const auto chen=[&](int repeat) {
            Array2d uv_a,uv_b;
            const auto start=Time::now();
            const double t=Solver::solveCCD(a,stationary,b,velocity,uv_a,uv_b,BoundingBoxType::OBB,.0005,1);
            const auto us=std::chrono::duration<double,std::micro>(Time::now()-start).count();
            if (repeat<0) return;
            const double residual=t<0 ? missing :
                (a.evaluatePatchPoint(uv_a)-b.evaluatePatchPoint(uv_b)-t*velocity.evaluatePatchPoint(uv_b)).norm();
            std::cout<<c.name<<",chen-obb,"<<repeat<<','<<reference<<','<<(t<0 ? "clear" : "candidate")
                     <<','<<(t<0 ? missing : t)<<','<<missing<<','<<residual<<','<<us<<'\n';
        };
        const auto native=[&](int repeat) {
            const auto start=Time::now();
            const auto r=nc::sweep(native_a,native_b,motion,{0,1},clearance,options);
            const auto us=std::chrono::duration<double,std::micro>(Time::now()-start).count();
            if (repeat<0) return;
            const char* status=r.status==nc::SweepStatus::clear ? "clear" :
                r.status==nc::SweepStatus::hit ? "hit" : "unresolved";
            const double residual=r.witness ? nc::norm(r.witness->a.position-r.witness->b.position).hi : missing;
            std::cout<<c.name<<(positive ? ",native-positive," : ",native,")<<repeat<<','<<reference
                     <<','<<status<<','<<r.safe_prefix
                     <<','<<r.witness_time.value_or(missing)<<','<<residual<<','<<us<<'\n';
        };
        if (!positive) chen(-1);
        native(-1);
        for (int i=0;i<9;++i) {
            if (positive) {
                native(i);
            } else if (i%2) {
                native(i);
                chen(i);
            } else {
                chen(i);
                native(i);
            }
        }
    }
}
