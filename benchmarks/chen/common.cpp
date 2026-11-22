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

template<class Patch>
nc::PreparedModel prepared(const Patch& patch, std::size_t side) {
    const nc::Rect chart{{0,1},{0,1}};
    nc::BezierSurface surface{side,side,{},chart};
    for (const auto& p:patch.ctrlp) surface.poles.push_back(nc::homogeneous({p.x(),p.y(),p.z()}));
    return nc::PreparedModel({{{std::move(surface),chart,0,0}},{{}},0});
}

void published_single() {
    // The original scene/scene_random.h singleTest. Both control nets are
    // affine grids. Keep both upstream velocities for the official query.
    RecCubicBezier a, b, va, vb;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const auto k = 4*i + j;
            a.ctrlp[k] = {double(j), double(i), 0};
            b.ctrlp[k] = {-1, double(i), double(j)};
            va.ctrlp[k] = {-.7, 0, 0};
            vb.ctrlp[k] = {1, 0, 0};
        }
    }
    const auto native_a = prepared(a, 4);
    const auto native_b = prepared(b, 4);
    const nc::Motion relative{{}, {1.7, 0, 0}};
    const nc::Options options{.0005, .0005, 25'000};
    using PublishedSolver = SolverTD<RecCubicBezier, RecCubicBezier, RecParamBound, RecParamBound>;
    using Duration = std::chrono::duration<double, std::micro>;
    std::cout << std::setprecision(17)
              << "method,repeat,outcome,lower_or_candidate,witness_time,point_residual,query_us\n";
    const auto run = [&](bool native, int repeat) {
        const auto start = Time::now();
        if (native) {
            const auto result = nc::sweep(native_a, native_b, relative, {0,1}, 0, options);
            const auto us = Duration(Time::now()-start).count();
            if (repeat < 0) return;
            const char* status = result.status == nc::SweepStatus::hit ? "hit" :
                                 result.status == nc::SweepStatus::clear ? "clear" : "unresolved";
            std::cout << "native-sweep," << repeat << ',' << status << ',' << result.safe_prefix << ',';
            if (result.witness_time) std::cout << *result.witness_time;
            std::cout << ",," << us << '\n';
        } else {
            Array2d uv_a, uv_b;
            const double time = PublishedSolver::solveCCD(a,va,b,vb,uv_a,uv_b,BoundingBoxType::OBB,.0005,1);
            const auto us = Duration(Time::now()-start).count();
            if (repeat < 0) return;
            std::cout << "chen-obb," << repeat << ',' << (time < 0 ? "clear" : "candidate") << ',';
            if (time >= 0) {
                const auto difference = (a.evaluatePatchPoint(uv_a)+time*va.evaluatePatchPoint(uv_a) -
                                         b.evaluatePatchPoint(uv_b)-time*vb.evaluatePatchPoint(uv_b)).eval();
                std::cout << time << ",," << difference.norm();
            } else {
                std::cout << ",,";
            }
            std::cout << ',' << us << '\n';
        }
    };
    run(true, -1);
    run(false, -1);
    for (int repeat = 0; repeat < 5; ++repeat) {
        run(repeat % 2 == 0, repeat);
        run(repeat % 2 != 0, repeat);
    }
}
}

int main(int argc,char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--published-single") {
        published_single();
        return 0;
    }
    const bool positive=argc==2 && std::string_view(argv[1])=="--positive-only";
    if (argc>1 && !positive) return 2;
    const double clearance=positive ? .025 : 0.;
    std::cout<<std::setprecision(17)
             <<"case,method,repeat,analytic_entry,outcome,lower_or_candidate,witness_time,point_residual,query_us\n";
    const auto a=patch(0,0,0);
    RecQuadBezier stationary;
    for (auto& p:stationary.ctrlp) p.setZero();
    const auto native_a=prepared(a, 3);
    const nc::Options options{.0005,.0005,25'000};
    const double missing=std::numeric_limits<double>::quiet_NaN();
    for (const auto& c:{Case{"planar-entry",.25,0,1,true},
                         Case{"curved-entry",0,.5,1,true},
                         Case{"curved-clear",0,.5,.125,false}}) {
        const auto b=patch(.375,c.slope,c.curvature);
        auto velocity=stationary;
        for (auto& p:velocity.ctrlp) p.z()=-c.speed;
        const auto native_b=prepared(b, 3);
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
