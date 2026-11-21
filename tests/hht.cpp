#include "nurbscontact/hht.hpp"
#include "nurbscontact/contact_dynamics.hpp"
#include "nurbscontact/triangle_bvh.hpp"

#include <iostream>
#include <random>
#include <stdexcept>

namespace {

using namespace nc;
using namespace nc::dynamics;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check_oscillator() {
    Hht integrator({1},[](const Vector& q,const Vector&,double) {
        return Forces{{-100*q[0]},{100},{0}};
    });
    double previous = infinity;
    for (double h : {0.02,0.01,0.005,0.0025}) {
        State state = integrator.initialize({1},{0});
        HhtStats stats;
        const int steps = static_cast<int>(std::round(1.2/h));
        double squared_error = 0;
        for (int i=0;i<steps;++i) {
            integrator.step(state,h,stats);
            const double x = state.q[0]-std::cos(10*state.time);
            squared_error += x*x;
        }
        const double rms = std::sqrt(squared_error/steps);
        require(rms*3.6<previous,"HHT oscillator lost second-order convergence");
        require(stats.maximum_residual<1e-7 && stats.accepted_steps==steps, "HHT oscillator residual");
        previous = rms;
    }
}

void check_rollback() {
    HhtOptions options;
    options.minimum_clearance = 0.8;
    options.max_subdivisions = 2;
    Hht integrator({1},[](const Vector&,const Vector&,double) {
        return Forces{{0},{0},{0}};
    }, options, [](const Vector& a,const Vector& b) {return std::min(a[0],b[0]);});
    State state = integrator.initialize({1},{-1});
    HhtStats stats;
    bool failed=false;
    try { integrator.step(state,.5,stats); }
    catch (const std::runtime_error&) { failed=true; }
    require(failed && state.time==0 && state.q[0]==1 && stats.accepted_steps==0, "failed HHT step must roll back");
    integrator.step(state,.1,stats);
    require(state.time==.1 && std::abs(state.q[0]-.9)<1e-15 && stats.minimum_clearance>.8,
            "HHT could not continue after rejected step");
}

void check_point_resistance() {
    const auto reference = ContactPose{};
    const std::vector<ContactBodyState> bodies{
        {nullptr,reference,{{{1,0,0},{}}}},
        {nullptr,reference,{{{0,0,7},{}}},{{0,1,0},{}}}
    };
    const PointResistance plate{0,1,{},{},{0,0,1},3,.1};
    const auto force = [&](double speed) {
        Forces output{{0},{0},{0}};
        add_point_resistances(output,bodies,{&plate,1},{speed});
        return output;
    };
    const auto value=force(2);
    require(std::abs(value.value[0]+6/std::sqrt(5.01))<1e-12,
            "sliding resistance did not project the relative planar velocity");
    const double slope=(force(2+1e-5).value[0]-force(2-1e-5).value[0])/2e-5;
    require(std::abs(value.damping[0]+slope)<1e-8,
            "sliding resistance tangent disagrees with virtual work");
}

void check_triangle_bvh() {
    const Triangle plate{{{0,0,0},{2,0,0},{0,2,0}}};
    const Triangle through{{{0.5,0.5,-1},{0.5,0.5,1},{1,1,1}}};
    const TriangleBvh one({plate}),two({through});
    TriangleStatistics counts;
    const auto hit=one.closest_within(two,{},0,nullptr,counts);
    require(hit && hit->distance==0,"zero cutoff missed a triangle crossing");
    const Triangle raised{{{0,0,1},{0,2,1},{2,0,1}}};
    const TriangleBvh ray_model({plate,raised},1);
    Vec raised_normal{},plate_normal{};
    const auto ray=ray_model.ray_first({.5,.5,2},{0,0,-1},3,counts,&raised_normal);
    const auto back=ray_model.ray_first({.5,.5,-1},{0,0,1},3,counts,&plate_normal);
    require(ray && *ray==1 && back && *back==1,"triangle ray skipped the closest two-sided hit");
    require(raised_normal==Vec{0,0,-4} && plate_normal==Vec{0,0,4},"triangle ray returned the wrong face normal");
    require(!ray_model.ray_first({.5,.5,2},{0,0,-1},.5,counts),"triangle ray exceeded its span");
    require(!ray_model.ray_first({2,2,2},{0,0,-1},3,counts),"triangle ray hit outside its triangles");
    std::mt19937 generator(1704);
    std::uniform_real_distribution<double> coordinate(-1,1);
    std::vector<Triangle> a,b;
    for (auto* geometry : {&a,&b}) {
        for (int i=0;i<18;++i) {
            Triangle triangle;
            for (Vec& p : triangle) p={coordinate(generator),coordinate(generator),coordinate(generator)};
            geometry->push_back(triangle);
        }
    }
    const TriangleBvh ma(a,2),mb(b,3);
    TriangleHint hint;
    for (int s=0;s<16;++s) {
        const double angle=s*.137;
        const Pose motion{{2.3+s*.02,0.23,-0.47},std::cos(angle),std::sin(angle)};
        auto answer=ma.closest_within(mb,motion,infinity,&hint,counts);
        auto cold=ma.closest_within(mb,motion,infinity,nullptr,counts);
        double oracle=infinity;
        for (const Triangle& ta : a) {
            for (const Triangle& tb : b) {
                const Triangle moved{motion.point(tb[0]),motion.point(tb[1]),motion.point(tb[2])};
                oracle=std::min(oracle,closest_triangles(ta,moved).distance);
            }
        }
        require(answer && cold && std::abs(answer->distance-oracle)<1e-12 &&
                std::abs(cold->distance-oracle)<1e-12,"BVH skipped the brute-force closest triangle pair");
        require(!ma.closest_within(mb,motion,oracle*.9,&hint,counts),"BVH exceeded the requested cutoff");
    }
}

}  // namespace

int main() {
    try {
        check_oscillator();
        check_rollback();
        check_point_resistance();
        check_triangle_bvh();
        std::cout<<"HHT and triangle BVH tests passed\n";
    } catch (const std::exception& e) {
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
