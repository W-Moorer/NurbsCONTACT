#include "nurbscontact/hht.hpp"
#include "nurbscontact/contact_dynamics.hpp"
#include "nurbscontact/free_rigid.hpp"
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

void check_shared_pair_stiffness() {
    struct FixedPatch final : PairRegionQuery {
        std::vector<ContactRegion> points;
        std::vector<ContactRegion> regions(ContactPose,ContactPose,ContactWork&) const override {return points;}
        SurfaceTransition transition(ContactPose,ContactPose,ContactPose,ContactPose,double) const override {return {};}
    };
    const auto geometry=std::make_shared<FixedPatch>();
    const std::array<ContactBodyState,2> bodies{{
        {nullptr,{},{{},{}}},
        {nullptr,{},{{{0.,1.,0.},{}},{{1.,0.,0.},{}}}}
    }};
    ContactPair pair{0,1,{},{},ContactDirection::world,{},{100.,1.,2.,.25,.01,1.}};
    pair.regions=geometry;
    const auto evaluate=[&](bool shared) {
        pair.share_penalty_stiffness=shared;
        Forces force{Vector(2),Vector(4),Vector(4)};
        ContactWork work;
        std::vector<ContactSample> samples;
        require(add_contact_forces(&force,bodies,std::span(&pair,1),{.5,.2},work,&samples),
                "pair penalty assembly failed");
        return std::pair{force,samples.at(0)};
    };
    // The third penetrating point has a separating damping force exceeding its elastic load.
    geometry->points={{{},{0,1,0},.1},{{},{0,1,0},.2},{{},{0,1,0},.001}};
    const auto [sum,old_sample]=evaluate(false);
    const auto [shared,sample]=evaluate(true);
    require(std::abs(sample.normal_force-14.)<1e-12 && std::abs(old_sample.normal_force-28.)<1e-12,
            "inactive point consumed part of the pair stiffness");
    require(length(sub(scale(old_sample.friction,.5),sample.friction))<1e-12,"shared friction output");
    for (std::size_t i=0;i<2;++i)
        require(std::abs(shared.value[i]-.5*sum.value[i])<1e-12,"shared force did not enter the residual");
    for (std::size_t i=0;i<4;++i)
        require(std::abs(shared.stiffness[i]-.5*sum.stiffness[i])<1e-12 &&
                std::abs(shared.damping[i]-.5*sum.damping[i])<1e-12,"inconsistent shared penalty tangent");
}

void check_spatial_wrench() {
    FreeRigidBody body(0,{2.,3.,4.},{.2,-.15,.1});
    ContactPose reference{{.3,.7,-.2},{0,.6,.8},.72};
    const Vec omega{.8,-1.2,2.},com{.3,-.2,.1},local_point{.9,.2,-.1},load{3.,2.,-5.};
    const auto attitude=body.initial_attitude(reference),rate=body.initial_rate(attitude,omega);
    const Vec center=reference.point(com);
    State state{};
    state.q={center[0],center[1],center[2]};
    state.v={0.,0.,0.};
    for (std::size_t i=0;i<4;++i) {
        state.q.push_back(1.7*attitude[i]);
        state.v.push_back(1.7*(rate[i]+.04*attitude[i]));
    }
    ContactBodyState placed{nullptr,{},std::vector<Twist>(7)};
    body.kinematics(state.q,com,placed);
    require(length(sub(reference.point(local_point),placed.pose.point(local_point)))<1e-13,
            "free-body placement lost the local mass offset");
    Forces forces{Vector(7),Vector(49),Vector(49)};
    for (std::size_t k=0;k<7;++k)
        forces.value[k]=dot3(load,placed.derivative[k].at(placed.pose.rotate(local_point)));
    body.apply_euler_equations(state.q,state.v,forces);
    state.a=state.force=forces.value;
    const auto alpha=body.body_angular_acceleration(state);
    const auto inertia=[&](Vec x) -> Vec {
        return {dot3(body.inertia[0],x),dot3(body.inertia[1],x),dot3(body.inertia[2],x)};
    };
    const auto torque=cross3(sub(local_point,com),reference.inverse_rotate(load));
    require(length(sub(add(inertia(alpha),cross3(omega,inertia(omega))),torque))<1e-12,
            "off-center point load violates body Euler equations");
    body.normalize(state);
    require(length(sub(body.body_angular_velocity(state.q,state.v),omega))<1e-13 &&
            length(sub(body.body_angular_acceleration(state),alpha))<1e-12,
            "normalization changed physical angular velocity or acceleration");
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

void check_bounded_force_decisions() {
    const auto force = [](double x) { return -100 * x; };
    std::size_t coarse_calls = 0;
    HhtOptions options;
    options.max_subdivisions = 0;
    Hht integrator({1}, {}, options, {}, {},
        [&](const Vector& q, const Vector&, double, const Vector&, double target) {
            coarse_calls += target > 0;
            const double bound = target / 2;
            const double nominal = force(q[0]);
            const double estimate = nominal + bound;
            return Forces{{estimate}, {100}, {0}, {}, up(std::abs(estimate - nominal))};
        });
    State state = integrator.initialize({1}, {0});
    HhtStats stats;
    for (int i = 0; i < 30; ++i) {
        const double previous = force(state.q[0]);
        integrator.step(state, .005, stats);
        const double next = force(state.q[0]);
        const double residual = state.a[0] - (1 + options.alpha) * next + options.alpha * previous;
        const double limit = options.absolute_tolerance + options.relative_tolerance *
            std::max(std::abs(previous), std::abs(next));
        require(std::abs(residual) <= limit && std::abs(residual) <= stats.maximum_residual,
                "bounded HHT accepted a state beyond the true residual tolerance");
    }
    require(coarse_calls > 0 && stats.force_refinements > 0 && stats.accepted_steps == 30,
            "bounded HHT did not exercise its coarse and refined decisions");

    Hht unresolved({1}, {}, options, {}, {},
        [](const Vector&, const Vector&, double time, const Vector&, double) {
            // The nominal force is 1 after the initial state. A zero estimate
            // cannot certify convergence while this uncertainty persists.
            return Forces{{0}, {0}, {0}, {}, time > 0 ? 1.0 : 0.0};
        });
    State old = unresolved.initialize({0}, {0});
    HhtStats rejected;
    bool failed = false;
    try {
        unresolved.step(old, .005, rejected);
    } catch (const std::runtime_error&) {
        failed = true;
    }
    require(failed && old.time == 0 && rejected.accepted_steps == 0,
            "unresolved force interval was mistaken for a converged step");
}

void check_rollback() {
    HhtOptions options;
    options.minimum_clearance = 0.8;
    options.max_subdivisions = 2;
    Hht integrator({1},{},options,[](const Vector& a,const Vector& b) {return std::min(a[0],b[0]);},
                   [](const Vector&,const Vector&,double,const Vector& history) {
        return Forces{{0},{0},{0},{history.empty() ? 0. : history[0]+1}};
    });
    State state = integrator.initialize({1},{-1});
    HhtStats stats;
    bool failed=false;
    try { integrator.step(state,.5,stats); }
    catch (const std::runtime_error&) { failed=true; }
    require(failed && state.time==0 && state.q[0]==1 && state.history==Vector{0} && stats.accepted_steps==0,
            "failed HHT step must roll back motion and internal history");
    integrator.step(state,.1,stats);
    require(state.time==.1 && std::abs(state.q[0]-.9)<1e-15 && stats.minimum_clearance>.8,
            "HHT could not continue after rejected step");
    require(state.history==Vector{1}, "only an accepted HHT evaluation may update history");

    Hht event_stepper({1},[](const Vector&,const Vector&,double) {return Forces{{0},{0},{0}};},
        {},{},{},{},[](const State& a,const State& b) -> std::optional<double> {
            return b.time-a.time <= .1875 ? std::nullopt : std::optional<double>{.75};
        });
    auto split=event_stepper.initialize({0},{1});
    HhtStats checks;
    event_stepper.step(split,.25,checks);
    require(split.q[0]==.25 && checks.accepted_steps==2 && checks.step_checks==3 && checks.rejected_steps==1,
            "step acceptance callback did not subdivide a converged candidate");
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

void check_translational_bushing() {
    const TranslationalBushing element{0,1,{},{},{1,0,0},{20,40,0},{3,7,0}};
    const auto evaluate = [&](double angle, double vx, double omega) {
        const std::vector<ContactBodyState> bodies{
            {nullptr,{{},{0,0,1},angle},{{},{ {},{0,0,1} }}},
            {nullptr,{{1,.2,0}},{{{1,0,0},{}},{}}}
        };
        Forces f{{0,0},Vector(4),Vector(4)};
        add_translational_bushings(f,bodies,{&element,1},{vx,omega});
        return f;
    };
    const auto at_rest=evaluate(.3,0,0);
    const double expected_torque=-(20*(std::cos(.3)+.2*std::sin(.3)-1)*(.2*std::cos(.3)-std::sin(.3))
        -40*(.2*std::cos(.3)-std::sin(.3))*(std::cos(.3)+.2*std::sin(.3)));
    require(std::abs(at_rest.value[1]-expected_torque)<1e-12,
            "rotating bushing did not transport the separation into its own frame");
    const auto slow=evaluate(.3,1-1e-6,2);
    const auto fast=evaluate(.3,1+1e-6,2);
    for (std::size_t i=0;i<2;++i)
        require(std::abs(at_rest.damping[2*i]+(fast.value[i]-slow.value[i])/2e-6)<1e-7,
                "bushing damping tangent disagrees with virtual work");
}

void check_triangle_bvh() {
    const Triangle plate{{{0,0,0},{2,0,0},{0,2,0}}};
    const Triangle through{{{0.5,0.5,-1},{0.5,0.5,1},{1,1,1}}};
    const TriangleBvh one({plate}),two({through});
    TriangleStatistics counts;
    const auto hit=one.closest_within(two,{},0,nullptr,counts);
    require(hit && hit->distance==0,"zero cutoff missed a triangle crossing");
    const Triangle interior{{{.25,.25,1},{.5,.25,1},{.25,.5,1}}};
    const TriangleBvh insert({interior});
    TriangleHint swept_hint;
    require(insert.swept_within(one, {}, {0,0,2}, 0, swept_hint, counts) &&
            !insert.swept_within(one, {}, {0,0,.8}, .19, swept_hint, counts) &&
            insert.swept_within(one, {}, {0,0,.8}, .21, swept_hint, counts),
            "swept triangle missed interior containment or used the wrong clearance");
    require(one.swept_within(one, {-4,0,0}, {4,0,0}, 0, swept_hint, counts),
            "coplanar swept triangle missed a pass-through");
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
        const ContactPose tilted{motion.origin,{0,.6,.8},angle};
        RigidPlacement rigid;
        rigid.origin=tilted.origin;
        for (auto& column : rigid.basis) column=tilted.rotate(column);
        const auto rotated=ma.closest_rigid_within(mb,rigid,infinity,&hint,counts);
        double rotated_oracle=infinity;
        for (const Triangle& ta : a) {
            for (const Triangle& tb : b) {
                const Triangle moved{tilted.point(tb[0]),tilted.point(tb[1]),tilted.point(tb[2])};
                rotated_oracle=std::min(rotated_oracle,closest_triangles(ta,moved).distance);
            }
        }
        require(rotated && std::abs(rotated->distance-rotated_oracle)<1e-12,
                "3D BVH placement lost the closest pair");
    }
}

}  // namespace

int main() {
    try {
        check_oscillator();
        check_shared_pair_stiffness();
        check_spatial_wrench();
        check_bounded_force_decisions();
        check_rollback();
        check_point_resistance();
        check_translational_bushing();
        check_triangle_bvh();
        std::cout<<"HHT and triangle BVH tests passed\n";
    } catch (const std::exception& e) {
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
