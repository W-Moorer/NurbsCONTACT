#include "nurbscontact/surface_impulse.hpp"

#include <cmath>
#include <iostream>
#include <string>

using namespace nc;
using namespace nc::dynamics;

namespace {
class Region final : public PairRegionQuery {
public:
    explicit Region(double depth) : depth_(depth) {}
    std::vector<ContactRegion> regions(ContactPose,ContactPose,ContactWork&) const override {
        return {{{},{0.,0.,1.},depth_}};
    }
    SurfaceTransition transition(ContactPose,ContactPose,ContactPose,ContactPose,double) const override {return {};}
private:
    double depth_;
};

void near(double a,double b,double tol=3e-8) {
    if (!(std::abs(a-b)<=tol)) throw std::runtime_error("expected "+std::to_string(b)+", got "+std::to_string(a));
}
ContactPair pair(unsigned a,unsigned b,double depth,double mu) {
    ContactPair p{};
    p.a=a; p.b=b;
    p.material={5000,1,0,mu,.001,.02};
    p.regions=std::make_shared<Region>(depth);
    return p;
}
SurfaceImpulseResult solve(ImpulseLaw law,const std::vector<ContactBodyState>& bodies,
                           const std::vector<ContactPair>& pairs,Vector v) {
    ContactWork work;
    return surface_impulses(law,bodies,pairs,v,.01,[](const Vector& x){return x;},work);
}
void coupled(ImpulseLaw law) {
    std::vector<ContactBodyState> bodies(3);
    for (std::size_t k=0;k<3;++k) {
        bodies[k].derivative.resize(3);
        bodies[k].derivative[k].linear[2]=1.;
    }
    const auto s=solve(law,bodies,{pair(0,1,.001,0),pair(1,2,.002,0)},Vector(3));
    // Shared middle mass gives the off-diagonal -1 in the contact system.
    const double diagonal=law==ImpulseLaw::sap ? 4. : 2.;
    const double p=(.1*diagonal+.2)/(diagonal*diagonal-1);
    const double q=(.2*diagonal+.1)/(diagonal*diagonal-1);
    near(s.contacts[0].normal_force*.01,p);
    near(s.contacts[1].normal_force*.01,q);
    near(s.velocity_change[0],-p);
    near(s.velocity_change[1],p-q);
    near(s.velocity_change[2],q);
}
void friction(ImpulseLaw law) {
    std::vector<ContactBodyState> bodies(2);
    for (auto& b:bodies) b.derivative.resize(3);
    for (std::size_t i=0;i<3;++i) bodies[1].derivative[i].linear[i]=1;
    const auto s=solve(law,bodies,{pair(0,1,.001,.5)},{1,2,0});
    const double speed=std::sqrt(5.);
    const double n=law==ImpulseLaw::sap ? (.1+.5*speed)/(3.+.25*1.001) : .1;
    near(s.velocity_change[2],n);
    if (law==ImpulseLaw::lcp) {
        near(s.velocity_change[0],0); near(s.velocity_change[1],-.5*n);
    } else {
        near(s.velocity_change[0],-.5*n/speed); near(s.velocity_change[1],-n/speed);
    }
    const auto stick=solve(law,bodies,{pair(0,1,.001,.8)},{.001,.002,0});
    const double denom=law==ImpulseLaw::sap ? 1.001 : 1.;
    near(stick.velocity_change[0],-.001/denom);
    near(stick.velocity_change[1],-.002/denom);
}
}
int main() {
    try {
        for (const auto name:{"sap","lcp","soccp","ncp","lagrange","augmented-lagrange"}) {
            const auto law=parse_impulse_law(name);
            if (impulse_law_name(law)!=name) return 1;
            coupled(law);
            friction(law);
        }
        std::vector<ContactBodyState> prescribed(2);
        prescribed[1].prescribed.linear[0]=1.;
        const auto fixed=solve(ImpulseLaw::sap,prescribed,{pair(0,1,.001,.5)},{});
        near(fixed.contacts[0].normal_force,30.);
        near(fixed.contacts[0].friction[0],-15.);
    } catch (const std::exception& e) {std::cerr<<e.what()<<'\n'; return 1;}
}
