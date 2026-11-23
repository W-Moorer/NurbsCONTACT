#include "nurbscontact/tdib.hpp"

#include <iostream>

namespace {
using namespace nc;
void require(bool condition,const char* msg) { if (!condition) throw std::runtime_error(msg); }

MovingBezier patch(double start,double end,double offset=0.) {
    MovingBezier p{2,2,{},{}};
    for (double u:{0.,1.}) for (double v:{0.,1.}) {
        const double weight=1.+u+2*v;
        p.first.push_back(homogeneous(exact({offset+u,v,start}),weight));
        p.last.push_back(homogeneous(exact({offset+u,v,end}),weight));
    }
    return p;
}
}
int main() {
    try {
        const auto a=patch(0,0,1e8), b=patch(3,-1,1e8);
        const auto hit=tdib_envelopes(a,b,.02,1.,0.,10000);
        require(hit.status==TdibStatus::candidate && hit.time.contains(.75),"missed interior rational patch event");
        require(tdib_envelopes(a,b,.02,.749,0.,10000).status==TdibStatus::clear,"horizon before entry");
        require(tdib_envelopes(a,b,.02,.75,0.,10000).status==TdibStatus::candidate,"closed right endpoint lost");
        const auto near=patch(1.,1e-10,1e8);
        require(tdib_envelopes(a,near,.02,1.,0.,10000).status==TdibStatus::clear,"small separation lost");
        require(tdib_envelopes(a,near,.02,1.,2e-10,10000).status==TdibStatus::candidate,"margin ignored");
        require(tdib_envelopes(a,b,.02,1.,0.,1).status==TdibStatus::unresolved,"budget reported separation");
        auto triangle=[](double z) {
            std::vector<HPoint> points;
            for (Vec q:{Vec{0,0,z},Vec{.5,0,z},Vec{1,0,z},Vec{0,.5,z},Vec{.5,.5,z},Vec{0,1,z}})
                points.push_back(homogeneous(exact(q)));
            return points;
        };
        MovingBezier t1{3,3,triangle(0),triangle(0),BezierLayout::quadratic_triangle};
        MovingBezier t2{3,3,triangle(1),triangle(0),BezierLayout::quadratic_triangle};
        const auto touch=tdib_envelopes(t1,t2,.02,1.,0.,10000);
        require(touch.status==TdibStatus::candidate && touch.time.contains(1.),"triangular endpoint contact missed");
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
