#include "cpu_mesh_ccd.hpp"

#include <tight_inclusion/ccd.hpp>

namespace nc::cpu {

TriangleTime linear_triangle_ccd(const Triangle& a0, const Triangle& a1,
                                 const Triangle& b0, const Triangle& b1,
                                 double margin, double distance_tolerance) {
    TriangleTime result;
    static_assert(std::is_same_v<ticcd::Scalar,double>, "CPU mesh CCD requires double precision");
    double scale = margin;
    for (const Triangle* tri : {&a0,&a1,&b0,&b1})
        for (const Vec& p : *tri)
            for (double x : p) scale=std::max(scale,std::abs(x));
    const int exponent = scale > 0 ? std::ilogb(scale)+1 : 0;
    // Power-of-two scaling puts every query in the unit box used by the
    // upstream filter. Use its worst-case double filter for VF and EE.
    const auto vertex = [exponent](Vec p) {
        return ticcd::Vector3{std::scalbn(p[0],-exponent),std::scalbn(p[1],-exponent),std::scalbn(p[2],-exponent)};
    };
    const double filter = 7.549516567451064e-15;
    const ticcd::Array3 error = ticcd::Array3::Constant(filter);
    const double precision = std::scalbn(distance_tolerance,-exponent);
    if (!std::isfinite(precision) || precision <= 0.) throw std::runtime_error("unrepresentable CCD tolerance");
    const auto query = [&](auto function, const Vec& p0, const Vec& q0, const Vec& r0, const Vec& s0,
                            const Vec& p1, const Vec& q1, const Vec& r1, const Vec& s1) {
        if (result.time == 0) return;
        double time = nc::infinity, actual = precision;
        // The upstream breadth-first search documents early termination and its achieved tolerance.
        if (function(vertex(p0),vertex(q0),vertex(r0),vertex(s0),vertex(p1),vertex(q1),vertex(r1),vertex(s1),
                     error,std::scalbn(margin,-exponent),time,precision,1.,1024L,actual,false,
                     ticcd::CCDRootFindingMethod::BREADTH_FIRST_SEARCH)) {
            if (!std::isfinite(time) || time < 0 || time > 1 || !std::isfinite(actual))
                throw std::runtime_error("invalid Tight Inclusion candidate");
            if (time < result.time) result = {time,std::scalbn(actual+filter,exponent)};
        }
    };
    for (std::size_t i=0;i<3;++i) {
        query(ticcd::vertexFaceCCD,a0[i],b0[0],b0[1],b0[2],a1[i],b1[0],b1[1],b1[2]);
        query(ticcd::vertexFaceCCD,b0[i],a0[0],a0[1],a0[2],b1[i],a1[0],a1[1],a1[2]);
        for (std::size_t j=0;j<3;++j) {
            const auto k=(i+1)%3, l=(j+1)%3;
            query(ticcd::edgeEdgeCCD,a0[i],a0[k],b0[j],b0[l],a1[i],a1[k],b1[j],b1[l]);
        }
    }
    return result;
}
}
