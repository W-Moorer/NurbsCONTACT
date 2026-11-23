#include "nurbscontact/cpu_mesh.hpp"
#ifdef NC_WITH_MESH_CCD
#include "cpu_mesh_ccd.hpp"
#include <CGAL/squared_distance_3.h>
#endif

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Intersections_3/Triangle_3_Triangle_3.h>

namespace nc::cpu {
namespace {
using Kernel = CGAL::Exact_predicates_exact_constructions_kernel;
using Point = Kernel::Point_3;

template<class K>
typename K::Triangle_3 triangle(const Triangle& tri) {
    const auto point = [](Vec v) { return typename K::Point_3(v[0], v[1], v[2]); };
    return {point(tri[0]), point(tri[1]), point(tri[2])};
}

Vec as_vec(const Point& p) {
    return {CGAL::to_double(p.x()), CGAL::to_double(p.y()), CGAL::to_double(p.z())};
}
}

#ifdef NC_WITH_MESH_CCD
std::pair<double,double> squared_distance_bounds(const Triangle& a, const Triangle& b) {
    return CGAL::to_interval(CGAL::squared_distance(triangle<Kernel>(a),triangle<Kernel>(b)));
}

TriangleTime translated_triangle_ccd(const Triangle& a, const Triangle& b,
                                     Vec a0, Vec a1, Vec b0, Vec b1) {
    using Number = Kernel::FT;
    using Vector = Kernel::Vector_3;
    const auto left = triangle<Kernel>(a), right = triangle<Kernel>(b);
    const auto origin = [](Vec v) { return Point(v[0], v[1], v[2]); };
    const auto velocity = (origin(b1) - origin(b0)) - (origin(a1) - origin(a0));
    Number enter(0), leave(1);
    const auto overlaps = [&](const Vector& axis) {
        if (axis == CGAL::NULL_VECTOR) return true;
        const auto project = [&](const auto& tri) {
            const Number p = (tri[0] - CGAL::ORIGIN) * axis, q = (tri[1] - CGAL::ORIGIN) * axis,
                         r = (tri[2] - CGAL::ORIGIN) * axis;
            return std::pair{std::min({p,q,r}),std::max({p,q,r})};
        };
        const auto [amin,amax] = project(left);
        const auto [bmin,bmax] = project(right);
        const Number rate = velocity * axis;
        if (rate == 0) return amin <= bmax && bmin <= amax;
        const Number t0 = (amin-bmax)/rate, t1 = (amax-bmin)/rate;
        enter = std::max(enter,std::min(t0,t1));
        leave = std::min(leave,std::max(t0,t1));
        return enter <= leave;
    };
    std::array<Vector,3> ea, eb;
    for (int k = 0; k < 3; ++k) {
        ea[k] = left[(k+1)%3] - left[k];
        eb[k] = right[(k+1)%3] - right[k];
    }
    const auto na = CGAL::cross_product(ea[0],ea[1]), nb = CGAL::cross_product(eb[0],eb[1]);
    if (!overlaps(na) || !overlaps(nb)) return {};
    for (int k = 0; k < 3; ++k) {
        // In-plane edge normals also separate disjoint coplanar triangles.
        if (!overlaps(CGAL::cross_product(na,ea[k])) || !overlaps(CGAL::cross_product(nb,eb[k]))) return {};
        for (const auto& e : eb) if (!overlaps(CGAL::cross_product(ea[k],e))) return {};
    }
    const auto [low, high] = CGAL::to_interval(enter);
    const double speed = std::sqrt(CGAL::to_interval(velocity.squared_length()).second);
    return {std::max(0.,low),nc::up((high-low)*speed)};
}
#endif

std::vector<Vec> intersect_triangles(const Triangle& a, const Triangle& b) {
    using Predicate = CGAL::Exact_predicates_inexact_constructions_kernel;
    const auto left = triangle<Predicate>(a), right = triangle<Predicate>(b);
    if (left.is_degenerate() || right.is_degenerate()) throw std::invalid_argument("degenerate triangle");
    if (!CGAL::do_intersect(left, right)) return {};
    const auto hit = CGAL::intersection(triangle<Kernel>(a), triangle<Kernel>(b));
    if (!hit) throw std::runtime_error("inconsistent triangle intersection");
    if (const auto* p = std::get_if<Point>(&*hit)) return {as_vec(*p)};
    if (const auto* s = std::get_if<Kernel::Segment_3>(&*hit)) return {as_vec(s->source()), as_vec(s->target())};
    if (const auto* t = std::get_if<Kernel::Triangle_3>(&*hit)) {
        return {as_vec(t->vertex(0)), as_vec(t->vertex(1)), as_vec(t->vertex(2))};
    }
    std::vector<Vec> vertices;
    for (const Point& p : std::get<std::vector<Point>>(*hit)) vertices.push_back(as_vec(p));
    return vertices;
}
} // namespace nc::cpu
