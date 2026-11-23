#include "nurbscontact/cpu_mesh.hpp"

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
