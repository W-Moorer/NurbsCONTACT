#include "nurbscontact/cpu_mesh.hpp"
#include "cpu_mesh_contours.hpp"
#include <map>
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

namespace {
std::vector<Point> exact_intersection(const Triangle& a, const Triangle& b) {
    using Predicate = CGAL::Exact_predicates_inexact_constructions_kernel;
    const auto left = triangle<Predicate>(a), right = triangle<Predicate>(b);
    if (left.is_degenerate() || right.is_degenerate()) throw std::invalid_argument("degenerate triangle");
    if (!CGAL::do_intersect(left, right)) return {};
    const auto hit = CGAL::intersection(triangle<Kernel>(a), triangle<Kernel>(b));
    if (!hit) throw std::runtime_error("inconsistent triangle intersection");
    if (const auto* p = std::get_if<Point>(&*hit)) return {*p};
    if (const auto* s = std::get_if<Kernel::Segment_3>(&*hit)) return {s->source(),s->target()};
    if (const auto* t = std::get_if<Kernel::Triangle_3>(&*hit)) {
        return {t->vertex(0),t->vertex(1),t->vertex(2)};
    }
    return std::get<std::vector<Point>>(*hit);
}

std::vector<Vec> world_vertices(const std::vector<Point>& intersection) {
    std::vector<Vec> vertices;
    vertices.reserve(intersection.size());
    for (const auto& p : intersection) vertices.push_back(as_vec(p));
    return vertices;
}
}

std::vector<Vec> intersect_triangles(const Triangle& a, const Triangle& b) {
    return world_vertices(exact_intersection(a,b));
}

struct TriangleContours::Data {
    struct Piece {
        std::optional<Kernel::Segment_3> edge;
        bool transverse{true};
    };
    std::map<Point,std::size_t,Kernel::Less_xyz_3> owner;
    std::vector<std::size_t> parent;
    std::vector<Piece> pieces;
    std::size_t root(std::size_t i) {
        while (parent[i]!=i) {
            parent[i]=parent[parent[i]];
            i=parent[i];
        }
        return i;
    }
    std::optional<Vec> vector_area(const std::vector<std::size_t>& group) const {
        std::map<Point,std::pair<unsigned,unsigned>,Kernel::Less_xyz_3> degree;
        std::optional<Point> origin;
        Kernel::Vector_3 area=CGAL::NULL_VECTOR;
        for (const auto i : group) {
            const auto& piece=pieces[i];
            if (!piece.transverse) return {};
            if (!piece.edge) continue;
            const auto& p=piece.edge->source();
            const auto& q=piece.edge->target();
            if (++degree[p].first>1 || ++degree[q].second>1) return {};
            if (!origin) origin=p;
            area=area+CGAL::cross_product(p-*origin,q-*origin);
        }
        if (degree.size()<3) return {};
        for (const auto& [point,counts] : degree)
            if (counts.first!=1 || counts.second!=1) return {};
        return Vec{CGAL::to_double(area.x()/2),CGAL::to_double(area.y()/2),CGAL::to_double(area.z()/2)};
    }
};

TriangleContours::TriangleContours() : data_(std::make_unique<Data>()) {}
TriangleContours::~TriangleContours() = default;

std::vector<Vec> TriangleContours::intersect(const Triangle& a, const Triangle& b) {
    const auto points=exact_intersection(a,b);
    if (points.empty()) return {};
    const auto id=data_->parent.size();
    data_->parent.push_back(id);
    Data::Piece piece;
    if (points.size()==2) {
        const auto na=triangle<Kernel>(a).supporting_plane().orthogonal_vector();
        const auto nb=triangle<Kernel>(b).supporting_plane().orthogonal_vector();
        // For the portion of A inside B, the outward co-normal points toward
        // the exterior of B. The positive boundary tangent is n_A cross n_B.
        const auto sign=(points[1]-points[0])*CGAL::cross_product(na,nb);
        piece.transverse=sign!=0;
        if (piece.transverse)
            piece.edge=sign>0 ? Kernel::Segment_3(points[0],points[1]) : Kernel::Segment_3(points[1],points[0]);
    } else if (points.size()>2) piece.transverse=false;
    data_->pieces.push_back(std::move(piece));
    for (const auto& point : points) {
        const auto [it,fresh]=data_->owner.emplace(point,id);
        if (!fresh) data_->parent[data_->root(it->second)]=data_->root(id);
    }
    return world_vertices(points);
}

std::vector<IntersectionResult::Contour> TriangleContours::finish(std::span<const Intersection> hits) {
    if (hits.size()!=data_->parent.size()) throw std::logic_error("inconsistent triangle contour hits");
    std::map<std::size_t,std::vector<std::size_t>> groups;
    for (std::size_t i=0;i<hits.size();++i) groups[data_->root(i)].push_back(i);
    std::vector<IntersectionResult::Contour> result;
    for (auto& [root,group] : groups) {
        std::vector<std::array<std::size_t,2>> pairs;
        for (const auto i : group) pairs.push_back({hits[i].triangle_a,hits[i].triangle_b});
        std::sort(pairs.begin(),pairs.end());
        result.push_back({hits[0].body_a,hits[0].body_b,std::move(pairs),data_->vector_area(group)});
    }
    std::sort(result.begin(),result.end(),[](const auto& a,const auto& b) { return a.triangles.front()<b.triangles.front(); });
    return result;
}
} // namespace nc::cpu
