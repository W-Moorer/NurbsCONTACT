#include "nurbscontact/triangle_bvh.hpp"

#include <numeric>
#include <queue>

namespace nc::dynamics {
namespace {

using Box = TriangleBvh::Box;

struct Best {
    Vec a{};
    Vec b{};
    double squared_distance{infinity};
    void update(Vec x, Vec y) {
        const double d = squared(sub(x, y));
        if (d < squared_distance) {
            a = x;
            b = y;
            squared_distance = d;
        }
    }
    PointPair result() const { return {a, b, std::sqrt(squared_distance)}; }
};

Vec project_segment(Vec p, Vec a, Vec b) {
    const Vec d = sub(b, a);
    const double size = squared(d);
    if (size == 0) return a;
    return add(a, scale(d, std::clamp(dot3(d, sub(p, a)) / size, 0.0, 1.0)));
}

Vec project_triangle(Vec p, const Triangle& t) {
    const Vec u = sub(t[1], t[0]);
    const Vec v = sub(t[2], t[0]);
    const Vec n = cross3(u, v);
    const double area = squared(n);
    if (area > 0) {
        const Vec w = sub(p, t[0]);
        const double s = dot3(cross3(w, v), n) / area;
        const double r = dot3(cross3(u, w), n) / area;
        if (s >= 0 && r >= 0 && s + r <= 1) return add(t[0], add(scale(u, s), scale(v, r)));
    }
    Best best;
    for (std::size_t i = 0; i < 3; ++i) best.update(p, project_segment(p, t[i], t[(i + 1) % 3]));
    return best.b;
}

void segment_pair(Best& best, Vec a, Vec b, Vec c, Vec d) {
    const Vec u = sub(b, a);
    const Vec v = sub(d, c);
    const Vec w = sub(a, c);
    const Vec n = cross3(u, v);
    const double denom = squared(n);
    if (denom > 0) {
        const double s = dot3(cross3(v, w), n) / denom;
        const double t = dot3(cross3(u, w), n) / denom;
        if (s >= 0 && s <= 1 && t >= 0 && t <= 1) {
            best.update(add(a, scale(u, s)), add(c, scale(v, t)));
        }
    }
    // Endpoint-to-edge cases also occur in the vertex-to-triangle pass.
}

std::optional<Vec> crossing(Vec a, Vec b, const Triangle& t) {
    const Vec u = sub(t[1], t[0]);
    const Vec v = sub(t[2], t[0]);
    const Vec n = cross3(u, v);
    const Vec direction = sub(b, a);
    const double denom = dot3(n, direction);
    if (denom == 0) return std::nullopt;
    const double s = dot3(n, sub(t[0], a)) / denom;
    if (s < 0 || s > 1) return std::nullopt;
    const Vec p = add(a, scale(direction, s));
    const Vec w = sub(p, t[0]);
    const double x = dot3(cross3(w, v), n);
    const double y = dot3(cross3(u, w), n);
    return x >= 0 && y >= 0 && x + y <= squared(n) ? std::optional{p} : std::nullopt;
}

Box triangle_box(const Triangle& t) {
    Box box{t[0], t[0]};
    for (std::size_t i = 0; i < 3; ++i) {
        box.lo[i] = std::min({t[0][i], t[1][i], t[2][i]});
        box.hi[i] = std::max({t[0][i], t[1][i], t[2][i]});
    }
    return box;
}

Box merge_box(Box a, Box b) {
    for (std::size_t i = 0; i < 3; ++i) {
        a.lo[i] = std::min(a.lo[i], b.lo[i]);
        a.hi[i] = std::max(a.hi[i], b.hi[i]);
    }
    return a;
}

Box transform_box(Box box, Pose pose) {
    const Vec center = pose.point(scale(add(box.lo, box.hi), 0.5));
    const Vec extent = scale(sub(box.hi, box.lo), 0.5);
    const double c = std::abs(pose.cosine);
    const double s = std::abs(pose.sine);
    const Vec radius{c * extent[0] + s * extent[1], s * extent[0] + c * extent[1], extent[2]};
    return {sub(center, radius), add(center, radius)};
}

Box transform_box(Box box, const RigidPlacement& pose) {
    const Vec center = pose.point(scale(add(box.lo, box.hi), 0.5));
    const Vec extent = scale(sub(box.hi, box.lo), 0.5);
    Vec radius{};
    for (std::size_t axis = 0; axis < radius.size(); ++axis) {
        for (std::size_t column = 0; column < extent.size(); ++column)
            radius[axis] += std::abs(pose.basis[column][axis]) * extent[column];
    }
    return {sub(center, radius), add(center, radius)};
}

double separation(Box a, Box b) {
    double bound = 0;
    for (std::size_t i = 0; i < 3; ++i) {
        const double padding = 32 * std::numeric_limits<double>::epsilon() *
            std::max({1.0, std::abs(a.lo[i]), std::abs(a.hi[i]), std::abs(b.lo[i]), std::abs(b.hi[i])});
        const double gap = std::max({0.0, a.lo[i] - b.hi[i] - padding, b.lo[i] - a.hi[i] - padding});
        bound += gap * gap;
    }
    return bound;
}

double size(Box a) {
    const Vec extent = sub(a.hi, a.lo);
    return std::max({extent[0], extent[1], extent[2]});
}

Box translate_sweep(Box box, Vec start, Vec end) {
    for (std::size_t i = 0; i < 3; ++i) {
        box.lo[i] = down(box.lo[i] + std::min(start[i], end[i]));
        box.hi[i] = up(box.hi[i] + std::max(start[i], end[i]));
    }
    return box;
}

bool prism_within(const Triangle& a, const Triangle& b, Vec start, Vec end, double cutoff) {
    const Triangle bottom{add(b[0], start), add(b[1], start), add(b[2], start)};
    const Triangle top{add(b[0], end), add(b[1], end), add(b[2], end)};
    if (closest_triangles(a, bottom).distance <= cutoff) return true;
    if (start == end) return false;
    if (closest_triangles(a, top).distance <= cutoff) return true;

    // A triangle strictly inside the prism need not meet any boundary face.
    const Vec u = sub(bottom[1], bottom[0]);
    const Vec v = sub(bottom[2], bottom[0]);
    const Vec w = sub(end, start);
    const double volume = dot3(u, cross3(v, w));
    if (volume != 0) {
        for (Vec vertex : a) {
            const Vec p = sub(vertex, bottom[0]);
            const double r = dot3(p, cross3(v, w)) / volume;
            const double s = dot3(u, cross3(p, w)) / volume;
            const double t = dot3(u, cross3(v, p)) / volume;
            if (r >= 0 && s >= 0 && r + s <= 1 && t >= 0 && t <= 1) return true;
        }
    }
    // The end triangles and six side triangles also cover the whole swept
    // set when the translation is in the plane of B.
    for (std::size_t i = 0; i < 3; ++i) {
        const std::size_t j = (i + 1) % 3;
        if (closest_triangles(a, {bottom[i], bottom[j], top[j]}).distance <= cutoff ||
            closest_triangles(a, {bottom[i], top[j], top[i]}).distance <= cutoff) return true;
    }
    return false;
}

}  // namespace

Pose relative_pose(Pose a, Pose b) {
    return {a.inverse_rotate(sub(b.origin, a.origin)),
            a.cosine*b.cosine + a.sine*b.sine, a.cosine*b.sine - a.sine*b.cosine};
}

PointPair closest_triangles(const Triangle& a, const Triangle& b) {
    Best best;
    for (std::size_t i = 0; i < 3; ++i) {
        best.update(a[i], project_triangle(a[i], b));
        best.update(project_triangle(b[i], a), b[i]);
        if (auto p = crossing(a[i], a[(i + 1) % 3], b)) return {*p, *p, 0};
        if (auto p = crossing(b[i], b[(i + 1) % 3], a)) return {*p, *p, 0};
    }
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            segment_pair(best, a[i], a[(i + 1) % 3], b[j], b[(j + 1) % 3]);
        }
    }
    return best.result();
}

TriangleBvh::TriangleBvh(std::vector<Triangle> triangles, std::size_t leaf_size)
    : triangles_(std::move(triangles)), ids_(triangles_.size()) {
    if (triangles_.empty() || leaf_size == 0) throw std::invalid_argument("empty triangle BVH or zero leaf size");
    for (const Triangle& t : triangles_) {
        for (Vec p : t) {
            if (!std::all_of(p.begin(), p.end(), [](double x) { return std::isfinite(x); })) {
                throw std::invalid_argument("nonfinite triangle vertex");
            }
        }
        triangle_boxes_.push_back(triangle_box(t));
    }
    std::iota(ids_.begin(), ids_.end(), std::size_t{0});
    nodes_.reserve(2 * triangles_.size());
    build(0, ids_.size(), leaf_size);
}

std::size_t TriangleBvh::build(std::size_t begin, std::size_t end, std::size_t leaf_size) {
    Box box = triangle_boxes_[ids_[begin]];
    for (std::size_t i = begin + 1; i < end; ++i) box = merge_box(box, triangle_boxes_[ids_[i]]);
    const std::size_t node = nodes_.size();
    nodes_.push_back({box, begin, end, 0, 0});
    if (end - begin <= leaf_size) return node;
    const Vec extent = sub(box.hi, box.lo);
    const auto axis = static_cast<std::size_t>(std::max_element(extent.begin(), extent.end()) - extent.begin());
    const std::size_t middle = begin + (end - begin) / 2;
    std::nth_element(ids_.begin() + static_cast<std::ptrdiff_t>(begin),
                     ids_.begin() + static_cast<std::ptrdiff_t>(middle),
                     ids_.begin() + static_cast<std::ptrdiff_t>(end), [&](std::size_t a, std::size_t b) {
        return triangle_boxes_[a].lo[axis] + triangle_boxes_[a].hi[axis] <
               triangle_boxes_[b].lo[axis] + triangle_boxes_[b].hi[axis];
    });
    const std::size_t left = build(begin, middle, leaf_size);
    const std::size_t right = build(middle, end, leaf_size);
    nodes_[node].first = left;
    nodes_[node].second = right;
    return node;
}

std::optional<PointPair> TriangleBvh::closest_within(const TriangleBvh& b, Pose placement, double cutoff,
                                                     TriangleHint* hint, TriangleStatistics& statistics) const {
    return closest_impl(b, placement, cutoff, hint, statistics);
}

std::optional<PointPair> TriangleBvh::closest_rigid_within(const TriangleBvh& b, const RigidPlacement& placement,
                                                           double cutoff, TriangleHint* hint,
                                                           TriangleStatistics& statistics) const {
    return closest_impl(b, placement, cutoff, hint, statistics);
}

template<class Placement>
std::optional<PointPair> TriangleBvh::closest_impl(const TriangleBvh& b, const Placement& placement, double cutoff,
                                                   TriangleHint* hint, TriangleStatistics& statistics) const {
    if (!(cutoff >= 0)) throw std::invalid_argument("invalid triangle proximity cutoff");
    struct Pair { std::size_t a; std::size_t b; double bound; };
    const auto smaller = [](Pair a, Pair b2) { return a.bound > b2.bound; };
    std::priority_queue<Pair, std::vector<Pair>, decltype(smaller)> pending(smaller);
    double limit = cutoff * cutoff;
    std::optional<PointPair> result;
    TriangleHint best;
    const auto test = [&](std::size_t i, std::size_t j) {
        ++statistics.triangles;
        const Triangle& original = b.triangles_[j];
        const Triangle moved{placement.point(original[0]), placement.point(original[1]), placement.point(original[2])};
        const PointPair pair = closest_triangles(triangles_[i], moved);
        const double d = squared(sub(pair.a, pair.b));
        if (d <= limit) {
            limit = d;
            result = pair;
            best = {i, j, true};
        }
    };
    const auto enqueue = [&](std::size_t i, std::size_t j) {
        const double lower = separation(nodes_[i].box, transform_box(b.nodes_[j].box, placement));
        if (lower <= limit) pending.push({i, j, lower});
    };
    enqueue(0, 0);
    if (pending.empty()) return std::nullopt;
    if (hint && hint->valid && hint->a < triangles_.size() && hint->b < b.triangles_.size()) test(hint->a, hint->b);
    while (!pending.empty() && pending.top().bound <= limit && (!result || limit > 0)) {
        const Pair pair = pending.top();
        pending.pop();
        ++statistics.nodes;
        const Node& left = nodes_[pair.a];
        const Node& right = b.nodes_[pair.b];
        if (left.leaf() && right.leaf()) {
            for (std::size_t i = left.begin; i < left.end; ++i) {
                for (std::size_t j = right.begin; j < right.end; ++j) test(ids_[i], b.ids_[j]);
            }
        } else if (right.leaf() || (!left.leaf() && size(left.box) >= size(right.box))) {
            enqueue(left.first, pair.b);
            enqueue(left.second, pair.b);
        } else {
            enqueue(pair.a, right.first);
            enqueue(pair.a, right.second);
        }
    }
    if (hint && result) *hint = best;
    return result;
}

bool TriangleBvh::swept_within(const TriangleBvh& b, Vec start, Vec end, double cutoff,
                               TriangleHint& hint, TriangleStatistics& statistics) const {
    if (!(cutoff >= 0) || !std::isfinite(cutoff)) throw std::invalid_argument("invalid swept proximity cutoff");
    const double limit = up(cutoff * cutoff);
    const auto test = [&](std::size_t i, std::size_t j) {
        if (separation(triangle_boxes_[i], translate_sweep(b.triangle_boxes_[j], start, end)) > limit) return false;
        ++statistics.triangles;
        if (!prism_within(triangles_[i], b.triangles_[j], start, end, cutoff)) return false;
        hint = {i, j, true};
        return true;
    };
    const auto visit = [&](const auto& self, std::size_t i, std::size_t j) -> bool {
        const Node& a_node = nodes_[i];
        const Node& b_node = b.nodes_[j];
        if (separation(a_node.box, translate_sweep(b_node.box, start, end)) > limit) return false;
        ++statistics.nodes;
        if (a_node.leaf() && b_node.leaf()) {
            for (std::size_t a_id = a_node.begin; a_id < a_node.end; ++a_id) {
                for (std::size_t b_id = b_node.begin; b_id < b_node.end; ++b_id) {
                    if (test(ids_[a_id], b.ids_[b_id])) return true;
                }
            }
            return false;
        }
        if (b_node.leaf() || (!a_node.leaf() && size(a_node.box) >= size(b_node.box))) {
            return self(self, a_node.first, j) || self(self, a_node.second, j);
        }
        return self(self, i, b_node.first) || self(self, i, b_node.second);
    };
    return (hint.valid && hint.a < triangles_.size() && hint.b < b.triangles_.size() && test(hint.a, hint.b)) ||
        visit(visit, 0, 0);
}

std::optional<double> TriangleBvh::ray_first(Vec origin, Vec direction, double maximum,
                                            TriangleStatistics& statistics, Vec* hit_normal) const {
    const auto enters = [&](const Box& box, double limit) {
        double low = 0;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            constexpr double margin = 1e-10;
            if (std::abs(direction[axis]) < 1e-15) {
                if (origin[axis] < box.lo[axis] - margin || origin[axis] > box.hi[axis] + margin) return false;
            } else {
                const double a = (box.lo[axis] - margin - origin[axis]) / direction[axis];
                const double b = (box.hi[axis] + margin - origin[axis]) / direction[axis];
                low = std::max(low, std::min(a, b));
                limit = std::min(limit, std::max(a, b));
                if (low > limit) return false;
            }
        }
        return true;
    };
    std::optional<double> result;
    const auto visit = [&](const auto& self, std::size_t index) -> void {
        const Node& node = nodes_[index];
        ++statistics.nodes;
        if (!enters(node.box, maximum)) return;
        if (!node.leaf()) {
            self(self, node.first);
            self(self, node.second);
            return;
        }
        for (std::size_t i = node.begin; i < node.end; ++i) {
            ++statistics.triangles;
            const Triangle& triangle = triangles_[ids_[i]];
            const Vec a = sub(triangle[1], triangle[0]);
            const Vec b = sub(triangle[2], triangle[0]);
            const Vec p = cross3(direction, b);
            const double determinant = dot3(a, p);
            if (std::abs(determinant) <= 1e-14 * length(a) * length(b)) continue;
            const Vec offset = sub(origin, triangle[0]);
            const double u = dot3(offset, p) / determinant;
            const Vec q = cross3(offset, a);
            const double v = dot3(direction, q) / determinant;
            constexpr double edge_tolerance = 2e-10;
            if (u < -edge_tolerance || v < -edge_tolerance || u + v > 1 + edge_tolerance) continue;
            const double distance = dot3(b, q) / determinant;
            if (distance >= 0 && distance <= maximum) {
                maximum = distance;
                result = distance;
                if (hit_normal) *hit_normal = cross3(a,b);
            }
        }
    };
    visit(visit, 0);
    return result;
}

}  // namespace nc::dynamics
