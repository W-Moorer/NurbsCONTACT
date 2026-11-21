#include "nurbscontact/query.hpp"

#include "axial.hpp"

#include <algorithm>
#include <cfenv>
#include <deque>
#include <iterator>
#include <map>
#include <numbers>
#include <queue>
#include <tuple>
#include <utility>

namespace nc {
namespace {

constexpr std::size_t missing = std::numeric_limits<std::size_t>::max();

struct Node {
    IVec box;
    std::vector<IVec> supports;
    Rect domain;
    std::size_t patch{missing};
    std::size_t first{missing};
    std::size_t second{missing};
    unsigned depth{};
    TrimClass trim{TrimClass::mixed};
    bool classified{};
    bool sampled{};
    std::vector<Sample> witnesses;
};

struct Terms {
    IVec constant;
    IVec linear;
    IVec cosine;
    IVec sine;

    IVec eval(Interval time, SinCos sc) const {
        return constant + linear * time + cosine * sc.cos + sine * sc.sin;
    }
};

class Trajectory {
public:
    explicit Trajectory(const Motion& motion)
        : position_(exact(motion.translation)), velocity_(exact(motion.velocity)),
          angle_(motion.angle), omega_(motion.angular_velocity) {
        const IVec axis = exact(motion.axis);
        if (!finite(axis) || !finite(position_) || !finite(velocity_) ||
            !std::isfinite(angle_) || !std::isfinite(omega_)) {
            throw std::invalid_argument("motion values must be finite");
        }
        const Interval length = norm(axis);
        if (length.lo <= 0 || !std::isfinite(length.hi)) {
            throw std::invalid_argument("motion axis must be nonzero and finite");
        }
        axis_ = axis / length;
    }

    Terms terms(IVec p) const {
        if (angle_ == 0 && omega_ == 0) return {position_ + p, velocity_, {}, {}};
        const IVec axial = axis_ * dot(axis_, p);
        return {position_ + axial, velocity_, p - axial, cross(axis_, p)};
    }

    SinCos trig(Interval time) const { return sin_cos(angle_ + time * omega_); }
    IVec point(IVec p, Interval time, SinCos sc) const { return terms(p).eval(time, sc); }
    Vec local_guess(Vec world, double time, SinCos sc) const {
        const IVec relative = exact(world) - position_ - velocity_ * time;
        const IVec axial = axis_ * dot(axis_, relative);
        return midpoint(axial + (relative - axial) * sc.cos - cross(axis_, relative) * sc.sin);
    }
    bool constant_rotation() const { return omega_ == 0; }
    IVec translation_velocity() const { return velocity_; }

    detail::AxialEnvelope initial_envelope(detail::AxialEnvelope envelope) const {
        const SinCos sc = trig(0);
        envelope.origin = point(envelope.origin, 0, sc);
        if (angle_ != 0) {
            const IVec axial = axis_ * dot(axis_, envelope.axis);
            envelope.axis = axial + (envelope.axis - axial) * sc.cos + cross(axis_, envelope.axis) * sc.sin;
        }
        return envelope;
    }
    double chord_error(Interval cosine, Interval sine, Interval time) const {
        const Interval width = Interval{time.hi} - time.lo;
        const Interval amplitude = Interval{magnitude(cosine)} + magnitude(sine);
        return (square(Interval{omega_}) * square(width) * amplitude / 8).hi;
    }
    double travel(IVec box, Interval time) const {
        return (Interval{norm(velocity_).hi} + Interval{std::abs(omega_)} * norm(box).hi).hi * time.width();
    }

private:
    static bool finite(IVec vector) {
        return std::all_of(vector.begin(), vector.end(), [](Interval x) { return std::isfinite(x.lo); });
    }

    IVec position_;
    IVec velocity_;
    IVec axis_;
    double angle_;
    double omega_;
};

IVec moved_box(const Node& node, const Trajectory& motion, Interval time) {
    const SinCos sc = motion.trig(time);
    IVec box = motion.point(node.supports.front(), time, sc);
    for (std::size_t k = 1; k < node.supports.size(); ++k) {
        box = hull(box, motion.point(node.supports[k], time, sc));
    }
    return box;
}

IVec chord_box(const Node& node, const Trajectory& motion, Interval time) {
    const auto first = motion.trig(time.lo);
    const auto last = motion.trig(time.hi);
    IVec result{{{infinity, -infinity}, {infinity, -infinity}, {infinity, -infinity}}};
    for (IVec point : node.supports) {
        const Terms terms = motion.terms(point);
        IVec box = hull(terms.eval(time.lo, first), terms.eval(time.hi, last));
        for (std::size_t i = 0; i < box.size(); ++i) {
            const double error = motion.chord_error(terms.cosine[i], terms.sine[i], time);
            box[i] = box[i] + Interval{-error, error};
        }
        result = hull(result, box);
    }
    return result;
}

double box_distance(IVec a, IVec b) {
    IVec separation;
    for (std::size_t i = 0; i < 3; ++i) {
        separation[i] = std::max({0.0, (Interval{a[i].lo} - b[i].hi).lo,
                                 (Interval{b[i].lo} - a[i].hi).lo});
    }
    return norm(separation).lo;
}

double size(IVec box) {
    return std::max({box[0].width(), box[1].width(), box[2].width()});
}

Node make_leaf(const Model& model, std::size_t patch, Rect domain, unsigned depth) {
    Node result;
    result.supports = support(model.patches[patch].surface, domain);
    result.box = bounding_box(result.supports);
    result.domain = domain;
    result.patch = patch;
    result.depth = depth;
    return result;
}

std::size_t build_bvh(std::deque<Node>& nodes, std::vector<std::size_t>::iterator begin,
                      std::vector<std::size_t>::iterator end) {
    if (end - begin == 1) return *begin;
    IVec box = nodes[*begin].box;
    for (auto it = std::next(begin); it != end; ++it) box = hull(box, nodes[*it].box);
    const auto axis = static_cast<std::size_t>(std::max_element(box.begin(), box.end(),
        [](Interval a, Interval b) { return a.width() < b.width(); }) - box.begin());
    auto middle = begin + (end - begin) / 2;
    std::nth_element(begin, middle, end, [&](std::size_t a, std::size_t b) {
        return nodes[a].box[axis].center() < nodes[b].box[axis].center();
    });
    const std::size_t first = build_bvh(nodes, begin, middle);
    const std::size_t second = build_bvh(nodes, middle, end);
    Node result;
    result.box = box;
    result.supports = corners(box);
    result.first = first;
    result.second = second;
    nodes.push_back(std::move(result));
    return nodes.size() - 1;
}

void validate(const Options& options) {
    if (std::fegetround() != FE_TONEAREST) throw std::invalid_argument("round-to-nearest is required");
    if (!std::isfinite(options.distance_tolerance) || options.distance_tolerance <= 0 ||
        !std::isfinite(options.time_tolerance) || options.time_tolerance <= 0 ||
        options.max_nodes == 0 || options.max_spatial_depth > 96) {
        throw std::invalid_argument("invalid query tolerances or budget");
    }
}

}  // namespace

struct PreparedModel::Data {
    Model model;
    std::deque<Node> nodes;
    std::vector<std::optional<detail::AxialEnvelope>> axial;
    std::size_t root{};
};

namespace {

class SearchTree {
public:
    explicit SearchTree(const PreparedModel::Data& prepared) : prepared_(prepared), nodes_(prepared.nodes) {}

    const Node& operator[](std::size_t id) const { return nodes_[id]; }
    std::size_t root() const { return prepared_.root; }
    bool leaf(std::size_t id) const { return nodes_[id].patch != missing; }

    bool can_split(std::size_t id, const Options& options) const {
        const Node& node = nodes_[id];
        if (node.first != missing) return true;
        const Rect box = node.domain;
        return node.depth < options.max_spatial_depth &&
               ((box.u.lo < box.u.center() && box.u.center() < box.u.hi) ||
                (box.v.lo < box.v.center() && box.v.center() < box.v.hi));
    }

    bool outside(std::size_t id, Stats& stats) {
        Node& node = nodes_[id];
        if (!leaf(id)) return false;
        if (!node.classified) {
            node.trim = trim(id).classify(node.domain);
            node.classified = true;
            if (node.trim == TrimClass::outside) ++stats.trim_rejected;
        }
        return node.trim == TrimClass::outside;
    }

    const std::vector<Sample>& witnesses(std::size_t id, Stats& stats) {
        Node& node = nodes_[id];
        if (!leaf(id) || outside(id, stats)) return node.witnesses;
        if (!node.sampled) {
            node.witnesses = samples(patch(id), trim(id), node.domain, node.trim);
            node.sampled = true;
        }
        return node.witnesses;
    }

    std::optional<Sample> planar_projection(std::size_t id, Vec target) const {
        const FacePatch& face = patch(id);
        const auto* plane = std::get_if<Plane>(&face.surface);
        if (!plane) return std::nullopt;
        const IVec delta = exact(target) - plane->origin;
        const double xx = dot(plane->x, plane->x).center(), xy = dot(plane->x, plane->y).center();
        const double yy = dot(plane->y, plane->y).center();
        const double dx = dot(delta, plane->x).center(), dy = dot(delta, plane->y).center();
        const double determinant = xx*yy - xy*xy;
        if (!(determinant > 0)) return std::nullopt;
        const double u = (dx*yy - dy*xy) / determinant;
        const double v = (dy*xx - dx*xy) / determinant;
        const auto contained = [&](UV p) {
            return face.domain.u.lo <= p[0].lo && p[0].hi <= face.domain.u.hi &&
                face.domain.v.lo <= p[1].lo && p[1].hi <= face.domain.v.hi;
        };
        if (contained({u, v}) && trim(id).classify(Vec{u, v, 0}) == TrimClass::inside) {
            return Sample{{u, v}, plane->eval({u, v}), {FeatureKind::face, face.face_id}};
        }

        // The one-dimensional search proposes parameters. Only the outward
        // evaluation of a point on the original trim can serve as a witness.
        std::optional<Sample> best;
        double nearest = infinity;
        for (const auto& loop : trim(id).loops) {
            for (const TrimCurve& edge : loop) {
                const auto distance = [&](double t) {
                    const IVec uv = evaluate(edge.geometry, t);
                    const double x = uv[0].center() - u, y = uv[1].center() - v;
                    return xx*x*x + 2*xy*x*y + yy*y*y;
                };
                constexpr int intervals = 12;
                int index = 0;
                double lowest = distance(0);
                for (int i = 1; i <= intervals; ++i) {
                    const double d = distance(static_cast<double>(i) / intervals);
                    if (d < lowest) { index = i; lowest = d; }
                }
                double lo = static_cast<double>(std::max(0, index - 1)) / intervals;
                double hi = static_cast<double>(std::min(intervals, index + 1)) / intervals;
                for (int i = 0; i < 32; ++i) {
                    const double left = std::lerp(lo, hi, 1.0 / 3);
                    const double right = std::lerp(lo, hi, 2.0 / 3);
                    if (distance(left) < distance(right)) hi = right;
                    else lo = left;
                }
                for (double t : {std::midpoint(lo, hi), static_cast<double>(index) / intervals}) {
                    const IVec full = evaluate(edge.geometry, t);
                    const UV uv{full[0], full[1]};
                    if (!contained(uv)) continue;
                    const IVec point = plane->eval(uv);
                    const double d = norm(point - exact(target)).hi;
                    if (d < nearest) {
                        const Feature feature = t == 0 ? edge.start : t == 1 ? edge.end : edge.feature;
                        best = Sample{uv, point, feature};
                        nearest = d;
                    }
                }
            }
        }
        return best;
    }

    std::pair<std::size_t, std::size_t> children(std::size_t id) {
        Node& node = nodes_[id];
        if (node.first != missing) return {node.first, node.second};

        const Rect rect = node.domain;
        const Surface& surface = patch(id).surface;
        const IVec u1 = evaluate(surface, {rect.u.lo, rect.v.center()});
        const IVec u2 = evaluate(surface, {rect.u.hi, rect.v.center()});
        const IVec v1 = evaluate(surface, {rect.u.center(), rect.v.lo});
        const IVec v2 = evaluate(surface, {rect.u.center(), rect.v.hi});
        const double du = norm(u2 - u1).hi;
        const double dv = norm(v2 - v1).hi;
        const bool u_available = rect.u.lo < rect.u.center() && rect.u.center() < rect.u.hi;
        const bool v_available = rect.v.lo < rect.v.center() && rect.v.center() < rect.v.hi;
        // Alternate on every fourth level to resolve regions with coincident
        // endpoints or a highly anisotropic, folded parameterization.
        const bool along_u = u_available && (!v_available ||
            (node.depth % 4 == 2 ? true : node.depth % 4 == 3 ? false : du >= dv));
        const auto [first_rect, second_rect] = split(rect, along_u);
        node.first = nodes_.size();
        nodes_.push_back(make_leaf(prepared_.model, node.patch, first_rect, node.depth + 1));
        node.second = nodes_.size();
        nodes_.push_back(make_leaf(prepared_.model, node.patch, second_rect, node.depth + 1));
        if (node.classified && node.trim == TrimClass::inside) {
            for (std::size_t child : {node.first, node.second}) {
                nodes_[child].classified = true;
                nodes_[child].trim = TrimClass::inside;
            }
        }
        return {node.first, node.second};
    }

private:
    const FacePatch& patch(std::size_t id) const { return prepared_.model.patches[nodes_[id].patch]; }
    const TrimDomain& trim(std::size_t id) const { return prepared_.model.trims[patch(id).trim_id]; }

    const PreparedModel::Data& prepared_;
    std::deque<Node> nodes_;
};

struct Pair {
    std::size_t a;
    std::size_t b;
};

bool choose_a(const SearchTree& a, const SearchTree& b, Pair pair, const Options& options) {
    if (!b.can_split(pair.b, options)) return true;
    if (!a.can_split(pair.a, options)) return false;
    if (a.leaf(pair.a) != b.leaf(pair.b)) return !a.leaf(pair.a);
    return size(a[pair.a].box) >= size(b[pair.b].box);
}

std::array<Pair, 2> children(SearchTree& a, SearchTree& b, Pair pair, const Options& options) {
    const bool split_a = choose_a(a, b, pair, options);
    const auto [first, second] = split_a ? a.children(pair.a) : b.children(pair.b);
    if (split_a) return {Pair{first, pair.b}, Pair{second, pair.b}};
    return {Pair{pair.a, first}, Pair{pair.a, second}};
}

bool exclude_trim(SearchTree& a, SearchTree& b, Pair pair, Stats& stats) {
    return a.outside(pair.a, stats) || b.outside(pair.b, stats);
}

double sample_pair(SearchTree& a, SearchTree& b, Pair pair, const Trajectory& motion,
                   double time, double upper, std::optional<Witness>& witness, Stats& stats) {
    if (!a.leaf(pair.a) || !b.leaf(pair.b)) return upper;
    const auto& points_a = a.witnesses(pair.a, stats);
    const auto& points_b = b.witnesses(pair.b, stats);
    if (points_a.empty() || points_b.empty()) return upper;
    const SinCos sc = motion.trig(time);
    for (const Sample& local : points_b) {
        Sample world = local;
        world.position = motion.point(local.position, time, sc);
        for (const Sample& on_a : points_a) {
            const double bound = norm(world.position - on_a.position).hi;
            if (bound >= upper) continue;
            upper = bound;
            witness = Witness{on_a, world, time};
        }
    }
    return upper;
}

std::optional<Witness> guided_planar_pair(const SearchTree& a, const SearchTree& b, Pair pair,
                                          const Trajectory& motion, double t, double threshold) {
    const SinCos sc = motion.trig(t);
    const auto run = [&](bool start_from_b) -> std::optional<Witness> {
        std::optional<Sample> point_a, point_b;
        if (start_from_b) {
            const Vec center_a = midpoint(a[pair.a].box);
            point_b = b.planar_projection(pair.b, motion.local_guess(center_a, t, sc));
            if (!point_b) return std::nullopt;
            point_b->position = motion.point(point_b->position, t, sc);
            point_a = a.planar_projection(pair.a, midpoint(point_b->position));
        } else {
            const Vec center_b = midpoint(motion.point(exact(midpoint(b[pair.b].box)), t, sc));
            point_a = a.planar_projection(pair.a, center_b);
            if (!point_a) return std::nullopt;
            point_b = b.planar_projection(pair.b, motion.local_guess(midpoint(point_a->position), t, sc));
            if (point_b) point_b->position = motion.point(point_b->position, t, sc);
        }
        if (point_a && point_b && norm(point_b->position - point_a->position).hi <= threshold) {
            return Witness{*point_a, *point_b, t};
        }
        return std::nullopt;
    };
    if (auto witness = run(false)) return witness;
    return run(true);
}

struct DistanceTask {
    Pair pair;
    double lower;
    bool operator<(const DistanceTask& other) const { return lower > other.lower; }
};

struct TimeTask {
    Pair pair;
    Interval time;
    bool operator<(const TimeTask& other) const {
        return std::tie(time.lo, time.hi) > std::tie(other.time.lo, other.time.hi);
    }
};

struct Projection {
    Interval constant;
    Interval linear;
    Interval cosine;
    Interval sine;

    Interval eval(Interval time, SinCos sc) const {
        return constant + linear * time + cosine * sc.cos + sine * sc.sin;
    }
};

std::vector<Interval> merge(std::vector<Interval> windows) {
    std::sort(windows.begin(), windows.end(), [](Interval a, Interval b) { return a.lo < b.lo; });
    std::vector<Interval> result;
    for (Interval window : windows) {
        if (result.empty() || window.lo > result.back().hi) {
            result.push_back(window);
        } else {
            result.back().hi = std::max(result.back().hi, window.hi);
        }
    }
    return result;
}

void append_linear_window(std::vector<Interval>& kept, Interval span, double constant, double slope) {
    if (!std::isfinite(constant) || !std::isfinite(slope)) {
        kept.push_back(span);
        return;
    }
    if (slope == 0) {
        if (constant <= 0) kept.push_back(span);
        return;
    }
    const Interval root = Interval{-constant} / slope;
    if (slope > 0) span.hi = std::min(span.hi, root.hi);
    else span.lo = std::max(span.lo, root.lo);
    if (span.lo <= span.hi) kept.push_back(span);
}

std::vector<Interval> linear_windows(const std::vector<Projection>& coefficients, Interval input, SinCos sc) {
    std::vector<Interval> kept;
    for (const Projection& projection : coefficients) {
        const double constant = projection.eval(0, sc).lo;
        if (input.lo < 0) {
            append_linear_window(kept, {input.lo, std::min(input.hi, 0.0)}, constant, projection.linear.hi);
        }
        if (input.hi >= 0) {
            append_linear_window(kept, {std::max(input.lo, 0.0), input.hi}, constant, projection.linear.lo);
        }
    }
    return merge(std::move(kept));
}

std::vector<Interval> direction_windows(const Node& a, const Node& b, const Trajectory& motion,
                                        Interval input, double clearance, double tolerance) {
    const IVec mid_box = moved_box(b, motion, input.center());
    const Vec center_b = midpoint(mid_box);
    const Vec center_a = midpoint(a.box);
    Vec direction{center_b[0] - center_a[0], center_b[1] - center_a[1], center_b[2] - center_a[2]};
    if (a.patch != missing && b.patch != missing && (a.trim == TrimClass::mixed || b.trim == TrimClass::mixed)) {
        Vec nearest{};
        for (std::size_t i = 0; i < nearest.size(); ++i) {
            if (a.box[i].hi < mid_box[i].lo) nearest[i] = mid_box[i].lo - a.box[i].hi;
            else if (mid_box[i].hi < a.box[i].lo) nearest[i] = mid_box[i].hi - a.box[i].lo;
        }
        if (std::count_if(nearest.begin(), nearest.end(), [](double x) { return x != 0; }) >= 2) {
            direction = nearest;
        }
    }
    const IVec axis = exact(direction);
    const double axis_length = norm(axis).hi;
    if (axis_length == 0 || !std::isfinite(axis_length)) return {input};
    double alpha = -infinity;
    for (IVec support : a.supports) alpha = std::max(alpha, dot(axis, support).hi);
    const Interval limit = Interval{alpha} + Interval{clearance} * axis_length;
    if (!std::isfinite(limit.hi)) return {input};

    std::vector<Projection> coefficients;
    coefficients.reserve(b.supports.size());
    for (IVec support : b.supports) {
        const Terms terms = motion.terms(support);
        coefficients.push_back({dot(axis, terms.constant) - limit.hi, dot(axis, terms.linear),
                                dot(axis, terms.cosine), dot(axis, terms.sine)});
    }
    if (motion.constant_rotation()) return linear_windows(coefficients, input, motion.trig(0));

    // Evaluate the trigonometric motion once per subinterval, irrespective of
    // control-net size. A union test determines which intervals can be removed.
    std::vector<Interval> kept;
    std::vector<Interval> stack{input};
    std::size_t splits = 0;
    while (!stack.empty()) {
        const Interval time = stack.back();
        stack.pop_back();
        const SinCos trig = motion.trig(time);
        bool possible = false;
        bool entirely_possible = false;
        for (const Projection& projection : coefficients) {
            const Interval range = projection.eval(time, trig);
            possible = possible || !std::isfinite(range.lo) || range.lo <= 0;
            if (range.hi <= 0) {
                entirely_possible = true;
                break;
            }
        }
        if (!possible) continue;
        const double middle = time.center();
        if (entirely_possible || time.width() <= tolerance || splits >= 24 ||
            middle == time.lo || middle == time.hi) {
            kept.push_back(time);
        } else {
            stack.push_back({middle, time.hi});
            stack.push_back({time.lo, middle});
            ++splits;
        }
    }
    return merge(std::move(kept));
}

}  // namespace

PreparedModel::PreparedModel(Model model) {
    if (model.patches.empty()) throw std::invalid_argument("model has no supported boundary patches");
    auto data = std::make_shared<Data>();
    data->model = std::move(model);
    std::vector<std::size_t> roots;
    for (std::size_t i = 0; i < data->model.patches.size(); ++i) {
        const FacePatch& patch = data->model.patches[i];
        if (patch.trim_id >= data->model.trims.size()) throw std::invalid_argument("invalid trim index");
        for (Interval axis : {patch.domain.u, patch.domain.v}) {
            if (!std::isfinite(axis.lo) || !std::isfinite(axis.hi) || axis.lo >= axis.hi) {
                throw std::invalid_argument("patch domain must be finite and nonempty");
            }
        }
        roots.push_back(data->nodes.size());
        data->nodes.push_back(make_leaf(data->model, i, patch.domain, 0));
        data->axial.push_back(detail::axial_envelope(patch, data->model.trims[patch.trim_id]));
    }
    data->root = build_bvh(data->nodes, roots.begin(), roots.end());
    data_ = std::move(data);
}

DistanceResult distance(const PreparedModel& a, const PreparedModel& b, const Motion& motion,
                        double time, const Options& options) {
    validate(options);
    if (!std::isfinite(time)) throw std::invalid_argument("query time must be finite");
    const Trajectory trajectory{motion};
    SearchTree tree_a{*a.data_};
    SearchTree tree_b{*b.data_};
    DistanceResult result;
    double terminal_lower = infinity;
    std::priority_queue<DistanceTask> queue;
    const auto enqueue = [&](Pair pair) {
        if (options.eager_trim && exclude_trim(tree_a, tree_b, pair, result.statistics)) return;
        const double lower = box_distance(tree_a[pair.a].box, moved_box(tree_b[pair.b], trajectory, time));
        if (lower < result.upper) queue.push({pair, lower});
    };
    enqueue({tree_a.root(), tree_b.root()});
    while (!queue.empty() && result.statistics.visited < options.max_nodes) {
        const auto [pair, lower] = queue.top();
        if (result.upper <= lower ||
            up(result.upper - std::min(lower, terminal_lower)) <= options.distance_tolerance) break;
        queue.pop();
        ++result.statistics.visited;
        if (exclude_trim(tree_a, tree_b, pair, result.statistics)) continue;
        result.upper = sample_pair(tree_a, tree_b, pair, trajectory, time, result.upper,
                                   result.witness, result.statistics);
        if (lower >= result.upper) continue;
        if (up(result.upper - lower) <= options.distance_tolerance ||
            (!tree_a.can_split(pair.a, options) && !tree_b.can_split(pair.b, options))) {
            terminal_lower = std::min(terminal_lower, lower);
            continue;
        }
        ++result.statistics.spatial_splits;
        for (Pair child : children(tree_a, tree_b, pair, options)) enqueue(child);
    }
    result.lower = std::min(result.upper, terminal_lower);
    if (!queue.empty()) result.lower = std::min(result.lower, queue.top().lower);
    return result;
}

SweepResult sweep(const PreparedModel& a, const PreparedModel& b, const Motion& motion,
                  Interval time, double clearance, const Options& options) {
    validate(options);
    if (!std::isfinite(time.lo) || !std::isfinite(time.hi) || time.lo > time.hi ||
        !std::isfinite(clearance) || clearance < 0) {
        throw std::invalid_argument("invalid sweep interval or clearance");
    }
    const Trajectory trajectory{motion};
    SearchTree tree_a{*a.data_};
    SearchTree tree_b{*b.data_};
    SweepResult result;
    std::priority_queue<TimeTask> queue;
    queue.push({{tree_a.root(), tree_b.root()}, time});
    double terminal = infinity;
    double confirmed = infinity;
    const double time_resolution = options.time_tolerance * 0.25;
    const bool axial_enabled = options.axial_time && trajectory.constant_rotation();
    std::map<std::pair<std::size_t, std::size_t>, std::optional<Interval>> axial_windows;
    std::map<std::pair<std::size_t, std::size_t>, double> last_guidance;

    while (!queue.empty() && result.statistics.visited < options.max_nodes) {
        const auto [pair, pending] = queue.top();
        if (pending.lo >= confirmed ||
            up(confirmed - std::min(terminal, pending.lo)) <= options.time_tolerance) break;
        queue.pop();
        ++result.statistics.visited;
        Interval span{pending.lo, std::min(pending.hi, confirmed)};
        const Node& node_a = tree_a[pair.a];
        const Node& node_b = tree_b[pair.b];
        bool axial_start = false;
        if (axial_enabled && node_a.patch != missing && node_b.patch != missing) {
            const auto& envelope_a = a.data_->axial[node_a.patch];
            const auto& envelope_b = b.data_->axial[node_b.patch];
            if (envelope_a && envelope_b) {
                const auto [it, fresh] = axial_windows.try_emplace({node_a.patch, node_b.patch});
                if (fresh) {
                    it->second = detail::axial_window(*envelope_a, trajectory.initial_envelope(*envelope_b),
                        trajectory.translation_velocity(), time, clearance);
                }
                if (!it->second || disjoint(span, *it->second)) {
                    ++result.statistics.axial_rejected;
                    continue;
                }
                const Interval clipped{std::max(span.lo, it->second->lo), std::min(span.hi, it->second->hi)};
                if (clipped.lo != span.lo || clipped.hi != span.hi) ++result.statistics.axial_clipped;
                span = clipped;
                axial_start = it->second->lo > time.lo;
            }
        }
        if (options.eager_trim && exclude_trim(tree_a, tree_b, pair, result.statistics)) continue;
        const IVec swept = options.chord_box ? chord_box(node_b, trajectory, span)
                                            : moved_box(node_b, trajectory, span);
        if (box_distance(node_a.box, swept) > clearance) continue;
        if (exclude_trim(tree_a, tree_b, pair, result.statistics)) continue;
        const auto windows = options.directional_time
            ? direction_windows(node_a, node_b, trajectory, span, clearance, time_resolution)
            : std::vector<Interval>{span};
        if (windows.empty()) {
            ++result.statistics.direction_rejected;
            continue;
        }
        if (windows.size() > 1 || windows.front().lo != span.lo || windows.front().hi != span.hi) {
            result.statistics.direction_windows += windows.size();
        }
        for (Interval window : windows) {
            if (window.lo >= confirmed) break;
            window.hi = std::min(window.hi, confirmed);
            double sampled_upper = infinity;
            double sample_at = window.center();
            const double early = axial_start
                ? std::min(window.center(), window.lo + options.time_tolerance / 2) : window.lo;
            double last = -infinity;
            for (double sample_time : {window.lo, early, window.center(), window.hi}) {
                if (sample_time >= confirmed) break;
                if (sample_time == last) continue;
                last = sample_time;
                std::optional<Witness> candidate;
                const double bound = sample_pair(tree_a, tree_b, pair, trajectory, sample_time,
                    infinity, candidate, result.statistics);
                if (bound < sampled_upper) {
                    sampled_upper = bound;
                    sample_at = sample_time;
                }
                if (bound <= clearance && candidate) {
                    confirmed = sample_time;
                    result.witness = candidate;
                    break;
                }
            }
            const double proposal_time = axial_start ? early : window.center();
            if (options.guided_witness && tree_a.leaf(pair.a) && tree_b.leaf(pair.b) && proposal_time < confirmed &&
                (axial_start || window.width() <= options.time_tolerance) && result.statistics.guided_attempts < 24) {
                const auto key = std::pair{node_a.patch, node_b.patch};
                const auto found = last_guidance.find(key);
                if (found == last_guidance.end() || std::abs(proposal_time - found->second) >= time_resolution) {
                    last_guidance[key] = proposal_time;
                    ++result.statistics.guided_attempts;
                    if (auto candidate = guided_planar_pair(tree_a, tree_b, pair, trajectory, proposal_time, clearance)) {
                        confirmed = proposal_time;
                        result.witness = std::move(candidate);
                    }
                }
            }
            if (window.lo >= confirmed) break;
            window.hi = std::min(window.hi, confirmed);
            if (up(confirmed - window.lo) <= options.time_tolerance) {
                terminal = std::min(terminal, window.lo);
                continue;
            }
            const double spatial_size = std::max(size(node_a.box), size(node_b.box));
            const bool can_split_space = tree_a.can_split(pair.a, options) || tree_b.can_split(pair.b, options);
            const double middle = window.center();
            const bool can_split_time = window.width() > time_resolution &&
                                        window.lo < middle && middle < window.hi;
            const bool root_pair = !tree_a.leaf(pair.a) || !tree_b.leaf(pair.b);
            double spatial_uncertainty = spatial_size;
            if (can_split_time && can_split_space && !root_pair && std::isfinite(sampled_upper)) {
                const double static_lower = box_distance(node_a.box, moved_box(node_b, trajectory, sample_at));
                spatial_uncertainty = std::min(spatial_size, std::max(0.0, sampled_upper - static_lower));
            }
            const bool prefer_time = !root_pair && (spatial_uncertainty <= options.distance_tolerance ||
                trajectory.travel(node_b.box, window) > spatial_uncertainty);
            if (can_split_time && (!can_split_space || prefer_time)) {
                queue.push({pair, {window.lo, middle}});
                queue.push({pair, {middle, window.hi}});
                ++result.statistics.time_splits;
            } else if (!can_split_time && sampled_upper <= clearance + options.distance_tolerance * 0.25) {
                // This time interval already has a feasible pair as close to
                // the threshold as the requested spatial resolution permits.
                terminal = std::min(terminal, window.lo);
            } else if (can_split_space && (root_pair || spatial_size > options.distance_tolerance)) {
                const auto child_pairs = children(tree_a, tree_b, pair, options);
                Interval inherited = options.inherit_time ? window : span;
                inherited.hi = std::min(inherited.hi, confirmed);
                if (inherited.lo < confirmed) {
                    for (Pair child : child_pairs) queue.push({child, inherited});
                    ++result.statistics.spatial_splits;
                }
                // With inheritance disabled, each child rechecks the entire
                // parent span once. Additional windows would duplicate work.
                if (!options.inherit_time) break;
            } else {
                terminal = std::min(terminal, window.lo);
            }
        }
    }

    const double pending = queue.empty() ? infinity : queue.top().time.lo;
    result.safe_prefix = std::min({terminal, pending, confirmed, time.hi});
    if (std::isfinite(confirmed)) {
        result.status = SweepStatus::hit;
        result.witness_time = confirmed;
    } else if (queue.empty() && !std::isfinite(terminal)) {
        result.status = SweepStatus::clear;
    }
    return result;
}

}  // namespace nc
