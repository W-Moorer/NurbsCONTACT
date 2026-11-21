#include "scene.hpp"

#include <iomanip>
#include <ostream>

namespace nc::hht_bench {
namespace {

using namespace dynamics;

constexpr double contact_skin = 0.012;
constexpr double stiffness = 5000;

Scene drop(const std::filesystem::path& models, unsigned facets) {
    Scene s;
    s.name = "drop";
    s.geometry.push_back(load_geometry(models/"floor.step", facets));
    s.geometry.push_back(load_geometry(models/"ball.step", facets));
    s.bodies = {{"floor", 0}, {"ball", 1, {}, {0,0,1}, Joint::slide, 0}};
    s.contacts = {{0,1,contact_skin,stiffness}};
    s.mass = {1};
    s.q = {0.055 + contact_skin + 0.018};
    s.v = {-0.45};
    s.load = {0};
    return s;
}

Scene cams(const std::filesystem::path& models, unsigned facets) {
    Scene s;
    s.name = "cam-array";
    s.geometry.push_back(load_geometry(models/"follower.step", facets));
    s.geometry.push_back(load_geometry(models/"cam.step", facets));
    constexpr std::size_t count = 6;
    constexpr double speed = 4.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double angle = 0.35 * static_cast<double>(i);
        const double x = 0.38 * static_cast<double>(i);
        const double height = 0.105 + 0.024 * std::sin(angle) + 0.010;
        s.bodies.push_back({"cam", 1, {x,0,0}, {}, Joint::hinge, 2*i});
        s.bodies.push_back({"follower", 0, {x,height,0}, {0,1,0}, Joint::slide, 2*i+1});
        s.contacts.push_back({2*i+1,2*i,contact_skin,stiffness});
        s.mass.insert(s.mass.end(), {0.007, 0.48 + 0.035 * i});
        s.q.insert(s.q.end(), {angle, 0});
        s.v.insert(s.v.end(), {speed, 0.024 * speed * std::cos(angle)});
        s.load.insert(s.load.end(), {0, -3.2});
        s.springs.push_back({2*i, {}, 1.8, 0.10, angle, speed});
        s.springs.push_back({2*i+1, {}, 70, 0.65, 0, 0});
        if (i) {
            s.springs.push_back({2*i, 2*(i-1), 0.30, 0.016, 0.35, 0});
            s.springs.push_back({2*i+1, 2*(i-1)+1, 95, 0.45, 0, 0});
        }
    }
    return s;
}

Scene train(const std::filesystem::path& models, unsigned facets) {
    Scene s;
    s.name = "ball-train";
    s.geometry.push_back(load_geometry(models/"ball.step", facets));
    s.geometry.push_back(load_geometry(models/"left-stop.step", facets));
    s.geometry.push_back(load_geometry(models/"right-stop.step", facets));
    constexpr std::size_t count = 12;
    constexpr double spacing = 0.120;
    constexpr double end_gap = 0.065;
    s.bodies.push_back({"left stop", 1, {-end_gap,0,0}});
    for (std::size_t i = 0; i < count; ++i) {
        s.bodies.push_back({"slider ball",0,{static_cast<double>(i)*spacing,0,0},{1,0,0},Joint::slide,i});
        s.mass.push_back(0.65 + 0.035 * i);
        s.q.push_back(0);
        s.v.push_back(0);
        s.load.push_back(0);
        s.springs.push_back({i,{},0,0.24,0,0});
        for (std::size_t earlier = 0; earlier < i; ++earlier) {
            s.contacts.push_back({earlier+1,i+1,contact_skin,stiffness});
        }
        s.contacts.push_back({0,i+1,contact_skin,stiffness});
    }
    s.bodies.push_back({"right stop",2,{static_cast<double>(count-1)*spacing+end_gap,0,0}});
    for (std::size_t i = 0; i < count; ++i) s.contacts.push_back({count+1,i+1,contact_skin,stiffness});
    return s;
}

BoundedPose bounded_pose(const Body& body, const Vector& q) {
    const IVec anchor = exact(body.anchor);
    if (body.joint == Joint::slide) return {anchor + exact(body.direction) * q[body.dof], 0};
    if (body.joint == Joint::hinge) return {anchor, q[body.dof]};
    return {anchor, 0};
}

double travel(const Body& body, const Geometry& g, const Vector& start, const Vector& end) {
    if (body.joint == Joint::fixed) return 0;
    const double delta = magnitude(Interval{end[body.dof]} - start[body.dof]);
    if (body.joint == Joint::hinge) return (Interval{delta} * g.motion_radius).hi;
    return (Interval{delta} * norm(exact(body.direction))).hi;
}

double gradient(const Body& body, Vec witness, Vec n) {
    if (body.joint == Joint::slide) return dot3(n, body.direction);
    if (body.joint == Joint::hinge) {
        return dot3(n, cross3(Vec{0,0,1}, sub(witness, body.anchor)));
    }
    return 0;
}

void spring_tangent(Vector& matrix, const Spring& spring, double x, std::size_t n) {
    const auto i = spring.a;
    matrix[i*n+i] += x;
    if (spring.b) {
        const auto j = *spring.b;
        matrix[j*n+j] += x;
        matrix[i*n+j] -= x;
        matrix[j*n+i] -= x;
    }
}

std::string_view joint_name(Joint joint) {
    if (joint == Joint::slide) return "slide";
    if (joint == Joint::hinge) return "hinge";
    return "fixed";
}

std::string_view shape_name(Shape kind) {
    if (kind == Shape::sphere) return "sphere";
    if (kind == Shape::cylinder) return "cylinder";
    return "plate";
}

void vector_json(std::ostream& out, Vec v) {
    out << '[' << v[0] << ',' << v[1] << ',' << v[2] << ']';
}

}  // namespace

Scene make_scene(std::string_view name, const std::filesystem::path& models, unsigned facets) {
    if (name == "drop") return drop(models, facets);
    if (name == "cam-array") return cams(models, facets);
    if (name == "ball-train") return train(models, facets);
    throw std::invalid_argument("unknown HHT scene");
}

Pose pose(const Body& body, const Vector& q) {
    if (body.joint == Joint::slide) return {add(body.anchor, scale(body.direction, q[body.dof])),1,0};
    if (body.joint == Joint::hinge) return {body.anchor,std::cos(q[body.dof]),std::sin(q[body.dof])};
    return {body.anchor,1,0};
}

Vector ClearanceGuard::endpoints(const Vector& q) {
    ++evaluations_;
    std::vector<BoundedPose> poses;
    poses.reserve(scene_.bodies.size());
    for (const Body& body : scene_.bodies) poses.push_back(bounded_pose(body,q));
    Vector result;
    result.reserve(scene_.contacts.size());
    for (const Contact& pair : scene_.contacts) {
        const auto& a = scene_.geometry[scene_.bodies[pair.a].geometry];
        const auto& b = scene_.geometry[scene_.bodies[pair.b].geometry];
        result.push_back(safe_gap(a, poses[pair.a], b, poses[pair.b]));
    }
    return result;
}

double ClearanceGuard::check(const Vector& start, const Vector& end) {
    Vector start_storage;
    const Vector* first = &last_gaps_;
    if (!cached_ || last_q_ != start) {
        start_storage = endpoints(start);
        first = &start_storage;
    }
    Vector last = endpoints(end);
    std::vector<double> moves;
    moves.reserve(scene_.bodies.size());
    for (const Body& body : scene_.bodies) {
        moves.push_back(travel(body, scene_.geometry[body.geometry], start, end));
    }
    double lower = infinity;
    for (std::size_t i = 0; i < scene_.contacts.size(); ++i) {
        const Contact& pair = scene_.contacts[i];
        const Interval displacement = (Interval{moves[pair.a]} + moves[pair.b]) / 2;
        lower = std::min(lower, (Interval{std::min((*first)[i],last[i])} - displacement).lo);
    }
    if (cached_) {
        last_q_ = end;
        last_gaps_ = std::move(last);
    }
    return lower;
}

Dynamics::Dynamics(const Scene& scene, Backend backend, bool warm)
    : scene_(scene), backend_(backend), warm_(warm), hints_(scene.contacts.size()) {}

Forces Dynamics::forces(const Vector& q, const Vector& v, double time) {
    const std::size_t n = q.size();
    Forces result{scene_.load, Vector(n*n), Vector(n*n)};
    peak_force_ = 0;
    if (scene_.name == "ball-train") {
        result.value[0] += 5.5 * std::sin(7.5*time) + 2.5 * std::sin(18.0*time);
    }
    for (const Spring& spring : scene_.springs) {
        const auto i = spring.a;
        double stretch = q[i] - spring.rest - spring.speed*time;
        double rate = v[i] - spring.speed;
        if (spring.b) {
            stretch -= q[*spring.b];
            rate -= v[*spring.b];
        }
        const double force = -spring.stiffness*stretch - spring.damping*rate;
        result.value[i] += force;
        if (spring.b) result.value[*spring.b] -= force;
        spring_tangent(result.stiffness, spring, spring.stiffness, n);
        spring_tangent(result.damping, spring, spring.damping, n);
    }

    std::vector<Pose> positions;
    positions.reserve(scene_.bodies.size());
    for (const Body& body : scene_.bodies) positions.push_back(pose(body, q));
    for (std::size_t k = 0; k < scene_.contacts.size(); ++k) {
        const Contact& pair = scene_.contacts[k];
        const Body& a = scene_.bodies[pair.a];
        const Body& b = scene_.bodies[pair.b];
        const auto result_pair = closest(scene_.geometry[a.geometry], positions[pair.a],
            scene_.geometry[b.geometry], positions[pair.b], backend_, pair.skin,
            warm_ ? &hints_[k] : nullptr, statistics_);
        if (!result_pair || result_pair->distance >= pair.skin) continue;
        const Vec difference = sub(result_pair->b, result_pair->a);
        const double separation = length(difference);
        if (!(separation > 1e-12)) {
            result.value[0] = std::numeric_limits<double>::quiet_NaN();
            return result;
        }
        const Vec normal = scale(difference, 1/separation);
        const double force = pair.stiffness*(pair.skin - result_pair->distance);
        peak_force_ = std::max(peak_force_,force);
        const std::array<std::pair<std::size_t,double>,2> jacobian{{
            {a.dof,-gradient(a,result_pair->a,normal)}, {b.dof,gradient(b,result_pair->b,normal)}
        }};
        for (const auto& [i, ji] : jacobian) {
            if (ji == 0) continue;
            result.value[i] += force*ji;
            for (const auto& [j, jj] : jacobian) result.stiffness[i*n+j] += pair.stiffness*ji*jj;
        }
    }
    return result;
}

Diagnostics diagnostics(const Scene& scene, const State& state) {
    Diagnostics out{infinity,0,0,0};
    for (std::size_t i = 0; i < scene.mass.size(); ++i) {
        out.stored_energy += 0.5 * scene.mass[i] * state.v[i] * state.v[i] - scene.load[i] * state.q[i];
    }
    for (const Spring& spring : scene.springs) {
        const double stretch = state.q[spring.a] - spring.rest - spring.speed*state.time -
                              (spring.b ? state.q[*spring.b] : 0);
        out.stored_energy += 0.5*spring.stiffness*stretch*stretch;
    }
    std::vector<BoundedPose> positions;
    positions.reserve(scene.bodies.size());
    for (const Body& body : scene.bodies) positions.push_back(bounded_pose(body, state.q));
    for (const Contact& c : scene.contacts) {
        const auto& a = scene.geometry[scene.bodies[c.a].geometry];
        const auto& b = scene.geometry[scene.bodies[c.b].geometry];
        const double gap = safe_gap(a,positions[c.a],b,positions[c.b]);
        out.cad_gap = std::min(out.cad_gap,gap);
        const double depth = std::max(0.0,c.skin-gap);
        out.cad_peak_force = std::max(out.cad_peak_force,c.stiffness*depth);
        out.stored_energy += 0.5*c.stiffness*depth*depth;
        out.contacts += depth > 0;
    }
    return out;
}

void write_description(std::ostream& out, const Scene& scene) {
    out << std::setprecision(17) << "{\n\"scene\":\"" << scene.name << "\",\n\"geometry\":[\n";
    for (std::size_t i = 0; i < scene.geometry.size(); ++i) {
        const Geometry& g = scene.geometry[i];
        if (i) out << ",\n";
        out << "{\"shape\":\"" << shape_name(g.kind) << "\",\"center\":";
        vector_json(out,g.center);
        out << ",\"u\":"; vector_json(out,g.u);
        out << ",\"v\":"; vector_json(out,g.v);
        out << ",\"axis\":"; vector_json(out,g.axis);
        out << ",\"radius\":" << g.radius << ",\"half\":" << g.half
            << ",\"half_u\":" << g.half_u << ",\"half_v\":" << g.half_v << '}';
    }
    out << "\n],\n\"bodies\":[\n";
    for (std::size_t i = 0; i < scene.bodies.size(); ++i) {
        const Body& body = scene.bodies[i];
        if (i) out << ",\n";
        out << "{\"name\":\"" << body.name << "\",\"geometry\":" << body.geometry
            << ",\"joint\":\"" << joint_name(body.joint) << "\",\"dof\":" << body.dof << ",\"anchor\":";
        vector_json(out,body.anchor);
        out << ",\"direction\":"; vector_json(out,body.direction);
        out << '}';
    }
    out << "\n],\n\"contacts\":[";
    for (std::size_t k = 0; k < scene.contacts.size(); ++k) {
        if (k) out << ',';
        out << '[' << scene.contacts[k].a << ',' << scene.contacts[k].b << ']';
    }
    out << "]\n}\n";
}

}  // namespace nc::hht_bench
