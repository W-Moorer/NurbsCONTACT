#pragma once

#include "contact.hpp"
#include "nurbscontact/hht.hpp"

#include <iosfwd>

namespace nc::hht_bench {

enum class Joint { fixed, slide, hinge };

struct Body {
    std::string_view name;
    std::size_t geometry{};
    Vec anchor{};
    Vec direction{};
    Joint joint{Joint::fixed};
    std::size_t dof{};
};

struct Contact {
    std::size_t a{};
    std::size_t b{};
    double skin{};
    double stiffness{};
};

struct Spring {
    std::size_t a{};
    std::optional<std::size_t> b;
    double stiffness{};
    double damping{};
    double rest{};
    double speed{};
};

struct Scene {
    std::string_view name;
    std::vector<Geometry> geometry;
    std::vector<Body> bodies;
    std::vector<Contact> contacts;
    std::vector<Spring> springs;
    dynamics::Vector mass;
    dynamics::Vector q;
    dynamics::Vector v;
    dynamics::Vector load;
};

Scene make_scene(std::string_view name, const std::filesystem::path& models, unsigned facets);
Pose pose(const Body& body, const dynamics::Vector& q);
void write_description(std::ostream& out, const Scene& scene);

class ClearanceGuard {
public:
    ClearanceGuard(const Scene& scene, bool cached) : scene_(scene), cached_(cached) {}
    double check(const dynamics::Vector& a, const dynamics::Vector& b);
    std::size_t endpoint_evaluations() const { return evaluations_; }

private:
    dynamics::Vector endpoints(const dynamics::Vector& q);

    const Scene& scene_;
    bool cached_;
    dynamics::Vector last_q_;
    dynamics::Vector last_gaps_;
    std::size_t evaluations_{};
};

class Dynamics {
public:
    Dynamics(const Scene& scene, Backend backend, bool warm);
    dynamics::Forces forces(const dynamics::Vector& q, const dynamics::Vector& v, double time);
    const Statistics& statistics() const { return statistics_; }
    double peak_force() const { return peak_force_; }

private:
    const Scene& scene_;
    Backend backend_;
    bool warm_;
    std::vector<TriangleHint> hints_;
    Statistics statistics_;
    double peak_force_{};
};

struct Diagnostics {
    double cad_gap{};
    double cad_peak_force{};
    double stored_energy{};
    unsigned contacts{};
};

Diagnostics diagnostics(const Scene& scene, const dynamics::State& state);

}  // namespace nc::hht_bench
