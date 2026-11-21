#pragma once

#include "nurbscontact/hht.hpp"
#include "nurbscontact/triangle_bvh.hpp"

#include <memory>
#include <span>

namespace nc::dynamics {

// Rigid kinematics in SI units. Twists are world-space derivatives per unit q.
struct ContactPose {
    Vec origin{};
    Vec axis{0, 0, 1};
    double angle{};
    Vec rotate(Vec local) const;
    Vec inverse_rotate(Vec world) const;
    Vec point(Vec local) const { return add(origin, rotate(local)); }
};

struct Twist {
    Vec linear{};
    Vec angular{};
    Vec at(Vec lever) const { return add(linear, cross3(angular, lever)); }
};

struct RayHit {
    double distance;
    Vec normal;
};

struct ContactWork {
    std::size_t pairs{};
    std::size_t culled{};
    std::size_t rays{};
    std::size_t missing{};
    TriangleStatistics mesh{};
};

class ContactSurface {
public:
    virtual ~ContactSurface() = default;
    virtual TriangleBvh::Box bounds() const = 0;
    virtual std::optional<RayHit> first_hit(Vec origin, Vec direction, double maximum,
                                            ContactWork& work) const = 0;
};

struct ContactBodyState {
    const ContactSurface* surface;
    ContactPose pose;
    std::vector<Twist> derivative;
    Twist prescribed{};
};

struct ContactMaterial {
    double stiffness;
    double exponent;
    double rate; // coefficient of depth^((exponent-1)/2) times closing speed
    double friction;
    double slip_speed;
    double maximum_depth;
    double impact_dissipation{}; // dimensionless multiplier of closing speed / incidence speed
    double approach_floor{};     // positive for an incidence-dependent law
};

enum class ContactDirection { world, body_a, anchors, contained, contained_radial };

struct ContactPair {
    std::size_t a, b;
    Vec anchor_a, anchor_b;
    ContactDirection direction;
    Vec normal;
    ContactMaterial material;
};

struct ContactSample {
    double gap{std::numeric_limits<double>::quiet_NaN()};
    double normal_force{};
    Vec friction{};
    Vec witness_a{}, witness_b{};
    bool resolved{};
    double gap_rate{std::numeric_limits<double>::quiet_NaN()};
    double impact_speed{};
};

struct ImpactHistory {
    double gap{std::numeric_limits<double>::quiet_NaN()};
    double gap_rate{std::numeric_limits<double>::quiet_NaN()};
    double impact_speed{};
};

// Sliding friction with a prescribed preload. The normal is in world space.
struct PointResistance {
    std::size_t a, b;
    Vec anchor_a, anchor_b, normal;
    double force_limit, slip_speed;
};

void add_point_resistances(Forces& forces, std::span<const ContactBodyState> bodies,
                           std::span<const PointResistance> resistances, const Vector& velocity);

// Translational Kelvin–Voigt element with principal axes fixed to body A.
struct TranslationalBushing {
    std::size_t a, b;
    Vec anchor_a, anchor_b;
    Vec rest, stiffness, damping;
};

void add_translational_bushings(Forces& forces, std::span<const ContactBodyState> bodies,
                                std::span<const TranslationalBushing> bushings, const Vector& velocity);

// Exterior probes treat an absent hit as open. A contained probe starts at
// the cavity and part anchors and rejects a missing wall or part. The radial
// mode uses the centerline for the force differential of circular sections.
bool add_contact_forces(Forces* forces, std::span<const ContactBodyState> bodies,
                        std::span<const ContactPair> pairs, const Vector& velocity,
                        ContactWork& work, std::vector<ContactSample>* observations = nullptr,
                        std::span<const ImpactHistory> previous = {});

} // namespace nc::dynamics
