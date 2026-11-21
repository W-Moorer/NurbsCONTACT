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
};

struct ContactMaterial {
    double stiffness;
    double exponent;
    double rate; // coefficient of depth^((exponent-1)/2) times closing speed
    double friction;
    double slip_speed;
    double maximum_depth;
};

enum class ContactDirection { world, body_a, anchors };

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
};

// Each probe finds the two facing trimmed surfaces independently. An absent
// hit is an open probe. The caller can put multiple probes on a component.
bool add_contact_forces(Forces* forces, std::span<const ContactBodyState> bodies,
                        std::span<const ContactPair> pairs, const Vector& velocity,
                        ContactWork& work, std::vector<ContactSample>* observations = nullptr);

} // namespace nc::dynamics
