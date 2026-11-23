#pragma once

#include "nurbscontact/contact_dynamics.hpp"
#include <string_view>

namespace nc::dynamics {

enum class ImpulseLaw { sap, lcp, ncp, soccp, lagrange, augmented_lagrange };
ImpulseLaw parse_impulse_law(std::string_view name);
std::string_view impulse_law_name(ImpulseLaw law);
using InverseMass = std::function<Vector(const Vector&)>;

struct SurfaceImpulseResult {
    Vector velocity_change;
    std::vector<ContactSample> contacts;
    std::size_t regions{}, iterations{};
    double residual_m_s{};
};

// Geometry and twists are evaluated at the predicted end of a free-motion step.
// InverseMass maps generalized impulses to velocity increments at that pose.
SurfaceImpulseResult surface_impulses(ImpulseLaw law, std::span<const ContactBodyState> bodies,
    std::span<const ContactPair> pairs, const Vector& free_velocity, double step,
    const InverseMass& inverse_mass, ContactWork& work, std::span<const ContactSample> previous = {});

} // namespace nc::dynamics
