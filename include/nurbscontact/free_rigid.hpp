#pragma once

#include "nurbscontact/contact_dynamics.hpp"

namespace nc::dynamics {

using Quaternion = std::array<double,4>; // scalar first, rotates local to world

// Six physical degrees of freedom. The seven integration coordinates store
// the world center of mass and a homogeneous quaternion.
struct FreeRigidBody {
    std::size_t first;
    std::array<Vec,3> inertia, inverse_inertia; // local, about the center of mass
    FreeRigidBody(std::size_t first_coordinate,Vec diagonal,Vec products_xy_xz_yz);
    Quaternion initial_attitude(const ContactPose& pose) const;
    Quaternion initial_rate(const Quaternion& attitude,Vec body_omega) const;
    void angular_impulse_response(const Vector& q,const Vector& generalized,Vector& delta_velocity) const;
    void kinematics(const Vector& q,Vec local_com,ContactBodyState& state) const;
    void apply_euler_equations(const Vector& q,const Vector& v,Forces& forces) const;
    Vec body_angular_velocity(const Vector& q,const Vector& v) const;
    Vec body_angular_acceleration(const State& state) const;
    void normalize(State& state) const;
};

} // namespace nc::dynamics
