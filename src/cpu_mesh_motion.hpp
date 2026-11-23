#pragma once

#include "nurbscontact/cpu_mesh.hpp"

#include <Eigen/Geometry>

namespace nc::cpu {

// Translations interpolate linearly and orientations follow the shortest quaternion arc.
// A driver supplies additional key frames for rotations greater than pi.
struct MeshMotion {
    Eigen::Vector3d from, to;
    Eigen::Quaterniond q0, q1;
    double angle;

    MeshMotion(const Eigen::Isometry3d& start, const Eigen::Isometry3d& end)
        : from(start.translation()), to(end.translation()), q0(start.linear()), q1(end.linear()),
          angle(q0.angularDistance(q1)) {}

    Eigen::Isometry3d at(double t) const {
        Eigen::Isometry3d pose=Eigen::Isometry3d::Identity();
        pose.linear()=q0.slerp(t,q1).normalized().toRotationMatrix();
        pose.translation()=(1-t)*from+t*to;
        return pose;
    }
    double chord_error(double radius, double time_width) const {
        const double s=std::sin(angle*time_width*.25);
        return nc::up(2*radius*s*s);
    }
};
}
