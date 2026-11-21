#pragma once

#include "nurbscontact/geometry.hpp"

namespace nc::detail {

struct AxialEnvelope {
    IVec origin;
    IVec axis;
    Interval radial;
    Interval axial;
};

std::optional<AxialEnvelope> axial_envelope(const FacePatch& patch, const TrimDomain& trim);

// b is B's envelope at time zero, with its orientation already transformed.
std::optional<Interval> axial_window(
    const AxialEnvelope& a, const AxialEnvelope& b, IVec velocity, Interval time, double clearance
);

}  // namespace nc::detail
