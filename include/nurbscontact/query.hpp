#pragma once

#include "nurbscontact/geometry.hpp"

#include <memory>

namespace nc {

// Coordinates of body B in body A's frame. The axis is normalized internally.
struct Motion {
    Vec translation{};
    Vec velocity{};
    Vec axis{0, 0, 1};
    double angle{};
    double angular_velocity{};
};

struct Options {
    double distance_tolerance{1e-5};
    double time_tolerance{1e-4};
    std::size_t max_nodes{100'000};
    unsigned max_spatial_depth{44};
    bool directional_time{true};
    bool chord_box{false};
    bool inherit_time{true};
    bool eager_trim{true};
    bool axial_time{true};
};

struct Stats {
    std::size_t visited{};
    std::size_t spatial_splits{};
    std::size_t time_splits{};
    std::size_t trim_rejected{};
    std::size_t direction_rejected{};
    std::size_t direction_windows{};
    std::size_t axial_rejected{};
    std::size_t axial_clipped{};
};

struct Witness {
    Sample a;
    Sample b;  // World position, UV and feature from B's local chart.
    double time{};
};

struct DistanceResult {
    double lower{};
    double upper{infinity};
    std::optional<Witness> witness;
    Stats statistics;
};

enum class SweepStatus { clear, hit, unresolved };

struct SweepResult {
    SweepStatus status{SweepStatus::unresolved};
    // [safe_prefix, witness_time] brackets the first event when status is hit.
    // With no witness, safe_prefix still excludes every earlier time.
    double safe_prefix{};
    std::optional<double> witness_time;
    std::optional<Witness> witness;
    Stats statistics;
};

class PreparedModel {
public:
    explicit PreparedModel(Model model);
    struct Data;

private:
    std::shared_ptr<const Data> data_;
    friend DistanceResult distance(const PreparedModel&, const PreparedModel&, const Motion&, double, const Options&);
    friend SweepResult sweep(const PreparedModel&, const PreparedModel&, const Motion&, Interval, double, const Options&);
};

DistanceResult distance(
    const PreparedModel& a, const PreparedModel& b, const Motion& motion = {},
    double time = 0, const Options& options = {}
);
SweepResult sweep(
    const PreparedModel& a, const PreparedModel& b, const Motion& motion,
    Interval time, double clearance, const Options& options = {}
);

}  // namespace nc
