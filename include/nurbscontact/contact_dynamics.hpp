#pragma once

#include "nurbscontact/hht.hpp"
#include "nurbscontact/triangle_bvh.hpp"

#include <memory>
#include <span>
#include <stdexcept>

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
    std::size_t closest_queries{};
    std::size_t unresolved_queries{};
    std::size_t native_nodes{};
    std::optional<Interval> last_unresolved_distance;
    std::size_t boolean_queries{}, boolean_input_faces{}, local_solid_uses{}, local_solid_builds{};
    double mesh_detection_s{}, mesh_placement_s{}, mesh_boolean_s{};
    std::size_t cad_boolean_queries{};
    std::size_t cad_region_cache_hits{};
};

struct ContactLineSample {
    PointPair points; // Coordinates in a shared frame, in meters.
    double weight_m{};
};

struct ContactWrench {
    Vec force{}, moment{};
};

struct ContactPairWrenches {
    ContactWrench a, b;
};

// Linear, frictionless normal stiffness per meter of contact line, in N/m².
ContactPairWrenches line_contact_wrenches(std::span<const ContactLineSample> samples, double activation_m,
                                         double stiffness_per_m, Vec center_a, Vec center_b);

class ContactSurface {
public:
    virtual ~ContactSurface() = default;
    virtual TriangleBvh::Box bounds() const = 0;
    virtual std::optional<RayHit> first_hit(Vec origin, Vec direction, double maximum,
                                            ContactWork& work) const = 0;
};

enum class PairStatus { separated, within_cutoff, unresolved };

struct PairProximity {
    PairStatus status{PairStatus::unresolved};
    PointPair points{}; // World-space points and distance in meters.
    std::optional<Interval> distance_bound_m{};
};

class PairContactQuery {
public:
    virtual ~PairContactQuery() = default;
    virtual PairProximity closest(ContactPose a, ContactPose b, double cutoff, ContactWork& work) const = 0;
};

struct ContactRegion {
    Vec point, normal; // Common world-space point, unit normal from A toward B.
    double depth_m;
};

class UnresolvedContact : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct SurfaceTransition {
    std::optional<double> fraction;
    bool start_in_contact{};
    bool end_in_contact{};
    std::size_t nodes{};
    std::size_t cad_distances{}, chen_queries{}, chen_confirmed{};
    std::size_t tdib_cells{}, tdib_candidates{}, tdib_unresolved{};
    std::size_t cad_box_tests{}, cad_extrema{}, cad_classifications{};
    std::size_t cad_distance_cache_hits{};
    std::size_t cad_witness_tests{}, cad_witness_improvements{};
};

class PairRegionQuery {
public:
    virtual ~PairRegionQuery() = default;
    virtual std::vector<ContactRegion> regions(ContactPose a, ContactPose b, ContactWork& work) const = 0;
    // A sweep from a separated endpoint locates an entry or a release.
    virtual SurfaceTransition transition(ContactPose a0,ContactPose b0,ContactPose a1,ContactPose b1,
                                         double tolerance_m) const = 0;
    virtual SurfaceTransition transition(ContactPose a0,ContactPose b0,ContactPose a1,ContactPose b1,
                                         double tolerance_m,double /*fraction_tolerance*/) const {
        return transition(a0,b0,a1,b1,tolerance_m);
    }
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
    double activation{};
    std::shared_ptr<const PairContactQuery> closest{};
    std::shared_ptr<const PairRegionQuery> regions{};
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
