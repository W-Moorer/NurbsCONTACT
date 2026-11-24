#pragma once

#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <vector>

namespace nc::dynamics {

using Vector = std::vector<double>;

struct Forces {
    Vector value;
    // Row-major approximations of -dF/dq and -dF/dv.
    Vector stiffness;
    Vector damping;
    // Proposed internal state for this force evaluation.
    Vector history{};
    // Infinity-norm error bound on value, for bounded force evaluations.
    double error_bound{};
};

using ForceFunction = std::function<Forces(const Vector&, const Vector&, double)>;
// The final argument is the history at the start of the attempted step.
using HistoryForceFunction = std::function<Forces(const Vector&, const Vector&, double, const Vector&)>;
// Receives the step-start history and the requested force error bound. Zero
// requests the tightest available evaluation. Return the actual error bound.
using BoundedForceFunction = std::function<Forces(const Vector&, const Vector&, double, const Vector&, double)>;
using ClearanceFunction = std::function<double(const Vector&, const Vector&)>;

struct HhtOptions {
    double alpha{-0.10};
    double absolute_tolerance{1e-8};
    double relative_tolerance{1e-8};
    double minimum_clearance{};
    unsigned max_iterations{18};
    unsigned max_subdivisions{8};
    bool nonsymmetric_tangent{};
    // Generalized-alpha spectral radius at infinite frequency. When unset,
    // retain the HHT force weighting and its unweighted mass.
    std::optional<double> spectral_radius{};
};

struct State {
    double time{};
    Vector q;
    Vector v;
    Vector a;
    Vector force;
    Vector history{};
    double force_error{};
};

// Return no value to accept, or the split fraction for a converged candidate.
using StepAcceptanceFunction = std::function<std::optional<double>(const State&, const State&)>;

struct HhtStats {
    std::size_t accepted_steps{};
    std::size_t subdivisions{};
    std::size_t evaluations{};
    std::size_t newton_iterations{};
    std::size_t rejected_trials{};
    double maximum_residual{};
    double minimum_clearance{std::numeric_limits<double>::infinity()};
    std::size_t force_refinements{};
    std::size_t step_checks{};
    std::size_t rejected_steps{};
};

class Hht {
public:
    Hht(Vector diagonal_mass, ForceFunction forces, HhtOptions options = {}, ClearanceFunction clearance = {},
        HistoryForceFunction historical_forces = {}, BoundedForceFunction bounded_forces = {},
        StepAcceptanceFunction accept_step = {}, std::function<void()> begin_trial = {});

    State initialize(Vector q, Vector v, double time = 0) const;
    void step(State& state, double dt, HhtStats& statistics) const;

private:
    bool advance(State& state, double dt, HhtStats& statistics, unsigned depth) const;
    bool solve(const State& start, double dt, State& result, HhtStats& statistics) const;

    Vector mass_;
    ForceFunction forces_;
    HhtOptions options_;
    ClearanceFunction clearance_;
    HistoryForceFunction historical_forces_;
    BoundedForceFunction bounded_forces_;
    StepAcceptanceFunction accept_step_;
    std::function<void()> begin_trial_;
};

}  // namespace nc::dynamics
