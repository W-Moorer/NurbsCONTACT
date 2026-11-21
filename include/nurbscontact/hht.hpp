#pragma once

#include <cstddef>
#include <functional>
#include <limits>
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

struct HhtStats {
    std::size_t accepted_steps{};
    std::size_t subdivisions{};
    std::size_t evaluations{};
    std::size_t newton_iterations{};
    std::size_t rejected_trials{};
    double maximum_residual{};
    double minimum_clearance{std::numeric_limits<double>::infinity()};
    std::size_t force_refinements{};
};

class Hht {
public:
    Hht(Vector diagonal_mass, ForceFunction forces, HhtOptions options = {}, ClearanceFunction clearance = {},
        HistoryForceFunction historical_forces = {}, BoundedForceFunction bounded_forces = {});

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
};

}  // namespace nc::dynamics
