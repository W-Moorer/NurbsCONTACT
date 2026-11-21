#include "nurbscontact/hht.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace nc::dynamics {
namespace {

bool finite(const Vector& x) {
    return std::all_of(x.begin(), x.end(), [](double value) { return std::isfinite(value); });
}

double max_norm(const Vector& x) {
    double result = 0;
    for (double value : x) result = std::max(result, std::abs(value));
    return result;
}

void validate(const Forces& f, std::size_t n) {
    if (f.value.size() != n || f.stiffness.size() != n * n || f.damping.size() != n * n) {
        throw std::invalid_argument("force and tangent sizes must agree with mass");
    }
}

bool solve_positive_definite(Vector& matrix, Vector& rhs) {
    const std::size_t n = rhs.size();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double x = matrix[i * n + j];
            for (std::size_t k = 0; k < j; ++k) x -= matrix[i * n + k] * matrix[j * n + k];
            if (i == j) {
                if (!(x > 0) || !std::isfinite(x)) return false;
                matrix[i * n + i] = std::sqrt(x);
            } else {
                matrix[i * n + j] = x / matrix[j * n + j];
            }
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < i; ++j) rhs[i] -= matrix[i * n + j] * rhs[j];
        rhs[i] /= matrix[i * n + i];
    }
    for (std::size_t i = n; i-- > 0;) {
        for (std::size_t j = i + 1; j < n; ++j) rhs[i] -= matrix[j * n + i] * rhs[j];
        rhs[i] /= matrix[i * n + i];
    }
    return finite(rhs);
}

struct Trial {
    State state;
    Forces forces;
    Vector residual;
    double error{};
};

}  // namespace

Hht::Hht(Vector diagonal_mass, ForceFunction forces, HhtOptions options, ClearanceFunction clearance)
    : mass_(std::move(diagonal_mass)), forces_(std::move(forces)),
      options_(options), clearance_(std::move(clearance)) {
    if (!forces_ || mass_.empty() || !finite(mass_) ||
        std::any_of(mass_.begin(), mass_.end(), [](double m) { return m <= 0; }) ||
        !(options.alpha >= -1.0 / 3 && options.alpha <= 0) ||
        !(std::isfinite(options.absolute_tolerance) && options.absolute_tolerance > 0) ||
        !(std::isfinite(options.relative_tolerance) && options.relative_tolerance >= 0) ||
        !std::isfinite(options.minimum_clearance) || !options.max_iterations || options.max_subdivisions > 20) {
        throw std::invalid_argument("invalid HHT mass, force or integration options");
    }
}

State Hht::initialize(Vector q, Vector v, double time) const {
    const std::size_t n = mass_.size();
    if (q.size() != n || v.size() != n || !finite(q) || !finite(v) || !std::isfinite(time)) {
        throw std::invalid_argument("invalid initial HHT state");
    }
    if (clearance_ && !(clearance_(q, q) > options_.minimum_clearance)) {
        throw std::invalid_argument("initial state fails the clearance guard");
    }
    Forces f = forces_(q, v, time);
    validate(f, n);
    if (!finite(f.value)) throw std::invalid_argument("nonfinite initial force");
    Vector acceleration(n);
    for (std::size_t i = 0; i < n; ++i) acceleration[i] = f.value[i] / mass_[i];
    return {time, std::move(q), std::move(v), std::move(acceleration), std::move(f.value)};
}

bool Hht::solve(const State& start, double dt, State& result, HhtStats& statistics) const {
    const std::size_t n = mass_.size();
    const double alpha = options_.alpha;
    const double beta = 0.25 * (1 - alpha) * (1 - alpha);
    const double gamma = 0.5 - alpha;
    const double position_scale = beta * dt * dt;
    const double velocity_scale = gamma * dt;
    const double residual_scale = 1 + alpha;
    Vector q_predict(n);
    Vector v_predict(n);
    for (std::size_t i = 0; i < n; ++i) {
        q_predict[i] = start.q[i] + dt * start.v[i] + dt * dt * (0.5 - beta) * start.a[i];
        v_predict[i] = start.v[i] + dt * (1 - gamma) * start.a[i];
    }

    const auto evaluate = [&](Vector acceleration) {
        Trial trial;
        trial.state = {start.time + dt, Vector(n), Vector(n), std::move(acceleration), {}};
        for (std::size_t i = 0; i < n; ++i) {
            trial.state.q[i] = q_predict[i] + position_scale * trial.state.a[i];
            trial.state.v[i] = v_predict[i] + velocity_scale * trial.state.a[i];
        }
        trial.forces = forces_(trial.state.q, trial.state.v, trial.state.time);
        ++statistics.evaluations;
        validate(trial.forces, n);
        trial.residual.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            trial.residual[i] = mass_[i] * trial.state.a[i] - residual_scale * trial.forces.value[i] +
                                alpha * start.force[i];
        }
        trial.error = finite(trial.residual) ? max_norm(trial.residual) : std::numeric_limits<double>::infinity();
        return trial;
    };

    Trial current = evaluate(start.a);
    for (unsigned iteration = 0; iteration <= options_.max_iterations; ++iteration) {
        const double tolerance = options_.absolute_tolerance + options_.relative_tolerance *
            std::max(max_norm(current.forces.value), max_norm(start.force));
        if (current.error <= tolerance && std::isfinite(current.error)) {
            statistics.maximum_residual = std::max(statistics.maximum_residual, current.error);
            current.state.force = std::move(current.forces.value);
            result = std::move(current.state);
            return true;
        }
        if (iteration == options_.max_iterations) break;
        Vector tangent(n * n);
        Vector correction(n);
        for (std::size_t i = 0; i < n * n; ++i) {
            tangent[i] = residual_scale * (position_scale * current.forces.stiffness[i] +
                                          velocity_scale * current.forces.damping[i]);
        }
        for (std::size_t i = 0; i < n; ++i) {
            tangent[i * n + i] += mass_[i];
            correction[i] = -current.residual[i];
        }
        if (!solve_positive_definite(tangent, correction)) return false;
        ++statistics.newton_iterations;
        bool improved = false;
        for (double fraction = 1; fraction >= 1.0 / 128; fraction *= 0.5) {
            Vector acceleration = current.state.a;
            for (std::size_t i = 0; i < n; ++i) acceleration[i] += fraction * correction[i];
            Trial next = evaluate(std::move(acceleration));
            if (next.error < current.error) {
                current = std::move(next);
                improved = true;
                break;
            }
            ++statistics.rejected_trials;
        }
        if (!improved) return false;
    }
    return false;
}

bool Hht::advance(State& state, double dt, HhtStats& statistics, unsigned depth) const {
    State candidate;
    if (solve(state, dt, candidate, statistics)) {
        const double lower = clearance_ ? clearance_(state.q, candidate.q) : std::numeric_limits<double>::infinity();
        if (lower > options_.minimum_clearance) {
            state = std::move(candidate);
            statistics.minimum_clearance = std::min(statistics.minimum_clearance, lower);
            ++statistics.accepted_steps;
            return true;
        }
    }
    if (depth >= options_.max_subdivisions) return false;
    ++statistics.subdivisions;
    const double half = dt / 2;
    return advance(state, half, statistics, depth + 1) && advance(state, half, statistics, depth + 1);
}

void Hht::step(State& state, double dt, HhtStats& statistics) const {
    const std::size_t n = mass_.size();
    if (!(dt > 0) || !std::isfinite(dt) || state.time + dt == state.time || state.q.size() != n ||
        state.v.size() != n || state.a.size() != n || state.force.size() != n) {
        throw std::invalid_argument("invalid HHT time step or state");
    }
    State candidate = state;
    HhtStats attempted = statistics;
    if (!advance(candidate, dt, attempted, 0)) {
        throw std::runtime_error("HHT residual or clearance guard did not converge within the subdivision budget at t=" +
                                 std::to_string(state.time));
    }
    state = std::move(candidate);
    statistics = attempted;
}

}  // namespace nc::dynamics
