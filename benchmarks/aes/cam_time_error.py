"""Independent fixed-step HHT check using the analytic cam normal and moment arm."""

import argparse
import json
from pathlib import Path

import numpy as np
from numba import njit
from cam_accuracy_cost import reference_data, score


@njit(cache=True)
def exact_surface_hht(count, h):
    q = np.zeros(2)
    v = np.zeros(2)
    mass = np.array([0.48, 0.007])
    prev = np.array([-3.2 - 0.65 * 0.096, 0.0])
    acc = prev / mass
    hist = np.zeros((count, 8))
    for step in range(1, count):
        t = step * h
        trial = acc.copy()
        for _ in range(30):
            x = q + h * v + h * h * (0.1975 * acc + 0.3025 * trial)
            u = v + h * (0.4 * acc + 0.6 * trial)
            theta = 4 * t + x[1]
            arm = 0.024 * np.cos(theta)
            delta = 0.105 + 0.024 * np.sin(theta) - (0.115 + 0.096 * t + x[0])
            force = 5000 * max(0.0, delta)
            applied = np.array(
                [
                    -3.2 - 70 * (x[0] + 0.096 * t) - 0.65 * (u[0] + 0.096) + force,
                    -1.8 * x[1] - 0.10 * u[1] - arm * force,
                ]
            )
            res = mass * trial - 0.9 * applied - 0.1 * prev
            if np.max(np.abs(res)) < 1e-11 * (1 + np.max(np.abs(applied))):
                break
            k = 5000.0 if delta > 0.0 else 0.0
            beta = 0.9 * 0.3025 * h * h
            a = mass[0] + 0.54 * h * 0.65 + beta * (70 + k)
            b = -beta * k * arm
            c = mass[1] + 0.54 * h * 0.10 + beta * (1.8 + k * arm * arm - 0.024 * np.sin(theta) * force)
            det = a * c - b * b
            trial -= np.array([c * res[0] - b * res[1], a * res[1] - b * res[0]]) / det
        else:
            raise ValueError("Analytic-geometry HHT failed to converge")
        q, v, acc, prev = x, u, trial, applied
        hist[step, :2], hist[step, 2:4], hist[step, 4] = q, v, force
    return hist


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    ref = reference_data(args.work, 160001, 0.000125)
    rows = []
    for divisor in (1, 2, 4):
        h = 0.000125 / divisor
        trace = exact_surface_hht(160000 * divisor + 1, h)
        rows.append(dict(step_us=h * 1e6, scoring_grid_us=125.0, **score(trace[::divisor], ref, 0.000125)))
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
