"""Compare a saved step-size attempt with the archived GPU trace, without extrapolation."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Archived responses.npz")
    parser.add_argument("attempt", type=Path, help="Directory containing response.npy and run.json")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(args.reference) as archive:
        reference = archive["gpu"]
        columns = archive["columns"].tolist()
    attempt = np.load(args.attempt / "response.npy")
    run = json.loads((args.attempt / "run.json").read_text())
    if len(attempt) < 2 or attempt.shape[1] != len(columns):
        raise ValueError("At least two saved frames with the reference columns are required")
    dt = reference[1, 0] - reference[0, 0]
    indices = np.rint((attempt[:, 0] - reference[0, 0]) / dt).astype(int)
    if np.any(indices < 0) or np.any(indices >= len(reference)):
        raise ValueError("Attempt lies outside the reference interval")
    if np.max(abs(reference[indices, 0] - attempt[:, 0])) > 1e-9:
        raise ValueError("Saved grids must coincide; no interpolated peak comparison is allowed")
    n = sum(key.startswith("v") for key in columns)
    if sum(key.startswith("normal_load") for key in columns) != 2:
        raise ValueError("This reducer comparison expects two contact pairs")
    completed = bool(run["completed"] and abs(attempt[-1, 0] - run["duration_s"]) < 1e-9)
    selected = [f"v{i}" for i in range(n)] + [f"force{i}" for i in range(n)]
    selected += [key for key in columns if key.startswith("normal_load")]
    metrics = {}
    native = {}
    reference_mask = ((reference[:, 0] >= attempt[0, 0] - 1e-10)
                      & (reference[:, 0] <= attempt[-1, 0] + 1e-10))
    for key in selected:
        col = columns.index(key)
        delta = attempt[:, col] - reference[indices, col]
        if not np.isfinite(delta).all():
            raise ValueError(f"Nonfinite response for {key}")
        metrics[key] = dict(max_abs=float(abs(delta).max()), rms=float(np.sqrt(np.mean(delta**2))))
        native[key] = {}
        for name, data in (("reference", reference[reference_mask]), ("attempt", attempt)):
            native[key][name] = dict(
                saved_min=float(data[:, col].min()), saved_max=float(data[:, col].max()),
                trapezoidal_integral=float(np.sum(
                    np.diff(data[:, 0]) * (data[1:, col] + data[:-1, col]) / 2
                )),
            )
    contacts = {}
    for key in columns:
        if key.startswith("contact"):
            col = columns.index(key)
            contacts[key] = int(np.count_nonzero(attempt[:, col] != reference[indices, col]))
    report = dict(
        completed=completed,
        requested_duration_s=run["duration_s"],
        observed_interval_s=[float(attempt[0, 0]), float(attempt[-1, 0])],
        common_samples=len(attempt),
        scope="Differences at saved common times only; not continuous-time extrema or accuracy against truth.",
        long_time_validation_available=completed,
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        attempt_sha256=hashlib.sha256((args.attempt / "response.npy").read_bytes()).hexdigest(),
        response_differences=metrics,
        native_saved_grid_statistics=native,
        native_statistics_scope="Extrema and trapezoidal integrals of saved samples; internal substep peaks are not recorded.",
        contact_presence_mismatches=contacts,
    )
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(n + 1, 2, figsize=(12, 2.4 * (n + 1)), sharex=True, layout="constrained")
    reference_on_common_grid = reference[indices]
    reference = reference[reference_mask]
    time_scale = 1.0 if attempt[-1, 0] >= 0.1 else 1000.0
    time_unit = "s" if time_scale == 1 else "ms"
    fields = [(axes[i, 0], f"v{i}", f"Shaft {i + 1} speed (rad/s)") for i in range(n)]
    fields += [(axes[i, 1], f"force{i}", f"Shaft {i + 1} generalized torque (N m)") for i in range(n)]
    fields += [(axes[n, i], f"normal_load{i}", f"Pair {i + 1} normal load (N)") for i in range(2)]
    for ax, key, title in fields:
        col = columns.index(key)
        ax.plot(time_scale * reference[:, 0], reference[:, col], color="#245a81", lw=0.8, label=f"{dt * 1000:g} ms main step")
        ax.plot(time_scale * attempt[:, 0], attempt[:, col], "o--", color="#c65f29", lw=0.8, ms=3,
                markevery=max(1, len(attempt) // 100), label=f"{run['step_s'] * 1000:g} ms main step (adaptive)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.2)
        ax.set_xlim(time_scale * attempt[0, 0], time_scale * attempt[-1, 0])
    axes[0, 0].legend(fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel(f"Physical time ({time_unit})")
    if completed:
        title = f"Completed step-size comparison over {run['duration_s']:g} s"
    else:
        title = (
            f"{run['step_s'] * 1000:g} ms attempt stopped after {attempt[-1, 0] * 1000:g} ms: startup diagnostic only\n"
            f"Requested duration: {run['duration_s']:g} s. No valid full-history result or speedup measurement."
        )
    fig.suptitle(title, fontsize=12)
    fig.savefig(args.out / ("responses.png" if completed else "startup-comparison.png"), dpi=150)
    for ax, key, title in fields:
        ax.clear()
        col = columns.index(key)
        ax.plot(time_scale * attempt[:, 0], attempt[:, col] - reference_on_common_grid[:, col],
                color="#784a79", lw=0.8)
        ax.set_title(title + " difference", fontsize=10)
        ax.grid(alpha=0.2)
        ax.set_xlim(time_scale * attempt[0, 0], time_scale * attempt[-1, 0])
    for ax in axes[-1]:
        ax.set_xlabel(f"Physical time ({time_unit})")
    fig.suptitle("Coarse minus fine at common saved times\nNumerical differences, not errors against physical truth", fontsize=12)
    fig.savefig(args.out / "response-differences.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
