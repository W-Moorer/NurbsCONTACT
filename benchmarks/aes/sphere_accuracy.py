"""Run the two half-sine analytic checks at three HHT resolutions."""

import argparse
import csv
from math import pi, sin, sqrt
from pathlib import Path
from statistics import median
import subprocess


SPEED = .45
STIFFNESS = 5000.0
APPROACH = .018
DURATION = .16
STEPS = (.0005, .00025, .000125)
METHODS = (
    ("fixed", "brep", None),
    ("bounded-exact", "brep", "bounded-exact"),
    ("bounded", "brep", "bounded"),
    ("bounded-cache", "brep", "bounded-cache"),
    ("mesh-24", "mesh", None),
)


def reference(scene, time, initial):
    relative_speed = SPEED
    omega = sqrt(STIFFNESS * (1 if scene == "drop" else 2))
    onset = APPROACH / relative_speed
    release = onset + pi / omega
    compression = 0.0
    if time < onset:
        closure = relative_speed * time
    elif time <= release:
        compression = relative_speed / omega * sin(omega * (time - onset))
        closure = APPROACH + compression
    else:
        closure = APPROACH - relative_speed * (time - release)
    positions = ((initial[0] - closure,) if scene == "drop"
                 else (initial[0] + closure / 2, initial[1] - closure / 2))
    return positions, STIFFNESS * compression


def metrics(scene, prefix, step, repeats):
    dimensions = 1 if scene == "drop" else 2
    with prefix.with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
        frames = list(csv.DictReader(stream))
    with prefix.with_suffix(".stats.csv").open(newline="", encoding="utf-8") as stream:
        times = list(csv.DictReader(stream))
    assert len(times) == repeats and len(frames) == round(DURATION / step) + 1
    assert float(frames[0]["time"]) == 0 and abs(float(frames[-1]["time"]) - DURATION) < 1e-14
    initial = tuple(float(frames[0][f"q{i}"]) for i in range(dimensions))
    position_error, force_error = [], []
    for row in frames:
        position, force = reference(scene, float(row["time"]), initial)
        position_error.extend(float(row[f"q{i}"]) - q for i, q in enumerate(position))
        force_error.append(float(row["solver_peak_force"]) - force)
    return {
        "position_rms_m": sqrt(sum(e*e for e in position_error) / len(position_error)),
        "position_max_m": max(map(abs, position_error)),
        "force_rms_n": sqrt(sum(e*e for e in force_error) / len(force_error)),
        "force_max_error_n": max(map(abs, force_error)),
        "setup_median_us": median(float(row["prep_us"]) for row in times),
        "online_median_us": median(float(row["wall_us"]) for row in times),
        "clearance_lower_m": min(float(row["min_guard"]) for row in times),
        "bounded_pairs": times[0]["bounded_pairs"], "approximated_pairs": times[0]["approximated_pairs"],
        "reused_exclusions": times[0]["reused_exclusions"], "force_refinements": times[0]["force_refinements"],
    }


def run(executable, output, repeats):
    output.mkdir(parents=True, exist_ok=True)
    models = output / "models"
    subprocess.run([str(executable), "models", str(models)], check=True, stdout=subprocess.DEVNULL)
    for scene in ("drop", "two-ball"):
        for step in STEPS:
            for method, backend, option in METHODS:
                prefix = output / f"{scene}-{method}-{round(step * 1e6)}us"
                args = [str(executable), "run", scene, backend, "24", str(step), str(DURATION),
                        str(models), str(prefix), str(repeats)]
                subprocess.run(args + ([option] if option else []), check=True, stdout=subprocess.DEVNULL)
                yield {"scene": scene, "method": method, "step_s": step, "repeats": repeats,
                       **metrics(scene, prefix, step, repeats)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    rows = list(run(args.executable.resolve(), args.output.resolve(), args.repeats))
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Verified trace lengths and measured analytic errors in {len(rows)} configurations")


if __name__ == "__main__":
    main()
