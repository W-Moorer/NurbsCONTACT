"""Compare completed whole-body cam traces with the refined cam reference."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from industrial_accuracy import CRITERIA, trace


def compare(prefix, reference, duration, step):
    with prefix.with_suffix(".stats.csv").open(newline="", encoding="utf-8") as stream:
        stats, = list(csv.DictReader(stream))
    steps = round(duration / step)
    if (int(stats["steps"]) != steps or int(stats["accepted"]) < steps or
            int(stats["closest_queries"]) == 0):
        raise ValueError(f"Incomplete closest-pair run: {prefix}")
    with prefix.with_suffix(".csv").open(encoding="utf-8") as stream:
        columns = stream.readline().strip().split(",")
        data = np.loadtxt(stream, delimiter=",", usecols=[columns.index(key) for key in
                                                          ("time_s", "q1", "p0_normal_N")])
    t, q, force = data.T
    if (not np.isfinite(data).all() or t[0] != 0 or abs(t[-1] - duration) > 2e-9 or
            np.any(np.diff(t) <= 0)):
        raise ValueError(f"Incomplete or invalid trace: {prefix}")
    q_ref, force_ref = [np.interp(t, reference[:, 0], reference[:, i]) for i in (1, 2)]
    position = float(np.linalg.norm(q - q_ref) * 1e6 / np.sqrt(t.size))
    force_pct = float(100 * np.linalg.norm(force - force_ref) / np.linalg.norm(force_ref))
    position_limit, force_limit = CRITERIA[0]
    return {
        "position_rms_um": position,
        "force_rms_pct": force_pct,
        "meets_target": position <= position_limit and force_pct <= force_limit,
        "samples": int(t.size),
        "online_s": float(stats["wall_s"]),
        "setup_s": float(stats["setup_s"]),
        "nominal_steps": int(stats["steps"]),
        "accepted_steps": int(stats["accepted"]),
        "closest_queries": int(stats["closest_queries"]),
        "unresolved_queries": int(stats["unresolved_queries"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("runs", nargs="+", type=Path, help="nc_contact output prefixes")
    parser.add_argument("--duration", type=float, default=20)
    parser.add_argument("--step", type=float, default=.000125)
    args = parser.parse_args()
    ref = trace(args.reference, "cam", args.duration)
    result = {"reference": args.reference.name, "duration_s": args.duration, "step_s": args.step,
              "runs": {run.name: compare(run, ref, args.duration, args.step) for run in args.runs}}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
