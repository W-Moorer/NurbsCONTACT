"""Replay measured eccentric-cam transitions as continuous STEP queries."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import mpmath as mp


def extract(trace):
    with trace.open(newline="", encoding="utf-8") as stream:
        rows = [{key: float(row[key]) for key in ("time", "q0", "q1", "solver_peak_force")}
                for row in csv.DictReader(stream)]
    for i in range(1, len(rows) - 1):
        if rows[i]["solver_peak_force"] == 0 < rows[i + 1]["solver_peak_force"]:
            if rows[i-1]["solver_peak_force"] == 0:
                yield f"cam-before-{i}", rows[i-1], rows[i], False
            yield f"cam-onset-{i}", rows[i], rows[i+1], True


def fixtures(trace, output):
    # The formula is the independent reference for the published benchmark
    # geometry. The tested backends consume only STEP files and motions.
    reference = {}
    with (output / "cam-paths.txt").open("w", encoding="utf-8") as stream, mp.workdps(65):
        stream.write("# name A B x y z dx dy dz activation-mm initial-rad rotation-rad\n")
        for name, start, end, onset in extract(trace):
            y = -1000 * (.115 + start["q1"])
            dy = -1000 * (end["q1"] - start["q1"])
            angle, omega = start["q0"], end["q0"] - start["q0"]
            height, vertical, alpha, rate = map(mp.mpf, (y, dy, angle, omega))

            def residual(t):
                return -height - vertical*t - 117 - 24*mp.sin(alpha + rate*t)

            # Sign of the derivative throughout the accepted step, using
            # the unit Lipschitz constant of cosine.
            if -vertical - 24*rate*mp.cos(alpha) + 24*rate**2 >= 0:
                raise ValueError(f"Cannot establish monotonic approach for {name}")
            if residual(0) <= 0 or (residual(1) <= 0) != onset:
                raise ValueError(f"Trace and analytic crossing disagree for {name}")
            lo, hi = mp.mpf(0), mp.mpf(1)
            if onset:
                for _ in range(200):
                    midpoint = (lo + hi) / 2
                    if residual(midpoint) > 0:
                        lo = midpoint
                    else:
                        hi = midpoint
            reference[name] = {"start_s": start["time"], "end_s": end["time"],
                               "onset_lower": str(lo) if onset else None,
                               "onset_upper": str(hi) if onset else None}
            stream.write(f"{name} 1 2 0 {y!r} 0 0 {dy!r} 0 12 {angle!r} {omega!r}\n")
    if not reference:
        raise ValueError("The trace contains no contact starts")
    (output / "cam-reference.json").write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")


def audit(raw, reference):
    with raw.open(newline="", encoding="utf-8") as stream:
        groups = defaultdict(list)
        for row in csv.DictReader(stream):
            groups[row["scene"], row["method"], float(row["deflection_mm"])].append(row)
    by_scene = defaultdict(set)
    for (scene, method, deflection), samples in groups.items():
        if sorted(int(row["repeat"]) for row in samples) != list(range(len(samples))):
            raise ValueError(f"Unexpected repeat indices for {scene} and {method}")
        by_scene[scene].add((method, deflection, len(samples)))
    if set(by_scene) != set(reference):
        raise ValueError("The measured cases do not match the trajectory reference")
    settings = by_scene[next(iter(reference))]
    if any(actual != settings for actual in by_scene.values()):
        raise ValueError("The tested methods and repeat counts differ between trajectories")
    summary = []
    with mp.workdps(65):
        for (scene, method, deflection), samples in sorted(groups.items()):
            row = samples[0]
            if len({(r["status"], r["lower_time"], r["upper_time"]) for r in samples}) != 1:
                raise ValueError(f"Non-reproducible result for {scene} and {method}")
            expected = reference[scene]
            lower = mp.mpf(float(row["lower_time"]))
            upper = mp.mpf(float(row["upper_time"])) if row["upper_time"] else None
            if not 0 <= lower <= 1 or (upper is not None and not lower <= upper <= 1):
                raise ValueError(f"Invalid onset interval for {scene} and {method}")
            if expected["onset_lower"] is None:
                compatible = row["status"] != "hit"
                complete = row["status"] == "clear"
            else:
                a, b = (mp.mpf(expected[name]) for name in ("onset_lower", "onset_upper"))
                compatible = row["status"] != "clear" and lower <= b and (upper is None or upper >= a)
                complete = upper is not None and upper-lower <= mp.mpf(float(row["time_tolerance"]))
            if not compatible and method in ("native-sweep", "cad-ca-bounds"):
                raise ValueError(f"Unexpected incompatible CAD bound for {scene} and {method}")
            summary.append({"scene": scene, "method": method, "deflection_mm": deflection,
                            "status": row["status"], "compatible": compatible, "within_target": complete,
                            "lower_time": row["lower_time"], "upper_time": row["upper_time"],
                            "median_ms": median(float(r["query_us"]) for r in samples)/1000,
                            "prepare_ms": float(row["prepare_us"])/1000, "repeats": len(samples)})
    if not summary:
        raise ValueError("No measured queries were found")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.audit:
        reference = json.loads((args.output / "cam-reference.json").read_text(encoding="utf-8"))
        summary = audit(args.audit, reference)
        with (args.output / "cam-summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=summary[0], lineterminator="\n")
            writer.writeheader()
            writer.writerows(summary)
    else:
        fixtures(args.trace, args.output)


if __name__ == "__main__":
    main()
