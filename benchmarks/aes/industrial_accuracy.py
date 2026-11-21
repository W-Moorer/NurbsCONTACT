"""Check complete industrial runs and report accuracy versus timing."""

import argparse
import csv
from pathlib import Path
from statistics import median

import numpy as np


FIELDS = {
    "cam": (("follower", "q1", 1e6, "solver_peak_force"),),
    "spur": (("stage 1", "te0_um", 1., "normal0_N"),),
    "reducer": (("stage 1", "te0_um", 1., "normal0_N"),
                ("stage 2", "te1_um", 1., "normal1_N")),
}
REFERENCE = {
    "cam": "cam-native-24-62us",
    "spur": "spur-native-time-check-0nm-2p5us-31",
    "reducer": "reducer-native-axial-check-0nm-5us-63",
}
CRITERIA = ((.5, 2.), (1., 5.), (2., 10.))


def trace(prefix, case, duration):
    fields = ["time"]
    fields.extend(key for _, position, _, force in FIELDS[case] for key in (position, force))
    with prefix.with_suffix(".csv").open(encoding="utf-8") as stream:
        names = stream.readline().strip().split(",")
        data = np.loadtxt(stream, delimiter=",", usecols=[names.index(key) for key in fields])
    time = data[:, 0]
    if (not np.isfinite(data).all() or time[0] != 0 or abs(time[-1] - duration) > 2e-9
            or np.any(np.diff(time) <= 0)):
        raise ValueError(f"Incomplete or invalid trace: {prefix}")
    return data


def difference(data, reference, case):
    # Compare on the common coarse output grid. The finer integrations
    # advance at smaller steps regardless of how often a trace is saved.
    stride = .001 if case == "cam" else .000020
    time = np.arange(round(reference[-1, 0] / stride) + 1) * stride

    def sampled(values, channel):
        return np.interp(time, values[:, 0], values[:, channel])

    for i, (label, _, scale, _) in enumerate(FIELDS[case]):
        p = (sampled(data, 2*i + 1) - sampled(reference, 2*i + 1)) * scale
        f = sampled(data, 2*i + 2)
        fr = sampled(reference, 2*i + 2)
        force_scale = np.sqrt(np.mean(fr**2))
        yield {
            "response": label,
            "position_rms_um": np.sqrt(np.mean(p**2)),
            "position_max_um": np.max(np.abs(p)),
            "force_rms_N": np.sqrt(np.mean((f-fr)**2)),
            "force_rms_pct": 100 * np.sqrt(np.mean((f-fr)**2)) / force_scale,
            "force_peak_error_pct": 100 * abs(np.max(f) - np.max(fr)) / np.max(fr),
        }


def summarize(folders):
    configs = {}
    for folder in folders:
        with (folder / "observations.csv").open(newline="", encoding="utf-8") as stream:
            configs.update((row["prefix"], (folder, row)) for row in csv.DictReader(stream))
    refs = {}
    for name, key in REFERENCE.items():
        if key in configs:
            folder, config = configs[key]
            if config["status"] == "complete":
                refs[name] = trace(folder / key, name, float(config["duration_s"]))
    for key, (folder, row) in configs.items():
        case, duration = row["case"], float(row["duration_s"])
        if row["status"] != "complete" or case not in refs:
            yield row
            continue
        prefix = folder / key
        with prefix.with_suffix(".stats.csv").open(newline="", encoding="utf-8") as stream:
            stats = list(csv.DictReader(stream))
        if any(int(s["accepted"]) < round(duration / float(row["step_s"])) for s in stats):
            raise ValueError(f"Missing accepted steps: {prefix}")
        times = [float(s["wall_us"]) / 1e6 if "wall_us" in s else float(s["wall_s"]) for s in stats]
        setup = [float(s["prep_us"]) / 1e6 if "prep_us" in s else float(s["setup_s"]) for s in stats]
        rows = list(difference(trace(prefix, case, duration), refs[case], case))
        accuracy = max(r["position_rms_um"] for r in rows), max(r["force_rms_pct"] for r in rows)
        criteria = {f"within_{p:g}um_{f:g}pct": bool(accuracy[0] <= p and accuracy[1] <= f)
                    for p, f in CRITERIA}
        for response in rows:
            yield {**row, **response, **criteria, "online_s": median(times), "setup_s": median(setup),
                   "online_min_s": min(times), "online_max_s": max(times), "repeats": len(times),
                   "force_refinements": stats[0].get("force_refinements"),
                   "certificate_hits": stats[0].get("certificate_hits", stats[0].get("reused_exclusions")),
                   "reference": REFERENCE[case]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    rows = list(summarize(args.runs))
    header = list(dict.fromkeys(key for row in rows for key in row))
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
