"""Audit the serial paper timing batch and the bounded reducer diagnostic."""

import argparse
import csv
import json
from pathlib import Path
from statistics import median

from industrial_accuracy import difference, trace


TIMINGS = {
    "cam-native": ("cam", 20., 3),
    "cam-polygon": ("cam", 20., 3),
    "cam-polygon-relaxed": ("cam", 20., 3),
    "spur-native": ("spur", 2., 3),
    "spur-polyline": ("spur", 2., 3),
    "reducer-native": ("reducer", 3., 2),
    "reducer-polyline": ("reducer", 3., 2),
}
REFERENCES = {
    "cam": "industrial-cam2/cam-native-24-62us",
    "spur": "industrial-spur-time/spur-native-time-check-0nm-2p5us-31",
    "reducer": "industrial-pilot/reducer-native-axial-check-0nm-5us-63",
}


def measurement(prefix, case, duration, repeats, ref):
    with prefix.with_suffix(".stats.csv").open(newline="", encoding="utf-8") as stream:
        stats = list(csv.DictReader(stream))
    if len(stats) != repeats or any(row["steps"] != row["accepted"] for row in stats):
        raise ValueError(f"Incomplete result: {prefix}")
    first = stats[0]
    online = [float(row["wall_s"]) if "wall_s" in row else float(row["wall_us"])*1e-6 for row in stats]
    setup = float(first["setup_s"]) if "setup_s" in first else float(first["prep_us"])*1e-6
    return {
        "steps": int(first["steps"]), "step_s": float(first.get("dt_s",first.get("dt"))),
        "duration_s": duration, "setup_s": setup, "online_s": online, "median_online_s": median(online),
        "force_refinements": int(first["force_refinements"]),
        "rechecked_certificates": int(first.get("certificate_hits",first.get("reused_exclusions"))),
        "responses": list(difference(trace(prefix,case,duration),ref,case)),
    }


def summarize(work):
    durations = {case:duration for case,duration,_ in TIMINGS.values()}
    refs = {case:trace(work/REFERENCES[case],case,duration) for case,duration in durations.items()}
    timings = {name:{**measurement(work/"final-timing"/name,case,duration,repeats,refs[case]),
                    "reference":REFERENCES[case]}
               for name,(case,duration,repeats) in TIMINGS.items()}
    ref = trace(work/"reducer-fixed-ablation","reducer",.2)
    ablations = {name:measurement(work/stem,"reducer",.2,1,ref) for name,stem in (
        ("fixed","reducer-fixed-ablation"), ("bounded-exact","reducer-bounded-exact-ablation"),
        ("bounded-cache","reducer-bounded-cache-guardfix"))}
    return {"serial_timing_windows":timings,"reducer_bounded_diagnostic":ablations}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.work),indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
