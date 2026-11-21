"""Measure the two Linux cam solvers against the finest native trajectory."""

import argparse
import csv
import json
from pathlib import Path
from statistics import median

from industrial_accuracy import difference, trace


def compare(folder):
    with (folder / "drake" / "summary.json").open(encoding="utf-8") as stream:
        engine = json.load(stream)
    duration = engine["duration_s"]
    native = []
    for stats in sorted(folder.glob("native-*.stats.csv")):
        with stats.open(newline="", encoding="utf-8") as stream:
            timings = list(csv.DictReader(stream))
        step = float(timings[0]["dt"])
        assert len(timings) == engine["repeats"]
        native.append((stats.with_suffix("").with_suffix(""), step, timings))
    reference_path = min(native, key=lambda item: item[1])[0]
    reference = trace(reference_path, "cam", duration)
    rows = []
    for prefix, step, timings in native:
        measured = next(difference(trace(prefix,"cam",duration),reference,"cam"))
        rows.append({"engine": "NurbsCONTACT", "step_s": step,
                     "online_s": [float(row["wall_us"])/1e6 for row in timings],
                     "setup_s": median(float(row["prep_us"])/1e6 for row in timings), **measured})
    for trial in engine["runs"]:
        prefix = folder / "drake" / trial["trace"]
        measured = next(difference(trace(prefix,"cam",duration),reference,"cam"))
        rows.append({"engine": "Drake", "version": engine["version"], "step_s": trial["step_s"],
                     "online_s": trial["online_s"], "setup_s": trial["median_setup_s"],
                     "max_force_law_error_N": trial["max_force_law_error_N"], **measured})
    for row in rows:
        row["median_online_s"] = median(row["online_s"])
    return {"duration_s": duration, "reference": reference_path.name,
            "repeats": engine["repeats"], "runs": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(compare(args.folder), indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
