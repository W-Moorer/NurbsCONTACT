"""Recheck archived continuous-query time outputs against independent references."""

import argparse
import csv
import json
import math
from pathlib import Path


TIME_TARGET = 5e-4


def audit(path, analytic):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    widths = []
    clear = hit = unresolved = wide = disagreements = 0
    for row in rows:
        start = float(row["safe_prefix"])
        outcome = row["status"]
        if analytic:
            reference = float(row["reference"]) if row["reference"] else math.nan
            expects_clear = not math.isfinite(reference)
            lower = upper = reference
        else:
            expects_clear = row["reference_status"] == "clear"
            lower = float(row["reference_lower"]) if row["reference_lower"] else math.nan
            upper = float(row["reference_upper"]) if row["reference_upper"] else math.nan
        if outcome == "clear":
            clear += 1
            disagreements += not expects_clear or start < 1
        elif outcome == "hit":
            hit += 1
            end = float(row["witness_time"])
            widths.append(end-start)
            wide += end-start > TIME_TARGET
            disagreements += (expects_clear or not math.isfinite(lower) or not math.isfinite(upper) or
                              start > end or start > upper or end < lower)
        else:
            unresolved += 1
    latency = sorted(float(row["query_us"]) for row in rows)
    return {
        "runs": len(rows), "clear": clear, "witnessed": hit, "unresolved": unresolved,
        "witnessed_above_time_target": wide, "reference_disagreements": disagreements,
        "maximum_witnessed_width": max(widths, default=None),
        "p95_query_us": latency[math.ceil(.95*len(latency))-1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    inputs = ("results.csv","axial-results.csv","assemblies/wheel-results.csv",
              "assemblies/lidar-results.csv","assemblies/lidar-diagnostic.csv")
    report = {
        "time_target": TIME_TARGET,
        "scope": "Recorded time certificates and reference intervals. CSVs do not store witness coordinates.",
        "files": {name:audit(root/name,index<2) for index,name in enumerate(inputs)},
    }
    if any(item["reference_disagreements"] for item in report["files"].values()):
        raise ValueError("a recorded interval disagrees with its reference")
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
