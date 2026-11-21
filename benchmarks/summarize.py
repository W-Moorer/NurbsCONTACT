"""Summarize the query-only measurements written by nc_bench."""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def read_groups(path):
    groups = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            groups[(row["scene"], row["variant"])].append(row)
    if not groups:
        raise ValueError("benchmark file contains no measurements")
    return groups


def summarize(rows):
    durations = sorted(float(row["query_us"]) / 1000 for row in rows)
    statuses = sorted({row["status"] for row in rows})
    widths = [float(row["bracket_width"]) for row in rows if row["bracket_width"]]
    repeat_ids = sorted(int(row["repeat"]) for row in rows)
    if repeat_ids != list(range(len(rows))):
        raise ValueError("missing or duplicate repetition")
    return {
        "repeats": len(rows),
        "status": "/".join(statuses),
        "median_ms": f"{statistics.median(durations):.6f}",
        "p95_ms": f"{durations[math.ceil(0.95 * len(durations)) - 1]:.6f}",
        "median_nodes": f"{statistics.median(int(row['nodes']) for row in rows):g}",
        "max_bracket": f"{max(widths):.9g}" if widths else "",
        "time_target_met": sum(
            row["status"] == "clear" or (
                row["status"] == "hit" and float(row["bracket_width"]) <= 5e-4
            )
            for row in rows
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    groups = read_groups(args.measurements)
    records = [
        {"scene": scene, "variant": variant, **summarize(rows)}
        for (scene, variant), rows in groups.items()
    ]
    if args.output:
        with args.output.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    else:
        for row in records:
            print("\t".join(map(str, row.values())))


if __name__ == "__main__":
    main()
