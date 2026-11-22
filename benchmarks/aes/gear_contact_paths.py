"""Extract one recorded load-rise segment per gear stage for a geometry diagnostic."""

import argparse
import csv
import json
import math
from pathlib import Path


def angle(part, row):
    speed = math.tau * float(part["start_rpm"]) / 60
    return float(part["angle_zero_rad"]) + int(part["direction"]) * (
        speed * float(row["time"]) + float(row["q" + part["shaft"]]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path)
    parser.add_argument("traces", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for case, trace in (("spur", "spur-native-0nm-5us-31.csv"), ("reducer", "reducer-native-0nm-10us-31.csv")):
        with (args.models / (case+".csv")).open(newline="", encoding="utf-8") as stream:
            parts = list(csv.DictReader(stream))
        with (args.traces / trace).open(newline="", encoding="utf-8") as stream:
            rows = csv.DictReader(stream)
            start = next(row for row in rows if float(row["time"]) >= .10)
            end = next(rows)
        for a, b in zip(parts[::2], parts[1::2], strict=True):
            center = float(b["center_x_mm"]) - float(a["center_x_mm"])
            a0, a1, b0, b1 = angle(a, start), angle(a, end), angle(b, start), angle(b, end)
            x, y = center*math.cos(a0), -center*math.sin(a0)
            dx, dy = center*math.cos(a1)-x, -center*math.sin(a1)-y
            theta, rotation = b0-a0, (b1-a1)-(b0-a0)
            name = f"{case}-stage-{a['stage']}"
            (args.output / (name+".txt")).write_text(
                f"{name} 1 2 {x:.17g} {y:.17g} 0 {dx:.17g} {dy:.17g} 0 0.055 {theta:.17g} {rotation:.17g}\n",
                encoding="ascii")
            metadata[name] = {"step_a": a["step"], "step_b": b["step"], "trace": trace,
                              "start_s": float(start["time"]), "end_s": float(end["time"]),
                              "translation_chord_deviation_mm": abs(center)*(a1-a0)**2/8}
    (args.output / "gear-paths.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
