"""Check whole-STEP contact candidates at the recorded double-tooth spur pose."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import subprocess

from gear_contact_paths import angle


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def run(executable, models, a, b, frame, output, seeds=None):
    initial_a, initial_b = angle(a, frame), angle(b, frame)
    center_x = (float(b["center_x_mm"]) - float(a["center_x_mm"])) * .001
    center_z = (float(b["z_mm"]) - float(a["z_mm"])) * .001
    translation = (center_x * math.cos(initial_a), -center_x * math.sin(initial_a), center_z)
    activation = 55e-6
    motion = [*translation, 0, 0, 1, initial_b - initial_a]
    command = [str(executable.resolve()), "regions", str((models / a["step"]).resolve()),
               str((models / b["step"]).resolve()), str(activation), *map(str, motion), str(output)]
    if seeds is not None:
        command.append(str(seeds))
    subprocess.run(command, check=True)
    rows = read_rows(Path(str(output) + ".csv"))
    stats, = read_rows(Path(str(output) + ".stats.csv"))
    if len(rows) != int(stats["witnesses"]):
        raise ValueError("Contact witness count disagrees with the query statistics")
    by_faces = defaultdict(list)
    for row in rows:
        if not (0 <= float(row["distance_m"]) <= activation):
            raise ValueError("A candidate is outside the requested distance")
        if row["facing_interiors"] == "1":
            by_faces[int(row["face_a"]), int(row["face_b"])].append(row)
    if len(by_faces) != 2:
        raise ValueError(f"Expected two facing interior face pairs at the recorded pose, found {list(by_faces)}")
    if len({key[0] for key in by_faces}) != 2 or len({key[1] for key in by_faces}) != 2:
        raise ValueError("The facing candidates share a face at the double-tooth pose")
    representatives = {}
    for (first, second), samples in sorted(by_faces.items()):
        representatives[first, second] = min(samples, key=lambda row: float(row["distance_m"]))
    summary = {"time_s": float(frame["time"]), "activation_um": activation * 1e6,
               "reference_loaded_teeth": int(float(frame["active0"])),
               "candidate_face_pairs": len({(row["face_a"], row["face_b"]) for row in rows}),
               **{key: float(val) if key.endswith("_s") else int(val) for key, val in stats.items()}}
    summary["facing_face_pairs"] = [
        {"face_a": first, "face_b": second, "distance_um": float(witness["distance_m"]) * 1e6,
         "point_on_a_m": [float(witness[key]) for key in ("ax_m", "ay_m", "az_m")]}
        for (first, second), witness in representatives.items()]
    return representatives, summary


def check(executable, models, trace, output):
    a, b = read_rows(models / "spur.csv")
    with trace.open(newline="", encoding="utf-8") as stream:
        frames = csv.DictReader(stream)
        frame = next((row for row in frames if float(row["active0"]) == 2), None)
        following = next(frames, None)
    if frame is None or following is None or float(following["active0"]) != 2:
        raise ValueError("Expected two consecutive double-tooth frames in the archived spur trajectory")
    initial, full = run(executable, models, a, b, frame, output)
    seeds = Path(str(output) + ".seeds.txt")
    seed_fields = ("face_a", "face_b", "ua", "va", "ub", "vb")
    seeds.write_text("\n".join(" ".join(row[key] for key in seed_fields) for row in initial.values()) + "\n",
                     encoding="ascii")
    cold, cold_report = run(executable, models, a, b, following, output.with_name(output.name + "-cold"))
    tracked, tracked_report = run(executable, models, a, b, following, output.with_name(output.name + "-tracked"), seeds)
    if set(initial) != set(cold) or set(cold) != set(tracked) or tracked_report["unresolved_face_pairs"]:
        raise ValueError("The tracked face pairs do not match the new complete candidate search")
    difference = max(abs(float(cold[key]["distance_m"]) - float(tracked[key]["distance_m"])) for key in cold)
    if difference > 1e-7:
        raise ValueError(f"Tracked distance differs by {difference * 1e6:g} micrometers from the cold search")
    summary = {"trace": trace.name, "initial": full, "next_frame_cold": cold_report,
               "next_frame_tracked": tracked_report, "max_tracking_gap_difference_um": difference * 1e6}
    Path(str(output) + ".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("models", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path, help="Path prefix for the CSV and JSON outputs")
    args = parser.parse_args()
    print(json.dumps(check(args.executable, args.models, args.trace, args.output), indent=2))


if __name__ == "__main__":
    main()
