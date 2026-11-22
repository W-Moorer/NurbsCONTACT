"""Replay the existing eccentric cam and spur stage to verify CAD line loads."""

import argparse
import json
import math
from pathlib import Path
import subprocess

from cad_contact_regions import check as check_regions, read_rows
from gear_contact_paths import angle


def cross(a, b):
    return tuple(a[i - 2] * b[i - 1] - a[i - 1] * b[i - 2] for i in range(3))


def difference(a, b):
    return tuple(x - y for x, y in zip(a, b))


def vector(row, prefix, units):
    return tuple(float(row[f"{prefix}{i}_{units}"]) for i in "xyz")


def verify_vector(measured, expected, tolerance, label):
    error = math.dist(measured, expected)
    if not error <= tolerance:
        raise ValueError(f"{label}: error {error:.6g} exceeds {tolerance:g}")
    return error


def run(executable, a, b, cutoff, stiffness, pose, output, seeds=None):
    command = [executable, "line-load", a, b, cutoff, stiffness, *pose, output]
    if seeds is not None:
        command.append(seeds)
    subprocess.run(list(map(str, command)), check=True)
    modes = {row["mode"]: {key: float(value) if key.endswith(("_s", "_m")) else int(value)
                           for key, value in row.items() if key != "mode"}
             for row in read_rows(Path(str(output) + ".stats.csv"))}
    loads = {mode: {} for mode in modes}
    for row in read_rows(Path(str(output) + ".csv")):
        mode = row["mode"]
        pair = (int(row["face_a"]), int(row["face_b"]))
        if pair in loads[mode]:
            raise ValueError(f"Duplicated line load for {mode} face pair {pair}")
        fa, fb = (vector(row, "force_" + side, "N") for side in "ab")
        ma, mb = (vector(row, "moment_" + side, "Nm") for side in "ab")
        zero = (0., 0., 0.)
        transfer = cross(pose[:3], fb)
        verify_vector(tuple(x + y for x, y in zip(fa, fb)), zero, 1e-10, "net internal force")
        verify_vector(tuple(x + y + z for x, y, z in zip(ma, mb, transfer)),
                      zero, 1e-10, "net internal moment about the origin of A")
        loads[mode][pair] = {"length_m": float(row["length_m"]), "samples": int(row["samples"]),
                             "force_b_N": fb, "moment_b_Nm": mb}
    for mode, stats in modes.items():
        if stats["unresolved_face_pairs"] or stats["unresolved_seeds"] or stats["lines"] != len(loads[mode]):
            raise ValueError(f"Unresolved or missing line loads in {output}: {stats}")
    return loads, modes


def cam_check(executable, models, output):
    width = .06
    cutoff, stiffness = .012, 5000 / width
    pose = (0., -.115, 0., 0, 0, 1, 0.)
    loads, stats = run(executable, models / "follower.step", models / "cam.step",
                       cutoff, stiffness, pose, output)
    load, = loads["cold"].values()
    expected_force, expected_moment = (0., -10., 0.), (0., 0., -.24)
    verify_vector((load["length_m"],), (width,), 1e-9, "cam face width")
    verify_vector(load["force_b_N"], expected_force, 1e-8, "cam force")
    verify_vector(load["moment_b_Nm"], expected_moment, 1e-9, "cam moment")
    return {"cutoff_m": cutoff, "line_stiffness_N_per_m2": stiffness, "reference_gap_m": .010,
            "reference_force_b_N": expected_force, "reference_moment_b_Nm": expected_moment,
            "load": load, "statistics": stats["cold"]}


def spur_check(executable, models, trace, output):
    region_path = output.with_name(output.name + "-regions")
    regions = check_regions(executable, models, trace, region_path)
    time = regions["next_frame_cold"]["time_s"]
    frame = next(row for row in read_rows(trace) if float(row["time"]) == time)
    a, b = read_rows(models / "spur.csv")
    angle_a, angle_b = angle(a, frame), angle(b, frame)
    center_x = (float(b["center_x_mm"]) - float(a["center_x_mm"])) * .001
    center_z = (float(b["z_mm"]) - float(a["z_mm"])) * .001
    pose = (center_x * math.cos(angle_a), -center_x * math.sin(angle_a), center_z, 0, 0, 1, angle_b - angle_a)
    cutoff, stiffness = 55e-6, 1e9
    loads, stats = run(executable, models / a["step"], models / b["step"], cutoff, stiffness,
                       pose, output, Path(str(region_path) + ".seeds.txt"))
    cold_rows = read_rows(Path(str(region_path) + "-cold.csv"))
    tracked_rows = read_rows(Path(str(region_path) + "-tracked.csv"))
    pairs = {(row["face_a"], row["face_b"]) for row in cold_rows if row["facing_interiors"] == "1"}
    width = min(float(gear["width_mm"]) for gear in (a, b)) * .001
    if (set(loads["cold"]) != {tuple(map(int, p)) for p in pairs} or
            set(loads["tracked"]) != set(loads["cold"]) or
            stats["tracked"]["replaced_witnesses"] != len(pairs) or stats["tracked"]["max_witness_shift_m"] < .001):
        raise ValueError("Expected the same tooth pairs with substantially shifted interior points")
    measured = []
    for pair, cold in loads["cold"].items():
        tracked = loads["tracked"][pair]
        boundaries = [row for row in cold_rows
                      if (row["face_a"], row["face_b"]) == tuple(map(str, pair)) and row["facing_interiors"] == "0"]
        lo, hi = min(boundaries, key=lambda r: float(r["az_m"])), max(boundaries, key=lambda r: float(r["az_m"]))
        verify_vector((cold["length_m"], tracked["length_m"]), (width, width), 1e-9, "spur face width")
        # Independent midpoint calculation from the discovered end points and the specified face width.
        midpoint_a, midpoint_b = (tuple((x + y) / 2 for x, y in zip(vector(lo, side, "m"), vector(hi, side, "m")))
                                  for side in "ab")
        delta = difference(midpoint_b, midpoint_a)
        gap = math.hypot(*delta)
        reference = tuple(stiffness * width * (cutoff - gap) * x / gap for x in delta)
        moment = cross(difference(midpoint_b, pose[:3]), reference)
        representatives = [
            min((row for row in rows if (int(row["face_a"]), int(row["face_b"])) == pair and
                 row["facing_interiors"] == "1"), key=lambda row: float(row["distance_m"]))
            for rows in (cold_rows, tracked_rows)
        ]
        # Use the same resultant so that only the point of application changes.
        point_moments = [cross(difference(vector(row, "b", "m"), pose[:3]), cold["force_b_N"])
                         for row in representatives]
        measured.append({"face_pair": pair, **cold, "reference_force_b_N": reference,
                         "reference_moment_b_Nm": moment,
                         "midpoint_force_error_N": verify_vector(cold["force_b_N"], reference, 1e-3, "spur force"),
                         "midpoint_moment_error_Nm": verify_vector(cold["moment_b_Nm"], moment, 1e-4, "spur moment"),
                         "tracked_force_change_N": verify_vector(cold["force_b_N"], tracked["force_b_N"], 1e-8, "spur tracking force"),
                         "tracked_moment_change_Nm": verify_vector(cold["moment_b_Nm"], tracked["moment_b_Nm"], 1e-9, "spur tracking moment"),
                         "single_point_moment_change_Nm_at_same_force": math.dist(*point_moments)})
    return {"trace": trace.name, "seed_time_s": regions["initial"]["time_s"], "time_s": time,
            "cutoff_m": cutoff, "line_stiffness_N_per_m2": stiffness, "loads": measured, "statistics": stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("cam_models", type=Path)
    parser.add_argument("spur_models", type=Path)
    parser.add_argument("spur_trace", type=Path)
    parser.add_argument("output", type=Path, help="Prefix for the summary JSON and the diagnostic CSVs")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = {"eccentric_cam": cam_check(args.executable, args.cam_models, args.output.with_name(args.output.name + "-cam")),
               "spur_double_tooth": spur_check(args.executable, args.spur_models, args.spur_trace,
                                               args.output.with_name(args.output.name + "-spur"))}
    Path(str(args.output) + ".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
