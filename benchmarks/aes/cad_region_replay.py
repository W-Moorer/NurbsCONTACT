"""Compare cached CAD region queries on recorded entry and exit windows."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import subprocess

from cad_contact_regions import read_rows
from gear_contact_paths import angle


def window_frames(trace, radius, count, transitions):
    rows = read_rows(trace)
    counts = list(map(count, rows))
    windows = {}
    for previous, current in transitions:
        i = next(i for i in range(radius, len(rows) - radius)
                 if (counts[i - 1], counts[i]) == (previous, current))
        windows[f"{previous}_to_{current}"] = rows[i-radius:i+radius+1]
    return windows


def write_poses(windows, a, b, output):
    x = (float(b["center_x_mm"]) - float(a["center_x_mm"])) * .001
    z = (float(b["z_mm"]) - float(a["z_mm"])) * .001
    with output.open("w", encoding="ascii") as stream:
        for frames in windows.values():
            for row in frames:
                a0, b0 = angle(a, row), angle(b, row)
                pose = (row["time"], x * math.cos(a0), -x * math.sin(a0), z, 0, 0, 1, b0 - a0)
                print(*pose, file=stream)


def read_report(prefix):
    return tuple(read_rows(Path(str(prefix) + suffix)) for suffix in (".csv", ".loads.csv", ".stats.csv"))


def totals(stats):
    return {
        **{key: sum(int(row.get(key, 0)) for row in stats)
           for key in ("tested_face_pairs", "reused_face_exclusions", "attempted_face_exclusions",
                       "unresolved_face_pairs", "unresolved_line_pairs")},
        **{key: sum(float(row[key]) for row in stats) for key in ("query_s", "assembly_s")},
        "setup_s": float(stats[0]["setup_s"]),
    }


def evaluate(cold, reused, frames):
    witnesses, loads, stats = cold
    warm_witnesses, warm_loads, warm_stats = reused
    if witnesses != warm_witnesses or loads != warm_loads:
        raise ValueError("Reuse changed the candidates or the assembled line wrenches")
    if len(stats) != len(warm_stats) or len(stats) != sum(map(len, frames.values())):
        raise ValueError("The replay did not evaluate all requested poses")
    for base, current in zip(stats, warm_stats, strict=True):
        if base["time_s"] != current["time_s"]:
            raise ValueError("The replay modes disagree on the sample time")
        for key in ("unresolved_face_pairs", "unresolved_line_pairs"):
            if base[key] != current[key]:
                raise ValueError(f"Reuse changed {key} at {base['time_s']} s")
        if int(base["tested_face_pairs"]) != int(current["tested_face_pairs"]) + int(current["reused_face_exclusions"]):
            raise ValueError("Reused exclusions do not account for skipped face queries")
    ref, tracked = totals(stats), totals(warm_stats)
    if ref["unresolved_face_pairs"] or ref["unresolved_line_pairs"]:
        raise ValueError("A CAD query or line assembly was unresolved in the replay windows")
    per_pose = [
        {"time_s": float(base["time_s"]), "cold_query_s": float(base["query_s"]),
         "reuse_query_s": float(current["query_s"]),
         "reused_face_exclusions": int(current["reused_face_exclusions"]),
         "attempted_face_exclusions": int(current["attempted_face_exclusions"])}
        for base, current in zip(stats, warm_stats, strict=True)
    ]
    return {"poses": len(stats), "identical_candidate_and_load_rows": True, "cold": ref, "reuse": tracked,
            "query_speedup": ref["query_s"] / tracked["query_s"], "per_pose": per_pose}


def spur_events(windows, witnesses, loads):
    facing, assembled = defaultdict(set), defaultdict(set)
    for row in witnesses:
        if row["facing_interiors"] == "1":
            facing[float(row["time_s"])].add((int(row["face_a"]), int(row["face_b"])))
    for row in loads:
        assembled[float(row["time_s"])].add((int(row["face_a"]), int(row["face_b"])))
    events = {}
    for label, frames in windows.items():
        previous, summaries = None, []
        for row in frames:
            time = float(row["time"])
            pairs = facing[time]
            summaries.append({"time_s": time, "gear_trace_loaded_teeth": int(float(row["active0"])),
                              "cad_facing_face_pairs": sorted(pairs), "assembled_line_pairs": sorted(assembled[time]),
                              "new_face_pairs": sorted(pairs - previous) if previous is not None else [],
                              "departing_face_pairs": sorted(previous - pairs) if previous is not None else []})
            previous = pairs
        events[label] = summaries
    if (not any(row["new_face_pairs"] for row in events["1_to_2"]) or
            not any(row["departing_face_pairs"] for row in events["2_to_1"])):
        raise ValueError("The sampled windows did not capture both CAD face-pair transitions")
    return events


def run_replays(exe, a, b, cutoff, stiffness, poses, output):
    for mode in ("cold", "reuse"):
        subprocess.run([exe.resolve(), "regions-replay", a, b, str(cutoff), str(stiffness),
                        poses, output / mode, mode], check=True)
    return read_report(output / "cold"), read_report(output / "reuse")


def cam_replay(exe, models, trace, output, radius):
    output.mkdir(parents=True, exist_ok=True)
    windows = window_frames(trace, radius, lambda row: int(float(row["solver_peak_force"]) > 0), ((0, 1), (1, 0)))
    poses = output / "cam-poses.txt"
    with poses.open("w", encoding="ascii") as stream:
        for frames in windows.values():
            for row in frames:
                print(row["time"], 0, -.115 - float(row["q1"]), 0, 0, 0, 1, row["q0"], file=stream)
    cutoff, stiffness = .012, 5000 / .060
    cold, reuse = run_replays(exe, models / "follower.step", models / "cam.step", cutoff, stiffness, poses, output)
    report = evaluate(cold, reuse, windows)
    loads = defaultdict(list)
    for row in cold[1]:
        loads[float(row["time_s"])].append(row)
    frames_out = {}
    for label, frames in windows.items():
        frames_out[label] = []
        for frame in frames:
            time, q0, q1 = (float(frame[key]) for key in ("time", "q0", "q1"))
            reference_force = 5000 * max(0, cutoff - (.010 + q1 - .024 * math.sin(q0)))
            if len(loads[time]) != int(reference_force > 0):
                raise ValueError(f"Cam active-set mismatch at {time} s")
            force = sum(-float(load["force_by_N"]) for load in loads[time])
            moment = sum(float(load["moment_bz_Nm"]) for load in loads[time])
            force_error = abs(force - reference_force)
            moment_error = abs(moment + reference_force * .024 * math.cos(q0))
            if force_error > 1e-8 or moment_error > 1e-9:
                raise ValueError(f"Cam wrench mismatch at {time} s")
            frames_out[label].append({"time_s": time, "contact": bool(reference_force), "force_N": force,
                                      "force_error_N": force_error, "moment_error_Nm": moment_error})
    return {"trace": trace.name, "cutoff_m": cutoff, "line_stiffness_N_per_m2": stiffness,
            **report, "windows": frames_out}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("models", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--radius", type=int, default=2, help="Frames before and after each transition")
    parser.add_argument("--cam-models", type=Path)
    parser.add_argument("--cam-trace", type=Path)
    args = parser.parse_args()
    if args.radius < 1:
        parser.error("--radius must be at least 1")
    if (args.cam_models is None) != (args.cam_trace is None):
        parser.error("--cam-models and --cam-trace must be used together")
    args.output.mkdir(parents=True, exist_ok=True)
    a, b = read_rows(args.models / "spur.csv")
    windows = window_frames(args.trace, args.radius, lambda row: int(float(row["active0"])), ((1, 2), (2, 1)))
    poses = args.output / "spur-poses.txt"
    write_poses(windows, a, b, poses)
    cutoff, stiffness = 55e-6, 1e9
    cold, reuse = run_replays(args.executable, args.models / a["step"], args.models / b["step"],
                             cutoff, stiffness, poses, args.output)
    stats = evaluate(cold, reuse, windows)
    if not stats["reuse"]["reused_face_exclusions"]:
        raise ValueError("The spur replay did not exercise any cached exclusions")
    report = {"trace": args.trace.name, "cutoff_m": cutoff, "line_stiffness_N_per_m2": stiffness,
              **stats, "windows": spur_events(windows, *cold[:2])}
    if args.cam_models is not None:
        report["cam"] = cam_replay(args.executable, args.cam_models, args.cam_trace, args.output / "cam", args.radius)
    (args.output / "cad-region-replay.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
