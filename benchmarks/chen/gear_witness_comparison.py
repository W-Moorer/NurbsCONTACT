"""Compare separately timed gear runs on the same saved HHT grid."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_run(path):
    result = json.loads(path.read_text())
    trace = path.with_suffix("") / "response.csv"
    if hashlib.sha256(trace.read_bytes()).hexdigest() != result["trace_sha256"]:
        raise ValueError(f"trace differs from timing report: {trace}")
    data = np.genfromtxt(trace, delimiter=",", names=True)
    if len(data) != result["steps"] + 1:
        raise ValueError("expected a saved state per main step and the initial state")
    return result, data


def replay_entry(runs, scene, cad_probe, mesh_probe):
    bodies = [shlex.split(s) for s in scene.read_text().splitlines() if s.startswith("body ")]
    if len(bodies) != 2 or hashlib.sha256(scene.read_bytes()).hexdigest() != runs["before"][0]["scene_sha256"]:
        raise ValueError("expected the same two-body gear scene")
    axes = np.array([[float(x) for x in row[6:9]] for row in bodies])
    axes /= np.linalg.norm(axes, axis=1)[:,None]
    assets = [scene.parent / row[1] for row in bodies]
    splits = 6
    results = {"subdivisions_of_saved_interval":2**splits, "cad_distance_tolerance_m":1e-7,
               "scene_sha256":runs["before"][0]["scene_sha256"],
               "trace_sha256":{name:r["trace_sha256"] for name,(r,_) in runs.items()},
               "interpolation":"Linear shaft origins and fixed-axis angles between adjacent saved states."}
    def endpoints(data):
        active = data["p0_normal_N"] > 0
        entries = np.flatnonzero(~active[:-1] & active[1:])
        if len(entries) != 1:
            raise ValueError("expected one saved entry in the loading episode")
        idx = entries[0]
        frames = data[idx:idx+2]
        states = np.array([[[row[f"b{i}_{k}"] for k in ("x_m","y_m","z_m","angle_rad")]
                            for i in range(2)] for row in frames])
        if np.any(np.abs(states[1,:,3]-states[0,:,3]) >= np.pi):
            raise ValueError("endpoint angles do not describe a short shaft rotation")
        return frames["time_s"], states
    def input_lines(rows):
        return "".join(" ".join(format(x, ".17g") for x in row)+"\n" for row in rows)
    for name in ("before","ours"):
        ts, states = endpoints(runs[name][1])
        if name == "ours" and np.array_equal(states, previous_states) and np.array_equal(ts, previous_ts):
            results["ours_path"] = {"identical_to":"before_path"}
            continue
        poses = [[*s[:3], *axes[i], s[3]] for state in states for i,s in enumerate(state)]
        child = subprocess.run([cad_probe,*assets,"--entry"], input=input_lines(poses)+f"0.0001 0.5 {splits}\n",
                               capture_output=True, text=True, timeout=1800)
        if child.returncode:
            raise RuntimeError(child.stderr)
        rows = list(csv.DictReader(io.StringIO(child.stdout)))
        if {r["start"] for r in rows} != {"cold","previous_feature","whole_solid_load"}:
            raise ValueError("incomplete CAD entry replay")
        results[name+"_path"] = {r["start"]:{
            "bracket_s":[float(ts[0]+float(r[k])*(ts[1]-ts[0])) for k in ("lo","hi")],
            "extrema":int(r["extrema"]),"witness_tests":int(r["witness_tests"]),"status":r["status"]} for r in rows}
        previous_ts, previous_states = ts, states
    ts, states = endpoints(runs["mesh"][1])
    fractions = np.linspace(0,1,2**splits+1)
    poses = []
    for f in fractions:
        frame = []
        for i, s in enumerate((1-f)*states[0]+f*states[1]):
            frame.extend([*s[:3], np.cos(s[3]/2), *(axes[i]*np.sin(s[3]/2))])
        poses.append(frame)
    child = subprocess.run([mesh_probe,"--method","mesh-obb","--regions","local",*[p.with_suffix(".msh") for p in assets]],
                           input=input_lines(poses), text=True, capture_output=True, timeout=1800)
    if child.returncode:
        raise RuntimeError(child.stderr)
    active = np.array([bool(json.loads(s)["regions"]) for s in child.stdout.splitlines()])
    if len(active)!=len(fractions) or active[0] or not active[-1] or np.count_nonzero(active[1:] != active[:-1]) != 1:
        raise ValueError("mesh replay did not contain a single load transition")
    idx = int(np.flatnonzero(active)[0])
    results["mesh_path"] = {"volume_load_bracket_s":(ts[0]+(ts[1]-ts[0])*fractions[idx-1:idx+1]).tolist()}
    return results


def compare(paths, output, replay=None, repeats=None):
    runs = {name: load_run(path) for name, path in paths.items()}
    reference, ref = runs["before"]
    if [r["method"] for r, _ in runs.values()] != ["cad-occt", "cad-occt", "mesh-obb"]:
        raise ValueError("expected the STEP control, updated STEP and mesh OBB")
    step_assets = reference["step_assets"]
    if runs["ours"][0]["step_assets"] != step_assets:
        raise ValueError("STEP control and updated solver used different CAD solids")
    mesh_sources = [item["step_sha256"] for item in runs["mesh"][0]["meshes"]]
    if mesh_sources != [item["step_sha256"] for item in step_assets]:
        raise ValueError("mesh control does not represent the same CAD solids")
    time = ref["time_s"]
    summary = {}
    columns = {"time_s": time}
    fig, (loads, residuals) = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    styles = [("STEP before", "#666666", "o", "-"), ("STEP (ours)", "#006e82", "x", ":"),
              ("Mesh OBB", "#aa4926", "s", "--")]
    for (name, (r, data)), (label, color, marker, line) in zip(runs.items(), styles):
        for key in ("scene_sha256", "duration_s", "dt_s", "event_tolerance_s", "solver"):
            if r[key] != reference[key]:
                raise ValueError(f"mismatched {key} in {name}")
        if data.shape != ref.shape or not np.allclose(data["time_s"], time, rtol=0, atol=8*np.spacing(time[-1])):
            raise ValueError("output grids differ")
        force = data["p0_normal_N"]
        delta = force-ref["p0_normal_N"]
        active = force > 0
        events = [dict(kind="load" if active[i] else "release", left_s=float(time[i-1]), right_s=float(time[i]))
                  for i in range(1, len(time)) if active[i] != active[i-1]]
        summary[name] = dict(measurement=r, saved_switch_intervals=events,
                             loaded_frames_different_from_before=int(np.count_nonzero(active != (ref["p0_normal_N"] > 0))),
                             maximum_force_difference_from_before_N=float(np.max(np.abs(delta))),
                             maximum_speed_differences_from_before_rad_s=[
                                 float(np.max(np.abs(data[k]-ref[k]))) for k in ("v0","v1")])
        for k in ("p0_normal_N", "v0", "v1"):
            columns[f"{name}_{k}"] = data[k]
        loads.plot(time*1e3, force/1000, color=color, ls=line, marker=marker, mfc="none", ms=5, label=label)
        if name != "before":
            residuals.plot(time*1e3, delta, color=color, ls=line, marker=marker, mfc="none", ms=5, label=label)
    if repeats:
        for name, path in zip(("before", "ours", "mesh"), repeats):
            repeated, _ = load_run(path)
            first = runs[name][0]
            assets = "meshes" if name == "mesh" else "step_assets"
            for key in ("method", "scene_sha256", assets, "dt_s", "duration_s",
                        "event_tolerance_s", "solver", "trace_sha256"):
                if repeated[key] != first[key]:
                    raise ValueError(f"{name} repeat differs in {key}")
            summary[name]["repeat_measurement"] = repeated
    output.mkdir(parents=True, exist_ok=True)
    (output/"runs.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    with (output/"responses.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))
    for ax, ylabel in ((loads, "Normal force (kN)"), (residuals, "Force minus STEP before (N)")):
        ax.set(xlabel="Simulated time (ms)", ylabel=ylabel, xticks=[0,.1,.2,.3,.35])
        ax.spines[["right","top"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.legend(frameon=False)
    fig.savefig(output/"loads.png", dpi=180)
    plt.close(fig)
    if replay:
        entry = replay_entry(runs, *replay)
        (output/"entry.json").write_text(json.dumps(entry, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("before", "ours", "mesh", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--replay", nargs=3, type=Path, metavar=("SCENE","CAD_PROBE","MESH_PROBE"))
    parser.add_argument("--repeat", nargs=3, type=Path, metavar=("STEP_BEFORE","STEP_AFTER","MESH"))
    args = parser.parse_args()
    compare({k:getattr(args,k) for k in ("before","ours","mesh")}, args.output, args.replay, args.repeat)
