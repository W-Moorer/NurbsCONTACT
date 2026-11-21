"""Validate HHT measurements and aggregate the recorded trajectories."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from run import CASES


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_trace(root, label):
    return np.genfromtxt(root / "runs" / f"{label}.csv", delimiter=",", names=True)


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def drop_reference(t):
    skin, radius, rate, frequency = .012, .055, .45, np.sqrt(5000.)
    entry = .018/rate
    release = entry + np.pi/frequency
    z = np.where(t < entry, radius+skin+.018-rate*t, radius+skin+rate*(t-release))
    inside = (t >= entry) & (t < release)
    z[inside] = radius + skin - rate/frequency*np.sin(frequency*(t[inside]-entry))
    force = 5000*np.maximum(0, radius+skin-z)
    return z, force


def pose_error(trace, reference, scene):
    q = [x for x in trace.dtype.names if x.startswith("q")]
    indices = range(1, len(q), 2) if scene == "cam-array" else range(len(q))
    differences = np.array([
        trace[q[i]] - np.interp(trace["time"], reference["time"], reference[q[i]]) for i in indices
    ])
    force = trace["solver_peak_force"] - np.interp(
        trace["time"], reference["time"], reference["solver_peak_force"]
    )
    result = {
        "rms_translation_mm": rms(differences)*1000,
        "max_translation_mm": float(np.max(np.abs(differences))*1000),
        "rms_peak_force_N": rms(force),
    }
    if scene == "cam-array":
        angles = np.array([
            trace[q[i]]-np.interp(trace["time"],reference["time"],reference[q[i]])
            for i in range(0,len(q),2)
        ])
        result["rms_angle_deg"] = rms(angles)*180/np.pi
    return result


def queries(root, name, scale):
    raw = read_rows(root/"metrics"/f"{name}.csv")
    expected = 1812 if name == "cam-queries" else 125
    if len(raw) != expected:
        raise ValueError(f"incomplete {name}")
    key = "facets" if name == "cam-queries" else "deflection"
    groups = defaultdict(list)
    for row in raw:
        groups[(row["backend"], row[key])].append(row)
    out = []
    for (backend,parameter), items in groups.items():
        resolved = [r for r in items if r["status"]=="ok"]
        errors = np.array([float(r["gap"])-float(r["reference"]) for r in resolved])*scale
        if name == "trimmed-queries" and backend == "brep":
            for r in items:
                value = float(r["reference"])
                if not float(r["lower"])-1e-10 <= value <= float(r["gap"])+1e-10:
                    raise ValueError("trimmed B-rep interval disagrees with the analytic reference")
        out.append({
            "backend":backend, "parameter":float(parameter), "cases":len(items), "resolved":len(resolved),
            "median_us":statistics.median(float(r["latency_us"]) for r in items),
            "rms_error_mm":rms(errors) if len(errors) else None,
            "max_error_mm":float(np.max(np.abs(errors))) if len(errors) else None,
            "triangles":int(items[0]["triangles"]) if name == "trimmed-queries" else None,
        })
    return out


def production_queries(root):
    groups = defaultdict(list)
    files = [root/"metrics"/f"production-{name}.csv" for name in ("wheel", "lidar")]
    if not any(path.exists() for path in files):
        return []
    for path, expected in zip(files, (204, 68), strict=True):
        rows = read_rows(path)
        if len(rows) != expected:
            raise ValueError(f"Incomplete production clearance data in {path}")
        for row in rows:
            key = row["scene"], row["backend"], float(row["deflection_mm"])
            groups[key].append(row)
    results = []
    for (scene, backend, deflection), rows in groups.items():
        if len(rows) != 17 or len({float(r["time"]) for r in rows}) != 17:
            raise ValueError(f"Unexpected pose counts for {scene} {backend}")
        resolved = [r for r in rows if r["status"] == "ok"]
        error = [abs(float(r["upper_mm"])-float(r["reference_mm"])) for r in resolved]
        missed_oracle = None
        if backend == "brep":
            missed_oracle = sum(
                not float(r["lower_mm"])-1e-6 <= float(r["reference_mm"]) <= float(r["upper_mm"])+1e-6
                for r in rows
            )
        results.append({
            "scene": scene, "backend": backend, "deflection_mm": deflection,
            "poses": len(rows), "resolved": len(resolved),
            "median_us": statistics.median(float(r["query_us"]) for r in rows),
            "median_resolved_us": statistics.median(float(r["query_us"]) for r in resolved) if resolved else None,
            "max_error_mm": max(error) if error else None, "oracle_outside_interval": missed_oracle,
            "triangles": int(rows[0]["triangles"]),
        })
    return results


def summarize(root):
    oscillator = read_rows(root/"metrics"/"oscillator.csv")
    rms_error = np.array([float(r["rms_displacement"]) for r in oscillator])
    order = list(np.log2(rms_error[:-1]/rms_error[1:]))
    if len(oscillator) != 4 or min(order) < 1.9:
        raise ValueError("oscillator convergence regression")
    raw = {}
    traces = {}
    for case in CASES:
        label = case.label
        trace = load_trace(root,label)
        rows = read_rows(root/"runs"/f"{label}.stats.csv")
        steps = round(case.seconds/case.dt)
        if len(rows)!=case.repeats or len(trace)!=steps+1 or abs(trace["time"][-1]-case.seconds)>1e-10:
            raise ValueError(f"incomplete trajectory {label}")
        if not all(np.isfinite(trace[k]).all() for k in trace.dtype.names):
            raise ValueError(f"nonfinite trajectory {label}")
        if not np.all(np.diff(trace["time"])>0):
            raise ValueError(f"nonmonotonic trajectory {label}")
        guards = [float(r["min_guard"]) for r in rows]
        if min(guards) <= 0.0005:
            raise ValueError(f"invalid endpoint-motion guard for {label}")
        if any(int(r["accepted"])<steps for r in rows):
            raise ValueError(f"substep inconsistency for {label}")
        times = [float(r["wall_us"])/1e6 for r in rows]
        raw[label] = {
            "scene":case.scene, "backend":case.backend, "dt":case.dt, "duration":case.seconds,
            "steps":steps, "repeats":len(rows), "median_s":statistics.median(times),
            "min_s":min(times), "max_s":max(times), "min_guard_mm":1000*min(guards),
            "min_pose_gap_mm":1000*float(np.min(trace["cad_gap"])),
            "peak_force_N":float(np.max(trace["solver_peak_force"])),
            "max_residual":max(float(r["max_residual"]) for r in rows),
            "substeps":int(rows[0]["subdivisions"]), "queries":int(rows[0]["queries"]),
            "triangles":int(rows[0]["triangles"]), "primitive_tests":int(rows[0]["triangle_tests"]),
            "guard_endpoints":int(rows[0]["guard_endpoint_evaluations"]),
            "cad_fallback":int(rows[0]["cad_fallback"]),
        }
        if case.scene == "drop":
            exact, force = drop_reference(trace["time"])
            raw[label].update(rms_to_exact_mm=rms(trace["q0"]-exact)*1000,
                              rms_force_to_exact_N=rms(trace["solver_peak_force"]-force))
        traces[label] = trace
    for label in ["cam-brep","cam-mesh-12","cam-mesh-24","cam-mesh-48","cam-mesh-96"]:
        raw[label]["fine_reference"] = pose_error(traces[label],traces["cam-brep-finest"],"cam-array")
        if label!="cam-brep":
            raw[label]["matched_reference"] = pose_error(traces[label],traces["cam-brep"],"cam-array")
    raw["train-brep"]["fine_reference"] = pose_error(traces["train-brep"],traces["train-brep-finest"],"ball-train")
    refinement = {}
    for prefix, scene in (("cam", "cam-array"), ("train", "ball-train")):
        reference = traces[f"{prefix}-brep-finest"]
        refinement[scene] = {
            "5ms": pose_error(traces[f"{prefix}-brep"], reference, scene),
            "2.5ms": pose_error(traces[f"{prefix}-brep-fine"], reference, scene),
        }
        for run in (f"{prefix}-brep", f"{prefix}-brep-fine"):
            track = traces[run]
            refinement[scene][run+"_first_5s"] = pose_error(track[track["time"] <= 5], reference, scene)
    train_mesh = ["train-mesh-24","train-mesh-24-cold"]
    for label in train_mesh:
        raw[label]["fine_reference"] = pose_error(traces[label],traces["train-brep-finest"],"ball-train")
        raw[label]["matched_reference"] = pose_error(traces[label],traces["train-brep"],"ball-train")
    raw["cache-off"]["matched_reference"] = pose_error(traces["cache-off"],traces["cache-on"],"cam-array")
    raw["cold-mesh"]["matched_reference"] = pose_error(traces["cold-mesh"],traces["warm-mesh"],"ball-train")
    if raw["cache-off"]["matched_reference"]["max_translation_mm"] > 1e-9:
        raise ValueError("guard reuse changed the accepted trajectory")
    # Use the absolute, unit-based 1 µm trajectory agreement criterion.
    if any(raw[label]["matched_reference"]["max_translation_mm"] > 0.001 for label in train_mesh):
        raise ValueError("ball-train comparison exceeds 1 µm maximum trajectory disagreement")
    best_train_mesh = min(train_mesh,key=lambda label:raw[label]["median_s"])
    allocation = {
        "train_speedup":raw[best_train_mesh]["median_s"]/raw["train-brep"]["median_s"],
        "train_mesh_setting":best_train_mesh,
        "cam_96_speedup":raw["cam-mesh-96"]["median_s"]/raw["cam-brep"]["median_s"],
        "guard_cache_speedup":raw["cache-off"]["median_s"]/raw["cache-on"]["median_s"],
        "mesh_hint_speedup":raw["cold-mesh"]["median_s"]/raw["warm-mesh"]["median_s"],
    }
    if allocation["train_speedup"] <= 1:
        raise ValueError("matched-accuracy comparison did not exhibit a speedup")
    return {
        "oscillator_order":order,
        "cam_queries":queries(root,"cam-queries",1000),
        "trimmed_queries":queries(root,"trimmed-queries",1),
        "production_queries":production_queries(root),
        "runs":raw,"speedups":allocation,"temporal_refinement":refinement,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path)
    args=parser.parse_args()
    result=summarize(args.results)
    path=args.results/"metrics"/"summary.json"
    path.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result["speedups"],indent=2))
    print(f"Validated {len(result['runs'])} trajectories")


if __name__ == "__main__":
    main()
