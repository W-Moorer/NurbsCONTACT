"""Run one archived gear mechanism and record every stage on the shared CPU backend."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from cpu_mesh_memory import dynamics


def normal_impulse(force, time, solver):
    if solver == "penalty":
        return float(np.trapezoid(force, time))
    return float(np.diff(time) @ force[1:])


def compare_trace(reference, data, stages, shafts, solver):
    previous = np.genfromtxt(reference, names=True, delimiter=",")
    previous_solver = solver
    metadata = reference.with_suffix(".stats.csv")
    if metadata.is_file():
        with metadata.open(newline="") as stream:
            previous_solver = next(csv.DictReader(stream)).get("solver", "penalty")
    t = data["time_s"]
    tolerance = 8 * np.finfo(float).eps * max(abs(t[-1]), abs(previous["time_s"][-1]))
    if previous.shape != data.shape or not np.allclose(previous["time_s"], t, rtol=0, atol=tolerance):
        raise ValueError("gear responses have different output times")
    loads = []
    for stage in stages:
        column = f"p{stage-1}_normal_N"
        before, after = previous[column], data[column]
        loads.append(dict(
            stage=stage,
            different_active_frames=int(np.count_nonzero((before > 0) != (after > 0))),
            max_force_difference_N=float(np.max(np.abs(after - before))),
            normal_impulse_difference_Ns=normal_impulse(after, t, solver) - normal_impulse(before, t, previous_solver),
        ))
    return dict(
        reference_trace_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        frames=len(t), stages=loads, reference_solver=previous_solver,
        max_shaft_speed_difference_rad_s=[float(np.max(np.abs(data[f"v{i}"] - previous[f"v{i}"])))
                                          for i in shafts],
    )


def run(args):
    root = args.scene.parent
    with args.scene.with_suffix(".csv").open(newline="") as stream:
        parts = list(csv.DictReader(stream))
    meshes = []
    cad = args.method in ("cad-occt","chen-2024","chen-2026")
    step_assets = []
    for part in parts:
        step = root / part["step"]
        if cad:
            step_assets.append(dict(asset=step.name, step_sha256=hashlib.sha256(step.read_bytes()).hexdigest()))
            continue
        mesh = step.with_suffix(".msh")
        meta = json.loads(mesh.with_suffix(".quality.json").read_text())
        for path, field in ((step, "step_sha256"), (mesh, "mesh_sha256")):
            if hashlib.sha256(path.read_bytes()).hexdigest() != meta[field]:
                raise ValueError(f"{path.name} differs from its mesh quality report")
        fields = (
            "triangles",
            "step_sha256",
            "mesh_sha256",
            "max_normal_angle_deg",
            "max_sampled_surface_error_m",
            "min_sicn",
        )
        meshes.append({"asset": mesh.name, "file_bytes": mesh.stat().st_size, **{key: meta[key] for key in fields}})
    start = time.perf_counter()
    try:
        report = dynamics(
            args.exe.resolve(), None if cad else root, args.scene, args.method, args.dt, args.duration, args.work,
            None if cad else args.scope, args.event_tol_s, args.solver
        )
    except TimeoutError:
        stopped = dict(completed=False, stop_reason="timeout", time_limit_s=1800,
                       process_s=time.perf_counter() - start, duration_s=args.duration, dt_s=args.dt,
                       event_tolerance_s=args.event_tol_s or (.2 * args.dt if args.method.startswith("chen-") else 0.),
                       method=args.method, solver=args.solver, scene_sha256=hashlib.sha256(args.scene.read_bytes()).hexdigest())
        stopped["step_assets" if cad else "meshes"] = step_assets if cad else meshes
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(stopped, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        raise
    data = np.genfromtxt(args.work / "response.csv", names=True, delimiter=",")
    stages = sorted({int(part["stage"]) for part in parts})
    shafts = sorted({int(part["shaft"]) for part in parts})
    activity = []
    report["stages"] = []
    for stage in stages:
        force = data[f"p{stage-1}_normal_N"]
        if not np.all(np.isfinite(force)):
            raise ValueError(f"nonfinite contact force for stage {stage}")
        activity.append(force > 0)
        report["stages"].append(
            dict(
                stage=stage,
                active_frames=int(activity[-1].sum()),
                peak_normal_N=float(force.max()),
                normal_impulse_Ns=normal_impulse(force, data["time_s"], args.solver),
            )
        )
    report["simultaneously_loaded_frames"] = int(np.logical_and.reduce(activity).sum())
    report["shaft_speed_range_rad_s"] = [
        [float(data[f"v{i}"].min()), float(data[f"v{i}"].max())] for i in shafts
    ]
    if cad:
        report["step_assets"] = step_assets
    else:
        report["meshes"] = meshes
    report["method"] = args.method
    report["solver"] = args.solver
    if args.compare_trace:
        report["compared_with_previous_run"] = compare_trace(args.compare_trace, data, stages, shafts, args.solver)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("exe", "scene", "work", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--method", choices=["mesh-aabb", "mesh-obb", "mesh-sap", "cad-occt", "chen-2024", "chen-2026"], default="mesh-obb")
    parser.add_argument("--event-tol-s", type=float, default=0.)
    parser.add_argument("--solver", choices=("penalty","sap","lagrange","augmented-lagrange","lcp","ncp","soccp"),
                        default="penalty")
    parser.add_argument("--scope", choices=["local", "full"], default="local")
    parser.add_argument("--dt", type=float, default=50e-6)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--compare-trace", type=Path, help="gear response on the same scene and time grid")
    run(parser.parse_args())
