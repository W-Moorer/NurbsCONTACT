"""Run one archived gear mechanism and record every stage on the shared CPU backend."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from cpu_mesh_memory import dynamics


def compare_trace(reference, data, stages, shafts):
    previous = np.genfromtxt(reference, names=True, delimiter=",")
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
            normal_impulse_difference_Ns=float(np.trapezoid(after - before, t)),
        ))
    return dict(
        reference_trace_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        frames=len(t), stages=loads,
        max_shaft_speed_difference_rad_s=[float(np.max(np.abs(data[f"v{i}"] - previous[f"v{i}"])))
                                          for i in shafts],
    )


def run(args):
    root = args.scene.parent
    with args.scene.with_suffix(".csv").open(newline="") as stream:
        parts = list(csv.DictReader(stream))
    meshes = []
    for part in parts:
        step = root / part["step"]
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
    report = dynamics(
        args.exe.resolve(), root, args.scene, args.method, args.dt, args.duration, args.work, args.scope
    )
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
                normal_impulse_Ns=float(np.trapezoid(force, data["time_s"])),
            )
        )
    report["simultaneously_loaded_frames"] = int(np.logical_and.reduce(activity).sum())
    report["shaft_speed_range_rad_s"] = [
        [float(data[f"v{i}"].min()), float(data[f"v{i}"].max())] for i in shafts
    ]
    report["meshes"] = meshes
    report["method"] = args.method
    if args.compare_trace:
        report["compared_with_previous_run"] = compare_trace(args.compare_trace, data, stages, shafts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("exe", "scene", "work", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--method", choices=["mesh-aabb", "mesh-obb", "mesh-sap"], default="mesh-obb")
    parser.add_argument("--scope", choices=["local", "full"], default="local")
    parser.add_argument("--dt", type=float, default=50e-6)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--compare-trace", type=Path, help="gear response on the same scene and time grid")
    run(parser.parse_args())
