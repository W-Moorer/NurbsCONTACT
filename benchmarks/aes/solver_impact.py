"""Switch contact laws on the archived two-slider impact, with one scene for all methods."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/engines"))
from rigid import DURATION, LEGACY, SPEED, STANDOFF, exact

SOURCE = ROOT / "results/engine-comparison/nurbs/spheres.scene"
LAWS = ("penalty", "sap", "lagrange", "augmented-lagrange", "lcp", "ncp", "soccp")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args, scene, law):
    prefix = args.work / law
    cmd = [args.exe.resolve(), scene, "0", str(args.dt), ".16", "1", prefix, "--method", args.method,
           "--solver", law, "--event-tol-s", str(args.event_tol)]
    try:
        child = subprocess.run(cmd, text=True, capture_output=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return dict(solver=law, completed=False, time_limit_s=1800, reason="timeout")
    if child.returncode:
        return dict(solver=law, completed=False, reason=child.stderr.strip(), returncode=child.returncode)
    data = np.genfromtxt(prefix.with_suffix(".csv"), names=True, delimiter=",")
    with prefix.with_suffix(".stats.csv").open() as stream:
        stats = next(csv.DictReader(stream))
    t, v0, v1, f = (data[name] for name in ("time_s", "v0", "v1", "p0_normal_N"))
    positions = np.column_stack([data["q0"], data["q1"]])
    if (len(t) != round(DURATION / args.dt) + 1 or abs(t[-1] - DURATION) > 1e-10
            or not np.isfinite(positions).all() or not np.isfinite([v0, v1, f]).all()):
        raise ValueError("incomplete or nonfinite collision")
    if f.min()<0 or not np.any(f>0) or stats["solver"]!=law:
        raise ValueError("unexpected solver or contact impulse")
    if args.method.startswith("chen-") and int(stats["chen_patch_queries"])==0:
        raise ValueError("Chen sweep did not run")
    if args.method=="chen-2026" and int(stats["tdib_cells"])==0:
        raise ValueError("safeguarded envelope search did not run")
    # Two unit masses, zero initial speed on the first, -0.45 m/s on the second.
    if law in ("penalty","sap"):
        reference, _ = exact("spheres", t, LEGACY)
        impulse_ref, end_v = SPEED, [-SPEED, 0.]
    else:
        q0 = -.5 * SPEED * np.maximum(t - STANDOFF / SPEED, 0.)
        reference = np.column_stack([q0, -SPEED * t - q0])
        impulse_ref, end_v = SPEED / 2, [-SPEED / 2] * 2
    impulse = np.trapezoid(f,t) if law=="penalty" else np.diff(t) @ f[1:]
    return dict(solver=law, completed=True, frames=len(t), final_velocity_m_s=[v0[-1],v1[-1]],
                reference_final_velocity_m_s=end_v, normal_impulse_Ns=float(impulse),
                reference_impulse_Ns=impulse_ref, max_momentum_drift_kg_m_s=float(np.abs(v0+v1+SPEED).max()),
                max_displacement_error_m=float(np.abs(positions-reference).max()),
                peak_normal_N=float(f.max()), integration_s=float(stats["wall_s"]),
                impulse_residual_m_s=float(stats["impulse_residual_m_s"]),
                iterations=int(stats["impulse_iterations"]), ccd_checks=int(stats["ccd_checks"]),
                chen_patch_queries=int(stats["chen_patch_queries"]), trace_sha256=digest(prefix.with_suffix(".csv")),
                **{k:int(stats[k]) for k in ("tdib_cells","tdib_candidates","tdib_unresolved") if k in stats})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("exe","work","output"):
        parser.add_argument(key,type=Path)
    parser.add_argument("--method",choices=("mesh-aabb","mesh-obb","mesh-sap","cad-occt","chen-2024","chen-2026"),required=True)
    parser.add_argument("--mesh-dir",type=Path)
    parser.add_argument("--solvers",nargs="+",choices=LAWS,default=LAWS)
    parser.add_argument("--dt",type=float,default=.00025)
    parser.add_argument("--event-tol",type=float,default=.00005)
    args=parser.parse_args()
    args.work.mkdir(parents=True,exist_ok=True)
    scene=args.work/SOURCE.name
    step=scene.with_name("offset-sphere.step")
    shutil.copyfile(SOURCE,scene)
    shutil.copyfile(SOURCE.with_name(step.name),step)
    report=dict(method=args.method, dt_s=args.dt, event_tolerance_s=args.event_tol, duration_s=.16,
                scene_sha256=digest(scene),step_sha256=digest(step))
    if args.method.startswith("mesh-"):
        if not args.mesh_dir: parser.error("mesh methods require --mesh-dir")
        mesh=step.with_suffix(".msh")
        quality=json.loads((args.mesh_dir/mesh.with_suffix(".quality.json").name).read_text())
        shutil.copyfile(args.mesh_dir/mesh.name,mesh)
        if digest(mesh)!=quality["mesh_sha256"] or digest(step)!=quality["step_sha256"]:
            raise ValueError("input geometry differs from the mesh quality record")
        report.update(triangles_per_sphere=quality["triangles"],mesh_sha256=quality["mesh_sha256"])
    report["runs"]=[run(args,scene,solver) for solver in args.solvers]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    raise SystemExit(int(not all(r["completed"] for r in report["runs"])))
