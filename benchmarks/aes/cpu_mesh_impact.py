"""Run the archived two-slider impact with mesh-driven HHT and independent equal-mass reference."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/engines"))
from rigid import DURATION, LEGACY, SPEED, STANDOFF, STIFFNESS, case_values, exact


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare(exe, mesh_exe, mesh, method, step, work, event_tol=0.):
    scene = ROOT / "results/engine-comparison/nurbs/spheres.scene"
    source = scene.with_name("offset-sphere.step")
    quality = json.loads(mesh.with_suffix(".quality.json").read_text())
    if quality["step_sha256"] != digest(source) or quality["mesh_sha256"] != digest(mesh):
        raise ValueError("mesh and quality report must match the archived contact geometry")
    if mesh.name != source.with_suffix(".msh").name:
        raise ValueError("use a separate mesh directory for each fidelity, with the scene's asset name")
    work.mkdir(parents=True, exist_ok=True)
    prefix = work / "impact"
    start = time.perf_counter()
    command = [exe, scene, "0", str(step), str(DURATION), "1", prefix, "--method", method,
               "--solver", "penalty", "--mesh-dir", mesh.parent]
    if event_tol:
        command.extend(["--event-tol-s", str(event_tol)])
    run = subprocess.run(command,capture_output=True,text=True,timeout=1800)
    if run.returncode:
        raise RuntimeError(f"nc_contact exited {run.returncode}: {run.stderr.strip()}")
    elapsed = time.perf_counter() - start
    data = np.genfromtxt(prefix.with_suffix(".csv"), names=True, delimiter=",")
    with prefix.with_suffix(".stats.csv").open() as stream:
        stats = next(csv.DictReader(stream))
    t = data["time_s"]
    if len(t) != round(DURATION / step) + 1 or abs(t[-1] - DURATION) > 1e-10:
        raise ValueError("incomplete impact trace")
    force = data["p0_normal_N"]
    q = np.column_stack([data["q0"], data["q1"]])
    v = np.column_stack([data["v0"], data["v1"]])
    if not np.isfinite(q).all() or not np.isfinite(v).all() or not np.isfinite(force).all():
        raise ValueError("nonfinite dynamics output")
    transitions = np.flatnonzero(np.diff((force > 0).astype(int)))
    if len(transitions) != 2 or force[0] or force[-1]:
        raise ValueError("expected one complete loading and unloading interval")
    height, reduced_mass, _ = case_values("spheres", LEGACY)
    z = height + q[:, 1] - q[:, 0]
    def surface_event(i, j):
        poses = "".join(f"0 0 0 1 0 0 0 0 0 {z[k]:.17g} 1 0 0 0\n" for k in (i,j))
        result = subprocess.run([mesh_exe, "--method", method, "--detection", "continuous", mesh, mesh],
                                input=poses, capture_output=True, text=True, check=True,
                                timeout=max(.001,1800-(time.perf_counter()-start)))
        hit = json.loads(result.stdout)["candidate"]
        if hit is None:
            raise ValueError("force changed state without a mesh surface crossing")
        return float(t[i] + (t[j] - t[i]) * hit["fraction"])
    begin, end = map(int, transitions)
    entry, release = surface_event(begin, begin+1), surface_event(end+1, end)
    omega = math.sqrt(STIFFNESS / reduced_mass)
    q_ref, f_ref = exact("spheres", t, LEGACY)
    return dict(method=method, solver="penalty", step_s=step, event_tolerance_s=event_tol, duration_s=DURATION, frames=len(t),
                scene_sha256=digest(scene), mesh_sha256=quality["mesh_sha256"], step_sha256=quality["step_sha256"],
                triangles=quality["triangles"], mesh_size_m=quality["mesh_size_m"], entry_s=entry, release_s=release,
                active_duration_s=release-entry, reference_entry_s=STANDOFF/SPEED,
                reference_active_duration_s=math.pi/omega, peak_force_N=float(force.max()),
                reference_peak_force_N=STIFFNESS*SPEED/omega, impulse_Ns=float(np.trapezoid(force,t)),
                reference_impulse_Ns=2*reduced_mass*SPEED, final_velocity_m_s=v[-1].tolist(),
                reference_final_velocity_m_s=[-SPEED,0.],
                max_momentum_drift_kg_m_s=float(np.max(np.abs(v.sum(axis=1)-v[0].sum()))),
                position_rmse_m=np.sqrt(np.mean((q-q_ref)**2,axis=0)).tolist(),
                force_rmse_N=float(np.sqrt(np.mean((force-f_ref)**2))), integration_s=float(stats["wall_s"]),
                setup_s=float(stats["setup_s"]), process_wall_s=elapsed, force_evaluations=int(stats["force_evaluations"]),
                subdivisions=int(stats["subdivisions"]), step_checks=int(stats["ccd_checks"]),
                event_rejections=int(stats["ccd_rejected"]), maximum_hht_residual=float(stats["max_residual"]))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("executable","mesh_executable","mesh","work","output"):
        parser.add_argument(key,type=Path)
    parser.add_argument("--method",choices=("mesh-aabb","mesh-obb","mesh-sap"),required=True)
    parser.add_argument("--step",type=float,default=.00025)
    parser.add_argument("--event-tol-s",type=float,default=0.)
    args=parser.parse_args()
    result=compare(args.executable.resolve(),args.mesh_executable.resolve(),args.mesh.resolve(),args.method,args.step,
                   args.work,args.event_tol_s)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
