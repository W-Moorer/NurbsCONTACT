"""Recheck every zero-normal-load output against its accepted and first-trial mesh geometry."""
import argparse
import json
from pathlib import Path
import subprocess
from time import monotonic

import numpy as np

from cpu_backend_pose_replay import read_model
from cpu_backend_run import digest


def regions_at(exe, bodies, pair, angles):
    a, b, _, depth = pair
    rows = []
    for q in angles:
        values = np.concatenate([np.r_[bodies[i][1], bodies[i][2], bodies[i][3] + q[bodies[i][4]]]
                                 for i in (a, b)])
        rows.append(" ".join(format(v, ".17g") for v in values))
    args = [exe, "--method", "mesh-obb", "--regions", "contour-ray", "--axis-angle",
            "--max-depth-m", str(depth), *(bodies[i][0].with_suffix(".msh") for i in (a, b))]
    output = []
    deadline = monotonic() + 1800
    while len(output) < len(rows):
        run = subprocess.run(args, input="\n".join(rows[len(output):]) + "\n", capture_output=True,
                             text=True, timeout=max(.01, deadline-monotonic()))
        for line in run.stdout.splitlines():
            regs = np.array(json.loads(line)["regions"]).reshape(-1, 7)
            if not np.isfinite(regs).all() or (regs[:, 6] < 0).any():
                raise ValueError("Invalid contact region in mesh replay")
            output.append(regs[regs[:, 6] > 0])
        if run.returncode:
            if "intersecting patch has no finite feature witness" not in run.stderr or len(output) >= len(rows):
                raise RuntimeError(run.stderr)
            output.append(None)
        elif len(output) != len(rows):
            raise ValueError("Incomplete mesh replay")
    return output


def audit(exe, scene, trace, out, spectral_radius=None, deck=None):
    bodies, pairs = read_model(scene)
    fields = ["time_s", *(f"{key}{i}" for key in ("q", "v", "a") for i in range(3)),
              *(f"p{i}_normal_N" for i in range(len(pairs)))]
    data = np.genfromtxt(trace, names=True, delimiter=",", usecols=fields)
    t = data["time_s"]
    np.testing.assert_allclose(t, np.arange(50001)*.0001, atol=1e-10, rtol=0)
    q, v, acc = [np.column_stack([data[f"{key}{i}"] for i in range(3)]) for key in ("q", "v", "a")]
    report = dict(scene_sha256=digest(scene), trace_sha256=digest(trace), mesh_exe_sha256=digest(exe),
                  steps=len(t)-1, step_s=.0001, stages={})
    beta = .25*(1+.1)**2
    if spectral_radius is not None:
        if not 0 <= spectral_radius <= 1:
            raise ValueError("Spectral radius must lie between zero and one")
        beta = 1/(1+spectral_radius)**2
        report["spectral_radius"] = spectral_radius
    for stage, pair in enumerate(pairs):
        rows = np.flatnonzero((data[f"p{stage}_normal_N"] == 0) & (t > 0))
        accepted = regions_at(exe, bodies, pair, q[rows])
        candidates = [(i, g) for i, g in zip(rows, accepted, strict=True) if g is not None and len(g)]
        indices = np.array([i for i, _ in candidates], dtype=int)
        h = .0001
        predictor = q[indices-1] + h*v[indices-1] + h*h*(.5-beta)*acc[indices-1]
        initial = predictor + (beta*h*h)*acc[indices-1]
        endpoint = predictor + (beta*h*h)*acc[indices]
        single = np.all(abs(endpoint-q[indices]) <= 64*np.finfo(float).eps*np.maximum(1., abs(q[indices])), axis=1)
        first_trial = iter(regions_at(exe, bodies, pair, initial[single]))
        records = []
        for (i, g), full_step in zip(candidates, single, strict=True):
            first = next(first_trial) if full_step else None
            depth = g[:, 6]
            force = pair[2]*depth[:, None]*g[:, 3:6]
            torque = np.cross(g[:, :3]-bodies[pair[1]][1], force) @ bodies[pair[1]][2]
            records.append(dict(step=int(i), initial_active=len(first) if first is not None else None,
                                accepted_active=len(g), accepted_max_um=float(depth.max()*1e6),
                                fresh_mean_normal_N=float(pair[2]*depth.mean()),
                                fresh_driven_axis_torque_Nm=float(torque.mean())))
        confirmed = [r for r in records if r["initial_active"] == 0]
        per_second = np.bincount([min((r["step"]-1)//10000, 4) for r in confirmed], minlength=5)
        report["stages"][stage+1] = dict(recorded_zero=len(rows), unresolved_accepted_steps=[
            int(i) for i, g in zip(rows, accepted, strict=True) if g is None],
            accepted_with_positive_depth=len(records), first_trial_empty=len(confirmed),
            first_trial_empty_per_second=per_second.tolist(), positive_depth_records=records)
    if deck is not None:
        from reducer_matched_responses import read_run
        solution=read_run(trace,deck)
        ids=np.arange(0,len(t),100)
        regions=regions_at(exe,bodies,pairs[-1],q[ids])
        _,driven,k,_=pairs[-1]
        _,origin,_,_,shaft_coordinate=bodies[driven]
        moments=np.array([np.nan if g is None else
            np.cross(g[:,:3]-origin,k*g[:,6:7]*g[:,3:6])[:,2].mean() if len(g) else 0.
            for g in regions])
        valid=np.isfinite(moments)
        errors=moments[valid]-solution["torque"][ids[valid],shaft_coordinate]
        report["sampled_output_torque"]=dict(samples=len(ids),unresolved_steps=ids[~valid].tolist(),
            rms_difference_Nm=float(np.sqrt(np.mean(errors**2))) if len(errors) else None)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("exe", "scene", "trace", "out"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--spectral-radius", type=float)
    parser.add_argument("--deck",type=Path,help="also compare fresh output-shaft torques every 100th accepted output")
    audit(**vars(parser.parse_args()))
