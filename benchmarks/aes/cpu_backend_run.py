"""Run one full reducer duty cycle through the CPU scene executable with a hard process deadline."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def run(exe, scene, directory, method, scope, solver, penalty_stiffness=None, geometry_update=None,
        spectral_radius=None, substeps_per_output=1, launch=subprocess.run):
    if not isinstance(substeps_per_output, int) or substeps_per_output < 1:
        raise ValueError("Expected a positive integer number of integration steps per output")
    dt=.0001/substeps_per_output
    directory.mkdir(parents=True, exist_ok=False)
    prefix = directory / "response"
    progress = directory / "progress.csv"
    args = [exe, scene, "0", str(dt), "5", str(substeps_per_output), prefix, "--method", method,
            "--solver", solver, "--progress", progress]
    if method.startswith("mesh"):
        args += ["--mesh-region-scope", scope]
    if penalty_stiffness:
        args += ["--penalty-stiffness", penalty_stiffness]
    if geometry_update:
        args += ["--geometry-update", geometry_update]
    if spectral_radius is not None:
        args += ["--spectral-radius", str(spectral_radius)]
    tic = perf_counter()
    status = "complete"
    exit_code = None
    with (directory / "process.log").open("wb") as log:
        try:
            exit_code = launch(args, stdout=log, stderr=subprocess.STDOUT, timeout=1800).returncode
            if exit_code:
                status = "runtime_loader_failure" if exit_code & 0xFFFFFFFF == 0xC0000135 else "solver_failure"
        except subprocess.TimeoutExpired:
            status = "timeout"
    wall = perf_counter() - tic
    rows = []
    if progress.exists():
        with progress.open(newline="") as f:
            rows = list(csv.DictReader(f))
        # The deadline may interrupt a write to the final line.
        if rows and any(v is None or v == "" for v in rows[-1].values()):
            rows.pop()
    last = {k: float(v) for k, v in rows[-1].items()} if rows else None
    report = dict(method=method, solver=solver, scope=scope if method.startswith("mesh") else None,
                  executable_sha256=digest(exe), scene_sha256=digest(scene), requested_s=5, dt_s=dt, output_dt_s=.0001,
                  budget_s=1800, process_s=wall, status=status, exit_code=exit_code, last_accepted=last)
    if penalty_stiffness: report["penalty_stiffness"]=penalty_stiffness
    if geometry_update: report["geometry_update"]=geometry_update
    if spectral_radius is not None: report["spectral_radius"]=spectral_radius
    lines = (directory / "process.log").read_text(errors="replace").splitlines()
    if exit_code:
        report["error"] = lines[-1] if lines else "No diagnostic"
    if status == "complete":
        values = np.genfromtxt(prefix.with_suffix(".csv"), names=True, delimiter=",")
        np.testing.assert_allclose(values["time_s"], np.arange(50001)*.0001, atol=1e-10, rtol=0)
        for column in values.dtype.names:
            data = values[column]
            if column.endswith("_gap_m"):
                pair = column.removesuffix("_gap_m")
                hit = values[pair+"_hit"].astype(bool)
                if np.isinf(data).any() or not np.isfinite(data[hit]).all() or np.any(values[pair+"_normal_N"][~hit] != 0):
                    raise ValueError(f"Invalid gap or load in {pair}")
            elif not np.isfinite(data).all():
                raise ValueError(f"Nonfinite CPU output in {column}")
        if not last or last["step"] != 50000*substeps_per_output:
            raise ValueError("Incomplete CPU response")
        with prefix.with_suffix(".stats.csv").open(newline="") as f:
            report["statistics"] = next(csv.DictReader(f))
        report["peak_load_N"] = [float(values[c].max()) for c in values.dtype.names if c.endswith("_normal_N")]
        report["response_sha256"] = digest(prefix.with_suffix(".csv"))
    (directory/"run.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("exe", "scene", "out"):
        p.add_argument(name, type=Path)
    p.add_argument("--method", required=True, choices=("mesh-obb", "mesh-aabb", "mesh-sap", "cad-occt", "cad-patch", "cad-bezier", "cad-bezier-bounds", "cad-bezier-relocate"))
    p.add_argument("--scope", choices=("local", "full", "fcl", "finite", "contour-ray"), default="local")
    p.add_argument("--solver", default="penalty")
    p.add_argument("--penalty-stiffness", choices=("per-region","shared"))
    p.add_argument("--geometry-update", choices=("newton","predictor"))
    p.add_argument("--spectral-radius", type=float)
    p.add_argument("--substeps-per-output", type=int, default=1)
    a = p.parse_args()
    run(a.exe.resolve(), a.scene.resolve(), a.out.resolve(), a.method, a.scope, a.solver,
        a.penalty_stiffness,a.geometry_update,a.spectral_radius,a.substeps_per_output)
