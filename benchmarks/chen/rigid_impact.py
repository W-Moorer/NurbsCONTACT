"""Replay the archived two-slider STEP impact through named CPU contact methods."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/engines"))
sys.path.insert(0, str(ROOT / "benchmarks/aes"))
from rigid import DURATION, LEGACY, SPEED, STANDOFF, STIFFNESS, case_values, exact
from cpu_mesh_memory import measured_run


def run(executable, directory, method, step, event_tol, mesh_dir):
    scene = ROOT / "results/engine-comparison/nurbs/spheres.scene"
    prefix = directory / method
    mesh = None
    if mesh_dir:
        path = mesh_dir / "offset-sphere.msh"
        quality = json.loads(path.with_suffix(".quality.json").read_text())
        for asset, key in ((path, "mesh_sha256"), (scene.with_name("offset-sphere.step"), "step_sha256")):
            if hashlib.sha256(asset.read_bytes()).hexdigest() != quality[key]:
                raise ValueError(f"{asset} differs from the mesh quality record")
        mesh = {k: quality[k] for k in ("mesh_size_m", "mesh_sha256", "triangles", "max_normal_angle_deg",
                                        "max_sampled_surface_error_m", "min_sicn")}
        mesh["file_bytes"] = path.stat().st_size
        prefix = directory / f"{method}-{mesh['triangles']}-{mesh['mesh_sha256'][:10]}"
    command = [executable, scene, "0", str(step), str(DURATION), "1", prefix,
               "--method", method, "--solver", "penalty", "--event-tol-s", str(event_tol)]
    if mesh_dir:
        command.extend(["--mesh-dir", mesh_dir])
    usage = measured_run(command)
    data = np.genfromtxt(prefix.with_suffix(".csv"), names=True, delimiter=",")
    with prefix.with_suffix(".stats.csv").open() as stream:
        stats = next(csv.DictReader(stream))
    t, force = data["time_s"], data["p0_normal_N"]
    if len(t) != round(DURATION / step) + 1 or abs(t[-1] - DURATION) > 1e-12:
        raise ValueError("incomplete impact run")
    displacement = np.column_stack([data["q0"], data["q1"]])
    qref, fref = exact("spheres", t, LEGACY)
    change = np.flatnonzero(np.diff((force > 0).astype(int)))
    if len(change) != 2 or force[0] or force[-1] or not np.isfinite(force).all() or not np.isfinite(displacement).all():
        raise ValueError("expected one complete impact")
    onset, end = change
    result = dict(method=method, onset_bracket_s=t[onset:onset+2].tolist(), release_bracket_s=t[end:end+2].tolist(),
                  peak_force_N=float(force.max()), normal_impulse_Ns=float(np.trapezoid(force, t)),
                  max_force_error_N=float(np.max(np.abs(force-fref))),
                  max_displacement_error_m=float(np.max(np.abs(displacement-qref))),
                  final_velocity_m_s=[float(data[k][-1]) for k in ("v0", "v1")],
                  integration_s=float(stats["wall_s"]), setup_s=float(stats["setup_s"]),
                  trace_sha256=hashlib.sha256(prefix.with_suffix(".csv").read_bytes()).hexdigest(), **usage)
    if mesh is not None:
        result["mesh_per_sphere"] = mesh
    result.update({k: int(stats[k]) for k in ("ccd_checks","ccd_rejected","ccd_unresolved","ccd_nodes","cad_boolean_queries",
                                              "cad_event_distances","chen_patch_queries","chen_confirmed_candidates",
                                              "mesh_boolean_queries")})
    result.update({k:int(stats[k]) for k in ("cad_region_cache_hits","cad_distance_cache_hits",
                                             "tdib_cells","tdib_candidates","tdib_unresolved") if k in stats})
    if method.startswith("chen-") and result["chen_patch_queries"]==0:
        raise ValueError("the Chen solver did not run during this impact")
    if not method.startswith("mesh-") and (result["mesh_boolean_queries"] or not result["cad_boolean_queries"]):
        raise ValueError("unexpected contact geometry used by the CAD method")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("exe", "work", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--step", type=float, default=.00025)
    parser.add_argument("--event-tol-s", type=float, default=.00005)
    control = parser.add_mutually_exclusive_group()
    control.add_argument("--mesh-dir", type=Path, nargs="+", default=[],
                         help="use these meshes in place of the default 8 mm Gmsh sphere")
    control.add_argument("--cad-only", action="store_true", help="run only the selected STEP methods")
    parser.add_argument("--cad-methods", nargs="+", choices=["cad-occt","chen-2024","chen-2026"],
                        default=["cad-occt","chen-2024","chen-2026"])
    parser.add_argument("--mesh-only", action="store_true", help="measure only the Gmsh controls")
    args = parser.parse_args()
    if args.cad_only and args.mesh_only:
        parser.error("choose one of --cad-only and --mesh-only")
    args.work.mkdir(parents=True, exist_ok=True)
    scene = ROOT / "results/engine-comparison/nurbs/spheres.scene"
    if not args.cad_only and not args.mesh_dir:
        directory = args.work / "mesh-8mm"
        subprocess.run([sys.executable, ROOT / "apps/prepare_contact_mesh.py", scene.with_name("offset-sphere.step"),
                        directory / "offset-sphere.msh", "--size-m", ".008", "--max-error-m", ".0005",
                        "--max-normal-deg", "8", "--min-sicn", ".3"], check=True, timeout=1800)
        args.mesh_dir = [directory]
    methods = [] if args.mesh_only else [(method, None) for method in args.cad_methods]
    methods.extend(("mesh-obb", mesh_dir) for mesh_dir in args.mesh_dir)
    runs = [run(args.exe.resolve(), args.work, method, args.step, args.event_tol_s, mesh_dir)
            for method, mesh_dir in methods]
    _, mass, _ = case_values("spheres", LEGACY)
    omega = np.sqrt(STIFFNESS / mass)
    reference = dict(onset_s=STANDOFF/SPEED, release_s=STANDOFF/SPEED+np.pi/omega,
                     peak_force_N=STIFFNESS*SPEED/omega, impulse_Ns=2*mass*SPEED)
    files = [scene, scene.with_name("offset-sphere.step")]
    report = dict(duration_s=DURATION, step_s=args.step, event_tol_s=args.event_tol_s, reference=reference,
                  asset_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files}, runs=runs)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
