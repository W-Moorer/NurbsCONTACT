"""Prepare and run the archived two-DOF eccentric cam with solid Gmsh contact."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from pair_dynamics import cam
from resident_cam import backed_plate, reference

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "apps"))
from prepare_contact_mesh import prepare, sha256


def model(directory):
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer

    directory.mkdir(parents=True, exist_ok=True)
    source = ROOT / "results/hht/models"
    shutil.copyfile(source / "cam.step", directory / "cam.step")
    reader = STEPControl_Reader()
    if reader.ReadFile(str(source / "follower.step")) != IFSelect_RetDone or not reader.TransferRoots():
        raise ValueError("could not load follower working surface")
    faces = TopExp_Explorer(reader.OneShape(), TopAbs_FACE)
    if not faces.More():
        raise ValueError("missing follower working surface")
    face = faces.Current()
    faces.Next()
    if faces.More():
        raise ValueError("expected the single archived working surface")
    backed_plate(face, (0, 50, 0), directory / "follower.step")
    cam(directory, directory / "cam.scene", 20.0, surface=True)


def mesh_assets(directory, output, size, deviation):
    for asset in ("cam", "follower"):
        prepare(directory / f"{asset}.step", output / f"{asset}.msh", size, deviation, 8.0, 0.2)


def run(args):
    if not (0 < args.dt <= args.duration <= 20):
        raise ValueError("cam load table covers 0 to 20 seconds")
    quality = {}
    for asset in ("cam", "follower"):
        path = args.meshes / f"{asset}.msh"
        report = json.loads(path.with_suffix(".quality.json").read_text())
        if report["mesh_sha256"] != sha256(path) or report["step_sha256"] != sha256(
            args.model / f"{asset}.step"
        ):
            raise ValueError("geometry does not match the quality report")
        quality[asset] = {
            key: report[key]
            for key in (
                "triangles",
                "mesh_sha256",
                "step_sha256",
                "min_sicn",
                "max_sampled_surface_error_m",
                "max_normal_angle_deg",
            )
        }
    scene = args.model / "cam.scene"
    args.work.mkdir(parents=True, exist_ok=True)
    prefix = args.work / "response"
    start = time.perf_counter()
    subprocess.run(
        [
            args.exe.resolve(),
            scene,
            "0",
            str(args.dt),
            str(args.duration),
            "1",
            prefix,
            "--method",
            args.method,
            "--mesh-dir",
            args.meshes,
            "--solver",
            "penalty",
            *(["--mesh-region-scope", args.mesh_region_scope] if args.mesh_region_scope else []),
        ],
        check=True,
        timeout=1800,
    )
    elapsed = time.perf_counter() - start
    path = prefix.with_suffix(".csv")
    data = np.genfromtxt(path, names=True, delimiter=",")
    t = data["time_s"]
    if abs(t[-1] - args.duration) > 1e-9 or len(t) != round(args.duration / args.dt) + 1:
        raise ValueError("incomplete cam response")
    sol, entering = reference(args.duration, 0.0, 0.001)
    ref = sol.sol(t)
    q, force = data["q1"], data["p0_normal_N"]
    expected_force = 5000 * np.maximum(0.0, 0.105 + 0.024 * np.sin(ref[0]) - 0.115 - ref[2])
    if not np.isfinite(np.column_stack([q, force, data["q0"], data["v1"]])).all():
        raise ValueError("nonfinite response")
    active = force > 0
    switches = np.flatnonzero(active[1:] != active[:-1])
    expected = sol.t_events[0]
    same_types = len(switches) == len(expected) and np.array_equal(active[switches + 1], entering)
    with prefix.with_suffix(".stats.csv").open() as f:
        stats = next(csv.DictReader(f))
    rms = lambda x: float(np.sqrt(np.mean(x * x)))
    result = dict(
        source="archived eccentric cam, zero skin, backed follower; DOP853 smooth-cylinder reference",
        method=args.method,
        solver="penalty",
        duration_s=args.duration,
        step_s=args.dt,
        scene_sha256=sha256(scene),
        trace_sha256=sha256(path),
        meshes=quality,
        frames=len(t),
        integration_s=float(stats["wall_s"]),
        process_s=elapsed,
        subdivisions=int(stats["subdivisions"]),
        force_evaluations=int(stats["force_evaluations"]),
        angle_rmse_rad=rms(data["q0"] - ref[0]),
        follower_position_rmse_m=rms(q - ref[2]),
        follower_velocity_rmse_m_s=rms(data["v1"] - ref[3]),
        force_rmse_N=rms(force - expected_force),
        mesh_peak_N=float(force.max()),
        reference_peak_N=float(expected_force.max()),
        mesh_event_count=len(switches),
        reference_event_count=len(expected),
        same_event_sequence=bool(same_types),
        reference_event_s=expected.tolist(),
        mesh_event_brackets_s=[[t[i], t[i + 1]] for i in switches],
    )
    if "saved_contact_reuses" in stats:
        result["saved_contact_reuses"] = int(stats["saved_contact_reuses"])
    if "mesh_boolean_queries" in stats:
        result["region_scope"] = args.mesh_region_scope or "local"
        result["boolean_queries"] = int(stats["mesh_boolean_queries"])
        result["boolean_input_faces"] = int(stats["mesh_boolean_input_faces"])
    if same_types and len(expected):
        result["max_event_time_outside_mesh_bracket_s"] = float(
            np.maximum.reduce(
                [t[switches] - expected, expected - t[switches + 1], np.zeros(len(expected))]
            ).max()
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    make = modes.add_parser("model")
    make.add_argument("directory", type=Path)
    mesh = modes.add_parser("mesh")
    mesh.add_argument("directory", type=Path)
    mesh.add_argument("output", type=Path)
    mesh.add_argument("--size-m", type=float, required=True)
    mesh.add_argument("--error-m", type=float, required=True)
    simulate = modes.add_parser("run")
    for name in ("exe", "model", "meshes", "work", "output"):
        simulate.add_argument(name, type=Path)
    simulate.add_argument("--method", choices=["mesh-aabb", "mesh-obb", "mesh-sap"], required=True)
    simulate.add_argument("--mesh-region-scope", choices=["local", "full"])
    simulate.add_argument("--dt", type=float, required=True)
    simulate.add_argument("--duration", type=float, default=20.0)
    args = parser.parse_args()
    if args.mode == "model":
        model(args.directory)
    elif args.mode == "mesh":
        mesh_assets(args.directory, args.output, args.size_m, args.error_m)
    else:
        run(args)
