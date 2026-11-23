"""Validate the independent CPU node-to-NURBS baseline against analytic and archived references."""

import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere
from literature_validation import write_step
from pair_dynamics import gears

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/engines"))
from rigid import AES, DURATION, LAYER, exact


def run(exe, args, output, rows=()):
    # Each subprocess, including STEP preparation and all HHT trials, has a separate wall-time limit.
    inputs = "".join(" ".join(f"{x:.17g}" for x in row) + "\n" for row in rows)
    p = subprocess.run(
        [exe, *map(str, args), output],
        input=inputs,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=1800,
    )
    if p.returncode:
        raise RuntimeError(f"{args[0]} exited with {p.returncode}: {p.stderr}")
    record = {k: float(v) if k.endswith("_s") else int(v) for k, v in re.findall(r"(\w+)=(\S+)", p.stderr)}
    print(output.stem, record, flush=True)
    return np.atleast_1d(np.genfromtxt(output, delimiter=",", names=True)), record


def sphere_points(exe, sphere, radius, work):
    # Include both signs, the periodic seam and the singular parameter values at both poles.
    directions = np.vstack((np.eye(3), -np.eye(3), np.random.default_rng(19).normal(size=(24, 3))))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    normals = np.repeat(directions, 3, axis=0)
    offsets = np.tile([-0.75, 0, 0.5], len(directions))
    rows, record = run(
        exe, ["points", sphere], work / "sphere-points.csv", (radius + offsets[:, None]) * normals
    )

    def columns(names):
        return np.column_stack([rows[k] for k in names])

    q = columns(("qx_mm", "qy_mm", "qz_mm"))
    n = columns(("nx", "ny", "nz"))
    if np.any(rows["face"] == 0):
        raise ValueError("Unresolved projection on a smooth sphere")
    np.testing.assert_allclose(q, radius * normals, atol=1e-6, rtol=0)
    np.testing.assert_allclose(rows["gap_mm"], offsets, atol=1e-6, rtol=0)
    np.testing.assert_allclose(n, normals, atol=1e-7, rtol=0)
    regular = rows["normal_derivative_order"] == 1
    tangent_errors = []
    for prefix in ("du", "dv"):
        tangent = columns([prefix + axis for axis in "xyz"])[regular]
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        tangent_errors.extend(np.abs(np.sum(tangent * normals[regular], axis=1)))
    np.testing.assert_allclose(tangent_errors, 0, atol=1e-7, rtol=0)
    return dict(
        **record,
        nodes=len(rows),
        max_gap_error_um=float(np.max(abs(rows["gap_mm"] - offsets)) * 1000),
        max_position_error_um=float(np.linalg.norm(q - radius * normals, axis=1).max() * 1000),
        max_normal_vector_error=float(np.linalg.norm(n - normals, axis=1).max()),
        max_abs_tangent_dot_analytic_normal=float(max(tangent_errors)),
        max_tangential_residual_mm=float(rows["tangential_mm"].max()),
    )


def sphere_impact(exe, sphere, tilt, work, traces, deflection):
    tag = f"{'aligned' if tilt == 0 else 'rotated'}_{deflection:g}mm"
    h = 0.000125
    rows, record = run(exe, ["impact", sphere, sphere, deflection, tilt, h], work / f"impact-{tag}.csv")
    t = np.arange(round(DURATION / h) + 1) * h
    np.testing.assert_allclose(rows["time_s"], t, atol=1e-12, rtol=0)
    q, force = exact("spheres", t, AES)
    simulated = np.column_stack((rows["q0_m"], rows["q1_m"]))
    actual_force = rows["axial_force_N"]
    if not np.isfinite(simulated).all() or not np.isfinite(actual_force).all() or actual_force.max() <= 0:
        raise ValueError("Invalid sphere response")
    np.testing.assert_allclose(simulated.sum(axis=1), 0, rtol=0, atol=1e-11)
    traces.update(time_s=t, exact=np.column_stack((q, force)))
    traces[tag] = np.column_stack((simulated, actual_force))
    return dict(
        **record,
        step_us=125,
        duration_s=DURATION,
        rows=len(rows),
        source_deflection_mm=deflection,
        body_rotations_rad=[tilt, -0.7 * tilt],
        rotation_axis=[1, 2, 0.5],
        displacement_rms_um=float(np.sqrt(np.mean((simulated - q) ** 2)) * 1e6),
        axial_force_rms_N=float(np.sqrt(np.mean((actual_force - force) ** 2))),
        peak_axial_force_N=float(actual_force.max()),
    )


def trim_check(exe, work):
    box = work / "trim-control.step"
    write_step(BRepPrimAPI_MakeBox(10, 15, 20).Shape(), box)
    # The last node projects beyond the trimmed top face onto a physical edge.
    points = [[5, 7, 19.5], [5, 7, 20.5], [11, 7, 21]]
    rows, record = run(exe, ["points", box], work / "trim-control.csv", points)
    np.testing.assert_allclose(rows["gap_mm"], [-0.5, 0.5, np.sqrt(2)], atol=1e-6, rtol=0)
    np.testing.assert_array_equal(rows["face"] > 0, [True, True, False])
    np.testing.assert_allclose(rows["nz"][:2], 1, atol=1e-12, rtol=0)
    if not np.isnan(rows["nz"][2]):
        raise ValueError("Unexpected smooth normal returned for a sharp edge")
    return dict(**record, gaps_mm=rows["gap_mm"].tolist(), sharp_edge_reported=True)


def gear_pose(work, models, physical, time):
    # Same shaft conventions, preloads and reference rotations as reducer_switch_replay.py.
    scene = work / "reducer.scene"
    gears(models, scene, 3.0, 148.0)
    deck = scene.read_text().splitlines()
    coordinates = np.array(
        [list(map(float, line.split()[1:])) for line in deck if line.startswith("coordinate ")]
    )
    with np.load(physical / "native-exports/rd-switch-geometric-physics.npz") as data:
        i = round(time / 0.000125)
        q = data["angle_rad"][i] + coordinates[:, 1] - data["reference_rad"][i]
    with (models / "reducer.csv").open(newline="") as f:
        pair = [r for r in csv.DictReader(f) if r["stage"] == "2"]
    poses = []
    for part in pair:
        j, sign = int(part["shaft"]), int(part["direction"])
        angle = sign * float(part["angle_zero_rad"]) + coordinates[j, 2] * time + q[j]
        poses.extend((float(part["center_x_mm"]), 0, float(part["z_mm"]), 0, 0, sign, angle))
    return [models / part["step"] for part in pair], poses


def check(exe, work, output):
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    radius = 1000 * (AES.radius + LAYER / 2)
    sphere = work / "sphere.step"
    write_step(BRepPrimAPI_MakeSphere(radius).Shape(), sphere)
    metrics = dict(baseline="CPU node-to-NURBS (independent implementation)", sphere_radius_mm=radius)
    metrics["sphere_geometry"] = sphere_points(exe, sphere, radius, work)
    metrics["trim_control"] = trim_check(exe, work)
    traces = {}
    metrics["impacts"] = [
        sphere_impact(exe, sphere, angle, work, traces, deflection)
        for angle, deflection in ((0, 0.5), (0.47, 0.5), (0.47, 0.025))
    ]

    models = ROOT / "results/gears/models"
    physical = ROOT / "results/aes/resident-reducer/physical"
    witness = json.loads((physical / "switch.json").read_text())["witness"]
    rows, record = run(
        exe, ["points", models / "reducer-2b.step"], work / "gear-node.csv", [witness["target_point_mm"]]
    )
    gap = float(rows["gap_mm"][0]) * 1000
    reference = witness["original_STEP_outside_gap_um"]
    np.testing.assert_allclose(gap, reference, rtol=0, atol=0.001)
    if not rows["face"][0] or not np.isfinite(rows["tangential_mm"][0]):
        raise ValueError("Unresolved gear face normal")
    metrics["gear_node"] = dict(
        **record,
        time_s=witness["time_s"],
        gap_um=gap,
        original_step_gap_um=reference,
        face=int(rows["face"][0]),
        tangential_residual_mm=float(rows["tangential_mm"][0]),
    )
    metrics["gear_pairs"] = [gear_pair(exe, work, models, physical, limit) for limit in (0.25, 0.025)]
    np.savez_compressed(output / "sphere-traces.npz", **traces)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def gear_pair(exe, work, models, physical, deflection):
    time = 1.25275
    pair, pose = gear_pose(work, models, physical, time)
    rows, record = run(exe, ["pair", *pair, deflection], work / f"gear-pair-{deflection:g}mm.csv", [pose])
    normal = np.array([rows[k][0] for k in ("nx", "ny", "nz")])
    np.testing.assert_allclose(np.linalg.norm(normal), 1, rtol=0, atol=1e-9)
    pa = np.array([rows[k][0] for k in ("ax_mm", "ay_mm", "az_mm")])
    pb = np.array([rows[k][0] for k in ("bx_mm", "by_mm", "bz_mm")])
    depth = float(rows["depth_mm"][0])
    np.testing.assert_allclose(pa - pb, depth * normal, rtol=0, atol=1e-6)
    force = depth * 0.001 * 1.6e8
    torque = np.cross((pb - pose[7:10]) * 0.001, force * normal)[2]
    with (physical / "fixed-dense-native-torques.csv").open(newline="") as f:
        ref = next(r for r in csv.DictReader(f) if abs(float(r["time_s"]) - time) < 1e-10)
    return dict(
        **record,
        time_s=time,
        source_deflection_mm=deflection,
        pose_mm_axis_rad=pose,
        depth_mm=depth,
        unit_normal=normal.tolist(),
        on_a_mm=pa.tolist(),
        on_b_mm=pb.tolist(),
        pair_stiffness_N_m=1.6e8,
        output_torque_Nm=float(torque),
        recurDyn_output_torque_Nm=float(ref["native_output_Nm"]),
        sampled_STEP_output_torque_Nm=float(ref["cad_pair_output_Nm"]),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("executable", type=Path)
    p.add_argument("work", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    check(args.executable.resolve(), args.work.resolve(), args.output.resolve())
