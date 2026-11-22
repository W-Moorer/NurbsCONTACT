"""Score the generic two-slider solver and existing RecurDyn sphere impact against the same closed form."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import shutil

import numpy as np
from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
from literature_validation import write_step
from resident_cam import mesh
from resident_cad_probes import prepare as project

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engines"))
from rigid import AES, DURATION, LAYER, STIFFNESS, case_values, initial_velocity, exact


def simulate(work, name, us, *, vertices=False, scope="region"):
    data = work / f"{name}.npy"
    report = work / f"{name}.json"
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("resident_assembly.py")),
        str(work),
        str(report),
        "--trace",
        str(data),
        "--method",
        "cad",
        "--pitch",
        "4",
        "--duration",
        str(DURATION),
        "--step-us",
        str(us),
        "--ticks",
        "4096",
        "--contact-samples",
        "16",
        "--stiffness-scope",
        scope,
    ]
    if vertices:
        cmd.append("--include-vertices")
    subprocess.run(cmd, check=True, timeout=1800, stdout=subprocess.DEVNULL)
    result = json.loads(report.read_text())
    if result["terminated"] != "complete":
        raise ValueError(result)
    return np.load(data), result


def run(work, reference, output):
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    stepfile = work / "sphere.step"
    write_step(BRepPrimAPI_MakeSphere(1000 * (AES.radius + LAYER / 2)).Shape(), stepfile)
    height, _, count = case_values("spheres", AES)
    speeds = initial_velocity(count, AES)
    h = min(AES.steps)
    coord = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], float)
    bodies = np.array([[1, 0, 0, 1, 0, 0, z, 0, s, 1, 0, 0, 0] for z, s in zip((0, height * 1000), speeds)])
    np.savez_compressed(
        work / "assembly.npz",
        coord=coord,
        bodies=bodies,
        mapping=[0, 1],
        pairs=[[0, 1]],
        stiffness=[STIFFNESS],
        loads=np.zeros((round(DURATION / h) + 1, 2)),
        h=h,
    )
    records = []
    series = {}
    for deflection in (8, 2):
        tri, face = mesh(stepfile, deflection)
        # OCCT can emit collapsed facets at a pole after vertex welding.
        valid = np.any(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) != 0, axis=1)
        tri, face = tri[valid], face[valid]
        points = project(stepfile, tri, face, 1)[0]
        archive = f"sphere-{deflection}.npz"
        np.savez_compressed(work / archive, tri=tri, faces=face, points=points)
        (work / "models.json").write_text(json.dumps(dict(parts=[dict(archive=archive)] * 2)))
        for step in AES.steps:
            us = round(1e6 * step)
            prefix = f"sphere-{deflection}-{us}"
            trace, _ = simulate(work, prefix, us)
            if deflection == 8 and us == 125:
                other, record = simulate(work, prefix + "-pair", us, scope="pair")
                np.testing.assert_array_equal(trace, other)
                control = dict(
                    benchmark="two spheres",
                    rows=len(trace),
                    step_s=step,
                    equal_all_trace_columns=True,
                    regions=record["max_active_regions"],
                    scope=record["stiffness_scope"],
                )
                (output / "single-region-control.json").write_text(json.dumps(control, indent=2) + "\n")
            time = np.arange(len(trace)) * step
            pos = trace[:, :2] + time[:, None] * speeds
            truth, force = exact("spheres", time, AES)
            rd = np.genfromtxt(reference / f"spheres-{us}us.csv", delimiter=",", names=True)
            np.testing.assert_allclose(time, rd["time_s"], rtol=0, atol=1e-12)
            for method, q, fn in (
                ("STEP", pos, trace[:, 4]),
                ("RecurDyn", np.column_stack((rd["q0_m"], rd["q1_m"])), rd["force_N"]),
            ):
                # Both columns represent each body's displacement from its initial center.
                records.append(
                    dict(
                        method=method,
                        deflection_um=deflection if method == "STEP" else None,
                        triangles=len(tri) if method == "STEP" else None,
                        step_us=us,
                        displacement_RMS_um=1e6 * np.sqrt(np.mean((q - truth) ** 2)),
                        force_RMS_N=np.sqrt(np.mean((fn - force) ** 2)),
                        peak_force_N=max(fn),
                        com_drift_um=1e6 * np.max(abs(q.mean(axis=1))),
                    )
                )
                if us == 125:
                    series[method + str(deflection)] = np.column_stack((q, fn))
            if us == 125:
                series.update(time_s=time, exact=np.column_stack((truth, force)))
            print(records[-2], flush=True)
    unique = [r for i, r in enumerate(records) if r not in records[:i]]
    (output / "sphere-exact.json").write_text(json.dumps(unique, indent=2) + "\n")
    np.savez_compressed(output / "sphere-exact.npz", **series)


def export_coverage(work, output):
    study = work / "coverage"
    time = np.arange(round(DURATION / 0.000125) + 1) * 0.000125
    speeds = initial_velocity(2, AES)
    truth, force = exact("spheres", time, AES)
    series = {"t": time, "exact": np.column_stack((truth, force))}
    for orientation in ("original", "rotated"):
        for vertices in (0, 1):
            trace = np.load(study / f"{orientation}-v{vertices}-125.npy")
            q = trace[:, :2] + time[:, None] * speeds
            series[f"{orientation}_v{vertices}"] = np.column_stack((q, trace[:, 4]))
    np.savez_compressed(output / "sphere-coverage-traces.npz", **series)
    with np.load(output / "sphere-exact.npz") as r:
        series["native"] = r["RecurDyn2"]
    # This interval is free flight in the exact solution and every recorded simulation.
    first = np.searchsorted(time, 0.10)
    balance = {}
    for name, channel in series.items():
        if name == "t":
            continue
        if np.any(abs(channel[first:, 2]) > 1e-8):
            raise ValueError("Expected post-impact free flight")
        v = (channel[-1, :2] - channel[first, :2]) / (time[-1] - time[first])
        balance[name] = dict(
            restitution=(v[1] - v[0]) / (speeds[0] - speeds[1]),
            final_to_initial_kinetic_energy=float((v @ v) / (speeds @ speeds)),
        )
    (output / "sphere-rebound.json").write_text(json.dumps(balance, indent=2) + "\n")


def coverage(work, output):
    """Rotate each unchanged sphere's tessellation independently of the impact axis."""
    from scipy.spatial.transform import Rotation

    study = work / "coverage"
    study.mkdir(exist_ok=True)
    shutil.copyfile(work / "assembly.npz", study / "assembly.npz")
    speeds = initial_velocity(2, AES)
    results = []
    with np.load(work / "sphere-2.npz") as d:
        tri, points, faces = d["tri"], d["points"], d["faces"]
    for orientation, angles in (
        ("original", ((0, 0, 0), (0, 0, 0))),
        ("rotated", ((19, 31, 7), (-23, 17, 43))),
    ):
        parts = []
        for i, ang in enumerate(angles):
            path = study / f"{orientation}-{i}.npz"
            if orientation == "original":
                shutil.copyfile(work / "sphere-2.npz", path)
            else:
                rotation = Rotation.from_euler("xyz", ang, degrees=True)
                # Transform each unique vertex once to preserve exact welded connectivity.
                vertices, inv = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
                oriented = rotation.apply(vertices)[inv].reshape(tri.shape)
                np.savez_compressed(path, tri=oriented, faces=faces, points=rotation.apply(points))
            parts.append(dict(archive=path.name))
        (study / "models.json").write_text(json.dumps(dict(parts=parts)))
        for vertices in (False, True):
            for us in (125, 62.5):
                name = f"{orientation}-v{int(vertices)}-{us}"
                trace, record = simulate(study, name, us, vertices=vertices)
                time = np.arange(len(trace)) * us * 1e-6
                truth, force = exact("spheres", time, AES)
                q = trace[:, :2] + time[:, None] * speeds
                results.append(
                    dict(
                        orientation=orientation,
                        xyz_euler_degrees=angles,
                        vertices=vertices,
                        step_us=us,
                        displacement_RMS_um=1e6 * np.sqrt(np.mean((q - truth) ** 2)),
                        force_RMS_N=np.sqrt(np.mean((trace[:, 4] - force) ** 2)),
                        peak_force_N=trace[:, 4].max(),
                        max_COM_drift_um=1e6 * np.max(abs(q.mean(axis=1))),
                        online_s=record["online_s"],
                    )
                )
                (output / "sphere-coverage.json").write_text(json.dumps(results, indent=2) + "\n")
                print(name, results[-1]["displacement_RMS_um"], flush=True)
    export_coverage(work, output)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("work", type=Path)
    ap.add_argument("reference", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument(
        "--coverage-only",
        action="store_true",
        help="Test the prepared meshes under independent rotations, with and without extra vertex probes",
    )
    args = ap.parse_args()
    if args.coverage_only:
        coverage(args.work, args.output)
    else:
        run(args.work, args.reference, args.output)
