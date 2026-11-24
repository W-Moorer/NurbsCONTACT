"""Compare CPU CAD and triangle-mesh force regions at identical recorded hinge poses."""
import argparse
import csv
import json
from pathlib import Path
import subprocess

import meshio
import numpy as np
from scipy.spatial.transform import Rotation

from _gpu_source import read_model
from cpu_backend_run import digest
from reducer_contact_rays import surface_hit


def check_mesh_chords(scene, frames, report):
    """Remeasure the mesh and STEP depths on the same physical line."""
    bodies, pairs = read_model(scene)
    with frames.open(newline="") as stream:
        samples = list(csv.DictReader(stream))
    for stage, (a, b, _, limit) in zip(report["stages"], pairs, strict=True):
        meshes = [meshio.read(bodies[i][0].with_suffix(".msh")) for i in (a, b)]
        for index, sample in enumerate(samples):
            triangles = []
            for i, mesh in zip((a, b), meshes):
                angle = float(sample[f"b{i}_angle_rad"])
                rotation = Rotation.from_rotvec(bodies[i][2] * angle)
                origin = [float(sample[f"b{i}_{k}_m"]) for k in "xyz"]
                triangles.append((rotation.apply(mesh.points) + origin)[mesh.get_cells_type("triangle")])
            for run in stage["runs"].values():
                row = run["rows"][index]
                row["mesh_line_depths_m"] = []
                for region in row["regions"]:
                    point, normal = np.asarray(region[:3]), np.asarray(region[3:6])
                    offsets = [surface_hit(tri, point, direction, 2 * limit)
                               for tri, direction in zip(triangles, (normal, -normal))]
                    depth = None if any(d is None for d in offsets) else float(sum(offsets))
                    row["mesh_line_depths_m"].append(depth)


def replay(scene, frames, cad, mesh):
    bodies, pairs = read_model(scene)
    with frames.open(newline="") as stream:
        samples = list(csv.DictReader(stream))
    report = dict(scene_sha256=digest(scene), poses_sha256=digest(frames),
                  executables={p.name: digest(p) for p in (cad, mesh)}, stages=[])
    for stage, (a, b, stiffness, limit) in enumerate(pairs, start=1):
        poses = []
        for sample in samples:
            row = []
            for i in (a, b):
                row.extend(float(sample[f"b{i}_{k}_m"]) for k in "xyz")
                row.extend(bodies[i][2])
                row.append(float(sample[f"b{i}_angle_rad"]))
            poses.append(" ".join(format(x, ".17g") for x in row))
        assets = [bodies[i][0] for i in (a, b)]
        meshes = [p.with_suffix(".msh") for p in assets]
        runs = {}
        commands = {
            "cad-occt": [cad, *assets, "--profile-regions"],
            "mesh-obb": [mesh, "--method", "mesh-obb", "--regions", "contour-ray", "--axis-angle",
                         "--max-depth-m", str(limit), *meshes],
        }
        for name, command in commands.items():
            proc = subprocess.run(command, input="\n".join(poses) + "\n", text=True,
                                  capture_output=True, timeout=1800, check=True)
            rows = [json.loads(line) for line in proc.stdout.splitlines()]
            if len(rows) != len(samples):
                raise ValueError(f"Incomplete {name} stage {stage}")
            for record, sample in zip(rows, samples):
                record["time_s"] = float(sample["time_s"])
                record["step"] = int(sample["step"])
                if not record.get("resolved", True):
                    continue
                regions = np.array(record["regions"]).reshape(-1, 7)
                if not np.isfinite(regions).all() or np.any(regions[:, 6] < 0):
                    raise ValueError("Invalid force geometry")
                loaded = regions[regions[:, 6] > 0]
                record["exceeds_depth_limit"] = bool(np.any(loaded[:, 6] > limit))
                record["shared_load_N"] = 0.0
                record["driven_axis_torque_Nm"] = 0.0
                if len(loaded) and not record["exceeds_depth_limit"]:
                    forces = stiffness * loaded[:, 6:7] * loaded[:, 3:6] / len(loaded)
                    center = [float(sample[f"b{b}_{k}_m"]) for k in "xyz"]
                    record["shared_load_N"] = float(stiffness * loaded[:, 6].mean())
                    record["driven_axis_torque_Nm"] = float(np.cross(loaded[:, :3] - center, forces).sum(axis=0) @ bodies[b][2])
                elif record["exceeds_depth_limit"]:
                    record["shared_load_N"] = record["driven_axis_torque_Nm"] = None
            runs[name] = dict(rows=rows, diagnostics=proc.stderr.strip())
        # Hold the point and direction fixed and change only the depth measurement.
        lines, records = [], []
        for run in runs.values():
            for pose, record in zip(poses, run["rows"]):
                record["cad_normal_chords"] = []
                for region in record["regions"]:
                    values = [*region[:6], 2 * limit]
                    lines.append(pose + " " + " ".join(format(x, ".17g") for x in values))
                    records.append(record)
        if lines:
            proc = subprocess.run([cad, *assets, "--normal-chords"], input="\n".join(lines) + "\n",
                                  text=True, capture_output=True, timeout=1800, check=True)
            chords = [json.loads(line) for line in proc.stdout.splitlines()]
            for record, chord in zip(records, chords, strict=True):
                record["cad_normal_chords"].append(chord)
        report["stages"].append(dict(stage=stage, assets={p.name: digest(p) for p in assets + meshes},
                                     stiffness_N_per_m=stiffness, depth_limit_m=limit, runs=runs))
    check_mesh_chords(scene, frames, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("scene", "poses", "cad_benchmark", "mesh_benchmark", "output"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    result = replay(args.scene, args.poses, args.cad_benchmark, args.mesh_benchmark)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
