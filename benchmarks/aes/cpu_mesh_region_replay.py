"""Compare all connected mesh penalty regions at identical prescribed body poses."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def gear_poses(trace, manifest, stage):
    """Read hinge axes from the gear benchmark manifest and world poses from nc_contact."""
    with manifest.open(newline="") as stream:
        parts = [(i, int(p["direction"])) for i, p in enumerate(csv.DictReader(stream))
                 if int(p["stage"]) == stage]
    if len(parts) != 2 or any(abs(sign) != 1 for _, sign in parts):
        raise ValueError("expected one pair of Z-axis hinges in the requested gear stage")
    samples = np.genfromtxt(trace, delimiter=",", names=True, ndmin=1)
    columns = [samples["time_s"]]
    for body, sign in parts:
        columns.extend(samples[f"b{body}_{axis}_m"] for axis in "xyz")
        half_angle = 0.5 * sign * samples[f"b{body}_angle_rad"]
        zero = np.zeros_like(half_angle)
        columns.extend((np.cos(half_angle), zero, zero, np.sin(half_angle)))
    return np.column_stack(columns)


def replay(executable, meshes, poses, method, manifest=None, stage=1):
    frames = gear_poses(poses, manifest, stage) if manifest else np.loadtxt(poses, ndmin=2)
    if frames.shape[1] != 15:
        raise ValueError("expected time and two world poses with unit quaternions")
    inputs = "".join(" ".join(format(x, ".17g") for x in row[1:]) + "\n" for row in frames)
    outputs = []
    for scope in ("full", "local"):
        run = subprocess.run(
            [executable, "--method", method, "--regions", scope, *meshes],
            input=inputs, text=True, stdout=subprocess.PIPE, check=True, timeout=1800,
        )
        found = [json.loads(line) for line in run.stdout.splitlines()]
        if len(found) != len(frames):
            raise ValueError("incomplete contact region replay")
        outputs.append(found)
    max_point = max_normal = max_depth = 0.0
    counts = []
    for idx, (full, local) in enumerate(zip(*outputs, strict=True)):
        first, second = (np.array(row["regions"]) for row in (full, local))
        if len(first) != len(second):
            raise ValueError(f"region count differs at pose {idx}")
        counts.append(len(first))
        if not len(first):
            continue
        left, right = linear_sum_assignment(cdist(first[:, :3], second[:, :3]))
        delta = second[right] - first[left]
        max_point = max(max_point, np.linalg.norm(delta[:, :3], axis=1).max())
        max_normal = max(max_normal, np.linalg.norm(delta[:, 3:6], axis=1).max())
        max_depth = max(max_depth, np.abs(delta[:, 6]).max())
    result = {
        "method": method,
        "poses_sha256": hashlib.sha256(poses.read_bytes()).hexdigest(),
        "meshes_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in meshes],
        "regions_per_pose": counts,
        "max_point_difference_m": max_point,
        "max_normal_vector_difference": max_normal,
        "max_depth_difference_m": max_depth,
        "total_boolean_input_faces_full_local": [sum(r["input_faces"] for r in run) for run in outputs],
        "localized_solids_per_pose": [r["local_solids"] for r in outputs[1]],
    }
    if manifest:
        result["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        result["stage"] = stage
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("exe", "mesh_a", "mesh_b", "poses", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--method", choices=["mesh-aabb", "mesh-obb", "mesh-sap"], default="mesh-obb")
    parser.add_argument("--gear-manifest", type=Path, help="read nc_contact gear response CSV as input")
    parser.add_argument("--stage", type=int, default=1)
    args = parser.parse_args()
    report = replay(args.exe.resolve(), [args.mesh_a, args.mesh_b], args.poses, args.method,
                    args.gear_manifest, args.stage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
