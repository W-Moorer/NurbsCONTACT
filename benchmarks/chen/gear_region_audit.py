"""Replay a two-shaft, undamped linear penalty trace on full and localized STEP solids."""
import argparse
import csv
import hashlib
import io
import json
import math
import shlex
import subprocess
from pathlib import Path

from scipy.optimize import linear_sum_assignment


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(exe, scene, trace):
    fields = [shlex.split(line) for line in scene.read_text().splitlines() if line.strip()]
    bodies = [row for row in fields if row[0] == "body"]
    surfaces = [row for row in fields if row[0] == "surface"]
    if len(bodies) != 2 or len(surfaces) != 1 or surfaces[0][1:3] != ["0", "1"]:
        raise ValueError("expected one surface pair between the two shaft bodies")
    stiffness, exponent, damping, friction = map(float, surfaces[0][3:7])
    if (exponent, damping, friction) != (1, 0, 0):
        raise ValueError("force replay requires an undamped linear normal law")
    assets = [scene.parent / row[1] for row in bodies]
    axes = [list(map(float, row[6:9])) for row in bodies]
    with trace.open(newline="") as stream:
        frames = list(csv.DictReader(stream))
    samples = []
    for frame in frames:
        pose = []
        for i, axis in enumerate(axes):
            pose.extend(float(frame[f"b{i}_{axis_name}_m"]) for axis_name in "xyz")
            pose.extend(axis)
            pose.append(float(frame[f"b{i}_angle_rad"]))
        t, force = float(frame["time_s"]), float(frame["p0_normal_N"])
        if samples and (samples[-1][2] > 0) != (force > 0):
            start, prev, _ = samples[-1]
            for f in (.25, .5, .75):
                samples.append((start + f*(t-start), [x+f*(y-x) for x,y in zip(prev,pose)], None))
        samples.append((t, pose, force))
    data = "\n".join(" ".join(format(v, ".17g") for v in pose) for _, pose, _ in samples)
    result = subprocess.run([exe, *assets, "--regions"], input=data+"\n", capture_output=True,
                            text=True, timeout=1800)
    if result.returncode:
        raise RuntimeError(result.stderr)
    records = {}
    for row in csv.DictReader(io.StringIO(result.stdout)):
        key = (int(row["pose"]), row["scope"])
        regions = records.setdefault(key, [])
        if int(row["count"]):
            point = tuple(float(row[f"point_{k}_m"]) for k in "xyz")
            normal = tuple(float(row[f"n{k}"]) for k in "xyz")
            regions.append((point, normal, float(row["depth_m"])))
    rows = []
    for i, (t, _, force) in enumerate(samples):
        full, local = records[i,"full"], records[i,"local"]
        if len(full) != len(local):
            raise ValueError(f"region count differs at {t} s: {len(full)} and {len(local)}")
        differences = []
        if full:
            cost = [[math.dist(a[0], b[0]) for b in local] for a in full]
            for ia, ib in zip(*linear_sum_assignment(cost)):
                a, b = full[ia], local[ib]
                differences.append((cost[ia][ib], math.dist(a[1], b[1]), abs(a[2]-b[2])))
        point, normal, depth = (max((diff[k] for diff in differences), default=0.) for k in range(3))
        full_force = stiffness * sum(r[2] for r in full)
        local_force = stiffness * sum(r[2] for r in local)
        rows.append(dict(time_s=t, pose_source="output" if force is not None else "linear_interpolation",
                         regions=len(full), full_force_N=full_force, local_force_N=local_force,
                         trace_force_N=force, max_point_delta_m=point,
                         max_normal_vector_delta=normal, max_depth_delta_m=depth))
    return dict(scene_sha256=digest(scene), trace_sha256=digest(trace),
                step_sha256={p.name:digest(p) for p in assets}, stiffness_N_m=stiffness, poses=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("exe", "scene", "trace", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    payload = audit(args.exe.resolve(), args.scene, args.trace)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
