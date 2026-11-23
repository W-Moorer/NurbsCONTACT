"""Compare all three CPU surface-intersection methods on a two-body pose replay."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np


def replay(executable, paths, poses):
    rows = []
    times = []
    for t, x, y, z, ax, ay, az, angle in np.loadtxt(poses, ndmin=2):
        axis = np.asarray([ax, ay, az])
        if np.linalg.norm(axis) == 0:
            raise ValueError("pose file contains a zero rotation axis")
        axis *= math.sin(angle / 2) / np.linalg.norm(axis)
        row = [0, 0, 0, 1, 0, 0, 0, x, y, z, math.cos(angle / 2), *axis]
        rows.append(" ".join(format(value, ".17g") for value in row))
        times.append(t)
    input_text = "\n".join(rows) + "\n"
    reference = None
    summaries = {}
    for method in ("mesh-aabb", "mesh-obb", "mesh-sap"):
        output = subprocess.run([executable, "--method", method, *paths], input=input_text,
                                text=True, capture_output=True, timeout=1800)
        if output.returncode:
            raise RuntimeError(f"{method} failed: {output.stderr.strip()}")
        frames = [json.loads(row) for row in output.stdout.splitlines()]
        if len(frames) != len(times):
            raise ValueError(f"{method} returned an incomplete replay")
        surfaces = [(frame["hits"], frame["enclosed"]) for frame in frames]
        if reference is None:
            reference = surfaces
        elif reference != surfaces:
            raise ValueError(f"{method} changed a triangle intersection or enclosure result")
        summaries[method] = [{**{k: v for k, v in f.items() if k not in ("hits", "enclosed")},
                              "intersection_pairs": len(f["hits"]), "enclosed_components": len(f["enclosed"])}
                             for f in frames]
    return dict(times_s=times, pose_sha256=hashlib.sha256(poses.read_bytes()).hexdigest(),
                mesh_sha256=[hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],
                matching_boundary_witnesses=True, methods=summaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("executable", "a", "b", "poses", "output"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    report = replay(args.executable.resolve(), [args.a, args.b], args.poses)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
