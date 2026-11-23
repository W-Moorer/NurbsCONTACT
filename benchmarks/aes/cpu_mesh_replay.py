"""Compare all three CPU surface-intersection methods on a two-body pose replay."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np


def replay(executable, paths, poses, compare_overlap=False):
    rows = []
    times = []
    for t, *values in np.loadtxt(poses, ndmin=2):
        if len(values)==14:
            rows.append(" ".join(format(value,".17g") for value in values))
            times.append(t)
            continue
        if len(values)!=7:
            raise ValueError("expected a relative axis-angle pose or two world quaternion poses per time")
        x, y, z, ax, ay, az, angle = values
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
                              "intersection_pairs": len(f["hits"]), "enclosed_components": len(f["enclosed"]),
                              "witness_sha256": hashlib.sha256(json.dumps([f["hits"],f["enclosed"]],
                                  separators=(",",":")).encode()).hexdigest()}
                             for f in frames]
        if compare_overlap:
            result=subprocess.run([executable,"--method",method,"--any",*paths],input=input_text,
                                  text=True,capture_output=True,check=True,timeout=1800)
            decisions=[json.loads(row) for row in result.stdout.splitlines()]
            if len(decisions)!=len(rows):
                raise ValueError("incomplete overlap replay")
            for summary,decision,(hits,contained) in zip(summaries[method],decisions,surfaces,strict=True):
                if decision["overlap"]!=bool(hits or contained):
                    raise ValueError(f"{method} disagrees with its full witnesses")
                summary["decision"]=decision
    return dict(times_s=times, pose_sha256=hashlib.sha256(poses.read_bytes()).hexdigest(),
                mesh_sha256=[hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],
                matching_boundary_witnesses=True, methods=summaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("executable", "a", "b", "poses", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--compare-overlap",action="store_true")
    args = parser.parse_args()
    report = replay(args.executable.resolve(), [args.a, args.b], args.poses,args.compare_overlap)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
