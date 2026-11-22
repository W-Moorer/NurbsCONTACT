"""Compare tracked face identities with a full native STEP query at selected poses."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import cupy as cp
import numpy as np
from scipy.spatial.transform import Rotation

from contact_tracking_replay import load_tags
from gpu_mesh_line import read_full_mesh
from gpu_patch_tracker import PatchTracker


def run(args):
    a, b = read_full_mesh(args.mesh_a), read_full_mesh(args.mesh_b)
    fa, fb = load_tags(args.faces_a, args.mesh_a), load_tags(args.faces_b, args.mesh_b)
    tracker = PatchTracker(a, b, fa, fb)
    samples = []
    for path, indices in args.window:
        trajectory = np.loadtxt(path, ndmin=2)
        selected = set(map(int, indices.split(",")))
        tracker.reset()
        for i, row in enumerate(trajectory[: max(selected) + 1]):
            pose = row[1:].reshape(3, 4)
            hits, _, _ = tracker.query(cp.asarray(pose), 1.0, np.max(np.abs(pose[:, 3])))
            if i in selected:
                h = hits.get()
                pairs = set(zip(fa[h[:, 0]].tolist(), fb[h[:, 1]].tolist()))
                samples.append((row[0], pose, pairs))
    with TemporaryDirectory() as work:
        work = Path(work)
        with (work / "poses.txt").open("w") as stream:
            for t, p, _ in samples:
                v = Rotation.from_matrix(p[:, :3]).as_rotvec()
                theta = np.linalg.norm(v)
                print(t, *(p[:, 3] * 0.001), *(v / theta if theta else [0, 0, 1]), theta, file=stream)
        subprocess.run(
            [
                str(args.executable.resolve()),
                "regions-replay",
                str(args.step_a),
                str(args.step_b),
                "1e-9",
                "0",
                str(work / "poses.txt"),
                str(work / "native"),
                "cold",
            ],
            check=True,
            timeout=1800,
        )
        native = {t: set() for t, _, _ in samples}
        with (work / "native.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                native[float(row["time_s"])].add((int(row["face_a"]), int(row["face_b"])))
        with (work / "native.stats.csv").open(newline="") as stream:
            stats = {float(r["time_s"]): r for r in csv.DictReader(stream)}
    return [
        dict(
            time_s=t,
            mesh_pairs=sorted(pairs),
            occt_pairs=sorted(native[t]),
            only_cad=sorted(native[t] - pairs),
            only_mesh=sorted(pairs - native[t]),
            native_unresolved=int(stats[t]["unresolved_face_pairs"]),
        )
        for t, _, pairs in samples
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("executable", "step_a", "step_b", "mesh_a", "mesh_b", "faces_a", "faces_b", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--window", nargs=2, action="append", required=True, metavar=("POSES", "FRAME_IDS"))
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args), indent=2) + "\n", encoding="utf-8")
