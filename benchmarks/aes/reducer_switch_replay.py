"""Replay both probe placements at identical reducer poses, retaining the force of every face pair."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import cupy as cp
from scipy.spatial.transform import Rotation
from resident_contact import ResidentMeshContact
from resident_dynamics import swept_samples


def poses(work, h, start, end, audits, native_only):
    result = []
    for source in () if native_only else ("cad", "mesh"):
        trace = np.load(work / f"{source}-h62.npy", mmap_mode="r")
        for step in range(round(start / h), round(end / h) + 1):
            result.append((source, step, -1, step * h, trace[step, :3]))
    with np.load(native_only or work / "rd-joints.npz") as rd, np.load(work / "assembly.npz") as d:
        for step in range(round(start / h), round(end / h) + 1):
            if step % 2 == 0:
                q = rd["angle_rad"][step // 2] + d["coord"][:, 3] - rd["reference_rad"][step // 2]
                result.append(("recurdyn", step, -1, step * h, q))
    for path in audits:
        with np.load(path) as d:
            tr = d["trials"]
            for i, row in enumerate(tr):
                step = int(row[0])
                if 1.252625 - 1e-12 <= step * h <= 1.2528125 + 1e-12:
                    t = (step - 1 + (row[1] + row[2]) / int(d["ticks"])) * h
                    result.append((path.stem, step, i, t, row[8:11]))
    return result


def replay(work, output, start, end, audits, native_only=None):
    with np.load(work / "assembly.npz") as d:
        bodies, mapping, pairs, stiffness, h = (
            d[k] for k in ("bodies", "mapping", "pairs", "stiffness", "h")
        )
    h = float(h) / 2
    paths = json.loads((work / "models.json").read_text())["parts"]
    samples = poses(work, h, start, end, audits, native_only)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "trajectory",
                "step",
                "trial",
                "t",
                "method",
                "stage",
                "face_a",
                "face_b",
                "probe_body",
                "depth_mm",
                "world_x_mm",
                "world_y_mm",
                "world_z_mm",
                "normal_x",
                "normal_y",
                "normal_z",
                "normal_N",
                "torque_a_Nm",
                "torque_b_Nm",
            ]
        )
        for p, (ia, ib) in enumerate(pairs):
            models = []
            for index in (ia, ib):
                with np.load(work / paths[index]["archive"]) as d:
                    models.append((d["tri"], d["faces"], d["points"]))
            joints = bodies[[ia, ib]]
            for method in ("cad", "mesh"):
                directions = []
                for a, b in ((0, 1), (1, 0)):
                    tri, tags, _ = models[a]
                    src, labels, pts = models[b]
                    model = ResidentMeshContact(
                        tri,
                        src,
                        tags,
                        labels,
                        pitch=0.8,
                        lanes=1,
                        source_points=pts if method == "cad" else None,
                        merge_planar=True,
                    )
                    keep = swept_samples(tri, model.samples[0].get(), joints[[a, b]])
                    if keep.any() and not keep.all():
                        idx = cp.asarray(keep)
                        model.samples = tuple(cp.ascontiguousarray(v[idx]) for v in model.samples)
                        model.n = int(keep.sum())
                    directions.append(model)
                prefer_reverse = (
                    sha256(models[0][0].tobytes()).digest() < sha256(models[1][0].tobytes()).digest()
                )
                for source, step, trial, t, q in samples:
                    angle = joints[:, 7] + t * joints[:, 8] + q[mapping[[ia, ib]]]
                    rot = Rotation.from_rotvec(joints[:, 1:4] * angle[:, None]).as_matrix()
                    origin = joints[:, 4:7]
                    hits = {}
                    for direction, (a, b) in enumerate(((0, 1), (1, 0))):
                        pose = np.column_stack((rot[a].T @ rot[b], rot[a].T @ (origin[b] - origin[a])))
                        for key, w in directions[direction].evaluate(pose).items():
                            if direction:
                                key = key[::-1]
                            depth = w[0]
                            if key in hits:
                                old = hits[key][0]
                                if depth < old - 1e-12 or (abs(depth - old) <= 1e-12 and not prefer_reverse):
                                    continue
                            point = rot[a] @ w[1:4] + origin[a]
                            normal = (1 if not direction else -1) * (rot[a] @ w[4:7])
                            force = 0.001 * stiffness[p] * depth
                            torque = np.einsum(
                                "ij,ij->i", joints[:, 1:4], np.cross(0.001 * (point - origin), normal) * force
                            )
                            hits[key] = (depth, b, point, normal, force, -torque[0], torque[1])
                    if not hits:
                        writer.writerow([source, step, trial, t, method, p + 1, -1, -1, -1] + [0.0] * 10)
                    for (fa, fb), (depth, b, pt, n, fn, ta, tb) in sorted(hits.items()):
                        writer.writerow(
                            [source, step, trial, t, method, p + 1, fa, fb, b, depth, *pt, *n, fn, ta, tb]
                        )
                print("finished stage", p + 1, method, len(samples), "poses", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--start", type=float, default=1.251)
    p.add_argument("--end", type=float, default=1.254)
    p.add_argument("--audit", type=Path, nargs="*", default=[])
    p.add_argument(
        "--native-only",
        type=Path,
        help="Replay only this native joint trajectory sampled at the deck time step",
    )
    args = p.parse_args()
    replay(args.work, args.output, args.start, args.end, args.audit, args.native_only)
