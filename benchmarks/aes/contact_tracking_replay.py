"""Compare full GPU overlap search with spatial patches and motion-bounded reuse."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import cupy as cp
import numpy as np

from gpu_mesh_line import read_full_mesh
from gpu_mesh_pairs import MeshPair
from gpu_patch_tracker import FacePairHistory, PatchTracker


def load_tags(path, mesh):
    with np.load(path) as tags:
        if str(tags["mesh_sha256"]) != sha256(mesh.read_bytes()).hexdigest():
            raise ValueError("CAD tags correspond to a different mesh")
        return tags["face"]


def normals(mesh):
    normal = np.cross(mesh[:, 1] - mesh[:, 0], mesh[:, 2] - mesh[:, 0])
    return normal / np.linalg.norm(normal, axis=1)[:, None]


def run(args):
    a, b = read_full_mesh(args.mesh_a), read_full_mesh(args.mesh_b)
    fa, fb = load_tags(args.faces_a, args.mesh_a), load_tags(args.faces_b, args.mesh_b)
    na, nb = normals(a), normals(b)
    raw = np.loadtxt(args.poses, ndmin=2)
    poses = raw[:, 1:].reshape(-1, 3, 4)
    matrices = [cp.asarray(x) for x in poses]
    translations = np.max(np.abs(poses[:, :, 3]), axis=1)
    orders = {}
    ref = MeshPair(a, b, orders=orders)
    tracker = PatchTracker(a, b, fa, fb, args.leaf_size)
    spatial = (
        None
        if args.patches_only
        else PatchTracker(a, b, np.zeros_like(fa), np.zeros_like(fb), args.leaf_size)
    )
    out = cp.empty((4096, 2), cp.int32)
    count = cp.empty(1, cp.int32)

    def reference(i):
        nonlocal out
        while True:
            count.fill(0)
            ref.query(matrices[i], 0.0, 1e-10, translations[i], True, out, count)
            n = int(count.get()[0])
            if n <= len(out):
                return out[:n]
            out = cp.empty((n, 2), cp.int32)

    oracle, face_reference = [], []
    previous = set()
    face_events = []
    digest = sha256()
    n_entry = n_exit = total = 0
    for i, t in enumerate(raw[:, 0]):
        hits = reference(i).get()
        ia, ib = orders[0][hits[:, 0]], hits[:, 1]
        keys = np.sort(ia.astype(np.int64) * len(b) + ib)
        oracle.append(keys)
        face_reference.append(np.unique((fa[ia].astype(np.uint64) << 32) | fb[ib].astype(np.uint64)))
        total += len(keys)
        digest.update(np.asarray([len(keys)], "<i8").tobytes())
        digest.update(keys.astype("<i8", copy=False).tobytes())
        opposed = np.einsum("ij,ij->i", na[ia], nb[ib] @ poses[i, :, :3].T) < -0.95
        faces = set(zip(fa[ia[opposed]].tolist(), fb[ib[opposed]].tolist()))
        if faces != previous:
            n_entry += len(faces - previous)
            n_exit += len(previous - faces)
            face_events.append(
                dict(
                    time_s=t,
                    frame=i,
                    active_pairs=sorted(faces),
                    entered=sorted(faces - previous),
                    left=sorted(previous - faces),
                )
            )
        previous = faces
    controls = [] if args.patches_only else [("full", None, 0), ("spatial", 0.0, 0)]
    methods = controls + [(f"patch_{guard:g}mm", guard, 0) for guard in args.guards]
    methods += [(f"envelope_{n}step", args.slack_mm, n) for n in args.lookahead]
    timing = {key: [] for key, _, _ in methods}
    with_faces = {key: [] for key, _, _ in methods}
    builds, maximum = {}, {}
    for repeat in range(args.repeats + 1):
        order = methods[repeat % len(methods) :] + methods[: repeat % len(methods)]
        for key, guard, lookahead in order:
            detector = spatial if key == "spatial" else tracker
            detector.reset()
            history = FacePairHistory(fa[orders[0]] if guard is None else fa, fb, tracker.module)
            rebuilt = peak = 0
            elapsed, complete = [], []
            cp.cuda.get_current_stream().synchronize()
            for i, pose in enumerate(matrices):
                start = perf_counter()
                if guard is None:
                    result = reference(i)
                else:
                    result, n, reset = detector.query(pose, guard, translations[i], lookahead)
                    rebuilt += reset
                    peak = max(peak, n)
                elapsed.append(1000 * (perf_counter() - start))
                current, entered, left = history.update(result)
                cp.cuda.get_current_stream().synchronize()
                complete.append(1000 * (perf_counter() - start))
                if repeat == 0:
                    hits = result.get()
                    ia = orders[0][hits[:, 0]] if guard is None else hits[:, 0]
                    keys = np.sort(ia.astype(np.int64) * len(b) + hits[:, 1])
                    if not np.array_equal(keys, oracle[i]):
                        missing = np.setdiff1d(oracle[i], keys).size
                        extra = np.setdiff1d(keys, oracle[i]).size
                        raise ValueError(f"{key} frame {i}: missing={missing}, extra={extra}")
                    prev = face_reference[i - 1] if i else np.empty(0, np.uint64)
                    for found, truth in [
                        (current, face_reference[i]),
                        (entered, np.setdiff1d(face_reference[i], prev)),
                        (left, np.setdiff1d(prev, face_reference[i])),
                    ]:
                        np.testing.assert_array_equal(np.sort(found.get()), truth)
            builds[key], maximum[key] = rebuilt, peak
            if repeat:
                timing[key].append(elapsed)
                with_faces[key].append(complete)
    metrics = {}
    for key, guard, lookahead in methods:
        mat = np.asarray(timing[key])
        metrics[key] = dict(
            mean_ms=float(mat.mean()),
            median_ms=float(np.median(mat)),
            p95_ms=float(np.percentile(mat, 95)),
            run_s=(mat.sum(axis=1) / 1000).tolist(),
            mean_with_face_updates_ms=float(np.mean(with_faces[key])),
        )
        if guard is not None:
            metrics[key].update(
                guard_mm=guard,
                lookahead_steps=lookahead,
                rebuilds=builds[key],
                max_candidate_patch_pairs=maximum[key],
            )
    return dict(
        gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        repeats=args.repeats,
        triangles=[len(a), len(b)],
        patches=[len(tracker.ba), len(tracker.bb)],
        spatial_patches=[len(spatial.ba), len(spatial.bb)] if spatial is not None else None,
        leaf_size=args.leaf_size,
        frames=len(raw),
        start_s=raw[0, 0],
        end_s=raw[-1, 0],
        input_sha256={
            p.name: sha256(p.read_bytes()).hexdigest()
            for p in (args.poses, args.mesh_a, args.mesh_b, args.faces_a, args.faces_b)
        },
        all_hit_pair_sets_equal=True,
        full_search_hit_sha256=digest.hexdigest(),
        intersecting_pairs_over_frames=total,
        nonempty_frames=sum(bool(len(x)) for x in oracle),
        opposed_face_entries=n_entry,
        opposed_face_exits=n_exit,
        face_events=face_events,
        metrics=metrics,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mesh_a", "mesh_b", "faces_a", "faces_b", "poses", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--leaf-size", type=int, default=16)
    parser.add_argument(
        "--patches-only", action="store_true", help="Time only CAD patch policies; audit against full search"
    )
    parser.add_argument("--guards", nargs="+", type=float, default=[0.0, 1.0, 4.0])
    parser.add_argument("--lookahead", nargs="*", type=int, default=[4])
    parser.add_argument("--slack-mm", type=float, default=0.05)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    if (
        args.repeats < 1
        or args.leaf_size < 1
        or min(args.guards) < 0
        or not np.isfinite(args.guards).all()
        or any(n < 1 for n in args.lookahead)
        or not np.isfinite(args.slack_mm)
        or args.slack_mm < 0
    ):
        parser.error("Invalid replay settings")
    args.output.write_text(json.dumps(run(args), indent=2, allow_nan=False) + "\n", encoding="utf-8")
