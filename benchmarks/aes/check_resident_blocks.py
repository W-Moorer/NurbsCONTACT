"""Check the reusable GPU candidates against CPU box inclusion under rigid transforms."""

import argparse
import json
from pathlib import Path

import cupy as cp
import numpy as np
from scipy.spatial.transform import Rotation

from resident_blocks import block_index
from resident_contact import ResidentMeshContact


def box(half, subdivisions):
    vertices = np.linspace(-1.0, 1.0, subdivisions + 1)
    u, v = np.meshgrid(vertices, vertices)
    tris = []
    for axis in range(3):
        for side in (-1, 1):
            xyz = np.stack((np.full_like(u, side), u, v), axis=-1)
            xyz = np.roll(xyz, axis, axis=-1) * half
            a, b, c, d = xyz[:-1, :-1], xyz[:-1, 1:], xyz[1:, 1:], xyz[1:, :-1]
            face = np.stack((a, b, c, a, c, d), axis=2).reshape(-1, 3, 3)
            tris.append(face if side == 1 else face[:, ::-1])
    return np.ascontiguousarray(np.concatenate(tris)), np.repeat(np.arange(6), 2 * subdivisions**2)


def check_module():
    path = Path(__file__).parent
    code = (path / "resident_contact.cu").read_text() + "\n" + (path / "resident_blocks.cuh").read_text()
    module = cp.RawModule(
        code=code
        + """
    extern "C" __global__ void anchor(State *f, const PatchIndex *p, const double *pose) {
        for (int i=0; i<12; ++i) f->pose[i]=pose[i];
        prepare_patches(*f,*p);
    }
    extern "C" __global__ void rebuild(State *f, const PatchIndex *p, const MeshView *m) {
        if (f->cold) rebuild_patches(*f,*m,*p,blockDim.x*blockIdx.x+threadIdx.x);
    }
    extern "C" __global__ void size(const State *f, int *out) {
        out[0]=f->nc; out[1]=f->rebuilds;
    }
    extern "C" __global__ void subset(const State *f, const PatchIndex *p, const MeshView *m) {
        int thread = blockIdx.x*blockDim.x+threadIdx.x;
        for (int k=thread/4; k<f->nc; k+=gridDim.x*blockDim.x/4)
            sample_contact<4>(p->active[k], f->pose, VIEW((*m)));
    }
    """,
        options=("--std=c++17", "-DQUERY_LANES=4"),
        enable_cooperative_groups=True,
    )
    return module


def candidates(module, model, guard, poses):
    view, patches = cp.empty(320, cp.uint8), cp.empty(104, cp.uint8)
    module.get_function("bind_mesh")((1,), (1,), (*model.args(), view))
    index = block_index(model, guard)
    active = index[5]
    module.get_function("bind_patches")((1,), (1,), (patches, *index))
    frame, info = cp.zeros(1024, cp.uint8), cp.empty(2, cp.int32)
    for pose in poses:
        module.get_function("anchor")((1,), (1,), (frame, patches, cp.asarray(pose)))
        module.get_function("rebuild")((32,), (128,), (frame, patches, view))
        module.get_function("size")((1,), (1,), (frame, info))
        n, rebuilt = info.get()
        yield pose, active[:n], int(rebuilt), (frame, patches, view)


def check():
    half = np.array([4.0, 3.0, 2.0])
    target, ta = box(half, 1)
    source, sb = box(np.array([5.0, 2.0, 1.5]), 20)
    model = ResidentMeshContact(target, source, ta, sb, pitch=0.8, lanes=4)
    points = model.samples[0].get()
    module = check_module()
    rows = []
    # Start separated, enter within the guard, move far enough to rebuild, then rotate back out.
    # The central poses also put probes deep inside the target, away from its surface.
    for guard in (0.0, 0.24, 2.5, 20.0):
        prev = None
        hits = deep = entrants = warm = cold = 0
        poses = [
            np.column_stack(
                (Rotation.from_rotvec(np.array([1.0, 2.0, -1.0]) * angle).as_matrix(), [dx, 0.0, 0.0])
            )
            for dx, angle in (
                (9.05, 0),
                (8.95, 0),
                (8.88, 0.004),
                (7.9, 0.08),
                (0.0, 0.5),
                (0.02, 0.51),
                (12.0, 1.0),
            )
        ]
        for pose, active, rebuilt, _ in candidates(module, model, guard, poses):
            rot, shift = pose[:, :3], pose[:, 3]
            ids = active.get()
            assert len(ids) == len(np.unique(ids)) and np.all((ids >= 0) & (ids < model.n))
            kept = np.zeros(model.n, bool)
            kept[ids] = True
            depth = (half - np.abs(points @ rot.T + shift)).min(axis=1)
            inside = depth > 1e-9
            assert not (inside & ~kept).any()
            if rebuilt == cold:
                warm += 1
                entrants += int((inside & ~prev).sum())
            cold = int(rebuilt)
            hits += int(inside.sum())
            deep += int((depth > 1.0).sum())
            prev = inside
        assert hits and deep and (not guard or (warm and entrants))
        rows.append(
            dict(
                guard_mm=guard,
                poses=7,
                rebuilds=cold,
                reused=warm,
                inside_visits=hits,
                deeper_than_1mm=deep,
                entering_on_reuse=entrants,
                missed=0,
            )
        )
    return rows


def check_reducer(work):
    from resident_dynamics import swept_samples

    module = check_module()
    with np.load(work / "assembly.npz") as deck:
        pairs, bodies, mapping = (deck[k] for k in ("pairs", "bodies", "mapping"))
    parts = []
    for entry in json.loads((work / "models.json").read_text())["parts"]:
        with np.load(work / entry["archive"]) as data:
            parts.append(tuple(data[k] for k in ("tri", "faces", "points")))
    rows = []
    for method in ("cad", "mesh"):
        trace = np.load(work / f"pair-{method}.npy", mmap_mode="r")
        steps = [39, 40, 41, 42, 1599, 1600, 1601, 20042, 20043, 20044, 20045]
        times = [(step * 62.5e-6, trace[step, :3]) for step in steps]
        with np.load(work / f"full-{method}.npz") as audit:
            for row in audit["trials"]:
                if int(row[0]) == 20044:
                    times.append(
                        ((row[0] - 1 + (row[1] + row[2]) / int(audit["ticks"])) * 62.5e-6, row[8:11])
                    )
        times.sort(key=lambda item: item[0])
        for stage, (ia, ib) in enumerate(pairs, 1):
            for a, b in ((ia, ib), (ib, ia)):
                target, fa, _ = parts[a]
                source, fb, points = parts[b]
                model = ResidentMeshContact(
                    target,
                    source,
                    fa,
                    fb,
                    pitch=0.8,
                    lanes=4,
                    merge_planar=True,
                    source_points=points if method == "cad" else None,
                )
                joints = bodies[[a, b]]
                keep = swept_samples(target, model.samples[0].get(), joints)
                if keep.any() and not keep.all():
                    model.samples = tuple(cp.ascontiguousarray(x[keep]) for x in model.samples)
                    model.n = int(keep.sum())
                poses = []
                for t, q in times:
                    angle = joints[:, 7] + t * joints[:, 8] + q[mapping[[a, b]]]
                    rot = Rotation.from_rotvec(joints[:, 1:4] * angle[:, None]).as_matrix()
                    poses.append(
                        np.column_stack((rot[0].T @ rot[1], rot[0].T @ (joints[1, 4:7] - joints[0, 4:7])))
                    )
                hits = skipped = different = 0
                depth_error = 0.0
                for pose, active, rebuilt, args in candidates(module, model, 0.2, poses):
                    # The scalar full search reads every source probe without a neighbour cache.
                    model.witness.fill(0)
                    model.evaluate(pose)
                    reference = model.witness[: model.n, 0].copy()
                    member = cp.zeros(model.n, cp.bool_)
                    member[active] = True
                    hits += int(cp.count_nonzero(reference).get())
                    skipped += int(cp.count_nonzero((reference > 0) & ~member).get())
                    model.reset((128,), (128,), (model.keys, model.best, model.active, model.count))
                    model.count.fill(0)
                    model.witness.fill(0)
                    module.get_function("subset")((64,), (128,), args)
                    actual = model.witness[: model.n, 0]
                    different += int(cp.count_nonzero((reference > 0) != (actual > 0)).get())
                    depth_error = max(depth_error, float(cp.max(cp.abs(actual - reference)).get()))
                rows.append(
                    dict(
                        method=method,
                        stage=stage,
                        from_body=int(b),
                        to_body=int(a),
                        poses=len(poses),
                        rebuilds=rebuilt,
                        reference_hits=hits,
                        missed_candidates=skipped,
                        different_contact_flags=different,
                        max_depth_difference_mm=depth_error,
                    )
                )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reducer", type=Path)
    args = parser.parse_args()
    result = check_reducer(args.reducer) if args.reducer else check()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.reducer and any(
        r["missed_candidates"] or r["different_contact_flags"] or r["max_depth_difference_mm"] > 1e-9
        for r in result
    ):
        raise AssertionError(f"GPU candidates or witnesses differ from the exhaustive replay; see {args.output}")
