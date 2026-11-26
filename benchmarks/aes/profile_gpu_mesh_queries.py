"""Time complete finite triangle-pair detection at prescribed poses from an nc_contact trace."""

import argparse
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path

import cupy as cp
import meshio
import numpy as np
from scipy.spatial.transform import Rotation

import _gpu_source
from nurbscontact_gpu.bvh import IntersectionBatch, TriangleMesh


def read_poses(scene, response):
    frames = np.genfromtxt(response, delimiter=",", names=True)
    files, rotations, positions, pairs = [], [], [], []
    for line in scene.read_text().splitlines():
        if line.startswith("body "):
            fields = shlex.split(line)
            i = len(files)
            files.append((scene.parent / fields[1]).with_suffix(".msh"))
            axis = np.asarray(fields[6:9], float)
            axis /= np.linalg.norm(axis)
            rotations.append(
                Rotation.from_rotvec(
                    frames[f"b{i}_angle_rad"][:, None] * axis
                ).as_matrix()
            )
            positions.append(
                np.column_stack([frames[f"b{i}_{axis}_m"] for axis in "xyz"])
            )
        elif line.startswith("surface "):
            pairs.append(tuple(map(int, line.split()[1:3])))
    transforms = []
    for a, b in pairs:
        inverse = rotations[b].transpose(0, 2, 1)
        rel = np.empty((len(frames), 3, 4))
        rel[:, :, :3] = inverse @ rotations[a]
        rel[:, :, 3] = (inverse @ (positions[a] - positions[b])[:, :, None])[:, :, 0]
        transforms.append(cp.asarray(rel[1:]))
    return files, pairs, transforms, frames["time_s"]


STAMP = r"""
extern "C" __global__ void mark(unsigned long long* out) {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    *out = value;
}
"""


def measure(batch_query, transforms, profile, serial):
    """Replay in temporal order. Graphs reuse workspace, never previous intersections."""
    queries = batch_query.queries
    steps = len(transforms)
    batch = min(50, steps)
    stream = cp.cuda.Stream(non_blocking=True)
    poses = cp.empty_like(transforms[:batch])
    counts = cp.empty((batch, len(queries), 2), dtype=cp.int32)
    groups = len(queries) if serial else 1
    times = cp.empty((batch, groups, 3), dtype=cp.uint64)
    output = cp.empty((steps, len(queries), 2), dtype=counts.dtype)
    stamps = cp.empty((steps, groups, 3), dtype=times.dtype) if profile else None
    clock = cp.RawKernel(STAMP, "mark")
    clock.compile()
    batch_query.enqueue(transforms[0])
    if serial:
        for k, query in enumerate(queries):
            query.enqueue(transforms[0, k])
    cp.cuda.get_current_stream().synchronize()
    graphs = {}
    with stream:
        for length in sorted({batch, steps % batch} - {0}):
            stream.begin_capture()
            for j in range(length):
                for k, q in enumerate(queries if serial else (batch_query,)):
                    pose = poses[j, k] if serial else poses[j]
                    if profile:
                        clock((1,), (1,), (times[j, k, 0:1],))
                    q.enqueue_candidates(pose)
                    if profile:
                        clock((1,), (1,), (times[j, k, 1:2],))
                    q.enqueue_intersections(pose)
                    if profile:
                        clock((1,), (1,), (times[j, k, 2:3],))
                cp.copyto(counts[j], batch_query.counts)
            graphs[length] = stream.end_capture()
        cp.copyto(poses, transforms[:batch])
        graphs[batch].launch(stream)
        stream.synchronize()
        start = time.perf_counter()
        for i in range(0, steps, batch):
            size = min(batch, steps - i)
            cp.copyto(poses[:size], transforms[i : i + size])
            graphs[size].launch(stream)
            cp.copyto(output[i : i + size], counts[:size])
            if profile:
                cp.copyto(stamps[i : i + size], times[:size])
        stream.synchronize()
        elapsed = time.perf_counter() - start
    values = output.get()
    for k, q in enumerate(queries):
        if values[:, k, 0].max() > len(q.pairs) or values[:, k, 1].max() > len(
            q.candidates
        ):
            raise OverflowError("geometry workspace exhausted")
    return elapsed, values, stamps.get() if profile else None


def check_cpu_ids(exe, files, pairs, query, transforms, samples):
    """Match the native CPU predicate at identical placements, including coplanar hits."""
    matrices = transforms[samples].get()
    native = []
    for k, (a, b) in enumerate(pairs):
        lines = []
        for frame in matrices:
            pose = frame[k]
            x, y, z, w = Rotation.from_matrix(pose[:, :3]).as_quat()
            lines.append(
                " ".join(
                    format(v, ".17g")
                    for v in (*pose[:, 3], w, x, y, z, 0, 0, 0, 1, 0, 0, 0)
                )
            )
            # Reconstruct using the CPU tool's quaternion convention. This
            # matters for grazing triangles whose planes differ by a few ulps.
            pose[:, :3] = [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        proc = subprocess.run(
            [exe, "--method", "mesh-obb", files[a], files[b]],
            input="\n".join(lines) + "\n",
            text=True,
            capture_output=True,
            check=True,
            timeout=1800,
        )
        native.append(
            [
                {(h[2], h[3]) for h in json.loads(line)["hits"]}
                for line in proc.stdout.splitlines()
            ]
        )
        if len(native[-1]) != len(samples):
            raise ValueError("incomplete CPU reference")
    missed = extra = duplicates = total = 0
    for index, frame in enumerate(matrices):
        query.enqueue(cp.asarray(frame))
        for k, q in enumerate(query.queries):
            hits = q.read().tolist()
            found = set(map(tuple, hits))
            expected = native[k][index]
            missed += len(expected - found)
            extra += len(found - expected)
            duplicates += len(hits) - len(found)
            total += len(expected)
    return dict(
        steps=(samples + 1).tolist(),
        expected_triangle_pairs=total,
        missing=missed,
        extra=extra,
        duplicate=duplicates,
    )


def check_reference_ids(source, query, transforms):
    """Compare integer pair sets at every pose against the archived FP64-only kernels."""
    module = cp.RawModule(
        code=source.read_text(), options=("--std=c++14", "--fmad=false")
    )
    broad = module.get_function("leaf_candidates")
    narrow = module.get_function("intersect_leaves")
    compare = cp.RawKernel(
        r"""
    extern "C" __global__ void compare_pairs(const int2* old,const int* old_n,const int2* now,
            const int* now_n,int hit_cap,int leaf_cap,int* errors) {
        if(old_n[0]>hit_cap||now_n[0]>hit_cap||old_n[1]>leaf_cap||now_n[1]>leaf_cap) {
            if(!blockIdx.x&&!threadIdx.x) atomicAdd(errors+1,1);
            return;
        }
        if(old_n[0]!=now_n[0]&&!blockIdx.x&&!threadIdx.x) atomicAdd(errors,1);
        for(int j=blockIdx.x*blockDim.x+threadIdx.x;j<old_n[0];j+=blockDim.x*gridDim.x) {
            int matches=0;
            for(int k=0;k<now_n[0];++k)
                matches+=(old[j].x==now[k].x&&old[j].y==now[k].y);
            if(matches!=1) atomicAdd(errors,1);
        }
    }""",
        "compare_pairs",
    )
    compare.compile()
    errors = cp.zeros(2, cp.int32)
    buffers = [
        (cp.empty_like(q.candidates), cp.empty_like(q.pairs), cp.empty_like(q.count))
        for q in query.queries
    ]
    size = min(32, len(transforms))
    poses = cp.empty_like(transforms[:size])
    cp.cuda.get_current_stream().synchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    graphs = {}
    with stream:
        for length in sorted({size, len(transforms) % size} - {0}):
            stream.begin_capture()
            for frame in poses[:length]:
                query.enqueue(frame)
                for q, pose, (leaves, hits, totals) in zip(
                    query.queries, frame, buffers
                ):
                    a, b = q.first, q.second
                    totals.fill(0)
                    broad(
                        ((len(a.leaves) + 127) // 128,),
                        (128,),
                        (
                            a.leaves,
                            np.int32(len(a.leaves)),
                            a.bounds,
                            b.bounds,
                            b.links,
                            np.int32(len(b.links)),
                            pose,
                            leaves,
                            np.int32(len(leaves)),
                            totals,
                        ),
                    )
                    narrow(
                        (128,),
                        (128,),
                        (
                            a.triangles,
                            a.ids,
                            a.links,
                            b.triangles,
                            b.ids,
                            b.links,
                            pose,
                            leaves,
                            np.int32(len(leaves)),
                            hits,
                            np.int32(len(hits)),
                            totals,
                        ),
                    )
                    compare(
                        (8,),
                        (128,),
                        (
                            hits,
                            totals,
                            q.pairs,
                            q.count,
                            np.int32(len(hits)),
                            np.int32(len(leaves)),
                            errors,
                        ),
                    )
            graphs[length] = stream.end_capture()
        for start in range(0, len(transforms), size):
            chunk = transforms[start : start + size]
            cp.copyto(poses[: len(chunk)], chunk)
            graphs[len(chunk)].launch(stream)
        stream.synchronize()
    mismatches, overflows = map(int, errors.get())
    return dict(
        body_pair_poses=len(query.queries) * len(transforms),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        mismatches=mismatches,
        workspace_overflows=overflows,
    )


def run(
    scene,
    response,
    output,
    leaf_size,
    serial,
    cpu_check,
    reuse_distance,
    reference_source,
    profile_all,
    repeats,
):
    files, pairs, transforms, times = read_poses(scene, response)
    meshes = []
    for name in files:
        m = meshio.read(name)
        meshes.append(
            TriangleMesh(m.points, m.get_cells_type("triangle"), leaf_size=leaf_size)
        )
    batch_query = IntersectionBatch(
        [(meshes[a], meshes[b]) for a, b in pairs], reuse_distance=reuse_distance
    )
    transforms = cp.stack(transforms, axis=1)
    elapsed, counts, _ = measure(batch_query, transforms, False, serial)
    timings = [elapsed]
    for _ in range(1, repeats):
        elapsed, repeat_counts, _ = measure(batch_query, transforms, False, serial)
        np.testing.assert_array_equal(counts[:, :, 0], repeat_counts[:, :, 0])
        timings.append(elapsed)
    elapsed = float(np.median(timings))
    # Use the same per-pair order on a uniform subset to isolate GPU phases.
    samples = (
        np.arange(len(times) - 1)
        if reuse_distance or profile_all
        else np.unique(
            np.linspace(0, len(times) - 2, min(501, len(times) - 1), dtype=int)
        )
    )
    _, observed, stamps = measure(batch_query, transforms[samples], True, serial)
    np.testing.assert_array_equal(observed[:, :, 0], counts[samples, :, 0])
    us = np.diff(stamps, axis=2) * 1.0e-3
    report = {
        "measurement": "complete triangle-pair detection on prescribed poses",
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "physical_interval_s": float(times[-1] - times[0]),
        "steps": len(times) - 1,
        "body_pairs": pairs,
        "triangles": [m.count for m in meshes],
        "triangles_per_leaf_limit": leaf_size,
        "candidate_reuse_distance_m": reuse_distance,
        "mesh_sha256": [hashlib.sha256(f.read_bytes()).hexdigest() for f in files],
        "pose_trace_sha256": hashlib.sha256(response.read_bytes()).hexdigest(),
        "online_wall_s": elapsed,
        "online_wall_s_repeats": timings,
        "mean_us_per_frame": 1e6 * elapsed / (len(times) - 1),
        "profile_frames": len(samples),
        "phase_order": ["broad", "narrow"],
        "pair_dispatch": "sequential" if serial else "simultaneous",
        "mean_us_per_dispatch_per_phase": us.mean(axis=0).tolist(),
        "profiled_detection_p99_us": float(np.percentile(us.sum(axis=(1, 2)), 99)),
        "profiled_detection_steps_over_100_us": int(
            np.count_nonzero(us.sum(axis=(1, 2)) > 100)
        ),
        "median_hits_per_pair": np.median(counts[:, :, 0], axis=0).tolist(),
        "median_candidate_leaves_per_pair": np.median(counts[:, :, 1], axis=0).tolist(),
        "hit_count_history_sha256": hashlib.sha256(
            counts[:, :, 0].tobytes()
        ).hexdigest(),
        "max_hits_per_pair": counts[:, :, 0].max(axis=0).tolist(),
        "count_history_sha256": hashlib.sha256(counts.tobytes()).hexdigest(),
    }
    if cpu_check:
        activity = counts[:, :, 0] > 0
        switching = np.flatnonzero(np.any(activity[1:] != activity[:-1], axis=1)) + 1
        chosen = np.unique(
            np.r_[
                np.linspace(0, len(times) - 2, 26, dtype=int),
                switching[:12],
                counts[:, :, 0].argmax(axis=0),
            ]
        )
        report["cpu_triangle_id_check"] = check_cpu_ids(
            cpu_check, files, pairs, batch_query, transforms, chosen
        )
    if reference_source:
        report["full_fp64_id_check"] = check_reference_ids(
            reference_source, batch_query, transforms
        )
    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    audit = report.get("cpu_triangle_id_check", {})
    if any(audit.get(key, 0) for key in ("missing", "extra", "duplicate")):
        raise ValueError("triangle identities differ from CPU")
    if reference_source and any(
        report["full_fp64_id_check"][k] for k in ("mismatches", "workspace_overflows")
    ):
        raise ValueError("triangle identities differ from FP64 reference")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("scene", "response", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--leaf-size", type=int, default=8)
    parser.add_argument(
        "--serial",
        action="store_true",
        help="time independent pairs one after the other",
    )
    parser.add_argument(
        "--reuse-distance",
        type=float,
        default=0.0,
        help="maximum displacement for cached candidates, in scene units",
    )
    parser.add_argument(
        "--cpu-check",
        type=Path,
        help="native nc_mesh executable for independent triangle-ID validation",
    )
    parser.add_argument(
        "--reference-source",
        type=Path,
        help="original FP64 triangle kernel source for the full trajectory ID check",
    )
    parser.add_argument(
        "--profile-all",
        action="store_true",
        help="time both geometry phases individually for every main step",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        choices=range(1, 31),
        default=3,
        help="online timing repetitions; report their median",
    )
    run(**vars(parser.parse_args()))
