"""Compare GPU triangle-query scheduling on the entire frozen CPU trajectory."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import cupy as cp
import meshio
import numpy as np
from profile_gpu_mesh_queries import read_poses, measure, check_reference_ids
from nurbscontact_gpu.bvh import TriangleMesh, IntersectionBatch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scene", type=Path)
    p.add_argument("response", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--sweep", action="store_true", help="Also compare alternative group widths")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    files, pairs, poses, times = read_poses(a.scene, a.response)
    meshes = []
    for path in files:
        m = meshio.read(path)
        meshes.append(TriangleMesh(m.points, m.get_cells_type("triangle"), leaf_size=4))
    poses = cp.stack(poses, axis=1)
    reference = None
    results = {}
    queries = {}
    cases = (1, 4, 8, 16, 32) if a.sweep else (1, 8)
    for lanes in cases:
        name = f"lanes-{lanes}"
        q = IntersectionBatch([(meshes[i], meshes[j]) for i, j in pairs], fine_lanes=lanes)
        queries[name] = q
        durations = []
        for _ in range(3):
            seconds, counts, _ = measure(q, poses, False, False)
            durations.append(seconds)
            if reference is None:
                reference = counts.copy()
            np.testing.assert_array_equal(counts, reference)
        results[name] = dict(
            online_s=durations, median_s=float(np.median(durations)), fine_lanes=lanes
        )
        print(name, results[name], flush=True)
    best = min(results, key=lambda k: results[k]["median_s"])
    _, _, stamps = measure(queries[best], poses, True, False)
    phase = np.diff(stamps, axis=2) * 1.0e-3
    _, _, baseline_stamps = measure(queries["lanes-1"], poses, True, False)
    baseline_phase = np.diff(baseline_stamps, axis=2) * 1.0e-3
    raw = a.out / "fp64.cu"
    raw.write_bytes(
        subprocess.check_output(
            ["git", "show", "7a87521:nurbscontact_gpu/kernels/gpu_triangle_intersections.cu"]
        )
    )
    check = check_reference_ids(raw, queries[best], poses)
    if check["mismatches"] or check["workspace_overflows"]:
        raise ValueError(check)
    report = dict(
        measurement="triangle-pair detection only, prescribed full 5 s trajectory",
        gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        steps=len(poses),
        physical_s=float(times[-1]),
        triangles=[m.count for m in meshes],
        mesh_sha256=[hashlib.sha256(f.read_bytes()).hexdigest() for f in files],
        response_sha256=hashlib.sha256(a.response.read_bytes()).hexdigest(),
        runs=results,
        best=best,
        hit_count_sha256=hashlib.sha256(reference[:, :, 0].tobytes()).hexdigest(),
        full_fp64_id_check=check,
        baseline_mean_phase_us=baseline_phase.mean(axis=0).tolist(),
        source_sha256={
            str(f): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in (
                Path("nurbscontact_gpu/bvh.py"),
                Path("nurbscontact_gpu/kernels/gpu_triangle_intersections.cu"),
            )
        },
        mean_phase_us=phase.mean(axis=0).tolist(),
        p99_us=float(np.percentile(phase.sum(axis=(1, 2)), 99)),
        steps_over_100_us=int(np.count_nonzero(phase.sum(axis=(1, 2)) > 100)),
    )
    (a.out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        a.out / "profile.npz",
        time_s=times[1:],
        before_us=baseline_phase.sum(axis=(1, 2)),
        after_us=phase.sum(axis=(1, 2)),
        intersections=reference[:, :, 0],
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
