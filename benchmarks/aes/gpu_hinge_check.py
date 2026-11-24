"""Check the accelerated triangle gate and equilibrium outputs at independent poses."""

import argparse
import io
import json
import subprocess
from pathlib import Path

import cupy as cp
import numpy as np
from cpu_backend_pose_replay import read_model
from gpu_hinge_mesh import module, pack_mesh
from gpu_triangle_bvh import IntersectionQuery, TriangleMesh


def inspect(deck, work, run, cpu_archive, exe, scene, out):
    cfg = np.load(deck / "assembly.npz")
    ref = np.load(cpu_archive)
    gpu = np.load(run / "response.npz")
    parts = json.loads((deck / "models.json").read_text())["parts"]
    bodies, pairs = read_model(scene)
    q, t = gpu["q"], gpu["time_s"]
    np.testing.assert_allclose(t, ref["time_s"], rtol=0, atol=2e-11)
    delta = gpu["output_contact_torque_Nm"] - ref["output_contact_torque_Nm"]
    # Include the largest error, the complete startup, the brake and evenly spaced steady samples.
    ids = np.unique(
        np.r_[
            np.arange(0, 101),
            np.arange(1000, 1151, 5),
            np.linspace(0, len(t) - 1, 73, dtype=int),
            np.argmax(abs(delta)),
        ]
    )
    rng = np.random.default_rng(5179)
    diagnostic = []
    for stage, (a, b, k, limit) in enumerate(pairs):
        selected = [a, b]
        qa, qb = (q[ids, cfg["mapping"][i]] for i in selected)
        rows = []
        for i, angle in ((a, qa), (b, qb)):
            _, origin, axis, offset, _ = bodies[i]
            rows.append(
                np.column_stack((np.tile(np.r_[origin, axis], (len(ids), 1)), angle + offset))
            )
        buf = io.StringIO()
        np.savetxt(buf, np.column_stack(rows), fmt="%.17g")
        command = [
            str(exe),
            "--method",
            "mesh-obb",
            "--regions",
            "contour-ray",
            "--axis-angle",
            "--max-depth-m",
            str(limit),
            *(str(bodies[i][0].with_suffix(".msh")) for i in selected),
        ]
        proc = subprocess.run(
            command, input=buf.getvalue(), capture_output=True, text=True, check=True, timeout=1800
        )
        exact = np.zeros((len(ids), 3))
        regions = []
        for row, line in enumerate(proc.stdout.splitlines()):
            r = np.array(json.loads(line)["regions"])
            regions.append(len(r))
            if not len(r):
                continue
            p, n, d = r[:, :3], r[:, 3:6], r[:, 6]
            exact[row, 0] = k * d.mean()
            for col, (i, s) in enumerate(((a, -1), (b, 1)), 1):
                exact[row, col] = k * np.mean(
                    d * (np.cross(p - bodies[i][1], s * n) @ bodies[i][2])
                )
        if len(regions) != len(ids):
            raise ValueError("Incomplete CPU response")

        triangles = [pack_mesh(deck / parts[i]["archive"]) for i in selected]
        av, bv = (cp.asarray(m[0]) for m in triangles)
        tree = [TriangleMesh(m[1], m[2], leaf_size=6) for m in triangles]
        query = IntersectionQuery(*tree, capacity=8192)
        window = np.load(work / f"windows-{stage}.npz")
        offsets, packed = window["offsets"], window["pairs"]
        period, ratio, band = (window[key].item() for key in ("period", "ratio", "band"))
        geom = cfg["bodies"][selected]
        origin = geom[1, 12] - ratio * geom[0, 12]
        param = cp.asarray(
            np.r_[
                geom[0, 4:7] * 0.001,
                geom[1, 4:7] * 0.001,
                geom[:, 3],
                geom[:, 7],
                period,
                ratio,
                origin,
                band,
                k,
            ]
        )
        raw = module()
        kern = raw.get_function("check_window")
        pose_kernel = raw.get_function("audit_pose")
        device_pairs = cp.asarray(packed)
        m = cp.empty((3, 4), cp.float64)
        count = len(offsets) - 1
        extra_phase = np.r_[0.0, -1e-12, period - 1e-12, rng.uniform(-2 * period, 2 * period, 24)]
        extra_slip = np.resize([band, -band], len(extra_phase))
        tests = list(zip(qa, qb)) + list(
            zip(extra_phase, extra_phase * ratio + origin + extra_slip)
        )
        misses = extra = full_hits = 0
        for p1, p2 in tests:
            idx = int(np.floor((p1 / period % 1) * count + 0.5)) % count
            start, end = offsets[idx : idx + 2]
            mask = cp.empty(int(end - start), cp.int32)
            kern(
                ((len(mask) + 127) // 128,),
                (128,),
                (
                    param,
                    av,
                    bv,
                    device_pairs,
                    start,
                    np.int32(len(mask)),
                    np.float64(p1),
                    np.float64(p2),
                    mask,
                ),
            )
            hit = packed[start:end][mask.get().astype(bool)]
            pose_kernel((1,), (1,), (param, np.float64(p1), np.float64(p2), m))
            query.enqueue(m)
            reference = set(map(tuple, query.read()))
            actual = set(map(tuple, hit))
            misses += len(reference - actual)
            extra += len(actual - reference)
            full_hits += len(reference)
        model_load = gpu["stage_normal_load_N"][ids, stage]
        worst = int(np.argmax(abs(delta)))
        row = int(np.flatnonzero(ids == worst)[0])
        report = dict(
            stage=stage + 1,
            full_bvh_poses=len(tests),
            reference_crossing_pairs=full_hits,
            missing=misses,
            extra=extra,
            exact_cpu_geometry_poses=len(ids),
            region_count_range=[int(min(regions)), int(max(regions))],
            same_pose_normal_load_rms_N=float(np.sqrt(np.mean((exact[:, 0] - model_load) ** 2))),
            worst_time_s=float(t[worst]),
            worst_exact_load_N=exact[row, 0],
            worst_online_load_N=model_load[row],
        )
        if stage == len(pairs) - 1:
            report.update(
                worst_same_gpu_pose_cpu_torque_Nm=float(exact[row, 2]),
                worst_online_torque_Nm=float(gpu["output_contact_torque_Nm"][worst]),
                worst_cpu_trajectory_torque_Nm=float(ref["output_contact_torque_Nm"][worst]),
            )
        diagnostic.append(report)
    out.write_text(json.dumps(diagnostic, indent=2) + "\n")
    print(json.dumps(diagnostic), flush=True)
    if any(row["missing"] or row["extra"] for row in diagnostic):
        raise AssertionError("Windowed finite-triangle tests disagree with full BVH")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("deck", "work", "run", "cpu", "exe", "scene", "out"):
        p.add_argument(arg, type=Path)
    a = p.parse_args()
    inspect(a.deck, a.work, a.run, a.cpu, a.exe, a.scene, a.out)
