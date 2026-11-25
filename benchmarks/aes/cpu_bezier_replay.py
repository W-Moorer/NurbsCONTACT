"""Compare warmed CPU contact geometry on identical poses of the reducer."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from _gpu_source import read_model
from cpu_backend_run import digest
from cpu_cad_pressure_report import wrench


def run(scene, archive, binaries, out):
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((archive / "summary.json").read_text())
    trajectory = archive / "trajectory.npz"
    if digest(scene) != manifest["scene_sha256"] or digest(trajectory) != manifest["trajectory_sha256"]:
        raise ValueError("scene or reference trajectory mismatch")
    bodies, pairs = read_model(scene)
    with np.load(trajectory) as trace:
        origin, angles = trace["origins_m"], trace["angles_rad"]
    starts = [0, 280, 10000, 16650, 24950, 33320, 46290, 49900]
    indices = np.concatenate([np.arange(start, start + 100) for start in starts])
    measured = np.tile(np.arange(100) >= 20, len(starts))
    probes, mesh = binaries / "nc_cad_probes.exe", binaries / "nc_mesh.exe"
    summary = dict(
        scene_sha256=digest(scene),
        trajectory_sha256=digest(trajectory),
        binaries={exe.name: digest(exe) for exe in (probes, mesh)},
        starts=starts,
        window_length=100,
        warmup=20,
        stages=[],
    )
    rows = []
    methods = ["cad-patch", "cad-bezier", "mesh-obb"]
    for stage, (a, b, stiff, limit) in enumerate(pairs):
        info = {}
        geometry = [bodies[i][0] for i in (a, b)]
        for path in geometry + [g.with_suffix(".msh") for g in geometry]:
            if digest(path) != manifest["stages"][stage]["geometry_sha256"][path.name]:
                raise ValueError(f"geometry mismatch: {path}")
        all_columns = [
            col
            for i in (a, b)
            for col in [*origin[:, i].T, *[np.full(len(angles), x) for x in bodies[i][2]], angles[:, i]]
        ]
        poses = np.column_stack(all_columns)[indices]
        source = out / f"stage-{stage+1}.poses"
        np.savetxt(source, poses, fmt="%.17g")
        data = {}
        # Reverse the CAD order between the two gear pairs.
        for name in methods if stage == 0 else methods[::-1]:
            if name == "mesh-obb":
                cmd = [
                    mesh,
                    "--method",
                    name,
                    "--regions",
                    "contour-ray",
                    "--axis-angle",
                    "--keep-unresolved",
                    "--max-depth-m",
                    str(limit),
                    *[g.with_suffix(".msh") for g in geometry],
                ]
            else:
                cmd = [probes, *geometry, "6", str(2 * limit), "--loaded"]
                if name == "cad-bezier":
                    cmd.append("--bezier")
            tic = time.perf_counter()
            with source.open() as stdin:
                result = subprocess.run(
                    cmd, stdin=stdin, capture_output=True, text=True, check=True, timeout=1800
                )
            raw = list(map(json.loads, result.stdout.splitlines()))
            if len(raw) != len(indices):
                raise ValueError("incomplete replay")
            info[name] = dict(process_s=time.perf_counter() - tic, setup_log=result.stderr.strip())
            (out / f"{stage+1}-{name}.jsonl").write_text(result.stdout)
            samples = []
            for i, (step, record) in enumerate(zip(indices, raw, strict=True)):
                regions = record["regions"] if record.get("resolved", True) else None
                w = wrench(regions, origin[step, b], bodies[b][2], stiff, limit, name != "mesh-obb")
                sample = dict(
                    stage=stage + 1,
                    step=int(step),
                    measured=bool(measured[i]),
                    method=name,
                    regions=len(regions) if regions is not None else -1,
                    candidate_face_pairs=record.get("candidate_face_pairs"),
                    triangle_tests=record.get("triangle_tests"),
                    node_pairs=record.get("node_pairs"),
                    resolved=w is not None,
                    shaft_Nm=None if w is None else w[-1],
                    query_ms=1000 * record.get("loaded_s", record["query_s"]),
                )
                samples.append(sample)
            rows.extend(samples)
            data[name] = samples
            times = [s["query_ms"] for i, s in enumerate(samples) if measured[i]]
            info[name].update(
                mean_ms=np.mean(times),
                median_ms=np.median(times),
                p95_ms=np.percentile(times, 95),
                unresolved=sum(not s["resolved"] for s in samples),
                first_11=[s["regions"] for s in samples[:11]],
                step5_shaft_Nm=samples[5]["shaft_Nm"],
                ray_counts=raw[-1].get("ray_counts"),
            )
            if name != "mesh-obb":
                info[name]["mean_candidate_face_pairs"] = float(
                    np.mean([r["candidate_face_pairs"] for i, r in enumerate(raw) if measured[i]])
                )
            print(stage + 1, name, info[name], flush=True)
        for name in ("cad-bezier", "mesh-obb"):
            reference, current = data["cad-patch"], data[name]
            diffs = [
                abs(x["shaft_Nm"] - y["shaft_Nm"])
                for x, y in zip(reference, current)
                if x["resolved"] and y["resolved"]
            ]
            info[name].update(
                region_count_mismatches=sum(
                    x["regions"] != y["regions"] for x, y in zip(reference, current)
                ),
                resolved_flag_mismatches=sum(
                    x["resolved"] != y["resolved"] for x, y in zip(reference, current)
                ),
                max_resolved_torque_change_Nm=max(diffs, default=None),
                torque_pairs=len(diffs),
            )

        def grouped_faces(row):
            groups = {}
            for label, w in zip(row["labels"], row["witnesses"], strict=True):
                groups.setdefault(label, []).append(tuple(w[:2]))
            return sorted(sorted(faces) for faces in groups.values())

        old = (out / f"{stage+1}-cad-patch.jsonl").read_text().splitlines()
        new = (out / f"{stage+1}-cad-bezier.jsonl").read_text().splitlines()
        info["cad-bezier"]["face_partition_mismatches"] = sum(
            grouped_faces(json.loads(x)) != grouped_faces(json.loads(y))
            for x, y in zip(old, new, strict=True)
        )
        with Path("results/aes/cpu-cad-pressure/wrenches.csv").open() as f:
            reference = {
                int(r["step"]): float(r["shaft_Nm"])
                for r in csv.DictReader(f)
                if r["method"] == "pressure" and r["resolved"] == "True" and int(r["stage"]) == stage + 1
            }
        common = [
            i
            for i, step in enumerate(indices)
            if step in reference and all(data[n][i]["resolved"] for n in methods)
        ]
        for name in methods:
            delta = np.array([data[name][i]["shaft_Nm"] - reference[indices[i]] for i in common])
            info[name]["whole_step_comparison"] = dict(
                poses=len(common),
                rms_torque_Nm=np.sqrt(np.mean(delta**2)),
                max_abs_torque_Nm=np.max(abs(delta)),
            )
        summary["stages"].append(info)
    parts = list(dict.fromkeys(bodies[i][0] for a, b, _, _ in pairs for i in (a, b)))
    offline = dict(parts=[path.name for path in parts])
    with tempfile.TemporaryDirectory(prefix="nc-bezier-") as cache:
        env = {**os.environ, "NC_CAD_CACHE_DIR": cache}
        for mode, expected in (("cold", "built"), ("warm", "loaded")):
            tic = time.perf_counter()
            result = subprocess.run(
                [probes, "--prepare-bezier", *parts],
                env=env,
                capture_output=True,
                text=True,
                check=True,
                timeout=1800,
            )
            observations = list(map(json.loads, result.stdout.splitlines()))
            if len(observations) != len(parts) or any(row[expected] != 1 for row in observations):
                raise ValueError(f"incomplete {mode} offline preprocessing")
            offline[mode] = dict(process_s=time.perf_counter() - tic, by_part=observations)
        offline["bytes"] = sum(p.stat().st_size for p in Path(cache).glob("*.bin"))
    summary["bezier_tables"] = offline
    with (out / "poses.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("scene", "binaries", "out"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--archive", type=Path, default=Path("results/aes/cpu-local-cad"))
    args = parser.parse_args()
    run(args.scene, args.archive, args.binaries, args.out)
