"""Validate and run current-pose GPU BVH mesh dynamics on the original CPU scene."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter
import numpy as np
from scipy.optimize import linear_sum_assignment
import _gpu_source
from nurbscontact_gpu.mesh_dynamics import MeshDynamics


def rotations(axis, angles):
    """Use the joint Rodrigues map, including exact invariant-axis entries."""
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    cosine = np.cos(angles)[:, None, None]
    sine = np.sin(angles)[:, None, None]
    return cosine * np.eye(3) + sine * skew + (1 - cosine) * np.outer(axis, axis)


def audit(model, trace, executable, out, predictor_step=None, extra_steps=()):
    frames = np.genfromtxt(trace, delimiter=",", names=True)
    if predictor_step is not None:
        coordinates = np.stack(
            [
                frames[f"q{i}"]
                + predictor_step * frames[f"v{i}"]
                + 0.5 * predictor_step**2 * frames[f"a{i}"]
                for i in range(model.n)
            ],
            axis=1,
        )
        for i, b in enumerate(model.bodies):
            angle = (
                coordinates[:, int(b["hinge"])]
                if b["hinge"] >= 0
                else np.zeros(len(frames))
            )
            rotation = rotations(b["axis"], angle)
            origin = (
                b["pivot"] + (rotation @ (b["origin"] - b["pivot"])[:, None])[:, :, 0]
            )
            if b["translation"] >= 0:
                origin += coordinates[:, int(b["translation"]), None] * b["direction"]
            for j, axis in enumerate("xyz"):
                frames[f"b{i}_{axis}_m"] = origin[:, j]
            frames[f"b{i}_angle_rad"] = b["angle"] + angle
    indices = np.unique(
        np.r_[
            np.arange(65),
            100,
            500,
            1000,
            np.linspace(0, len(frames) - 1, 17, dtype=int),
        ]
    )
    indices = np.unique(np.r_[indices, np.asarray(extra_steps, dtype=int)])
    if np.any(indices < 0) or np.any(indices >= len(frames)):
        raise ValueError("Audit step outside CPU trace")
    matrices = []
    origins = []
    for i, b in enumerate(model.bodies):
        matrices.append(rotations(b["axis"], frames[f"b{i}_angle_rad"][indices]))
        origins.append(
            np.column_stack([frames[f"b{i}_{axis}_m"][indices] for axis in "xyz"])
        )
    transforms = []
    native = []
    for law in model.laws:
        a, b = int(law["a"]), int(law["b"])
        ra, rb = matrices[a], matrices[b]
        m = np.empty((len(indices), 3, 4))
        m[:, :, :3] = rb.transpose(0, 2, 1) @ ra
        m[:, :, 3] = (rb.transpose(0, 2, 1) @ (origins[a] - origins[b])[:, :, None])[
            :, :, 0
        ]
        transforms.append(m)
        rows = []
        for j, index in enumerate(indices):
            row = []
            for body in (a, b):
                row.extend(
                    [
                        *origins[body][j],
                        *model.bodies[body]["axis"],
                        frames[f"b{body}_angle_rad"][index],
                    ]
                )
            rows.append(" ".join(format(x, ".17g") for x in row))
        result = subprocess.run(
            [
                str(executable),
                "--method",
                "mesh-obb",
                "--regions",
                "contour-ray",
                "--axis-angle",
                "--max-depth-m",
                str(law["max_depth"]),
                str(model.files[a]),
                str(model.files[b]),
            ],
            input="\n".join(rows) + "\n",
            text=True,
            capture_output=True,
            check=True,
            timeout=1800,
        )
        native.append(
            [json.loads(line)["regions"] for line in result.stdout.splitlines()]
        )
    records = []
    worst = np.zeros(3)
    count_errors = 0
    for j, index in enumerate(indices):
        placements = np.array(
            [
                [
                    *origins[b][j],
                    *model.bodies[b]["axis"],
                    frames[f"b{b}_angle_rad"][index],
                ]
                for b in range(model.nb)
            ]
        )
        found = model.probe_world(placements)
        for k, rows in enumerate(found):
            rows = rows.copy()
            expected = np.asarray(native[k][j]).reshape(-1, 7)
            record = dict(
                step=int(index), pair=k, cpu=expected.tolist(), gpu=rows.tolist()
            )
            if len(expected) != len(rows):
                count_errors += 1
                record["count_mismatch"] = True
            elif len(rows):
                x, y = linear_sum_assignment(
                    np.linalg.norm(expected[:, None, :3] - rows[None, :, :3], axis=2)
                )
                delta = expected[x] - rows[y]
                errors = np.array(
                    [
                        np.linalg.norm(delta[:, :3], axis=1).max(),
                        np.linalg.norm(delta[:, 3:6], axis=1).max(),
                        abs(delta[:, 6]).max(),
                    ]
                )
                record["max_point_normal_depth_difference"] = errors.tolist()
                worst = np.maximum(worst, errors)
            records.append(record)
    report = dict(
        scope="same-pose CPU finite/contour-ray versus GPU triangle BVH contacts",
        predictor_step=predictor_step,
        steps=indices.tolist(),
        count_mismatches=count_errors,
        max_point_normal_depth_difference=worst.tolist(),
        records=records,
    )
    (out / "geometry-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Geometry audit", count_errors, worst, flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scene", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--step", type=float, default=0.0001)
    p.add_argument("--cpu-trace", type=Path)
    p.add_argument("--cpu-query", type=Path)
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--predictor-audit", action="store_true")
    p.add_argument(
        "--audit-steps", type=Path, help="JSON list of additional saved step indices"
    )
    p.add_argument("--register-limit", type=int)
    p.add_argument("--serial-contour", action="store_true")
    p.add_argument(
        "--execution", choices=("graph", "cooperative"), default="cooperative"
    )
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    model = MeshDynamics(
        a.scene,
        round(a.duration / a.step),
        blocks=a.blocks,
        cache_dir=a.out.parent / "preprocessing",
        register_limit=a.register_limit,
        parallel_contour=not a.serial_contour,
    )
    print("Prepared", model.setup_s, flush=True)
    if a.cpu_trace and a.cpu_query:
        audit(
            model,
            a.cpu_trace,
            a.cpu_query,
            a.out,
            a.step if a.predictor_audit else None,
            json.loads(a.audit_steps.read_text()) if a.audit_steps else (),
        )
    if a.audit_only:
        return
    started = perf_counter()
    try:
        data, stats = model.run(a.step, execution=a.execution)
        stats["completed"] = True
    except RuntimeError as error:
        status = model.status().tolist()
        if not status[0]:
            raise
        saved = np.flatnonzero(model.finished.get())
        last = int(saved[-1]) if len(saved) else -1
        data = model.output.get()[: last + 1]
        stats = dict(
            completed=False,
            error=str(error),
            status=status,
            setup_s=model.setup_s,
            attempt_wall_s=perf_counter() - started,
            attempt_timing_scope=(
                "Includes graph instantiation when applicable and error/status downloads; "
                "not a completed simulation timing."
            ),
            last_saved_step=last,
        )
    np.save(a.out / "response.npy", data)
    stats.update(
        duration_s=a.duration,
        step_s=a.step,
        scene_sha256=hashlib.sha256(a.scene.read_bytes()).hexdigest(),
        mesh_sha256=[hashlib.sha256(f.read_bytes()).hexdigest() for f in model.files],
        method="GPU BVH triangle intersection + finite/contour-ray support",
        steps=model.steps,
    )
    (a.out / "run.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k != "kernels"}, indent=2), flush=True)
    if not stats["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
