"""Separate time and geometry refinement on the same cam and publish repeated controls."""

import argparse
import json
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
from cam_accuracy_cost import prepare_level, reference_data, run_case
from resident_cad_probes import geometry_hash, parametric_probes


def inputs(work, base, level, step, cad):
    if not (work / f"deck-{level:g}.npz").exists():
        prepare_level(base, work, level)
    with np.load(work / f"deck-{level:g}.npz") as data:
        model = dict(data)
    with np.load(base / f"deck-0mm-{step:g}us.npz") as data:
        model.update(h=data["h"], loads=data["loads"])
    deck = work / f"level-{level:g}-h{step:g}.npz"
    np.savez_compressed(deck, **model)
    if not cad:
        return deck, None
    if level == 1.0:
        return deck, base / "cam-probes.npz"
    cache = work / f"cad-probes-{level:g}.npz"
    if not cache.exists():
        source = Path("results/hht/models/cam.step")
        start = perf_counter()
        points, faces, shift = parametric_probes(source, model["b"], model["fb"], level)
        np.savez_compressed(
            cache,
            points=points,
            face_probes=faces,
            edge_probes=len(points) - faces,
            geometry_sha256=geometry_hash(model["b"], model["fb"]),
            step_sha256=sha256(source.read_bytes()).hexdigest(),
            per_edge=1,
            max_shift_um=shift,
            projection_s=perf_counter() - start,
        )
    return deck, cache


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--base", type=Path, default=Path("build/resident-cam"))
    p.add_argument("--confirm", action="store_true", help="Repeat selected pure-mesh and SDF controls")
    p.add_argument(
        "--symmetric",
        action="store_true",
        help="Repeat comparisons after bidirectional contact and planar merging",
    )
    p.add_argument(
        "--symmetric-mesh-125", action="store_true", help="Matched coarse-step bidirectional mesh controls"
    )
    p.add_argument(
        "--mesh-density",
        action="store_true",
        help="Screen intermediate mesh vertex densities with bidirectional SDF contact",
    )
    p.add_argument(
        "--confirm-density",
        action="store_true",
        help="Repeat the first mesh density meeting the RMS ceiling in the 62.5 us sweep",
    )
    args = p.parse_args()
    if args.symmetric_mesh_125 or args.mesh_density or args.confirm_density:
        args.symmetric = True
    configs = [
        (125, 1.0, "cad", "sdf", 8.0, 3),
        (125, 1.0, "cad", "grid", 4.0, 3),
        (125, 0.0625, "cad", "grid", 4.0, 1),
        (62.5, 1.0, "cad", "grid", 4.0, 1),
        (62.5, 0.25, "mesh", "grid", 4.0, 1),
        (62.5, 0.0625, "mesh", "grid", 4.0, 1),
        (62.5, 0.015625, "mesh", "grid", 4.0, 1),
        (62.5, 0.0625, "vertices", "grid", 4.0, 1),
        (62.5, 0.0625, "mesh", "sdf", 8.0, 1),
    ]
    if args.confirm:
        configs = [
            (62.5, 0.0625, "vertices", "grid", 4.0, 3),
            (62.5, 0.0625, "mesh", "sdf", 8.0, 3),
            (62.5, 0.0625, "vertices", "sdf", 8.0, 3),
            (62.5, 1.0, "cad", "sdf", 8.0, 3),
        ]
    if args.symmetric:
        configs = [
            (125, 1.0, "cad", "grid", 4.0, 3),
            (125, 1.0, "cad", "sdf", 8.0, 3),
            (62.5, 0.0625, "vertices", "grid", 4.0, 3),
            (62.5, 0.0625, "vertices", "sdf", 8.0, 3),
            (62.5, 1.0, "cad", "sdf", 8.0, 3),
        ]
    if args.symmetric_mesh_125:
        configs = [(125, 1.0, "mesh", "grid", 4.0, 3), (125, 0.0625, "vertices", "sdf", 8.0, 3)]
    if args.mesh_density:
        configs = [
            (step, level, "vertices", "sdf", 8.0, 1)
            for step in (62.5, 125)
            for level in (1.0, 0.5, 0.25, 0.125)
        ]
    if args.confirm_density:
        configs = [(62.5, 0.125, "vertices", "sdf", 8.0, 3)]
    rows = []
    for step, level, method, backend, pitch, repeats in configs:
        ref = reference_data(args.work, round(20e6 / step) + 1, step * 1e-6)
        deck, probes = inputs(args.work, args.base, level, step, method == "cad")
        prefix = "symmetric" if args.symmetric else "checked"
        name = f"{prefix}-h{step:g}-{method}-{level:g}-{backend}"
        row = run_case(
            args.work,
            name,
            deck,
            ref,
            probes=probes,
            pitch=pitch,
            backend=backend,
            repeats=repeats,
            extra=["--lanes", "1"]
            + (["--vertex-probes"] if method == "vertices" else [])
            + (["--two-sided", "--merge-planar"] if args.symmetric else []),
        )
        row.update(step_us=step, deflection_um=level, method=method, backend=backend, pitch_mm=pitch)
        rows.append(row)
        args.output.write_text(json.dumps(rows, indent=2) + "\n")
        print(name, row["status"], row.get("rms_um"), row.get("online_s"), flush=True)
