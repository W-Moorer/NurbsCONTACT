"""Precompute mesh force and moment per unit penalty stiffness in two hinge coordinates.

This optional reduced contact model uses the CPU contour-ray operator on independent
configuration samples. Runtime interpolation is approximate at contact switches.
"""

import argparse
import hashlib
import io
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np
from .scene import read_model


def prepare(
    exe,
    scene,
    deck,
    out,
    phases,
    layers,
    workers,
    save_anchors=False,
    slip_bounds=(-0.0024, 0.0008),
):
    start = time.perf_counter()
    if phases < 2 or layers < 2 or not slip_bounds[0] < slip_bounds[1]:
        raise ValueError("Invalid configuration grid")
    cfg = np.load(deck / "assembly.npz")
    bodies, pairs = read_model(scene)
    offsets = np.linspace(*slip_bounds, layers)
    tables = np.empty((len(pairs), phases, layers, 3))
    valid = np.zeros(tables.shape[:3], bool)
    patches = np.empty(tables.shape[:3], object)
    patches.fill(None)
    periods = []
    ratios = []
    origins = []
    for k, (a, b, _, _) in enumerate(pairs):
        r = Fraction(float(cfg["bodies"][b, 8] / cfg["bodies"][a, 8])).limit_denominator(32)
        if abs(float(r) - cfg["bodies"][b, 8] / cfg["bodies"][a, 8]) > 1e-12:
            raise ValueError("nonperiodic velocity ratio")
        ratios.append(float(r))
        periods.append(2 * np.pi * r.denominator)
        origins.append(cfg["coord"][bodies[b][4], 3] - float(r) * cfg["coord"][bodies[a][4], 3])

    def sample(task):
        k, layer, begin, end = task
        a, b, _, limit = pairs[k]
        qa = np.arange(begin, end) * periods[k] / phases
        qb = qa * ratios[k] + origins[k] + offsets[layer]
        rows = []
        for i, angle in ((a, qa), (b, qb)):
            _, origin, axis, offset, j = bodies[i]
            rows.append(
                np.column_stack((np.tile(np.r_[origin, axis], (len(qa), 1)), angle + offset))
            )
        args = [
            str(exe),
            "--method",
            "mesh-obb",
            "--regions",
            "contour-ray",
            "--axis-angle",
            "--max-depth-m",
            str(limit),
            *(str(bodies[i][0].with_suffix(".msh")) for i in (a, b)),
        ]
        data = np.column_stack(rows)
        index = 0
        while index < len(data):
            buf = io.StringIO()
            np.savetxt(buf, data[index:], fmt="%.17g")
            proc = subprocess.run(
                args, input=buf.getvalue(), capture_output=True, text=True, timeout=900
            )
            for line in proc.stdout.splitlines():
                contacts = np.asarray(json.loads(line)["regions"])
                anchors = []
                for region in contacts if save_anchors else ():
                    point, normal, depth = region[:3], region[3:6], region[6]
                    record = []
                    for i, side in ((a, 1), (b, -1)):
                        _, origin, axis, offset, j = bodies[i]
                        theta = axis[2] * ((qa if i == a else qb)[index] + offset)
                        c, s = np.cos(theta), np.sin(theta)
                        inverse = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
                        record.extend(inverse @ (point + side * 0.5 * depth * normal - origin))
                        record.extend(inverse @ normal)
                    anchors.append(record)
                if save_anchors:
                    patches[k, begin + index, layer] = np.array(anchors).reshape(-1, 12)
                value = np.zeros(3)
                if len(contacts):
                    pos, n, d = contacts[:, :3], contacts[:, 3:6], contacts[:, 6]
                    value[0] = d.mean()
                    for side, (i, sign) in enumerate(((a, -1), (b, 1)), 1):
                        _, origin, axis, _, _ = bodies[i]
                        value[side] = np.mean(d * (np.cross(pos - origin, sign * n) @ axis))
                tables[k, begin + index, layer] = value
                valid[k, begin + index, layer] = True
                index += 1
            if proc.returncode:
                if "unresolved" not in proc.stderr.lower() and "witness" not in proc.stderr.lower():
                    raise RuntimeError(f"Mesh query exited {proc.returncode}: {proc.stderr}")
                index += 1
            elif index < len(data):
                raise ValueError("incomplete geometry output")

    # Long phase sweeps amortize BVH builds. Every failed geometric sample is explicitly masked.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(
            pool.map(sample, [(k, l, 0, phases) for k in range(len(pairs)) for l in range(layers)])
        )
    tables[~valid] = np.nan
    out.parent.mkdir(parents=True, exist_ok=True)
    patch_data = {}
    if save_anchors:
        lengths = np.array([len(x) if x is not None else 0 for x in patches.flat], np.int32)
        pointers = np.r_[0, np.cumsum(lengths, dtype=np.int32)].astype(np.int32)
        nonempty = [x for x in patches.flat if x is not None and len(x)]
        patch_data = dict(
            anchor_offsets=pointers,
            anchors=np.concatenate(nonempty) if nonempty else np.empty((0, 12)),
        )
    np.savez_compressed(
        out,
        **patch_data,
        table=tables,
        valid=valid,
        period=periods,
        ratio=ratios,
        origin=origins,
        slip=offsets,
        joint_poses=cfg["bodies"][:, 1:8],
        joint_map=cfg["mapping"],
        pair_ids=cfg["pairs"],
        prep_s=time.perf_counter() - start,
        scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
        mesh_sha256=[
            hashlib.sha256(b[0].with_suffix(".msh").read_bytes()).hexdigest() for b in bodies
        ],
        operator_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
    )
    print("geometry table", tables.shape, valid.mean(), time.perf_counter() - start, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("exe", "scene", "deck", "out"):
        p.add_argument(arg, type=Path)
    p.add_argument("--phases", type=int, default=1024)
    p.add_argument("--layers", type=int, default=17)
    p.add_argument(
        "--anchors",
        action="store_true",
        help="Also store material point pairs for the transport comparison",
    )
    p.add_argument(
        "--slip-bounds", nargs=2, type=float, default=(-0.0024, 0.0008), metavar=("MIN", "MAX")
    )
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    prepare(a.exe, a.scene, a.deck, a.out, a.phases, a.layers, a.workers, a.anchors, a.slip_bounds)
