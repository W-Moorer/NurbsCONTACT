"""Time FCL's whole-mesh pair distance at the poses of an audited STEP replay."""

import argparse
import csv
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import fcl
import numpy as np
from scipy.spatial.transform import Rotation

from gear_precision import mesh_step, check_closed


def make_body(vertices, faces):
    check_closed(vertices, faces)
    tree = fcl.BVHModel()
    tree.beginModel(len(faces), len(vertices))
    tree.addSubModel(vertices, faces)
    tree.endModel()
    return fcl.CollisionObject(tree)


def run(args):
    inputs = [args.pinion_mesh, args.wheel, args.poses, args.cad_pairs]
    cad_gap_mm = {}
    with args.cad_pairs.open(newline='') as stream:
        for row in csv.DictReader(stream):
            t = float(row['time_s'])
            gap = 1000 * float(row['distance_m'])
            cad_gap_mm[t] = min(cad_gap_mm.get(t, np.inf), gap)
    poses = []
    for t, x, y, z, *axis, angle in np.loadtxt(args.poses, ndmin=2):
        rotation = Rotation.from_rotvec(angle * np.asarray(axis) / np.linalg.norm(axis))
        poses.append((t, fcl.Transform(rotation.as_matrix(), 1000 * np.array([x, y, z]))))
    if set(t for t, _ in poses) != cad_gap_mm.keys():
        raise ValueError('Pose times and reference times must match.')

    start = perf_counter()
    with np.load(args.pinion_mesh) as mesh:
        body_a = make_body(mesh['v'], mesh['f'])
        triangles_a = len(mesh['f'])
    if args.wheel.suffix.lower() == '.npz':
        with np.load(args.wheel) as mesh:
            v, f = mesh['v'], mesh['f']
    else:
        v, f = mesh_step(args.deflection_um, interior=False, step=args.wheel)
    body_b = make_body(v, f)
    setup = perf_counter() - start
    request = fcl.DistanceRequest(enable_nearest_points=True)
    rows = [dict(time_s=t, cad_min_gap_um=1000 * cad_gap_mm[t], query_ms=[]) for t, _ in poses]
    for repeat in range(args.repeats + 1):
        for row, (_, transform) in zip(rows, poses, strict=True):
            begin = perf_counter()
            body_b.setTransform(transform)
            result = fcl.DistanceResult()
            distance = fcl.distance(body_a, body_b, request, result)
            elapsed_ms = 1000 * (perf_counter() - begin)
            p, q = result.nearest_points
            if distance <= 0 or not np.isclose(np.linalg.norm(p-q), distance, rtol=1e-8, atol=1e-10):
                raise ValueError('This separated-gear replay expects a positive gap and consistent witnesses.')
            if repeat:
                row['query_ms'].append(elapsed_ms)
            row['mesh_min_gap_um'] = 1000 * distance
    times = [elapsed for row in rows for elapsed in row['query_ms']]
    return dict(python_fcl=version('python-fcl'), triangles=[triangles_a, len(f)],
                wheel_deflection_um=None if args.wheel.suffix.lower()=='.npz' else args.deflection_um,
                setup_s=setup, repeats=args.repeats,
                median_ms=float(np.median(times)), min_ms=min(times), max_ms=max(times),
                max_abs_gap_difference_um=max(abs(r['mesh_min_gap_um']-r['cad_min_gap_um']) for r in rows),
                input_sha256={p.name: sha256(p.read_bytes()).hexdigest() for p in inputs}, poses=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pinion_mesh', 'wheel', 'poses', 'cad_pairs', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--deflection-um', type=float, default=0.04)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()
    if args.repeats < 1 or not np.isfinite(args.deflection_um) or args.deflection_um <= 0:
        parser.error('Expected positive repeats and mesh deflection.')
    args.output.write_text(json.dumps(run(args), indent=2, allow_nan=False) + '\n', encoding='utf-8')
