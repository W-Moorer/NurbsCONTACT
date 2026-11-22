"""Rebuild or time the point queries from the two archived tooth-transition windows."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from gear_field_timing import accuracy


MODEL = Path('results/aes/gear-field')
SAMPLES = MODEL / 'transition-points.csv'


def prepare(executable, poses, out):
    from nhrep_gear import contact_samples
    out.mkdir(parents=True, exist_ok=True)
    parts = [Path('results/gears/models') / f'spur-1{side}.step' for side in 'ab']
    rows = []
    for i, row in enumerate(poses.read_text().splitlines()):
        time, *motion = row.split()
        prefix = out / f'regions-{i}'
        subprocess.run([executable, 'regions', *parts, '0.000055', *motion, prefix], check=True)
        with prefix.with_suffix('.stats.csv').open(newline='') as stream:
            stats, = csv.DictReader(stream)
        if int(stats['unresolved_face_pairs']):
            raise ValueError(f'Incomplete STEP contact search at {time} s')
        for point, weight, pair in contact_samples(prefix.with_suffix('.csv')):
            rows.append((time, pair, weight, *point))
    coordinates = out / 'transition-points.txt'
    np.savetxt(coordinates, np.asarray([r[3:] for r in rows]), fmt='%.17g')
    distances = out / 'transition-distances.csv'
    subprocess.run([executable, 'cad-point-distance', parts[0], coordinates, distances], check=True)
    truth = np.loadtxt(distances, delimiter=',', skiprows=1, usecols=0, ndmin=1)
    with SAMPLES.open('w', encoding='ascii', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(('time_s', 'face_pair', 'weight_m', 'x_mm', 'y_mm', 'z_mm', 'cad_distance_mm'))
        writer.writerows((*row, d) for row, d in zip(rows, truth, strict=True))


def timed(methods, frames, fixed, probes, ids, repeats):
    # Each measured frame is one call through the public signed-distance API.
    samples = {name: ([], [], [], []) for name, _, _ in methods}
    for rep in range(repeats):
        # Rotate the order so each backend sees different positions in the run.
        start_at = rep % len(methods)
        for name, predict, _ in methods[start_at:]+methods[:start_at]:
            moving, singles, batches, frozen = samples[name]
            for p in frames:
                start = perf_counter()
                predict(p)
                moving.append((perf_counter()-start)*1e6)
            for i in ids:
                one = probes[i:i+1]
                start = perf_counter()
                predict(one)
                singles.append((perf_counter()-start)*1e6)
            start = perf_counter()
            predict(probes)
            batches.append((perf_counter()-start)*1e3)
            start = perf_counter()
            for _ in range(8):
                predict(fixed)
            frozen.append((perf_counter()-start)*1e6/8)
    result = {}
    for name, (moving, singles, batches, frozen) in samples.items():
        moving = np.reshape(moving, (repeats, len(frames)))
        result[name] = {'two_points_us': float(np.median(moving[:,[len(p)==2 for p in frames]])),
                        'four_points_us': float(np.median(moving[:,[len(p)==4 for p in frames]])),
                        'all_frames_p95_us': float(np.percentile(moving,95)),
                        'fixed_four_points_us': float(np.median(frozen)),
                        'single_us': float(np.median(singles)), 'mixed_batch_ms': float(np.median(batches))}
    return result


def benchmark(repeats):
    import open3d as o3d
    from gear_field import FittedFaces
    from fitted_projection import refine
    with SAMPLES.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    times = np.array([float(row['time_s']) for row in rows])
    xyz = np.array([[float(row[k+'_mm']) for k in 'xyz'] for row in rows])
    frames = [xyz[times==t] for t in np.unique(times)]
    truth = np.array([float(row['cad_distance_mm']) for row in rows])
    pairs = np.array([row['face_pair'] for row in rows])
    weights = np.array([float(row['weight_m']) for row in rows])
    seeds = (1407,5119)
    probes, reference = [], []
    for seed, source in zip(seeds, ('build/aes-gear-field','build/aes-gear-field-fast'), strict=True):
        probes.append(np.loadtxt(Path(source) / f'probe-{seed}.txt'))
        reference.append(np.loadtxt(Path(source) / f'truth-{seed}.csv', delimiter=',', skiprows=1, usecols=0))
    order = np.random.default_rng(825).permutation(len(probes[0]))
    shuffled = probes[0][order]
    ids = np.r_[np.flatnonzero(order < 768)[:48], np.flatnonzero(order >= 768)[:48]]
    start = perf_counter()
    field = FittedFaces(MODEL/'guide.npz', MODEL/'faces.npz')
    if field.refine is None:
        raise RuntimeError('Numba is required for this benchmark')
    fitted_setup = perf_counter()-start
    start = perf_counter()
    field._init_reference_trees()
    libigl_extra_setup = perf_counter()-start

    def full_search(points):
        _, nearest, seed = field.tree.squared_distance(field.xyz, field.tri, points)
        crease = field.edge_tree.squared_distance(field.xyz, field.edges, points)
        return refine(points,nearest,seed,*crease,field.xyz,field.tri,field.owner,field.uv,
                      field.incident,field.edge_pairs,field.convex_sign,field.bary_grad,*field.packed,
                      field.crease_band_mm)

    start = perf_counter()
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.core.Tensor(field.xyz.astype(np.float32)),
                        o3d.core.Tensor(field.tri.astype(np.uint32)))
    mesh_setup = perf_counter()-start

    def mesh(p):
        return scene.compute_signed_distance(o3d.core.Tensor(p.astype(np.float32)), nthreads=1, nsamples=3).numpy()

    grid_bytes = sum(a.nbytes for grid in (field.tri_grid,field.edge_grid) for a in grid
                     if isinstance(a,np.ndarray))
    report = {'processor': os.environ.get('PROCESSOR_IDENTIFIER'), 'threads': 1,
              'open3d_version': o3d.__version__, 'frames': len(frames),
              'points_per_frame': list(map(len, frames)), 'grid_MiB': grid_bytes/2**20,
              'ablation_extra_libigl_setup_s': libigl_extra_setup, 'repeats': repeats, 'methods': {}}
    methods = [('fitted_faces_ours', field.predict, fitted_setup),
               ('same_projection_full_libigl_search', full_search, None),
               ('triangle_bvh_open3d', mesh, mesh_setup)]
    for name, predict, setup in methods:
        start = perf_counter()
        predict(frames[0])
        first = perf_counter()-start
        values = predict(xyz)
        by_seed = {}
        for seed, p, target in zip(seeds,probes,reference,strict=True):
            distance = predict(p)
            by_seed[seed] = {**accuracy(distance,target),
                             'wrong_sign': int(np.count_nonzero(np.sign(distance)!=np.sign(target)))}
        loads = []
        for t in np.unique(times):
            for pair in np.unique(pairs[times==t]):
                idx = (times==t) & (pairs==pair)
                def load(gaps):
                    return 1e6 * weights[idx] @ np.maximum(.055-gaps[idx], 0.)
                loads.append(abs(load(values)-load(truth)))
        report['methods'][name] = {'setup_s': setup, 'first_query_s': first,
                                   'random_probes': by_seed, 'moving_distances': accuracy(values,truth),
                                   'moving_line_force_max_error_N': float(max(loads))}
        if name == 'fitted_faces_ours' and (any(s['within_0p125_um'] != len(p) or s['wrong_sign']
               for s,p in zip(by_seed.values(),probes,strict=True)) or np.max(abs(values-truth)) > .000125):
            raise AssertionError('Fitted distances failed the CAD accuracy check')
    fixed = xyz[np.isclose(times,.00022,atol=1e-12,rtol=0.)]
    if len(fixed)!=4:
        raise ValueError('Missing the archived four-point pose at 0.22 ms')
    for name, values in timed(methods,frames,fixed,shuffled,ids,repeats).items():
        report['methods'][name].update(values)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action',required=True)
    prep = subs.add_parser('prepare')
    prep.add_argument('executable', type=Path)
    prep.add_argument('poses', type=Path)
    prep.add_argument('out', type=Path)
    bench = subs.add_parser('time')
    bench.add_argument('report', type=Path)
    bench.add_argument('--repeats', type=int, default=25)
    check = subs.add_parser('verify')
    check.add_argument('executable', type=Path)
    check.add_argument('scratch', type=Path)
    check.add_argument('report', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.executable.resolve(), args.poses, args.out)
    else:
        with threadpool_limits(1):
            if args.action == 'verify':
                from gear_field_timing import verify_acceleration
                args.scratch.mkdir(parents=True, exist_ok=True)
                result = verify_acceleration(args.executable.resolve(), args.scratch)
            else:
                if args.repeats < 1:
                    parser.error('--repeats must be positive')
                result = benchmark(args.repeats)
        args.report.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(result, indent=2))
