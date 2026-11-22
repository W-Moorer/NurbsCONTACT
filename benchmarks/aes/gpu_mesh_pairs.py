"""Discrete whole-mesh contact detection on the GPU at archived rigid poses."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import cupy as cp
import numpy as np
from scipy.spatial.transform import Rotation

from gpu_mesh_line import TRIANGLE_CUDA, device_bvhs, read_full_mesh


class MeshPair:
    def __init__(self, a, b):
        _, (self.nodes, self.boxes, self.a) = device_bvhs({0: a})
        self.tri_boxes = cp.concatenate((self.a.min(axis=1), self.a.max(axis=1)), axis=1)
        self.b = cp.asarray(b)
        self.scale = max(1., np.abs(a).max(), np.abs(b).max())
        self.distances = cp.empty(len(b), dtype=cp.float64)
        self.near = cp.empty(len(b), dtype=cp.int32)
        self.touch = cp.empty_like(self.near)
        source = Path(__file__).with_name('mesh_pair.cu').read_text()
        self.kernel = cp.RawKernel(TRIANGLE_CUDA + source, 'near_pairs')

    def query(self, pose, cutoff_mm, tolerance_mm, translation_scale, overlaps_only=False):
        # Pad only the broad phase. Triangle-pair distances decide membership.
        padding = 128 * np.finfo(float).eps * (self.scale + translation_scale)
        box_radius = tolerance_mm if overlaps_only else cutoff_mm
        for start in range(0, len(self.b), 8192):
            stop = min(len(self.b), start+8192)
            self.kernel(((stop-start+127)//128,), (128,),
                        (self.nodes, self.boxes, self.tri_boxes, self.a, self.b,
                         np.int32(start), np.int32(stop), pose, np.float64(cutoff_mm**2),
                         np.float64((box_radius+padding)**2), np.float64(tolerance_mm**2),
                         np.int32(overlaps_only), self.distances, self.near, self.touch))
        if overlaps_only:
            return self.touch.sum()
        return cp.stack((cp.sqrt(self.distances.min()), self.near.sum(), self.touch.sum()))


def check_kernel():
    # Crossing, coplanar overlap, shared edge, parallel clearance, edge-to-edge
    # clearance and distant triangles exercise both detection modes.
    a = np.array([[[0,0,0],[2,0,0],[0,2,0]]], dtype=float)
    identity = cp.asarray(np.eye(4)[:3].copy())
    cases = [(np.array([[.3,.3,-1],[.3,.3,1],[.7,.3,1]]), 0),
             (a[0]+[.1,.1,0], 0),
             (np.array([[0,0,0],[2,0,0],[1,-1,0]]), 0),
             (a[0]+[0,0,.04], .04),
             (np.array([[.7,-.4,.03],[1.3,.4,.03],[.7,-.4,1]]), .03),
             (np.array([[1.5,1.5,0],[2.,1.5,0],[1.5,2.,0]]), np.inf),
             (a[0]+[8,8,8], np.inf)]
    for b, gap in cases:
        pair = MeshPair(a, b.astype(float)[None])
        result = pair.query(identity, .055, 1e-10, 0).get()
        np.testing.assert_allclose(result, [gap, int(np.isfinite(gap)), int(gap==0)], atol=1e-10)
        assert pair.query(identity, .055, 1e-10, 0, overlaps_only=True).get() == int(gap==0)


def run(args):
    check_kernel()
    a, b = read_full_mesh(args.mesh_a), read_full_mesh(args.mesh_b)
    pair = MeshPair(a, b)
    ref = json.loads(args.cpu_reference.read_text())
    hashes = {p.name: sha256(p.read_bytes()).hexdigest() for p in (args.mesh_a,args.mesh_b,args.poses)}
    if any(ref['input_sha256'].get(k) != v for k,v in hashes.items()):
        raise ValueError('CPU and GPU must use the same meshes and pose input.')
    cpu = {row['time_s']: row for row in ref['poses']}
    poses = []
    for t, x, y, z, *axis, angle in np.loadtxt(args.poses, ndmin=2):
        R = Rotation.from_rotvec(np.asarray(axis)*angle/np.linalg.norm(axis)).as_matrix()
        translation = 1000*np.array([x,y,z])
        poses.append((t, cp.asarray(np.column_stack((R,translation))), np.abs(translation).max()))
    if set(cpu) != {t for t, _, _ in poses}:
        raise ValueError('CPU reference pose times differ.')
    cutoff_mm, touch_mm = args.cutoff_um/1000, 1e-10
    rows = [dict(time_s=t, query_ms=[], overlap_only_ms=[]) for t, _, _ in poses]
    begin, end = cp.cuda.Event(), cp.cuda.Event()
    for repeat in range(args.repeats+1):
        for row, (t, pose, scale) in zip(rows, poses, strict=True):
            for overlaps_only in ((False, True) if repeat%2 else (True, False)):
                begin.record()
                summary = pair.query(pose, cutoff_mm, touch_mm, scale, overlaps_only)
                end.record()
                end.synchronize()
                if repeat:
                    key = 'overlap_only_ms' if overlaps_only else 'query_ms'
                    row[key].append(cp.cuda.get_elapsed_time(begin, end))
                if overlaps_only:
                    overlaps = int(summary.get())
                else:
                    gap_mm, n, intersects = summary.get()
            if overlaps != intersects:
                raise ValueError(f'Intersection and distance tests disagree at {t}.')
            if not np.isclose(1000*gap_mm, cpu[t]['mesh_min_gap_um'], rtol=0, atol=1e-6):
                raise ValueError(f'GPU and independent FCL closest-pair gaps differ at {t}.')
            row.update(mesh_min_gap_um=1000*gap_mm, near_triangle_pairs=int(n),
                       touching_triangle_pairs=int(intersects))
    times = [x for r in rows for x in r['query_ms']]
    overlaps = [x for r in rows for x in r['overlap_only_ms']]
    return dict(gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                input_sha256=hashes, triangles=[len(a),len(b)], cutoff_um=args.cutoff_um,
                numerical_touch_tolerance_mm=touch_mm, repeats=args.repeats,
                median_ms=float(np.median(times)), min_ms=min(times), max_ms=max(times),
                overlap_only_median_ms=float(np.median(overlaps)), poses=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('mesh_a', 'mesh_b', 'poses', 'cpu_reference', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--cutoff-um', type=float, default=55.)
    args = parser.parse_args()
    if args.repeats<1 or not np.isfinite(args.cutoff_um) or args.cutoff_um<=0:
        parser.error('Expected positive repeats and query radius.')
    args.output.write_text(json.dumps(run(args), indent=2, allow_nan=False)+'\n', encoding='utf-8')
