"""Audit a CUDA broad phase against the cold OCCT output of cad_region_replay.py.

Inputs: the same two STEP solids, pose table (translation in m) and cold CSV
prefix as the CPU replay. Times cover one pose on one device. The candidate
mask stays on the GPU until the validation pass; it is not a contact result.

After generating the cold replay as described in results/contact_validation/README.md:
    python benchmarks/aes/gpu_face_candidates.py results/gears/models/spur-1a.step results/gears/models/spur-1b.step build/aes-region-replay/spur-poses.txt build/aes-region-replay/cold results/aes/gpu-face-candidates/spur.json --cutoff-mm 0.055
For the cam use the follower and cam from the same replay, its cam/cam-poses.txt
and cam/cold prefix, with --cutoff-mm 12. Requires OCP, SciPy and CUDA CuPy.
"""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import cupy as cp
import numpy as np
from scipy.spatial.transform import Rotation


KERNEL = r'''
extern "C" __global__ void candidates(const double* a, int na, const double* b,
    const double* pose, double limit, unsigned char* mask) {
    const int j = blockIdx.x;
    __shared__ double center[3], radius[3];
    if (threadIdx.x < 3) {
        const int d = threadIdx.x;
        double c = pose[4*d + 3], r = 0.0;
        for (int k = 0; k < 3; ++k) {
            const double rot = pose[4*d + k];
            c += rot * b[6*j + k];
            r += fabs(rot) * b[6*j + 3 + k];
        }
        center[d] = c;
        radius[d] = r;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < na; i += blockDim.x) {
        double sq = 0.0;
        for (int d = 0; d < 3; ++d) {
            const double gap = fmax(0.0, fabs(center[d] - a[6*i+d])
                                        - radius[d] - a[6*i+3+d]);
            sq += gap * gap;
        }
        mask[j*na+i] = sq <= limit*limit;
    }
}
'''


def face_boxes(path):
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape

    faces = TopTools_IndexedMapOfShape()
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone or not reader.TransferRoots():
        raise ValueError(f'Could not import STEP shape {path}')
    TopExp.MapShapes_s(reader.OneShape(), TopAbs_FACE, faces)
    if not faces.Extent():
        raise ValueError(f'No faces in {path}')
    boxes = []
    for i in range(1, faces.Extent() + 1):
        box = Bnd_Box()
        BRepBndLib.AddOptimal_s(faces.FindKey(i), box, False, True)
        if box.IsVoid() or box.IsOpen():
            raise ValueError(f'Unbounded or empty face {i} in {path}')
        boxes.append(box.Get())
    return np.array(boxes, dtype=np.float64).reshape(-1, 2, 3)


def cpu_mask(a, b, pose, cutoff):
    # Independent implementation: transform all eight box corners on CPU.
    corners = np.array(list(np.ndindex(2, 2, 2)))
    xyz = np.where(corners, b[:, None, 1], b[:, None, 0])
    moved = xyz @ pose[:, :3].T + pose[:, 3]
    gap = np.maximum(a[:, 0] - moved.max(axis=1)[:, None], moved.min(axis=1)[:, None] - a[:, 1])
    return np.linalg.norm(np.maximum(gap, 0), axis=2) <= cutoff


def read_reference(prefix, times):
    with Path(str(prefix) + '.stats.csv').open(newline='') as stream:
        stats = list(csv.DictReader(stream))
    if [float(row['time_s']) for row in stats] != list(times) or any(
        int(row[k]) for row in stats for k in ('unresolved_face_pairs', 'unresolved_line_pairs')
    ):
        raise ValueError('CPU reference poses differ or contain unresolved queries')
    pairs = {t: set() for t in times}
    with Path(str(prefix) + '.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            pairs[float(row['time_s'])].add((int(row['face_a']), int(row['face_b'])))
    return pairs


def resident_screen_us(launch, poses, repeats):
    # Serial replay on one stream. All transforms are known in advance here.
    # This diagnoses device cost only; the online timer takes a new host pose.
    inputs = [cp.asarray(pose) for pose in poses]
    cp.cuda.get_current_stream().synchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        for pose in inputs:
            launch(pose)
        graph = stream.end_capture()
        begin, end = cp.cuda.Event(), cp.cuda.Event()
        graph.launch(stream)
        stream.synchronize()
        times = []
        for _ in range(repeats):
            begin.record()
            graph.launch(stream)
            end.record()
            end.synchronize()
            times.append(1000*cp.cuda.get_elapsed_time(begin, end)/len(inputs))
    return float(np.median(times))


def benchmark(args):
    table = np.loadtxt(args.poses, ndmin=2)
    if table.shape[1] != 8 or not np.isfinite(table).all() or len(np.unique(table[:, 0])) != len(table):
        raise ValueError('Expected distinct times and finite rigid poses')
    axes = table[:, 4:7]
    lengths = np.linalg.norm(axes, axis=1, keepdims=True)
    if np.any(lengths == 0):
        raise ValueError('Rotation axes must have positive length')
    rotations = Rotation.from_rotvec(axes / lengths * table[:, 7:8]).as_matrix()
    poses = np.ascontiguousarray(np.concatenate((rotations, table[:, 1:4, None]*1000), axis=2))
    reference = read_reference(args.reference_prefix, table[:, 0])
    start = perf_counter()
    a, b = face_boxes(args.step_a), face_boxes(args.step_b)

    def upload(box):
        middle = (box[:, 1] + box[:, 0]) * .5
        half = (box[:, 1] - box[:, 0]) * .5
        return cp.asarray(np.column_stack((middle, half)))

    ga, gb = upload(a), upload(b)
    kernel = cp.RawKernel(KERNEL, 'candidates')
    transform = cp.empty((3, 4), dtype=cp.float64)
    mask = cp.empty((len(b), len(a)), dtype=cp.uint8)
    scale = max(1., np.abs(a).max(), np.abs(b).max(), np.abs(poses[:, :, 3]).max(), args.cutoff_mm)
    limit = np.float64(args.cutoff_mm + 64*np.finfo(float).eps*scale)

    def launch(matrix):
        kernel((len(b),), (128,), (ga, np.int32(len(a)), gb, matrix, limit, mask))

    def query(pose):
        transform.set(pose)
        launch(transform)

    query(poses[0])
    stream = cp.cuda.get_current_stream()
    stream.synchronize()
    setup = perf_counter() - start
    rows = []
    for row, pose in zip(table, poses, strict=True):
        query(pose)
        result = mask.get()
        if not np.array_equal(result, cpu_mask(a, b, pose, limit)):
            raise ValueError(f'GPU and CPU box masks differ at {row[0]}')
        pairs = reference[row[0]]
        if any(not (1 <= i <= len(a) and 1 <= j <= len(b) and result[j-1, i-1]) for i, j in pairs):
            raise ValueError(f'GPU broad phase dropped a CPU CAD face pair at {row[0]}')
        rows.append(dict(time_s=row[0], candidates=int(result.sum()), occt_pairs=len(pairs)))
    latencies = []
    for _ in range(args.repeats):
        for pose in poses:
            start = perf_counter()
            query(pose)
            stream.synchronize()
            latencies.append(1e6*(perf_counter()-start))
    sources = [args.step_a, args.step_b, args.poses, Path(str(args.reference_prefix) + '.csv'),
               Path(str(args.reference_prefix) + '.stats.csv')]
    return dict(gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(), cupy=cp.__version__,
                input_sha256={p.name: sha256(p.read_bytes()).hexdigest() for p in sources},
                face_counts=[len(a), len(b)], cutoff_mm=args.cutoff_mm, repeats=args.repeats,
                stage='face box screening; pose upload, kernel and host synchronization; mask stays on device',
                setup_and_first_query_s=setup, median_us=float(np.median(latencies)),
                p99_us=float(np.percentile(latencies, 99)), max_us=max(latencies),
                resident_graph_median_us_per_known_pose=resident_screen_us(launch, poses, args.repeats),
                poses=rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('step_a', 'step_b', 'poses', 'reference_prefix', 'output'):
        p.add_argument(name, type=Path)
    p.add_argument('--cutoff-mm', type=float, required=True)
    p.add_argument('--repeats', type=int, default=100)
    args = p.parse_args()
    if args.repeats < 1 or not np.isfinite(args.cutoff_mm) or args.cutoff_mm < 0:
        p.error('Use positive repeats and a finite nonnegative cutoff')
    report = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
