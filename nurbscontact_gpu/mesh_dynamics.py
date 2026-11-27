"""Current-pose triangle BVH contact coupled to the shared resident implicit solver."""

from pathlib import Path
import hashlib
import shlex
from time import perf_counter
import cupy as cp
import meshio
import numpy as np
from .bvh import TriangleMesh, IntersectionBatch
from .mesh_topology import support_topology

COORD = np.dtype(
    [(name, np.float64) for name in ("mass", "q", "v", "k", "c", "rest", "load")]
    + [("first", np.int32), ("end", np.int32)],
    align=True,
)
BODY = np.dtype(
    [(name, np.float64, (3,)) for name in ("origin", "axis", "pivot")]
    + [
        ("angle", np.float64),
        ("hinge", np.int32),
        ("translation", np.int32),
        ("direction", np.float64, (3,)),
    ],
    align=True,
)
LAW = np.dtype(
    [("a", np.int32), ("b", np.int32)]
    + [(name, np.float64) for name in ("k", "exponent", "rate", "max_depth")]
    + [("shared", np.bool_)],
    align=True,
)


def read_scene(path):
    path = Path(path)
    lines = iter(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if next(lines) not in ("nc-contact-v1", "nc-contact-v2"):
        raise ValueError("Unknown scene")
    head = next(lines).split()
    if len(head) != 7 or head[0] != "gravity" or any(float(x) != 0 for x in head[1:4]):
        raise ValueError("This driver requires zero gravity")
    n, nb, npair = map(int, head[4:])
    if not 1 <= n <= 16 or not 2 <= nb <= 1024 or not 1 <= npair <= 16384:
        raise ValueError("Invalid scene counts; expected one to sixteen coordinates")
    coordinates = np.zeros(n, COORD)
    for i in range(n):
        row = next(lines).split()
        if row[0] != "coordinate":
            raise ValueError(row)
        for key, value in zip(COORD.names[:7], row[1:], strict=True):
            coordinates[i][key] = float(value)
        if (
            not all(np.isfinite(coordinates[i][k]) for k in COORD.names[:7])
            or coordinates[i]["mass"] <= 0
            or coordinates[i]["k"] < 0
            or coordinates[i]["c"] < 0
        ):
            raise ValueError("Invalid coordinate inertia or spring")
    bodies = np.zeros(nb, BODY)
    files = []
    for i in range(nb):
        row = shlex.split(next(lines))
        if not row or row[0] != "body" or len(row) not in (18, 22):
            raise ValueError("Invalid body record")
        if not np.isfinite(np.asarray(row[2:], float)).all():
            raise ValueError("Nonfinite body data")
        if float(row[2]) != 0:
            raise ValueError(
                "Body mass must be represented by coordinate inertia in this driver"
            )
        files.append((path.parent / row[1]).with_suffix(".msh").resolve())
        bodies[i]["origin"] = row[3:6]
        bodies[i]["axis"] = row[6:9]
        bodies[i]["angle"] = float(row[9])
        bodies[i]["hinge"] = int(row[13])
        bodies[i]["pivot"] = row[14:17]
        bodies[i]["translation"] = -1
        length = np.linalg.norm(bodies[i]["axis"])
        if not length > 0:
            raise ValueError("Zero rotation axis")
        bodies[i]["axis"] /= length
        if not -1 <= bodies[i]["hinge"] < n:
            raise ValueError("Invalid hinge coordinate")
        if int(row[17]) == 1:
            if len(row) != 22:
                raise ValueError("Missing translation coordinate")
            bodies[i]["translation"] = int(row[18])
            bodies[i]["direction"] = row[19:22]
            if (
                not 0 <= bodies[i]["translation"] < n
                or not np.linalg.norm(bodies[i]["direction"]) > 0
            ):
                raise ValueError("Invalid translation coordinate or direction")
        elif int(row[17]) != 0:
            raise ValueError("At most one prismatic coordinate per body")
        elif len(row) != 18:
            raise ValueError("Unexpected body data")
    pairs = np.zeros(npair, LAW)
    for i in range(npair):
        row = next(lines).split()
        if len(row) != 9 or row[0] != "surface" or float(row[6]) != 0:
            raise ValueError("Expected frictionless surface contact")
        if not np.isfinite(np.asarray(row[1:], float)).all():
            raise ValueError("Nonfinite contact data")
        a, b = int(row[1]), int(row[2])
        if not (0 <= a < nb and 0 <= b < nb and a != b):
            raise ValueError("Invalid contact body pair")
        if not (
            float(row[3]) > 0
            and float(row[4]) >= 1
            and float(row[5]) >= 0
            and float(row[7]) > 0
            and float(row[8]) > 0
        ):
            raise ValueError("Invalid penalty material")
        pairs[i] = (
            int(row[1]),
            int(row[2]),
            float(row[3]),
            float(row[4]),
            float(row[5]),
            float(row[8]),
            True,
        )
    rows = []
    remaining = list(lines)
    if remaining:
        lines = iter(remaining)
        if next(lines) != "drives 0":
            raise ValueError("Prescribed drives unsupported by this driver")
        count = next(lines).split()
        if count[0] != "loads":
            raise ValueError(count)
        tables = {}
        if not 0 <= int(count[1]) <= n:
            raise ValueError("Invalid load table count")
        for _ in range(int(count[1])):
            kind, i, size = next(lines).split()
            i, size = int(i), int(size)
            if (
                kind != "load"
                or not 0 <= i < n
                or i in tables
                or not 2 <= size <= 10000000
            ):
                raise ValueError("Invalid load table")
            values = np.asarray(
                [list(map(float, next(lines).split())) for _ in range(size)]
            )
            if (
                values.shape != (size, 2)
                or not np.isfinite(values).all()
                or np.any(np.diff(values[:, 0]) <= 0)
            ):
                raise ValueError(
                    "Load times must increase strictly and all samples must be finite"
                )
            tables[i] = values.tolist()
        if list(lines):
            raise ValueError("Unrecognized trailing scene data")
        for i in range(n):
            coordinates[i]["first"] = len(rows)
            rows.extend(tables.get(i, []))
            coordinates[i]["end"] = len(rows)
    return (
        files,
        coordinates,
        bodies,
        pairs,
        np.asarray(rows, np.float64).reshape(-1, 2),
    )


def module(coordinates=16, parallel_contour=True, register_limit=None):
    folder = Path(__file__).parent / "kernels"
    detector = (
        (folder / "gpu_triangle_intersections.cu")
        .read_text()
        .replace("#include <cuda_runtime.h>", "")
    )
    detector = detector.replace('extern "C" __global__', "__device__")
    step_file = folder / "cuda_resident_step.cuh"
    if not step_file.exists():
        step_file = folder.parent.parent / "src/cuda_resident_step.cuh"
    step = step_file.read_text()
    step = step.replace(
        "std::is_same_v<Grid, ScalarGrid>", "mesh_same<Grid, ScalarGrid>"
    )
    code = (
        """#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_device_runtime_api.h>
#define INFINITY __longlong_as_double(0x7ff0000000000000ULL)
#define NAN __longlong_as_double(0x7ff8000000000000ULL)
template<class T,class U> constexpr bool mesh_same=false;
template<class T> constexpr bool mesh_same<T,T> = true;
#define NC_MESH_RESIDENT
#define NC_DEVICE_GEOMETRY_SCALE 1.
#define NC_DEVICE_QUERY_REACH 4.
namespace mesh_detect {\n"""
        + detector
        + "\n}\n"
        + (folder / "mesh_resident.cuh").read_text()
        + step
        + (folder / "mesh_resident_bind.cuh").read_text()
        + (folder / "mesh_graph.cuh").read_text()
    )
    options = (
        "--std=c++17",
        "--fmad=false",
        f"-DNC_DEVICE_COORDINATE_LIMIT={coordinates}",
    )
    if not parallel_contour:
        options += ("-DNC_SERIAL_MESH_CONTOUR",)
    if register_limit is not None:
        options += (f"--maxrregcount={register_limit}",)
    return cp.RawModule(code=code, options=options, enable_cooperative_groups=True)


class MeshDynamics:
    def __init__(
        self,
        scene,
        steps,
        *,
        capacity=4096,
        blocks=8,
        cache_dir=None,
        parallel_contour=True,
        register_limit=None,
    ):
        start = perf_counter()
        if capacity < 1 or capacity & (capacity - 1) or not 1 <= blocks <= capacity:
            raise ValueError(
                "Capacity must be a power of two and at least the positive block count"
            )
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("Expected a positive integer number of steps")
        if register_limit is not None and register_limit not in (64, 96, 128, 192, 255):
            raise ValueError("Unsupported register bound")
        self.files, self.coordinates, self.bodies, self.laws, self.loads = read_scene(
            Path(scene)
        )
        self.n, self.nb, self.np = (
            len(self.coordinates),
            len(self.bodies),
            len(self.laws),
        )
        self.blocks = blocks
        self.capacity = capacity
        self.keep = []
        self.steps = steps
        self.module = module(self.n, parallel_contour, register_limit)
        sizes = cp.empty(10, cp.int32)
        self.module.get_function("mesh_layout")((1,), (1,), (sizes,))
        sizes = sizes.get().tolist()
        if sizes[7:] != [COORD.itemsize, BODY.itemsize, LAW.itemsize]:
            raise ValueError(("Scene ABI", sizes))
        self.views = cp.zeros((self.nb, sizes[0]), cp.uint8)
        self.meshes = []
        vertex_capacity = 0
        face_capacity = 0
        for index, file in enumerate(self.files):
            data = meshio.read(file)
            faces = data.get_cells_type("triangle")
            vertex_capacity = max(vertex_capacity, len(data.points))
            face_capacity = max(face_capacity, len(faces))
            mesh = TriangleMesh(data.points, faces, leaf_size=4)
            self.meshes.append(mesh)
            cached = None
            if cache_dir is not None:
                folder = Path(cache_dir)
                folder.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256(
                    file.read_bytes()
                    + Path(__file__).with_name("mesh_topology.py").read_bytes()
                ).hexdigest()
                cached = folder / (digest + ".npz")
            if cached is not None and cached.exists():
                with np.load(cached, allow_pickle=False) as arrays:
                    topo = {k: arrays[k] for k in arrays.files}
            else:
                topo = support_topology(data.points, faces)
                if cached is not None:
                    np.savez(cached, **topo)
            keys = (
                "vertices",
                "faces",
                "normals",
                "edges",
                "face_edges",
                "panels",
                "panel_offsets",
                "panel_faces",
                "ray_offsets",
                "rays",
                "ridges",
                "edge_ridge",
                "ridge_offsets",
                "ridge_ids",
            )
            arrays = [cp.asarray(np.ascontiguousarray(topo[k])) for k in keys]
            self.keep.extend(arrays)
            support = [
                cp.asarray(np.ascontiguousarray(topo[k]))
                for k in (
                    "vertex_normals",
                    "neighbor_ids",
                    "components",
                    "component_seeds",
                )
            ]
            self.keep.extend(support)
            self.module.get_function("bind_mesh_body")(
                (1,),
                (1,),
                (
                    self.views[index],
                    *arrays,
                    mesh.triangles,
                    mesh.bounds,
                    mesh.links,
                    mesh.ids,
                    *support,
                    np.float64(topo["radius"]),
                    np.int32(len(data.points)),
                    np.int32(len(faces)),
                    np.int32(len(topo["edges"])),
                    np.int32(len(topo["ridges"])),
                    np.int32(topo["convex"]),
                    np.int32(len(topo["component_seeds"])),
                ),
            )
        # Traverse in the same reference frame used for contact construction.
        self.query = IntersectionBatch(
            [(self.meshes[int(l["b"])], self.meshes[int(l["a"])]) for l in self.laws],
            capacity=capacity,
        )
        self.scene = cp.zeros(sizes[2], cp.uint8)
        self.workspace = cp.zeros(sizes[1], cp.uint8)
        self.control = cp.zeros(sizes[3], cp.uint8)
        self.cache = cp.zeros((self.np, sizes[4]), cp.uint8)
        self.output = cp.zeros((steps + 1, 1 + 4 * self.n + 3 * self.np))
        self.finished = cp.zeros(steps + 1, cp.uint64)
        self.observed = cp.zeros(3 * self.np)
        patch_capacity = 1 << (7 * capacity - 1).bit_length()
        feature_capacity = 1 << (15 * patch_capacity - 1).bit_length()
        support_blocks = min(blocks, 8)
        buffers = [
            cp.empty(capacity, cp.int32),
            cp.empty(capacity, cp.int32),
            cp.empty((support_blocks, 2 * patch_capacity), cp.uint64),
            cp.empty((support_blocks, 2 * feature_capacity), cp.uint64),
            cp.empty((support_blocks, patch_capacity), cp.uint64),
            cp.empty((support_blocks, feature_capacity), cp.uint64),
        ]
        regions = [
            cp.empty((capacity, sizes[6]), cp.uint8),
            cp.empty((capacity, sizes[6]), cp.uint8),
            cp.empty((self.np * capacity, sizes[5]), cp.uint8),
        ]
        inputs = [
            cp.asarray(x)
            for x in (
                self.coordinates.view(np.uint8),
                self.bodies.view(np.uint8),
                self.laws.view(np.uint8),
                self.loads,
            )
        ]
        self.keep.extend(buffers + regions + inputs)
        vertices = [
            cp.empty(vertex_capacity, cp.int32),
            cp.empty(vertex_capacity, cp.int32),
        ]
        self.keep.extend(vertices)
        contributions = cp.empty((support_blocks * capacity, sizes[6]), cp.uint8)
        self.keep.append(contributions)
        placed = [cp.empty((vertex_capacity, 3)), cp.empty((face_capacity, 3))]
        self.keep.extend(placed)
        self.module.get_function("bind_mesh_scene")(
            (1,),
            (1,),
            (
                self.scene,
                *inputs,
                self.views,
                np.int32(self.n),
                np.int32(self.nb),
                np.int32(self.np),
                self.cache,
                self.workspace,
                self.control,
                self.observed,
                self.output,
                self.finished,
                self.query.views,
                *buffers,
                np.int32(capacity),
                np.int32(patch_capacity),
                np.int32(feature_capacity),
                *regions,
                *vertices,
                np.int32(vertex_capacity),
                contributions,
                *placed,
            ),
        )
        cp.cuda.get_current_stream().synchronize()
        self.setup_s = perf_counter() - start

    def status(self):
        out = cp.empty(10)
        self.module.get_function("mesh_status")((1,), (1,), (self.scene, out))
        return out.get()

    def probe(self, poses):
        result = cp.empty((self.np, self.capacity, 7))
        counts = cp.empty(self.np, cp.int32)
        self.module.get_function("probe_mesh")(
            (self.blocks,), (128,), (self.scene, cp.asarray(poses), result, counts)
        )
        status = self.status()
        if status[0]:
            raise RuntimeError(("mesh contact status", status.tolist()))
        return [r[:n] for r, n in zip(result.get(), counts.get())]

    def probe_world(self, placements):
        result = cp.empty((self.np, self.capacity, 7))
        counts = cp.empty(self.np, cp.int32)
        self.module.get_function("probe_mesh_world")(
            (self.blocks,),
            (128,),
            (self.scene, cp.asarray(placements, dtype=cp.float64), result, counts),
        )
        status = self.status()
        if status[0]:
            raise RuntimeError(("mesh contact status", status.tolist()))
        return [r[:n] for r, n in zip(result.get(), counts.get())]

    def run(self, step, *, execution="cooperative"):
        if not np.isfinite(step) or step <= 0:
            raise ValueError("Expected a positive finite time step")
        if execution == "graph":
            from .mesh_graph import run

            return run(self, step)
        if execution != "cooperative":
            raise ValueError("Unknown GPU execution mode")
        advance = self.module.get_function("advance_mesh")
        self.module.get_function("reset_mesh")((1,), (1,), (self.scene,))
        cp.cuda.get_current_stream().synchronize()
        start = perf_counter()
        for first in range(0, self.steps, 128):
            advance(
                (self.blocks,),
                (128,),
                (
                    self.scene,
                    np.int32(first),
                    np.int32(min(self.steps, first + 128)),
                    np.float64(step),
                ),
            )
        cp.cuda.get_current_stream().synchronize()
        online = perf_counter() - start
        status = self.status()
        output = self.output.get()
        stamps = self.finished.get()
        if status[0] or abs(status[1] - self.steps * step) > 1e-8:
            raise RuntimeError(("Incomplete mesh dynamics", status.tolist()))
        return output, dict(
            online_s=online,
            setup_s=self.setup_s,
            status=status.tolist(),
            p99_step_s=float(np.percentile(np.diff(stamps) * 1e-9, 99)),
            kernel=advance.attributes,
        )
