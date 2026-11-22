"""Device-resident sampled contact on closed, face-tagged rigid meshes."""

from pathlib import Path

import cupy as cp
import numpy as np


def surface_samples(triangles, faces, per_edge=1, crease_samples=False):
    # Interior lattice probes and shared-crease probes are disjoint.
    if per_edge < 1:
        raise ValueError("Expected at least one point per triangle")
    norms = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    if np.einsum("ij,ij->", triangles[:, 0] - triangles[0, 0], norms) <= 0:
        raise ValueError("Expected outward-oriented source triangles")
    norms /= np.linalg.norm(norms, axis=1)[:, None]
    points = []
    for i in range(per_edge):
        for j in range(per_edge - i):
            u, v = (i + 1 / 3) / per_edge, (j + 1 / 3) / per_edge
            points.append(
                triangles[:, 0]
                + u * (triangles[:, 1] - triangles[:, 0])
                + v * (triangles[:, 2] - triangles[:, 0])
            )
            if i + j + 1 < per_edge:
                u, v = (i + 2 / 3) / per_edge, (j + 2 / 3) / per_edge
                points.append(
                    triangles[:, 0]
                    + u * (triangles[:, 1] - triangles[:, 0])
                    + v * (triangles[:, 2] - triangles[:, 0])
                )
    tags, compact = np.unique(faces, return_inverse=True)
    p = np.concatenate(points)
    n = np.tile(norms, (len(points), 1))
    f = np.tile(compact.astype(np.int32), len(points))
    other = n.copy()
    alt = f.copy()
    if crease_samples:
        coords, indices = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
        ii = indices.reshape(-1, 3)
        directed = np.stack((ii, np.roll(ii, -1, axis=1)), axis=-1).reshape(-1, 2)
        ij = np.sort(directed, axis=1)
        edges, owner, cnt = np.unique(ij, axis=0, return_inverse=True, return_counts=True)
        if np.any(cnt != 2) or np.any(
            np.bincount(owner, weights=np.where(directed[:, 0] < directed[:, 1], 1.0, -1.0))
        ):
            raise ValueError("Source crease sampling requires a closed, consistently oriented surface")
        inc = np.argsort(owner, kind="stable").reshape(-1, 2) // 3
        sharp = compact[inc[:, 0]] != compact[inc[:, 1]]
        # A shared crease supplies one probe. Its two incident face normals
        # let the force choose the more opposing source face at this pose.
        left, right = inc[sharp].T
        p = np.vstack((p, coords[edges[sharp]].mean(axis=1)))
        n = np.vstack((n, norms[left]))
        other = np.vstack((other, norms[right]))
        f = np.r_[f, compact[left]].astype(np.int32)
        alt = np.r_[alt, compact[right]].astype(np.int32)
    return p, n, f, other, alt, tags


class ResidentMeshContact:
    def __init__(
        self,
        a,
        b,
        fa,
        fb,
        pitch=0.12,
        samples=1,
        lanes=16,
        crease_samples=True,
        backend="grid",
        source_points=None,
        impact_samples=0,
        vertex_probes=False,
        merge_planar=False,
        reverse=None,
        include_vertices=False,
    ):
        from resident_index import compile_mesh

        if backend not in ("grid", "bvh", "sdf") or lanes not in (1, 2, 4, 8, 16, 32):
            raise ValueError("Unknown backend or unsupported warp group size")
        if reverse is not None and (reverse.backend != backend or reverse.reverse is not None):
            raise ValueError("Reverse queries require a single-direction model with the same backend")
        self.backend = backend
        a, b = np.ascontiguousarray(a, dtype=np.float64), np.ascontiguousarray(b, dtype=np.float64)
        for mesh, labels in ((a, fa), (b, fb)):
            if (
                mesh.ndim != 3
                or mesh.shape[1:] != (3, 3)
                or len(mesh) == 0
                or np.shape(labels) != (len(mesh),)
            ):
                raise ValueError("Expected triangles (N,3,3) with one face label per triangle")
            areas = np.linalg.norm(np.cross(mesh[:, 1] - mesh[:, 0], mesh[:, 2] - mesh[:, 0]), axis=1)
            if not np.isfinite(mesh).all() or not np.all(np.isfinite(areas) & (areas > 0)):
                raise ValueError("Triangle coordinates must be finite and triangles nondegenerate")
        if merge_planar:
            from resident_index import planar_regions

            fa = planar_regions(a, fa)[np.unique(fa, return_inverse=True)[1]]
        geom, self.tags_a = compile_mesh(a, fa, pitch)
        p, normals, ids, other, alt, self.tags_b = surface_samples(b, fb, samples, crease_samples)
        if source_points is not None:
            source_points = np.ascontiguousarray(source_points, dtype=np.float64)
            if source_points.shape != p.shape or not np.isfinite(source_points).all():
                raise ValueError("Expected a finite CAD point for each source probe")
            p = source_points
        if vertex_probes or include_vertices:
            if vertex_probes and (source_points is not None or include_vertices):
                raise ValueError("Vertex probes use the supplied triangle vertices")
            xyz, vertices = np.unique(b.reshape(-1, 3), axis=0, return_inverse=True)
            keys, group = np.unique(
                np.column_stack((vertices.ravel(), np.repeat(ids[: len(b)], 3))), axis=0, return_inverse=True
            )
            # One normal per vertex and incident original face. Preserve sharp edges.
            vertex_normals = np.zeros((len(keys), 3))
            np.add.at(vertex_normals, group, np.repeat(normals[: len(b)], 3, axis=0))
            length = np.linalg.norm(vertex_normals, axis=1)
            if np.any(length <= 1e-14):
                raise ValueError("Ambiguous vertex normal")
            vertex_normals /= length[:, None]
            vertex_ids = keys[:, 1].astype(np.int32)
            samples_at_vertices = (xyz[keys[:, 0]], vertex_normals, vertex_ids, vertex_normals, vertex_ids)
            if include_vertices:
                p, normals, ids, other, alt = (
                    np.concatenate((base, extra))
                    for base, extra in zip((p, normals, ids, other, alt), samples_at_vertices, strict=True)
                )
            else:
                p, normals, ids, other, alt = samples_at_vertices
        if merge_planar:
            self.tags_b, regions = np.unique(planar_regions(b, fb), return_inverse=True)
            ids, alt = regions[ids].astype(np.int32), regions[alt].astype(np.int32)
        self.geometry = tuple(cp.asarray(v) for v in geom)
        self.samples = tuple(cp.asarray(v) for v in (p, normals, ids, other, alt))
        self.n = len(p)
        if int(impact_samples) != impact_samples or impact_samples < 0:
            raise ValueError("Expected a nonnegative number of impact samples")
        self.impact_samples = int(impact_samples)
        self.radius = float(np.linalg.norm(p, axis=1).max())
        # At most one entry per sample, even when many lie on the same face.
        self.reverse = reverse
        total = self.n + (reverse.n if reverse is not None else 0)
        self.capacity = 1 << (2 * total - 1).bit_length()
        self.keys = cp.full(self.capacity, np.uint64(0xFFFFFFFFFFFFFFFF), dtype=cp.uint64)
        self.best = cp.zeros(self.capacity, cp.uint64)
        self.active = cp.empty(total, cp.int32)
        self.count = cp.zeros(1, cp.int32)
        self.witness = cp.empty((total, 8), cp.float64)
        src = Path(__file__).with_name("resident_contact.cu").read_text()
        self.module = cp.RawModule(
            code=src,
            options=("--std=c++17", f"-DQUERY_LANES={1 if backend == 'sdf' else lanes}")
            + (("-DNC_BVH_CONTROL",) if backend == "bvh" else ())
            + (("-DNC_SDF",) if backend == "sdf" else ())
            + (("-DNC_TWO_SIDED",) if reverse is not None else ())
            + ((f"-DNC_IMPACT_SAMPLES={self.impact_samples}",) if self.impact_samples else ()),
            enable_cooperative_groups=True,
        )
        off = self.geometry[3]
        occ = self.geometry[9]
        cells = off.size - 1
        work = cp.flatnonzero((off[1:] > off[:-1]) & occ.ravel()).astype(cp.int32)
        planes = cp.zeros((cells, 4), cp.float32)
        planes[:, 3] = -cp.inf
        ends = off[1:].copy()
        seeds = cp.full(cells, -1, cp.int32)
        build = self.module.get_function("cell_planes")
        if len(work):
            build(
                (128,),
                (128,),
                (
                    work,
                    np.int32(len(work)),
                    self.geometry[0],
                    self.geometry[1],
                    self.geometry[6],
                    self.geometry[5],
                    off,
                    self.geometry[4],
                    ends,
                    planes,
                    seeds,
                ),
            )
        self.geometry += planes, ends, seeds
        tri = self.geometry[0].get()
        e = np.roll(tri, -1, axis=1) - tri
        edge = np.argmax(np.linalg.norm(e, axis=2), axis=1)
        ii = np.arange(len(e))
        u = e[ii, edge]
        length = np.linalg.norm(u, axis=1)
        u /= length[:, None]
        normal = self.geometry[1][:, 0].get()
        v = np.cross(normal, u)
        origin = tri[ii, edge]
        corner = tri[ii, (edge + 2) % 3] - origin
        cu = (corner * u).sum(axis=1)
        cv = (corner * v).sum(axis=1)
        chart2 = np.stack(
            [np.column_stack((di, -(di * origin).sum(axis=1))) for di in (u, v, normal)]
            + [np.column_stack((length, cu, cv, np.zeros_like(cu)))],
            axis=1,
        )
        self.geometry += (cp.asarray(chart2.astype(np.float32)),)
        self.field = ()
        if backend == "sdf":
            from resident_index import sampled_distance_field

            self.field = tuple(cp.asarray(v) for v in sampled_distance_field(a, fa, geom[5], pitch))

        self.outside = cp.zeros((self.n, 4))
        self.nearby = cp.zeros((self.n, 4))
        self.neighbors = cp.empty((self.n, lanes), cp.int32)
        self.fp_pad = 64 * np.finfo(np.float32).eps * max(1.0, np.max(np.abs(a)))
        self.probe = self.module.get_function("probe")
        self.reset = self.module.get_function("clear_active")
        self.reverse_args = ()
        if reverse is not None:
            from hashlib import sha256

            if not np.array_equal(reverse.tags_a, self.tags_b) or not np.array_equal(
                reverse.tags_b, self.tags_a
            ):
                raise ValueError("Forward and reverse region maps differ")
            # A stable geometry order resolves equal depths in the two queries.
            self.prefer_reverse = sha256(a.tobytes()).digest() < sha256(b.tobytes()).digest()
            view = cp.empty(320, cp.uint8)
            reverse.module.get_function("bind_mesh")((1,), (1,), (*reverse.args(), view))
            self.reverse_args = (view, np.int32(self.prefer_reverse))

    def args(self):
        return (
            *self.geometry,
            *self.samples,
            np.int32(self.n),
            self.outside,
            self.nearby,
            self.neighbors,
            np.float64(self.fp_pad),
            self.keys,
            self.best,
            np.uint32(self.capacity - 1),
            self.active,
            self.count,
            self.witness,
            *self.field,
        )

    def reset_history(self):
        self.keys.fill(np.uint64(0xFFFFFFFFFFFFFFFF))
        self.best.fill(0)
        self.count.fill(0)
        self.outside.fill(0)
        self.nearby.fill(0)
        if self.reverse is not None:
            self.reverse.reset_history()

    def evaluate(self, matrix):
        matrix = np.ascontiguousarray(matrix, dtype=np.float64)
        if (
            matrix.shape != (3, 4)
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix[:, :3].T @ matrix[:, :3], np.eye(3), rtol=0, atol=1e-10)
            or np.linalg.det(matrix[:, :3]) < 0
        ):
            raise ValueError("Expected a finite 3-by-4 rigid transform")

        self.reset(
            (min(128, (self.n + 127) // 128),), (128,), (self.keys, self.best, self.active, self.count)
        )
        self.count.fill(0)
        self.probe(
            (min(128, (self.n + 127) // 128),), (128,), (*self.args(), cp.asarray(matrix, dtype=cp.float64))
        )
        count = int(self.count.get()[0])
        slots = self.active[:count]
        keys = self.keys[slots].get()
        wins = self.best[slots].astype(cp.uint32).astype(cp.int64)
        hits = {
            (int(self.tags_a[int(k) >> 32]), int(self.tags_b[int(k) & 0xFFFFFFFF])): w
            for k, w in zip(keys, self.witness[wins].get(), strict=True)
        }
        if self.reverse is not None:
            rotation, offset = matrix[:, :3], matrix[:, 3]
            inverse = np.column_stack((rotation.T, -rotation.T @ offset))
            for (b, a), raw in self.reverse.evaluate(inverse).items():
                key = (a, b)
                if key in hits:
                    delta = raw[0] - hits[key][0]
                    if delta < -1e-12 or (abs(delta) <= 1e-12 and not self.prefer_reverse):
                        continue
                w = raw.copy()
                w[1:4] = rotation @ raw[1:4] + offset
                w[4:7] = -rotation @ raw[4:7]
                hits[key] = w
        return hits
