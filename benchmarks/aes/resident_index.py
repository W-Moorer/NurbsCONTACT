"""Conservative voxel lists and feature pseudonormals for a closed oriented mesh."""

import numpy as np
from numba import njit
from scipy.ndimage import binary_fill_holes


def sampled_distance_field(tri, faces, cells, pitch):
    import igl

    vertices, inverse = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    triangles = inverse.reshape(-1, 3).astype(np.int64)
    _, tags = np.unique(faces, return_inverse=True)
    nx, ny, nz = map(int, cells + 1)
    low = vertices.min(axis=0)
    distance = np.empty((nz, ny, nx), np.float32)
    owner = np.empty((nz, ny, nx), np.int32)
    y, x = np.indices((ny, nx))
    points = np.column_stack((low[0] + x.ravel() * pitch, low[1] + y.ravel() * pitch, np.zeros(x.size)))
    for z in range(nz):
        points[:, 2] = low[2] + z * pitch
        d, idx, _, _ = igl.signed_distance(points, vertices, triangles, igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)
        if not np.isfinite(d).all() or idx.min() < 0:
            raise ValueError("Failed to sample the closed mesh signed distance")
        distance[z] = d.reshape(ny, nx)
        owner[z] = tags[idx].reshape(ny, nx)
    return distance, owner


def planar_regions(tri, faces):
    """Join connected coplanar CAD faces; preserve separate or curved faces."""
    labels, f = np.unique(faces, return_inverse=True)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    first = np.unique(f, return_index=True)[1]
    axes = normals[first]
    deviation = np.zeros(len(labels))
    np.maximum.at(deviation, f, np.linalg.norm(normals - axes[f], axis=1))
    plane = np.einsum("ij,ij->i", axes, tri[first, 0] - tri[0, 0])
    height = np.einsum("ijk,ik->ij", tri - tri[0, 0], axes[f]) - plane[f, None]
    flatness = np.zeros(len(labels))
    np.maximum.at(flatness, f, abs(height).max(axis=1))
    tol = max(1.0, float(np.ptp(tri.reshape(-1, 3), axis=0).max())) * 1e-11
    planar = (deviation < 1e-9) & (flatness <= tol)
    _, idx = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    idx = idx.reshape(-1, 3)
    edges = np.sort(np.stack((idx, np.roll(idx, -1, axis=1)), axis=-1), axis=-1).reshape(-1, 2)
    _, group, count = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    if np.any(count != 2):
        raise ValueError("Planar regions require a closed manifold mesh")
    adjacent = f[np.argsort(group, kind="stable").reshape(-1, 2) // 3]
    pairs = np.unique(np.sort(adjacent[adjacent[:, 0] != adjacent[:, 1]], axis=1), axis=0)
    parent = np.arange(len(labels))

    def root(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        if (
            planar[a]
            and planar[b]
            and np.linalg.norm(axes[a] - axes[b]) < 1e-9
            and abs(plane[a] - plane[b]) <= tol
        ):
            i, j = sorted((root(a), root(b)))
            parent[j] = i
    return labels[[root(i) for i in range(len(labels))]]


@njit(cache=True)
def voxel_lists(tri, norm, low, sizes, pitch, radius):
    cells = int(sizes[0]) * sizes[1] * sizes[2]
    off = np.zeros(cells + 1, np.int32)
    out = np.empty(0, np.int32)
    cursor = np.empty(0, np.int32)
    for pass_id in range(2):
        if pass_id:
            total = np.cumsum(off.astype(np.int64))
            if total[-1] >= 256_000_000:
                raise ValueError("Voxel index too large; use a larger pitch")
            off = total.astype(np.int32)
            cursor = off[:-1].copy()
            out = np.empty(total[-1], np.int32)
        for j in range(len(tri)):
            t = tri[j]
            normal = norm[j]
            d = np.argmax(np.abs(normal))
            u = (d + 1) % 3
            v = (d + 2) % 3
            r = radius + 1e-9
            half = pitch * 0.5 + r
            x0 = max(0, int(np.floor((t[:, u].min() - r - low[u]) / pitch)))
            x1 = min(sizes[u] - 1, int(np.floor((t[:, u].max() + r - low[u]) / pitch)))
            y0 = max(0, int(np.floor((t[:, v].min() - r - low[v]) / pitch)))
            y1 = min(sizes[v] - 1, int(np.floor((t[:, v].max() + r - low[v]) / pitch)))
            zlo = t[:, d].min() - r
            zhi = t[:, d].max() + r
            du = normal[u] / normal[d]
            dv = normal[v] / normal[d]
            width = half * (abs(du) + abs(dv)) + r
            for y in range(y0, y1 + 1):
                py = low[v] + (y + 0.5) * pitch
                for x in range(x0, x1 + 1):
                    px = low[u] + (x + 0.5) * pitch
                    outside = False
                    for a in range(3):
                        b = (a + 1) % 3
                        dx = t[b, u] - t[a, u]
                        dy = t[b, v] - t[a, v]
                        det = dx * (py - t[a, v]) - dy * (px - t[a, u])
                        if det * np.sign(normal[d]) < -half * (abs(dx) + abs(dy)):
                            outside = True
                            break
                    if outside:
                        continue
                    z = t[0, d] - (px - t[0, u]) * du - (py - t[0, v]) * dv
                    first = max(0, int(np.floor((max(zlo, z - width) - low[d]) / pitch)))
                    last = min(sizes[d] - 1, int(np.floor((min(zhi, z + width) - low[d]) / pitch)))
                    for k in range(first, last + 1):
                        if d == 0:
                            cell = (y * sizes[1] + x) * sizes[0] + k
                        elif d == 1:
                            cell = (x * sizes[1] + k) * sizes[0] + y
                        else:
                            cell = (k * sizes[1] + y) * sizes[0] + x
                        if pass_id:
                            out[cursor[cell]] = j
                            cursor[cell] += 1
                        else:
                            off[cell + 1] += 1
    return off, out


def mesh_features(tri):
    vv, inv = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inv.reshape(-1, 3)
    start = faces.ravel()
    end = np.roll(faces, -1, axis=1).ravel()
    edges, indices, counts = np.unique(
        np.sort(np.column_stack((start, end)), axis=1), axis=0, return_inverse=True, return_counts=True
    )
    if np.any(counts != 2) or np.any(np.bincount(indices, weights=np.where(start < end, 1.0, -1.0))):
        raise ValueError("The target must be a closed, consistently oriented manifold mesh")
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1)[:, None]
    if (
        np.einsum("ij,ij->", tri[:, 0] - tri[0, 0], np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))
        <= 0
    ):
        raise ValueError("The target must use outward orientation")
    en = np.zeros((len(edges), 3))
    np.add.at(en, indices, np.repeat(n, 3, axis=0))
    vn = np.zeros_like(vv)
    for k in range(3):
        a = tri[:, (k + 1) % 3] - tri[:, k]
        b = tri[:, (k + 2) % 3] - tri[:, k]
        theta = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.einsum("ij,ij->i", a, b))
        np.add.at(vn, faces[:, k], theta[:, None] * n)
    return n, np.ascontiguousarray(
        np.concatenate((n[:, None], en[indices].reshape(-1, 3, 3), vn[faces]), axis=1)
    )


def compile_mesh(tri, fa, pitch, *, patch_shift=None):
    from gpu_mesh_line import device_bvhs

    if not np.isfinite(pitch) or pitch <= 0:
        raise ValueError("Expected positive finite voxel pitch")
    n, features = mesh_features(tri)
    order = {}
    _, (nodes, boxes, sorted_tri) = device_bvhs({0: tri}, orders=order)
    ids = order[0]
    tri = tri[ids]
    n = n[ids]
    features = features[ids]
    low = tri.min(axis=(0, 1))
    high = tri.max(axis=(0, 1))
    deviation = 0.0
    if patch_shift is not None:
        deviation = 2 * np.linalg.norm(patch_shift, axis=2).max() + 1e-9
        # Bounds per coordinate prevent curvature in one direction from
        # artificially thickening a flat end face in an unrelated direction.
        controls = 0.5 * (tri + np.roll(tri, -1, axis=1)) + 2 * patch_shift[ids]
        low = np.minimum(low, controls.min(axis=(0, 1)))
        high = np.maximum(high, controls.max(axis=(0, 1)))
    sizes = np.ceil((high - low) / pitch) + 1
    if not np.isfinite(sizes).all() or np.prod(sizes) > 64_000_000:
        raise ValueError("Voxel index exceeds budget; increase pitch")
    sizes = sizes.astype(np.int32)
    radius = 1.5 * pitch + 2 * deviation
    offsets, tri_ids = voxel_lists(tri, n, low, sizes, pitch, radius)
    thin, _ = voxel_lists(tri, n, low, sizes, pitch, deviation)
    occupied = binary_fill_holes((np.diff(thin) > 0).reshape(tuple(sizes[::-1]))).astype(np.uint8)
    tags, compact = np.unique(fa, return_inverse=True)
    geom = (
        sorted_tri,
        features,
        compact[ids].astype(np.int32),
        offsets,
        tri_ids,
        sizes,
        np.array([*low, *high, 1 / pitch, radius] + ([deviation] if patch_shift is not None else [])),
        nodes,
        boxes,
        occupied,
    )
    if patch_shift is not None:
        return geom, tags, ids
    return geom, tags
