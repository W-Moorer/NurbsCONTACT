"""Double-precision nearest-feature queries for the fitted CAD field."""
import numpy as np
from numba import njit
from scipy.spatial import cKDTree

from fitted_projection import refine
from trim_bounds import box_distance, segment_foot


def pack_tree(xyz, elements):
    """Partition primitive centres once, then bound the full primitives in each node."""
    if not len(elements):
        return (np.empty((0, 2, 3)), np.empty((0, 2), np.int32),
                np.empty(0, np.int64), 1)
    vertices = xyz[elements]
    partition = cKDTree(vertices.mean(axis=1), leafsize=8)
    order = np.ascontiguousarray(partition.indices, dtype=np.int64)
    nodes = [partition.tree]
    links = np.empty((partition.size, 2), np.int32)
    leaves = []
    depth = 0
    for i, node in enumerate(nodes):
        if node.split_dim < 0:
            links[i] = ~node.start_idx, node.end_idx
            leaves.append(i)
            depth = max(depth, node.level)
        else:
            links[i] = len(nodes), len(nodes) + 1
            nodes.extend((node.lesser, node.greater))
    leaves = np.asarray(leaves)
    starts = ~links[leaves, 0]
    leaves = leaves[np.argsort(starts)]
    bounds = np.empty((len(nodes), 2, 3))
    starts = ~links[leaves, 0]
    bounds[leaves, 0] = np.minimum.reduceat(vertices.min(axis=1)[order], starts, axis=0)
    bounds[leaves, 1] = np.maximum.reduceat(vertices.max(axis=1)[order], starts, axis=0)
    # Children have consecutive indices and are always written after their parent.
    for i in range(len(nodes) - 1, -1, -1):
        left, right = links[i]
        if left >= 0:
            bounds[i, 0] = np.minimum(bounds[left, 0], bounds[right, 0])
            bounds[i, 1] = np.maximum(bounds[left, 1], bounds[right, 1])
    return bounds, links, order, depth + 1


@njit(cache=True)
def cell_lists(low, high, leaves, dims):
    cells = dims[0]*dims[1]*dims[2]
    offsets = np.zeros(cells+1, np.int32)
    entries = np.empty(0, np.int32)
    cursor = np.empty(0, np.int32)
    for write in (False, True):
        for i, node in enumerate(leaves):
            for a in range(low[i,0], high[i,0]+1):
                for b in range(low[i,1], high[i,1]+1):
                    base = (a*dims[1]+b)*dims[2]
                    for c in range(low[i,2], high[i,2]+1):
                        cell = base+c
                        if write:
                            entries[cursor[cell]] = node
                            cursor[cell] += 1
                        else:
                            offsets[cell+1] += 1
        if not write:
            for j in range(cells):
                offsets[j+1] += offsets[j]
            cursor = offsets[:-1].copy()
            entries = np.empty(offsets[-1], np.int32)
    return offsets, entries


def pack_grid(xyz, tree, resolution):
    """Index nearby BVH leaves; oversized indices fall back to tree traversal."""
    span = np.ptp(xyz, axis=0)
    width = max(span.max()/resolution, np.finfo(float).tiny)
    origin = xyz.min(axis=0)-width
    dims = np.ceil(span/width).astype(np.int64)+3
    bounds, links, _, _ = tree
    leaves = np.flatnonzero(links[:,0]<0)
    band = .25*width
    # Pad the assignment outwards at cell boundaries before quantizing.
    low = np.floor((np.nextafter(bounds[leaves,0]-band, -np.inf)-origin)/width).astype(np.int64)
    high = np.floor((np.nextafter(bounds[leaves,1]+band, np.inf)-origin)/width).astype(np.int64)
    low, high = np.maximum(low, 0), np.minimum(high, dims-1)
    if not len(leaves) or np.prod(high-low+1, axis=1).sum() > 8_000_000:
        return origin, width, dims, np.empty(0,np.int32), np.empty(0,np.int32)
    offsets, entries = cell_lists(low, high, leaves, dims)
    return origin, width, dims, offsets, entries


@njit(cache=True, inline='always')
def primitive_foot(point, idx, xyz, elements, bary_grad):
    a, b = xyz[elements[idx,0]], xyz[elements[idx,1]]
    if elements.shape[1] == 2:
        return segment_foot(point, a, b)
    c = xyz[elements[idx,2]]
    v, w = 0., 0.
    for k in range(3):
        delta = point[k]-a[k]
        v += delta*bary_grad[idx,0,k]
        w += delta*bary_grad[idx,1,k]
    if v >= 0. and w >= 0. and v+w <= 1.:
        x = a[0] + v*(b[0]-a[0]) + w*(c[0]-a[0])
        y = a[1] + v*(b[1]-a[1]) + w*(c[1]-a[1])
        z = a[2] + v*(b[2]-a[2]) + w*(c[2]-a[2])
        return (point[0]-x)**2 + (point[1]-y)**2 + (point[2]-z)**2, x, y, z
    ab = segment_foot(point, a, b)
    bc = segment_foot(point, b, c)
    ca = segment_foot(point, c, a)
    best = bc if bc[0] < ab[0] else ab
    return ca if ca[0] < best[0] else best


@njit(cache=True)
def closest(point, limit, xyz, elements, bary_grad, tree, stack):
    bounds, links, order, _ = tree
    best, hit = limit, -1
    x, y, z = 0., 0., 0.
    if not len(order):
        return best, hit, x, y, z
    stack[0] = 0
    pending = 1
    while pending:
        pending -= 1
        node = stack[pending]
        if box_distance(point, bounds[node]) > best:
            continue
        left, right = links[node]
        if left < 0:
            for pos in range(~left, right):
                idx = order[pos]
                sq, px, py, pz = primitive_foot(point, idx, xyz, elements, bary_grad)
                if sq < best or (sq == best and (hit < 0 or idx < hit)):
                    best, hit, x, y, z = sq, idx, px, py, pz
        else:
            a = box_distance(point, bounds[left])
            b = box_distance(point, bounds[right])
            if a > b:
                left, right, a, b = right, left, b, a
            if b <= best:
                stack[pending] = right
                pending += 1
            if a <= best:
                stack[pending] = left
                pending += 1
    return best, hit, x, y, z


@njit(cache=True)
def nearby(point, limit, xyz, elements, bary_grad, tree, grid, stack):
    origin, width, dims, offsets, entries = grid
    # A leaf can be absent from this cell only when every primitive in
    # it is farther than the band. Beyond the band use the complete BVH.
    if len(offsets):
        cell = 0
        for k in range(3):
            coordinate = (point[k]-origin[k])/width
            if not 0. <= coordinate < dims[k]:
                cell = -1
                break
            cell = cell*dims[k]+int(coordinate)
        if cell >= 0:
            bounds, links, order, _ = tree
            band_sq = (.25*width)**2
            best, hit = min(limit, band_sq), -1
            x, y, z = 0., 0., 0.
            for pos in range(offsets[cell], offsets[cell+1]):
                node = entries[pos]
                if box_distance(point, bounds[node]) > best:
                    continue
                for j in range(~links[node,0], links[node,1]):
                    idx = order[j]
                    sq, px, py, pz = primitive_foot(point, idx, xyz, elements, bary_grad)
                    if sq < best or (sq == best and (hit < 0 or idx < hit)):
                        best, hit, x, y, z = sq, idx, px, py, pz
            if hit >= 0 or limit <= band_sq:
                return best, hit, x, y, z
    return closest(point, limit, xyz, elements, bary_grad, tree, stack)


@njit(cache=True)
def locate(points, xyz, tri, edges, bary_grad, tri_tree, edge_tree, tri_grid, edge_grid, crease_band_mm):
    nearest = np.empty(len(points), np.int64)
    seeds = np.empty((len(points), 3))
    edge = np.empty(len(points), np.int64)
    edge_sq = np.empty(len(points))
    edge_seeds = np.empty_like(seeds)
    stack = np.empty(max(tri_tree[3], edge_tree[3]), np.int32)
    for i, point in enumerate(points):
        sq, nearest[i], seeds[i,0], seeds[i,1], seeds[i,2] = nearby(
            point, np.inf, xyz, tri, bary_grad, tri_tree, tri_grid, stack)
        if sq > crease_band_mm*crease_band_mm:
            edge_sq[i], edge[i] = np.inf, -1
            continue
        # refine() only uses a crease within this radius. A miss need not
        # compute its actual distance or traverse more distant edge subtrees.
        limit = (2*np.sqrt(sq)+.005)**2
        dist, hit, x, y, z = nearby(point, limit, xyz, edges, bary_grad, edge_tree, edge_grid, stack)
        edge_sq[i] = dist if hit >= 0 else np.inf
        edge[i], edge_seeds[i,0], edge_seeds[i,1], edge_seeds[i,2] = hit, x, y, z
    return nearest, seeds, edge_sq, edge, edge_seeds


@njit(cache=True)
def signed_distances(points, xyz, tri, edges, bary_grad, tri_tree, edge_tree, tri_grid, edge_grid, owner, uv,
                     incident, edge_pairs, convex_sign, coefficients, offsets, frames, crease_band_mm,
                     trim_index=None, prune=True):
    nearest, seeds, edge_sq, edge, edge_seeds = locate(
        points, xyz, tri, edges, bary_grad, tri_tree, edge_tree, tri_grid, edge_grid, crease_band_mm)
    return refine(points, nearest, seeds, edge_sq, edge, edge_seeds, xyz, tri, owner, uv,
                  incident, edge_pairs, convex_sign, bary_grad, coefficients, offsets, frames, crease_band_mm,
                  trim_index, prune)
