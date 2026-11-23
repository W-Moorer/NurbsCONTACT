"""Rigid-local source patch bounds and an octree over the target's occupied cells."""

import cupy as cp
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter


def block_index(model, guard):
    points = model.samples[0].get()
    # Spatial order is an indirection. Contact keys retain the original probe IDs.
    ids = cKDTree(points, leafsize=16).indices.astype(np.int32)
    ordered = points[ids]
    boxes = []
    for width in (512, 32):
        starts = np.arange(0, len(points), width)
        lo = np.minimum.reduceat(ordered, starts, axis=0)
        hi = np.maximum.reduceat(ordered, starts, axis=0)
        boxes.append(cp.asarray(np.column_stack(((hi + lo) * 0.5, (hi - lo) * 0.5))))

    # Each octree parent encloses all its occupied children, including interior cells.
    grid = model.geometry[9].get()
    factor = 2 if grid.size <= 4_000_000 else 1
    bounds = model.geometry[6].get()
    inv = float(bounds[6])
    if factor == 2:
        # Conservative fine cells admitted by the same plane test as the narrow phase.
        planes = model.geometry[10].get().reshape((*grid.shape, 4))
        normal = planes[..., :3].astype(np.float64)
        padding = max(1e-9, 128 * np.finfo(float).eps * (1 + np.max(np.abs(bounds[:6])) + max(grid.shape) / inv))
        reach = np.abs(normal).sum(axis=-1) / (4 * inv)
        fine = np.empty(tuple(2 * n for n in grid.shape), np.uint8)
        for dz, dy, dx in np.ndindex(2, 2, 2):
            center = (
                normal[..., 0] * (2 * dx - 1) + normal[..., 1] * (2 * dy - 1) + normal[..., 2] * (2 * dz - 1)
            ) / (4 * inv)
            fine[dz::2, dy::2, dx::2] = grid & ~(planes[..., 3] + center > reach + padding)
    else:
        fine = grid
    if guard >= max(fine.shape) / (inv * factor):
        halo = np.full_like(fine, fine.any())
        halo_margin = np.inf
    else:
        radius_cells = int(np.floor(guard * inv * factor)) + 1
        halo = maximum_filter(fine, size=2 * radius_cells + 1, mode="constant")
        halo_margin = radius_cells / (inv * factor)
    levels, arrays, offset = [], [], 0
    while True:
        nz, ny, nx = grid.shape
        levels.append((offset, nx, ny, nz))
        arrays.append(grid.ravel())
        if grid.size == 1:
            break
        offset += grid.size
        even = np.pad(grid, ((0, nz % 2), (0, ny % 2), (0, nx % 2)))
        grid = even.reshape((nz + 1) // 2, 2, (ny + 1) // 2, 2, (nx + 1) // 2, 2).max(axis=(1, 3, 5))
    return (
        *boxes,
        cp.asarray(ids),
        cp.asarray(np.concatenate(arrays)),
        cp.asarray(levels, dtype=cp.int32),
        cp.empty(len(points), cp.int32),
        np.float64(np.linalg.norm(points, axis=1).max()),
        np.float64(guard),
        np.int32(len(boxes[0])),
        np.int32(len(levels)),
        cp.asarray(halo),
        np.int32(factor),
        np.float64(halo_margin),
    )
