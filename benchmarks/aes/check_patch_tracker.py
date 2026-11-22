"""GPU regression for newly engaging surfaces, large pose changes and buffer growth."""

from itertools import product

import cupy as cp
import numpy as np
from scipy.spatial.transform import Rotation

from gpu_mesh_pairs import MeshPair
from gpu_patch_tracker import FacePairHistory, PatchTracker


def check():
    rng = np.random.default_rng(1348)
    a = rng.uniform(-1.0, 1.0, (40, 3, 3))
    b = rng.uniform(-1.0, 1.0, (52, 3, 3))
    order = {}
    full = MeshPair(a, b, orders=order)
    tracker = PatchTracker(a, b, np.arange(len(a)) % 8, np.arange(len(b)) % 9, 4)
    ids = cp.empty((len(a) * len(b), 2), cp.int32)
    count = cp.empty(1, cp.int32)
    for guard, steps in ((0.0, 0), (0.15, 0), (0.02, 4)):
        tracker.reset()
        tracker.hits = cp.empty((1, 2), cp.int32)
        tracker.pairs = cp.empty((1, 2), cp.int32)
        for t, x in [
            (0.0, 10.0),
            (0.7, 0.0),
            (0.701, 0.001),
            (0.702, 0.002),
            (0.699, -0.003),
            (2.0, -15.0),
            (0.7, 0.0),
        ]:
            # Include non-axial rotations and a return to the original contact region.
            r = Rotation.from_rotvec(t * np.array([0.2, 0.5, 0.7])).as_matrix()
            transform = cp.asarray(np.column_stack((r, [x, 0.0, 0.0])))
            count.fill(0)
            full.query(transform, 0.0, 1e-10, abs(x), True, ids, count)
            ref = ids[: int(count.get()[0])].get()
            expected = set(zip(order[0][ref[:, 0]], ref[:, 1]))
            found = tracker.query(transform, guard, abs(x), steps)[0].get()
            assert len(found) == len(expected) and set(map(tuple, found)) == expected
    # A coplanar shared boundary has multiple valid contact IDs.
    tri = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    peer = np.repeat(tri, 3, axis=0)
    tracker = PatchTracker(peer, peer, np.arange(3), np.arange(3), 2)
    pairs = tracker.query(cp.asarray(np.eye(4)[:3].copy()), 0.0, 0)[0].get()
    assert set(map(tuple, pairs)) == set(product(range(3), repeat=2))
    for size in (3, 1200):
        tags = np.arange(size, dtype=np.int32) + 7
        history = FacePairHistory(tags, tags, tracker.module)
        previous = set()
        for rows in ([[0, 1], [1, 2], [0, 1]], [], [[2, 0]]):
            hits = cp.asarray(rows, dtype=cp.int32).reshape(-1, 2)
            current = {(int(tags[a]) << 32) | int(tags[b]) for a, b in rows}
            actual = [set(values.get().tolist()) for values in history.update(hits)]
            assert actual == [current, current - previous, previous - current]
            previous = current


if __name__ == "__main__":
    check()
    print("GPU patch tracker checks passed")
