"""Find overlapping triangle pairs on face-tagged rigid meshes at successive poses."""

from itertools import product
from pathlib import Path

import cupy as cp
import numpy as np
from scipy.spatial import cKDTree

from gpu_mesh_line import TRIANGLE_CUDA


class FacePairHistory:
    """Return device-side active, entering and leaving face IDs until the next call."""

    def __init__(self, face_a, face_b, module):
        tags_a, ia = np.unique(face_a, return_inverse=True)
        tags_b, ib = np.unique(face_b, return_inverse=True)
        self.columns = len(tags_b)
        cells = len(tags_a) * self.columns
        words = (cells + 31) // 32
        # Keep word-table storage linear in the input size. Large sparse
        # assemblies can still use packed IDs without a quadratic allocation.
        self.dense = cells < 2**31 and words <= 8 * (len(tags_a) + len(tags_b))
        if self.dense:
            self.tags_a, self.tags_b = (cp.asarray(t, dtype=cp.uint32) for t in (tags_a, tags_b))
            self.ia, self.ib = (cp.asarray(v, dtype=cp.int32) for v in (ia, ib))
            self.marked = cp.zeros(words, cp.uint32)
            self.previous = cp.zeros_like(self.marked)
            self.counts = cp.zeros(3, cp.int32)
            self.out = cp.empty((3, 1), cp.uint64)
            self.mark = module.get_function("mark_faces")
            self.changes = module.get_function("face_changes")
        else:
            self.face_a = cp.asarray(face_a, dtype=cp.uint64)
            self.face_b = cp.asarray(face_b, dtype=cp.uint64)
            self.previous = cp.empty(0, cp.uint64)

    def update(self, hits):
        if self.dense:
            if len(hits) > self.out.shape[1]:
                self.out = cp.empty((3, len(hits)), cp.uint64)
            self.counts.fill(0)
            if len(hits):
                self.mark(
                    ((len(hits) + 127) // 128,),
                    (128,),
                    (hits, np.int32(len(hits)), self.ia, self.ib, np.int32(self.columns), self.marked),
                )
            words = len(self.marked)
            self.changes(
                ((words + 127) // 128,),
                (128,),
                (
                    self.marked,
                    self.previous,
                    np.int32(words),
                    np.int32(self.columns),
                    self.tags_a,
                    self.tags_b,
                    self.counts,
                    *self.out,
                ),
            )
            return tuple(row[: int(n)] for row, n in zip(self.out, self.counts.get(), strict=True))
        current = cp.unique((self.face_a[hits[:, 0]] << 32) | self.face_b[hits[:, 1]])
        entered = cp.setdiff1d(current, self.previous, assume_unique=True)
        left = cp.setdiff1d(self.previous, current, assume_unique=True)
        self.previous = current
        return current, entered, left


def patch_layout(tri, face, leaf_size):
    if len(tri) != len(face) or not len(tri) or leaf_size < 1:
        raise ValueError("Invalid patch input")
    centers = tri.mean(axis=1)
    ids, boxes, bounds, offsets = [], [], [], [0]
    for tag in np.unique(face):
        members = np.flatnonzero(face == tag)
        tree = cKDTree(centers[members], leafsize=leaf_size)
        stack = [tree.tree]
        while stack:
            node = stack.pop()
            if node.split_dim >= 0:
                stack.extend((node.greater, node.lesser))
                continue
            ordered = members[tree.indices[node.start_idx : node.end_idx]]
            for first in range(0, len(ordered), leaf_size):
                items = ordered[first : first + leaf_size]
                xyz = tri[items].reshape(-1, 3)
                origin = xyz.mean(axis=0)
                centered = xyz - origin
                _, axes = np.linalg.eigh(centered.T @ centered)
                projected = centered @ axes
                low, high = projected.min(axis=0), projected.max(axis=0)
                boxes.append(np.vstack((origin + axes @ ((high + low) / 2), (high - low) / 2, axes.T)))
                bounds.append([xyz.min(axis=0), xyz.max(axis=0)])
                ids.extend(items)
                offsets.append(len(ids))
    return np.asarray(ids, np.int32), np.asarray(offsets, np.int32), np.asarray(boxes), np.asarray(bounds)


def patch_tree(boxes):
    mid = boxes.mean(axis=1)
    nodes, bounds = [], []

    def visit(ids):
        index = len(nodes)
        nodes.append(None)
        bounds.append([boxes[ids, 0].min(axis=0), boxes[ids, 1].max(axis=0)])
        if len(ids) > 1:
            axis = np.ptp(mid[ids], axis=0).argmax()
            half = len(ids) // 2
            ids = ids[np.argpartition(mid[ids, axis], half)]
            visit(ids[:half])
            visit(ids[half:])
        nodes[index] = [len(nodes), int(ids[0]) if len(ids) == 1 else -1]

    visit(np.arange(len(boxes)))
    return cp.asarray(nodes, dtype=cp.int32), cp.asarray(bounds, dtype=cp.float64)


class PatchTracker:
    """The guard enlarges only the cached search region. Forces never use it."""

    def __init__(self, a, b, face_a, face_b, leaf_size=16):
        ia, oa, ba, bounds = patch_layout(a, face_a, leaf_size)
        ib, ob, bb, _ = patch_layout(b, face_b, leaf_size)
        if len(a) * len(b) > np.iinfo(np.int64).max or len(ba) * len(bb) > np.iinfo(np.int32).max:
            raise ValueError("Patch table exceeds the supported index range")
        self.tree, self.bounds = patch_tree(bounds)
        self.ia, self.oa, self.ba = map(cp.asarray, (ia, oa, ba))
        self.ib, self.ob, self.bb = map(cp.asarray, (ib, ob, bb))
        self.a, self.b = cp.asarray(a[ia]), cp.asarray(b[ib])
        self.tri_boxes = cp.concatenate((self.a.min(axis=1), self.a.max(axis=1)), axis=1)
        self.moved = cp.empty_like(self.b)
        self.moved_boxes = cp.empty((len(b), 6))
        self.moved_patches = cp.empty_like(self.bb)
        self.envelopes = cp.empty_like(self.bb)
        low, high = b.min(axis=(0, 1)), b.max(axis=(0, 1))
        self.corners = cp.asarray(list(product(*zip(low, high))))
        self.scale = float(max(1, np.abs(a).max(), np.abs(b).max()))
        self.anchor = cp.zeros((3, 4))
        self.previous = cp.empty_like(self.anchor)
        self.state = cp.empty(3, cp.uint64)
        self.pairs = cp.empty((8192, 2), cp.int32)
        self.hits = cp.empty((4096, 2), cp.int32)
        self.grid = 4 * int(cp.cuda.Device().attributes["MultiProcessorCount"])
        self.touch = 1e-10
        folder = Path(__file__).parent
        source = (
            TRIANGLE_CUDA
            + (folder / "mesh_pair.cu").read_text()
            + (folder / "mesh_patch_tracking.cu").read_text()
        )
        self.module = cp.RawModule(code=source)
        self.gate, self.place_boxes, self.place_mesh, self.find, self.intersect = (
            self.module.get_function(name)
            for name in (
                "refresh_gate",
                "place_patches",
                "place_mesh",
                "discover_patches",
                "intersect_patches",
            )
        )
        self.begin_check, self.check, self.prepare, self.begin_discovery = (
            self.module.get_function(name)
            for name in ("begin_envelope_check", "check_envelopes", "prepare_envelopes", "begin_discovery")
        )
        self.reset()

    def reset(self):
        self.state.fill(2)
        self.policy = None
        self.primed = False

    def query(self, pose, guard_mm, translation_scale, lookahead=0):
        if not np.isfinite(guard_mm) or guard_mm < 0 or lookahead < 0:
            raise ValueError("Expected a nonnegative search guard in millimetres")
        if self.policy != (guard_mm, lookahead):
            self.reset()
            self.policy = (guard_mm, lookahead)
        pad = self.touch + 512 * np.finfo(float).eps * (self.scale + translation_scale + guard_mm)
        n, nb = len(self.b), len(self.bb)
        blocks = ((nb + 127) // 128,)
        self.place_boxes(blocks, (128,), (self.bb, self.moved_patches, np.int32(nb), pose))
        self.place_mesh(
            ((n + 127) // 128,), (128,), (self.b, self.moved, self.moved_boxes, np.int32(n), pose)
        )
        while True:
            search, margin = self.moved_patches, guard_mm + pad
            if lookahead:
                self.begin_check((1,), (1,), (self.state, np.int32(not self.primed)))
                if self.primed:
                    self.check(
                        blocks,
                        (128,),
                        (self.moved_patches, self.envelopes, np.int32(nb), self.state, np.float64(pad)),
                    )
                else:
                    cp.copyto(self.previous, pose)
                self.begin_discovery((1,), (1,), (self.state,))
                self.prepare(
                    blocks,
                    (128,),
                    (
                        self.moved_patches,
                        self.envelopes,
                        np.int32(nb),
                        self.previous,
                        pose,
                        np.int32(lookahead),
                        np.float64(guard_mm + 2 * pad),
                        self.state,
                    ),
                )
                search, margin = self.envelopes, pad
            else:
                self.gate((1,), (1,), (pose, self.anchor, self.corners, np.float64(guard_mm), self.state))
            self.find(
                blocks,
                (128,),
                (
                    self.tree,
                    self.bounds,
                    self.ba,
                    search,
                    np.int32(nb),
                    np.float64(margin),
                    self.state,
                    self.pairs,
                    np.int32(len(self.pairs)),
                ),
            )
            self.intersect(
                (self.grid,),
                (128,),
                (
                    self.pairs,
                    np.int32(len(self.pairs)),
                    self.state,
                    self.ba,
                    self.moved_patches,
                    self.oa,
                    self.ob,
                    self.a,
                    self.moved,
                    self.tri_boxes,
                    self.moved_boxes,
                    self.ia,
                    self.ib,
                    np.float64(pad),
                    np.float64(self.touch**2),
                    self.hits,
                    np.uint64(len(self.hits)),
                ),
            )
            if lookahead:
                cp.copyto(self.previous, pose)
            patches, hits, rebuilt = map(int, self.state.get())
            if patches <= len(self.pairs) and hits <= len(self.hits):
                self.primed = True
                return self.hits[:hits], patches, bool(rebuilt)
            if patches > len(self.pairs):
                self.pairs = cp.empty((patches, 2), cp.int32)
            if hits > len(self.hits):
                self.hits = cp.empty((hits, 2), cp.int32)
            self.state.fill(2)
            self.primed = False
