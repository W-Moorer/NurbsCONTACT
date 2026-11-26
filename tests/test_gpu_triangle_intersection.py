"""Optional CUDA regressions for contact at facets, edges, vertices and positive gaps."""
from pathlib import Path
import sys
import unittest
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp=None


class TriangleIntersectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if cp is None: raise unittest.SkipTest("CuPy unavailable")
        try:
            if not cp.cuda.runtime.getDeviceCount(): raise unittest.SkipTest("CUDA device unavailable")
        except cp.cuda.runtime.CUDARuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"benchmarks/aes"))

    def test_finite_contacts(self):
        from gpu_triangle_bvh import TriangleMesh,IntersectionQuery
        first=np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]])
        fixtures=[
            ("transverse",[[.25,.25,-1],[.5,.25,1],[.25,.5,1]],True),
            ("coplanar overlap",[[.25,.25,0],[.5,.25,0],[.25,.5,0]],True),
            ("disjoint coplanar triangles",[[1.5,1.5,0],[3,1.5,0],[1.5,3,0]],False),
            ("shared vertex",[[2,0,0],[3,0,1],[2,-1,0]],True),
            ("shared edge",[[0,0,0],[2,0,0],[1,0,1]],True),
            ("positive parallel gap",[[0,0,1e-16],[2,0,1e-16],[0,2,1e-16]],False)]
        transform=cp.asarray(np.eye(4)[:3].copy())
        for name,other,hit in fixtures:
            for a,b in ((first,other),(other,first)):
                with self.subTest(name=name):
                    query=IntersectionQuery(TriangleMesh(a,[[0,1,2]]),TriangleMesh(b,[[0,1,2]]))
                    sizes,points,normals=query.enqueue(transform,geometry=True)
                    self.assertEqual(query.read().tolist(),[[0,0]] if hit else [])
                    if hit:
                        n=int(sizes[0].get())
                        self.assertEqual(n,{"transverse":2,"coplanar overlap":3,"shared vertex":1,"shared edge":2}[name])
                        np.testing.assert_allclose(points[0,:n,2].get(),0,atol=1e-14)
                        np.testing.assert_allclose(np.linalg.norm(normals[0].get(),axis=1),1.,atol=1e-14)

    def test_independent_pairs_and_workspace(self):
        from nurbscontact_gpu.bvh import IntersectionBatch, TriangleMesh
        face = np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]])
        cross = np.array([[.25,.25,-1.],[.5,.25,1.],[.25,.5,1.]])
        triangles = [[0,1,2],[3,4,5]]
        a = TriangleMesh(np.vstack([face,face+[3,0,0]]),triangles)
        b = TriangleMesh(np.vstack([cross, cross + [3, 0, 0]]), triangles)
        batch = IntersectionBatch([(a, b), (b, a)], capacity=2)
        transforms = cp.asarray(np.repeat(np.eye(4)[None, :3], 2, axis=0))
        for active in (1, 0):
            transforms[:, :, 3] = 0.0
            transforms[1 - active, 2, 3] = 4.0
            geometry = batch.enqueue(transforms, geometry=True)
            ids = batch.queries[active].read().tolist()
            self.assertEqual(set(map(tuple, ids)), {(0, 0), (1, 1)})
            self.assertEqual(len(batch.queries[1 - active].read()), 0)
            self.assertTrue((geometry[active][0][:2].get() == 2).all())
        small = IntersectionBatch([(a, b)], capacity=1)
        small.enqueue(transforms[:1])
        with self.assertRaises(OverflowError):
            small.queries[0].read()

    def test_changing_spatial_pose_and_body_order(self):
        from nurbscontact_gpu.bvh import TriangleMesh, IntersectionQuery
        from scipy.spatial.transform import Rotation

        face = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        cross = np.array([[0.35, 0.35, -1.0], [0.75, 0.35, 1.0], [0.35, 0.75, 1.0]])
        separated = np.array([3.7, 2.8, 0.4])
        first = np.vstack([face, face + separated])
        second = np.vstack([cross, cross + separated])
        faces = [[0, 1, 2], [3, 4, 5]]
        rotation = Rotation.from_rotvec([0.63, -0.28, 0.49]).as_matrix()
        shift = np.array([0.7, -1.2, 0.33])
        # Embed the source in a different local frame. Its known world-space
        # intersection segments lie in the target z-planes.
        src = TriangleMesh((first - shift) @ rotation, faces, leaf_size=1)
        dst = TriangleMesh(second, faces, leaf_size=1)
        forward, reverse = IntersectionQuery(src, dst), IntersectionQuery(dst, src)
        for offset, angle, expected in [
            (0.0, 0.0, {(0, 0), (1, 1)}),
            (8.0, 0.37, set()),
            (0.0, 0.0, {(0, 0), (1, 1)}),
        ]:
            r = (
                Rotation.from_rotvec([angle, 0.7 * angle, -angle]).as_matrix()
                @ rotation
            )
            t = shift + np.array([offset, 0.4 * offset, -offset])
            for query, matrix, swapped in [
                (forward, np.c_[r, t], False),
                (reverse, np.c_[r.T, -r.T @ t], True),
            ]:
                sizes, points, _ = query.enqueue(
                    cp.asarray(np.ascontiguousarray(matrix)), geometry=True
                )
                ids = query.read()
                hits = {tuple(pair[::-1] if swapped else pair) for pair in ids}
                self.assertEqual(len(ids), len(hits))
                self.assertEqual(hits, expected)
                for k, (i, _) in enumerate(ids):
                    self.assertEqual(sizes[k].item(), 2)
                    segment = points[k, :2].get()
                    if swapped:
                        segment = segment @ r.T + t
                    origin = i * separated
                    reference = (
                        np.array([[0.55, 0.35, 0.0], [0.35, 0.55, 0.0]]) + origin
                    )
                    delta = np.linalg.norm(segment[:, None] - reference[None], axis=2)
                    np.testing.assert_allclose(delta.min(axis=0), 0, atol=3e-14)

    def test_cached_motion_finds_new_contacts(self):
        from nurbscontact_gpu.bvh import TriangleMesh, IntersectionBatch

        face = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        crossing = np.array([[0.25, 0.25, 0.02], [0.5, 0.25, 0.3], [0.25, 0.5, 0.3]])
        a = TriangleMesh(face, [[0, 1, 2]])
        b = TriangleMesh(crossing, [[0, 1, 2]])
        queries = IntersectionBatch([(a, b)], reuse_distance=0.1)
        matrix = cp.asarray(np.eye(4)[None, :3].copy())
        for height, rebuild, count in [
            (0.0, 1, 0),
            (0.05, 0, 1),
            (-2.0, 1, 0),
            (0.05, 1, 1),
        ]:
            matrix[0, 2, 3] = height
            queries.enqueue(matrix)
            self.assertEqual(queries.refresh.item(), rebuild)
            self.assertEqual(len(queries.queries[0].read()), count)


if __name__ == "__main__":
    unittest.main()
