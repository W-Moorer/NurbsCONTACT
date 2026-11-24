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


if __name__=="__main__":
    unittest.main()
