"""Check mixed contact laws against independent active-set solutions."""
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


class NormalLawTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if cp is None:
            raise unittest.SkipTest("CuPy unavailable")
        try:
            if not cp.cuda.runtime.getDeviceCount():
                raise unittest.SkipTest("CUDA device unavailable")
        except cp.cuda.runtime.CUDARuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/aes"))

    def test_mixed_coupled_contacts(self):
        from gpu_normal_solver import NormalContactSolve
        # Two pairs share a moving body. The third row moves both tangential
        # bodies with different normal projections.
        j = np.array([[1., -1., 0.], [0., 1., -1.], [.4, 0., .8]])
        inv_mass = np.diag([1., .7, .5])
        w = j @ inv_mass @ j.T
        h = .0001
        k = np.array([1.6e8, 2e8, 8e7])
        law = np.array(list(itertools.product(range(3), repeat=3)), dtype=np.int32)
        bias = np.broadcast_to(np.array([-.4, -.05, .12]), law.shape)
        solver = NormalContactSolve(*law.shape)
        result = solver.enqueue(cp.asarray(np.broadcast_to(w, (len(law), 3, 3)).copy()), cp.asarray(h*bias),
                                cp.asarray(law), cp.asarray(np.broadcast_to(k, law.shape).copy()), h, 1e-11).get()
        np.testing.assert_array_equal(solver.status.get(), 0)
        for p, types, speed in zip(result, law, bias, strict=True):
            system = w + np.diag((types == solver.PENALTY)/(h*h*k))
            for active in itertools.product((False, True), repeat=3):
                ix = np.flatnonzero(active)
                expected = np.zeros(3)
                expected[ix] = np.linalg.solve(system[np.ix_(ix, ix)], -speed[ix])
                if np.min(expected) >= -1e-12 and np.min(speed + system @ expected) >= -1e-12:
                    break
            else:
                self.fail("No feasible active set")
            np.testing.assert_allclose(p, expected, atol=5e-11, rtol=1e-10)
            np.testing.assert_allclose(inv_mass @ j.T @ p, inv_mass @ j.T @ expected, atol=5e-11)

    def test_unilateral_impact_and_failures(self):
        from gpu_normal_solver import NormalContactSolve
        solver = NormalContactSolve(5, 1)
        # Two unit masses at first touch and unit relative closing speed exchange 0.5 N s.
        w = cp.asarray([2., 2., 2., 0., 2.]).reshape(-1, 1, 1)
        g = cp.asarray([-.001, -.001, .01, -.001, -.001]).reshape(-1, 1)
        laws = cp.asarray([[1], [2], [1], [1], [0]], dtype=cp.int32)
        k = cp.zeros((5, 1))
        impulse = solver.enqueue(w, g, laws, k, .001, 1e-10, 100).get()[:, 0]
        np.testing.assert_allclose(impulse[:3], [.5, .5, 0.], atol=1e-10)
        np.testing.assert_array_equal(solver.status.get(), [0, 0, 0, 1, 2])
        self.assertGreater(solver.residual.get()[3], .9)

    def test_warp_stride(self):
        from gpu_normal_solver import NormalContactSolve
        rows = 40
        solver = NormalContactSolve(1, rows)
        impulse = solver.enqueue(cp.eye(rows)[None], cp.full((1, rows), -.001),
                                 cp.ones((1, rows), dtype=cp.int32), cp.zeros((1, rows)), .001)
        np.testing.assert_allclose(impulse.get(), 1.)
        np.testing.assert_array_equal(solver.status.get(), 0)

    def test_two_row_active_sets_and_redundancy(self):
        from gpu_normal_solver import NormalContactSolve
        cases = [(1., 2.), (-1., 2.), (2., -1.), (-1., -1.)]
        w = np.tile(np.array([[2., -1.], [-1., 2.]]), (5, 1, 1))
        w[-1] = 1.
        solver = NormalContactSolve(5, 2)
        p = solver.enqueue(cp.asarray(w), cp.asarray(np.array(cases + [(-1., -1.)])*.001),
                           cp.ones((5, 2), cp.int32), cp.zeros((5, 2)), .001).get()
        np.testing.assert_array_equal(solver.status.get(), 0)
        np.testing.assert_allclose(p[:4], [[0., 0.], [.5, 0.], [0., .5], [1., 1.]])
        self.assertAlmostEqual(p[-1].sum(), 1.)


if __name__ == "__main__":
    unittest.main()
