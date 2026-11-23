"""Run with python tests/test_contact_mesh.py from the checkout."""

import sys
import tempfile
import unittest
from pathlib import Path

import meshio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps"))
from prepare_contact_mesh import check_topology, prepare, sha256


class ContactMeshTest(unittest.TestCase):
    def test_sphere_refinement_and_failed_publication(self):
        sphere = ROOT / "results/engine-comparison/nurbs/sphere.step"
        with tempfile.TemporaryDirectory() as work:
            mesh = Path(work) / "sphere.msh"
            coarse = prepare(sphere, mesh, .006)
            fine = prepare(sphere, mesh, .003)
            self.assertLess(fine["max_sampled_surface_error_m"], coarse["max_sampled_surface_error_m"])
            self.assertEqual(fine["mesh_sha256"], sha256(mesh))
            data = meshio.read(mesh)
            self.assertTrue(np.allclose(np.ptp(data.points, axis=0), .070, atol=.0003))
            tri = data.points[data.get_cells_type("triangle")]
            volume = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6
            self.assertAlmostEqual(volume, fine["signed_volume_m3"])
            self.assertLess(abs(volume / (4 * np.pi * .035**3 / 3) - 1), .01)
            quality = mesh.with_suffix(".quality.json")
            before = sha256(quality)
            with self.assertRaisesRegex(ValueError, "quality limits"):
                prepare(sphere, mesh, .006, max_error_m=1e-9)
            self.assertEqual(fine["mesh_sha256"], sha256(mesh))
            self.assertEqual(before, sha256(quality))

    def test_topological_defects(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
        check_topology(vertices, faces)
        with self.assertRaisesRegex(ValueError, "boundary"):
            check_topology(vertices, faces[:-1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            check_topology(vertices, np.vstack((faces, faces[0])))
        faces[0] = faces[0, ::-1]
        with self.assertRaisesRegex(ValueError, "winding"):
            check_topology(vertices, faces)


if __name__ == "__main__":
    unittest.main()
