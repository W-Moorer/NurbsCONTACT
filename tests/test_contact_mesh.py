"""Run with python tests/test_contact_mesh.py from the checkout."""

import sys
import tempfile
import unittest
from pathlib import Path

import meshio
import gmsh
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps"))
from prepare_contact_mesh import check_topology, prepare, refine_slender_faces, sha256


class ContactMeshTest(unittest.TestCase):
    def test_sphere_refinement_and_failed_publication(self):
        sphere = ROOT / "results/engine-comparison/nurbs/sphere.step"
        with tempfile.TemporaryDirectory() as work:
            mesh = Path(work) / "sphere.msh"
            coarse = prepare(sphere, mesh, .006)
            fine = prepare(sphere, mesh, .003)
            parallel = prepare(sphere, Path(work) / "parallel.msh", .003, audit_workers=2)
            self.assertEqual(fine["faces"], parallel["faces"])
            self.assertEqual(fine["edges"], parallel["edges"])
            self.assertEqual(fine["mesh_sha256"], parallel["mesh_sha256"])
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

    def test_refine_slender_cad_faces(self):
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "strip.step"
            gmsh.initialize(readConfigFiles=False)
            try:
                gmsh.option.setNumber("General.Terminal", 0)
                gmsh.model.add("strip")
                gmsh.model.occ.addBox(0, 0, 0, 8, .04, 2)
                gmsh.model.occ.synchronize()
                gmsh.write(str(path))
            finally:
                gmsh.finalize()
            output = path.with_suffix(".msh")
            report = prepare(path, output, .001, min_sicn=.2)
            self.assertGreater(report["locally_refined_edges"], 0)
            self.assertGreaterEqual(report["min_sicn"], .2)

    def test_wheel_trim_end_elements(self):
        # The full wheel has a 24 mm long face bounded by 7.44 µm trims.
        # Isolate one copy to test the corner triangles without meshing every tooth.
        gmsh.initialize(readConfigFiles=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.occ.importShapes(str(ROOT / "results/gears/models/spur-1b.step"))
            gmsh.model.occ.synchronize()
            [(dim, tag)] = gmsh.model.occ.copy([(2, 1)])
            gmsh.model.occ.remove(gmsh.model.getEntities(3), recursive=True)
            gmsh.model.occ.synchronize()
            self.assertEqual(gmsh.model.getEntities(2), [(dim, tag)])
            gmsh.option.setNumber("Mesh.MeshSizeMax", 1.)
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.model.mesh.generate(2)
            def lowest_quality():
                elements, _ = gmsh.model.mesh.getElementsByType(2, tag)
                return gmsh.model.mesh.getElementQualities(elements, "minSICN").min()
            self.assertLess(lowest_quality(), .01)
            self.assertGreater(refine_slender_faces(.2, 1.), 0)
            self.assertGreaterEqual(lowest_quality(), .2)
        finally:
            gmsh.finalize()


if __name__ == "__main__":
    unittest.main()
