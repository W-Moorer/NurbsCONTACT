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
from prepare_contact_mesh import check_topology, open_step, prepare, refine_slender_faces, sha256


class ContactMeshTest(unittest.TestCase):
    def test_coarser_faces_preserve_edge_segments(self):
        sphere = ROOT / "results/engine-comparison/nurbs/sphere.step"
        with tempfile.TemporaryDirectory() as work:
            boundaries = []
            counts = []
            for name, face_samples, edge_samples in (("reference", 32, None), ("compact", 20, 32)):
                path = Path(work) / f"{name}.msh"
                report = prepare(sphere, path, .02, extend_boundary=False, curvature_samples=face_samples,
                                 edge_curvature_samples=edge_samples)
                mesh = meshio.read(path)
                segments = mesh.points[mesh.get_cells_type("line")]
                boundaries.append({tuple(sorted(map(tuple, segment))) for segment in segments})
                counts.append(report["triangles"])
            self.assertTrue(boundaries[0])
            self.assertEqual(*boundaries)
            self.assertLess(counts[1], counts[0])

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
            mapped=prepare(path,Path(work)/"mapped.msh",.005,extend_boundary=False,mapped_edge_error_m=.000002)
            self.assertEqual(mapped["mapped_faces"],6)
            self.assertLess(mapped["triangles"],report["triangles"])
            self.assertAlmostEqual(mapped["signed_volume_m3"],report["signed_volume_m3"],places=16)
            self.assertLess(mapped["max_sampled_surface_error_m"],1e-12)

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

    def test_cad_projection_matches_native_on_thin_and_curved_faces(self):
        try:
            from OCP.BRep import BRep_Tool
            from OCP.STEPControl import STEPControl_Reader
            from OCP.TopAbs import TopAbs_FACE
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopoDS import TopoDS
        except ImportError:
            self.skipTest("cadquery-ocp optional spline acceleration is unavailable")
        from cad_audit_projection import FaceProjection
        path = ROOT / "results/gears/models/spur-1b.step"
        reader = STEPControl_Reader()
        reader.ReadFile(str(path))
        reader.TransferRoots()
        explorer = TopExp_Explorer(reader.OneShape(),TopAbs_FACE)
        gmsh.initialize(readConfigFiles=False)
        try:
            open_step(path)
            for tag in (1,2):
                surface = BRep_Tool.Surface_s(TopoDS.Face_s(explorer.Current()))
                explorer.Next()
                kernel = FaceProjection(tag,surface)
                # Straddle a subpatch boundary, offset along CAD normals and
                # include a query outside the admissible control box.
                uv = np.array([[0.,.5],[.124,.1],[.125,.5],[.126,.9],[.37,.25],[.8,.55],[1.,.5]])
                xyz = gmsh.model.getValue(2,tag,uv.ravel()).reshape(-1,3)
                normals = gmsh.model.getNormal(tag,uv.ravel()).reshape(-1,3)
                points = np.vstack((xyz, xyz+.001*normals, xyz[0]+100.))
                old, old_uv = gmsh.model.getClosestPoint(2,tag,points.ravel())
                new, new_uv, fallback = kernel.project(points)
                self.assertGreater(fallback,0)
                self.assertLess(fallback,len(points))
                np.testing.assert_allclose(new,old.reshape(-1,3),rtol=0,atol=2e-7)
                np.testing.assert_allclose(gmsh.model.getNormal(tag,new_uv),gmsh.model.getNormal(tag,old_uv),
                                           rtol=0,atol=2e-7)
        finally:
            gmsh.finalize()


if __name__ == "__main__":
    unittest.main()
