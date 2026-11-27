"""Analytic finite-feature and implicit-contact regressions for the triangle backend."""

from pathlib import Path
import tempfile
import unittest
import meshio
import numpy as np


def box():
    vertices = np.array(
        [[x, y, z] for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0)]
    )
    faces = np.array(
        [
            [0, 2, 3],
            [0, 3, 1],
            [4, 5, 7],
            [4, 7, 6],
            [0, 1, 5],
            [0, 5, 4],
            [2, 6, 7],
            [2, 7, 3],
            [0, 4, 6],
            [0, 6, 2],
            [1, 3, 7],
            [1, 7, 5],
        ]
    )
    return vertices, faces


def tetra(vertices):
    vertices = np.array(vertices, dtype=float)
    faces = []
    for i in range(4):
        f = [(i + 1) % 4, (i + 2) % 4, (i + 3) % 4]
        p = vertices[f]
        if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), vertices[i] - p[0]) > 0:
            f[0], f[1] = f[1], f[0]
        faces.append(f)
    return vertices, np.array(faces)


def tip():
    return tetra(
        [[0.65, 0.25, 0], [0.60, 0.20, 0.20], [0.70, 0.20, 0.20], [0.65, 0.30, 0.20]]
    )


def make_scene(folder, meshes, *, initial_q=0.0, moving=False):
    for i, (v, f) in enumerate(meshes):
        meshio.write(
            folder / f"{i}.msh",
            meshio.Mesh(
                v,
                [("triangle", f)],
                cell_data={
                    "gmsh:physical": [np.ones(len(f), int)],
                    "gmsh:geometrical": [np.ones(len(f), int)],
                },
            ),
            file_format="gmsh22",
            binary=False,
        )
    lines = [
        "nc-contact-v2",
        "gravity 0 0 0 1 2 1",
        f"coordinate 1 {initial_q} 0 0 0 0 0",
    ]
    lines += [
        'body "0.msh" 0 0 0 0 0 0 1 0 0 0 0 -1 0 0 0 0',
        'body "1.msh" 0 0 0 '
        + ("1" if moving else "0")
        + " 0 0 1 0 0 0 0 -1 0 0 0 "
        + ("1 0 0 0 1" if moving else "0"),
    ]
    lines += ["surface 0 1 1000 1 0 0 .0005 .1", "drives 0", "loads 0"]
    path = folder / "test.scene"
    path.write_text("\n".join(lines) + "\n")
    return path


class MeshDynamicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp

            if not cp.cuda.runtime.getDeviceCount():
                raise unittest.SkipTest("CUDA unavailable")
        except (ImportError, RuntimeError) as e:
            raise unittest.SkipTest(str(e)) from e
        from nurbscontact_gpu.mesh_dynamics import MeshDynamics

        cls.driver = staticmethod(MeshDynamics)

    def model(self, meshes, **options):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = make_scene(Path(temporary.name), meshes, **options)
        return self.driver(path, 2000, capacity=128, blocks=4)

    def placements(self, z=0.0):
        p = np.zeros((2, 7))
        p[:, 5] = 1
        p[1, 2] = z
        return p

    def test_tip_plane_separation_and_body_order(self):
        model = self.model([box(), tip()])
        poses = self.placements(0.98)
        r = model.probe_world(poses)[0]
        self.assertEqual(len(r), 1)
        np.testing.assert_allclose(r[0, 3:], [0, 0, 1, 0.02], atol=1e-12)
        poses[1, 2] = 1.02
        self.assertEqual(len(model.probe_world(poses)[0]), 0)
        reverse = self.model([tip(), box()])
        poses = self.placements()
        poses[0, 2] = 0.98
        s = reverse.probe_world(poses)[0]
        self.assertEqual(len(s), 1)
        np.testing.assert_allclose(s[0, 3:], [0, 0, -1, 0.02], atol=1e-12)
        np.testing.assert_allclose(s[0, :3], r[0, :3], atol=1e-12)

    def test_crossed_edges_in_a_spatial_frame(self):
        a = tetra([[-1, 0, 0], [1, 0, 0], [0, -0.4, -1], [0, 0.4, -1]])
        b = tetra([[0, -1, 0], [0, 1, 0], [-0.4, 0, 1], [0.4, 0, 1]])
        model = self.model([a, b])
        p = self.placements(-0.02)
        r = model.probe_world(p)[0]
        self.assertEqual(len(r), 1)
        np.testing.assert_allclose(r[0], [0, 0, -0.01, 0, 0, 1, 0.02], atol=1e-11)
        axis = np.array([0.3, -0.4, 0.5])
        axis /= np.linalg.norm(axis)
        angle = 0.73
        from scipy.spatial.transform import Rotation

        rotation = Rotation.from_rotvec(axis * angle).as_matrix()
        shift = np.array([0.7, -0.4, 0.2])
        p[:, :3] = p[:, :3] @ rotation.T + shift
        p[:, 3:6] = axis
        p[:, 6] = angle
        s = model.probe_world(p)[0]
        expected = r.copy()
        expected[:, :3] = r[:, :3] @ rotation.T + shift
        expected[:, 3:6] = r[:, 3:6] @ rotation.T
        np.testing.assert_allclose(s, expected, atol=1e-10)

    def test_disconnected_supports_and_enclosure(self):
        v, f = tip()
        twin = (np.vstack([v, v + [-0.4, 0, 0]]), np.vstack([f, f + len(v)]))
        model = self.model([box(), twin])
        r = model.probe_world(self.placements(0.98))[0]
        self.assertEqual(len(r), 2)
        np.testing.assert_allclose(r[:, 6], 0.02, atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(r[1, :3] - r[0, :3]), 0.4, places=10)
        self.assertEqual(len(model.probe_world(self.placements(1.0))[0]), 0)
        with self.assertRaisesRegex(RuntimeError, "25.0"):
            model.probe_world(self.placements(0.4))

    def test_filtered_plane_signs_against_decimal(self):
        import cupy as cp
        from decimal import Decimal, localcontext
        from nurbscontact_gpu.mesh_dynamics import module

        rng = np.random.default_rng(872)
        triangles = rng.uniform(-0.05, 0.05, (400, 3, 3))
        points = triangles.mean(axis=1)
        # Near-coplanar input, including exact shared vertices and large offsets.
        points[::3] = triangles[::3, 0]
        points[1::3, 2] = np.nextafter(points[1::3, 2], np.inf)
        triangles[::7] += 10
        points[::7] += 10
        expected = []
        with localcontext() as context:
            context.prec = 100
            for t, p in zip(triangles, points):
                d = [
                    [Decimal(float(x)) - Decimal(float(y)) for x, y in zip(row, p)]
                    for row in t
                ]
                determinant = sum(
                    d[0][j]
                    * (
                        d[1][(j + 1) % 3] * d[2][(j + 2) % 3]
                        - d[1][(j + 2) % 3] * d[2][(j + 1) % 3]
                    )
                    for j in range(3)
                )
                expected.append(1 if determinant > 0 else -1 if determinant < 0 else 0)
        signs = cp.empty(len(points), cp.int32)
        module(1).get_function("mesh_plane_sides")(
            (4,),
            (128,),
            (cp.asarray(triangles), cp.asarray(points), signs, np.int32(len(points))),
        )
        actual = signs.get()
        expected = np.array(expected)
        self.assertTrue(np.all((actual == 0) | (actual == expected)))
        self.assertGreater(np.count_nonzero(actual), 200)

    def test_implicit_unilateral_oscillator(self):
        previous = None
        for execution in ("cooperative", "graph"):
            with self.subTest(execution=execution):
                model = self.model([box(), tip()], initial_q=-0.02, moving=True)
                data, status = model.run(0.0001, execution=execution)
                time = data[:, 0]
                omega = np.sqrt(1000.0)
                release = np.pi / (2 * omega)
                expected = np.where(
                    time <= release,
                    -0.02 * np.cos(omega * time),
                    0.02 * omega * (time - release),
                )
                self.assertLess(np.max(abs(data[:, 1] - expected)), 3e-6)
                self.assertLess(np.max(abs(data[time > release + 0.0002, 6])), 1e-9)
                self.assertEqual(status["status"][0], 0.0)
                if previous is not None:
                    np.testing.assert_allclose(data, previous, rtol=1e-11, atol=1e-11)
                previous = data

    def test_scene_validation(self):
        from nurbscontact_gpu.mesh_dynamics import read_scene

        with tempfile.TemporaryDirectory() as folder:
            path = make_scene(Path(folder), [box(), tip()])
            original = path.read_text()
            for text in (
                original.replace("coordinate 1 ", "coordinate nan "),
                original.replace('body "0.msh" 0 ', 'body "0.msh" 2 '),
                original.replace("surface 0 1", "surface 0 2"),
                original.replace("loads 0", "loads 1\nload 0 1\n0 1"),
                original.replace("loads 0", "loads 1\nload 0 2\n0 1\n0 2"),
            ):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_scene(path)

    def test_overflow_does_not_return_incomplete_contacts(self):
        with tempfile.TemporaryDirectory() as folder:
            path = make_scene(Path(folder), [box(), tip()])
            model = self.driver(path, 10, capacity=1, blocks=1)
            with self.assertRaises(RuntimeError):
                model.probe_world(self.placements(0.98))


if __name__ == "__main__":
    unittest.main()
