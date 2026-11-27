"""Check scene-sized HHT matrices with an independent constant-acceleration solution."""

import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


class ResidentLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp

            if not cp.cuda.runtime.getDeviceCount():
                raise unittest.SkipTest("CUDA unavailable")
        except (ImportError, RuntimeError) as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def test_preloaded_prisms_and_coordinate_counts(self):
        vertices = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
        facets = np.array(
            [
                [0, 1, 3],
                [0, 3, 2],
                [4, 6, 7],
                [4, 7, 5],
                [0, 4, 5],
                [0, 5, 1],
                [2, 3, 7],
                [2, 7, 6],
                [0, 2, 6],
                [0, 6, 4],
                [1, 5, 7],
                [1, 7, 3],
            ]
        )
        script = Path(__file__).resolve().parents[1] / "benchmarks/aes/resident_assembly.py"
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            parts = []
            for i, scale in enumerate(([2.0, 2.0, 1.0], [1.0, 1.0, 1.0])):
                filename = f"{i}.npz"
                np.savez(work / filename, tri=(vertices * scale)[facets], faces=np.arange(12) // 2)
                parts.append(dict(archive=filename))
            (work / "models.json").write_text(json.dumps(dict(parts=parts)))
            bodies = np.zeros((2, 13))
            bodies[:, 3] = [1.0, -1.0]
            bodies[1, 6] = 1.985  # 15 micrometres axial overlap.
            for n in (1, 2, 8):
                with self.subTest(coordinates=n):
                    coord = np.zeros((n, 4))
                    coord[:, 0] = np.linspace(0.4, 1.0, n)
                    load = np.linspace(0.1, 0.3, n)
                    np.savez(
                        work / "assembly.npz",
                        coord=coord,
                        bodies=bodies,
                        mapping=[0, min(1, n - 1)],
                        pairs=[[0, 1]],
                        stiffness=[1.0e6],
                        loads=np.tile(load, (101, 1)),
                        h=0.0001,
                    )
                    completed = subprocess.run(
                        [
                            sys.executable,
                            str(script),
                            str(work),
                            str(work / "run.json"),
                            "--trace",
                            str(work / "run.npy"),
                            "--method",
                            "mesh",
                            "--duration",
                            ".01",
                            "--stiffness-scope",
                            "pair",
                            "--solver-layout",
                            "sized",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=180,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                    report = json.loads((work / "run.json").read_text())
                    self.assertEqual(report["terminated"], "complete")
                    response = np.load(work / "run.npy")
                    t = np.arange(101)[:, None] * 0.0001
                    acceleration = load / coord[:, 0]
                    np.testing.assert_allclose(
                        response[:, :n], 0.5 * t * t * acceleration, atol=1e-12
                    )
                    np.testing.assert_allclose(response[:, n : 2 * n], t * acceleration, atol=1e-10)
                    np.testing.assert_allclose(response[:, 2 * n], 15.0, atol=1e-8)
                    np.testing.assert_allclose(response[:, 2 * n + 1 : 2 * n + 3], 0.0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
