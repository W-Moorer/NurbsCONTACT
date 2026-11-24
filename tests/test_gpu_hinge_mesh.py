"""Two independently driven, axially preloaded prisms have a known force and zero shaft torque."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


class HingeDynamicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if cp is None:
            raise unittest.SkipTest("CuPy unavailable")
        try:
            if not cp.cuda.runtime.getDeviceCount():
                raise unittest.SkipTest("CUDA unavailable")
        except cp.cuda.runtime.CUDARuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/aes"))

    def test_axial_preload_and_free_rotation(self):
        from gpu_hinge_mesh import prepare, run
        from itertools import product

        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            points = np.array(list(product((-1.0, 1.0), repeat=3)))
            faces = np.array(
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
            parts = []
            for j, r in enumerate((12.0, 4.0)):
                archive = f"{j}.npz"
                np.savez(work / archive, tri=(points * [r, r, 5.0])[faces])
                parts.append(dict(archive=archive))
            (work / "models.json").write_text(json.dumps(dict(parts=parts)))
            cfg = np.zeros((2, 13))
            cfg[:, 3] = [1, -1]
            cfg[1, 6] = 9.985  # 15 micrometres overlap along the shaft axis.
            cfg[:, 8] = [0.6, 0.3]
            cfg[:, 12] = [0.12, -0.046]
            coord = np.zeros((2, 4))
            coord[:, 0] = [0.4, 0.3]
            coord[:, 3] = cfg[:, 12]
            loads = np.tile([1.6, 0.6], (50001, 1))
            h = 0.0001
            np.savez(
                work / "assembly.npz",
                coord=coord,
                bodies=cfg,
                mapping=[0, 1],
                pairs=[[0, 1]],
                stiffness=[1.0e6],
                loads=loads,
                h=h,
            )
            table = np.zeros((1, 8, 5, 3))
            table[:, :, :, 0] = 0.000015
            np.savez(
                work / "geometry.npz",
                joint_poses=cfg[:, 1:8],
                joint_map=[0, 1],
                pair_ids=[[0, 1]],
                table=table,
                period=[4 * np.pi],
                ratio=[0.5],
                origin=[-0.106],
                slip=np.linspace(-0.0024, 0.0008, 5),
                prep_s=0.0,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(work, work, 16, 0.0025)
                run(work, work, work / "run", "mesh-wrench")
            with np.load(work / "run/response.npz") as trace:
                t = trace["time_s"]
                np.testing.assert_allclose(trace["stage_normal_load_N"], 15, atol=1e-12)
                np.testing.assert_allclose(trace["joint_contact_torque_Nm"], 0, atol=1e-12)
                np.testing.assert_allclose(
                    trace["q"],
                    cfg[:, 12] + t[:, None] * cfg[:, 8] + t[:, None] ** 2 * [2, 1],
                    atol=1e-8,
                )
                np.testing.assert_allclose(
                    trace["speed_rad_s"], (cfg[:, 8] + t[:, None] * [4, 2]) * cfg[:, 3], atol=1e-8
                )


if __name__ == "__main__":
    unittest.main()
