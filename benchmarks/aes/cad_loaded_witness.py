"""Recheck probes on a recorded load-bearing face pair against both original STEP shells."""

import argparse
import json
from pathlib import Path

import numpy as np
from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.TopAbs import TopAbs_IN, TopAbs_SHELL
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS_Compound
from OCP.gp import gp_Pnt
from scipy.spatial.transform import Rotation

from nhrep_gear import read_occt_solid
from resident_contact import surface_samples
from resident_index import planar_regions


class NativePart:
    def __init__(self, entry, deck, sources):
        self.solid = read_occt_solid(sources / entry["step"])
        self.side = BRepClass3d_SolidClassifier(self.solid)
        shell = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(shell)
        it = TopExp_Explorer(self.solid, TopAbs_SHELL)
        while it.More():
            builder.Add(shell, it.Current())
            it.Next()
        self.distance = BRepExtrema_DistShapeShape()
        self.distance.LoadS2(shell)
        self.distance.SetDeflection(1.0e-9)
        with np.load(deck / entry["archive"]) as z:
            tri, faces, shift, self.points = (z[k] for k in ("tri", "faces", "shift", "points"))
        _, self.normals, left, self.other, right, _ = surface_samples(tri, faces, 1, True)
        labels = planar_regions(tri, faces)
        self.faces = labels[np.unique(faces, return_inverse=True)[1]]
        self.left, self.right = labels[left], labels[right]
        self.hull = np.concatenate((tri, 0.5 * (tri + np.roll(tri, -1, axis=1)) + 2 * shift), axis=1)

    def shell_distance(self, p):
        self.distance.LoadS1(BRepBuilderAPI_MakeVertex(gp_Pnt(*p)).Vertex())
        self.distance.Perform()
        if not self.distance.IsDone():
            raise ValueError("Native shell distance failed")
        return self.distance.Value(), np.array(self.distance.PointOnShape2(1).Coord())


def check(deck, sources, trace, face_file, stage, steps, pad):
    data = np.load(trace)
    with np.load(face_file) as f:
        owners = f["faces"][:, stage]
    with np.load(deck / "assembly.npz") as d:
        n = len(d["coord"])
        pair = d["pairs"][stage]
        rows = d["bodies"][pair]
        coords = d["mapping"][pair]
        stiffness = float(d["stiffness"][stage])
    h = json.loads(trace.with_suffix(".json").read_text())["step_s"]
    entries = json.loads((deck / "models.json").read_text())["parts"]
    parts = [NativePart(entries[i], deck, sources) for i in pair]
    info = []
    for step in steps:
        q = rows[:, 7] + step * h * rows[:, 8] + data[step, coords]
        rot = Rotation.from_rotvec(np.where(rows[:, 0:1] == 0, rows[:, 1:4] * q[:, None], 0.0)).as_matrix()
        pos = rows[:, 4:7] + np.where(rows[:, 0:1] == 1, 1000 * rows[:, 1:4] * q[:, None], 0.0)
        tests = 0
        best = None
        for i, j in ((0, 1), (1, 0)):
            target, source = parts[i], parts[j]
            sel = (source.left == owners[step, j]) | (source.right == owners[step, j])
            hull = target.hull[target.faces == owners[step, i]]
            if not hull.size:
                continue
            points = ((source.points @ rot[j].T) + (pos[j] - pos[i])) @ rot[i]
            sel &= np.all(
                (points >= hull.min(axis=(0, 1)) - pad) & (points <= hull.max(axis=(0, 1)) + pad), axis=1
            )
            for idx in np.flatnonzero(sel):
                p = points[idx]
                tests += 1
                target.side.Perform(gp_Pnt(*p), 1.0e-8)
                if target.side.State() != TopAbs_IN:
                    continue
                depth, near = target.shell_distance(p)
                if depth <= 1.0e-10:
                    continue
                normal = (near - p) / depth
                source_n = np.stack((source.normals[idx], source.other[idx])) @ rot[j].T @ rot[i]
                if np.min(source_n @ normal) >= -1.0e-10:
                    continue
                if best is None or depth > best["native_depth_mm"]:
                    residual, _ = source.shell_distance(source.points[idx])
                    best = dict(
                        source=entries[pair[j]]["step"],
                        source_probe=int(idx),
                        native_depth_mm=depth,
                        source_shell_residual_mm=residual,
                        target_local_mm=p.tolist(),
                        native_normal=normal.tolist(),
                        equivalent_spring_load_N=0.001 * stiffness * depth,
                    )
        info.append(
            dict(
                step=step,
                time_s=step * h,
                load_pair=owners[step].tolist(),
                native_queries=tests,
                patch_load_N=float(data[step, 2 * n + 4 * stage]),
                witness=best,
            )
        )
    return info


def check_saved_points(deck, report):
    import cupy as cp
    from resident_contact import ResidentMeshContact

    with np.load(deck / "assembly.npz") as d:
        pair = d["pairs"][report["stage"]]
        stiffness = d["stiffness"][report["stage"]]
    entries = json.loads((deck / "models.json").read_text())["parts"]
    for i, j in (pair, pair[::-1]):
        selected = [
            s["witness"]
            for s in report["samples"]
            if s["witness"] and s["witness"]["source"] == entries[j]["step"]
        ]
        if not selected:
            continue
        with np.load(deck / entries[i]["archive"]) as z:
            t, f, shift = (z[k] for k in ("tri", "faces", "shift"))
        model = ResidentMeshContact(t, t, f, f, pitch=0.8, lanes=1, backend="patch", patch_shift=shift)
        model.n = 1
        model.tags_b = np.array([0, 1])
        for point in selected:
            n = np.array(point["native_normal"])[None, :]
            model.samples = tuple(
                cp.asarray(a)
                for a in (
                    np.array(point["target_local_mm"])[None, :],
                    n,
                    np.zeros(1, np.int32),
                    -n,
                    np.ones(1, np.int32),
                )
            )
            model.reset_history()
            model.witness[0].fill(0)
            model.evaluate(np.eye(3, 4))
            depth = float(model.witness[0, 0].get())
            point.update(
                same_point_patch_depth_mm=depth, same_point_patch_load_N=float(0.001 * stiffness * depth)
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("deck", "sources", "trace", "faces", "output"):
        p.add_argument(key, type=Path)
    p.add_argument("--stage", type=int, required=True, help="Zero-based model body-pair index")
    p.add_argument("--steps", type=int, nargs="+")
    p.add_argument(
        "--check-saved",
        action="store_true",
        help="Evaluate the curved GPU target at saved native witness coordinates",
    )
    p.add_argument(
        "--pad-mm", type=float, default=0.25, help="Search padding around the recorded target face hull"
    )
    a = p.parse_args()
    if a.check_saved:
        report = json.loads(a.output.read_text())
        if report["stage"] != a.stage:
            p.error("The saved stage differs from the requested stage")
        check_saved_points(a.deck, report)
    else:
        if not a.steps:
            p.error("Supply output steps for native STEP verification")
        out = check(a.deck, a.sources, a.trace, a.faces, a.stage, a.steps, a.pad_mm)
        report = dict(stage=a.stage, pad_mm=a.pad_mm, samples=out)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
