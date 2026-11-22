"""Split a STEP face and reverse cam contact roles without changing the physical scene."""

import argparse
import json
from pathlib import Path

import numpy as np
from cam_accuracy_cost import reference_data, run_case, score
from literature_validation import write_step
from nhrep_gear import read_occt_solid
from resident_cam import mesh
from resident_cad_probes import unify_step


def split_plate(source, dest):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepFeat import BRepFeat_SplitShape
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.gp import gp_Pnt

    solid = read_occt_solid(source)
    it = TopExp_Explorer(solid, TopAbs_FACE)
    while it.More():
        face = TopoDS.Face_s(it.Current())
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, props)
        if abs(props.CentreOfMass().Y()) < 1e-10:
            break
        it.Next()
    if not it.More():
        raise ValueError("Missing follower working face")
    cut = BRepFeat_SplitShape(solid)
    cut.Add(BRepBuilderAPI_MakeEdge(gp_Pnt(-172, 0, 0), gp_Pnt(172, 0, 0)).Edge(), face)
    cut.Build()
    if not cut.IsDone() or not BRepCheck_Analyzer(cut.Shape()).IsValid():
        raise ValueError("Invalid STEP working-face split")
    write_step(cut.Shape(), dest)


def decks(base, work):
    work.mkdir(parents=True, exist_ok=True)
    with np.load(base / "deck-0mm-125us.npz") as data:
        original = dict(data)
    plate = work / "split-plate.step"
    split_plate(base / "backed-follower.step", plate)
    a, fa = mesh(plate, 1.0)
    if len(np.unique(fa)) != len(np.unique(original["fa"])) + 1:
        raise ValueError("Expected exactly one additional face")
    split = work / "split.npz"
    np.savez_compressed(split, **(original | dict(a=a, fa=fa)))
    for stem, source in (("original", base / "backed-follower.step"), ("split", plate)):
        normalized = work / f"unified-{stem}.step"
        unify_step(source, normalized)
        a, fa = mesh(normalized, 1.0)
        np.savez_compressed(work / f"unified-{stem}.npz", **(original | dict(a=a, fa=fa)))
    swapped = work / "swapped.npz"
    np.savez_compressed(
        swapped,
        **(
            original
            | dict(
                a=original["b"],
                b=original["a"],
                fa=original["fb"],
                fb=original["fa"],
                rows=original["rows"][::-1],
                loads=original["loads"][:, ::-1],
            )
        ),
    )
    return split, swapped


def compare_wrenches(work, base):
    from resident_cad_probes import load_probes
    from resident_contact import ResidentMeshContact
    from scipy.spatial.transform import Rotation

    original = np.load(work / "original.npy")
    ids = np.unique(np.r_[np.arange(0, 24001, 100), np.arange(26000, len(original), 2000)])
    t = ids * 0.000125
    angles = t * 4 + original[ids, 1]
    height = 115 + 1000 * (0.096 * t + original[ids, 0])
    rotations = Rotation.from_rotvec(np.column_stack((0 * t, 0 * t, angles))).as_matrix()
    expected = None
    output = {}
    witnesses = {}
    for name, deck, reverse in (
        ("original", base / "deck-0mm-125us.npz", False),
        ("split", work / "split.npz", False),
        ("split-merged", work / "split.npz", False),
        ("reversed", work / "swapped.npz", True),
    ):
        with np.load(deck) as data:
            a, b, fa, fb = (data[key] for key in ("a", "b", "fa", "fb"))
        points = None if reverse else load_probes(base / "cam-probes.npz", b, fb, 1)[0]
        model = ResidentMeshContact(
            a, b, fa, fb, pitch=4, source_points=points, merge_planar=name == "split-merged"
        )
        result = []
        candidates = []
        for r, y in zip(rotations, height, strict=True):
            origin = np.array([0.0, y, 0.0])
            transform = np.column_stack((r.T, r.T @ origin)) if reverse else np.column_stack((r, -origin))
            force, torque = np.zeros(3), np.zeros(3)
            pairs = {}
            for key, w in model.evaluate(transform).items():
                # The cam is the source in forward queries and the target in reverse.
                normal = -r @ w[4:7] if reverse else w[4:7]
                point = r @ w[1:4] if reverse else w[1:4] + origin
                contribution = 5 * w[0] * normal
                force += contribution
                torque += np.cross(0.001 * point, contribution)
                pairs[key[::-1] if reverse else key] = (
                    w[0],
                    np.r_[contribution, np.cross(0.001 * point, contribution)],
                )
            result.append(np.r_[force, torque])
            candidates.append(pairs)
        result = np.asarray(result)
        if expected is None:
            expected = result
        delta = result - expected
        output[name] = dict(
            poses=len(t),
            loaded_poses=int(np.sum(np.linalg.norm(result[:, :3], axis=1) > 0)),
            max_force_vector_change_N=float(np.linalg.norm(delta[:, :3], axis=1).max()),
            max_cam_axial_torque_change_Nm=float(abs(delta[:, 5]).max()),
        )
        witnesses[name] = candidates
    joint = []
    for forward, reverse in zip(witnesses["original"], witnesses["reversed"], strict=True):
        strongest = dict(forward)
        for key, contact in reverse.items():
            if key not in strongest or contact[0] > strongest[key][0]:
                strongest[key] = contact
        joint.append(sum((w[1] for w in strongest.values()), start=np.zeros(6)))
    delta = np.asarray(joint) - expected
    output["two_sided_replay"] = dict(
        poses=len(t),
        max_force_vector_change_N=float(np.linalg.norm(delta[:, :3], axis=1).max()),
        max_cam_axial_torque_change_Nm=float(abs(delta[:, 5]).max()),
    )
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--base", type=Path, default=Path("build/resident-cam"))
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--score-existing", action="store_true", help="Score completed portable GPU runs")
    p.add_argument(
        "--unified-only", action="store_true", help="Run the two equivalent STEP solids after unification"
    )
    args = p.parse_args()
    split, swapped = decks(args.base, args.work)
    if args.prepare_only:
        return
    ref = reference_data(args.work, 160001, 0.000125)
    probes = args.base / "cam-probes.npz"
    rows = []
    cases = (
        ("original", args.base / "deck-0mm-125us.npz", probes, 0, ()),
        ("split", split, probes, 0, ()),
        ("split-merged", split, probes, 0, ("--merge-planar",)),
        ("reversed", swapped, None, 1, ()),
        ("both-forward", args.base / "deck-0mm-125us.npz", probes, 0, ("--two-sided", "--merge-planar")),
        ("both-split", split, probes, 0, ("--two-sided", "--merge-planar")),
        (
            "both-reversed",
            swapped,
            None,
            1,
            ("--two-sided", "--merge-planar", "--target-cad-probes", str(probes)),
        ),
    )
    if args.unified_only:
        cases = [
            (
                f"unified-{stem}",
                args.work / f"unified-{stem}.npz",
                probes,
                0,
                ("--two-sided", "--merge-planar"),
            )
            for stem in ("original", "split")
        ]
    for name, deck, source, j, flags in cases:
        if args.score_existing:
            report = json.loads((args.work / f"{name}.json").read_text())
            (trial,) = report["runs"]
            row = dict(name=name, status="incomplete", gpu=report["gpu"], runs=[trial])
            if trial["steps"] == 160000 and trial["newton_failure"] == 0:
                row.update(
                    status="complete", **score(np.load(args.work / f"{name}.npy"), ref, report["step_s"], j)
                )
        else:
            row = run_case(args.work, name, deck, ref, probes=source, follower=j, extra=flags, repeats=1)
        if row["status"] == "complete":
            hist = np.load(args.work / f"{name}.npy")
            base = np.load(args.work / ("unified-original.npy" if args.unified_only else "original.npy"))
            row["max_displacement_change_um"] = float(np.max(abs(hist[:, j] - base[:, 0])) * 1e6)
            row["max_normal_force_change_N"] = float(np.max(abs(hist[:, 4] - base[:, 4])))
        rows.append(row)
        print(name, row["status"], row.get("rms_um"), flush=True)
    result = dict(dynamics=rows)
    if not args.unified_only:
        result["fixed_poses"] = compare_wrenches(args.work, args.base)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
