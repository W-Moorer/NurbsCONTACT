"""Prepare the eccentric cam deck and check its closed-loop motion and contact events."""

import argparse
import json
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp

from gear_precision import mesh_step
from literature_validation import cam_reference, cam_shapes, write_step
from resident_cad_probes import geometry_hash, parametric_probes


def mesh(path, deflection_um):
    tags = []
    vertices, faces = mesh_step(deflection_um, False, step=path, face_ids=tags)
    return vertices[faces], np.concatenate(tags)


def backed_plate(face, vector, path):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.gp import gp_Vec

    write_step(BRepPrimAPI_MakePrism(face, gp_Vec(*vector)).Shape(), path)


def prepare(args):
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer

    root = Path("results/hht/models")
    args.work.mkdir(parents=True, exist_ok=True)
    reader = STEPControl_Reader()
    if reader.ReadFile(str(root / "follower.step")) != IFSelect_RetDone or not reader.TransferRoots():
        raise RuntimeError("Cannot load the follower STEP working face")
    faces = TopExp_Explorer(reader.OneShape(), TopAbs_FACE)
    face = faces.Current()
    faces.Next()
    if faces.More():
        raise ValueError("Expected the single working face of the archived follower")
    plate = args.work / "backed-follower.step"
    backed_plate(face, (0, 50, 0), plate)
    a, fa = mesh(plate, args.deflection_um)
    source = root / "cam.step"
    b, fb = mesh(source, args.deflection_um)
    start = perf_counter()
    points, interior, shift = parametric_probes(source, b, fb, args.deflection_um)
    np.savez_compressed(
        args.work / "cam-probes.npz",
        points=points,
        geometry_sha256=geometry_hash(b, fb),
        step_sha256=sha256(source.read_bytes()).hexdigest(),
        per_edge=1,
        face_probes=interior,
        edge_probes=len(points) - interior,
        max_shift_um=shift,
        projection_s=perf_counter() - start,
    )
    for skin in args.skin_mm:
        for step in args.steps_us:
            h = step * 1.0e-6
            count = round(args.duration / h)
            if abs(count * h - args.duration) > 1.0e-12:
                raise ValueError("Duration must contain whole steps")
            t = np.arange(count + 1) * h
            # The solver evolves corrections about constant-speed nominal coordinates.
            # The follower spring/damper act on absolute travel and speed in the repository scene.
            rows = [
                [1, 0, 1, 0, 0, 115 - skin, 0, 0, 0.096, 0.48, 70, 0.65, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 4, 0.007, 1.8, 0.10, 0],
            ]
            loads = np.column_stack((-3.2 - 70 * 0.096 * t - 0.65 * 0.096, np.zeros_like(t)))
            np.savez_compressed(
                args.work / f"deck-{skin:g}mm-{step:g}us.npz",
                case="eccentric-cam",
                h=h,
                a=a,
                b=b,
                fa=fa,
                fb=fb,
                rows=rows,
                loads=loads,
                stiffness=5000.0,
            )


def reference(duration, skin_mm, max_step):
    # State order: cam angle, cam speed, follower displacement, follower speed (SI).
    def compression(t, state):
        theta, _, q, _ = state
        return 0.105 + 0.024 * np.sin(theta) - (0.115 + q - 0.001 * skin_mm)

    def rhs(t, state):
        angle, omega, q, v = state
        fn = 5000 * max(0.0, compression(t, state))
        return [
            omega,
            (1.8 * (4 * t - angle) + 0.1 * (4 - omega) - fn * 0.024 * np.cos(angle)) / 0.007,
            v,
            (fn - 3.2 - 70 * q - 0.65 * v) / 0.48,
        ]

    sol = solve_ivp(
        rhs,
        (0, duration),
        [0, 4, 0, 0.096],
        method="DOP853",
        rtol=2.0e-12,
        atol=1.0e-14,
        max_step=max_step,
        events=compression,
        dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    state = sol.y_events[0]
    entering = (
        (0.024 * np.cos(state[:, 0]) * state[:, 1] - state[:, 3]) > 0 if len(state) else np.empty(0, bool)
    )
    return sol, entering


def audit(args):
    report = json.loads(args.report.read_text())
    data = np.load(args.trace, allow_pickle=False)
    h = report["step_s"]
    if (
        data.ndim != 2
        or data.shape != (round(args.duration / h) + 1, 8)
        or not np.isfinite(data).all()
        or report["case"] != "eccentric-cam"
        or not report["runs"]
        or not all(x["newton_failure"] == 0 and x["steps"] + 1 == len(data) for x in report["runs"])
    ):
        raise ValueError("Incomplete GPU response")
    t = np.arange(len(data)) * h
    ref, entering = reference(t[-1], args.skin_mm, 0.001)
    coarse, coarse_entering = reference(t[-1], args.skin_mm, 0.002)
    np.testing.assert_array_equal(entering, coarse_entering)
    expected = ref.t_events[0]
    observed = data[:, 4] > 0
    switches = np.flatnonzero(observed[:-1] != observed[1:]) + 1
    q = data[:, 0] + 0.096 * t
    exact = ref.sol(t)
    dq = q - exact[2]
    angles = data[:, 1] + 4 * t
    force = 5000 * np.maximum(
        0.0, 0.105 + 0.024 * np.sin(exact[0]) - (0.115 + exact[2] - 0.001 * args.skin_mm)
    )
    same = bool(len(switches) == len(expected) and np.array_equal(observed[switches], entering))
    events = []
    if same:
        for i, tref, is_entry in zip(switches, expected, entering, strict=True):
            events.append(
                dict(
                    type="engage" if is_entry else "release",
                    reference_s=float(tref),
                    gpu_left_s=t[i - 1],
                    gpu_right_s=t[i],
                    midpoint_error_us=1e6 * ((t[i - 1] + t[i]) * 0.5 - tref),
                    outside_bracket_us=1e6 * max(0.0, t[i - 1] - tref, tref - t[i]),
                )
            )
    baseline = json.loads(args.control_report.read_text())
    control = np.load(args.control_trace, allow_pickle=False)
    if (
        data.shape != control.shape
        or not np.isfinite(control).all()
        or baseline["step_s"] != h
        or baseline["backend"] != "bvh"
        or baseline["case"] != report["case"]
        or not baseline["runs"]
    ) or not all(x["newton_failure"] == 0 and x["steps"] + 1 == len(data) for x in baseline["runs"]):
        raise ValueError("Mismatched or incomplete control")
    delta = data - control
    out = dict(
        physical_s=t[-1],
        step_s=h,
        skin_mm=args.skin_mm,
        position_rms_um=1e6 * np.sqrt(np.mean(dq * dq)),
        position_max_um=1e6 * np.max(np.abs(dq)),
        cam_angle_rms_rad=np.sqrt(np.mean((angles - exact[0]) ** 2)),
        force_rms_N=np.sqrt(np.mean((data[:, 4] - force) ** 2)),
        reference_events=len(expected),
        gpu_events=len(switches),
        same_event_order=same,
        events=events,
        reference_displacement_refinement_max_um=1e6 * np.max(np.abs(coarse.sol(t)[2] - exact[2])),
        reference_event_refinement_max_us=1e6 * np.max(np.abs(coarse.t_events[0] - expected), initial=0.0),
        bvh_max_displacement_difference_um=1e6 * np.max(np.abs(delta[:, 0])),
        bvh_max_force_difference_N=np.max(np.abs(delta[:, 4])),
        bvh_different_contact_states=int(np.sum((control[:, 4] > 0) != observed)),
    )
    if args.mesh_trace:
        pure_mesh = np.load(args.mesh_trace, allow_pickle=False)
        if pure_mesh.shape != data.shape or not np.isfinite(pure_mesh).all():
            raise ValueError("Pure mesh trace differs in shape or contains invalid values")
        active = pure_mesh[:, 4] > 0
        ids = np.flatnonzero(active[:-1] != active[1:]) + 1
        matching = len(ids) == len(expected) and np.array_equal(active[ids], entering)
        mesh_error_um = 1.0e6 * (pure_mesh[:, 0] + 0.096 * t - exact[2])
        out["pure_mesh_bvh"] = dict(
            position_rms_um=float(np.sqrt(np.mean(mesh_error_um**2))),
            position_max_um=float(np.max(np.abs(mesh_error_um))),
            events=len(ids),
            same_event_order=bool(matching),
            max_midpoint_error_us=(
                float(1e6 * np.max(np.abs(t[ids] - 0.5 * h - expected), initial=0)) if matching else None
            ),
        )
    if not same:
        out.update(
            reference_switch_s=expected.tolist(),
            gpu_switch_brackets_s=np.c_[t[switches - 1], t[switches]].tolist(),
        )
    if args.archive:
        hist = np.genfromtxt(
            args.archive, delimiter=",", names=True, usecols=("time", "q1", "solver_peak_force")
        )
        if args.skin_mm != 12 or len(hist) != len(t) or not np.allclose(hist["time"], t, rtol=0, atol=2.0e-9):
            raise ValueError(
                "Archived comparison requires the 12 mm repository model and matching output times"
            )
        out["archived_single_cam"] = dict(
            source=args.archive.name,
            gpu_position_rms_um=float(1e6 * np.sqrt(np.mean((hist["q1"] - q) ** 2))),
            cpu_reference_position_rms_um=float(1e6 * np.sqrt(np.mean((hist["q1"] - exact[2]) ** 2))),
            gpu_force_rms_N=float(np.sqrt(np.mean((hist["solver_peak_force"] - data[:, 4]) ** 2))),
        )
    if args.plot_data:
        stride = max(1, round(0.002 / h))
        np.savez_compressed(
            args.plot_data,
            t=t[::stride],
            gpu_mm=1000 * q[::stride],
            reference_mm=1000 * exact[2, ::stride],
            error_um=1e6 * dq[::stride],
        )
    args.output.write_text(json.dumps(out, indent=2) + "\n")


def kinematics(args):
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.gp import gp_Vec
    from OCP.GeomAbs import GeomAbs_Cylinder
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopoDS import TopoDS
    from nhrep_gear import read_occt_solid
    from resident_contact import ResidentMeshContact

    args.work.mkdir(parents=True, exist_ok=True)
    wire, _, face, x0 = cam_shapes(6, 5, 7, 2.5)
    plate, cam = args.work / "literature-plate.step", args.work / "literature-solid.step"
    backed_plate(face, (3, 0, 0), plate)
    write_step(BRepPrimAPI_MakePrism(BRepBuilderAPI_MakeFace(wire).Face(), gp_Vec(0, 0, 1)).Shape(), cam)
    a, fa = mesh(plate, args.deflection_um)
    b, fb = mesh(cam, args.deflection_um)
    probes, _, _ = parametric_probes(cam, b, fb, args.deflection_um)
    model = ResidentMeshContact(a, b, fa, fb, pitch=0.5, source_points=probes)
    (working,) = np.unique(fa[np.all(np.isclose(a[:, :, 0], x0), axis=1)])
    angles = np.deg2rad(np.arange(3601) / 10)
    prev = b[:, :, 0].max()
    radius = np.linalg.norm(probes, axis=1).max()
    margin = radius * (angles[1] - angles[0]) + 0.002
    support = []
    owners = []
    for ang in angles:
        # The previous measured support and a bound on rigid travel place the
        # plate slightly inside the surface. The witness yields the current support.
        left = prev - margin
        c, s = np.cos(ang), np.sin(ang)
        matrix = np.array([[c, -s, 0, x0 - left], [s, c, 0, 0], [0, 0, 1, 0]])
        active = model.evaluate(matrix)
        hits = [(w[0], source) for (target, source), w in active.items() if target == working]
        if not hits:
            raise RuntimeError(f"No working-face witness at {ang} rad")
        depth, owner = max(hits)
        if depth > 2 * margin + 1.0e-6:
            raise ValueError("Support probe exceeded its motion bound")
        prev = left + depth
        support.append(prev)
        owners.append(owner)
    exact, _, _, switches = cam_reference(angles, 6, 5, 7)
    displacement = np.asarray(support) - 2.5
    err = displacement - exact
    cyclic = np.asarray(owners[:-1])
    idx = np.flatnonzero(cyclic != np.roll(cyclic, 1))
    face_map = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(read_occt_solid(cam), TopAbs_FACE, face_map)
    radii = {}
    for fid in np.unique(owners):
        surface = BRepAdaptor_Surface(TopoDS.Face_s(face_map.FindKey(int(fid))))
        if surface.GetType() != GeomAbs_Cylinder:
            raise ValueError("Unexpected contact on a cam end cap")
        radii[int(fid)] = surface.Cylinder().Radius()
    # Pair switches by the identities of the two arcs, including the 360°/0° seam.
    reference_radii = [2.5, 7.5, 0.5, 6.5, 1.5, 8.5]
    measured_radii = np.array([radii[int(cyclic[i])] for i in idx])
    start = np.flatnonzero(np.isclose(measured_radii, reference_radii[0], rtol=0, atol=1e-9))
    ordered = False
    if len(idx) == 6 and len(start) == 1:
        idx = np.roll(idx, -start[0])
        measured_radii = np.roll(measured_radii, -start[0])
        ordered = np.allclose(measured_radii, reference_radii, rtol=0, atol=1e-9)
    out = dict(
        source="Dhande and Rajaram, Example 1, Tables 1–2",
        mm_per_paper_length_unit=1,
        samples=len(angles),
        angular_step_deg=0.1,
        requested_deflection_um=args.deflection_um,
        source_triangles=len(b),
        source_probes=len(probes),
        position_rms_um=1000 * np.sqrt(np.mean(err**2)),
        position_max_um=1000 * np.max(np.abs(err)),
        original_1deg_grid_max_um=1000 * np.max(np.abs(err[::10])),
        exact_switch_deg=np.rad2deg(switches[:-1]).tolist(),
        observed_switch_brackets_deg=[[float((i - 1) / 10), float(i / 10)] for i in idx],
        entering_face_ids=cyclic[idx].tolist(),
        entering_arc_radii_mm=measured_radii.tolist(),
        expected_branch_radii_mm=reference_radii,
        branches_in_published_order=bool(ordered),
        reference_peak_to_peak_mm=float(np.ptp(exact)),
    )
    if ordered:
        diffs = (idx - 0.5) / 10 - np.rad2deg(switches[:-1])
        out["switch_midpoint_error_deg"] = ((diffs + 180) % 360 - 180).tolist()
    if args.plot_data:
        np.savez_compressed(
            args.plot_data,
            angle_deg=np.rad2deg(angles),
            gpu_mm=displacement,
            exact_mm=exact,
            source_face=np.asarray(owners),
        )
    args.output.write_text(json.dumps(out, indent=2) + "\n")


def refine_events(args):
    """Bisect observed transitions along the cubic Hermite path between saved states."""
    from scipy.spatial.transform import Rotation
    from resident_cad_probes import load_probes
    from resident_contact import ResidentMeshContact

    data = np.load(args.trace, allow_pickle=False)
    with np.load(args.deck, allow_pickle=False) as deck:
        a, b, fa, fb, rows = (deck[k] for k in ("a", "b", "fa", "fb", "rows"))
        h = float(deck["h"])
    points, _ = load_probes(args.probes, b, fb, 1)
    model = ResidentMeshContact(a, b, fa, fb, pitch=4.0, source_points=points)
    checks = json.loads(args.audit.read_text())
    if (
        not checks["same_event_order"]
        or len(data) != round(checks["physical_s"] / h) + 1
        or not np.isclose(checks["step_s"], h)
    ):
        raise ValueError("Expected a matched physical trajectory and event reference")
    on = data[:, 4] > 0
    indices = np.flatnonzero(on[:-1] != on[1:])

    def active(i, u):
        s0, s1 = data[i], data[i + 1]
        q = (
            (1 - u) ** 2 * (1 + 2 * u) * s0[:2]
            + u**2 * (3 - 2 * u) * s1[:2]
            + h * u * (1 - u) * ((1 - u) * s0[2:4] - u * s1[2:4])
        )
        coordinate = rows[:, 7] + rows[:, 8] * (i + u) * h + q
        hinge = rows[:, 0] == 0
        r = Rotation.from_rotvec(rows[:, 1:4] * (hinge * coordinate)[:, None]).as_matrix()
        origins = rows[:, 4:7] + 1000 * rows[:, 1:4] * ((~hinge) * coordinate)[:, None]
        matrix = np.column_stack((r[0].T @ r[1], r[0].T @ (origins[1] - origins[0])))
        return bool(model.evaluate(matrix))

    events = []
    queries = 0
    start = perf_counter()
    for i, ref in zip(indices, checks["events"], strict=True):
        if on[i] != active(i, 0.0) or on[i + 1] != active(i, 1.0):
            raise ValueError(f"Static probes differ from the closed-loop state at step {i}")
        queries += 2
        lo, hi = 0.0, 1.0
        while (hi - lo) * h > args.tolerance_us * 1e-6:
            mid = (lo + hi) * 0.5
            if active(i, mid) == on[i]:
                lo = mid
            else:
                hi = mid
            queries += 1
        left, right = (i + lo) * h, (i + hi) * h
        events.append(
            dict(
                type=ref["type"],
                left_s=left,
                right_s=right,
                reference_s=ref["reference_s"],
                midpoint_error_us=1e6 * ((left + right) * 0.5 - ref["reference_s"]),
            )
        )
    elapsed = perf_counter() - start
    err = np.array([r["midpoint_error_us"] for r in events])
    out = dict(
        step_s=h,
        bracket_tolerance_us=args.tolerance_us,
        queries=queries,
        offline_wall_s=elapsed,
        event_count=len(events),
        max_abs_error_us=float(np.max(abs(err), initial=0)),
        rms_error_us=float(np.sqrt(np.mean(err**2))) if len(err) else None,
        events=events,
    )
    args.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_subparsers(dest="mode", required=True)
    prep = modes.add_parser("prepare")
    prep.add_argument("work", type=Path)
    prep.add_argument("--duration", type=float, default=20)
    prep.add_argument("--steps-us", type=float, nargs="+", default=[125, 62.5])
    prep.add_argument("--skin-mm", type=float, nargs="+", default=[0, 12])
    prep.add_argument("--deflection-um", type=float, default=1.0)
    check = modes.add_parser("audit")
    for key in ("report", "trace", "control_report", "control_trace", "output"):
        check.add_argument(key, type=Path)
    check.add_argument("--skin-mm", type=float, required=True)
    check.add_argument("--duration", type=float, default=20)
    check.add_argument("--archive", type=Path)
    check.add_argument("--plot-data", type=Path)
    check.add_argument("--mesh-trace", type=Path)
    kin = modes.add_parser("kinematics")
    kin.add_argument("work", type=Path)
    kin.add_argument("output", type=Path)
    kin.add_argument("--deflection-um", type=float, default=0.25)
    kin.add_argument("--plot-data", type=Path)
    refine = modes.add_parser("refine-events")
    for key in ("deck", "probes", "trace", "audit", "output"):
        refine.add_argument(key, type=Path)
    refine.add_argument("--tolerance-us", type=float, default=0.1)
    args = p.parse_args()
    if args.mode == "refine-events" and not (np.isfinite(args.tolerance_us) and args.tolerance_us >= 1e-6):
        p.error("Event-bracket tolerance must be finite and at least 1e-6 µs")
    {"prepare": prepare, "audit": audit, "kinematics": kinematics, "refine-events": refine_events}[args.mode](
        args
    )
