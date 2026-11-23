"""Run one two-joint rigid system with device-resident contact and implicit HHT."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import cupy as cp
import numpy as np
from contact_tracking_replay import load_tags
from gpu_mesh_line import read_full_mesh
from resident_contact import ResidentMeshContact


def spur(args):
    from gpu_online_spur import scene

    _, coordinates, _, signs, _, _, _ = scene(Path("results/gears/models").resolve(), args.duration)
    with open("results/gears/models/spur.csv", newline="") as stream:
        parts = list(csv.DictReader(stream))
    rows = []
    for i, (part, coord) in enumerate(zip(parts, coordinates, strict=True)):
        mass, q, speed, k, damp, _, _ = coord
        rows.append(
            [
                0,
                0,
                0,
                signs[i],
                float(part["center_x_mm"]),
                0,
                float(part["z_mm"]),
                signs[i] * float(part["angle_zero_rad"]),
                speed,
                mass,
                k,
                damp,
                q,
            ]
        )
    t = np.arange(round(args.duration / args.h) + 1) * args.h
    loads = np.column_stack(
        (
            28 * (1 + 0.1 * np.sin(2 * np.pi * 23 * t)),
            -56
            * (
                1
                + 0.26 * 0.5 * (1 + np.tanh((t - 0.1) / 0.018))
                - 0.34 * 0.5 * (1 + np.tanh((t - 0.24) / 0.028))
                + 0.06 * np.sin(2 * np.pi * 13 * t)
            ),
        )
    )
    folder = Path("results/aes/mesh-comparison/meshes")
    tracking = Path("results/aes/online-gear/tracking")
    ma, mb = folder / "mesh-0.04.npz", folder / "wheel-0.4.npz"
    return (
        read_full_mesh(ma),
        read_full_mesh(mb),
        load_tags(tracking / "faces-a.npz", ma),
        load_tags(tracking / "faces-b.npz", mb),
        rows,
        loads,
        1.6e8,
    )


def spheres(args):
    from gear_precision import mesh_step

    labels = []
    v, f = mesh_step(args.sphere_um, False, step=Path("results/hht/models/ball.step"), face_ids=labels)
    tri = v[f]
    tags = np.concatenate(labels)
    valid = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) > 1e-15
    tri, tags = tri[valid], tags[valid]
    # Repository two-ball scene, with its original centres and zero-gap loading.
    # Nominal translation supplies the two initial velocities.
    rows = [[1, 1, 0, 0, 0, 0, 0, 0, 0.225, 1, 0, 0, 0], [1, 1, 0, 0, 140, 0, 0, 0, -0.225, 1, 0, 0, 0]]
    return tri, tri, tags, tags, rows, np.zeros((round(args.duration / args.h) + 1, 2)), 5000.0


def swept_samples(tri, points, joints, depth_floor=1.e-10):
    # A point on any rotating B describes a circle. The sphere enclosing the
    # rotating target about its joint is a conservative, trajectory-free bound.
    a, b = np.asarray(joints)
    if a[0] != 0 or b[0] != 0:
        return np.ones(len(points), bool)
    na, nb = a[1:4], b[1:4]
    shift = b[4:7] - a[4:7]
    radial = points @ nb
    circle = shift + radial[:, None] * nb
    axial = circle @ nb
    r = np.linalg.norm(points - radial[:, None] * nb, axis=1)
    rho = np.linalg.norm(circle - axial[:, None] * nb, axis=1)
    bound = np.linalg.norm(tri.reshape(-1, 3), axis=1).max()
    keep = axial**2 + (rho - r) ** 2 <= (bound + 1e-8) ** 2
    # An exactly shared coordinate axis additionally preserves height.
    if np.count_nonzero(na) == 1 and (np.array_equal(na, nb) or np.array_equal(na, -nb)):
        level = (points + shift) @ na
        height = tri.reshape(-1, 3) @ na
        keep &= (level > height.min() + depth_floor) & (level < height.max() - depth_floor)
    return keep


def projection_order(model):
    from itertools import product

    points = model.samples[0].get()
    directions, bins, order = [], [], []
    for v in product((-1, 0, 1), repeat=3):
        if not any(v) or next(x for x in v if x) != 1:
            continue
        axis = np.array(v) / np.linalg.norm(v)
        projections = points @ axis
        lower = projections.min() - 1e-8
        upper = projections.max() + 1e-8
        scale = 256 / (upper - lower)
        ranked = np.argsort(projections, kind="stable")
        cuts = np.searchsorted(projections[ranked], lower + np.arange(257) / scale)
        cuts[0], cuts[-1] = 0, len(points)
        directions.append([*axis, lower, scale])
        order.append(ranked)
        bins.append(cuts)
    tri = model.geometry[0].get().reshape(-1, 3)
    center = (tri.min(axis=0) + tri.max(axis=0)) * 0.5
    radius = np.float64(np.linalg.norm(tri - center, axis=1).max())
    return cp.asarray(directions), cp.asarray(bins, dtype=cp.int32), cp.asarray(order, dtype=cp.int32), radius


def solve(model, rows, loads, stiffness, h, chunk, blocks, guard=0.15, predict=True):
    initialization_start = perf_counter()
    if not cp.cuda.Device().attributes["CooperativeLaunch"]:
        raise RuntimeError("Device does not support cooperative launch")
    advance = model.module.get_function("advance")
    status = model.module.get_function("status")
    trace = cp.empty((len(loads), 8))
    state = cp.zeros(1024, cp.uint8)
    counters = cp.empty(10, cp.int32)
    from cupyx.scipy.ndimage import maximum_filter

    candidate = cp.empty(model.n, cp.int32)
    ip = int(np.ceil(guard * float(model.geometry[6][6].get())))
    halo = maximum_filter(model.geometry[9], size=2 * ip + 1, mode="constant") if guard else model.geometry[9]
    joint, applied = cp.asarray(rows, dtype=cp.float64), cp.asarray(loads, dtype=cp.float64)
    dirs, slices, order, target_radius = projection_order(model)
    elapsed = cp.empty(len(loads), cp.uint64)
    model.reset_history()

    def run(stop):
        advance(
            (blocks,),
            (128,),
            (
                *model.args(),
                joint,
                state,
                applied,
                np.float64(h),
                np.float64(stiffness),
                np.int32(stop),
                np.int32(len(loads)),
                np.int32(predict),
                trace,
                halo,
                candidate,
                np.float64(guard),
                np.float64(model.radius),
                elapsed,
                dirs,
                slices,
                order,
                target_radius,
                *model.reverse_args,
            ),
        )

    stream = cp.cuda.get_current_stream()
    run(1)
    stream.synchronize()
    start = perf_counter()
    initialization_s = start - initialization_start
    for end in range(1, len(loads), chunk):
        run(min(end + chunk, len(loads)))
    stream.synchronize()
    wall = perf_counter() - start
    status((1,), (1,), (state, counters))
    steps, trials, fail, max_iter, size, rebuilds, max_samples, splits, deepest, microsteps = map(
        int, counters.get()
    )
    if size > state.nbytes:
        raise RuntimeError("State buffer too small")
    data = trace[:steps].get()
    observed = elapsed[1:steps].get() * 0.001
    impact = {}
    if model.impact_samples:
        out = cp.empty(4, dtype=cp.float64)
        model.module.get_function("impact_stats")((1,), (1,), (state, out))
        windows, peak, when, first = out.get()
        impact = dict(
            impact_samples_per_half_cycle=model.impact_samples,
            impact_restarts=int(windows),
            substep_peak_N=float(peak),
            substep_peak_s=float(when),
            substep_first_loaded_s=float(first) if peak > 0 else None,
        )
    return (
        dict(
            steps=steps - 1,
            trials=trials,
            newton_failure=fail,
            max_newton=max_iter,
            subdivisions=splits,
            finest_divisor=max(1, deepest),
            accepted_substeps=microsteps - 1,
            rebuilds=rebuilds,
            peak_candidate_samples=max_samples,
            wall_s=wall,
            initialization_s=initialization_s,
            **impact,
            physical_s=(steps - 1) * h,
            us_per_step=wall * 1e6 / max(1, steps - 1),
            step_p95_us=float(np.percentile(observed, 95)) if len(observed) else None,
            step_p99_us=float(np.percentile(observed, 99)) if len(observed) else None,
            step_max_us=float(observed.max()) if len(observed) else None,
            steps_over_budget=int(np.sum(observed > h * 1e6)),
            realtime=bool(fail == 0 and steps == len(loads) and wall <= (steps - 1) * h),
        ),
        data,
    )


def main(args):
    start = perf_counter()
    case = args.case
    if args.deck:
        with np.load(args.deck, allow_pickle=False) as deck:
            if "case" in deck:
                case = str(deck["case"].item())
            a, b, fa, fb, rows, loads = (deck[key] for key in ("a", "b", "fa", "fb", "rows", "loads"))
            k = float(deck["stiffness"])
            if not np.isclose(float(deck["h"]), args.h, rtol=0, atol=1.0e-15):
                raise ValueError("Deck step mismatch")
            if len(loads) < round(args.duration / args.h) + 1:
                raise ValueError("Deck load interval too short")
            loads = loads[: round(args.duration / args.h) + 1]
    else:
        a, b, fa, fb, rows, loads, k = (spur if args.case == "spur" else spheres)(args)
    rows = np.asarray(rows, dtype=float)
    if (
        rows.shape != (2, 13)
        or not np.isfinite(rows).all()
        or not np.isin(rows[:, 0], [0, 1]).all()
        or not np.allclose(np.linalg.norm(rows[:, 1:4], axis=1), 1, rtol=0, atol=1e-12)
        or np.any(rows[:, 9] <= 0)
        or np.any(rows[:, 10:12] < 0)
        or not np.isfinite(k)
        or k <= 0
        or loads.shape != (round(args.duration / args.h) + 1, 2)
        or not np.isfinite(loads).all()
    ):
        raise ValueError("Invalid two-joint deck, loads or contact stiffness")
    points, source_report = None, None
    if args.cad_probes:
        from resident_cad_probes import load_probes

        points, source_report = load_probes(args.cad_probes, b, fb, args.samples)
    reverse = None
    if args.two_sided:
        target_points = None
        if args.target_cad_probes:
            from resident_cad_probes import load_probes

            target_points, _ = load_probes(args.target_cad_probes, a, fa, args.samples)
        reverse = ResidentMeshContact(
            b,
            a,
            fb,
            fa,
            args.pitch,
            args.samples,
            lanes=args.lanes,
            backend=args.backend,
            source_points=target_points,
            merge_planar=args.merge_planar,
            vertex_probes=args.vertex_probes,
        )
    model = ResidentMeshContact(
        a,
        b,
        fa,
        fb,
        args.pitch,
        args.samples,
        lanes=args.lanes,
        backend=args.backend,
        source_points=points,
        impact_samples=args.impact_samples,
        vertex_probes=args.vertex_probes,
        merge_planar=args.merge_planar,
        reverse=reverse,
    )
    original_samples = model.n
    keep = swept_samples(a, model.samples[0].get(), rows)
    if keep.any() and not keep.all():
        model.samples = tuple(cp.ascontiguousarray(v[cp.asarray(keep)]) for v in model.samples)
        model.n = int(keep.sum())
    blocks = args.blocks if args.blocks else cp.cuda.Device().attributes["MultiProcessorCount"]
    setup = perf_counter() - start
    report = dict(
        gpu=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(),
        setup_s=setup,
        step_s=args.h,
        case=case,
        backend=args.backend,
        vertex_probes=args.vertex_probes,
        merge_planar=args.merge_planar,
        two_sided=args.two_sided,
        predictor=not args.no_predict,
        guard_mm=args.guard,
        samples=model.n,
        original_samples=original_samples,
        triangles=[len(a), len(b)],
        pitch_mm=args.pitch,
        blocks=blocks,
        runs=[],
    )
    if source_report:
        report["cad_source"] = source_report
    for _ in range(args.repeats):
        info, data = solve(model, rows, loads, k, args.h, args.chunk, blocks, args.guard, not args.no_predict)
        info.update(
            peak_N=float(data[:, 4].max()),
            first_load_s=next((i * args.h for i in np.flatnonzero(data[:, 4] > 0)), None),
        )
        report["runs"].append(info)
        # Reserved device allocations of the CuPy pool, including cached blocks.
        info["cupy_pool_reserved_bytes"] = cp.get_default_memory_pool().total_bytes()
        if info["newton_failure"]:
            break
    if args.trace:
        np.save(args.trace, data)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument("--duration", type=float, default=2)
    p.add_argument("--step-us", type=float, default=100)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--blocks", type=int, default=0)
    p.add_argument("--lanes", type=int, choices=(1, 2, 4, 8, 16, 32), default=16)
    p.add_argument("--pitch", type=float, default=0.12)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--vertex-probes", action="store_true", help="Use oriented mesh vertices as source probes")
    p.add_argument("--merge-planar", action="store_true", help="Aggregate connected coplanar face labels")
    p.add_argument(
        "--two-sided",
        action="store_true",
        help="Query both surfaces and reduce both witnesses into one force per region pair",
    )
    p.add_argument("--target-cad-probes", type=Path, help="Target-side STEP probe cache for a two-sided run")
    p.add_argument(
        "--deck", type=Path, help="Portable input containing the two meshes, joints and load histories"
    )
    p.add_argument("--case", choices=("spur", "spheres"), default="spur")
    p.add_argument("--cad-probes", type=Path, help="Original trimmed STEP face and edge probe cache")
    p.add_argument("--sphere-um", type=float, default=40)
    p.add_argument("--backend", choices=("grid", "bvh", "sdf"), default="grid")
    p.add_argument("--no-predict", action="store_true")
    p.add_argument(
        "--impact-samples",
        type=int,
        default=0,
        help="Locally resolve each observed impact half-cycle with this many intervals (0 disables)",
    )
    p.add_argument("--guard", type=float, default=0.075)
    p.add_argument("--trace", type=Path)
    args = p.parse_args()
    if args.target_cad_probes and not args.two_sided:
        p.error("--target-cad-probes requires --two-sided")
    args.h = args.step_us * 1e-6
    if (
        min(args.h, args.duration, args.repeats, args.chunk, args.samples, args.pitch) <= 0
        or args.guard < 0
        or args.blocks < 0
        or not np.isclose(args.duration / args.h, round(args.duration / args.h), rtol=0, atol=1e-8)
    ):
        p.error("Expected positive settings and a duration divisible by the step")
    main(args)
