"""Integrate shared hinge/slider coordinates with bidirectional GPU contact and HHT."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import cupy as cp
import numpy as np
from resident_contact import ResidentMeshContact
from resident_dynamics import projection_order, swept_samples


def load_inputs(args):
    if args.cam_deck:
        from resident_cad_probes import load_probes

        with np.load(args.cam_deck) as d:
            a, b, fa, fb, rows, loads = (d[k] for k in ("a", "b", "fa", "fb", "rows", "loads"))
            k, h = float(d["stiffness"]), float(d["h"])
        points = load_probes(args.cam_cache, b, fb, 1)[0] if args.cam_cache else None
        return (
            [(a, fa, None), (b, fb, points)],
            rows[:, [9, 10, 11, 12]],
            rows,
            [0, 1],
            [[0, 1]],
            [k],
            loads,
            h,
        )
    models = []
    for entry in json.loads((args.work / "models.json").read_text())["parts"]:
        with np.load(args.work / entry["archive"]) as d:
            points = d["points"] if args.method == "cad" else None
            models.append((d["tri"], d["faces"], points))
    with np.load(args.work / "assembly.npz") as d:
        return (
            models,
            *(d[k] for k in ("coord", "bodies", "mapping", "pairs", "stiffness", "loads")),
            float(d["h"]),
        )


def solve(args):
    start = perf_counter()
    if args.broad_phase is None:
        args.broad_phase = "projection" if args.prefix_contact is not None else "blocks"
    models, coord, bodies, mapping, pairs, stiffness, loads, h = load_inputs(args)
    if args.step_us is not None:
        new_step = args.step_us * 1e-6
        if not np.isfinite(new_step) or new_step <= 0 or args.duration > (len(loads) - 1) * h + 1e-12:
            raise ValueError("Invalid integration interval")
        old = np.arange(len(loads)) * h
        new = np.arange(round(args.duration / new_step) + 1) * new_step
        loads = np.column_stack([np.interp(new, old, column) for column in loads.T])
        h = new_step
    count, n, m = round(args.duration / h) + 1, len(coord), len(pairs)
    if (
        not 1 <= n <= 8
        or not m
        or count > len(loads)
        or count < 2
        or abs((count - 1) * h - args.duration) > 1e-12
        or np.any(coord[:, 0] <= 0)
    ):
        raise ValueError("Unsupported coordinate count, inertia or duration")
    files = Path(__file__).parent
    audit_args = ()
    if (args.audit_window is None) != (args.audit_output is None):
        raise ValueError("Supply both the audit interval and the audit output")
    if args.audit_window is None and (
        args.audit_regions_at is not None or args.prefix_contact or args.audit_samples
    ):
        raise ValueError("Region logging and replay options require an audit window")
    first_audit, last_audit = count, count
    if args.audit_window is not None:
        first_audit, last_audit = (round(t / h) for t in args.audit_window)
        if not 1 <= first_audit <= last_audit < count or not np.allclose(
            np.array([first_audit, last_audit]) * h, args.audit_window, atol=1e-12, rtol=0
        ):
            raise ValueError("Audit window endpoints must be output times")
        capacity = 100000
        history = cp.empty((capacity, 8 + 5 * n + 4 * m), dtype=cp.float64)
        logged = cp.zeros(1, cp.int32)
        if args.audit_regions_at is not None:
            region_step = round(args.audit_regions_at / h)
            if (
                not first_audit <= region_step <= last_audit
                or abs(region_step * h - args.audit_regions_at) > 1e-12
            ):
                raise ValueError("Region audit must be a main step ending in the trial interval")
        else:
            region_step = -1
        region_capacity = 30000 if region_step >= 0 else 1
        region_log = cp.empty((region_capacity, 14))
        region_count = cp.zeros(1, cp.int32)
        audit_args = (
            np.int32(first_audit),
            np.int32(last_audit),
            history,
            logged,
            np.int32(capacity),
            np.int32(region_step),
            region_log,
            region_count,
            np.int32(region_capacity),
        )
    assembly = (files / "resident_assembly.cuh").read_text()
    if args.audit_probe_order is not None:
        assembly = (files / "resident_order_audit.cuh").read_text() + "\n" + assembly
    patch_args = ()
    options = (
        "--std=c++17",
        f"-DQUERY_LANES={args.lanes}",
        "-DNC_TWO_SIDED",
        f"-DASSEMBLY_TICKS={args.ticks}",
    )
    if audit_args:
        options += ("-DNC_ASSEMBLY_AUDIT",)
    if args.stiffness_scope == "pair":
        options += ("-DNC_PAIR_STIFFNESS",)
    if args.broad_phase == "blocks":
        from resident_blocks import block_index

        guard = args.pitch / 4 if args.guard_mm is None else args.guard_mm
        if not np.isfinite(guard) or guard < 0:
            raise ValueError("Expected a finite, nonnegative candidate reuse margin in mm")
        assembly = (files / "resident_blocks.cuh").read_text() + "\n" + assembly
        options += ("-DNC_PATCHES",)
        patch_args = (cp.empty((2 * m, 104), cp.uint8),)
    elif args.guard_mm is not None:
        raise ValueError("The reuse margin applies to the block broad phase")
    broad_phase_data = patch_args[0] if patch_args else cp.zeros(2 * m, cp.uint64)
    profile_args = ()
    if args.profile_phases:
        options += ("-DNC_ASSEMBLY_PROFILE",)
        profile_args = (cp.zeros(5 + 2 * m, cp.uint64),)
    blocks = cp.cuda.Device().attributes["MultiProcessorCount"]
    threads = 128
    search_args = ()
    search_stages = (
        "visited",
        "inside_bounds",
        "occupied_cell",
        "closest_triangle_query",
        "unrestricted_bvh",
        "projection_fallback_bvh",
        "contact_witness",
    )
    if args.count_search_paths:
        options += (f"-DNC_SEARCH_COUNTS={len(search_stages)}",)
        search_args = (cp.zeros((blocks * threads // args.lanes, len(search_stages)), cp.uint64),)
    order_scratch = []
    order_args = ()
    if args.audit_probe_order is not None:
        if not patch_args:
            raise ValueError("Candidate order can only be audited for a cached block list")
        options += ("-DNC_PROBE_ORDER_AUDIT",)
        order_scratch = [None] * (2 * m)

    def compile_module(contact):
        return cp.RawModule(
            code=contact.read_text() + "\n" + assembly, options=options, enable_cooperative_groups=True
        )

    module = compile_module(files / "resident_contact.cu")
    views = cp.empty((2 * m, 320), cp.uint8)
    orders = cp.empty((2 * m, 32), cp.uint8)
    frames = cp.zeros(2 * m * 1024, cp.uint8)
    state = cp.zeros(1024, cp.uint8)
    keep_alive, packed, ids, preferred, probes = [], [], [], [], []
    for p, (ia, ib) in enumerate(pairs):
        forward, reverse = None, None
        for d, (a, b) in ((1, (ib, ia)), (0, (ia, ib))):
            target, fa, _ = models[a]
            source, fb, pts = models[b]
            model = ResidentMeshContact(
                target,
                source,
                fa,
                fb,
                pitch=args.pitch,
                lanes=args.lanes,
                source_points=pts if args.method == "cad" else None,
                merge_planar=True,
                reverse=reverse,
                include_vertices=args.include_vertices,
            )
            select = swept_samples(target, model.samples[0].get(), bodies[[a, b]])
            if select.any() and not select.all():
                idx = cp.asarray(select)
                model.samples = tuple(cp.ascontiguousarray(v[idx]) for v in model.samples)
                model.n = int(select.sum())
            i = 2 * p + d
            if order_scratch:
                # The native and sorted controls use the same binary, buffers and launch geometry.
                axis = {"native": 2, "x": 0, "y": 1, "z": 2}[args.audit_probe_order]
                sorted_probes = np.argsort(model.samples[0][:, axis].get(), kind="stable")
                words = (model.n + 31) // 32
                scratch = cp.zeros(2 * model.n + 2 * words + 1, cp.int32)
                scratch[:model.n] = cp.asarray(sorted_probes, dtype=cp.int32)
                order_scratch[i] = scratch
            module.get_function("bind_mesh")((1,), (1,), (*model.args(), views[i]))
            if patch_args:
                order = block_index(model, guard)
                module.get_function("bind_patches")((1,), (1,), (patch_args[0][i], *order))
            else:
                order = projection_order(model)
                module.get_function("bind_order")((1,), (1,), (orders[i], *order))
            keep_alive.append((model, order))
            if d:
                reverse = model
            else:
                forward = model
        packed.extend(bodies[[ia, ib, ib, ia]])
        ids.extend([mapping[ia], mapping[ib], mapping[ib], mapping[ia]])
        preferred.append(forward.prefer_reverse)
        probes.append([forward.n, reverse.n])
    if order_scratch:
        order_args = (
            cp.asarray([a.data.ptr for a in order_scratch], dtype=cp.uint64),
            np.int32(args.audit_probe_order != "native"),
        )
    device_args = (
        state,
        views,
        frames,
        orders,
        cp.asarray(packed, dtype=cp.float64),
        cp.asarray(ids, dtype=cp.int32),
        cp.asarray(preferred, dtype=cp.int32),
        cp.asarray(np.ascontiguousarray(coord, dtype=np.float64)),
        cp.asarray(stiffness, dtype=cp.float64),
        cp.asarray(np.ascontiguousarray(loads[:count], dtype=np.float64)),
        np.int32(n),
        np.int32(m),
        np.float64(h),
    )
    output = cp.empty((count, 2 * n + 4 * m))
    elapsed = cp.zeros(count, cp.uint64)
    status = cp.zeros(5, cp.int32)
    module.get_function("assembly_start")((1,), (1,), (state, device_args[7], np.int32(n)))
    advance = module.get_function("assembly_advance")
    prefix_advance = None
    if args.prefix_contact:
        prefix_module = compile_module(args.prefix_contact)
        prefix_advance = prefix_module.get_function("assembly_advance")
    get_status = module.get_function("assembly_status")

    def run(stop, samples):
        kernel = prefix_advance if prefix_advance is not None and stop <= first_audit else advance
        kernel(
            (blocks,),
            (threads,),
            (
                *device_args,
                np.int32(stop),
                np.int32(samples),
                np.int32(args.explicit_crossing),
                output,
                elapsed,
                broad_phase_data,
                *profile_args,
                *search_args,
                *order_args,
                *audit_args,
            ),
        )

    run(1, args.contact_samples)
    if profile_args:
        profile_args[0].fill(0)
    if search_args:
        search_args[0].fill(0)
    cp.cuda.get_current_stream().synchronize()
    tic = perf_counter()
    for first in range(1, count, 32):
        stop = min(first + 32, count)
        if (args.audit_samples or prefix_advance is not None) and first < first_audit < stop:
            run(first_audit, args.contact_samples)
        samples = args.audit_samples if args.audit_samples and stop > first_audit else args.contact_samples
        run(stop, samples)
        if first % 512 == 1:
            get_status((1,), (1,), (state, status))
            if status.get()[1] or perf_counter() - start >= 1800:
                break
    cp.cuda.get_current_stream().synchronize()
    wall = perf_counter() - tic
    get_status((1,), (1,), (state, status))
    steps, failure, trials, increments, crossings = map(int, status.get())
    data = output[:steps].get()
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite accepted assembly state")
    np.testing.assert_array_equal(data[0, :n], coord[:, 3])
    times = elapsed[1:steps].get() * 0.001
    report = dict(
        gpu=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(),
        method=args.method,
        broad_phase=args.broad_phase,
        include_vertices=args.include_vertices,
        stiffness_scope=args.stiffness_scope,
        physical_s=(steps - 1) * h,
        requested_s=args.duration,
        step_s=h,
        terminated=("complete" if steps == count and not failure else "newton" if failure else "timeout"),
        coordinates=n,
        pairs=m,
        contact_samples=args.contact_samples,
        minimum_step_s=h / args.ticks,
        explicit_crossing_enabled=args.explicit_crossing,
        explicit_crossings=crossings,
        directional_probes=probes,
        steps=steps - 1,
        trials=trials,
        accepted_increments=increments - 1,
        setup_s=tic - start,
        online_s=wall,
        p99_us=float(np.percentile(times, 99)) if len(times) else None,
        over_budget=int(np.sum(times > h * 1e6)),
        device_pool_MiB=cp.get_default_memory_pool().total_bytes() / 2**20,
        peak_normal_N=data[:, 2 * n :: 4].max(axis=0).tolist(),
        max_active_regions=data[:, 2 * n + 3 :: 4].max(axis=0).tolist(),
    )
    if profile_args:
        counters = profile_args[0].get().tolist()
        report["phase_profile"] = dict(
            prepare_s=counters[0] * 1e-9,
            search_s=counters[1] * 1e-9,
            merge_and_hht_s=counters[2] * 1e-9,
            online_trials=counters[3],
            trials_after_newton_update=counters[4],
            directional_probe_visits=counters[5:],
        )
    if patch_args:
        counts = cp.empty((2 * m, 2), cp.uint64)
        module.get_function("patch_counts")(
            ((2 * m + 127) // 128,), (128,), (frames, broad_phase_data, counts, np.int32(2 * m))
        )
        rebuilds, visits = counts.get().T
        report.update(guard_mm=guard, directional_rebuilds=rebuilds.tolist(), directional_probe_visits=visits.tolist())
    else:
        report["directional_probe_visits"] = broad_phase_data.get().tolist()
    if search_args:
        totals = search_args[0].get().sum(axis=0).tolist()
        report["search_paths"] = dict(zip(search_stages, totals, strict=True))
    if order_scratch:
        if any(int(a[-1].get()) for a in order_scratch):
            raise RuntimeError("A rebuilt list did not preserve its unique candidates")
        report["audit_probe_order"] = args.audit_probe_order
    if failure:
        diagnostic = cp.empty(8)
        module.get_function("assembly_failure")((1,), (1,), (state, diagnostic))
        report["failed_trial"] = dict(
            zip(
                (
                    "code",
                    "tick",
                    "span",
                    "iteration",
                    "halvings",
                    "residual",
                    "frequency_squared",
                    "explicit",
                ),
                (float(x) if np.isfinite(x) else str(x) for x in diagnostic.get()),
                strict=True,
            )
        )
    np.save(args.trace, data)
    if audit_args:
        entries = int(logged.get()[0])
        if entries > capacity:
            raise ValueError("Audit capacity exceeded")
        columns = ["step", "tick", "span", "iteration", "halvings", "residual", "scale", "decision"]
        for label in ("q", "v", "previous_a", "trial_a", "applied"):
            columns.extend(f"{label}{j}" for j in range(n))
        for j in range(m):
            columns.extend(f"{name}{j}" for name in ("normal", "torque_a", "torque_b", "regions"))
        size = int(region_count.get()[0])
        if size > region_capacity:
            raise ValueError("Region audit capacity exceeded")
        metadata = {}
        if region_step >= 0:
            for p in range(m):
                forward = keep_alive[2 * p + 1][0]
                metadata[f"tags_a{p}"] = forward.tags_a
                metadata[f"tags_b{p}"] = forward.tags_b
                metadata[f"forward_n{p}"] = forward.n
        np.savez_compressed(
            args.audit_output,
            trials=history[:entries].get(),
            columns=columns,
            h=h,
            ticks=args.ticks,
            first=first_audit,
            last=last_audit,
            regions=region_log[:size].get(),
            **metadata,
        )
        report.update(audit_trials=entries, audit_samples=args.audit_samples or args.contact_samples)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--cam-deck", type=Path)
    p.add_argument("--cam-cache", type=Path)
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument(
        "--profile-phases",
        action="store_true",
        help="Measure GPU phases at existing grid barriers and count probe visits after interval culling",
    )
    p.add_argument(
        "--count-search-paths",
        action="store_true",
        help="Count the GPU search paths used by online probes; use a separate run for phase timing",
    )
    p.add_argument(
        "--step-us",
        type=float,
        help="Change the integration step, preserving the deck's piecewise-linear loads",
    )
    p.add_argument("--pitch", type=float, default=0.8)
    p.add_argument(
        "--broad-phase", choices=("projection", "blocks"),
        help="Use reusable blocks by default; archived contact-prefix replays default to projection",
    )
    p.add_argument("--guard-mm", type=float, help="Candidate reuse margin in mm (default: one quarter cell)")
    p.add_argument(
        "--audit-probe-order",
        choices=("native", "x", "y", "z"),
        help="Measure the effect of reordering the same block candidates along a body-local axis",
    )
    p.add_argument("--lanes", type=int, choices=(1, 2, 4, 8, 16, 32), default=4)
    p.add_argument(
        "--contact-samples",
        type=int,
        choices=(0, 4, 8, 16, 32),
        default=0,
        help="Resolve every active contact half-period locally (0 refines only observed onset)",
    )
    p.add_argument("--ticks", type=int, choices=(64, 1024, 4096, 65536), default=64)
    p.add_argument(
        "--explicit-crossing",
        action="store_true",
        help="Retry an unresolved minimum interval with a counted velocity-Verlet step",
    )
    p.add_argument("--method", choices=("cad", "mesh"), default="cad")
    p.add_argument(
        "--stiffness-scope",
        choices=("region", "pair"),
        default="region",
        help="Apply the input normal stiffness to each active region or once at the deepest contact of each body pair",
    )
    p.add_argument(
        "--include-vertices",
        action="store_true",
        help="Also test mesh vertices on each incident CAD face when selecting the contact witness",
    )
    p.add_argument("--audit-window", nargs=2, type=float, metavar=("FIRST_END_S", "LAST_END_S"))
    p.add_argument("--audit-output", type=Path)
    p.add_argument(
        "--audit-regions-at",
        type=float,
        help="Record individual witnesses throughout the main step ending at this time",
    )
    p.add_argument(
        "--prefix-contact",
        type=Path,
        help="Use this compatible reference contact kernel up to the first audited step; retain the complete state and caches",
    )
    p.add_argument(
        "--audit-samples",
        type=int,
        choices=(16, 32),
        help="Change local contact resolution from the first audited step onward",
    )
    solve(p.parse_args())
