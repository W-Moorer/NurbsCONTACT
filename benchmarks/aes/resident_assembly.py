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
            models.append((d["tri"], d["faces"], d["points"]))
    with np.load(args.work / "assembly.npz") as d:
        return (
            models,
            *(d[k] for k in ("coord", "bodies", "mapping", "pairs", "stiffness", "loads")),
            float(d["h"]),
        )


def solve(args):
    start = perf_counter()
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
    code = (files / "resident_contact.cu").read_text() + "\n" + (files / "resident_assembly.cuh").read_text()
    module = cp.RawModule(
        code=code,
        options=(
            "--std=c++17",
            f"-DQUERY_LANES={args.lanes}",
            "-DNC_TWO_SIDED",
            f"-DASSEMBLY_TICKS={args.ticks}",
        ),
        enable_cooperative_groups=True,
    )
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
            )
            select = swept_samples(target, model.samples[0].get(), bodies[[a, b]])
            if select.any() and not select.all():
                idx = cp.asarray(select)
                model.samples = tuple(cp.ascontiguousarray(v[idx]) for v in model.samples)
                model.n = int(select.sum())
            order = projection_order(model)
            i = 2 * p + d
            module.get_function("bind_mesh")((1,), (1,), (*model.args(), views[i]))
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
    blocks = cp.cuda.Device().attributes["MultiProcessorCount"]
    module.get_function("assembly_start")((1,), (1,), (state, device_args[7], np.int32(n)))
    advance = module.get_function("assembly_advance")
    get_status = module.get_function("assembly_status")

    def run(stop):
        advance(
            (blocks,),
            (128,),
            (
                *device_args,
                np.int32(stop),
                np.int32(args.contact_samples),
                np.int32(args.explicit_crossing),
                output,
                elapsed,
            ),
        )

    run(1)
    cp.cuda.get_current_stream().synchronize()
    tic = perf_counter()
    for first in range(1, count, 32):
        run(min(first + 32, count))
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
        "--step-us",
        type=float,
        help="Change the integration step, preserving the deck's piecewise-linear loads",
    )
    p.add_argument("--pitch", type=float, default=0.8)
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
    solve(p.parse_args())
