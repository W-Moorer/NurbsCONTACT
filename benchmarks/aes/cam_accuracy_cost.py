"""Measure full cam responses against the independent eccentric-cylinder ODE."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from resident_cam import mesh, reference


def prepare_level(base, work, tolerance):
    start = perf_counter()
    with np.load(base / "deck-0mm-125us.npz") as data:
        deck = dict(data)
    source = Path("results/hht/models/cam.step")
    deck["b"], deck["fb"] = mesh(source, tolerance)
    deck["a"], deck["fa"] = mesh(base / "backed-follower.step", tolerance)
    elapsed = perf_counter() - start
    dest = work / f"deck-{tolerance:g}.npz"
    np.savez_compressed(dest, **deck)
    return dest, elapsed


def reference_data(work, count, h):
    path = work / f"reference-{count}-{h:.12g}.npz"
    if not path.exists():
        t = np.arange(count) * h
        sol, entering = reference(t[-1], 0, 0.001)
        np.savez(path, q=sol.sol(t)[2], events=sol.t_events[0], entering=entering)
    with np.load(path) as data:
        return dict(data)


def score(trace, ref, h, follower=0):
    t = np.arange(len(trace)) * h
    error = (trace[:, follower] + 0.096 * t - ref["q"]) * 1e6
    active = trace[:, 4] > 0
    switches = np.flatnonzero(active[1:] != active[:-1]) + 1
    ordered = np.array_equal(active[switches], ref["entering"])
    return dict(
        rms_um=float(np.sqrt(np.mean(error**2))),
        max_um=float(abs(error).max()),
        events=len(switches),
        ordered=ordered,
        max_event_us=(
            float(np.max(abs((t[switches] - 0.5 * h) - ref["events"])) * 1e6)
            if ordered and len(switches)
            else None
        ),
    )


def run_case(
    work,
    name,
    deck,
    ref,
    *,
    probes=None,
    backend="grid",
    pitch=4.0,
    repeats=3,
    follower=0,
    extra=(),
    duration=20.0,
):
    work.mkdir(parents=True, exist_ok=True)
    report, trace = work / f"{name}.json", work / f"{name}.npy"
    with np.load(deck) as data:
        step_us = float(data["h"]) * 1e6
    args = [
        sys.executable,
        str(Path(__file__).with_name("resident_dynamics.py")),
        str(report),
        "--trace",
        str(trace),
        "--duration",
        str(duration),
        "--step-us",
        str(step_us),
        "--deck",
        str(deck),
        "--chunk",
        "64",
        "--impact-samples",
        "32",
        "--backend",
        backend,
        "--pitch",
        str(pitch),
        "--repeats",
        "1",
        *extra,
    ]
    if probes:
        args += ["--cad-probes", str(probes)]
    runs = []
    for _ in range(repeats):
        try:
            subprocess.run(args, check=True, timeout=1800, stdout=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return dict(name=name, status="timeout", limit_s=1800, completed_runs=runs)
        data = json.loads(report.read_text())
        (run,) = data["runs"]
        runs.append(run)
        if run["newton_failure"] or run["steps"] != round(duration / data["step_s"]):
            return dict(name=name, status="incomplete", completed_runs=runs)
    hist = np.load(trace)
    if not np.isfinite(hist).all():
        raise ValueError(f"Nonfinite response in {name}")
    return dict(
        name=name,
        status="complete",
        gpu=data["gpu"],
        step_us=step_us,
        triangles=data["triangles"],
        samples=data["samples"],
        setup_s=data["setup_s"],
        online_s=float(np.median([r["wall_s"] for r in runs])),
        reserved_MiB=max(r["cupy_pool_reserved_bytes"] for r in runs) / 2**20,
        **score(hist, ref, data["step_s"], follower),
        runs=runs,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base", type=Path, default=Path("build/resident-cam"))
    parser.add_argument("--levels", nargs="+", type=float, default=[1, 0.5, 0.25, 0.125, 0.0625])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ref = reference_data(args.work, 160001, 0.000125)
    rows = []
    for level in args.levels:
        deck, prep = prepare_level(args.base, args.work, level)
        for method in ["cad", "mesh"] if level == 1 else ["mesh"]:
            probes = args.base / "cam-probes.npz" if method == "cad" else None
            name = f"{method}-{level:g}"
            result = run_case(args.work, name, deck, ref, probes=probes, repeats=args.repeats)
            result.update(method=method, deflection_um=level, geometry_preparation_s=prep)
            if probes:
                with np.load(probes) as cache:
                    result["cad_projection_s"] = float(cache["projection_s"])
            rows.append(result)
            args.output.write_text(json.dumps(rows, indent=2) + "\n")
            print(name, result["status"], result.get("rms_um"), result.get("online_s"), flush=True)


if __name__ == "__main__":
    main()
