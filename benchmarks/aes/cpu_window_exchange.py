"""Cross local integration windows on one reducer pose and test seed selection on new poses."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import tempfile

from cpu_backend_run import digest
from cpu_boundary_replay import replay
from cpu_cad_pressure_report import wrench


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def probe(exe, assets, stage, previous, target, *, donor=None, order=None, current_seed=False):
    poses = dict(zip(stage["steps"], stage["poses"], strict=True))
    this_pose = poses[target]
    lines = [" ".join(format(x, ".17g") for x in poses[step]) for step in (previous, target)]
    files = [assets / (name + ".ncpart") for name in stage["parts"]]
    stiffness, limit = stage["stiffness_N_m"], stage["limit_m"]
    env = {k: v for k, v in os.environ.items() if not k.startswith("NC_DIAG_")}
    if donor is not None:
        env["NC_DIAG_DONOR"] = str(donor)
    if order is not None:
        env["NC_DIAG_ORDER"] = str(order)
    if current_seed:
        env["NC_DIAG_CURRENT_SEED"] = "1"
    run = subprocess.run([exe, *files, str(limit), "--trim-boundary", "adaptive", str(stiffness),
                          "1", "10", "0.5"], input="\n".join(lines) + "\n", text=True, capture_output=True,
                         env=env, timeout=1800)
    if run.returncode:
        raise RuntimeError(run.stderr)
    prior, row = map(json.loads, run.stdout.splitlines())
    if not prior["resolved"]:
        raise ValueError(f"Unresolved predecessor: {previous}")
    windows, orders = {}, {}
    work = None
    for line in run.stderr.splitlines():
        fields = line.split()
        if fields[:1] == ["window"]:
            _, group, side, face, *bounds, accepted = fields
            if group in orders and orders[group] != int(accepted):
                raise ValueError("Conflicting per-region orders")
            orders[group] = int(accepted)
            key = (group, int(side), int(face))
            if key in windows:
                raise ValueError("Duplicate window")
            windows[key] = list(map(float, bounds))
        if fields[:1] == ["work"]:
            work = dict(strip_rays=int(fields[1]), witness_trials=int(fields[2]), rules=row["rules"])
    result = dict(stage=stage["stage"], previous=previous, target=target, current_seed=current_seed,
                  order=order, resolved=row["resolved"])
    if not row["resolved"]:
        return dict(result, diagnostic=run.stderr)
    if work is None or len(orders) != len(row["regions"]):
        raise ValueError("Incomplete diagnostics; build the instrumented probe")
    load = wrench(row["regions"], this_pose[7:10], this_pose[10:13], stiffness, limit, distributed=True)
    if load is None:
        raise ValueError("Invalid load")
    return dict(result, group_faces=row["group_faces"], wrench=load.tolist(), work=work, orders=orders,
                windows=[[group, side, face, *uv] for (group, side, face), uv in sorted(windows.items())])


def save_windows(path, windows):
    path.write_text("".join(" ".join(map(str, w)) + "\n" for w in windows), encoding="ascii")


def swap_one_group(exe, assets, stage, native, temp):
    first, second = native
    a, b = first["windows"], second["windows"]
    if [w[:3] for w in a] != [w[:3] for w in b]:
        raise ValueError("Window keys differ between histories")
    groups = sorted({u[0] for u, v in zip(a, b, strict=True) if u[3:] != v[3:]})
    results = []
    for i, group in enumerate(groups):
        path = temp / f"one-group-{i}.txt"
        save_windows(path, [v if u[0] == group else u for u, v in zip(a, b, strict=True)])
        result = probe(exe, assets, stage, first["previous"], 16, donor=path, order=12)
        results.append(dict(result, replaced_group=group, donor_previous=second["previous"]))
    return results


def cross(exe, assets, archive, workers):
    stage = next(s for s in read(archive / "poses.json")["stages"] if s["stage"] == 2)
    previous_steps = (15, 17)
    with tempfile.TemporaryDirectory() as temp, ThreadPoolExecutor(workers) as pool:
        native = list(pool.map(lambda p: probe(exe, assets, stage, p, 16), previous_steps))
        files = {}
        for previous, row in zip(previous_steps, native, strict=True):
            path = Path(temp) / f"windows-{previous}.txt"
            save_windows(path, row["windows"])
            files[previous] = path
        def cell(spec):
            recipient, source, order = spec
            row = probe(exe, assets, stage, recipient, 16, donor=files[source], order=order)
            return dict(row, donor_previous=source)
        cells = list(pool.map(cell, [(r, s, o) for o in (12, None) for r in previous_steps for s in previous_steps]))
        fresh = list(pool.map(lambda p: probe(exe, assets, stage, p, 16, current_seed=True), previous_steps))
        single = swap_one_group(exe, assets, stage, native, Path(temp))
    return dict(native=native, crossed=cells, current_seed=fresh, one_group_at_a_time=single)


def holdout(exe, assets, archive, workers):
    cases = read(archive / "window-exchange/holdout-poses.json")
    def measure(spec):
        stage, target, neighbor, current = spec
        row = probe(exe, assets, stage, target + neighbor, target, current_seed=current)
        row.pop("windows", None)
        return row
    specs = [(stage, target, neighbor, current) for stage in cases["stages"] for target in cases["targets"]
             for neighbor in (-1, 1) for current in (False, True)]
    with ThreadPoolExecutor(workers) as pool:
        return dict(rows=list(pool.map(measure, specs)))


def target_reference(exe, assets, archive, method):
    cases = read(archive / "window-exchange/holdout-poses.json")
    for stage in cases["stages"]:
        poses = dict(zip(stage["steps"], stage["poses"], strict=True))
        stage.update(steps=cases["targets"], poses=[poses[step] for step in cases["targets"]])
    cold = method == "cold"
    return replay(cases, exe, assets, "adaptive" if cold else method, cold=cold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("cross", "holdout", "cold", "step", "step-refined"))
    parser.add_argument("exe", type=Path)
    parser.add_argument("assets", type=Path, help="Compiled parts or original STEP solids")
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent isolated probe processes")
    args = parser.parse_args()
    archive = Path(__file__).resolve().parents[2] / "results/aes/cpu-bezier/boundary"
    if args.experiment.startswith("step"):
        expected = read(archive / "step-reference.json")["assets"]
    else:
        expected = read(archive / "probe-builds.json")["adaptive"]["assets"]
    hashes = {name: digest(args.assets / name) for name in expected}
    if hashes != expected:
        raise ValueError("Parts differ from the recorded baseline")
    if args.experiment in ("cross", "holdout"):
        evaluate = cross if args.experiment == "cross" else holdout
        result = evaluate(args.exe.resolve(), args.assets, archive, args.workers)
    else:
        result = target_reference(args.exe.resolve(), args.assets, archive, args.experiment)
    fixture = archive / ("poses.json" if args.experiment == "cross" else "window-exchange/holdout-poses.json")
    result.update(probe_sha256=digest(args.exe), pose_sha256=digest(fixture), assets_sha256=hashes)
    args.out.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n", encoding="utf-8")
