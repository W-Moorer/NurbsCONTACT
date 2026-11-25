"""CPU reducer replay and distributed pressure audit. Every executable has an 1800 s deadline."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
from threading import Timer
from time import perf_counter

import numpy as np

from _gpu_source import read_model
from cpu_backend_run import digest


def read_rows(path):
    with path.open() as stream:
        yield from map(json.loads, stream)


def checked_run(command, source, target):
    start = perf_counter()
    with source.open() as stdin, target.open("w") as stdout, target.with_suffix(".err").open("w") as stderr:
        subprocess.run(command, stdin=stdin, stdout=stdout, stderr=stderr, timeout=1800, check=True)
    return perf_counter() - start


def timing_windows(stage, poses, args, paths, limit):
    # Include startup, tooth handoffs, peak load and the end of the 5 s trace.
    # Rewarm the cache for twenty poses after every discontinuous seek.
    starts = (0, 280, 10000, 16650, 24950, 33320, 46290, 49900)
    all_poses = poses.read_text().splitlines()
    window = args.work / f"windows-{stage}.poses"
    window.write_text("".join(all_poses[j] + "\n" for start in starts for j in range(start, start + 100)))
    paths_to_mesh = [p.with_suffix(".msh") for p in paths]
    commands = {
        "old": [args.old_exe, *paths, "6", str(2 * limit), "--indexed-groups" if args.old_indexed else "--groups"],
        "indexed": [args.exe, *paths, "6", str(2 * limit), "--indexed-groups"],
        "mesh": [
            args.mesh_exe,
            "--method",
            "mesh-obb",
            "--regions",
            "contour-ray",
            "--axis-angle",
            "--keep-unresolved",
            "--max-depth-m",
            str(limit),
            *paths_to_mesh,
        ],
    }
    # Interleave ten-pose blocks so background load and changes in CPU clock
    # affect all methods. Only one worker process computes at a time.
    positions = window.read_text().splitlines(keepends=True)
    buffers = {name: [] for name in commands}
    with ExitStack() as stack:
        active = {}
        for name, command in commands.items():
            err = stack.enter_context((args.work / f"paired-{name}-{stage}.err").open("w"))
            proc = stack.enter_context(
                subprocess.Popen(
                    command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True
                )
            )
            stack.callback(lambda p=proc: p.kill() if p.poll() is None else None)
            timer = Timer(1800, proc.kill)
            timer.start()
            stack.callback(timer.cancel)
            proc.stdin.write(positions[0])
            proc.stdin.flush()
            json.loads(proc.stdout.readline())  # setup and warmup before paired timing
            active[name] = proc
        names = list(active)
        for j in range(0, len(positions), 10):
            start = (j // 10) % len(names)
            for name in names[start:] + names[:start]:
                proc = active[name]
                proc.stdin.writelines(positions[j : j + 10])
                proc.stdin.flush()
                for _ in range(10):
                    buffers[name].append(json.loads(proc.stdout.readline()))
        for proc in active.values():
            proc.stdin.close()
            if proc.wait(timeout=30):
                raise RuntimeError("window replay failed")
    for name, command in commands.items():
        log = args.work / f"paired-{name}-{stage}.jsonl"
        with log.open("w") as dest:
            for row in buffers[name]:
                print(json.dumps(row, separators=(",", ":")), file=dest)
        log.with_suffix(".json").write_text(
            json.dumps(
                dict(
                    starts=starts,
                    warmup=20,
                    window_length=100,
                    interleave_block=10,
                    executable_sha256=digest(Path(command[0])),
                ),
                indent=2,
            )
        )


def replay(args, stage):
    bodies, pairs = read_model(args.scene)
    a, b, _, limit = pairs[stage]
    paths = [bodies[j][0] for j in (a, b)]
    prefix = args.work / f"{args.action}-{stage}"
    poses = args.work / f"stage-{stage}.poses"
    archive = json.loads((args.data / "summary.json").read_text())
    if digest(args.scene) != archive["scene_sha256"]:
        raise ValueError("Reducer scene differs from the archived baseline")
    if args.action == "prepare":
        if digest(args.data / "trajectory.npz") != archive["trajectory_sha256"]:
            raise ValueError("Pose history differs from the baseline")
        with np.load(args.data / "trajectory.npz") as trace:
            origin, angles = trace["origins_m"], trace["angles_rad"]
            fields = [
                col
                for j in (a, b)
                for col in [
                    *origin[:, j, :].T,
                    *[np.full(len(angles), x) for x in bodies[j][2]],
                    angles[:, j],
                ]
            ]
            np.savetxt(poses, np.column_stack(fields), fmt="%.17g")
        return

    for step in paths:
        for path in (step, step.with_suffix(".msh")):
            if digest(path) != archive["stages"][stage]["geometry_sha256"][path.name]:
                raise ValueError(f"Geometry differs from the archived baseline: {path}")
    if args.action == "windows":
        return timing_windows(stage, poses, args, paths, limit)
    if args.action in ("mesh", "indexed", "old"):
        if args.action == "mesh":
            cmd = [
                args.exe,
                "--method",
                "mesh-obb",
                "--regions",
                "contour-ray",
                "--axis-angle",
                "--keep-unresolved",
                "--max-depth-m",
                str(limit),
                *[p.with_suffix(".msh") for p in paths],
            ]
        else:
            option = "--groups" if args.action == "old" else "--indexed-groups"
            cmd = [args.exe, *paths, "6", str(2 * limit), option]
        source = poses
        count = archive["frames"]
    else:
        reference = json.loads((args.data / f"stage-{stage+1}-audit.json").read_text())["rows"]
        ids = sorted(map(int, reference))
        positions = poses.read_text().splitlines()
        if args.action == "area":
            prefix = args.work / f"area-{args.order}-{stage}"
        source = prefix.with_suffix(".input")
        count = len(ids)
        if args.action == "pressure":
            cmd = [args.exe, *paths, "--pressure-patches", str(2 * limit)]
            source.write_text("".join(positions[j] + "\n" for j in ids))
        else:
            cmd = [args.exe, *paths, "6", str(2 * limit), "--patch-wrench", str(args.order)]
            rows = read_rows(args.work / f"indexed-{stage}.jsonl")
            required = set(ids)
            with source.open("w") as out:
                for j, row in enumerate(rows):
                    if j in required:
                        values = [
                            x
                            for label, w in zip(row["labels"], row["witnesses"], strict=True)
                            for x in (label, *w[:8])
                        ]
                        print(positions[j], len(row["labels"]), *values, file=out)
        prefix.with_suffix(".ids").write_text(json.dumps(ids))
    output = prefix.with_suffix(".jsonl")
    elapsed = checked_run(cmd, source, output)
    if sum(1 for _ in read_rows(output)) != count:
        raise ValueError(f"Interrupted {args.action} replay")
    prefix.with_suffix(".json").write_text(
        json.dumps(
            dict(
                frames=count,
                process_s=elapsed,
                executable_sha256=digest(args.exe),
                scene_sha256=digest(args.scene),
                output_sha256=digest(output),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "mesh", "old", "indexed", "area", "pressure", "windows")
    )
    parser.add_argument("scene", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--data", type=Path, default=Path("results/aes/cpu-local-cad"))
    parser.add_argument("--exe", type=Path)
    parser.add_argument("--old-exe", type=Path)
    parser.add_argument("--old-indexed", action="store_true", help="compare two indexed CAD builds in timing windows")
    parser.add_argument("--mesh-exe", type=Path)
    parser.add_argument(
        "--order", type=int, default=0, help="0 retries unresolved groups at 12, 24, 48 and 96"
    )
    options = parser.parse_args()
    if options.action != "prepare" and options.exe is None:
        parser.error("an executable is required for this action")
    if options.action == "windows" and (options.old_exe is None or options.mesh_exe is None):
        parser.error("windows requires --old-exe and --mesh-exe")
    options.work.mkdir(parents=True, exist_ok=True)
    if options.action == "windows":
        for i in range(2):
            replay(options, i)
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda i: replay(options, i), range(2)))
