"""Run a bounded, full-history CAD GPU execution comparison on a compiled scene."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("scene", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--step", type=float, default=0.001)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--execution", choices=("cooperative", "cached", "batched", "wavefront", "coherent"), default="cooperative")
    parser.add_argument("--reduction", choices=("ordered", "audit"), default="ordered")
    parser.add_argument("--profile-links", action="store_true",
                        help="Collect intrusive connection-stage work counters; not an acceptance timing")
    parser.add_argument("--opt-mask", type=int, default=15)
    args = parser.parse_args()
    if not (math.isfinite(args.step) and math.isfinite(args.duration)
            and args.step > 0 and args.duration > 0):
        raise ValueError("Positive duration and step required")
    steps = round(args.duration / args.step)
    if abs(steps * args.step - args.duration) > 1e-10:
        raise ValueError("Duration must be an integer number of main steps")
    args.out.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.executable), str(args.scene), "0", str(args.step), str(args.duration),
        "1", str(args.out / "response"), "--method", "cad-compiled", "--device", "cuda-resident",
        "--solver", "penalty", "--penalty-stiffness", "shared", "--geometry-update", "predictor",
        "--spectral-radius", "0", "--region-reduction", args.reduction,
        "--gpu-execution", args.execution,
    ]
    if args.execution == "coherent":
        if not 0 <= args.opt_mask <= 255:
            raise ValueError("Optimization mask must be 0..255")
        command.extend(("--gpu-opt-mask", str(args.opt_mask)))
    if args.profile_links:
        command.extend(("--gpu-profile", "on"))
    executable_hash = hashlib.sha256(args.executable.read_bytes()).hexdigest()
    started = perf_counter()
    with (args.out / "console.log").open("w") as log:
        try:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    report = dict(
        command=command, exit_code=code, process_s=perf_counter() - started,
        requested_duration_s=args.duration, step_s=args.step, completed=False,
        executable_sha256=executable_hash,
        scene_sha256=hashlib.sha256(args.scene.read_bytes()).hexdigest(),
        profiled_links=args.profile_links,
    )
    if hashlib.sha256(args.executable.read_bytes()).hexdigest() != executable_hash:
        raise RuntimeError("Executable changed during the measured run")
    if code == 0:
        for name, key in (("gpu", "gpu"), ("stats", "statistics")):
            with (args.out / f"response.{name}.csv").open(newline="") as stream:
                report[key] = next(csv.DictReader(stream))
        with (args.out / "response.csv").open(newline="") as stream:
            rows = csv.DictReader(stream)
            count, last = 0, None
            for last in rows:
                count += 1
        report["frames"] = count
        report["completed"] = count == steps + 1 and abs(float(last["time_s"]) - args.duration) < 1e-8
        report["response_sha256"] = hashlib.sha256((args.out / "response.csv").read_bytes()).hexdigest()
    (args.out / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "statistics"}, indent=2), flush=True)
    if not report["completed"]:
        print((args.out / "console.log").read_text()[-3000:], flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
