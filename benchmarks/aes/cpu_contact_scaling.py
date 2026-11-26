"""Time a Windows CPU reducer run and verify every output row against a frozen response."""
import argparse
import csv
import json
import os
import statistics
import subprocess
from pathlib import Path

from cpu_backend_run import digest
from cpu_process_timing import observer


def measure(exe, scene, reference, out, method, threads, steps):
    out.mkdir(parents=True, exist_ok=False)
    prefix = out / "response"
    args = [exe, scene, "0", ".0001", str(steps / 10000), "1", prefix, "--method", method,
            "--solver", "penalty", "--penalty-stiffness", "shared", "--geometry-update", "predictor",
            "--spectral-radius", "0", "--progress", out / "progress.csv"]
    if method == "mesh-obb":
        args.extend(["--mesh-region-scope", "contour-ray"])
    os.environ["OMP_NUM_THREADS"] = str(threads)
    telemetry = out / "samples.json"
    with (out / "process.log").open("wb") as log:
        completed = observer(telemetry)(args, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    completed.check_returncode()
    with prefix.with_suffix(".stats.csv").open(newline="") as stream:
        stats = next(csv.DictReader(stream))
    if int(stats["steps"]) != steps:
        raise ValueError("Incomplete simulation")
    with reference.open("rb") as ref, prefix.with_suffix(".csv").open("rb") as response:
        for row in range(steps + 2):
            line = response.readline()
            if not line or line != ref.readline():
                raise ValueError(f"Reference mismatch at CSV row {row + 1}")
        if response.readline():
            raise ValueError("Extra response rows")
    samples = [r for r in json.loads(telemetry.read_text()) if 0 < r.get("step", 0) < steps]
    cores = None
    if len(samples) >= 2:
        a, b = samples[0], samples[-1]
        cores = ((b["system_s"] - a["system_s"]) + (b["user_s"] - a["user_s"])) / (b["wall_s"] - a["wall_s"])
    result = dict(method=method, threads=threads, steps=steps, dt_s=0.0001, wall_s=float(stats["wall_s"]),
                  setup_s=float(stats["setup_s"]), sampled_logical_cores=cores,
                  executable_sha256=digest(exe), scene_sha256=digest(scene),
                  response_sha256=digest(prefix.with_suffix(".csv")),
                  matched_reference_rows=steps+1, force_evaluations=int(stats["force_evaluations"]),
                  subdivisions=int(stats["subdivisions"]), triangle_tests=int(stats["triangle_tests"]),
                  cad_queries=int(stats["cad_patch_queries"]))
    result["sample_count_in_simulation"] = len(samples)
    for counter in ("busy_pct", "performance_pct"):
        values = [r[counter] for r in samples if counter in r]
        if values:
            result["host_median_" + counter] = statistics.median(values)
    (out / "timing.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ["exe", "scene", "reference", "out"]:
        parser.add_argument(arg, type=Path)
    parser.add_argument("--method", choices=("mesh-obb", "cad-compiled"), required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--steps", type=int, default=50000)
    args = parser.parse_args()
    if args.threads < 1 or not 1 <= args.steps <= 50000:
        parser.error("Use positive threads and 1 to 50000 steps")
    measure(args.exe.resolve(), args.scene.resolve(), args.reference.resolve(), args.out.resolve(),
            args.method, args.threads, args.steps)
