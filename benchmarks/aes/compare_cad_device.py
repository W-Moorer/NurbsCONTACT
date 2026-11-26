"""Run CPU and CUDA contact on the frozen reducer and check every response channel."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("exe", "scene", "reference", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--analyze-existing", action="store_true", help="Analyze two previously completed runs")
    args = parser.parse_args()
    output = args.output.resolve()
    if not args.analyze_existing:
        output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS="4")
    reference = np.genfromtxt(args.reference, delimiter=",", names=True)
    if len(reference) != 50001 or abs(reference["time_s"][-1] - 5) > 1e-9:
        raise ValueError("Expected the complete frozen five-second response")
    results = dict(executable_sha256=digest(args.exe), scene_sha256=digest(args.scene),
                   reference_sha256=digest(args.reference), runs=[])
    for device in ("cpu", "cuda-columns"):
        prefix = output / device
        command = [args.exe, args.scene, "0", "0.0001", "5", "1", prefix, "--method", "cad-compiled",
                   "--device", device, "--solver", "penalty", "--penalty-stiffness", "shared",
                   "--geometry-update", "predictor", "--spectral-radius", "0",
                   "--progress", output / (device + "-progress.csv")]
        if not args.analyze_existing:
            with prefix.with_suffix(".log").open("wb") as stream:
                subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=1800)
        response = prefix.with_suffix(".csv")
        data = np.genfromtxt(response, delimiter=",", names=True)
        if data.shape != reference.shape or data.dtype.names != reference.dtype.names:
            raise ValueError("Response shape or fields differ")
        maximum = {}
        for key in data.dtype.names:
            # The contact reporter writes NaN for the gap of an inactive pair.
            actual, expected = data[key], reference[key]
            valid = np.isfinite(expected)
            if not np.array_equal(np.isfinite(actual), valid) or not np.array_equal(actual[~valid], expected[~valid], equal_nan=True):
                raise ValueError(f"Different non-finite values in {device} channel {key}")
            maximum[key] = float(np.max(np.abs(actual[valid] - expected[valid]), initial=0))
        with prefix.with_suffix(".stats.csv").open() as stream:
            stats = next(csv.DictReader(stream))
        if int(stats["steps"]) != 50000:
            raise ValueError("Incomplete simulation")
        run = dict(device=device, frames=len(data), response_sha256=digest(response), maximum_error=maximum,
                   statistics={k:stats[k] for k in ("wall_s", "setup_s", "steps", "subdivisions", "force_evaluations",
                               "cad_patch_queries", "cuda_column_rays", "cuda_column_subdivisions",
                               "cad_patch_broad_s", "cad_patch_search_s", "cad_patch_group_s", "cad_patch_area_s")})
        results["runs"].append(run)
        (output / "comparison.json").write_text(json.dumps(results, indent=2)+"\n")
        print(device, stats["wall_s"], "s", max(maximum.values()), "maximum channel difference", flush=True)
        if device=="cpu" and run["response_sha256"]!=results["reference_sha256"]:
            raise ValueError("CPU build diverged from the frozen reference")
        if device!="cpu" and int(stats["cuda_column_rays"])==0:
            raise ValueError("GPU path was not exercised")


if __name__ == "__main__":
    main()
