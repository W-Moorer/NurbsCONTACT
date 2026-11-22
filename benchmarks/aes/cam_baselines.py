"""Compare four contact-force queries in the same published eccentric-cam run."""

import argparse
import csv
import json
from pathlib import Path
import subprocess

from industrial_accuracy import trace
from paper_measurements import measurement


DURATION = 20.
STEP = .000125
BACKENDS = {
    "cylinder-plane": ("brep", 24, 3),
    "occt-extrema": ("occt-distance", 24, 1),
    "triangle-bvh": ("mesh", 2048, 1),
    "polygon-support": ("mesh-support", 2048, 3),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("reference", type=Path, help="Path to the refined cam trace")
    parser.add_argument("output", type=Path)
    parser.add_argument("--reuse-triangle-run", action="store_true",
                        help="Use the existing complete BVH trace and timings in the output directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    models = args.reference.parent / "cam-models"
    ref = trace(args.reference, "cam", DURATION)
    results = {}
    for method, (backend, facets, repeats) in BACKENDS.items():
        prefix = args.output / f"cam-{method}"
        if not (method == "triangle-bvh" and args.reuse_triangle_run):
            subprocess.run([str(args.executable.resolve()), "run", "cam", backend, str(facets), str(STEP),
                            str(DURATION), str(models), str(prefix), str(repeats)], check=True)
        with prefix.with_suffix(".stats.csv").open(newline="", encoding="utf-8") as stream:
            records = list(csv.DictReader(stream))
            if any(row["scene"] != "cam" or row["backend"] != backend or int(row["facets"]) != facets
                   or row["force_strategy"] != "fixed" for row in records):
                raise ValueError(f"Unexpected backend or geometry in {method}")
        run = measurement(prefix, "cam", DURATION, repeats, ref)
        if run["step_s"] != STEP or run["steps"] != round(DURATION / STEP):
            raise ValueError(f"Unexpected time step or count in {method}")
        errors, = run["responses"]
        run["meets_target"] = bool(errors["position_rms_um"] <= .5 and errors["force_rms_pct"] <= 2.)
        results[method] = {"backend": backend, "polygon_sides": facets if backend.startswith("mesh") else None,
                           "geometry_queries": [int(row["queries"]) for row in records],
                           **run}
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
