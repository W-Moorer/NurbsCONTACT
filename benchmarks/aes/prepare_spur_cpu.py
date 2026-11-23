"""Prepare Gmsh contact surfaces and one shared scene for an archived gear mechanism."""

import argparse
import csv
from pathlib import Path
import shutil
import sys

from pair_dynamics import gears

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "apps"))
from prepare_contact_mesh import prepare


def main(directory, case="spur"):
    source = ROOT / "results/gears/models"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = f"{case}.csv"
    shutil.copyfile(source / manifest, directory / manifest)
    with (directory / manifest).open(newline="") as stream:
        parts = list(csv.DictReader(stream))
    for part in parts:
        step = directory / part["step"]
        shutil.copyfile(source / part["step"], step)
        # These two Gmsh resolutions describe the benchmark assets. The contact
        # backend only receives the resulting meshes, material and rigid poses.
        helical = float(part["helix_deg"]) != 0
        quality = prepare(
            step,
            step.with_suffix(".msh"),
            0.016 if helical else 0.03,
            max_error_m=0.00005,
            max_normal_deg=8,
            extend_boundary=False,
            mapped_edge_error_m=0.000010 if helical else 0.000002,
            curvature_samples=96 if helical else 64,
        )
        print(f"{step.stem}: {quality['triangles']} triangles", flush=True)
    gears(directory, directory / f"{case}.scene", 0.36, {"spur": 56, "reducer": 148}[case], surface=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--case", choices=["spur", "reducer"], default="spur")
    args = parser.parse_args()
    main(args.directory, args.case)
