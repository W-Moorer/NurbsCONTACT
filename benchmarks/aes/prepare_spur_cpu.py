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


def mesh_settings(helical, profile):
    edge_samples = 96 if helical else 64
    settings = dict(
        size_m=0.016 if helical else 0.03,
        mapped_edge_error_m=0.000010 if helical else 0.000002,
        extend_boundary=False,
        max_normal_deg=8,
    )
    if profile == "reference":
        return dict(settings, curvature_samples=edge_samples, max_error_m=0.00005)
    if profile == "compact":
        return dict(settings, curvature_samples=5 * edge_samples // 8,
                    edge_curvature_samples=edge_samples, max_error_m=0.00016)
    if profile == "light":
        return dict(settings, curvature_samples=20, edge_curvature_samples=edge_samples,
                    max_error_m=0.0005, max_normal_deg=16)
    if profile == "preview":
        return dict(settings, size_m=0.03, mapped_edge_error_m=0.000025,
                    curvature_samples=20, edge_curvature_samples=32,
                    max_error_m=0.0008, max_normal_deg=20)
    raise ValueError(f"unknown mesh profile {profile}")


def main(directory, case="spur", mesh_profile="light"):
    source = ROOT / "results/gears/models"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = f"{case}.csv"
    shutil.copyfile(source / manifest, directory / manifest)
    with (directory / manifest).open(newline="") as stream:
        parts = list(csv.DictReader(stream))
    for part in parts:
        step = directory / part["step"]
        shutil.copyfile(source / part["step"], step)
        helical = float(part["helix_deg"]) != 0
        quality = prepare(step, step.with_suffix(".msh"), **mesh_settings(helical, mesh_profile))
        print(f"{step.stem}: {quality['triangles']} triangles", flush=True)
    gears(directory, directory / f"{case}.scene", 0.36, {"spur": 56, "reducer": 148}[case], surface=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--case", choices=["spur", "reducer"], default="spur")
    parser.add_argument("--mesh-profile", choices=["light", "compact", "reference", "preview"], default="light",
                        help="light preserves mapped contact edges; preview also coarsens the edge geometry")
    args = parser.parse_args()
    main(args.directory, args.case, args.mesh_profile)
