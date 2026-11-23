"""Prepare a long reducer load history and the same Gmsh triangles for CPU and GPU."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import meshio
import numpy as np

from pair_dynamics import gears
from resident_reducer import equations


def prepare(mesh_dir, output, duration, step):
    output.mkdir(parents=True, exist_ok=True)
    gears(mesh_dir, output / "reducer.scene", duration, 148., surface=True)
    equations(output, h=step, duration=duration)
    with (mesh_dir / "reducer.csv").open() as stream:
        parts = list(csv.DictReader(stream))
    manifest = []
    for part in parts:
        path = (mesh_dir / part["step"]).with_suffix(".msh")
        quality = json.loads(path.with_suffix(".quality.json").read_text())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != quality["mesh_sha256"]:
            raise ValueError(f"unexpected mesh: {path}")
        mesh = meshio.read(path)
        faces = mesh.get_cells_type("triangle")
        labels = mesh.get_cell_data("gmsh:geometrical", "triangle")
        if len(faces) != quality["triangles"]:
            raise ValueError(f"triangle count differs from quality report: {path}")
        archive = path.stem + ".npz"
        np.savez_compressed(output / archive, tri=1000 * mesh.points[faces], faces=labels)
        measured={k:quality[k] for k in ("max_sampled_surface_error_m","max_normal_angle_deg","min_sicn","gmsh_version")}
        manifest.append(dict(archive=archive, mesh=path.name, step=part["step"], sha256=digest, triangles=len(faces), **measured))
    (output / "models.json").write_text(json.dumps(dict(parts=manifest), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--duration", type=float, default=5.)
    parser.add_argument("--step", type=float, default=.0001)
    args = parser.parse_args()
    prepare(args.mesh_dir.resolve(), args.output.resolve(), args.duration, args.step)
