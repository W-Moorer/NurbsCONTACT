"""Build the two-stage reducer from its four full STEP solids and shared shaft coordinates."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from _gpu_source import equations
from pair_dynamics import gears
from resident_cam import mesh
from resident_cad_probes import parametric_probes
from resident_index import mesh_features


def prepare(work, tolerance):
    work.mkdir(parents=True, exist_ok=True)
    models = Path("results/gears/models").resolve()
    scene = work / "reducer.scene"
    gears(models, scene, 3.0, 148.0)
    with (models / "reducer.csv").open(newline="") as stream:
        parts = list(csv.DictReader(stream))
    files = []
    for part in parts:
        step = models / part["step"]
        start = perf_counter()
        tri, labels = mesh(step, tolerance)
        mesh_features(tri)
        points, _, _ = parametric_probes(step, tri, labels, tolerance)
        dest = work / f"{step.stem}.npz"
        np.savez_compressed(dest, tri=tri, faces=labels, points=points)
        files.append(
            dict(
                step=step.name,
                archive=dest.name,
                triangles=len(tri),
                probes=len(points),
                prep_s=perf_counter() - start,
            )
        )
        print(files[-1], flush=True)
    (work / "models.json").write_text(json.dumps(dict(deflection_um=tolerance, parts=files), indent=2) + "\n")
    equations(work)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("--deflection-um", type=float, default=1.0)
    args = p.parse_args()
    prepare(args.work, args.deflection_um)
