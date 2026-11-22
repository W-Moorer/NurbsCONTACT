"""Build the two-stage reducer from its four full STEP solids and shared shaft coordinates."""

import argparse
import csv
import json
import shlex
from pathlib import Path
from time import perf_counter

import numpy as np
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


def equations(work, h=0.000125):
    """Convert the existing mechanism's loads to corrections about nominal joint motion."""
    lines = iter((work / "reducer.scene").read_text().splitlines())
    next(lines)
    n, nb, npairs = map(int, next(lines).split()[-3:])
    coords = np.array([list(map(float, next(lines).split()[1:])) for _ in range(n)])
    bodies, mapping = [], []
    for _ in range(nb):
        body = shlex.split(next(lines))
        j = int(body[13])
        origin = 1000 * np.array(body[3:6], float)
        bodies.append(
            [
                0,
                *map(float, body[6:9]),
                *origin,
                float(body[9]),
                coords[j, 2],
                coords[j, 0],
                coords[j, 3],
                coords[j, 4],
                coords[j, 1],
            ]
        )
        mapping.append(j)
    pairs, stiffness = [], []
    for _ in range(npairs):
        row = next(lines).split()
        pairs.append([int(row[1]), int(row[2])])
        stiffness.append(float(row[4]))
    if next(lines) != "drives 0":
        raise ValueError("Expected free shafts")
    t = np.arange(round(3 / h) + 1) * h
    loads = coords[:, 6] - coords[:, 4] * coords[:, 2] - t[:, None] * coords[:, 2] * coords[:, 3]
    for _ in range(int(next(lines).split()[1])):
        _, j, count = next(lines).split()
        samples = np.array([list(map(float, next(lines).split())) for _ in range(int(count))])
        loads[:, int(j)] += np.interp(t, samples[:, 0], samples[:, 1])
    np.savez_compressed(
        work / "assembly.npz",
        coord=coords[:, [0, 3, 4, 1]],
        bodies=bodies,
        mapping=mapping,
        pairs=pairs,
        stiffness=stiffness,
        loads=loads,
        h=h,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("--deflection-um", type=float, default=1.0)
    args = p.parse_args()
    prepare(args.work, args.deflection_um)
