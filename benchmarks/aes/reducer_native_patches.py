"""Measure exported RecurDyn contact-patch chords against the common fine STEP tessellation."""

import argparse
import json
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation


def row(section, name):
    return re.search(r"^, " + name + r" = ([^\n]*)", section, re.M)[1].strip()


def block(section, name, count, columns, dtype):
    lines = section.split(f", {name} = \n", 1)[1].splitlines()[:count]
    return np.array([line.lstrip(", ").split(",")[:columns] for line in lines], dtype=dtype)


def native_surfaces(work, file):
    raw = file.read_text(encoding="utf-8-sig")
    parts = json.loads((work / "models.json").read_text())["parts"]
    with np.load(work / "assembly.npz") as deck:
        bodies, mapping = deck["bodies"], deck["mapping"]
    for surface in re.split(r"^CSURFACE / ", raw, flags=re.M)[1:]:
        i = int(re.search(r"\.Part(\d+)'", row(surface, "NAME"))[1])
        marker = row(surface, "RM")
        coord = raw.split(f"\nMARKER / {marker}\n", 1)[1].split("\n \n", 1)[0]
        offset = np.fromstring(row(coord, "QP"), sep=",")
        euler = [float(v.strip().removesuffix("D")) for v in row(coord, "REULER").split(",")]
        np.testing.assert_array_equal(euler, np.zeros(3))
        nf, _, nv = map(
            int, re.search(r"NO_PATCH = (\d+) , NO_LINE = (\d+) , NO_NODE = (\d+)", surface).groups()
        )
        faces = block(surface, "PATCH", nf, 3, int) - 1
        nodes = block(surface, "NODE", nv, 3, float) + offset
        body = bodies[i]
        first = np.flatnonzero(mapping == mapping[i])[0]
        nodes[:, 2] -= body[6] - bodies[first, 6]
        rot = Rotation.from_rotvec(body[1:4] * (body[7] + body[12]))
        nodes = rot.inv().apply(nodes)
        yield i, parts[i], nodes, faces


def measure(work, file):
    import igl

    results = []
    for i, part, nodes, faces in native_surfaces(work, file):
        nf, nv = len(faces), len(nodes)
        with np.load(work / part["archive"]) as d:
            target, inv = np.unique(d["tri"].reshape(-1, 3), axis=0, return_inverse=True)
            indices = inv.reshape(-1, 3)
        sample = np.vstack((nodes, nodes[faces].mean(axis=1)))
        squared, _, _ = igl.point_mesh_squared_distance(sample, target, indices)
        dist = np.sqrt(squared) * 1000
        results.append(
            dict(
                part=i,
                patches=nf,
                reference_triangles=len(indices),
                node_median_um=float(np.median(dist[:nv])),
                node_p99_um=float(np.percentile(dist[:nv], 99)),
                facet_center_median_um=float(np.median(dist[nv:])),
                facet_center_p95_um=float(np.percentile(dist[nv:], 95)),
                facet_center_max_um=float(dist[nv:].max()),
            )
        )
    if sorted(r["part"] for r in results) != list(range(len(json.loads((work/"models.json").read_text())["parts"]))):
        raise ValueError("Unexpected native part count")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("deck", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(measure(args.work, args.deck), indent=2) + "\n")
