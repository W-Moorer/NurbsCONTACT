"""Replace only the native solid-contact triangles in a copied RecurDyn solver deck."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import meshio
import numpy as np
from scipy.spatial.transform import Rotation

from cpu_backend_pose_replay import read_model
from reducer_native_patches import native_surfaces, row


def replace_table(section, name, old_count, rows):
    key = f", {name} = \n"
    prefix, tail = section.split(key, 1)
    lines = tail.splitlines(keepends=True)
    if not all(line.startswith(", ") for line in lines[:old_count]):
        raise ValueError(f"Incomplete {name} table")
    return prefix + key + "".join(rows) + "".join(lines[old_count:])


def write_matched(deck, original, destination):
    parts = json.loads((deck / "models.json").read_text())["parts"]
    paths, _ = read_model(deck / "reducer.scene")
    with np.load(deck / "assembly.npz") as data:
        bodies, mapping = data["bodies"], data["mapping"]
    raw = original.read_text(encoding="utf-8-sig")
    expected = {}
    def surface(match):
        section = match.group()
        i = int(re.search(r"\.Part(\d+)'", row(section, "NAME"))[1])
        path = paths[i][0].with_suffix(".msh")
        if hashlib.sha256(path.read_bytes()).hexdigest() != parts[i]["sha256"]:
            raise ValueError(f"Unexpected mesh {path}")
        mesh = meshio.read(path)
        faces = mesh.get_cells_type("triangle")
        expected[i] = (mesh.points * 1000., faces)
        marker = row(section, "RM")
        coords = raw.split(f"\nMARKER / {marker}\n", 1)[1].split("\n \n", 1)[0]
        offset = np.fromstring(row(coords, "QP"), sep=",")
        body = bodies[i]
        nodes = Rotation.from_rotvec(body[1:4] * (body[7] + body[12])).apply(expected[i][0])
        nodes[:, 2] += body[6] - bodies[np.flatnonzero(mapping == mapping[i])[0], 6]
        nodes -= offset
        limits = np.fromstring(row(section, "MAX"), sep=",")
        if np.any(nodes < 0) or np.any(nodes > limits):
            raise ValueError("Geometry outside native spatial indexing bounds")
        vertices = nodes[faces]
        normals = np.cross(vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths == 0): raise ValueError("Degenerate Gmsh triangle")
        normals /= lengths[:, None]
        counts = r"NO_PATCH = (\d+) , NO_LINE = (\d+) , NO_NODE = (\d+)"
        nf, _, nv = map(int, re.search(counts, section).groups())
        edges = np.sort(faces[:, [[0,1],[1,2],[2,0]]].reshape(-1, 2),axis=1)
        ne = len(np.unique(edges,axis=0))
        section = re.sub(counts, f"NO_PATCH = {len(faces)} , NO_LINE = {ne} , NO_NODE = {len(nodes)}", section, count=1)
        def patch_rows():
            for face, normal in zip(faces, normals, strict=True):
                xyz = ", ".join(format(x, ".17g") for x in normal)
                yield ", " + ", ".join(map(str,face+1)) + ", " + ", ".join([xyz]*4) + "\n"
        section = replace_table(section, "PATCH", nf, patch_rows())
        return replace_table(section, "NODE", nv, (", " + ", ".join(format(x, ".17g") for x in p) + "\n" for p in nodes))
    modified = re.sub(r"^CSURFACE / .*?(?=^ \n|\Z)", surface, raw, flags=re.M|re.S)
    if len(expected) != len(parts): raise ValueError("Incorrect number of contact bodies")
    destination.write_text(modified, encoding="utf-8",newline="\n")
    for i, _, nodes, faces in native_surfaces(deck, destination):
        np.testing.assert_allclose(nodes, expected[i][0], atol=1e-11, rtol=0)
        np.testing.assert_array_equal(faces, expected[i][1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("deck","original","destination"): parser.add_argument(name,type=Path)
    write_matched(**vars(parser.parse_args()))
