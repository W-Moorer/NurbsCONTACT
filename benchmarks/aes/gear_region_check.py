"""Check that coplanar-face merging preserves concurrent spur contact regions on archived poses."""

import argparse
import json
from pathlib import Path

import numpy as np
from contact_tracking_replay import load_tags
from gpu_mesh_line import read_full_mesh
from resident_index import planar_regions


def main(output):
    tracks = Path("results/aes/online-gear/tracking")
    mesh = Path("results/aes/mesh-comparison/meshes")
    maps = []
    for name, archive in (("a", "mesh-0.04.npz"), ("b", "wheel-0.4.npz")):
        tri = read_full_mesh(mesh / archive)
        labels = load_tags(tracks / f"faces-{name}.npz", mesh / archive)
        maps.append(dict(zip(np.unique(labels), planar_regions(tri, labels), strict=True)))
    counts = []
    for name in ("startup", "revolution", "release-small", "hht-small"):
        data = json.loads((tracks / f"{name}-3070.json").read_text())
        frames = {event["frame"]: event["active_pairs"] for event in data["face_events"]}
        active = []
        simultaneous = collisions = 0
        for i in range(data["frames"]):
            active = frames.get(i, active)
            pairs = {(maps[0][a], maps[1][b]) for a, b in active}
            simultaneous += len(active) >= 2
            collisions += len(pairs) < len(active)
        counts.append(
            dict(
                window=name,
                poses=data["frames"],
                multiple_pair_poses=simultaneous,
                poses_with_collapsed_pairs=collisions,
            )
        )
    output.write_text(json.dumps(counts, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    main(p.parse_args().output)
