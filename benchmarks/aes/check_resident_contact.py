"""Check gear witnesses against mesh traversal and CPU distance, and arbitrary poses on a box."""

import argparse
import json
from pathlib import Path

import cupy as cp
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from contact_tracking_replay import load_tags
from gpu_mesh_line import read_full_mesh
from resident_contact import ResidentMeshContact, surface_samples


def tilted_boxes():
    box = trimesh.creation.box(extents=[4, 5, 6])
    a = np.ascontiguousarray(box.triangles)
    b = a * [0.6, 0.8, 0.7]
    tags = np.unique(box.face_normals, axis=0, return_inverse=True)[1]
    model = ResidentMeshContact(a, b, tags, tags, pitch=0.3)
    pts, normals, own, other, alt, _ = surface_samples(b, tags, crease_samples=True)
    half = np.array([2.0, 2.5, 3.0])
    max_depth = 0.0
    pairs = 0
    frames = 0
    for u in np.linspace(-1, 1, 33):
        rot = Rotation.from_rotvec([0.83 * u, 0.27 + 0.6 * u, -0.44 * u]).as_matrix()
        shift = np.array([4.9 * u, 0.44 * np.sin(u), 0.32 * u])
        actual = model.evaluate(np.column_stack((rot, shift)))
        points = pts @ rot.T + shift
        distance = half - np.abs(points)
        expected = {}
        for i in np.flatnonzero(distance.min(axis=1) > 1e-10):
            axis = np.argmin(distance[i])
            unit = np.eye(3)[axis] * np.sign(points[i, axis])
            normal_a = int(tags[np.argmax(box.face_normals @ unit)])
            face_dot = np.dot(rot @ normals[i], unit)
            alt_dot = np.dot(rot @ other[i], unit)
            if min(face_dot, alt_dot) >= -1e-10:
                continue
            key = (normal_a, int(alt[i] if alt_dot < face_dot else own[i]))
            expected[key] = max(expected.get(key, 0.0), distance[i, axis])
        if set(actual) != set(expected):
            raise AssertionError(f"Rotated box face IDs differ at pose {u}: {set(actual)^set(expected)}")
        for key, d in expected.items():
            max_depth = max(max_depth, abs(d - actual[key][0]))
        pairs += len(expected)
        frames += 1
    if max_depth > 1e-10:
        raise AssertionError(f"Rotated box depth error {max_depth} mm")
    return dict(poses=frames, cumulative_face_pairs=pairs, max_abs_depth_mm=max_depth)


def gear_audit(root, meshes, pitch, cad_probes=None):
    pa, pb = meshes / "mesh-0.04.npz", meshes / "wheel-0.4.npz"
    a, b = read_full_mesh(pa), read_full_mesh(pb)
    fa, fb = load_tags(root / "faces-a.npz", pa), load_tags(root / "faces-b.npz", pb)
    source = None
    if cad_probes:
        from resident_cad_probes import load_probes

        source, _ = load_probes(cad_probes, b, fb, 1)
    model = ResidentMeshContact(a, b, fa, fb, pitch, source_points=source)
    grid = model.probe
    control = cp.RawModule(
        code=Path(__file__).with_name("resident_contact.cu").read_text(),
        options=("--std=c++17", "-DQUERY_LANES=16", "-DNC_BVH_CONTROL"),
        enable_cooperative_groups=True,
    ).get_function("probe")
    v, f = np.unique(a.reshape(-1, 3), axis=0, return_inverse=True)
    cpu = trimesh.Trimesh(vertices=v, faces=f.reshape(-1, 3), process=False)
    windows = {}
    for window, archive in (
        ("startup", "startup"),
        ("revolution", "revolution"),
        ("release", "release-small"),
        ("hht", "hht-small"),
    ):
        poses = np.loadtxt(root / (window + ".txt"))
        ref = json.loads((root / (archive + "-3070.json")).read_text())
        events = {e["frame"]: set(map(tuple, e["active_pairs"])) for e in ref["face_events"]}
        if len(poses) != ref["frames"]:
            raise ValueError("Mismatched pose archive")
        opp = set()
        missed = extra = presence_miss = presence_extra = hits = changes = 0
        witnesses = 0
        cpu_points = []
        depths = []
        max_d = max_n = 0.0
        examples = []
        for i, row in enumerate(poses):
            matrix = row[1:].reshape(3, 4)
            model.probe = grid
            fast = model.evaluate(matrix)
            model.probe = control
            exact = model.evaluate(matrix)
            changes += len(set(exact) ^ set(fast))
            for key in set(exact) & set(fast):
                w = fast[key]
                max_d = max(max_d, abs(w[0] - exact[key][0]))
                max_n = max(max_n, np.linalg.norm(w[4:7] - exact[key][4:7]))
                witnesses += 1
                if i % max(1, len(poses) // 24) == 0:
                    cpu_points.append(w[1:4])
                    depths.append(w[0])
            opp = events.get(i, opp)
            found = {k for k, w in fast.items() if w[7] < -0.95}
            hits += len(opp)
            missed += len(opp - found)
            extra += len(found - opp)
            presence_miss += bool(opp) and not bool(found)
            presence_extra += bool(found) and not bool(opp)
            if found != opp and len(examples) < 5:
                examples.append(
                    dict(frame=i, time_s=row[0], missed=sorted(opp - found), extra=sorted(found - opp))
                )
        delta = (
            np.abs(trimesh.proximity.signed_distance(cpu, np.asarray(cpu_points)) - depths) if depths else []
        )
        if changes or max_d > 1.0e-9 or max_n > 1.0e-8:
            raise AssertionError(f"{window}: grid/BVH disagree on fixed-pose witnesses")
        windows[window] = dict(
            poses=len(poses),
            matching_sampled_pairs=witnesses,
            bvh_face_set_symmetric_difference=changes,
            bvh_max_depth_diff_mm=max_d,
            bvh_max_unit_normal_diff=float(max_n),
            cpu_witnesses=len(depths),
            cpu_max_signed_depth_diff_mm=float(max(delta, default=0)),
            intersection_opposed_pairs=hits,
            opposed_pair_frame_misses=missed,
            opposed_pair_frame_extras=extra,
            opposed_presence_missed_frames=presence_miss,
            opposed_presence_extra_frames=presence_extra,
            examples=examples,
        )
    return windows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pitch", type=float, default=0.12)
    parser.add_argument("--cad-probes", type=Path)
    args = parser.parse_args()
    checks = {
        "rotated_boxes": tilted_boxes(),
        "gear": gear_audit(
            Path("results/aes/online-gear/tracking"),
            Path("results/aes/mesh-comparison/meshes"),
            args.pitch,
            args.cad_probes,
        ),
    }
    args.output.write_text(json.dumps(checks, indent=2) + "\n")
