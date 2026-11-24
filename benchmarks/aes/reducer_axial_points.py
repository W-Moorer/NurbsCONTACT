"""Audit exceptionally large native forces against the material surfaces and the shaft moment arm."""
import argparse
import json
import subprocess
from pathlib import Path

import meshio
import numpy as np
from scipy.spatial.transform import Rotation
from trimesh.triangles import closest_point

from cpu_backend_pose_replay import read_model
from reducer_contact_rays import surface_hit


def radial_separation(mesh_exe, bodies, pair, parts, angles, steps):
    a, b = pair
    direction = bodies[b][1] - bodies[a][1]
    direction /= np.linalg.norm(direction)
    distances_um = np.array([0., 25., 50., 100., 200., 400., 800.])
    input_rows = []
    for step in steps:
        blocks = []
        bounds = []
        for i, (vertices, _) in zip(pair, parts, strict=True):
            _, center, axis, initial, coordinate = bodies[i]
            angle = initial + angles[step, coordinate]
            rotation = Rotation.from_rotvec(axis * angle)
            world = rotation.apply(vertices) + center
            bounds.append((world.min(0), world.max(0)))
            blocks.append(np.r_[center, rotation.as_quat()[[3, 0, 1, 2]]])
        for shift in distances_um:
            second = blocks[1].copy()
            delta = shift * 1e-6 * direction
            second[:3] += delta
            # For these connected closed surfaces, two disjoint boundaries could
            # still enclose one another. Each has a vertex outside the other's AABB.
            left, right = np.asarray(bounds[0]), np.asarray(bounds[1]) + delta
            if not ((left[0] < right[0]).any() or (left[1] > right[1]).any()):
                raise ValueError("First body potentially contained in the second")
            if not ((right[0] < left[0]).any() or (right[1] > left[1]).any()):
                raise ValueError("Second body potentially contained in the first")
            input_rows.append(" ".join(format(x, ".17g") for x in np.r_[blocks[0], second]))
    files = [bodies[i][0].with_suffix(".msh") for i in pair]
    run = subprocess.run([mesh_exe, "--method", "mesh-obb", *files],
                         input="\n".join(input_rows) + "\n", capture_output=True, text=True, check=True, timeout=1800)
    counts = np.array([len(json.loads(line)["hits"]) for line in run.stdout.splitlines()])
    counts = counts.reshape(len(steps), len(distances_um))
    if (counts[:, 0] == 0).any():
        raise ValueError("No triangle intersections at a reported high-load pose")
    checks = []
    for i, hits in zip(steps, counts, strict=True):
        clear = distances_um[hits == 0]
        checks.append(dict(output_step=int(i), hits=hits.tolist(),
                           disjoint_translation_bound_um=float(clear[0]) if len(clear) else None))
    return checks, distances_um


def plot_check(path, time, force, original_force, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    mask = time <= report["point_export_end_s"] + 1e-9
    for values, label, color in ((force, "Shared Gmsh facets", "#126965"),
                                 (original_force, "Original RecurDyn facets", "#bd542b")):
        axes[0].plot(1000*time[mask], values[mask]/1000, color=color, lw=1., label=label)
    axes[0].set(yscale="symlog", ylim=(-.2, 500), ylabel="Stage 2 resultant force (kN)",
                xlabel="Time (ms)", title="RecurDyn point-output interval")
    axes[0].set_yticks([0, 1, 10, 100, 400], labels=["0", "1", "10", "100", "400"])
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    shifts = report["radial_translation_trials_um"][:5]
    hits = np.asarray([d["hits"][:len(shifts)] for d in report["separation_checks"]])
    low, mid, high = np.percentile(hits, [0, 50, 100], axis=0)
    axes[1].fill_between(shifts, low, high, alpha=.2, color="#126965", label=f"Range across {len(hits)} poses")
    axes[1].plot(shifts, mid, "o-", color="#126965", lw=1.4, label="Median")
    axes[1].set(title="Same meshes at high-force poses", xlabel="Increase in shaft centre distance (µm)",
                ylabel="Intersecting triangle pairs", xticks=shifts)
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def audit(deck, point_file, matched, original, out, mesh_exe=None):
    bodies, pairs = read_model(deck / "reducer.scene")
    a, b, _, _ = pairs[1]
    with np.load(deck / "assembly.npz") as model, np.load(matched) as rd, np.load(original) as old:
        time = rd["reference_rad"][:, 0] / model["bodies"][0, 8]
        angles = rd["angle_rad"] + model["coord"][:, 3]
        force = rd["resultant_N"][:, 1]
        original_force = old["resultant_N"][:, 1]
    parts = []
    for i in (a, b):
        mesh = meshio.read(bodies[i][0].with_suffix(".msh"))
        parts.append((mesh.points, mesh.get_cells_type("triangle")))
    shaft_origin, shaft_axis = bodies[b][1:3]
    rows = []
    with point_file.open() as stream:
        next(stream)
        for line in stream:
            v = np.fromstring(line, sep=" ")
            if len(v) in (2, 24):
                step = round(v[0] / time[1])
                v = v[2:]
            if not len(v) or v[20] < 1e5:
                continue
            origin = v[7:10] * .001
            normal = v[10:13] / np.linalg.norm(v[10:13])
            distances = []
            nearest = []
            winding = []
            for (vertices, faces), i, n in zip(parts, (a, b), (normal, -normal), strict=True):
                _, position, axis, initial, coordinate = bodies[i]
                rotation = Rotation.from_rotvec(axis * (initial + angles[step, coordinate]))
                world = (rotation.apply(vertices) + position)[faces]
                distance = surface_hit(world, origin, n, .025)
                distances.append(None if distance is None else distance * 1000)
                feet = closest_point(world, np.broadcast_to(origin, (len(world), 3)))
                nearest.append(float(np.linalg.norm(feet - origin, axis=1).min() * 1000))
                # The solid angle of an oriented closed triangle surface
                # distinguishes a material point from a point outside a tooth.
                u, v1, w = (world - origin).transpose(1, 0, 2)
                lu, lv, lw = (np.linalg.norm(x, axis=1) for x in (u, v1, w))
                numerator = np.einsum("ij,ij->i", u, np.cross(v1, w))
                denominator = (lu*lv*lw + lw*np.einsum("ij,ij->i", u, v1)
                               + lu*np.einsum("ij,ij->i", v1, w) + lv*np.einsum("ij,ij->i", w, u))
                winding.append(float(np.arctan2(numerator, denominator).sum() / (2*np.pi)))
            torque = np.cross(origin - shaft_origin, v[20] * normal) @ shaft_axis
            rows.append(dict(time_s=float(time[step]), native_depth_mm=-v[16],
                             native_point_force_N=v[20], native_pair_force_N=float(force[step]),
                             normal_dot_shaft=float(normal @ shaft_axis), direct_shaft_torque_Nm=float(torque),
                             surface_intercepts_mm=distances, nearest_surface_distances_mm=nearest,
                             material_winding_numbers=winding,
                             previous_native_mesh_pair_force_N=float(original_force[step])))
    threshold = 100000
    stats = dict(force_threshold_N=threshold, matched_5s_exceedances=int((force > threshold).sum()),
                 original_5s_exceedances=int((original_force > threshold).sum()),
                 matched_5s_max_N=float(force.max()), original_5s_max_N=float(original_force.max()),
                 point_export_end_s=float(time[step]), exceptional_points=rows)
    if mesh_exe:
        checks, shifts = radial_separation(mesh_exe, bodies, (a, b), parts, angles, np.flatnonzero(force > threshold))
        stats.update(radial_translation_trials_um=shifts.tolist(), separation_checks=checks)
        plot_check(out.with_suffix(".png"), time, force, original_force, stats)
    out.write_text(json.dumps(stats, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("deck", "point_file", "matched", "original", "out"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--mesh-exe", type=Path)
    audit(**vars(parser.parse_args()))
