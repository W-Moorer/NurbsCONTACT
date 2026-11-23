"""Compare CAD projection on identical Gmsh face samples, independently of mesh generation."""

import argparse
import json
from pathlib import Path
import sys
import time

import gmsh
import meshio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "apps"))
from cad_audit_projection import projectors
from prepare_contact_mesh import open_step, sha256


def compare(step, mesh, per_face):
    gmsh.initialize(readConfigFiles=False)
    try:
        open_step(step)
        start = time.perf_counter()
        mapping = projectors(step)
        setup = time.perf_counter() - start
        data = meshio.read(mesh)
        tags = data.get_cell_data("gmsh:geometrical", "triangle")
        cells = data.get_cells_type("triangle")
        samples = reference_points = 0
        legacy_s = projection_s = max_position_mm = max_normal_rad = 0.0
        max_legacy_distance_mm = max_new_distance_mm = 0.0
        rng = np.random.default_rng(0)
        for _, tag in gmsh.model.getEntities(2):
            indexes = np.flatnonzero(tags == tag)
            if per_face and len(indexes) > per_face:
                indexes = np.sort(rng.choice(indexes, per_face, replace=False))
            for start in range(0, len(indexes), 1024):
                triangles = 1000.0 * data.points[cells[indexes[start : start + 1024]]]
                mids = 0.5 * (triangles + np.roll(triangles, -1, axis=1))
                points = np.concatenate([triangles.mean(axis=1)[:, None, :], mids], axis=1).reshape(-1, 3)
                tick = time.perf_counter()
                xyz, uv = gmsh.model.getClosestPoint(2, tag, points.ravel())
                legacy_s += time.perf_counter() - tick
                new_xyz, new_uv = xyz.reshape(-1, 3), uv
                if tag in mapping:
                    tick = time.perf_counter()
                    new_xyz, new_uv, fallback = mapping[tag].project(points)
                    projection_s += time.perf_counter() - tick
                    reference_points += fallback
                else:
                    tick = time.perf_counter()
                    new_xyz, new_uv = gmsh.model.getClosestPoint(2, tag, points.ravel())
                    projection_s += time.perf_counter() - tick
                    new_xyz = new_xyz.reshape(-1, 3)
                    reference_points += len(points)
                samples += len(points)
                xyz = xyz.reshape(-1, 3)
                old_n = gmsh.model.getNormal(tag, uv).reshape(-1, 3)
                new_n = gmsh.model.getNormal(tag, new_uv).reshape(-1, 3)
                angles = np.arctan2(
                    np.linalg.norm(np.cross(old_n, new_n), axis=1), (old_n * new_n).sum(axis=1)
                )
                if not all(np.isfinite(a).all() for a in (xyz, new_xyz, angles)):
                    raise ValueError("nonfinite projection comparison")
                max_position_mm = max(max_position_mm, np.linalg.norm(new_xyz - xyz, axis=1).max())
                max_normal_rad = max(max_normal_rad, angles.max())
                max_legacy_distance_mm = max(
                    max_legacy_distance_mm, np.linalg.norm(xyz - points, axis=1).max()
                )
                max_new_distance_mm = max(max_new_distance_mm, np.linalg.norm(new_xyz - points, axis=1).max())
        return dict(
            step_sha256=sha256(step),
            mesh_sha256=sha256(mesh),
            gmsh_version=gmsh.__version__,
            triangles_on_mesh=len(cells),
            sample_limit_triangles_per_face=per_face,
            seed=0,
            checked_points=samples,
            accelerated_faces=len(mapping),
            face_count=len(set(tags)),
            legacy_path_points=reference_points,
            setup_s=setup,
            legacy_projection_s=legacy_s,
            new_projection_s=projection_s,
            max_projected_point_difference_m=max_position_mm * 0.001,
            max_normal_difference_deg=np.degrees(max_normal_rad),
            max_legacy_sample_distance_m=max_legacy_distance_mm * 0.001,
            max_new_sample_distance_m=max_new_distance_mm * 0.001,
        )
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("step", "mesh", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--triangles-per-face", type=int, default=0, help="0 checks every triangle")
    args = parser.parse_args()
    if args.triangles_per_face < 0:
        parser.error("triangle count must be nonnegative")
    args.output.write_text(
        json.dumps(compare(args.step, args.mesh, args.triangles_per_face), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
