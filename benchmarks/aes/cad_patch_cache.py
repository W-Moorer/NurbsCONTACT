"""Bake STEP faces and common trim curves onto the supplied watertight Gmsh topology (mm)."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter

import gmsh
import meshio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "apps"))
from prepare_contact_mesh import open_step, sha256
from cad_audit_projection import projectors


def evaluate(tri, shift, u, v):
    w = 1 - u - v
    return (
        w * tri[:, 0]
        + u * tri[:, 1]
        + v * tri[:, 2]
        + 4 * (w * u * shift[:, 0] + u * v * shift[:, 1] + v * w * shift[:, 2])
    )


def area_vector(tri, shift, u, v):
    w = 1 - u - v
    du = tri[:, 1] - tri[:, 0] + 4 * ((w - u) * shift[:, 0] + v * (shift[:, 1] - shift[:, 2]))
    dv = tri[:, 2] - tri[:, 0] + 4 * ((w - v) * shift[:, 2] + u * (shift[:, 1] - shift[:, 0]))
    return np.cross(du, dv)


def normal(tri, shift, u, v):
    n = area_vector(tri, shift, u, v)
    length = np.linalg.norm(n, axis=1)
    if np.any(length <= 1e-16):
        raise ValueError("Singular CAD patch at a sampled parameter")
    return n / length[:, None]


def orientation_bound(tri, shift):
    flat = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    dual = flat / np.einsum("ij,ij->i", flat, flat)[:, None]
    values = np.array(
        [
            np.einsum("ij,ij->i", dual, area_vector(tri, shift, u, v))
            for u, v in ((0, 0), (1, 0), (0, 1), (0.5, 0), (0.5, 0.5), (0, 0.5))
        ]
    )
    # The oriented Jacobian is quadratic. Positive Bernstein coefficients
    # prove a positive local orientation everywhere, including between probes.
    values[3:] = 2 * values[3:] - 0.5 * (values[:3] + np.roll(values[:3], -1, axis=0))
    return values.min(axis=0)


def bake(part, mesh_dir, output):
    start = perf_counter()
    path = mesh_dir / part["mesh"]
    step = mesh_dir / part["step"]
    source_hash, mesh_hash = sha256(step), sha256(path)
    quality = json.loads(path.with_suffix(".quality.json").read_text())
    if (
        source_hash != quality["step_sha256"]
        or mesh_hash != part["sha256"]
        or mesh_hash != quality["mesh_sha256"]
    ):
        raise ValueError("STEP, face map or mesh differs from its quality record")
    key = hashlib.sha256(("quadratic-v2:" + source_hash + mesh_hash).encode()).hexdigest()
    dest = output / (key + ".npz")
    if dest.exists():
        with np.load(dest, allow_pickle=False) as f:
            info = json.loads(str(f["metadata"]))
            if info["key"] != key:
                raise ValueError("Cache identity mismatch")
        return dict(part, archive=dest.name, patch=info)
    mesh = meshio.read(path)
    nodes = mesh.points * 1000
    ids = mesh.get_cells_type("triangle")
    if not len(ids):
        raise ValueError("The Gmsh model has no first-order surface triangles")
    tags = mesh.get_cell_data("gmsh:geometrical", "triangle")
    tri = nodes[ids]
    area = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if not np.isfinite(tri).all() or np.any(np.linalg.norm(area, axis=1) == 0):
        raise ValueError("Invalid carrier coordinates or degenerate triangle")
    xyz, idx = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    idx = idx.reshape(-1, 3)
    all_edges = np.sort(np.stack((idx, np.roll(idx, -1, axis=1)), axis=2), axis=2)
    edges, edge_id, count = np.unique(
        all_edges.reshape(-1, 2), axis=0, return_inverse=True, return_counts=True
    )
    if np.any(count != 2):
        raise ValueError("The contact surface must be a closed manifold")
    inc = np.argsort(edge_id, kind="stable").reshape(-1, 2) // 3
    sharp = tags[inc[:, 0]] != tags[inc[:, 1]]
    faces = tags[inc[:, 0]]
    mid = xyz[edges].mean(axis=1)
    curved = mid.copy()
    vertex_ids = {tuple(p): i for i, p in enumerate(xyz)}
    lines = mesh.get_cells_type("line")
    line_tags = mesh.get_cell_data("gmsh:geometrical", "line") if len(lines) else []
    curves = {
        tuple(sorted((vertex_ids[tuple(nodes[i])], vertex_ids[tuple(nodes[j])]))): int(tag)
        for (i, j), tag in zip(lines, line_tags, strict=True)
    }
    curve_tags = np.array([curves.get(tuple(edge), 0) for edge in edges])
    if np.any(sharp & (curve_tags == 0)):
        raise ValueError("Missing STEP trim curve for a shared mesh edge")
    gmsh.initialize(readConfigFiles=False)
    try:
        open_step(step)
        mapping = projectors(step)

        def project(dim, tag, points):
            if dim == 2 and tag in mapping:
                return mapping[tag].project(points)[0]
            return np.asarray(gmsh.model.getClosestPoint(dim, int(tag), points.ravel())[0]).reshape(-1, 3)

        for tag in np.unique(curve_tags[curve_tags != 0]):
            pick = np.where(curve_tags == tag)[0]
            curved[pick] = project(1, tag, mid[pick])
        centers = np.empty((len(tri), 3))
        for tag in np.unique(tags):
            pick = np.where((faces == tag) & (curve_tags == 0))[0]
            if len(pick):
                curved[pick] = project(2, tag, mid[pick])
            pick = np.where(tags == tag)[0]
            centers[pick] = project(2, tag, tri[pick].mean(axis=1))
        shift = (curved - mid)[edge_id.reshape(-1, 3)]
        initial = orientation_bound(tri, shift)
        repair = np.unique(tags[initial <= 1e-7])
        for tag in repair:
            # On a narrow curved face an orthogonal projection can collapse
            # the carrier's diagonal onto a trim. Preserve its UV position.
            select = np.flatnonzero(tags == tag)
            vertices = np.unique(idx[select])
            uv = gmsh.model.getParametrization(2, int(tag), xyz[vertices].ravel()).reshape(-1, 2)
            check = gmsh.model.getValue(2, int(tag), uv.ravel()).reshape(-1, 3)
            if np.linalg.norm(check - xyz[vertices], axis=1).max() > 1e-6:
                raise ValueError(f"Unresolved face parameters for STEP face {tag}")
            inner = np.flatnonzero((faces == tag) & (curve_tags == 0))
            mid_uv = uv[np.searchsorted(vertices, edges[inner])].mean(axis=1)
            center_uv = uv[np.searchsorted(vertices, idx[select])].mean(axis=1)
            for params in (mid_uv, center_uv):
                if len(params) and gmsh.model.isInside(2, int(tag), params.ravel(), True) != len(params):
                    raise ValueError(f"Parametric repair crosses the trim on STEP face {tag}")
            if len(inner):
                curved[inner] = gmsh.model.getValue(2, int(tag), mid_uv.ravel()).reshape(-1, 3)
            centers[select] = gmsh.model.getValue(2, int(tag), center_uv.ravel()).reshape(-1, 3)
        shift = (curved - mid)[edge_id.reshape(-1, 3)]
        lower = orientation_bound(tri, shift)
        if not np.isfinite(lower).all() or lower.min() <= 1e-7:
            raise ValueError("Refine the input mesh: quadratic orientation could not be certified")
        errors, flat_errors = [], []
        for u, v in ((0.19, 0.27), (0.62, 0.17), (0.14, 0.68)):
            q = evaluate(tri, shift, u, v)
            flat = (1 - u - v) * tri[:, 0] + u * tri[:, 1] + v * tri[:, 2]
            for tag in np.unique(tags):
                pick = np.where(tags == tag)[0]
                pair = np.concatenate((q[pick], flat[pick]))
                residual = np.linalg.norm(project(2, tag, pair) - pair, axis=1) * 1000
                errors.extend(residual[: len(pick)])
                flat_errors.extend(residual[len(pick) :])
    finally:
        gmsh.finalize()
    if not np.isfinite([*errors, *flat_errors]).all() or not np.isfinite(centers).all():
        raise ValueError("Invalid STEP evaluation")
    # The quadratic displacement from the linear parameter triangle is a
    # Bernstein polynomial with edge controls 2*shift and zero vertex controls.
    bound = 2 * np.linalg.norm(shift, axis=2).max(axis=1)
    source = np.concatenate((centers, curved[sharp]))
    info = dict(
        key=key,
        step_sha256=source_hash,
        mesh_sha256=mesh_hash,
        triangles=len(tri),
        max_carrier_offset_mm=float(bound.max()),
        audit_samples=len(errors),
        initial_fold_candidates=int(np.sum(initial <= 1e-7)),
        repaired_faces=len(repair),
        min_oriented_jacobian_ratio=float(lower.min()),
        quadratic_p99_max_um=np.percentile(errors, [99, 100]).tolist(),
        linear_p99_max_um=np.percentile(flat_errors, [99, 100]).tolist(),
    )
    with tempfile.NamedTemporaryFile(dir=output, suffix=".npz", delete=False) as tmp:
        staged = Path(tmp.name)
    try:
        info["bake_s"] = perf_counter() - start
        np.savez_compressed(
            staged, tri=tri, faces=tags, points=source, shift=shift, metadata=json.dumps(info)
        )
        staged.replace(dest)
    finally:
        staged.unlink(missing_ok=True)
    print(step.name, info, flush=True)
    return dict(part, archive=dest.name, patch=info)


def prepare(deck, mesh_dir, output, jobs):
    import shutil

    start = perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    parts = json.loads((deck / "models.json").read_text())["parts"]
    if jobs == 1:
        entries = [bake(p, mesh_dir, output) for p in parts]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            jobs = [pool.submit(bake, p, mesh_dir, output) for p in parts]
            entries = [j.result() for j in jobs]
    assembly = deck / "assembly.npz"
    if not assembly.exists() and (output / "assembly.npz").exists():
        raise ValueError("A geometry-only deck would leave an unrelated assembly in the output directory")
    if assembly.exists() and assembly.resolve() != (output / "assembly.npz").resolve():
        shutil.copyfile(assembly, output / "assembly.npz")
    manifest = dict(parts=entries, prepare_wall_s=perf_counter() - start)
    (output / "models.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("deck", "mesh_dir", "output"):
        p.add_argument(field, type=Path)
    p.add_argument("--jobs", type=int, default=1)
    args = p.parse_args()
    prepare(args.deck, args.mesh_dir, args.output, args.jobs)
