"""Build a linear Gmsh contact surface in metres and audit it against the STEP faces."""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import tempfile
import time

import gmsh
import igl
import igl.copyleft.cgal as cgal
import numpy as np
from cad_audit_projection import projectors


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_topology(vertices, faces):
    if len(np.unique(np.sort(faces, axis=1), axis=0)) != len(faces):
        raise ValueError("duplicate triangles")
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, index, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True,
                                 return_counts=True)
    if np.any(counts != 2):
        raise ValueError(f"{np.count_nonzero(counts != 2)} boundary or non-manifold edges")
    signs = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    if np.any(np.bincount(index, weights=signs)):
        raise ValueError("inconsistent winding across a shared mesh edge")
    if not np.all(igl.is_vertex_manifold(faces)[np.unique(faces)]):
        raise ValueError("non-manifold mesh vertex")
    intersections = cgal.remesh_self_intersections(vertices, faces, detect_only=True, first_only=True)[2]
    if intersections.size:
        raise ValueError(f"self-intersecting triangles {intersections[0].tolist()}")


def surface_error(tag, triangles, mapping):
    edges = np.roll(triangles, -1, axis=1) - triangles
    area_vectors = np.cross(edges[:, 0], -edges[:, 2])
    lengths = np.linalg.norm(area_vectors, axis=1)
    if np.any(lengths == 0):
        raise ValueError(f"degenerate triangle on CAD face {tag}")
    normals = area_vectors / lengths[:, None]
    unit_edges = edges / np.linalg.norm(edges, axis=2)[:, :, None]
    cosines = (-unit_edges * np.roll(unit_edges, 1, axis=1)).sum(axis=2)
    min_angle = np.degrees(np.arccos(np.clip(cosines.max(), -1, 1)))
    # Probe the face interior and all three edge midpoints, including CAD feature edges.
    midpoints = (triangles + np.roll(triangles,-1,axis=1)) * .5
    samples = np.concatenate([triangles.mean(axis=1)[:,None,:], midpoints], axis=1)
    max_error, max_angle = 0., 0.
    unique_count, legacy_count = 0, 0
    for start in range(0, len(triangles), 1024):
        requested = samples[start:start + 1024].reshape(-1, 3)
        points, indices = np.unique(requested, axis=0, return_inverse=True)
        indices = indices.reshape(-1)
        if tag in mapping:
            projected, uv, fallback = mapping[tag].project(points)
        else:
            projected, uv = gmsh.model.getClosestPoint(2, tag, points.ravel())
            fallback = len(points)
        unique_count += len(points)
        legacy_count += fallback
        cad_normals = np.asarray(gmsh.model.getNormal(tag, uv)).reshape(-1, 3)
        cad_lengths = np.linalg.norm(cad_normals, axis=1)
        if np.any(cad_lengths == 0) or not np.isfinite(cad_lengths).all():
            raise ValueError(f"undefined CAD normal at face {tag}")
        distances = np.linalg.norm(projected.reshape(-1, 3) - points, axis=1)
        if not np.isfinite(distances).all():
            raise ValueError(f"invalid CAD projection at face {tag}")
        max_error = max(max_error, float(distances.max()))
        mesh_normals = np.repeat(normals[start:start + 1024], 4, axis=0)
        cos = np.einsum("ij,ij->i", mesh_normals, cad_normals[indices]) / cad_lengths[indices]
        max_angle = max(max_angle, float(np.degrees(np.arccos(np.clip(np.abs(cos), 0, 1))).max()))
    return float(min_angle), max_error * .001, max_angle, unique_count, legacy_count


def open_step(step):
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
    gmsh.model.add("contact")
    solids = gmsh.model.occ.importShapes(str(step.resolve()))
    if not solids or any(dim != 3 for dim, _ in solids):
        raise ValueError("contact mesh requires a STEP solid with a closed boundary")
    gmsh.model.occ.synchronize()


def cad_face_signature():
    return [(tag, gmsh.model.getType(2, tag), gmsh.model.occ.getMass(2, tag),
             gmsh.model.occ.getCenterOfMass(2, tag)) for _, tag in gmsh.model.getEntities(2)]


def init_audit_worker(step, signature, projection):
    global worker_mapping
    gmsh.initialize(readConfigFiles=False)
    open_step(step)
    if cad_face_signature() != signature:
        raise ValueError("worker STEP geometry does not match the meshed body's face tags")
    worker_mapping = projectors(step) if projection=="auto" else {}


def audit_chunk(tag, chunk):
    return surface_error(tag,chunk,worker_mapping)


def audit(step, workers, projection):
    start_time = time.perf_counter()
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    order = np.argsort(node_tags)
    vertices = coords.reshape(-1, 3)[order]
    sorted_tags = node_tags[order]
    blocks, reports = [], []
    def merge(row, values):
        angle, error, normal, evaluated, legacy = values
        row["min_angle_deg"] = min(row["min_angle_deg"], angle)
        row["max_sampled_surface_error_m"] = max(row["max_sampled_surface_error_m"], error)
        row["max_normal_angle_deg"] = max(row["max_normal_angle_deg"], normal)
        row["evaluated_points"] += evaluated
        row["gmsh_closest_points"] += legacy

    pool = (ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                                initializer=init_audit_worker, initargs=(step, cad_face_signature(), projection))
            if workers > 1 else nullcontext(None))
    mapping = projectors(step) if workers==1 and projection=="auto" else {}
    with pool as executor:
        pending = deque()
        for _, tag in gmsh.model.getEntities(2):
            types = gmsh.model.mesh.getElementTypes(2, tag)
            if list(types) != [2]:
                raise ValueError(f"expected linear triangles on face {tag}, found {list(types)}")
            element_ids, nodes = gmsh.model.mesh.getElementsByType(2, tag)
            indices = np.searchsorted(sorted_tags, nodes).reshape(-1, 3)
            blocks.append(indices)
            qualities = gmsh.model.mesh.getElementQualities(element_ids, "minSICN")
            if not np.isfinite(qualities).all() or qualities.min() <= 0:
                raise ValueError(f"non-positive element quality on face {tag}")
            row = dict(face=tag, triangles=len(indices), min_angle_deg=180.,
                       min_sicn=float(qualities.min()), p05_sicn=float(np.quantile(qualities, .05)),
                       max_sampled_surface_error_m=0., max_normal_angle_deg=0.,
                       evaluated_points=0, gmsh_closest_points=0)
            reports.append(row)
            for start in range(0, len(indices), 4096):
                chunk = vertices[indices[start:start + 4096]]
                if executor is None:
                    merge(row, surface_error(tag, chunk, mapping))
                else:
                    pending.append((row, executor.submit(audit_chunk, tag, chunk)))
                    if len(pending) >= 2 * workers:
                        earlier, future = pending.popleft()
                        merge(earlier, future.result())
        for row, future in pending:
            merge(row, future.result())
    face_elapsed = time.perf_counter()-start_time
    topology_start = time.perf_counter()
    faces = np.concatenate(blocks)
    check_topology(vertices, faces)
    volume_origin = vertices.mean(axis=0)
    a, b, c = (vertices[faces[:, k]] - volume_origin for k in range(3))
    volume = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
    if volume <= 0:
        raise ValueError("closed boundary has non-positive signed volume")
    topology_s = time.perf_counter()-topology_start
    edge_start = time.perf_counter()
    curves = []
    for _, tag in gmsh.model.getEntities(1):
        _, node_ids = gmsh.model.mesh.getElementsByType(1, tag)
        if len(node_ids) == 0:
            continue
        segments = vertices[np.searchsorted(sorted_tags, node_ids).reshape(-1, 2)]
        midpoints = segments.mean(axis=1)
        projected, _ = gmsh.model.getClosestPoint(1, tag, midpoints.ravel())
        error = np.linalg.norm(projected.reshape(-1, 3) - midpoints, axis=1)
        if not np.isfinite(error).all():
            raise ValueError(f"invalid projection on CAD edge {tag}")
        curves.append(dict(edge=tag, segments=len(segments), max_sampled_error_m=float(error.max()) * .001))
    return dict(vertices=len(vertices), triangles=len(faces), faces=reports, edges=curves,
                face_audit_s=face_elapsed, topology_s=topology_s, edge_audit_s=time.perf_counter()-edge_start,
                face_samples=4*len(faces), evaluated_points=sum(row["evaluated_points"] for row in reports),
                gmsh_closest_points=sum(row["gmsh_closest_points"] for row in reports),
                signed_volume_m3=float(volume) * 1e-9,
                max_sampled_edge_error_m=max((row["max_sampled_error_m"] for row in curves), default=0.),
                max_sampled_surface_error_m=max(row["max_sampled_surface_error_m"] for row in reports),
                max_normal_angle_deg=max(row["max_normal_angle_deg"] for row in reports),
                min_angle_deg=min(row["min_angle_deg"] for row in reports),
                min_sicn=min(row["min_sicn"] for row in reports))


def generate_surface(edge_samples=None):
    if edge_samples is not None:
        face_samples = gmsh.option.getNumber("Mesh.MeshSizeFromCurvature")
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", edge_samples)
        try:
            gmsh.model.mesh.generate(1)
        finally:
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", face_samples)
    gmsh.model.mesh.generate(2)


def refine_slender_faces(required_sicn, size_mm, edge_samples=None):
    """Resolve narrow CAD strips by dividing their long edges in proportion to local width."""
    edge_counts = {}
    spacing_by_edge = {}
    for _, face in gmsh.model.getEntities(2):
        element_ids, _ = gmsh.model.mesh.getElementsByType(2, face)
        if min(gmsh.model.mesh.getElementQualities(element_ids, "minSICN")) >= required_sicn:
            continue
        edges = {tag: gmsh.model.occ.getMass(1, tag)
                 for _, tag in gmsh.model.getBoundary([(2, face)], oriented=False)}
        perimeter = sum(edges.values())
        if perimeter <= 0:
            continue
        width = 2 * gmsh.model.occ.getMass(2, face) / perimeter
        if width <= 0 or max(edges.values()) < 8 * width:
            continue
        spacing = min(size_mm, width / required_sicn)
        for tag, length in edges.items():
            # A short trim can have several Gmsh segments even on a straight
            # CAD edge. Match the adjacent face size to the actual segments.
            _, nodes = gmsh.model.mesh.getElementsByType(1, tag)
            count = max(1, len(nodes) // 2)
            neighbor_size = min(spacing, length / (count * 2 * required_sicn))
            if neighbor_size > 0:
                spacing_by_edge[tag] = min(spacing_by_edge.get(tag, neighbor_size), neighbor_size)
            if length < 8 * width:
                continue
            count = math.ceil(length / spacing) + 1
            edge_counts[tag] = max(edge_counts.get(tag, 0), count)
    if edge_counts:
        gmsh.model.mesh.clear()
        for tag, count in edge_counts.items():
            # At the ends of a strip, CAD corner nodes meet shorter trim
            # segments. Concentrate some of the long-edge nodes at both ends.
            gmsh.model.mesh.setTransfiniteCurve(tag, count, "Bump", .7)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        generate_surface(edge_samples)
        _, tri_tags, _ = gmsh.model.mesh.getElements(2)
        if gmsh.model.mesh.getElementQualities(tri_tags[0], "minSICN").min() < required_sicn:
            # Adjacent faces can inherit a very short boundary edge. Grade their
            # interior sizes away from it as well, keeping the CAD seam intact.
            gmsh.model.mesh.clear()
            bands = {}
            for tag, spacing in spacing_by_edge.items():
                level = math.floor(math.log2(spacing / size_mm))
                bands.setdefault(level, []).append(tag)
            field = gmsh.model.mesh.field
            fields = []
            for level, tags in sorted(bands.items()):
                near = size_mm * 2.**level
                distance = field.add("Distance")
                field.setNumbers(distance, "CurvesList", tags)
                longest = max(gmsh.model.occ.getMass(1, tag) for tag in tags)
                field.setNumber(distance, "Sampling", max(10, math.ceil(longest / near)))
                threshold = field.add("Threshold")
                field.setNumber(threshold, "InField", distance)
                field.setNumber(threshold, "SizeMin", near)
                field.setNumber(threshold, "SizeMax", size_mm)
                field.setNumber(threshold, "DistMin", near)
                field.setNumber(threshold, "DistMax", 1.25 * size_mm)
                fields.append(threshold)
            minimum = field.add("Min")
            field.setNumbers(minimum, "FieldsList", fields)
            field.setAsBackgroundMesh(minimum)
            generate_surface(edge_samples)
            elements, _ = gmsh.model.mesh.getElementsByType(2)
            if gmsh.model.mesh.getElementQualities(elements, "minSICN").min() < required_sicn:
                gmsh.model.mesh.clear()
                for threshold in fields:
                    field.setNumber(threshold, "DistMax", 4 * size_mm)
                generate_surface(edge_samples)
    return len(edge_counts)


def map_four_sided_faces(size_mm, chord_mm):
    """Use compatible opposite edge counts on simply bounded quadrilateral CAD faces."""
    groups = {}
    ends = {}
    quads = []

    def root(tag):
        groups.setdefault(tag, tag)
        while groups[tag] != tag:
            groups[tag] = groups[groups[tag]]
            tag = groups[tag]
        return tag

    for _, face in gmsh.model.getEntities(2):
        edges = [tag for _, tag in gmsh.model.getBoundary([(2, face)], combined=False, oriented=False)]
        if len(edges) != 4 or len(set(edges)) != 4:
            continue
        for tag in edges:
            if tag not in ends:
                ends[tag] = {
                    v for _, v in gmsh.model.getBoundary([(1, tag)], combined=False, oriented=False)
                }
        endpoint_sets = [ends[tag] for tag in edges]
        corners = set.union(*endpoint_sets)
        if (
            len(corners) != 4
            or any(len(s) != 2 for s in endpoint_sets)
            or len({frozenset(s) for s in endpoint_sets}) != 4
        ):
            continue
        if any(sum(v in s for s in endpoint_sets) != 2 for v in corners):
            continue
        for tag in edges:
            opposite = next(k for k in edges if ends[tag].isdisjoint(ends[k]))
            groups[root(tag)] = root(opposite)
        quads.append(face)
    subdivisions = {}
    for tag in groups:
        length = gmsh.model.occ.getMass(1, tag)
        lo, hi = gmsh.model.getParametrizationBounds(1, tag)
        count = max(1, math.ceil(length / size_mm))
        # Midpoint and quarter-point distances to each chord control refinement
        # of the boundary curve. The final audit samples the actual Gmsh mesh.
        while True:
            t = np.linspace(lo[0], hi[0], 4 * count + 1)
            xyz = gmsh.model.getValue(1, tag, t).reshape(-1, 3)
            a, b = xyz[:-1:4], xyz[4::4]
            chord = b - a
            denom = np.einsum("ij,ij->i", chord, chord)
            worst = 0.0
            for k in (1, 2, 3):
                delta = xyz[k::4] - a
                along = np.clip(np.einsum("ij,ij->i", delta, chord) / np.maximum(denom, 1e-300), 0.0, 1.0)
                worst = max(worst, np.linalg.norm(delta - along[:, None] * chord, axis=1).max())
            if worst <= chord_mm:
                break
            if count >= 32768:
                raise ValueError(f"failed to resolve the requested chord deviation on curve {tag}")
            count *= 2
        key = root(tag)
        subdivisions[key] = max(subdivisions.get(key, 1), count)
    for tag in groups:
        gmsh.model.mesh.setTransfiniteCurve(tag, subdivisions[root(tag)] + 1)
    for tag in quads:
        gmsh.model.mesh.setTransfiniteSurface(tag, "AlternateLeft")
    return len(quads)


def prepare(step, output, size_m, max_error_m=None, max_normal_deg=None, min_sicn=0., audit_workers=1,
            projection="auto", extend_boundary=True, mapped_edge_error_m=None, curvature_samples=24,
            edge_curvature_samples=None, compact_planar_solids=False):
    limits = (size_m, max_error_m, max_normal_deg, mapped_edge_error_m)
    if any(v is not None and (not math.isfinite(v) or v <= 0) for v in limits):
        raise ValueError("size and requested tolerances must be positive and finite")
    if not 0 <= min_sicn <= 1 or output.suffix.lower() != ".msh":
        raise ValueError("expected a .msh output and min-SICN between zero and one")
    if not isinstance(audit_workers, int) or audit_workers < 1:
        raise ValueError("expected a positive integer number of audit workers")
    if projection not in ("auto","gmsh"):
        raise ValueError("unknown CAD projection strategy")
    if not isinstance(curvature_samples,int) or curvature_samples<3 or (edge_curvature_samples is not None and
            (not isinstance(edge_curvature_samples, int) or edge_curvature_samples < 3)):
        raise ValueError("expected at least three angular samples per full circle")
    start = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as work:
        staged = Path(work) / "contact.msh"
        gmsh.initialize(readConfigFiles=False)
        try:
            open_step(step)
            requested_size_m = size_m
            polyhedral = (compact_planar_solids and
                          all(gmsh.model.getType(2, tag) == "Plane" for _, tag in gmsh.model.getEntities(2)) and
                          all(gmsh.model.getType(1, tag) == "Line" for _, tag in gmsh.model.getEntities(1)))
            if polyhedral:
                bounds = gmsh.model.getBoundingBox(-1, -1)
                size_m = max(size_m, .001 * math.dist(bounds[:3], bounds[3:]))
                extend_boundary = False
                if mapped_edge_error_m is None:
                    mapped_edge_error_m = size_m
            gmsh.option.setNumber("Mesh.MeshSizeMax", size_m * 1000)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature_samples)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", int(extend_boundary))
            gmsh.option.setNumber("Mesh.ElementOrder", 1)
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            mapped_faces = (map_four_sided_faces(size_m*1000,mapped_edge_error_m*1000)
                            if mapped_edge_error_m is not None else 0)
            generate_surface(edge_curvature_samples)
            initial_mesh_s = time.perf_counter() - start
            if mapped_faces and min_sicn>0:
                elements,_=gmsh.model.mesh.getElementsByType(2)
                if gmsh.model.mesh.getElementQualities(elements,"minSICN").min()<min_sicn:
                    gmsh.model.mesh.clear()
                    gmsh.model.mesh.removeConstraints()
                    generate_surface(edge_curvature_samples)
                    mapped_faces=0
            refined_edges = refine_slender_faces(min_sicn, size_m * 1000, edge_curvature_samples) if min_sicn > 0 else 0
            if min_sicn > 0:
                elements, _ = gmsh.model.mesh.getElementsByType(2)
                qualities = gmsh.model.mesh.getElementQualities(elements, "minSICN")
                worst = int(np.argmin(qualities))
                if qualities[worst] < min_sicn:
                    _, _, _, tag = gmsh.model.mesh.getElement(int(elements[worst]))
                    raise ValueError(f"mesh exceeds quality limits on face {tag}: minSICN "
                                     f"{qualities[worst]:.6g}, {np.count_nonzero(qualities < min_sicn)} "
                                     f"of {len(elements)} triangles below {min_sicn:.6g}")
            audit_start = time.perf_counter()
            report = audit(step, audit_workers, projection)
            audit_s = time.perf_counter() - audit_start
            sampled_error = max(report["max_sampled_surface_error_m"], report["max_sampled_edge_error_m"])
            if ((max_error_m is not None and sampled_error > max_error_m)
                    or (max_normal_deg is not None and report["max_normal_angle_deg"] > max_normal_deg)
                    or report["min_sicn"] < min_sicn):
                raise ValueError(f"mesh exceeds quality limits: sampled CAD error "
                                 f"{sampled_error:.6g} m, "
                                 f"normal {report['max_normal_angle_deg']:.5g} deg, "
                                 f"minSICN {report['min_sicn']:.5g}")
            gmsh.option.setNumber("Mesh.ScalingFactor", .001)
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.option.setNumber("Mesh.Binary", 0)
            gmsh.write(str(staged))
        finally:
            gmsh.finalize()
        report.update(step_sha256=sha256(step), mesh_sha256=sha256(staged), gmsh_version=gmsh.__version__,
                      length_unit="m", mesh_size_m=size_m, requested_mesh_size_m=requested_size_m,
                      compact_planar_solid=polyhedral, requested_max_sampled_error_m=max_error_m,
                      requested_max_normal_angle_deg=max_normal_deg, requested_min_sicn=min_sicn,
                      locally_refined_edges=refined_edges, audit_workers=audit_workers, audit_s=audit_s,
                      preparation_s=time.perf_counter() - start, generation_s=audit_start-start,
                      initial_mesh_s=initial_mesh_s, refinement_and_quality_s=audit_start-start-initial_mesh_s,
                      projection=projection, extend_boundary=extend_boundary, mapped_faces=mapped_faces,
                      mapped_edge_error_m=mapped_edge_error_m, curvature_samples=curvature_samples,
                      edge_curvature_samples=edge_curvature_samples)
        metadata = Path(work) / "quality.json"
        metadata.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staged.replace(output)
        metadata.replace(output.with_suffix(".quality.json"))
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--size-m", required=True, type=float)
    parser.add_argument("--max-error-m", type=float, help="maximum sampled distance to CAD faces and edges")
    parser.add_argument("--max-normal-deg", type=float)
    parser.add_argument("--min-sicn", type=float, default=0.)
    parser.add_argument("--audit-workers", type=int, default=1, help="independent processes for CAD projections")
    parser.add_argument("--projection", choices=["auto","gmsh"], default="auto")
    parser.add_argument("--no-extend-boundary", action="store_true",
                        help="keep short trim edges from prescribing the mesh size across neighboring faces")
    parser.add_argument("--compact-planar-solids", action="store_true",
                        help="allow larger triangles for solids bounded entirely by planes and straight edges")
    parser.add_argument("--mapped-edge-error-m",type=float,
                        help="map four-sided CAD faces with edge counts adapted to curvature")
    parser.add_argument("--curvature-samples",type=int,default=24,help="Gmsh curvature sampling per full circle")
    parser.add_argument("--edge-curvature-samples", type=int,
                        help="mesh CAD edges at this angular resolution before filling faces")
    args = parser.parse_args()
    result = prepare(args.step, args.output, args.size_m, args.max_error_m, args.max_normal_deg, args.min_sicn,
                     args.audit_workers,args.projection,not args.no_extend_boundary,args.mapped_edge_error_m,
                     args.curvature_samples, args.edge_curvature_samples, args.compact_planar_solids)
    print(json.dumps({k: v for k, v in result.items() if k not in ("faces", "edges")}, allow_nan=False))
