"""Build rigid local contact probes directly on trimmed STEP faces and shared edges."""

import argparse
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from contact_tracking_replay import load_tags
from gpu_mesh_line import read_full_mesh
from resident_contact import surface_samples


def geometry_hash(triangles, faces):
    h = sha256()
    for values, dtype in ((triangles, "<f8"), (faces, "<i8")):
        h.update(np.ascontiguousarray(values, dtype=dtype).tobytes())
    return h.hexdigest()


def load_probes(path, triangles, faces, density):
    with np.load(path, allow_pickle=False) as data:
        if (
            str(data["geometry_sha256"]) != geometry_hash(triangles, faces)
            or int(data["per_edge"]) != density
        ):
            raise ValueError("CAD probes belong to a different source mesh, face map or sampling density")
        return data["points"], dict(
            source_step_sha256=str(data["step_sha256"]),
            face_probes=int(data["face_probes"]),
            edge_probes=int(data["edge_probes"]),
            maximum_source_shift_um=float(data["max_shift_um"]),
            offline_projection_s=float(data["projection_s"]),
        )


def prepare(step, triangles, labels, density):
    from OCP.BRep import BRep_Tool
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.BRepClass import BRepClass_FaceClassifier
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.BRepTools import BRepTools
    from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_IN, TopAbs_ON
    from OCP.TopoDS import TopoDS
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.gp import gp_Pnt, gp_Pnt2d
    from nhrep_gear import read_occt_solid

    original, _, owner, _, alternative, tags = surface_samples(triangles, labels, density, True)
    solid = read_occt_solid(step)
    faces, edges = TopTools_IndexedMapOfShape(), TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(solid, TopAbs_FACE, faces)
    TopExp.MapShapes_s(solid, TopAbs_EDGE, edges)
    if not np.array_equal(np.unique(labels), np.arange(1, faces.Extent() + 1)):
        raise ValueError("Expected the indexed face labels of this closed STEP body")
    face_edges = {}
    for tag in tags:
        it = TopExp_Explorer(faces.FindKey(int(tag)), TopAbs_EDGE)
        ids = set()
        while it.More():
            ids.add(edges.FindIndex(it.Current()))
            it.Next()
        face_edges[tag] = ids
    queries = {}
    projectors = {}
    classifier = BRepClass_FaceClassifier()
    out = np.empty_like(original)
    for i, (seed, own, alt) in enumerate(zip(original, tags[owner], tags[alternative], strict=True)):
        key = tuple(sorted((own, alt)))
        point = gp_Pnt(*seed)
        if own == alt:
            if own not in projectors:
                face = TopoDS.Face_s(faces.FindKey(int(own)))
                project = GeomAPI_ProjectPointOnSurf()
                project.Init(BRep_Tool.Surface_s(face), *BRepTools.UVBounds_s(face), 1.0e-9)
                projectors[own] = face, project
            face, project = projectors[own]
            project.Perform(point)
            if project.NbPoints():
                uv = gp_Pnt2d(*project.LowerDistanceParameters())
                classifier.Perform(face, uv, 1.0e-9)
                if classifier.State() in (TopAbs_IN, TopAbs_ON):
                    out[i] = project.NearestPoint().Coord()
                    continue
        if key not in queries:
            if own == alt:
                shapes = [faces.FindKey(int(own))]
            else:
                shapes = [edges.FindKey(e) for e in sorted(face_edges[own] & face_edges[alt])]
                if not shapes:
                    raise ValueError(f"STEP faces {key} share no topological edge")
            queries[key] = []
            for s in shapes:
                query = BRepExtrema_DistShapeShape()
                query.SetDeflection(1.0e-9)
                query.LoadS2(s)
                queries[key].append(query)
        vertex = BRepBuilderAPI_MakeVertex(point).Vertex()
        best = float("inf")
        for q in queries[key]:
            q.LoadS1(vertex)
            if not q.Perform() or not q.IsDone() or q.NbSolution() < 1:
                raise RuntimeError(f"STEP projection failed for probe {i}")
            if q.Value() < best:
                best = q.Value()
                out[i] = q.PointOnShape2(1).Coord()
    return (
        out,
        int(np.sum(owner == alternative)),
        float(1000 * np.max(np.linalg.norm(out - original, axis=1))),
    )


def parametric_probes(step, triangles, labels, deflection_um):
    """Evaluate matching OCCT UV triangles and polygon-on-edge parameter intervals."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.BRepTopAdaptor import BRepTopAdaptor_FClass2d
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_IN, TopAbs_ON
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.gp import gp_Pnt, gp_Pnt2d
    from scipy.spatial import cKDTree
    from gear_precision import mesh_step

    tags, centers, edge_seeds, edge_points = [], [], [], []

    def visit(face, location, mesh, xyz, indices):
        surface = BRepAdaptor_Surface(face)
        classifier = BRepTopAdaptor_FClass2d(face, 1.0e-9)
        uv = np.array([mesh.UVNode(i).Coord() for i in range(1, mesh.NbNodes() + 1)])
        block = []
        for w, tri in zip(uv[indices].mean(axis=1), indices, strict=True):
            if classifier.Perform(gp_Pnt2d(*w)) in (TopAbs_IN, TopAbs_ON):
                block.append(surface.Value(*w).Coord())
            else:
                vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(*xyz[tri].mean(axis=0))).Vertex()
                q = BRepExtrema_DistShapeShape(vertex, face)
                if not q.IsDone() or not q.NbSolution():
                    raise ValueError("Unresolved trimmed face sample")
                block.append(q.PointOnShape2(1).Coord())
        centers.append(np.asarray(block))
        it = TopExp_Explorer(face, TopAbs_EDGE)
        while it.More():
            edge = TopoDS.Edge_s(it.Current())
            polygon = BRep_Tool.PolygonOnTriangulation_s(edge, mesh, location)
            if polygon is not None and polygon.HasParameters() and not BRep_Tool.Degenerated_s(edge):
                curve = BRepAdaptor_Curve(edge)
                for k in range(1, polygon.NbNodes()):
                    i, j = polygon.Node(k) - 1, polygon.Node(k + 1) - 1
                    edge_seeds.append((xyz[i] + xyz[j]) * 0.5)
                    edge_points.append(
                        curve.Value((polygon.Parameter(k) + polygon.Parameter(k + 1)) * 0.5).Coord()
                    )
            it.Next()

    v, f = mesh_step(deflection_um, False, step=step, face_ids=tags, visit_face=visit)
    np.testing.assert_array_equal(v[f], triangles)
    np.testing.assert_array_equal(np.concatenate(tags), labels)
    seeds = surface_samples(triangles, labels, 1, True)[0]
    out = np.empty_like(seeds)
    n = len(triangles)
    out[:n] = np.concatenate(centers)
    if len(seeds) > n:
        if not edge_seeds:
            raise ValueError("The STEP triangulation has no mapped boundary curves")
        # Match the two endpoints recovered from welded mesh topology by segment midpoint.
        # Coincident midpoints must also have coincident curved witnesses.
        tree = cKDTree(edge_seeds)
        candidates = tree.query_ball_point(seeds[n:], 2e-6)
        coords = np.asarray(edge_points)
        for i, inds in enumerate(candidates, n):
            if not inds or np.linalg.norm(coords[inds] - coords[inds[0]], axis=1).max() > 2e-6:
                raise ValueError("Ambiguous STEP boundary segment; use trimmed-face projection")
            out[i] = coords[inds[0]]
    return out, n, float(1000 * np.linalg.norm(out - seeds, axis=1).max())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("step", "mesh", "tags", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument(
        "--deflection-um",
        type=float,
        help="Reuse matching OCCT UV nodes and native edge parameters (one interior probe per triangle)",
    )
    args = p.parse_args()
    tri = read_full_mesh(args.mesh)
    labels = load_tags(args.tags, args.mesh)
    start = perf_counter()
    if args.deflection_um is None:
        points, faces, shift = prepare(args.step, tri, labels, args.samples)
    else:
        if args.samples != 1:
            p.error("UV-centroid sampling uses one interior probe per triangle")
        points, faces, shift = parametric_probes(args.step, tri, labels, args.deflection_um)
    np.savez_compressed(
        args.output,
        points=points,
        geometry_sha256=geometry_hash(tri, labels),
        step_sha256=sha256(args.step.read_bytes()).hexdigest(),
        per_edge=args.samples,
        face_probes=faces,
        edge_probes=len(points) - faces,
        max_shift_um=shift,
        projection_s=perf_counter() - start,
    )
