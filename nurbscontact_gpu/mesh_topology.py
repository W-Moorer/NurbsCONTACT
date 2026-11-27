"""Precompute closed triangle-mesh support topology, independent of motion and load."""

import numpy as np


def support_topology(vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or not np.isfinite(vertices).all()
        or faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValueError("Expected finite vertices and indexed nonempty triangles")
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = np.linalg.norm(normals, axis=1)
    if np.any(areas == 0):
        raise ValueError("Degenerate contact triangle")
    normals /= areas[:, None]
    vertex_normals = np.zeros_like(vertices)
    for j in range(3):
        a = tri[:, (j + 1) % 3] - tri[:, j]
        b = tri[:, (j + 2) % 3] - tri[:, j]
        angle = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), (a * b).sum(axis=1))
        np.add.at(vertex_normals, faces[:, j], angle[:, None] * normals)
    radius = float(np.linalg.norm(vertices, axis=1).max())
    edges, lookup = [], {}
    face_edges = np.empty_like(faces)
    incident = [[] for _ in vertices]
    rays = [[] for _ in vertices]
    neighbors = [[] for _ in vertices]
    for f, ids in enumerate(faces):
        for j in range(3):
            a, b = int(ids[j]), int(ids[(j + 1) % 3])
            key = min(a, b), max(a, b)
            if key not in lookup:
                index = len(edges)
                lookup[key] = index
                edges.append([a, b, f, -1])
                incident[a].append(index)
                incident[b].append(index)
                direction = vertices[b] - vertices[a]
                direction /= np.linalg.norm(direction)
                rays[a].append(direction)
                rays[b].append(-direction)
                neighbors[a].append(b)
                neighbors[b].append(a)
            else:
                index = lookup[key]
                if edges[index][3] >= 0 or edges[index][:2] != [b, a]:
                    raise ValueError(
                        "Expected consistently oriented two-manifold edges"
                    )
                edges[index][3] = f
            face_edges[f, j] = index
    edges = np.asarray(edges, dtype=np.int32)
    if np.any(edges[:, 3] < 0):
        raise ValueError("Contact mesh must be closed")
    components = np.full(len(faces), -1, np.int32)
    seeds = []
    volume = 0.0
    for face in range(len(faces)):
        if components[face] >= 0:
            continue
        component = len(seeds)
        seeds.append(int(faces[face, 0]))
        queue = [face]
        components[face] = component
        for current in queue:
            for e in face_edges[current]:
                left, right = edges[e, 2:]
                other = right if left == current else left
                if components[other] < 0:
                    components[other] = component
                    queue.append(int(other))
        shell = tri[queue]
        volume += np.einsum(
            "ij,ij->i", shell[:, 0], np.cross(shell[:, 1], shell[:, 2])
        ).sum()
    if volume <= 0:
        raise ValueError("Expected a positively oriented material volume")
    panels = np.full(len(faces), -1, np.int32)
    panel_faces = []
    for f in range(len(faces)):
        if panels[f] >= 0:
            continue
        index = len(panel_faces)
        queue = [f]
        panels[f] = index
        for current in queue:
            for e in face_edges[current]:
                left, right = edges[e, 2:]
                other = right if left == current else left
                if (
                    panels[other] >= 0
                    or np.sum((normals[f] - normals[other]) ** 2) > 1e-18
                ):
                    continue
                if np.any(abs((tri[other] - tri[f, 0]) @ normals[f]) > radius * 1e-12):
                    continue
                panels[other] = index
                queue.append(int(other))
        panel_faces.append(queue)
    edge_ridge = np.full(len(edges), -1, np.int32)
    ridges = []
    vertex_ridges = [[] for _ in vertices]
    for e, (a, b, f, g) in enumerate(edges):
        pair = tuple(sorted((panels[f], panels[g])))
        if (
            edge_ridge[e] >= 0
            or pair[0] == pair[1]
            or np.sum(np.cross(normals[f], normals[g]) ** 2) < 1e-18
        ):
            continue
        index = len(ridges)
        queue = [e]
        edge_ridge[e] = index
        members = set()
        for current in queue:
            for v in edges[current, :2]:
                members.add(int(v))
                for other in incident[v]:
                    if edge_ridge[other] >= 0:
                        continue
                    left, right = edges[other, 2:]
                    if tuple(sorted((panels[left], panels[right]))) == pair:
                        edge_ridge[other] = index
                        queue.append(other)
        ids = sorted(members)
        projections = (vertices[ids] - vertices[a]) @ (vertices[b] - vertices[a])
        ridges.append(
            [ids[int(np.argmin(projections))], ids[int(np.argmax(projections))]]
        )
        for v in ids:
            vertex_ridges[v].append(index)
    convex = len(seeds) == 1
    for a, b, f, g in edges:
        other = next(v for v in faces[g] if v != a and v != b)
        if normals[f] @ (vertices[other] - vertices[a]) > 2e-12 * radius:
            convex = False
            break

    def csr(rows, dtype):
        offsets = np.r_[0, np.cumsum([len(row) for row in rows])].astype(np.int32)
        return offsets, np.asarray([v for row in rows for v in row], dtype=dtype)

    ray_offsets, ray_vectors = csr(rays, np.float64)
    _, neighbor_ids = csr(neighbors, np.int32)
    ridge_offsets, ridge_ids = csr(vertex_ridges, np.int32)
    panel_offsets, panel_ids = csr(panel_faces, np.int32)
    return dict(
        vertices=vertices,
        faces=faces,
        normals=normals,
        edges=edges,
        face_edges=face_edges,
        panels=panels,
        panel_offsets=panel_offsets,
        panel_faces=panel_ids,
        ray_offsets=ray_offsets,
        rays=ray_vectors,
        ridges=np.asarray(ridges, np.int32).reshape(-1, 2),
        edge_ridge=edge_ridge,
        ridge_offsets=ridge_offsets,
        ridge_ids=ridge_ids,
        radius=radius,
        convex=int(convex),
        vertex_normals=vertex_normals,
        neighbor_ids=neighbor_ids,
        components=components,
        component_seeds=np.asarray(seeds, np.int32),
    )
