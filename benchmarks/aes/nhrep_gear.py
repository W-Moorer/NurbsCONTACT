"""Prepare one STEP solid for upstream NH-Rep and check the field against CAD."""

import argparse
import csv
from hashlib import sha256
import json
from math import gcd
from collections import defaultdict
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np


def read_solid(step):
    import cadquery as cq

    source = cq.importers.importStep(str(step)).val()
    solids = source.Solids()
    if len(solids) != 1 or not solids[0].isValid():
        raise ValueError("NH-Rep requires one valid closed STEP solid")
    return solids[0]


def prepare(solid, output, deflection_mm):
    import trimesh

    vertices, triangles = solid.tessellate(deflection_mm, angularTolerance=.1)
    xyz = np.array([point.toTuple() for point in vertices])
    # Stitch tessellations of incident CAD faces with a sub-micrometer key.
    keys = np.rint(xyz / 1e-6).astype(np.int64)
    _, indices, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    xyz = xyz[indices]
    faces = inverse[np.asarray(triangles)]
    mesh = trimesh.Trimesh(vertices=xyz, faces=faces, process=False)
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError("The stitched training mesh is not an outward-oriented watertight solid")
    incident = defaultdict(list)
    for i, face in enumerate(faces):
        for a, b in zip(face, np.roll(face, -1)):
            incident[tuple(sorted((int(a), int(b))))].append(i)
    edges = [(a, b, f, g) for (a, b), (f, g) in incident.items()
             if np.dot(mesh.face_normals[f], mesh.face_normals[g]) < np.cos(np.deg2rad(25))]
    if not edges:
        raise ValueError("No sharp feature segments found for the NH-Rep evaluation")
    low, high = xyz.min(axis=0), xyz.max(axis=0)
    center, scale = (low + high) / 2, (high - low).max() / 1.8
    with (output / "spur.obj").open("w", encoding="ascii") as stream:
        np.savetxt(stream, (xyz - center) / scale, fmt="v %.15g %.15g %.15g")
        np.savetxt(stream, faces + 1, fmt="f %d %d %d")
    edge_records = np.asarray(edges)
    with (output / "spur.fea").open("w", encoding="ascii") as stream:
        print(len(edges), file=stream)
        np.savetxt(stream, edge_records[:, :2], fmt="%d")
    np.savez(output / "spur-geometry.npz", vertices=xyz, feature_segments=edge_records,
             normals=mesh.face_normals)
    return {"center_mm": center.tolist(), "mm_per_network_unit": float(scale), "deflection_mm": deflection_mm,
            "mesh_vertices": len(xyz), "mesh_triangles": len(faces), "feature_segments": len(edges)}


def read_occt_solid(step):
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape

    reader = STEPControl_Reader()
    if reader.ReadFile(str(step)) != IFSelect_RetDone or not reader.TransferRoots():
        raise ValueError(f"Could not read STEP solid: {step}")
    solids = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(reader.OneShape(), TopAbs_SOLID, solids)
    if solids.Extent() != 1 or not BRepCheck_Analyzer(solids.FindKey(1)).IsValid():
        raise ValueError("NH-Rep requires one valid closed STEP solid")
    return solids.FindKey(1)


def cad_triangle_parameters(step, deflection_mm, mesh, center, scale):
    """Associate each triangle of the original training mesh with its CAD chart."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedMapOfShape

    solid = read_occt_solid(step)
    BRepMesh_IncrementalMesh(solid, deflection_mm, True, .1)
    faces = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(solid, TopAbs_FACE, faces)
    owners, triangle_uv = [], []
    start = 0
    for face_id in range(1, faces.Extent() + 1):
        face = TopoDS.Face_s(faces.FindKey(face_id))
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is None or not triangulation.HasUVNodes():
            raise ValueError(f"Missing parameter coordinates on CAD face {face_id}")
        triangles = np.asarray([t.Get() for t in triangulation.Triangles()]) - 1
        if face.Orientation() == TopAbs_REVERSED:
            triangles = triangles[:, [0, 2, 1]]
        count = triangulation.NbNodes()
        transform = location.Transformation()
        positions = np.asarray([triangulation.Node(i).Transformed(transform).Coord() for i in range(1, count + 1)])
        uv = np.asarray([triangulation.UVNode(i).Coord() for i in range(1, count + 1)])
        end = start + len(triangles)
        np.testing.assert_allclose(mesh.triangles[start:end] * scale + center, positions[triangles], atol=2e-6, rtol=0)
        triangle_uv.append(uv[triangles])
        owners.append(np.full(len(triangles), face_id))
        start = end
    if start != len(mesh.faces):
        raise ValueError("The original tessellation could not be matched to the training mesh")
    return np.concatenate(owners), np.concatenate(triangle_uv)


def nearest_training_triangles(mesh, points, labels, face_labels):
    # The upstream XYZ writer rounds samples to six significant digits.
    # Query just the triangles intersecting that coordinate error box.
    radius = 3e-6
    tree = mesh.triangles_tree
    ids, counts = tree.intersection_v(points - radius, points + radius)
    source_ids = np.repeat(np.arange(len(points)), counts.astype(np.int64))
    matches = face_labels[ids] == labels[source_ids]
    ids, source_ids = ids[matches], source_ids[matches]
    triangles = mesh.triangles[ids]
    samples = points[source_ids]
    a, b, c = triangles.transpose(1, 0, 2)
    ab, ac, ap = b - a, c - a, samples - a
    normal = np.cross(ab, ac)
    normal_squared = np.einsum("ij,ij->i", normal, normal)
    if np.any(normal_squared == 0):
        raise ValueError("Degenerate triangle in the source mesh")
    # Cross products avoid cancellation for long, narrow triangles.
    v = np.einsum("ij,ij->i", np.cross(ap, ac), normal) / normal_squared
    w = np.einsum("ij,ij->i", np.cross(ab, ap), normal) / normal_squared
    barycentric = np.column_stack((1 - v - w, v, w))
    planar = np.einsum("ni,nij->nj", barycentric, triangles)
    distance = np.linalg.norm(planar - samples, axis=1)
    distance[np.any(barycentric < 0, axis=1)] = np.inf
    for start, end in ((0, 1), (1, 2), (2, 0)):
        base = triangles[:, start]
        edge = triangles[:, end] - base
        t = np.clip(np.einsum("ij,ij->i", samples - base, edge) / np.einsum("ij,ij->i", edge, edge), 0, 1)
        edge_distance = np.linalg.norm(base + t[:, None] * edge - samples, axis=1)
        better = edge_distance < distance
        distance[better] = edge_distance[better]
        barycentric[better] = 0
        barycentric[better, start] = 1 - t[better]
        barycentric[better, end] = t[better]
    order = np.lexsort((ids, distance, source_ids))
    found, first = np.unique(source_ids[order], return_index=True)
    if len(found) != len(points):
        raise ValueError(f"No triangle with the original label for {len(points) - len(found)} training samples")
    best = order[first]
    if np.max(distance[best]) > radius:
        raise ValueError(f"Some upstream training points are {np.max(distance[best]):g} units from their patch")
    return barycentric[best], ids[best]


def prepare_parametric(step, mesh_trial, output, executable):
    import trimesh

    started = time.perf_counter()
    metadata = json.loads((mesh_trial / "spur.json").read_text(encoding="utf-8"))
    center, scale = np.asarray(metadata["center_mm"]), metadata["mm_per_network_unit"]
    mesh = trimesh.load_mesh(mesh_trial / "spur.obj", process=False)
    samples = np.loadtxt(mesh_trial / "spur_50k.xyz")
    labels = np.loadtxt(mesh_trial / "spur_50k_mask.txt", dtype=int)
    face_labels = np.loadtxt(mesh_trial / "spur_50k_fid.txt", dtype=int) + 1
    if len(face_labels) != len(mesh.faces) or samples.shape != (len(labels), 6):
        raise ValueError("The mesh and upstream training labels have incompatible sizes")
    owners, triangle_uv = cad_triangle_parameters(step, metadata["deflection_mm"], mesh, center, scale)
    barycentric, ids = nearest_training_triangles(mesh, samples[:, :3], labels, face_labels)
    parameters = np.einsum("ni,nij->nj", barycentric, triangle_uv[ids])
    def evaluate(uv, face_ids, name):
        params_path = output / f"{name}-uv.txt"
        samples_path = output / f"{name}-samples.csv"
        np.savetxt(params_path, np.column_stack((face_ids, uv)), fmt=("%d", "%.17g", "%.17g"))
        subprocess.run([executable.resolve(), "cad-surface-samples", step.resolve(), params_path, samples_path], check=True)
        return np.genfromtxt(samples_path, delimiter=",", skip_header=1, ndmin=2)

    cad = evaluate(parameters, owners[ids], "spur-cad")
    outside = cad[:, 0] != 1
    if outside.any():
        # A flat UV triangle can slightly overrun a curved trim. The nearest
        # vertex of the same triangle supplies parameters on the original face.
        vertex = np.argmax(barycentric[outside], axis=1)
        cad[outside] = evaluate(triangle_uv[ids[outside], vertex], owners[ids[outside]], "spur-trim-vertex")
    if not (cad[:, 0] == 1).all() or not np.isfinite(cad).all():
        raise ValueError("Some training samples cannot be evaluated on their original CAD face")
    xyz, normals = cad[:, 1:4], cad[:, 4:]
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-9, rtol=0)
    correction_um = np.linalg.norm(xyz - samples[:, :3] * scale - center, axis=1) * 1000
    if np.max(correction_um[~outside]) > 10 or np.any(np.sum(samples[:, 3:] * normals, axis=1) <= 0):
        raise ValueError("The original CAD positions or normals disagree with their source triangles")
    np.savetxt(output / "spur_50k.xyz", np.column_stack(((xyz - center) / scale, normals)), fmt="%.16g")
    for file in ("spur_50k_mask.txt", "spur_50k_csg.conf", "spur-reference.npz"):
        shutil.copyfile(mesh_trial / file, output / file)
    metadata.update(training_geometry="original CAD parametric surface", training_samples=len(samples),
                    trim_vertex_fallbacks=int(outside.sum()),
                    paired_position_shift_p95_um=float(np.percentile(correction_um, 95)),
                    paired_position_shift_max_um=float(correction_um.max()),
                    preparation_s=time.perf_counter() - started)
    (output / "spur.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def contact_samples(witnesses):
    """Take the two Gauss locations along each recorded straight CAD contact."""
    with witnesses.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    active = {(row["face_a"], row["face_b"]) for row in rows if row["facing_interiors"] == "1"}
    if not active:
        raise ValueError("The supplied frame has no facing CAD candidates")
    for pair in sorted(active):
        boundary = [row for row in rows if (row["face_a"], row["face_b"]) == pair and row["facing_interiors"] == "0"]
        lo, hi = (pick(boundary, key=lambda row: float(row["bz_m"])) for pick in (min, max))
        p, q = (np.array([float(row[f"b{axis}_m"]) for axis in "xyz"]) for row in (lo, hi))
        unit = np.linalg.norm(q - p) / 2
        if unit <= 0:
            raise ValueError("The recorded CAD line endpoints coincide")
        for s in (1 / np.sqrt(3), -1 / np.sqrt(3)):
            x_mm = 1000 * ((p + q) / 2 + s * (q - p) / 2)
            yield x_mm, unit, "/".join(pair)


def references(step, output, count, executable, witnesses):
    import trimesh

    rng = np.random.default_rng(42)
    geometry = np.load(output / "spur-geometry.npz")
    triangles = trimesh.load_mesh(output / "spur.obj", process=False).faces
    sharp_vertices = np.unique(geometry["feature_segments"][:, :2])
    smooth = ~np.isin(triangles, sharp_vertices).any(axis=1)
    sampled_faces = rng.choice(np.flatnonzero(smooth), size=count, replace=False)
    points = geometry["vertices"][triangles[sampled_faces, 0]]
    normals = geometry["normals"][sampled_faces]
    segments = rng.choice(geometry["feature_segments"], size=count, replace=False)
    # The shared endpoints originate on the STEP edge, independently of the field.
    edge_points = geometry["vertices"][segments[:, 0]]
    edge_normals = geometry["normals"][segments[:, 2]] + geometry["normals"][segments[:, 3]]
    edge_normals /= np.linalg.norm(edge_normals, axis=1)[:, None]
    offsets_mm = np.array([-.05, -.005, -.001, 0., .001, .005, .05])
    xs, labels, targets = [], [], []
    for label, p, n in (("face", points, normals), ("sharp", edge_points, edge_normals)):
        samples = (p[:, None, :] + n[:, None, :] * offsets_mm[None, :, None]).reshape(-1, 3)
        xs.extend(samples)
        labels.extend([label] * len(samples))
        targets.extend(np.tile(offsets_mm, len(p)))
    weights, pairs = [0.] * len(xs), [""] * len(xs)
    if witnesses is not None:
        for x, weight, pair in contact_samples(witnesses):
            xs.append(x)
            labels.append("contact")
            targets.append(0.)
            weights.append(weight)
            pairs.append(pair)
    points_path = output / "spur-points-mm.txt"
    distances_path = output / "spur-cad-distances.csv"
    np.savetxt(points_path, xs, fmt="%.17g")
    start = time.perf_counter()
    subprocess.run([executable.resolve(), "cad-point-distance", step.resolve(), points_path, distances_path], check=True)
    wall = time.perf_counter() - start
    raw = np.genfromtxt(distances_path, delimiter=",", skip_header=1, ndmin=2)
    truth, grads = raw[:, 0], raw[:, 1:]
    if truth.shape != (len(xs),) or not np.isfinite(truth).all():
        raise ValueError("The CAD reference did not return one finite distance per point")
    surface = (np.asarray(targets) == 0) & (np.asarray(labels) != "contact")
    if not np.all(abs(truth[surface]) < 1e-5):
        raise ValueError("A CAD boundary sample is more than 10 nm from the source")
    mesh = trimesh.Trimesh(vertices=geometry["vertices"], faces=triangles, process=False)
    mesh_distance = -trimesh.proximity.signed_distance(mesh, np.asarray(xs))
    np.savez(output / "spur-reference.npz", points_mm=xs, distance_mm=truth, group=labels, offsets_mm=targets,
             weights_m=weights, contact_pair=pairs, gradient=grads, cad_wall_s=wall, mesh_distance_mm=mesh_distance)
    print(f"Computed {len(xs)} CAD reference distances in {wall:.3f} s")


def audit(output, model):
    import torch

    metadata = json.loads((output / "spur.json").read_text(encoding="utf-8"))
    reference = np.load(output / "spur-reference.npz")
    points = reference["points_mm"]
    unit = metadata["mm_per_network_unit"]
    center = np.array(metadata["center_mm"])
    query = torch.tensor((points - center) / unit, dtype=torch.float32)
    field = torch.jit.load(str(model), map_location="cpu").eval()
    torch.set_num_threads(1)
    with torch.inference_mode():
        predicted = field(query).reshape(-1).numpy() * unit
        field(query)
        started = time.perf_counter()
        for _ in range(100):
            field(query)
        query_us = (time.perf_counter() - started) * 1e6 / (100 * len(points))
    if not np.isfinite(predicted).all() or len(predicted) != len(points):
        raise ValueError("The NH-Rep field did not return one finite scalar per CAD point")
    query.requires_grad_(True)
    predicted_normals = torch.autograd.grad(field(query).sum(), query)[0].numpy()
    single = query[-1:].detach().requires_grad_(True)
    torch.autograd.grad(field(single).sum(), single)
    started = time.perf_counter()
    for _ in range(100):
        torch.autograd.grad(field(single).sum(), single)
    single_us = (time.perf_counter() - started) * 1e4
    norm = np.linalg.norm(predicted_normals, axis=1)
    predicted_normals /= np.maximum(norm, 1e-12)[:, None]
    reference_gradients = reference["gradient"]
    valid_gradient = np.isfinite(reference_gradients).all(axis=1)
    normal_errors_deg = np.rad2deg(np.arccos(np.clip(np.sum(predicted_normals * reference_gradients, axis=1), -1, 1)))
    actual = reference["distance_mm"]
    error_um = abs(predicted - actual) * 1000
    mesh_error_um = abs(reference["mesh_distance_mm"] - actual) * 1000
    groups = {}
    for name in ("face", "sharp"):
        selected = reference["group"] == name
        boundary = selected & (reference["offsets_mm"] == 0)
        sign = selected & (abs(actual) > 1e-6)
        defined = selected & valid_gradient
        groups[name] = {"queries": int(selected.sum()), "mean_distance_error_um": float(error_um[selected].mean()),
                        "p95_distance_error_um": float(np.percentile(error_um[selected], 95)),
                        "max_distance_error_um": float(error_um[selected].max()),
                        "max_boundary_error_um": float(error_um[boundary].max()),
                        "source_mesh_max_distance_error_um": float(mesh_error_um[selected].max()),
                        "within_0p125_um": int(np.count_nonzero(error_um[selected] <= .125)),
                        "normal_checks": int(defined.sum()),
                        "normal_error_p95_deg": float(np.percentile(normal_errors_deg[defined], 95)),
                        "wrong_sign": int(np.count_nonzero(np.sign(predicted[sign]) != np.sign(actual[sign]))),
                        "checked_sign": int(sign.sum())}
    training = json.loads((output / "spur-training.json").read_text(encoding="utf-8"))
    report = {**metadata, **training, "model": model.name, "gap_check_um": .125,
              "batched_cpu_query_us_per_point": query_us,
              "single_cpu_value_and_gradient_us": single_us,
              "point_reference_total_s_including_setup": float(reference["cad_wall_s"]), "groups": groups}
    pairs = reference["contact_pair"]
    report["frozen_pose_line_loads"] = []
    for pair in sorted(set(pairs) - {""}):
        selected = pairs == pair
        stiffness = 1e9 * reference["weights_m"][selected]
        result = {"face_pair": pair}
        for name, distance in (("cad", actual), ("mesh", reference["mesh_distance_mm"]), ("nhrep", predicted)):
            gap = distance[selected]
            result[name + "_gaps_um"] = (gap * 1000).tolist()
            result[name + "_load_N"] = float(np.dot(stiffness, np.maximum(.055 - gap, 0)) * .001)
        result["gap_budget_um_for_2pct_load"] = float(.02 * result["cad_load_N"] / stiffness.sum() * 1e6)
        report["frozen_pose_line_loads"].append(result)
    (output / "spur-audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def nhrep_composite(self, values, operation, convex=True):
    import torch

    if not hasattr(self, "_reduction_indices"):
        self._reduction_indices = {}
    key = (id(operation), values.device)
    if key not in self._reduction_indices:
        direct = [node for node in operation if not isinstance(node, list)]
        children = [node for node in operation if isinstance(node, list)]
        direct_ids = torch.tensor(direct, dtype=torch.int64, device=values.device) if direct else None
        child_ids = None
        if children and all(not isinstance(node, list) for child in children for node in child):
            width = max(map(len, children))
            # Pad with the last index so ties select the same feature as upstream.
            padded = [child + [child[-1]] * (width - len(child)) for child in children]
            child_ids = torch.tensor(padded, dtype=torch.int64, device=values.device)
        self._reduction_indices[key] = (direct_ids, child_ids, children)
    direct_ids, child_ids, children = self._reduction_indices[key]
    terms = [values.index_select(1, direct_ids)] if direct_ids is not None else []
    reduce = torch.max if convex else torch.min
    if child_ids is not None:
        combine = torch.min if convex else torch.max
        terms.append(combine(values[:, child_ids], dim=2).values)
    else:
        terms.extend(nhrep_composite(self, values, child, not convex) for child in children)
    return reduce(torch.cat(terms, dim=1), dim=1, keepdim=True).values


def train(upstream, output, epochs, batch, trial):
    import torch
    from pyhocon import ConfigFactory

    upstream = upstream.resolve()
    code = upstream / "code"
    output = output.resolve()
    for suffix in (".xyz", "_mask.txt", "_csg.conf"):
        if not (output / ("spur_50k" + suffix)).is_file():
            raise ValueError("Run the official NH-Rep feature sampler on spur.obj before training")
    if not torch.cuda.is_available():
        raise ValueError("The upstream NH-Rep training loop requires CUDA")
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    sys.path.insert(0, str(code))
    import utils.general as general

    # The upstream runner already passes absolute paths. This is its HOME-free
    # equivalent for a Windows host. Shape extraction only affects visual checks.
    general.concat_home_dir = lambda path: path
    entry = code / "conversion/run.py"
    sys.argv = [str(entry), "--cpu", "--nepoch", str(epochs), "--points_batch", str(batch), "--pt", str(output)]
    source = entry.read_text(encoding="utf-8")
    # Preserve the upstream per-branch draws and losses, using one index copy
    # and one differentiable gather in place of repeated GPU copies and slices.
    first = source.index("            indices = torch.empty(0,dtype=torch.int64).cuda()")
    last = source.index("            cur_data = self.data[indices]", first)
    source = source[:first] + """            picks = [
                patch_id[i][np.random.choice(patch_id_n[i], n_patch_batch if i < n_branch - 1 else n_patch_last, True)]
                for i in range(n_branch)
            ]
            indices = torch.from_numpy(np.concatenate(picks)).cuda()

""" + source[last:]
    first = source.index("            all_fi = torch.zeros([n_batchsize, 1], device = 'cuda')")
    last = source.index("            # manifold loss for patches", first)
    source = source[:first] + (
        "            all_fi = mnfld_pred_all.gather(1, (single_branch_mask_id + 1).unsqueeze(1))\n\n"
    ) + source[last:]
    source = source.replace("self.lr_schedules[0].get_learning_rate(0)", "float(self.lr_schedules[0].get_learning_rate(0))")
    source = source.replace("self.lr_schedules[i].get_learning_rate(epoch)",
                            "float(self.lr_schedules[i].get_learning_rate(epoch))")
    source = source.replace("        traced_script_module = torch.jit.trace(network, example)",
                            "        with torch.no_grad():\n"
                            "            network(example)\n"
                            "        traced_script_module = torch.jit.trace(network, example)")
    namespace = {"__name__": "nhrep_replay", "__file__": str(entry)}
    exec(compile(source, str(entry), "exec"), namespace)
    runner_type = namespace["ReconstructionRunner"]
    settings = ConfigFactory.parse_file(str(code / "conversion/setup.conf"))
    settings["train"]["input_path"] = str(output)
    settings["train"]["checkpoint_frequency"] = gcd(epochs, 1000)
    prefix = "gear_parametric_trial_" if trial == "parametric" else "gear_trial_"
    runner = runner_type(conf=settings, folder_prefix=prefix, file_prefix="spur_50k", expname="single_shape",
                         gpu_index="0", is_continue=False, checkpoint="latest", eval=False,
                         flag_list=True, points_batch=batch, nepochs=epochs)
    if not runner.flag_data_load:
        raise ValueError("The upstream NH-Rep runner did not load the gear training data")
    net = runner.network
    if batch < net.n_branch:
        raise ValueError("The batch must include at least one sample from every upstream patch")
    probes = torch.rand(128, net.n_branch, device="cuda", requires_grad=True)
    original = net.nested_cvx_output(probes, net.csg_tree, net.flag_convex)
    batched = nhrep_composite(net, probes, net.csg_tree, net.flag_convex)
    torch.testing.assert_close(original, batched, atol=0, rtol=0)
    torch.testing.assert_close(torch.autograd.grad(original.sum(), probes)[0],
                               torch.autograd.grad(batched.sum(), probes)[0], atol=0, rtol=0)
    type(net).nested_cvx_output = nhrep_composite
    runner.plot_shapes = lambda *args, **kwargs: None
    start = time.perf_counter()
    runner.run_nhrepnet_training()
    timing = {"epochs": epochs, "start_epoch": runner.startepoch, "batch": batch,
              "training_s": time.perf_counter() - start, "device": torch.cuda.get_device_name(),
              "patches": int(runner.network.n_branch),
              "upstream_commit": subprocess.check_output(
                  ["git", "-C", upstream, "rev-parse", "HEAD"], text=True).strip(),
              "sample_sha256": sha256((output / "spur_50k.xyz").read_bytes()).hexdigest()}
    (output / "spur-training.json").write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "prepare-parametric", "reference", "train", "audit"))
    parser.add_argument("source", type=Path, help="STEP solid, NH-Rep checkout for training, or field for audit")
    parser.add_argument("output", type=Path)
    parser.add_argument("--deflection-mm", type=float, default=.0001)
    parser.add_argument("--anchors", type=int, default=32, help="Points per geometry group for the CAD reference")
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--trial", choices=("mesh", "parametric"), default="mesh")
    parser.add_argument("--mesh-trial", type=Path, help="Existing mesh-sampled trial for prepare-parametric")
    parser.add_argument("--exe", type=Path, help="nc_contact executable for the CAD reference")
    parser.add_argument("--witnesses", type=Path, help="regions CSV for the optional frozen contact pose")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch <= 0 or args.anchors <= 0:
        parser.error("epochs, batch, and anchors must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "audit":
        audit(args.output, args.source)
        return
    if args.stage == "train":
        train(args.source, args.output, args.epochs, args.batch, args.trial)
        return
    if args.stage == "prepare-parametric":
        if args.mesh_trial is None or args.exe is None:
            parser.error("prepare-parametric requires --mesh-trial and --exe")
        if args.mesh_trial.resolve() == args.output.resolve():
            parser.error("prepare-parametric requires a separate output directory")
        prepare_parametric(args.source, args.mesh_trial, args.output, args.exe)
        return
    if args.stage == "reference":
        if args.exe is None:
            parser.error("reference requires --exe")
        references(args.source, args.output, args.anchors, args.exe, args.witnesses)
        return
    solid = read_solid(args.source)
    metadata = {"step": args.source.name, **prepare(solid, args.output, args.deflection_mm)}
    (args.output / "spur.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
