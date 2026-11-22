"""Prepare one STEP solid for upstream NH-Rep and check the field against CAD."""

import argparse
import csv
from hashlib import sha256
import json
from math import gcd
from collections import defaultdict
from pathlib import Path
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

    direct = [node for node in operation if not isinstance(node, list)]
    children = [node for node in operation if isinstance(node, list)]
    terms = [values[:, direct]] if direct else []
    reduce = torch.max if convex else torch.min
    if children and all(not isinstance(node, list) for child in children for node in child):
        width = max(map(len, children))
        # Pad with the last index so ties select the same feature as upstream.
        indices = [child + [child[-1]] * (width - len(child)) for child in children]
        combine = torch.min if convex else torch.max
        terms.append(combine(values[:, indices], dim=2).values)
    else:
        terms.extend(nhrep_composite(self, values, child, not convex) for child in children)
    return reduce(torch.cat(terms, dim=1), dim=1, keepdim=True).values


def train(upstream, output, epochs, batch):
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
    namespace = {"__name__": "nhrep_replay", "__file__": str(entry)}
    exec(compile(source, str(entry), "exec"), namespace)
    runner_type = namespace["ReconstructionRunner"]
    settings = ConfigFactory.parse_file(str(code / "conversion/setup.conf"))
    settings["train"]["input_path"] = str(output)
    settings["train"]["checkpoint_frequency"] = gcd(epochs, 1000)
    runner = runner_type(conf=settings, folder_prefix="gear_trial_", file_prefix="spur_50k", expname="single_shape",
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
    parser.add_argument("stage", choices=("prepare", "reference", "train", "audit"))
    parser.add_argument("source", type=Path, help="STEP solid, NH-Rep checkout for training, or field for audit")
    parser.add_argument("output", type=Path)
    parser.add_argument("--deflection-mm", type=float, default=.0001)
    parser.add_argument("--anchors", type=int, default=32, help="Points per geometry group for the CAD reference")
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=16384)
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
        train(args.source, args.output, args.epochs, args.batch)
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
