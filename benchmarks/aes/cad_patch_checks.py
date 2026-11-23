"""Check the GPU load-producing point query against distance to the original STEP shells."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from cad_patch_cache import evaluate, normal


def samples(tri, tags, shift):
    rng = np.random.default_rng(701)
    curve = 2 * np.linalg.norm(shift, axis=2).max(axis=1)
    points, normals, kinds = [], [], []
    for name, mask in (("flat", curve < 1e-9), ("curved", curve > max(1e-6, np.percentile(curve, 65)))):
        ix = np.flatnonzero(mask)
        if not len(ix):
            continue
        ix = rng.choice(ix, min(12, len(ix)), replace=False)
        points.extend(evaluate(tri[ix], shift[ix], 0.26, 0.33))
        normals.extend(normal(tri[ix], shift[ix], 0.26, 0.33))
        kinds.extend([name] * len(ix))
    _, idx = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    idx = idx.reshape(-1, 3)
    edges = np.sort(np.stack((idx, np.roll(idx, -1, axis=1)), axis=2), axis=2)
    _, ids = np.unique(edges.reshape(-1, 2), axis=0, return_inverse=True)
    incidence = np.argsort(ids, kind="stable").reshape(-1, 2)
    shared = incidence[tags[incidence[:, 0] // 3] != tags[incidence[:, 1] // 3]]
    for left, right in shared[rng.choice(len(shared), min(16, len(shared)), replace=False)]:
        i, k = divmod(left, 3)
        j, l = divmod(right, 3)
        uv = ((0.5, 0), (0.5, 0.5), (0, 0.5))
        n1 = normal(tri[[i]], shift[[i]], *uv[k])[0]
        n2 = normal(tri[[j]], shift[[j]], *uv[l])[0]
        for name, direction in (("bisector", n1 + n2), ("off_bisector", n1 + 0.37 * n2)):
            if np.linalg.norm(direction) < 1e-7:
                continue
            points.append(evaluate(tri[[i]], shift[[i]], *uv[k])[0])
            normals.append(direction / np.linalg.norm(direction))
            kinds.append(name)
    offset = np.array([-0.050, -0.005, 0.005, 0.050])
    n = np.array(normals)
    p = (np.asarray(points)[:, None, :] + offset[None, :, None] * n[:, None, :]).reshape(-1, 3)
    return p, np.repeat(n, len(offset), axis=0), np.repeat(kinds, len(offset))


def reference(entry, deck, mesh_dir, output):
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.TopAbs import TopAbs_IN, TopAbs_OUT, TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS_Compound
    from OCP.gp import gp_Pnt
    from nhrep_gear import read_occt_solid

    with np.load(deck / entry["archive"]) as d:
        pts, n, k = samples(d["tri"], d["faces"], d["shift"])
    solid = read_occt_solid(mesh_dir / entry["step"])
    shells = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(shells)
    ex = TopExp_Explorer(solid, TopAbs_SHELL)
    while ex.More():
        builder.Add(shells, ex.Current())
        ex.Next()
    distance = BRepExtrema_DistShapeShape()
    distance.LoadS2(shells)
    distance.SetDeflection(1e-9)
    classify = BRepClass3d_SolidClassifier(solid)
    gaps = []
    directions = []
    alternatives = []
    owners = []
    for p in pts:
        point = gp_Pnt(*p)
        classify.Perform(point, 1e-8)
        if classify.State() not in (TopAbs_IN, TopAbs_OUT):
            raise ValueError("Ambiguous reference sign")
        distance.LoadS1(BRepBuilderAPI_MakeVertex(point).Vertex())
        distance.Perform()
        if not distance.IsDone():
            raise ValueError("Unresolved native distance")
        gaps.append(distance.Value() * (-1 if classify.State() == TopAbs_IN else 1))
        directions.append((p - np.array(distance.PointOnShape2(1).Coord())) / gaps[-1])
        for sol in range(1, distance.NbSolution() + 1):
            d = p - np.array(distance.PointOnShape2(sol).Coord())
            norm = np.linalg.norm(d)
            if norm > 1e-12 and abs(norm - abs(gaps[-1])) < 1e-7:
                alternatives.append(np.sign(gaps[-1]) * d / norm)
                owners.append(len(gaps) - 1)
    np.savez_compressed(
        output / (entry["step"] + ".npz"),
        points=pts,
        normals=n,
        kind=k,
        gap=gaps,
        directions=directions,
        alternatives=alternatives,
        owners=owners,
    )
    print(entry["step"], len(pts), flush=True)


def check(entry, deck, output, lanes, patch_search):
    import cupy as cp
    from resident_contact import ResidentMeshContact, contact_source

    with np.load(deck / entry["archive"]) as d:
        t, f, s = d["tri"], d["faces"], d["shift"]
    with np.load(output / (entry["step"] + ".npz")) as d:
        p, n, k, g, alts, owners = (
            d[a] for a in ("points", "normals", "kind", "gap", "alternatives", "owners")
        )
    result = {}
    for kind in ("grid", "patch"):
        m = ResidentMeshContact(
            t, t, f, f, pitch=0.8, backend=kind, lanes=lanes, patch_shift=s if kind == "patch" else None
        )
        m.samples = tuple(
            cp.asarray(v) for v in (p, n, np.zeros(len(p), np.int32), -n, np.ones(len(p), np.int32))
        )
        if len(p) > m.n:
            m.outside = cp.zeros((len(p), 4))
            m.nearby = cp.zeros_like(m.outside)
            m.neighbors = cp.empty((len(p), lanes), cp.int32)
            m.witness = cp.zeros((len(p), 8))
            m.capacity = 1 << (2 * len(p) - 1).bit_length()
            m.keys = cp.full(m.capacity, np.iinfo(np.uint64).max, cp.uint64)
            m.best = cp.zeros(m.capacity, cp.uint64)
            m.active = cp.empty(len(p), cp.int32)
        m.n = len(p)
        m.witness.fill(0)
        files = Path(__file__).resolve().parent
        code = (
            contact_source(files / "resident_contact.cu", kind == "patch")
            + """
extern "C" __global__ void check_probes(GEOM, const double* pose, unsigned long long* stages) {
    int i=(blockIdx.x*blockDim.x+threadIdx.x)/QUERY_LANES;
    if (i<n) sample_contact<QUERY_LANES,true>(i,pose,PASS,stages+7*i);
}"""
        )
        opts = ("--std=c++17", f"-DQUERY_LANES={lanes}", "-I" + str(files)) + (
            ("-DNC_CAD_PATCH",) if kind == "patch" else ()
        )
        if kind == "patch" and patch_search == "carrier-filter":
            opts += ("-DNC_CARRIER_FILTER",)
        if kind == "patch" and patch_search == "direct":
            opts += ("-DNC_DIRECT_PATCH_SEARCH",)
        mod = cp.RawModule(code=code, options=opts, enable_cooperative_groups=True)
        kernel = mod.get_function("check_probes")
        stage_counts = cp.empty((len(p), 7), cp.uint64)

        def query(offset):
            # Preserve pose history but clear the force accumulator.
            m.keys.fill(np.iinfo(np.uint64).max)
            m.best.fill(0)
            m.count.fill(0)
            m.witness[: len(p)].fill(0)
            stage_counts.fill(0)
            pose = np.eye(3, 4)
            pose[:, 3] = offset * np.array([1.0, -0.32, 0.57])
            kernel(((len(p) * lanes + 127) // 128,), (128,), (*m.args(), cp.asarray(pose), stage_counts))
            return m.witness[: len(p)].get()

        hits = query(0.0)
        inside = g < 0
        loaded = hits[:, 0] > 0
        error = np.abs(hits[:, 0] - np.maximum(-g, 0)) * 1000
        match = inside & loaded
        cos = np.full(len(p), -1.0)
        np.maximum.at(cos, owners, (alts * hits[owners, 4:7]).sum(axis=1))
        angles = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        result[kind] = dict(
            query_stage_counts=stage_counts.sum(axis=0).get().tolist(),
            samples=len(p),
            false_negative=int(np.sum(inside & ~loaded)),
            false_positive=int(np.sum(~inside & loaded)),
            depth_rms_um=float(np.sqrt(np.mean(error**2))),
            depth_max_um=float(error.max()),
            normal_p95_deg=float(np.percentile(angles[match], 95)) if match.any() else None,
            normal_p95_by_kind_deg={
                name: float(np.percentile(angles[match & (k == name)], 95))
                for name in np.unique(k)
                if (match & (k == name)).any()
            },
            maximum_by_kind_um={name: float(error[k == name].max()) for name in np.unique(k)},
        )
        np.savez_compressed(
            output / (entry["step"] + "-" + kind + ".npz"), depth_mm=hits[:, 0], normal=hits[:, 4:7]
        )
        if kind == "patch":
            offsets = [0.00001, 0.001, 0.015, 0.06, -0.018, -0.001, 0.0]
            warm = np.stack([query(offset) for offset in offsets])
            cold = []
            for offset in offsets:
                m.reset_history()
                cold.append(query(offset))
            cold = np.stack(cold)
            result[kind]["history_check"] = dict(
                queries=len(p) * len(offsets),
                load_flag_mismatches=int(np.sum((warm[:, :, 0] > 0) != (cold[:, :, 0] > 0))),
                max_depth_difference_um=1000 * float(np.abs(warm[:, :, 0] - cold[:, :, 0]).max()),
                max_normal_vector_difference=float(
                    np.linalg.norm(warm[:, :, 4:7] - cold[:, :, 4:7], axis=-1).max()
                ),
            )
    print(entry["step"], result, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("reference", "check"))
    parser.add_argument("--lanes", type=int, choices=(1, 2, 4, 8, 16, 32), default=4)
    parser.add_argument("--patch-search", choices=("bounded", "direct", "carrier-filter"), default="bounded")
    for name in ("deck", "mesh_dir", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    entries = json.loads((args.deck / "models.json").read_text())["parts"]
    if args.mode == "reference":
        with ProcessPoolExecutor(max_workers=len(entries)) as pool:
            jobs = [pool.submit(reference, e, args.deck, args.mesh_dir, args.output) for e in entries]
            for j in jobs:
                j.result()
    else:
        report = {e["step"]: check(e, args.deck, args.output, args.lanes, args.patch_search) for e in entries}
        (args.output / "query-check.json").write_text(json.dumps(report, indent=2) + "\n")
