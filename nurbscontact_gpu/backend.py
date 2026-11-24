"""Run the fixed-axis reducer with finite triangles and generalized-alpha on CUDA."""

import ctypes
import platform
import hashlib
import json
import time
from fractions import Fraction
from pathlib import Path

import cupy as cp
import numpy as np

from .bvh import TriangleMesh
from .scene import equations, read_model

KERNELS = Path(__file__).with_name("kernels")
SOURCE = KERNELS / "gpu_hinge_mesh.cu"
INTERSECTION = KERNELS / "gpu_triangle_intersections.cu"


def module(n=1, pairs=1, warps=1, exchange=False):
    return cp.RawModule(
        code=INTERSECTION.read_text() + SOURCE.read_text(),
        options=(
            "--std=c++14",
            "--fmad=false",
            f"-DNC_COORDS={n}",
            f"-DNC_PAIRS={pairs}",
            f"-DNC_WARPS={warps}",
            f"-DNC_EXCHANGE={int(exchange)}",
        ),
    )


def pack_mesh(path):
    with np.load(path) as data:
        tri = np.ascontiguousarray(data["tri"] * 0.001)
    vertices, faces = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    faces = faces.reshape(-1, 3)
    return tri, vertices, faces


def make_windows(mod, source, target, tree, b, bins, band):
    """One nominal cycle only seeds candidate lists. Motion bounds cover both independent hinges."""
    speed_ratio = Fraction(float(b[1, 8] / b[0, 8])).limit_denominator(32)
    if abs(float(speed_ratio) - b[1, 8] / b[0, 8]) > 1e-12:
        raise ValueError(
            "This periodic window builder requires a commensurate pair of nominal speeds"
        )
    period = 2 * np.pi * speed_ratio.denominator
    ratio = float(speed_ratio)
    qa = np.arange(bins) * period / bins
    qb = ratio * qa + b[1, 12] - ratio * b[0, 12]
    a, beta = b[0, 3] * (qa + b[0, 7]), b[1, 3] * (qb + b[1, 7])
    offset = 0.001 * (b[0, 4:7] - b[1, 4:7])
    pose = np.zeros((bins, 3, 4))
    pose[:, 0, 0] = pose[:, 1, 1] = np.cos(a - beta)
    pose[:, 1, 0] = np.sin(a - beta)
    pose[:, 0, 1] = -pose[:, 1, 0]
    pose[:, 2, 2] = 1
    pose[:, 0, 3] = np.cos(beta) * offset[0] + np.sin(beta) * offset[1]
    pose[:, 1, 3] = -np.sin(beta) * offset[0] + np.cos(beta) * offset[1]
    pose[:, 2, 3] = offset[2]
    radius = [np.linalg.norm(t[:, :, :2], axis=2).max() for t in (source, target)]
    pad = (radius[0] + abs(ratio) * radius[1]) * 0.5 * period / bins + radius[1] * band + 1e-12
    arg = (
        cp.asarray(source),
        np.int32(len(source)),
        tree.triangles,
        tree.ids,
        tree.bounds,
        tree.links,
        np.int32(len(tree.links)),
        cp.asarray(pose),
        np.float64(pad),
    )
    kernel = mod.get_function("window_pairs")
    counts = cp.zeros(bins, cp.int32)
    offsets = cp.zeros(bins + 1, cp.int32)
    pairs = cp.empty((1, 2), cp.int32)
    for write in (0, 1):
        for pos in range(0, bins, 8):
            kernel(
                ((len(source) + 127) // 128, min(8, bins - pos)),
                (128,),
                (*arg, np.int32(pos), counts, offsets, pairs, np.int32(write)),
            )
        if not write:
            totals = counts.get()
            prefix = np.r_[0, np.cumsum(totals, dtype=np.int64)]
            if prefix[-1] > np.iinfo(np.int32).max:
                raise OverflowError("Triangle window offsets exceed 32 bits")
            offsets = cp.asarray(prefix, dtype=cp.int32)
            pairs = cp.empty((int(prefix[-1]), 2), cp.int32)
            counts.fill(0)
    cp.cuda.get_current_stream().synchronize()
    return dict(offsets=offsets.get(), pairs=pairs.get(), period=period, ratio=ratio, band=band)


def export_scene(scene, deck, duration=5.0, step=0.0001):
    """Preserve the solver's mesh face order and dimensional conventions."""
    import meshio

    scene, deck = Path(scene), Path(deck)

    bodies, _ = read_model(scene)
    deck.mkdir(parents=True, exist_ok=True)
    equations(deck, h=step, duration=duration, scene=scene)
    manifest = []
    for i, body in enumerate(bodies):
        path = body[0].with_suffix(".msh")
        mesh = meshio.read(path)
        triangles = mesh.points[mesh.get_cells_type("triangle")] * 1000
        archive = f"body-{i}.npz"
        np.savez_compressed(deck / archive, tri=triangles)
        manifest.append(
            dict(
                archive=archive,
                mesh=path.name,
                triangles=len(triangles),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    (deck / "models.json").write_text(json.dumps(dict(parts=manifest), indent=2) + "\n")


def prepare(deck, output, bins=4096, band=0.0025):
    """Build conservative per-pair windows from an exported device deck."""
    deck, output = Path(deck), Path(output)
    if bins < 2 or not np.isfinite(band) or band <= 0:
        raise ValueError("Invalid angular window bounds")
    output.mkdir(parents=True, exist_ok=True)
    cfg = np.load(deck / "assembly.npz")
    parts = json.loads((deck / "models.json").read_text())["parts"]
    mod = module()
    for stage, ids in enumerate(cfg["pairs"]):
        start = time.perf_counter()
        files = [deck / parts[i]["archive"] for i in ids]
        first, second = [pack_mesh(path) for path in files]
        tree = TriangleMesh(second[1], second[2], leaf_size=6)
        windows = make_windows(mod, first[0], second[0], tree, cfg["bodies"][ids], bins, band)
        key = signature(files, cfg["bodies"][ids])
        np.savez(
            output / f"windows-{stage}.npz",
            **windows,
            signature=key,
            prep_s=time.perf_counter() - start,
        )
        sizes = np.diff(windows["offsets"])
        print(
            f"stage {stage+1}: {len(sizes)} windows, candidates median {np.median(sizes)}, "
            f"max {sizes.max()}, build {time.perf_counter()-start:.2f} s",
            flush=True,
        )


def signature(files, bodies):
    key = hashlib.sha256(b"fixed-axis-conservative-windows-v1")
    for path in files:
        key.update(path.read_bytes())
    key.update(np.ascontiguousarray(bodies).tobytes())
    return key.hexdigest()


def run(
    deck,
    work,
    destination,
    method="mesh-wrench",
    steps=None,
    geometry=None,
    warps=4,
    stream_io=False,
    exchange_lib=None,
    *,
    exchange=False,
):
    """Integrate a device deck, write response.npz and timing.json, and return its timing report."""
    if sum((bool(exchange), exchange_lib is not None, bool(stream_io))) > 1:
        raise ValueError("Select a single host/device transfer mode")
    if exchange:
        filename = "nc_hinge_exchange.dll" if platform.system() == "Windows" else "libnc_hinge_exchange.so"
        exchange_lib = Path(__file__).with_name(filename)
        if not exchange_lib.is_file():
            raise FileNotFoundError("Install the backend wheel for --exchange or pass --exchange-lib PATH")
    if exchange_lib is not None:
        exchange_lib = Path(exchange_lib)
    if geometry is not None:
        geometry = Path(geometry)
    deck, work = Path(deck), Path(work)
    setup = time.perf_counter()
    cfg = np.load(deck / "assembly.npz")
    mapping = cfg["mapping"]
    pairs = cfg["pairs"]
    ncoord = len(cfg["coord"])
    npair = len(pairs)
    if not 1 <= npair * warps <= 32 or not 1 <= ncoord <= min(32 * npair, 32):
        raise ValueError("GPU block capacity exceeded")
    paircoords = np.ascontiguousarray(mapping[pairs], dtype=np.int32)
    if np.any(paircoords[:, 0] == paircoords[:, 1]):
        raise ValueError("Contact within one rigidly connected coordinate")
    bodies = cfg["bodies"]
    if np.any(bodies[:, 0:3]) or np.any(np.abs(bodies[:, 3]) != 1):
        raise ValueError("Expected parallel Z hinges")
    geometry = geometry or work / ("anchors.npz" if method == "mesh-anchors" else "geometry.npz")
    g = np.load(geometry)
    table = g["table"]
    _, phases, layers, _ = table.shape
    parts = json.loads((deck / "models.json").read_text())["parts"]
    for key, actual in (
        ("joint_poses", bodies[:, 1:8]),
        ("joint_map", mapping),
        ("pair_ids", pairs),
    ):
        if not np.array_equal(g[key], actual):
            raise ValueError(f"Stale geometry cache: {key}")
    if table.shape[-1] != 3 or not np.allclose(
        g["origin"],
        cfg["coord"][paircoords[:, 1], 3] - g["ratio"] * cfg["coord"][paircoords[:, 0], 3],
        atol=1e-14,
        rtol=0,
    ):
        raise ValueError("Geometry table and reference angles differ")
    if len(table) != npair:
        raise ValueError("Geometry and assembly pair counts differ")
    if "mesh_sha256" in g and g["mesh_sha256"].tolist() != [part["sha256"] for part in parts]:
        raise ValueError("Geometry table and input meshes differ")
    joint_body = np.array([np.flatnonzero(mapping == i)[0] for i in range(ncoord)])
    mod = module(ncoord, npair, warps, exchange_lib is not None)
    seed = mod.get_function("seed_windows")
    advance = mod.get_function("advance")
    devices = []
    ptrs = []
    window_bytes = 0
    window_prep = []
    for k, ids in enumerate(pairs):
        files = [deck / parts[i]["archive"] for i in ids]
        mesh = []
        for path in files:
            with np.load(path) as data:
                mesh.append(cp.asarray(data["tri"] * 0.001))
        w = np.load(work / f"windows-{k}.npz")
        if k and len(w["offsets"]) - 1 != bins:
            raise ValueError("All pair caches must use the same window count")
        if "prep_s" in w:
            window_prep.append(w["prep_s"].item())
        if str(w["signature"]) != signature(files, bodies[ids]):
            raise ValueError("Stale candidate window cache")
        b = bodies[ids]
        if float(w["period"]) != g["period"][k] or float(w["ratio"]) != g["ratio"][k]:
            raise ValueError("Map convention mismatch")
        par = cp.asarray(
            np.r_[
                b[0, 4:7] * 0.001,
                b[1, 4:7] * 0.001,
                b[:, 3],
                b[:, 7],
                w["period"],
                w["ratio"],
                g["origin"][k],
                w["band"],
                cfg["stiffness"][k],
            ]
        )
        offsets = cp.asarray(w["offsets"])
        candidates = cp.asarray(w["pairs"])
        bins = len(offsets) - 1
        window_bytes += offsets.nbytes + candidates.nbytes
        seeds = cp.empty((bins, 32), cp.int32)
        seed((bins,), (128,), (par, np.int32(bins), offsets, candidates, *mesh, seeds))
        if method == "mesh-anchors":
            addresses = g["anchor_offsets"][
                k * phases * layers : (k + 1) * phases * layers + 1
            ].copy()
            anchors = cp.asarray(g["anchors"][addresses[0] : addresses[-1]])
            addresses -= addresses[0]
        else:
            addresses = np.zeros(1, np.int32)
            anchors = cp.empty(1)
        row = [
            par,
            *mesh,
            candidates,
            offsets,
            seeds,
            cp.asarray(table[k]),
            cp.asarray(addresses),
            anchors,
        ]
        devices += row
        ptrs.append([v.data.ptr for v in row])
    pointer = cp.asarray(ptrs, dtype=cp.uint64)
    coord = cp.asarray(np.ascontiguousarray(cfg["coord"]))
    nominal = bodies[joint_body, 8].copy()
    graph = cp.asarray(paircoords)
    stride = 3 * ncoord + 3 * npair
    n = len(cfg["loads"]) - 1 if steps is None else steps
    if n < 1 or n >= len(cfg["loads"]):
        raise ValueError("Invalid step count")
    vel = cp.asarray(nominal)
    h = cfg["h"].item()
    host_load = cfg["loads"][: n + 1].copy()
    data = cp.empty((n + 1, stride), cp.float64)
    history = cp.zeros((n + 1, 2), cp.int32)
    ticks = cp.zeros(n + 1, cp.uint64)
    status = cp.zeros(1, cp.int32)
    slip = g["slip"]
    ik = np.int32
    cp.cuda.get_current_stream().synchronize()
    chan = cp.cuda.get_current_stream()
    host_ns = np.zeros(n + 1, dtype=np.int64)
    if stream_io:
        upload = cp.cuda.alloc_pinned_memory(host_load.nbytes)
        inputs = np.frombuffer(upload, dtype=np.float64, count=host_load.size).reshape(
            host_load.shape
        )
        inputs[:] = host_load
        download = cp.cuda.alloc_pinned_memory(3 * npair * 8)
    init = mod.get_function("initialize_state")
    mode = ik({"wrench-only": 0, "mesh-wrench": 1, "mesh-anchors": 2}[method])
    header = (pointer, graph, coord, vel)
    geometry_args = (
        ik(bins),
        ik(phases),
        ik(layers),
        np.float64(slip[0]),
        np.float64(slip[1] - slip[0]),
        mode,
    )
    mailbox_ptr = 0
    try:
        if exchange_lib:
            if stream_io:
                raise ValueError("Choose one transfer mode")
            # Mapped page remains owned by this process until all kernels have finished.
            mailbox_ptr = cp.cuda.runtime.hostAlloc(4096, cp.cuda.runtime.hostAllocMapped)
            ctypes.memset(mailbox_ptr, 0, 4096)
            driver = (
                ctypes.WinDLL("nvcuda")
                if platform.system() == "Windows"
                else ctypes.CDLL("libcuda.so.1")
            )
            address = ctypes.c_uint64()
            rc = driver.cuMemHostGetDevicePointer_v2(
                ctypes.byref(address), ctypes.c_void_p(mailbox_ptr), 0
            )
            if rc:
                raise RuntimeError(f"cuMemHostGetDevicePointer returned {rc}")
            callback = ctypes.CDLL(str(exchange_lib.resolve())).exchange_steps
            callback.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            received = np.zeros((n + 1, 3 * npair))
            event = cp.cuda.Event(disable_timing=True)
        else:
            address = ctypes.c_uint64(0)
        setup_s = time.perf_counter() - setup
        started = time.perf_counter()
        if exchange_lib:
            load = cp.asarray(host_load[:1])
        elif stream_io:
            load = cp.empty(host_load.shape, dtype=cp.float64)
            cp.cuda.runtime.memcpyAsync(load.data.ptr, upload.ptr, 8 * ncoord, 1, chan.ptr)
        else:
            load = cp.asarray(host_load)
        init((1,), (32 * npair * warps,), (*header, load, *geometry_args, data, status))
        chunk = 1 if stream_io else 256
        step_args = (*header, load, np.float64(h), *geometry_args)
        for begin in range(0, n, chunk):
            end = min(n, begin + chunk)
            if stream_io:
                stamp = time.perf_counter_ns()
                pos = end * ncoord * 8
                cp.cuda.runtime.memcpyAsync(
                    load.data.ptr + pos, upload.ptr + pos, 8 * ncoord, 1, chan.ptr
                )
            advance(
                (1,),
                (32 * npair * warps,),
                (
                    *step_args,
                    ik(begin),
                    ik(end),
                    data,
                    history,
                    ticks,
                    status,
                    np.uint64(address.value),
                ),
            )
            if exchange_lib:
                event.record(chan)
                _ = event.done  # Flush WDDM dispatch before entering the host wait.
                failed = callback(
                    mailbox_ptr,
                    begin,
                    end,
                    ncoord,
                    3 * npair,
                    host_load.ctypes.data,
                    received.ctypes.data,
                    host_ns.ctypes.data,
                )
                if failed:
                    break
            if stream_io:
                cp.cuda.runtime.memcpyAsync(
                    download.ptr,
                    data.data.ptr + 8 * (end * stride + 3 * ncoord),
                    3 * npair * 8,
                    2,
                    chan.ptr,
                )
                chan.synchronize()
                host_ns[end] = time.perf_counter_ns() - stamp
        values = data.get()
        hist = history.get()
        timings = ticks.get()
        error = int(status.get()[0])
        elapsed = time.perf_counter() - started
    finally:
        if mailbox_ptr:
            try:
                chan.synchronize()
            finally:
                cp.cuda.runtime.freeHost(mailbox_ptr)
    complete = int(np.count_nonzero(hist[:, 0]))
    if error:
        np.savez(
            work / f"failure-{method}.npz",
            state=values[: complete + 1],
            iterations=hist[: complete + 1],
        )
        raise RuntimeError(
            f"GPU stopped after {complete} steps, status {error} (1 domain, 2 invalid geometry, 3 Newton, 4 normal, 5 host timeout)"
        )
    if complete != n or not np.isfinite(values).all():
        raise RuntimeError("Incomplete or nonfinite GPU trace")
    dest = Path(destination)
    dest.mkdir(exist_ok=True, parents=True)
    t = np.arange(n + 1) * h
    q = values[:, :ncoord]
    delta = q - q[0]
    contact = values[:, 3 * ncoord :].reshape(n + 1, npair, 3)
    if exchange_lib:
        np.testing.assert_array_equal(received[1:], contact[1:].reshape(n, -1))
    te = delta[:, paircoords[:, 1]] - delta[:, paircoords[:, 0]] * g["ratio"]
    generalized = np.zeros_like(q)
    for k, (a, b) in enumerate(paircoords):
        generalized[:, a] += contact[:, k, 1]
        generalized[:, b] += contact[:, k, 2]
    torque = generalized * bodies[joint_body, 3]
    np.savez_compressed(
        dest / "response.npz",
        time_s=t,
        q=q,
        speed_rad_s=values[:, ncoord : 2 * ncoord] * bodies[joint_body, 3],
        transmission_error_mrad=1e3 * te,
        stage_normal_load_N=contact[:, :, 0],
        joint_contact_torque_Nm=torque,
        output_contact_torque_Nm=torque[:, -1],
        newton_iterations=hist[:, 0],
        full_scan_warps=hist[:, 1],
        step_ns=timings,
        host_io_step_ns=host_ns,
    )
    device = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    report = dict(
        device=device,
        method=method,
        steps=n,
        step_s=h,
        physical_s=n * h,
        online_wall_s=elapsed,
        setup_s=setup_s,
        geometry_preparation_s=float(g["prep_s"]),
        step_us_p50_p95_p99_max=np.percentile(timings[1:] / 1000, [50, 95, 99, 100]).tolist(),
        steps_over_budget=int(np.count_nonzero(timings[1:] > h * 1e9)),
        newton_max=int(hist.max(axis=0)[0]),
        peak_output_Nm=float(torque[:, -1].max()),
        mean_output_Nm=float(torque[:, -1].mean()),
        phases=phases,
        layers=layers,
        geometry_windows=bins,
        warps_per_pair=warps,
        window_cache_bytes=window_bytes,
        window_preparation_s=window_prep,
        geometry_file_bytes=geometry.stat().st_size,
    )
    report["delivery"] = (
        "one load upload and one wrench download per step"
        if stream_io
        else "device-resident trajectory"
    )
    if exchange_lib:
        report["delivery"] = "synchronous mapped command and wrench exchange per step"
    if stream_io or exchange_lib:
        report["host_step_us_p50_p95_p99_max"] = np.percentile(
            host_ns[1:] / 1000, [50, 95, 99, 100]
        ).tolist()
        report["host_steps_over_budget"] = int(np.count_nonzero(host_ns[1:] > h * 1e9))
    (dest / "timing.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report
