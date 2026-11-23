"""Measure native CPU mesh memory on the archived two-gear pose replay."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time

import numpy as np
import psutil


def measure(exe, meshes, poses, reference, method):
    assets = [meshes / name for name in ("spur-1a.msh", "spur-1b.msh")]
    quality = [json.loads(p.with_suffix(".quality.json").read_text()) for p in assets]
    source = json.loads(reference.read_text())
    pose_sha256 = hashlib.sha256(poses.read_bytes()).hexdigest()
    if pose_sha256 != source["pose_sha256"]:
        raise ValueError("pose file differs from the archived dense replay")
    known = source["compact_grading"]["surface_triangle_pairs"]
    for path, report in zip(assets, quality, strict=True):
        with path.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != report["mesh_sha256"]:
                raise ValueError("mesh differs from its geometry quality report")
    frames = np.loadtxt(poses, ndmin=2)
    if len(frames) != len(known) or frames.shape[1] != 15:
        raise ValueError("expected archived world poses matching the reference trace")
    outputs = []
    started = time.perf_counter()
    with tempfile.TemporaryFile() as errors, subprocess.Popen(
        [exe, "--method", method, "--any", *assets],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=errors,
        text=True,
        bufsize=1,
    ) as child:
        monitor = psutil.Process(child.pid)
        deadline = threading.Timer(1800, child.kill)
        deadline.start()
        try:
            for row in frames:
                child.stdin.write(" ".join(format(x, ".17g") for x in row[1:]) + "\n")
                child.stdin.flush()
                message = child.stdout.readline()
                if not message:
                    raise RuntimeError("native mesh query exited without a result")
                outputs.append(json.loads(message))
                usage = monitor.memory_info()
                if len(outputs) == 1:
                    first_rss = usage.rss
                if not hasattr(usage, "peak_wset"):
                    raise ValueError(
                        "the archived Windows memory experiment requires peak working-set counters"
                    )
            child.stdin.close()
            if child.wait() != 0:
                raise RuntimeError("native mesh query failed")
        finally:
            deadline.cancel()
            if child.poll() is None:
                child.kill()
                child.wait()
    changed = [i for i, (r, k) in enumerate(zip(outputs, known, strict=True)) if r["overlap"] != bool(k)]
    return dict(
        method=method,
        pose_sha256=pose_sha256,
        triangles=[q["triangles"] for q in quality],
        mesh_sha256=[q["mesh_sha256"] for q in quality],
        sampled_surface_error_um=[q["max_sampled_surface_error_m"] * 1e6 for q in quality],
        min_sicn=[q["min_sicn"] for q in quality],
        file_bytes=sum(p.stat().st_size for p in assets),
        first_frame_working_set_bytes=first_rss,
        final_working_set_bytes=usage.rss,
        peak_working_set_bytes=usage.peak_wset,
        final_private_bytes=usage.private,
        elapsed_s=time.perf_counter() - started,
        frames=len(frames),
        positive_frames=sum(r["overlap"] for r in outputs),
        differing_from_dense_frame_indices=changed,
        mesh_positive=[r["overlap"] for r in outputs],
        median_decision_ms=1000 * float(np.median([r["query_s"] for r in outputs])),
        measurement="One native nc_mesh process, two complete solids, resident OBB/AABB trees, discrete predicates and containment; Windows working set includes DLLs and temporary allocations.",
    )


def measured_run(command):
    started = time.perf_counter()
    peak_rss = peak_commit = 0
    with tempfile.TemporaryFile() as errors, subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=errors,
    ) as proc:
        monitor = psutil.Process(proc.pid)
        try:
            while proc.poll() is None:
                try:
                    info = monitor.memory_info()
                    peak_rss = max(peak_rss, getattr(info, "peak_wset", info.rss))
                    peak_commit = max(peak_commit, getattr(info, "peak_pagefile", 0))
                except psutil.NoSuchProcess:
                    break
                if time.perf_counter() - started >= 1800:
                    raise TimeoutError("HHT memory run reached 30 minutes")
                time.sleep(0.025)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        if proc.returncode:
            errors.seek(0)
            detail = errors.read().decode(errors="replace").strip()
            raise RuntimeError(f"{command[0]} exited with code {proc.returncode}. {detail}")
    result = dict(process_s=time.perf_counter() - started,
                  observed_peak_working_set_bytes=peak_rss, monitor_interval_s=0.025)
    if peak_commit:
        result["observed_peak_commit_bytes"] = peak_commit
    return result


def dynamics(exe, meshes, scene, method, dt, duration, work, region_scope=None, event_tolerance=0., solver="penalty"):
    work.mkdir(parents=True, exist_ok=True)
    prefix = work / "response"
    process = measured_run([
        exe, scene, "0", str(dt), str(duration), "1", prefix,
        "--method", method, "--solver", solver,
        *([] if meshes is None else ["--mesh-dir", meshes]),
        *(["--mesh-region-scope", region_scope] if region_scope else []),
        *(["--event-tol-s", str(event_tolerance)] if event_tolerance else []),
    ])
    data = np.genfromtxt(prefix.with_suffix(".csv"), names=True, delimiter=",")
    with prefix.with_suffix(".stats.csv").open() as f:
        stats = next(csv.DictReader(f))
    if abs(data["time_s"][-1] - duration) > 1e-9 or not np.max(data["p0_normal_N"]) > 0:
        raise ValueError("expected a completed loaded gear run")
    result = dict(
        duration_s=duration,
        dt_s=dt,
        steps=int(stats["steps"]),
        active_frames=int((data["p0_normal_N"] > 0).sum()),
        scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
        trace_sha256=hashlib.sha256(prefix.with_suffix(".csv").read_bytes()).hexdigest(),
        normal_impulse_Ns=float(np.trapezoid(data["p0_normal_N"], data["time_s"]) if solver=="penalty"
                                else np.diff(data["time_s"]) @ data["p0_normal_N"][1:]),
        peak_normal_N=float(data["p0_normal_N"].max()),
        subdivisions=int(stats["subdivisions"]),
        integration_s=float(stats["wall_s"]),
        **process,
    )
    if "cad_boolean_queries" in stats:
        result.update({k:int(stats[k]) for k in ("ccd_checks","ccd_rejected","ccd_unresolved","ccd_nodes",
                                                  "cad_boolean_queries","cad_event_distances",
                                                  "chen_patch_queries","chen_confirmed_candidates")})
        result.update({k:int(stats[k]) for k in ("tdib_cells","tdib_candidates","tdib_unresolved",
                                                 "cad_box_tests","cad_extrema","cad_classifications",
                                                 "cad_region_cache_hits","cad_distance_cache_hits",
                                                 "cad_witness_tests","cad_witness_improvements") if k in stats})
        result.update(event_tolerance_s=float(stats.get("event_tolerance_s",event_tolerance)), setup_s=float(stats["setup_s"]),
                      evaluations=int(stats["force_evaluations"]))
    if meshes is not None and "mesh_boolean_queries" in stats:
        result.update(
            region_scope=region_scope or "local",
            evaluations=int(stats["force_evaluations"]),
            boolean_queries=int(stats["mesh_boolean_queries"]),
            boolean_input_faces=int(stats["mesh_boolean_input_faces"]),
            local_solid_uses=int(stats["mesh_local_solid_uses"]),
            local_solid_builds=int(stats["mesh_local_solid_builds"]),
            detection_s=float(stats["mesh_detection_s"]),
            placement_s=float(stats["mesh_placement_s"]),
            boolean_s=float(stats["mesh_boolean_s"]),
        )
    return result


def compare_gear_traces(first, second):
    tables = [np.genfromtxt(p, names=True, delimiter=",") for p in (first, second)]
    a, b = tables
    tolerance = 8 * np.finfo(float).eps * max(abs(a["time_s"][-1]), abs(b["time_s"][-1]))
    if a.shape != b.shape or not np.allclose(a["time_s"], b["time_s"], rtol=0, atol=tolerance):
        raise ValueError("gear responses have different time grids")
    t = a["time_s"]
    force_a, force_b = a["p0_normal_N"], b["p0_normal_N"]
    return dict(
        frames=len(t),
        duration_s=t[-1],
        trace_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in (first, second)],
        peak_normal_N=[float(force_a.max()), float(force_b.max())],
        normal_impulse_Ns=[float(np.trapezoid(f, t)) for f in (force_a, force_b)],
        normal_force_rms_difference_N=float(np.sqrt(np.mean((force_a - force_b) ** 2))),
        different_active_frames=int(np.count_nonzero((force_a > 0) != (force_b > 0))),
        max_shaft_angle_difference_rad=[float(np.abs(a[key] - b[key]).max()) for key in ("q0", "q1")],
        max_shaft_speed_difference_rad_s=[float(np.abs(a[key] - b[key]).max()) for key in ("v0", "v1")],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("exe", "meshes", "poses", "reference", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--method", choices=["mesh-aabb", "mesh-obb", "mesh-sap"], default="mesh-obb")
    parser.add_argument("--mesh-region-scope", choices=["local", "full"])
    parser.add_argument("--dynamics-exe", type=Path)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--dt", type=float, default=0.00005)
    parser.add_argument("--duration", type=float, default=0.005)
    parser.add_argument(
        "--compare-trace", type=Path, help="previous gear response on the identical time grid"
    )
    args = parser.parse_args()
    result = measure(args.exe.resolve(), args.meshes, args.poses, args.reference, args.method)
    if args.dynamics_exe:
        if not args.scene or not args.work:
            parser.error("--dynamics-exe requires --scene and --work")
        result["dynamics"] = dynamics(
            args.dynamics_exe.resolve(),
            args.meshes,
            args.scene,
            args.method,
            args.dt,
            args.duration,
            args.work,
            args.mesh_region_scope,
        )
        if args.compare_trace:
            result["compared_with_previous_run"] = compare_gear_traces(
                args.compare_trace, args.work / "response.csv"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
