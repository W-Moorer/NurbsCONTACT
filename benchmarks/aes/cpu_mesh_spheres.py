"""Replay the existing two-sphere HHT trajectory and compare mesh and sphere contact times."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from cpu_mesh_replay import replay

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/engines"))
from rigid import LEGACY, LAYER, case_values


def swept_time(exe, method, mesh, t, separation, start, end):
    poses = "".join(f"0 0 0 1 0 0 0 0 0 {separation[k]:.17g} 1 0 0 0\n" for k in (start, end))
    output = subprocess.run([exe, "--method", method, "--detection", "continuous", mesh, mesh],
                            input=poses, capture_output=True, text=True, check=True, timeout=1800)
    hit = json.loads(output.stdout)["candidate"]
    if hit is None:
        raise ValueError(f"{method} missed the change of contact state at frame {start}")
    return float(t[start] + hit["fraction"] * (t[end] - t[start]))


def compare(exe, mesh, trace):
    quality = json.loads(mesh.with_suffix(".quality.json").read_text())
    source = ROOT / "results/engine-comparison/nurbs/offset-sphere.step"
    if hashlib.sha256(source.read_bytes()).hexdigest() != quality["step_sha256"]:
        raise ValueError("expected the Gmsh mesh of the legacy contact surface offset-sphere.step")
    data = np.genfromtxt(trace, names=True, delimiter=",")
    t = data["time_s"]
    height, _, _ = case_values("spheres", LEGACY)
    z = height + data["q1_m"] - data["q0_m"]
    gap = z - (2 * LEGACY.radius + LAYER)
    if not np.all(np.diff(t) > 0):
        raise ValueError("expected increasing time samples")
    rows = np.zeros((len(t), 8))
    rows[:, 0], rows[:, 3], rows[:, 6] = t, z, 1.
    with tempfile.TemporaryDirectory() as work:
        poses = Path(work) / "relative.txt"
        np.savetxt(poses, rows)
        geometry = replay(exe, [mesh, mesh], poses)
    if quality["mesh_sha256"] != geometry["mesh_sha256"][0]:
        raise ValueError("the mesh does not match its quality report")
    methods = {}
    reference_contours = None
    sphere_events = []
    changes = np.flatnonzero(np.diff((gap < 0).astype(int)))
    if len(changes) != 2:
        raise ValueError("expected one approach and one release on the sphere trajectory")
    for i in changes:
        sphere_events.append(float(t[i] - (t[i + 1] - t[i]) * gap[i] / (gap[i + 1] - gap[i])))
    for method, frames in geometry["methods"].items():
        counts = np.array([frame["intersection_pairs"] for frame in frames])
        changes = np.flatnonzero(np.diff((counts > 0).astype(int)))
        if len(changes) != 2 or counts[0] or counts[-1]:
            raise ValueError(f"unexpected contact transitions for {method}")
        i, j = map(int, changes)
        entry = swept_time(exe, method, mesh, t, z, i, i + 1)
        release = swept_time(exe, method, mesh, t, z, j + 1, j)
        peak = counts.argmax()
        compressed = int(np.argmin(z))
        sample = "".join(f"0 0 0 1 0 0 0 0 0 {z[k]:.17g} 1 0 0 0\n" for k in (peak,compressed))
        run = subprocess.run([exe, "--method", method, "--contours", mesh, mesh], text=True,
                             input=sample, capture_output=True, check=True, timeout=1800)
        peak_contours, contours = [json.loads(line)["contours"] for line in run.stdout.splitlines()]
        if (len(peak_contours) != 1 or len(peak_contours[0]["triangle_pairs"]) != counts[peak]
                or len(contours) != 1 or len(contours[0]["triangle_pairs"]) != counts[compressed]):
            raise ValueError("expected one connected boundary where the two spheres intersect")
        members = (peak_contours[0]["triangle_pairs"], contours[0]["triangle_pairs"])
        if reference_contours is not None and members != reference_contours:
            raise ValueError("inconsistent contour membership across broad phases")
        reference_contours = members
        area = np.asarray(contours[0]["vector_area_a_m2"], dtype=float)
        if area.shape != (3,) or not np.isfinite(area).all() or area[2] <= 0:
            raise ValueError("expected a closed loop oriented toward the upper sphere")
        smooth_area = np.pi * ((LEGACY.radius + LAYER / 2)**2 - (z[compressed] / 2)**2)
        methods[method] = dict(mesh_event_times_s=[entry, release], peak_intersecting_triangle_pairs=int(max(counts)),
                               peak_frame_contours=len(peak_contours), peak_frame_time_s=float(t[peak]),
                               max_compression_time_s=float(t[compressed]), projected_area_at_max_compression_m2=float(area[2]),
                               sphere_projected_area_same_pose_m2=float(smooth_area),
                               max_compression_normal_error_deg=float(np.degrees(np.arctan2(np.linalg.norm(area[:2]),area[2]))),
                               active_frames=int(np.count_nonzero(counts)))
        if len(methods) > 1 and not np.allclose([entry, release], next(iter(methods.values()))["mesh_event_times_s"],
                                                rtol=0, atol=1e-12):
            raise ValueError("contact event times disagree across broad phases")
    return dict(trace=trace.name, trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(), frames=len(t),
                mesh_sha256=quality["mesh_sha256"], step_sha256=quality["step_sha256"],
                mesh_size_m=quality["mesh_size_m"], triangles=quality["triangles"],
                sphere_radius_m=LEGACY.radius + LAYER / 2, sphere_event_times_same_trajectory_s=sphere_events,
                matching_boundary_witnesses=geometry["matching_boundary_witnesses"], methods=methods)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("executable", "mesh", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--trace", type=Path, default=ROOT / "results/engine-comparison/nurbs/spheres-HHT-250us.csv")
    args = parser.parse_args()
    result = compare(args.executable.resolve(), args.mesh.resolve(), args.trace)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
