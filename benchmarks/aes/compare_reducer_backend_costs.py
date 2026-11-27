"""Report archived CPU and measured GPU costs for the five-second reducer."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cpu_cad_pressure_report import history


def read(path):
    return json.loads(path.read_text())


def main(a):
    archive = Path(__file__).resolve().parents[2] / "results/aes"
    frozen = read(archive / "cpu-bezier/standalone/simulation-timing/cpu-worker-scaling.json")
    cpu = {r["method"]: r for r in frozen["cpu_runs"]
           if r["variant"] == "fused" and r["steps"] == 50000 and r["threads"] == 4}
    for method, path in (("mesh-obb", a.cpu_mesh), ("cad-compiled", a.cpu_cad)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != cpu[method]["response_sha256"]:
            raise ValueError(f"unexpected archived {method} response")
    gpu = read(a.out / "comparison.json")["runs"]
    gpu_time = np.mean([gpu[k]["wall_s"] for k in ("final", "repeat")])
    sampled = read(a.cross / "mesh-dynamics.json")
    bvh = read(a.cross / "mesh-bvh.json")
    original_bvh = read(archive / "gpu-mesh-geometry/geometry.json")
    for key in ("pose_trace_sha256", "mesh_sha256", "hit_count_history_sha256"):
        if bvh[key] != original_bvh[key]:
            raise ValueError(f"mesh detector replay changed {key}")
    rays = read(a.cross / "rays/comparison.json")
    original_rays = read(archive / "gpu-mesh-geometry/exit-queries.json")
    if len(rays) != len(original_rays):
        raise ValueError("incomplete matched ray test")
    for r, old in zip(rays, original_rays, strict=True):
        for key in ("ray_sha256", "cad_hits", "mesh_hits", "different_hit_status", "cad_unresolved"):
            if r[key] != old[key]:
                raise ValueError(f"{r['part']} changed {key}")
    raw = np.load(a.cross / "mesh-dynamics.npy", allow_pickle=False)
    if sampled["terminated"] != "complete" or raw.shape[0] != 50001 or not np.isfinite(raw).all():
        raise ValueError("incomplete GPU mesh dynamics")
    curve = history(a.gpu_cad, a.scene)
    t = curve["time_s"]
    np.testing.assert_allclose(t, np.arange(len(raw)) * sampled["step_s"], atol=1e-11, rtol=0)
    labels = ("CPU mesh OBB", "CPU compiled CAD", "GPU sampled mesh", "GPU compiled CAD")
    seconds = [cpu["mesh-obb"]["wall_s"], cpu["cad-compiled"]["wall_s"], sampled["online_s"], gpu_time]
    torque = raw[:, 12]
    cad_torque = curve["output_torque_Nm"]
    report = {
        "device": sampled["gpu"], "cpu_runs_from_archive": cpu, "closed_loop_s": dict(zip(labels, seconds)),
        "sampled_gpu_mesh": sampled, "gpu_mesh_bvh_geometry_replay": bvh, "identical_rays": rays,
        "hinge_map_archived_median_s": float(np.median([
            r["online_wall_s"] for r in read(archive / "gpu-hinge-mesh/runs.json")["resident"]
        ])),
        "rms_sampled_gpu_mesh_minus_gpu_cad_torque_Nm": float(np.sqrt(np.mean((torque - cad_torque)**2))),
        "peak_sampled_mesh_cad_Nm": [float(max(torque)), float(max(cad_torque))],
        "sampled_response_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        "sampled_archive_sha256": read(archive / "long-mesh/dynamics.json")["raw_array_sha256"],
    }
    (a.out / "backend-costs.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.2), layout="constrained")
    bars = ax[0].barh(labels, seconds, color=["#7695a4", "#477583", "#dc9d60", "#b56235"])
    ax[0].bar_label(bars, fmt="%.1f s", padding=3)
    ax[0].set(xlabel="Online time for 5 s [s]", xlim=(0, max(seconds) * 1.20))
    ax[0].invert_yaxis()
    for key, color, label in (("mesh_grid", "#80a7b7", "Mesh grid"), ("mesh", "#31647c", "Mesh BVH"),
                              ("cad", "#b56235", "Compiled CAD")):
        ax[1].plot([r["part"].removeprefix("reducer-") for r in rays],
                   [r["median_mean_batch_us"][key] for r in rays], "o-", color=color, label=label)
    ax[1].set(xlabel="Reducer part", ylabel="Matched exit query [µs/batch]", ylim=(0, None))
    ax[1].legend(fontsize=8)
    ax[1].grid(axis="y", alpha=.2)
    fig.savefig(a.out / "backend-costs.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("scene", "cpu_mesh", "cpu_cad", "gpu_cad", "cross", "out"):
        p.add_argument(name, type=Path)
    main(p.parse_args())
