"""Archive complete CAD execution experiments and plot their dynamics and costs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cpu_cad_pressure_report import smooth_loads


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(folder):
    meta = folder / "run.json"
    if not meta.exists():
        meta = folder / "command.json"
    result = json.loads(meta.read_text()) if meta.exists() else {}
    for suffix, key in (("gpu", "gpu"), ("stats", "statistics")):
        with (folder / f"response.{suffix}.csv").open(newline="") as stream:
            result[key] = next(csv.DictReader(stream))
    result["response_sha256"] = digest(folder / "response.csv")
    return result


def trace(folder):
    data = np.genfromtxt(folder / "response.csv", delimiter=",", names=True)
    result = {name: data[name] for name in data.dtype.names}
    for name, values in result.items():
        if np.isinf(values).any() or (not name.endswith("_gap_m") and not np.isfinite(values).all()):
            raise ValueError(f"Nonfinite response channel: {folder} {name}")
    return result


def compare(a, b):
    np.testing.assert_array_equal(a["time_s"], b["time_s"])
    result = {}
    for name in a:
        if name == "time_s":
            continue
        valid = np.isfinite(a[name]) & np.isfinite(b[name])
        if not np.array_equal(np.isnan(a[name]), np.isnan(b[name])):
            raise ValueError(f"Missing-value pattern changed: {name}")
        delta = a[name][valid] - b[name][valid]
        result[name] = {"max_abs": float(np.max(abs(delta), initial=0)),
                        "rms": float(np.sqrt(np.mean(delta ** 2))) if len(delta) else 0}
    return result


def draw_responses(a, b, mesh, scene, out, title, before_label="GPU CAD before"):
    contact = []
    for data in (a, b):
        q = np.column_stack([data[f"q{i}"] for i in range(3)])
        v = np.column_stack([data[f"v{i}"] for i in range(3)])
        _, smooth = smooth_loads(scene, data["time_s"], q, v)
        contact.append(data["generalized_force2"] - smooth[:, 2])
    _, mesh_smooth = smooth_loads(scene, mesh[:, 0], mesh[:, [1, 5, 9]], mesh[:, [2, 6, 10]])
    mesh_contact = mesh[:, 12] - mesh_smooth[:, 2]
    fig, axes = plt.subplots(3, 2, figsize=(12, 8.5), layout="constrained")
    panels = [("v0", 2, "Input speed [rad/s]"), ("v1", 6, "Intermediate speed [rad/s]"),
              ("v2", 10, "Output speed [rad/s]"), ("generalized_force2", 12, "Output contact torque [N m]"),
              ("p0_normal_N", 14, "Stage 1 normal load [N]"),
              ("p1_normal_N", 17, "Stage 2 normal load [N]")]
    for ax, (key, column, label) in zip(axes.flat, panels):
        torque = key == "generalized_force2"
        ax.plot(mesh[:, 0], mesh_contact if torque else mesh[:, column], color="#a6acb4", lw=.7, alpha=.75, label="GPU mesh")
        ax.plot(a["time_s"], contact[0] if torque else a[key], color="#286a8c", lw=1, label=before_label)
        ax.plot(b["time_s"], contact[1] if torque else b[key], color="#bd5e35", ls="--", lw=.8, label="GPU CAD after")
        ax.set(xlabel="Time [s]", ylabel=label, xlim=(0, 5))
        ax.grid(alpha=.15)
    axes[0, 0].legend(ncol=3, fontsize=8)
    fig.suptitle(title)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("winner")
    parser.add_argument("out", type=Path)
    parser.add_argument("--fine", type=Path)
    parser.add_argument("--fine-before", type=Path)
    parser.add_argument("--cpu", type=Path, help="Directory of an archived, complete CPU CAD run; never rerun")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = [p.name for p in sorted(args.runs.iterdir())
             if p.is_dir() and (p / "response.gpu.csv").exists() and p.name.endswith("1ms")]
    sequence = ["before-1ms", "initial-graph-1ms", "staged-graph-1ms", "view-graph-1ms",
                "view-cooperative-1ms", "block-graph-1ms", "active-graph-1ms", "active-repeat-1ms",
                "cached-1ms", "prepared-1ms", "spread-1ms", "warp-segments-1ms"]
    names.sort(key=lambda n: sequence.index(n) if n in sequence else len(sequence))
    measurements = {name: run(args.runs / name) for name in names}
    scene = Path(measurements["before-1ms"]["command"][1])
    a, b = trace(args.runs / "before-1ms"), trace(args.runs / args.winner)
    if len(a["time_s"]) != 5001 or abs(a["time_s"][-1] - 5) > 1e-10:
        raise ValueError("A complete five-second, 1 ms trace is required")
    differences = compare(a, b)
    report = {"device": "NVIDIA RTX 3070", "duration_s": 5,
              "main_step_s": .001, "selected_run": args.winner,
              "measurements": measurements, "differences": differences}
    for name, data in (("before", a), ("after", b)):
        np.savez_compressed(args.out / f"{name}-1ms.npz", **data)
    if args.fine:
        fine_before, fine = trace(args.fine_before), trace(args.fine)
        if len(fine["time_s"]) != 50001 or abs(fine["time_s"][-1] - 5) > 1e-10:
            raise ValueError("A complete five-second, 0.1 ms regression is required")
        report["fine_regression"] = {"before": run(args.fine_before), "after": run(args.fine),
                                      "differences": compare(fine_before, fine)}
        np.savez_compressed(args.out / "after-0.1ms.npz", **fine)
        if args.cpu:
            cpu = trace(args.cpu)
            with (args.cpu / "response.stats.csv").open(newline="") as stream:
                cpu_stats = next(csv.DictReader(stream))
            cpu_wall = float(cpu_stats["wall_s"])
            gpu_wall = float(report["fine_regression"]["after"]["gpu"]["online_s"])
            report["cpu_reference"] = {"source": str(args.cpu), "rerun": False,
                "response_sha256": digest(args.cpu / "response.csv"), "online_s": cpu_wall,
                "gpu_speedup": cpu_wall / gpu_wall, "gpu_faster_than_cpu": gpu_wall < cpu_wall,
                "differences": compare(cpu, fine)}

    mesh_dir = Path("results/aes/gpu-mesh-dynamics/step-1ms-repaired")
    mesh = np.load(mesh_dir / "response.npy")
    np.testing.assert_allclose(mesh[:, 0], b["time_s"], atol=1e-10, rtol=0)
    report["mesh_reference"] = {"source": str(mesh_dir),
                                "timing": json.loads((mesh_dir / "run.json").read_text()),
                                "response_sha256": digest(mesh_dir / "response.npy")}
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    draw_responses(a, b, mesh, scene, args.out / "responses.png",
                   "Complete 5 s dynamics, 1 ms main step (automatic substeps retained)")
    if args.fine:
        with np.load("results/aes/gpu-mesh-dynamics/responses.npz") as archive:
            mesh_fine = archive["gpu"]
        np.testing.assert_allclose(mesh_fine[:, 0], fine["time_s"], atol=1e-10, rtol=0)
        draw_responses(fine_before, fine, mesh_fine, scene, args.out / "responses-0.1ms.png",
                       "Complete 5 s dynamics, 0.1 ms main step (automatic substeps retained)")
        if args.cpu:
            draw_responses(cpu, fine, mesh_fine, scene, args.out / "responses-cpu.png",
                           "Archived CPU CAD and current GPU CAD, complete 5 s / 0.1 ms", "CPU CAD")

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), layout="constrained")
    colors = ["#7394b1", "#abc1ce", "#bd6542", "#dca45e", "#6f997b", "#b6b6b6"]
    bottom = np.zeros(len(names))
    phases = ["broad_s", "sort_s", "search_s", "group_s", "integral_s"]
    for phase, color in zip(phases + ["other"], colors):
        values = np.array([float(measurements[n]["gpu"][phase]) if phase != "other" else
                           float(measurements[n]["gpu"]["online_s"]) -
                           sum(float(measurements[n]["gpu"][p]) for p in phases) for n in names])
        axes[0].bar(np.arange(len(names)), values, bottom=bottom, label=phase.replace("_s", ""), color=color)
        bottom += values
    labels = {"before-1ms": "Original", "initial-graph-1ms": "Three kernels",
              "staged-graph-1ms": "Split stages", "view-graph-1ms": "Shared views",
              "view-cooperative-1ms": "Cooperative views", "block-graph-1ms": "Block search",
              "active-graph-1ms": "Active size", "active-repeat-1ms": "Repeat",
              "cached-1ms": "Blocker cache", "prepared-1ms": "Fixed rules",
              "spread-1ms": "Spread work", "warp-segments-1ms": "Warp segments"}
    axes[0].set_xticks(np.arange(len(names)), [labels.get(n, n) for n in names], rotation=20, ha="right")
    for i, value in enumerate(bottom):
        axes[0].text(i, value + 1, f"{value:.2f}", ha="center", fontsize=8)
    axes[0].set_ylim(0, max(bottom) * 1.22)
    axes[0].set_ylabel("Online time [s]")
    axes[0].legend(ncol=6, fontsize=8)
    for key, label in (("p0_normal_N", "Stage 1 load"), ("p1_normal_N", "Stage 2 load"),
                       ("generalized_force2", "Output torque")):
        axes[1].plot(b["time_s"], b[key] - a[key], label=label)
    axes[1].set(xlim=(0, 5), xlabel="Time [s]", ylabel="After minus before [N or N m]")
    axes[1].legend(fontsize=8)
    fig.savefig(args.out / "execution.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"runs": {n: measurements[n]["gpu"]["online_s"] for n in names},
                      "largest_change": max(v["max_abs"] for v in differences.values())}, indent=2))


if __name__ == "__main__":
    main()
