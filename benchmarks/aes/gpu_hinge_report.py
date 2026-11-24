"""Compare independent five-second trajectories without shifting or smoothing the samples."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def comparison(cpu_file, run, deck, animate):
    reference = np.load(cpu_file)
    with np.load(run / "response.npz") as raw:
        gpu = dict(raw)
    t = gpu["time_s"]
    np.testing.assert_allclose(t, reference["time_s"], rtol=0, atol=2e-11)
    if len(t) != 50001 or abs(t[-1] - 5) > 1e-10:
        raise ValueError("Expected a complete 5 s trajectory")
    metrics = {}
    for key in (
        "speed_rad_s",
        "transmission_error_mrad",
        "stage_normal_load_N",
        "output_contact_torque_Nm",
    ):
        a, b = reference[key], gpu[key]
        error = b - a
        metrics[key] = dict(
            rms=np.sqrt(np.mean(error**2, axis=0)).tolist(),
            max_abs=np.max(abs(error), axis=0).tolist(),
            cpu_peak=a.max(axis=0).tolist(),
            gpu_peak=b.max(axis=0).tolist(),
        )
    cpu, g = reference["output_contact_torque_Nm"], gpu["output_contact_torque_Nm"]
    imax = int(np.argmax(abs(g - cpu)))
    metrics["torque_summary"] = dict(
        cpu_mean_Nm=cpu.mean(),
        gpu_mean_Nm=g.mean(),
        cpu_peak_time_s=t[np.argmax(cpu)],
        gpu_peak_time_s=t[np.argmax(g)],
        cpu_last_s_peak_Nm=cpu[t >= 4].max(),
        gpu_last_s_peak_Nm=g[t >= 4].max(),
        worst_time_s=t[imax],
        worst_cpu_Nm=cpu[imax],
        worst_gpu_Nm=g[imax],
    )
    metrics["pair_loading"] = []
    for j in range(2):
        record = {}
        for name, source in (("cpu", reference), ("gpu", gpu)):
            load = source["stage_normal_load_N"][:, j] > 0
            changes = np.diff(load.astype(int))
            record[name + "_load_times_s"] = t[1:][changes > 0].tolist()
            record[name + "_release_times_s"] = t[1:][changes < 0].tolist()
        record["different_output_samples"] = int(
            np.count_nonzero(
                (reference["stage_normal_load_N"][:, j] > 0)
                != (gpu["stage_normal_load_N"][:, j] > 0)
            )
        )
        metrics["pair_loading"].append(record)
    (run / "accuracy.json").write_text(json.dumps(metrics, indent=2) + "\n")

    colors = ("#385a78", "#b26231")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "path.simplify": False,
        }
    )
    fig, axes = plt.subplots(
        3, 1, figsize=(8, 6.5), layout="constrained", height_ratios=(1, 1, 1.4)
    )
    for ax, y, name, color in zip(axes[:2], (cpu, g), ("CPU mesh", "GPU mesh"), colors):
        ax.plot(t, y, lw=0.42, color=color)
        ax.set(xlim=(0, 5), ylim=(-12, 455), ylabel="Torque (N m)", xlabel="Time (s)")
        ax.set_title(name, loc="left")
    sub = t <= 0.010
    for y, name, color in zip((cpu, g), ("CPU", "GPU"), colors):
        axes[2].plot(1e3 * t[sub], y[sub], color=color, lw=1.1, marker=".", ms=3, label=name)
    axes[2].set(xlim=(0, 10), xlabel="Time (ms)", ylabel="Torque (N m)")
    axes[2].set_title("Start-up contact and release", loc="left")
    axes[2].axvline(t[imax] * 1000, color=".55", lw=0.8, ls=":")
    axes[2].legend(frameon=False, ncol=2)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Output shaft contact torque", fontsize=13)
    save(fig, run / "torque")

    series = [
        ("speed_rad_s", i, title, "rad/s")
        for i, title in enumerate(("Input shaft", "Intermediate shaft", "Output shaft"))
    ]
    series += [
        (key, i, f"Stage {i+1} {title}", unit)
        for key, title, unit in (
            ("transmission_error_mrad", "transmission error", "mrad"),
            ("stage_normal_load_N", "normal load", "N"),
        )
        for i in range(2)
    ]
    fig, axes = plt.subplots(
        len(series), 2, figsize=(8.3, 10), sharex=True, sharey="row", layout="constrained"
    )
    for idx, (key, j, title, unit) in enumerate(series):
        for col, (name, data, color) in enumerate(
            zip(("CPU mesh", "GPU mesh"), (reference, gpu), colors)
        ):
            ax = axes[idx, col]
            ax.plot(t, data[key][:, j], color=color, lw=0.4)
            ax.grid(axis="y", alpha=0.18)
            if idx == 0:
                ax.set_title(name, color=color)
            if col == 0:
                ax.set_ylabel(f"{title}\n({unit})", fontsize=8)
    for ax in axes[-1]:
        ax.set(xlabel="Time (s)", xlim=(0, 5))
    save(fig, run / "shafts-and-loads")

    if animate:
        from rigid_movie import render_reducer_animation

        render_reducer_animation(
            deck,
            run / "reducer.mp4",
            gpu["q"],
            gpu["stage_normal_load_N"],
            t,
            "Two-stage reducer  |  GPU mesh dynamics",
        )


def save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("cpu", "run", "deck"):
        p.add_argument(arg, type=Path)
    p.add_argument("--animate", action="store_true")
    a = p.parse_args()
    comparison(a.cpu, a.run, a.deck, a.animate)
