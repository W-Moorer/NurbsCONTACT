"""Create figures and multi-view animations for the two external experiments."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np


MEASURED = "#343d47"
BREP = "#007e92"
MESH = "#bd522d"


def data(path):
    return np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")


def save(fig, directory, name):
    for suffix in (".pdf", ".png"):
        fig.savefig(directory / (name + suffix), dpi=200)
    plt.close(fig)


def slider_figure(root, output):
    configs = json.loads((root / "summary.json").read_text(encoding="utf-8"))["runs"]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.4), layout="constrained")
    position = data(root / "position-200rpm-0.25mm-brep.csv")
    phase = position["angle_deg"]
    axes[0].plot(phase, position["measured_position_m"]*1000, c=MEASURED, label="Experiment")
    axes[0].plot(phase, position["simulated_position_m"]*1000, c=BREP, ls="--", label="B-rep")
    axes[0].set(xlabel="Crank angle (°)", ylabel="Slider position (mm)", title="a   Slider travel",
                xlim=(0, 360))
    axes[0].legend(frameon=False, loc="lower right")
    b = data(root / "paired-200rpm-0.25mm-brep.csv")
    axes[1].plot(b["angle_deg"], b["measured_acceleration_m_s2"], color=MEASURED,
                 label="Experiment", lw=1.2)
    axes[1].plot(b["angle_deg"], b["simulated_acceleration_m_s2"], color=BREP,
                 label="B-rep", lw=1.2)
    axes[1].plot(b["angle_deg"], b["ideal_joint_m_s2"], color="#7d8c95", ls=":", lw=1.2,
                 label="Ideal joint")
    axes[1].legend(frameon=False, loc="lower left", ncol=2, fontsize=8)
    axes[1].set(xlabel="Crank angle (°)", ylabel="Slider acceleration (m/s²)",
                title="b   Acceleration, 200 rpm / 0.25 mm", xlim=(0, 720))
    axes[1].set_xticks([0, 180, 360, 540, 720])
    for label, color, marker, legend in (("brep", BREP, "o", "B-rep"),
                                          ("mesh5um", "#6b638b", "s", "BVH 5 µm"),
                                          ("mesh25um", MESH, "^", "BVH 25 µm")):
        group = [r for r in configs if r["backend"] == label]
        valid = [(idx, r["uniform_phase_acceleration_rmse_m_s2"]) for idx, r in enumerate(group)
                 if "error" not in r]
        if valid:
            x, y = np.array(valid).T
            axes[2].plot(x, y, c=color, marker=marker, label=legend, ls="none", ms=5)
    labels = [f'{r["rpm"]}\n{r["clearance_mm"]:.2f}' for r in configs if r["backend"] == "brep"]
    axes[2].set(xlabel="Speed (rpm) / clearance (mm)", ylabel="Acceleration RMSE (m/s²)",
                title="c   Seven experimental conditions", yscale="log", xticks=range(len(labels)),
                xticklabels=labels)
    axes[2].legend(frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=.23)
        ax.spines[["right", "top"]].set_visible(False)
    save(fig, output, "clearance-slider-validation")


def ring_trials(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    groups = {}
    for row in rows:
        key = (int(row["slot"]), row["offset"], row["trial"])
        groups.setdefault(key, []).append(tuple(float(row[key]) for key in (
            "sample_from_command_turnaround", "relative_z_m", "measured_axial_N", "brep_axial_N")))
    return {key: np.array(group) for key, group in groups.items()}


def ring_figure(trials, output):
    held = {key: trial for key, trial in trials.items() if key[0] in (3, 4)}
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.4), layout="constrained")
    for letter, ax, offset, description in zip(
        "ab", axes, ("0", "x+3"), ("Nominal offset", "Commanded x offset +3 mm")
    ):
        group = np.stack([trial for key, trial in held.items() if key[1] == offset])
        sample = group[0, :, 0]
        p25, med, p75 = np.percentile(group[:, :, 2], [25, 50, 75], axis=0)
        pred = np.median(group[:, :, 3], axis=0)
        ax.fill_between(sample, p25, p75, alpha=.20, color=MEASURED, linewidth=0)
        ax.plot(sample, med, c=MEASURED, label="Measured median")
        ax.plot(sample, pred, c=BREP, label="STEP replay median")
        ax.axvline(0, c="#9ca7af", ls=":", lw=1)
        ax.set(xlabel="Samples from commanded turnaround", ylabel="Axial force (N)",
               title=f"{letter}   {description}", ylim=(-.5, 12))
    axes[0].legend(frameon=False, loc="upper left")
    peaks = np.array([np.max(trial[:, 2:4], axis=0) for trial in held.values()])
    axes[2].plot([0, 15], [0, 15], ls=":", c="#7d8c95")
    axes[2].scatter(*peaks.T, color=BREP, s=12, alpha=.50, edgecolors="none")
    axes[2].set(xlabel="Measured peak force (N)", ylabel="Predicted peak force (N)",
                title="c   All 150 held-out trials", xlim=(0, 15), ylim=(0, 15),
                aspect="equal")
    for ax in axes:
        ax.grid(axis="y", alpha=.23)
        ax.spines[["right", "top"]].set_visible(False)
    save(fig, output, "copper-ring-validation")


def make_animation(fig, update, frames, path):
    playback = FuncAnimation(fig, update, frames=frames, interval=1000/24, blit=False)
    playback.save(path, writer=FFMpegWriter(fps=24, codec="h264",
                                           extra_args=["-pix_fmt", "yuv420p"]), dpi=125)
    plt.close(fig)


def slider_animation(trace, path, elevation, azimuth):
    fig = plt.figure(figsize=(7.2, 5.0), layout="constrained")
    ax = fig.add_subplot(2, 1, 1, projection="3d")
    graph = fig.add_subplot(2, 1, 2)
    ax.set(xlim=(-65, 375), ylim=(-70, 70), zlim=(-25, 25), title="Slider-crank simulation")
    ax.set_box_aspect((440, 140, 60))
    ax.view_init(elevation, azimuth)
    ax.set_axis_off()
    ax.plot([-65, 375], [0, 0], [-11, -11], c="#aebdc4", lw=3)
    crank, = ax.plot([], [], [], c=BREP, lw=4)
    rod, = ax.plot([], [], [], c="#253e50", lw=5)
    bore, = ax.plot([], [], [], c=MESH, lw=3)
    pin, = ax.plot([], [], [], c=BREP, lw=2)
    phase = trace["angle_deg"]
    graph.plot(phase, trace["a1"], c=BREP, lw=1.4)
    cursor = graph.axvline(0, c="#343d47", lw=1)
    graph.set(xlim=(0, 360), xlabel="Crank angle (°)", ylabel="Slider acceleration (m/s²)")
    circle = np.linspace(0, 2*math.pi, 65)

    def update(i):
        p = math.radians(phase[i])
        x, y = 50*math.cos(p), 50*math.sin(p)
        px, py = 1000*trace["b1_x_m"][i], 1000*trace["b1_y_m"][i]
        sx = 1000*trace["q1"][i]
        crank.set_data_3d([0, x], [0, y], [0, 0])
        rod.set_data_3d([x, px], [y, py], [0, 0])
        bore.set_data_3d(sx+11.125*np.cos(circle), 11.125*np.sin(circle), 6+0*circle)
        pin.set_data_3d(px+10.875*np.cos(circle), py+10.875*np.sin(circle), 7+0*circle)
        cursor.set_xdata([phase[i], phase[i]])

    make_animation(fig, update, np.linspace(0, len(phase)-1, 72).astype(int), path)


def ring_animation(trial, stop, path, elevation, azimuth):
    fig = plt.figure(figsize=(7.2, 5.0), layout="constrained")
    ax = fig.add_subplot(2, 1, 1, projection="3d")
    graph = fig.add_subplot(2, 1, 2)
    ax.set(xlim=(-24, 24), ylim=(-24, 24), zlim=(-3, 20), title="Reduced copper-ring axial replay")
    ax.set_box_aspect((48, 48, 23))
    ax.view_init(elevation, azimuth)
    ax.set_axis_off()
    x, y = np.meshgrid([-21, 21], [-21, 21])
    ax.plot_surface(x, y, np.full_like(x, 1000*stop, dtype=float), alpha=.3, color="#8396a4")
    a = np.linspace(0, 2*math.pi, 65)
    rings = [ax.plot(radius*np.cos(a), radius*np.sin(a), 0*a, c="#a9532c", lw=2)[0]
             for radius in (8, 10) for _ in range(2)]
    sample = trial[:, 0]
    graph.plot(sample, trial[:, 2], color=MEASURED, label="Experiment")
    graph.plot(sample, trial[:, 3], color=BREP, label="STEP replay")
    cursor = graph.axvline(sample[0], c="#343d47", lw=1)
    graph.set(xlabel="Samples from commanded turnaround", ylabel="Axial force (N)")
    graph.legend(frameon=False, loc="upper left")

    def update(i):
        for curve, (radius, thickness) in zip(rings, ((8, 0), (8, 3), (10, 0), (10, 3))):
            curve.set_data_3d(radius*np.cos(a), radius*np.sin(a), 0*a+1000*trial[i, 1]+thickness)
        cursor.set_xdata([sample[i], sample[i]])

    make_animation(fig, update, np.repeat(np.arange(len(trial)), 2), path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--animate", action="store_true")
    args = parser.parse_args()
    root = args.results
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "axes.titleweight": "semibold"})
    slider_figure(root / "clearance", figures)
    trials = ring_trials(root / "insertion/paired.csv")
    ring_figure(trials, figures)
    if args.animate:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        directory = root / "animations"
        trace = data(root / "clearance/display-200rpm-0.25mm-brep.csv")
        trial = next(val for key, val in trials.items() if key[:2] == (3, "x+3"))
        summary = json.loads((root / "insertion/summary.json").read_text(encoding="utf-8"))
        stop = summary["identified_stop_relative_to_command_target_m"]["x+3"]
        slider_animation(trace, directory / "clearance-slider-plan.mp4", 90, -90)
        slider_animation(trace, directory / "clearance-slider-isometric.mp4", 28, -60)
        ring_animation(trial, stop, directory / "copper-ring-front.mp4", 8, -90)
        ring_animation(trial, stop, directory / "copper-ring-isometric.mp4", 29, -57)


if __name__ == "__main__":
    main()
