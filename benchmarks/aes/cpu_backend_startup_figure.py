"""Plot accepted CPU states and same-pose contact geometry for a failed duty cycle."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw(directory, reference):
    cpu = np.genfromtxt(directory/"local-progress.csv", names=True, delimiter=",", ndmin=1)
    replay = np.genfromtxt(directory/"pose-replay.csv", names=True, delimiter=",", dtype=None, encoding="utf-8")
    indices = cpu["step"].astype(int)
    t = cpu["time_s"] * 1000
    with np.load(reference) as rd:
        tr = rd["reference_rad"][indices,0]/rd["global_speed_rad_s"][0,0]
        np.testing.assert_allclose(tr*1000, t, rtol=0, atol=1e-6)
        speed = rd["global_speed_rad_s"][indices, 2]
        torque = rd["contact_raw_TZ_Nm"][indices, 1]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.0), sharex=True, layout="constrained")
    blue, red, gray = "#126485", "#c65f28", "#414754"
    fig.suptitle("Two-stage reducer | accepted CPU states and same-pose replay", fontsize=15)
    for stage, ax in enumerate(axes[0], start=1):
        for scope, name, color in (("local", "Overlap width", blue), ("fcl", "FCL triangle depth", red)):
            r = replay[(replay["stage"]==stage) & (replay["scope"]==scope)]
            ax.plot(r["time_s"]*1000, r["max_depth_um"], color=color, label=name, lw=1.4)
        ax.axhline(53, ls="--", c=gray, lw=1, label="Scene limit")
        ax.set_yscale("symlog", linthresh=2)
        ax.set_yticks([0, 2, 10, 53, 200, 1000], labels=["0", "2", "10", "53", "200", "1000"])
        ax.set_ylim(-.08, 1650)
        ax.set_title(f"Stage {stage} | same shaft positions", loc="left")
        ax.set_ylabel("Maximum depth (µm)")
    axes[0, 0].legend(loc="upper right", fontsize=8, frameon=False)
    picks = replay[(replay["stage"]==2) & (replay["scope"]=="local")]
    np.testing.assert_allclose(picks["time_s"]*1000, t, rtol=0, atol=1e-8)
    for ax, series, ref, title, unit in ((axes[1,0], cpu["v2"], speed, "Output shaft speed", "rad/s"),
                    (axes[1,1], picks["driven_global_Tz_Nm"], torque, "Output contact torque", "N·m")):
        ax.plot(t, series, c=blue, label="CPU overlap", lw=1.4)
        ax.plot(t, ref, c=gray, ls="--", label="RecurDyn", lw=1.3)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Physical time (ms)")
        ax.set_ylabel(unit)
    axes[1, 0].legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_xlim(0, t[-1])
    fig.savefig(directory/"cpu-startup.png", dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("recurdyn", type=Path)
    a = p.parse_args()
    draw(a.directory,a.recurdyn)
