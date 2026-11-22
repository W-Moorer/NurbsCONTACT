"""Plot the two cam displacement checks and the zero-gap switching-time audit."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw(folder):
    with np.load(folder / "kinematics-curves.npz") as kin, np.load(folder / "zero-gap-curves.npz") as dyn:
        d = json.loads((folder / "accuracy-0mm-125us.json").read_text())
        fig, ax = plt.subplots(2, 2, figsize=(7.15, 4.55), layout="constrained")
        teal, ink = "#007477", "#303944"
        for label, axes, title in zip(
            "abcd",
            ax.flat,
            (
                "Constant-breadth cam",
                "Eccentric cam",
                "Follower displacement error",
                "Engagement and release",
            ),
            strict=True,
        ):
            axes.set_title(title, loc="left", fontsize=9, pad=6)
            axes.text(-0.17, 1.07, label, transform=axes.transAxes, fontweight="bold", fontsize=11)
            axes.spines[["right", "top"]].set_visible(False)
            axes.tick_params(length=3, width=0.65)
            axes.grid(axis="y", linewidth=0.4, color="#e0e5e8", zorder=0)
        a, b, c, e = ax.flat
        a.plot(kin["angle_deg"], kin["exact_mm"], color=ink, lw=1.2, label="Published solution")
        a.plot(
            kin["angle_deg"][::120],
            kin["gpu_mm"][::120],
            "o",
            ms=2.5,
            mec=teal,
            mfc="white",
            mew=0.8,
            label="GPU contact",
        )
        a.set(
            xlabel="Cam angle (°)",
            ylabel="Follower travel (mm)",
            xlim=(0, 360),
            xticks=[0, 90, 180, 270, 360],
        )
        a.legend(frameon=False, loc="upper left")
        b.plot(dyn["t"], dyn["reference_mm"], color=ink, lw=0.85, label="CPU reference")
        b.plot(
            dyn["t"][::120],
            dyn["gpu_mm"][::120],
            "o",
            ms=2.0,
            mec=teal,
            mfc="white",
            mew=0.65,
            label="GPU HHT",
        )
        b.set(xlabel="Time (s)", ylabel="Follower travel (mm)", xlim=(0, d["physical_s"]))
        lo, hi = b.get_ylim()
        b.set_ylim(lo, hi + (hi - lo) * 0.14)
        b.legend(frameon=False, loc="upper center", ncols=2)
        c.plot(dyn["t"], dyn["error_um"], color=teal, lw=0.65)
        c.set(xlabel="Time (s)", ylabel="GPU − reference (µm)", xlim=(0, d["physical_s"]))
        if not d["same_event_order"] or not d["events"]:
            raise ValueError("Event figure requires paired independent reference and GPU transitions")
        half_step = d["step_s"] * 5e5
        e.axhspan(-half_step, half_step, color="#f1f3f4", zorder=0)
        e.scatter(
            [r["reference_s"] for r in d["events"]],
            [r["midpoint_error_us"] for r in d["events"]],
            color="#97a1a8",
            s=7,
            label="Output midpoint",
        )
        refined = json.loads((folder / "events-0mm-125us.json").read_text())
        for event, marker, color in (("engage", "o", teal), ("release", "^", ink)):
            rows = [r for r in refined["events"] if r["type"] == event]
            e.scatter(
                [r["reference_s"] for r in rows],
                [r["midpoint_error_us"] for r in rows],
                marker=marker,
                facecolor="white",
                edgecolor=color,
                s=12,
                linewidth=0.8,
                label="Refined " + event,
                zorder=3,
            )
        e.set(xlabel="Reference event time (s)", ylabel="Switch-time error (µs)")
        e.legend(
            frameon=False,
            ncols=3,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.29),
            columnspacing=0.8,
            handlelength=1,
        )
        for fmt in ("pdf", "png"):
            fig.savefig(folder / f"cam-validation.{fmt}", dpi=250, facecolor="white", pad_inches=0.03)
        plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
        }
    )
    draw(p.parse_args().folder)
