"""Plot measured GPU mesh-query costs, including the complete five-second interval."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    a = parser.parse_args()
    report = json.loads((a.work / "comparison.json").read_text())
    profile = np.load(a.work / "profile.npz")
    if report["steps"] != 50000 or report["full_fp64_id_check"]["mismatches"]:
        raise ValueError("Incomplete or discrepant geometry audit")
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), layout="constrained")
    names = list(report["runs"])
    medians = [report["runs"][name]["median_s"] for name in names]
    labels = ["Scalar predicate" if name == "lanes-1" else "Grouped predicate" for name in names]
    axes[0].barh(labels, medians, color=["#c18e60", "#28778b"])
    axes[0].set(xlabel="Time for 50,000 two-pair frames [s]", xlim=(0, max(medians) * 1.23))
    axes[0].invert_yaxis()
    for y, name in enumerate(names):
        axes[0].plot(report["runs"][name]["online_s"], [y] * 3, "|", color="#222222", ms=10)
        axes[0].text(
            max(report["runs"][name]["online_s"]) + 0.13,
            y,
            f"{report['runs'][name]['median_s']:.3f} s",
            va="center",
        )
    for key, color, label in (("before_us", "#c18e60", "Before"), ("after_us", "#28778b", "After")):
        sorted_us = np.sort(profile[key])
        axes[1].semilogx(
            sorted_us, np.arange(1, len(sorted_us) + 1) / len(sorted_us), label=label, color=color
        )
    axes[1].axvline(100, color="#777777", ls=":", lw=0.8)
    axes[1].set(
        xlabel="Instrumented detection latency [µs]", ylabel="Cumulative fraction", ylim=(0, 1.02)
    )
    axes[1].legend()
    fig.savefig(a.work / "timing.png", dpi=180)
    fig.savefig(a.work / "timing.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.6), layout="constrained", sharex=True)
    t = profile["time_s"]
    for ax, key, color, label in zip(
        axes[:2], ("before_us", "after_us"), ("#c18e60", "#28778b"), ("Before", "After")
    ):
        ax.plot(t, profile[key], color=color, lw=0.4)
        ax.axhline(100, color="#555555", ls=":", lw=0.7)
        ax.set(yscale="log", ylabel=f"{label} latency [µs]", xlim=(0, 5))
    for i in range(profile["intersections"].shape[1]):
        axes[2].plot(t, profile["intersections"][:, i], lw=0.5, label=f"Contact pair {i+1}")
    axes[2].set(ylabel="Intersecting triangle pairs", xlabel="Prescribed trajectory time [s]")
    axes[2].legend(ncols=2)
    fig.suptitle("Complete geometry replay; all 100,000 triangle-pair sets match FP64")
    fig.savefig(a.work / "full-interval.png", dpi=180)
    fig.savefig(a.work / "full-interval.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
