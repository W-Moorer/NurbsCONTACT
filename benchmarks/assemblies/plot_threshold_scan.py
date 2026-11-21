"""Plot the costs of the threshold-scan results that pass the analytic audit."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator


STYLES = {
    "native-sweep": ("Native CAD sweep", "#136665", "o", "-"),
    "cad-ca-bounds": ("CAD interval advancement", "#36588F", "s", "-"),
    "mesh-ccd-cad-bounds": ("Mesh-guided CAD verification", "#744595", "D", "-"),
    "mesh-bvh-ccd": ("Fastest compatible mesh", "#AA4C18", "^", "--"),
    "occt-ca": ("Numerical OCCT advancement", "#57575C", "v", "--"),
}
PANELS = (("trim-edge", "Trimmed plate and disk"), ("counterbore", "Counterbore and pin"))
COSTS = (("median_ms", "Online query"), ("first_query_ms", "Preparation + first query"))


def plot(summary, output):
    with summary.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["offset_direction"] == "x"]
    gaps = sorted({int(row["gap_from_threshold_nm"]) for row in rows})
    values = defaultdict(list)
    for row in rows:
        if row["accuracy"] == "within-target":
            values[row["fixture"], row["method"], int(row["gap_from_threshold_nm"])].append(row)

    font = "Times New Roman", "Liberation Serif", "DejaVu Serif"
    plt.rcParams.update({"font.family": "serif", "font.serif": font, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#82868A", "axes.labelcolor": "#252B30",
                         "xtick.color": "#252B30", "ytick.color": "#252B30"})
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.105, right=.98, bottom=.175, top=.825, hspace=.55, wspace=.13)
    for i, (fixture, fixture_label) in enumerate(PANELS):
        for j, (metric, metric_label) in enumerate(COSTS):
            ax = axes[i, j]
            ax.set_title(f"{fixture_label} · {metric_label}", loc="left", fontsize=10.8, pad=8)
            ax.set_xscale("symlog", linthresh=10, linscale=.65)
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
            ax.axvline(0, color="#BDC2C5", linewidth=.75)
            ax.grid(axis="y", which="major", color="#DFE2E4", linewidth=.65)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", labelsize=9, length=3, width=.65)
            for method, (_, color, marker, line) in STYLES.items():
                # Different mesh settings may minimize online and first-query
                # costs. Break the curve wherever a method has no valid result.
                points = [min((float(row[metric]) for row in values[fixture, method, gap]), default=float("nan"))
                          for gap in gaps]
                ax.plot(gaps, points, color=color, linestyle=line, marker=marker,
                        markerfacecolor="white" if line == "--" else color, markeredgewidth=.9,
                        markersize=4.6, linewidth=1.15)
    labels = {-10000: "−10k", -1000: "−1k", -100: "−100", 0: "0", 100: "+100", 1000: "+1k", 10000: "+10k"}
    axes[1, 0].set_xticks(list(labels), list(labels.values()))
    axes[1, 0].set_xlim(min(gaps) * 1.45, max(gaps) * 1.45)
    fig.supxlabel("Minimum separation minus threshold (nm, symmetric log scale)", y=.096, fontsize=11)
    fig.supylabel("Time (ms, logarithmic scale)", x=.025, fontsize=11)
    handles = [Line2D([0], [0], color=color, marker=marker, linestyle=line, label=label,
                      markerfacecolor="white" if line == "--" else color, markeredgewidth=.9,
                      markersize=4.9, linewidth=1.15) for label, color, marker, line in STYLES.values()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .992), ncol=3,
               columnspacing=1.3, frameon=False, fontsize=9.6)
    fig.text(.52, .040, "In-plane offset along x. Solid markers = CAD bounds. Open markers = numerical results.",
             ha="center", fontsize=9.5)
    fig.text(.52, .016, "Only results compatible with analytic truth and meeting the time-width target are shown.",
             ha="center", fontsize=9.5)
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.summary, args.output)


if __name__ == "__main__":
    main()
