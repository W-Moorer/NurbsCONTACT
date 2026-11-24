"""Plot the measured fraction of host replies that exceed a given latency."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


def main():
    root = Path(__file__).resolve().parent
    fig, ax = plt.subplots(figsize=(7.8, 4.6), layout="constrained")
    for stem, label, color in [
        ("windows-0.1.0", "RTX 3070 · Windows · original 0.1.0", "#68737d"),
        ("linux-0.1.0", "RTX 3090 · Linux · original 0.1.0", "#b84923"),
        ("linux-0.1.1", "RTX 3090 · Linux · updated 0.1.1", "#12698b"),
    ]:
        with np.load(root / (stem + ".npz")) as data:
            values, counts = np.unique(data["host_io_step_ns"][1:] / 1000, return_counts=True)
        above = 100 * (1 - counts.cumsum() / counts.sum())
        # The lowest non-zero fraction corresponds to one physical step.
        ax.step(values[:-1], above[:-1], where="post", color=color, label=label, lw=1.8)
    ax.axvline(100, color="#454545", lw=1, ls="--")
    ax.text(108, 0.011, "100 µs step budget", rotation=90, color="#454545")
    ax.set(xscale="log", yscale="log", xlim=(20, 6000), ylim=(0.0015, 110),
           xlabel="Host command to returned wrench (µs)", ylabel="Steps exceeding this latency (%)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_xticks([30, 100, 300, 1000, 3000], labels=["30", "100", "300", "1,000", "3,000"])
    ax.legend(frameon=False, fontsize=9)
    ax.grid(which="major", alpha=0.2)
    fig.savefig(root / "host-latency.png", dpi=180)


if __name__ == "__main__":
    main()
