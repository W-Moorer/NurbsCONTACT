"""Plot the archived eight output frames and the independent whole-solid force replay."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[2] / "results/aes/general-contact"
    def trace(method):
        return np.genfromtxt(root / f"spur-local-{method}-response.csv", delimiter=",", names=True)

    cad, mesh = trace("cad"), trace("mesh")
    time_tol = 8*np.spacing(cad["time_s"].max())
    if cad.shape != mesh.shape or not np.allclose(cad["time_s"], mesh["time_s"], atol=time_tol, rtol=0):
        raise ValueError("mismatched output times")
    audit = json.loads((root / "spur-local-regions.json").read_text())["poses"]
    saved = [p for p in audit if p["pose_source"] == "output"]
    if not np.array_equal(cad["time_s"], [p["time_s"] for p in saved]):
        raise ValueError("replay times differ from CAD trace")
    time = cad["time_s"] * 1e3
    fig, (force, speed) = plt.subplots(1, 2, figsize=(9, 3.3), layout="constrained")
    blue, rust = "#116782", "#b34927"
    force.plot(time, mesh["p0_normal_N"]/1000, "--s", color=rust, ms=4, label="Triangle mesh")
    force.plot(time, cad["p0_normal_N"]/1000, "-o", color=blue, ms=4, label="Local STEP (three methods)")
    force.scatter(time, [p["full_force_N"]/1000 for p in saved], marker="o", s=63,
                  facecolors="none", edgecolors="#303641", linewidths=.9, zorder=4,
                  label="Whole STEP at saved poses")
    force.set(ylabel="Normal force (kN)", title="First loaded output frames")
    for i, color, marker in ((0, blue, "o"), (1, rust, "s")):
        speed.plot(time, cad[f"v{i}"]-mesh[f"v{i}"], marker=marker, color=color, ms=4,
                   label=("Pinion" if i == 0 else "Wheel"))
    speed.set(ylabel="STEP minus mesh (rad/s)", title="Shaft speed differences")
    for ax in (force, speed):
        ax.set_xlabel("Simulated time (ms)")
        ax.set_xticks([0, .1, .2, .3, .35])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.savefig(root / "spur-local-response.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
