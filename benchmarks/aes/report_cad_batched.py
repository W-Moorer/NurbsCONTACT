"""Archive full-duration CAD execution comparisons, without filtering response curves."""
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


def load(folder):
    info = json.loads((folder / "run.json").read_text())
    if not info["completed"] or info["requested_duration_s"] != 5 or info["step_s"] != .0001:
        raise ValueError(f"Expected a complete 5 s / 0.1 ms experiment: {folder}")
    data = np.genfromtxt(folder / "response.csv", delimiter=",", names=True)
    if len(data) != 50001 or abs(data["time_s"][-1] - 5) > 1e-8:
        raise ValueError("Incomplete history")
    for name in data.dtype.names:
        if np.isinf(data[name]).any() or (not name.endswith("_gap_m") and not np.isfinite(data[name]).all()):
            raise ValueError(f"Invalid channel {name}")
    return info, data


def differences(a, b):
    if a.dtype.names != b.dtype.names:
        raise ValueError("Response schema changed")
    np.testing.assert_array_equal(a["time_s"], b["time_s"])
    result = {}
    for name in a.dtype.names:
        missing = np.array_equal(np.isnan(a[name]), np.isnan(b[name]))
        valid = np.isfinite(a[name]) & np.isfinite(b[name])
        delta = a[name][valid] - b[name][valid]
        result[name] = dict(same_missing_pattern=missing,
                            maximum=float(np.max(abs(delta), initial=0)),
                            rms=float(np.sqrt(np.mean(delta * delta))) if len(delta) else 0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("cpu", type=Path)
    parser.add_argument("scene", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    before, a = load(args.before)
    after, b = load(args.after)
    if before["scene_sha256"] != after["scene_sha256"]:
        raise ValueError("Scene changed between experiments")
    c = np.genfromtxt(args.cpu / "response.csv", delimiter=",", names=True)
    with (args.cpu / "response.stats.csv").open(newline="") as f:
        cpu = next(csv.DictReader(f))
    delta = differences(a, b)
    cpu_delta = differences(c, b)
    exact = all(v["same_missing_pattern"] and v["maximum"] == 0 for v in delta.values())
    report = dict(before=before, after=after, cpu_source=str(args.cpu), cpu_statistics=cpu,
                  cpu_rerun=False, exact_response_match=exact, differences=delta,
                  cpu_differences=cpu_delta,
                  cpu_sha256=hashlib.sha256((args.cpu / "response.csv").read_bytes()).hexdigest(),
                  gpu_speedup=float(before["gpu"]["online_s"]) / float(after["gpu"]["online_s"]),
                  cpu_speedup=float(cpu["wall_s"]) / float(after["gpu"]["online_s"]))
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.out / "response.npz", **{k: b[k] for k in b.dtype.names})

    histories = []
    for data in (a, b):
        q = np.column_stack([data[f"q{i}"] for i in range(3)])
        v = np.column_stack([data[f"v{i}"] for i in range(3)])
        _, external = smooth_loads(args.scene, data["time_s"], q, v)
        torque = np.column_stack([data[f"generalized_force{i}"] for i in range(3)]) - external
        histories.append([*[data[f"v{i}"] for i in range(3)],
                          *[torque[:, i] for i in range(3)],
                          data["p0_normal_N"], data["p1_normal_N"], data["a1"]])
    labels = ["Input coordinate speed [rad/s]", "Middle coordinate speed [rad/s]",
              "Output coordinate speed [rad/s]", "Input contact torque [N m]",
              "Middle contact torque [N m]", "Output contact torque [N m]",
              "Stage 1 normal load [N]", "Stage 2 normal load [N]", "Middle acceleration [rad/s²]"]
    fig, axes = plt.subplots(3, 3, figsize=(14, 9), layout="constrained")
    for i, ax in enumerate(axes.flat):
        ax.plot(a["time_s"], histories[0][i], color="#255e83", lw=.85, label="GPU CAD control")
        ax.plot(b["time_s"], histories[1][i], color="#cf6433", lw=.7, ls="--", label="GPU CAD batched (ours)")
        ax.set(xlim=(0, 5), xlabel="Time [s]", ylabel=labels[i])
        ax.grid(alpha=.18)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Complete reducer simulation · 0.1 ms main step · unfiltered responses")
    fig.savefig(args.out / "responses.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    times = [float(cpu["wall_s"]), float(before["gpu"]["online_s"]), float(after["gpu"]["online_s"])]
    bars = axes[0].bar(["CPU CAD\narchived", "GPU CAD\ncontrol", "GPU CAD\nbatched (ours)"], times,
                       color=["#9aa6ac", "#255e83", "#cf6433"])
    axes[0].bar_label(bars, fmt="%.2f s", padding=3)
    axes[0].set(ylabel="Online time [s]", ylim=(0, max(times) * 1.15))
    phases = ["broad_s", "sort_s", "search_s", "group_s", "integral_s"]
    x = np.arange(len(phases))
    axes[1].bar(x - .18, [float(before["gpu"][k]) for k in phases], .36, color="#255e83", label="Control")
    axes[1].bar(x + .18, [float(after["gpu"][k]) for k in phases], .36, color="#cf6433", label="Batched")
    axes[1].set(xticks=x, xticklabels=["Broad", "Sort", "Confirm", "Connect", "Integrate"], ylabel="Stage time [s]")
    axes[1].legend()
    fig.suptitle("Same reducer · 5 s / 0.1 ms · preprocessing excluded")
    fig.savefig(args.out / "timing.png", dpi=180)
    plt.close(fig)
    print(json.dumps({k: report[k] for k in ["exact_response_match", "gpu_speedup", "cpu_speedup"]}, indent=2))
    if not exact:
        raise SystemExit("Execution-only response preservation failed")


if __name__ == "__main__":
    main()
