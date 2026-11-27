"""Compare complete CAD histories after arithmetic / segment scheduling changes.

Report absolute response differences; this is not an independent accuracy proof.
Keep the execution-only report's bitwise acceptance rule unchanged.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from report_cad_batched import load
from cpu_cad_pressure_report import smooth_loads


def differences(a, b):
    if a.dtype.names != b.dtype.names or len(a) != len(b):
        raise ValueError("Response schema or frame count changed")
    # The solver accumulates time rather than multiplying the step index by dt.
    # Check every increment and the cross-history timestamps; never resample.
    allowance = 4 * np.maximum(np.spacing(a["time_s"]), np.spacing(b["time_s"]))
    for data in (a, b):
        if data["time_s"][0] != 0 or np.any(np.abs(np.diff(data["time_s"]) - .0001) > allowance[1:]):
            raise ValueError("Response timestamps do not follow the fixed main-step grid")
    if np.any(np.abs(a["time_s"] - b["time_s"]) > allowance):
        raise ValueError("Different main-step output times")
    result = {}
    for name in a.dtype.names:
        valid = np.isfinite(a[name]) & np.isfinite(b[name])
        delta = a[name][valid] - b[name][valid]
        result[name] = dict(same_missing_pattern=bool(np.array_equal(np.isnan(a[name]), np.isnan(b[name]))),
                            maximum=float(np.max(abs(delta), initial=0)),
                            rms=float(np.sqrt(np.mean(delta * delta))) if len(delta) else 0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before", "after", "cpu", "scene", "out"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    old, a = load(args.before)
    new, b = load(args.after)
    if old["scene_sha256"] != new["scene_sha256"]:
        raise ValueError("Different input scenes")
    if hashlib.sha256(args.scene.read_bytes()).hexdigest() != new["scene_sha256"]:
        raise ValueError("Scene changed")
    c = np.genfromtxt(args.cpu / "response.csv", delimiter=",", names=True)
    with (args.cpu / "response.stats.csv").open(newline="") as stream:
        cpu = next(csv.DictReader(stream))
    delta, cpu_delta = differences(a, b), differences(c, b)
    flags = [k for k in b.dtype.names if k.endswith("_hit")]
    if not flags:
        raise ValueError("Contact-state channels missing")
    events = {}
    for name in flags:
        def switches(data):
            return np.flatnonzero(np.diff(data[name]) != 0) + 1
        events[name] = dict(
            different_frames_old=int(np.count_nonzero(a[name] != b[name])),
            different_frames_cpu=int(np.count_nonzero(c[name] != b[name])),
            transitions=int(len(switches(b))),
            same_switch_frames_old=bool(np.array_equal(switches(a), switches(b))),
            same_switch_frames_cpu=bool(np.array_equal(switches(c), switches(b))),
        )
    histories = []
    for data in (a, b, c):
        q = np.column_stack([data[f"q{i}"] for i in range(3)])
        v = np.column_stack([data[f"v{i}"] for i in range(3)])
        _, external = smooth_loads(args.scene, data["time_s"], q, v)
        torque = np.column_stack([data[f"generalized_force{i}"] for i in range(3)]) - external
        histories.append(np.column_stack([
            *[data[f"v{i}"] for i in range(3)],
            *[torque[:, i] for i in range(3)],
            data["p0_normal_N"], data["p1_normal_N"], data["a1"],
        ]))
    labels = ["Input speed [rad/s]", "Middle speed [rad/s]", "Output speed [rad/s]",
              "Input contact torque [N m]", "Middle contact torque [N m]", "Output contact torque [N m]",
              "Stage 1 normal load [N]", "Stage 2 normal load [N]", "Middle acceleration [rad/s^2]"]
    peaks = {}
    for i, label in enumerate(labels):
        x, y, z = (h[:, i] for h in histories)
        # Differences of raw adjacent increments expose added spikes without smoothing.
        peaks[label] = dict(maximum_difference_old=float(np.max(abs(y - x))),
                            maximum_difference_cpu=float(np.max(abs(y - z))),
                            increment_difference_old=float(np.max(abs(np.diff(y) - np.diff(x)))),
                            peak_time_old=float(a["time_s"][np.argmax(abs(x))]),
                            peak_time_new=float(b["time_s"][np.argmax(abs(y))]),
                            peak_time_cpu=float(c["time_s"][np.argmax(abs(z))]))
    report = dict(before=old, after=new, cpu_statistics=cpu, cpu_source=str(args.cpu), cpu_rerun=False,
                  cpu_sha256=hashlib.sha256((args.cpu / "response.csv").read_bytes()).hexdigest(),
                  differences_old=delta, differences_cpu=cpu_delta, contact_events=events, curves=peaks,
                  exact_response_old=all(v["same_missing_pattern"] and v["maximum"] == 0 for v in delta.values()),
                  time_alignment="integer main-step index; increments and cross-history time differences checked within four FP64 ulps; no interpolation",
                  gpu_speedup=float(old["gpu"]["online_s"]) / float(new["gpu"]["online_s"]),
                  cpu_speedup=float(cpu["wall_s"]) / float(new["gpu"]["online_s"]),
                  scope="Absolute numerical agreement; independent physical accuracy not inferred")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.out / "response.npz", **{k: b[k] for k in b.dtype.names})
    fig, axes = plt.subplots(3, 3, figsize=(14, 9), layout="constrained")
    for i, ax in enumerate(axes.flat):
        ax.plot(c["time_s"], histories[2][:, i], color="#8c969c", lw=1.3, label="CPU CAD (archived)")
        ax.plot(a["time_s"], histories[0][:, i], color="#255e83", lw=.8, label="GPU CAD before")
        ax.plot(b["time_s"], histories[1][:, i], color="#cf6433", lw=.65, ls="--", label="GPU CAD (ours)")
        ax.set(xlim=(0, 5), xlabel="Time [s]", ylabel=labels[i])
        ax.grid(alpha=.18)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Complete reducer - 5 s / 0.1 ms - raw responses, no smoothing")
    fig.savefig(args.out / "responses.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(14, 9), layout="constrained")
    for i, ax in enumerate(axes.flat):
        ax.plot(b["time_s"], histories[1][:, i] - histories[2][:, i], color="#255e83", lw=.6,
                label="New GPU minus CPU")
        ax.plot(b["time_s"], histories[1][:, i] - histories[0][:, i], color="#cf6433", lw=.55,
                label="New GPU minus old GPU")
        ax.set(xlim=(0, 5), xlabel="Time [s]", ylabel="Difference: " + labels[i])
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(alpha=.18)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Absolute response differences over the full history")
    fig.savefig(args.out / "differences.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    times = [float(cpu["wall_s"]), float(old["gpu"]["online_s"]), float(new["gpu"]["online_s"])]
    bars = axes[0].bar(["CPU CAD\narchived", "GPU CAD\nbefore", "GPU CAD\nours"], times,
                       color=["#9aa6ac", "#255e83", "#cf6433"])
    axes[0].bar_label(bars, fmt="%.2f s", padding=3)
    axes[0].set(ylabel="Online time [s]", ylim=(0, max(times) * 1.15))
    phases = ["broad_s", "sort_s", "search_s", "group_s", "integral_s"]
    x = np.arange(len(phases))
    axes[1].bar(x - .18, [float(old["gpu"][k]) for k in phases], .36, color="#255e83", label="Before")
    axes[1].bar(x + .18, [float(new["gpu"][k]) for k in phases], .36, color="#cf6433", label="Ours")
    axes[1].set(xticks=x, xticklabels=["Broad", "Sort", "Confirm", "Connect", "Integrate"], ylabel="Stage time [s]")
    axes[1].legend()
    fig.suptitle("Same input and integration settings - preprocessing excluded")
    fig.savefig(args.out / "timing.png", dpi=180)
    plt.close(fig)
    print(json.dumps({k: report[k] for k in ("gpu_speedup", "cpu_speedup", "contact_events")}, indent=2))
    if any(not e["same_switch_frames_old"] or not e["same_switch_frames_cpu"] or
           e["different_frames_old"] or e["different_frames_cpu"] for e in events.values()):
        raise SystemExit("Contact switching changed; investigate before accepting")


if __name__ == "__main__":
    main()
