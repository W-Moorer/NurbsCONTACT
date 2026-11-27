"""Audit complete 5 s sampled-mesh responses and plot scheduling measurements."""

import argparse
import hashlib
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("work", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument(
        "--compare", default="reference", help="Explicit curve comparison, not the fastest run"
    )
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ref = np.load(a.reference.with_suffix(".npy"))
    baseline = json.loads(a.reference.with_suffix(".json").read_text())
    if ref.shape != (50001, 14) or not np.isfinite(ref).all():
        raise ValueError("Expected complete five-second reference")
    values = {"reference": baseline}
    audits = {}
    best, best_time, best_name = ref, baseline["online_s"], a.compare
    if a.compare != "reference" and a.compare not in a.runs:
        raise ValueError("The selected comparison must be included in --runs")
    for name in a.runs:
        data = np.load(a.work / f"{name}.npy")
        run = json.loads((a.work / f"{name}.json").read_text())
        if (
            run["terminated"] != "complete"
            or data.shape != ref.shape
            or not np.isfinite(data).all()
        ):
            raise ValueError(f"Incomplete run: {name}")
        for key in (
            "step_s",
            "contact_samples",
            "minimum_step_s",
            "stiffness_scope",
            "directional_probes",
        ):
            if run[key] != baseline[key]:
                raise ValueError(f"Changed experiment: {name} {key}")
        identical = np.array_equal(data, ref)
        audits[name] = dict(
            bitwise_equal=identical,
            response_sha256=hashlib.sha256(data.tobytes()).hexdigest(),
            max_abs_difference_by_column=np.max(abs(data - ref), axis=0).tolist(),
            max_abs_normal_load_difference_N=float(np.max(abs(data[:, [6, 10]] - ref[:, [6, 10]]))),
            max_abs_contact_torque_difference_Nm=float(
                np.max(abs(data[:, [7, 8, 11, 12]] - ref[:, [7, 8, 11, 12]]))
            ),
            load_switch_mismatches=np.count_nonzero(
                (data[:, [6, 10]] > 0) != (ref[:, [6, 10]] > 0), axis=0
            ).tolist(),
            output_torque_peak_Nm=float(data[:, 12].max()),
            output_torque_peak_step=int(np.argmax(data[:, 12])),
            output_torque_max_jump_Nm=float(abs(np.diff(data[:, 12])).max()),
        )
        values[name] = run
        if name == a.compare:
            best, best_time = data, run["online_s"]
    report = dict(
        scope="sampled-mesh closed loop; distinct from CPU contour-ray force definition",
        steps=50000,
        duration_s=5.0,
        step_s=baseline["step_s"],
        runs=values,
        audits=audits,
        reference_sha256=hashlib.sha256(ref.tobytes()).hexdigest(),
        comparison_run=best_name,
        speedup=baseline["online_s"] / best_time,
    )
    (a.out / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(
        a.out / "response.npz", time_s=np.arange(50001) * 0.0001, original=ref, optimized=best
    )
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), layout="constrained")
    names = list(values)
    phases = ("prepare_s", "search_s", "merge_and_hht_s")
    left = np.zeros(len(names))
    for phase, color in zip(phases, ("#8ba6b1", "#277588", "#d99662")):
        times = np.array([values[name]["phase_profile"][phase] for name in names])
        axes[0].barh(names, times, left=left, color=color, label=phase.replace("_s", ""))
        left += times
    axes[0].invert_yaxis()
    axes[0].set_xlabel("GPU phase time for 5 s [s]")
    axes[0].legend(ncols=3, fontsize=8)
    bars = axes[1].bar(names, [values[name]["online_s"] for name in names], color="#277588")
    axes[1].bar_label(bars, fmt="%.2f s", padding=3)
    axes[1].set_ylabel("Total online time [s]")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_ylim(0, max(v["online_s"] for v in values.values()) * 1.18)
    fig.savefig(a.out / "timing.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(11, 7), layout="constrained", sharex=True)
    t = np.arange(len(ref)) * 0.0001
    channels = (
        (3, "Input speed ripple [rad/s]"),
        (4, "Intermediate speed ripple [rad/s]"),
        (6, "Stage 1 normal force [N]"),
        (10, "Stage 2 normal force [N]"),
        (8, "Stage 1 driven torque [N m]"),
        (12, "Output contact torque [N m]"),
    )
    for ax, (col, label) in zip(axes.flat, channels):
        ax.plot(t, ref[:, col], color="#db985e", lw=0.7, label="Before")
        ax.plot(t, best[:, col], color="#277588", lw=0.5, ls="--", label="After")
        ax.set(ylabel=label, xlim=(0, 5))
        ax.grid(alpha=0.15)
    axes[0, 0].legend()
    for ax in axes[-1]:
        ax.set_xlabel("Simulated time [s]")
    fig.suptitle("GPU sampled-mesh dynamics: original and scene-sized HHT")
    fig.savefig(a.out / "response.png", dpi=180)
    fig.savefig(a.out / "response.pdf")
    plt.close(fig)
    print(
        json.dumps(
            {
                "comparison": best_name,
                "online_s": best_time,
                "speedup": report["speedup"],
                "audits": audits,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
