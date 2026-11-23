"""Compare both broad phases at the same HHT settings and plot the recorded contact switch."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text())


def responses(trace, nominal):
    speed = trace[:, 3:6] + nominal
    torque = np.column_stack((trace[:, 7], trace[:, 8] + trace[:, 11], trace[:, 12]))
    return speed, torque


def compare(work, out):
    out.mkdir(parents=True, exist_ok=True)
    deck = np.load(work / "assembly.npz")
    nominal = deck["bodies"][[0, 1, 3], 8]
    signs = deck["bodies"][[0, 1, 3], 3]
    native = np.load(work / "rd-switch-geometric-physics.npz")
    native_time = np.load(work / "rd-reducer-1.26s-125us-patch0.0625.npz")["t"]
    native_speed = native["global_speed_rad_s"] * signs
    native_torque = native["global_accel_rad_s2"] * signs * deck["coord"][:, 0] - native["support_raw_TZ_Nm"]
    native_torque[:, [0, 2]] -= native["load_raw_TZ_Nm"]
    curves = {"refined RecurDyn": (native_time, native_speed[:, 1], native_torque[:, 2])}
    metrics = {}
    for method in ("cad", "mesh"):
        runs = {}
        series = {}
        for mode in ("projection", "blocks"):
            stem = work / f"{mode}-{method}"
            data = read(stem.with_suffix(".json"))
            trace = np.load(stem.with_suffix(".npy"))
            assert data["terminated"] == "complete" and data["physical_s"] == 3
            assert data["steps"] + 1 == len(trace) and np.isfinite(trace).all()
            assert (data["contact_samples"], data["step_s"], data["stiffness_scope"]) == (16, 62.5e-6, "pair")
            assert 0 < data["online_s"] < 1800
            run = {
                k: data[k]
                for k in (
                    "gpu",
                    "steps",
                    "trials",
                    "accepted_increments",
                    "over_budget",
                    "p99_us",
                    "setup_s",
                    "online_s",
                    "device_pool_MiB",
                    "directional_probe_visits",
                )
            }
            run["visits_per_step"] = sum(run["directional_probe_visits"]) / run["steps"]
            if mode == "blocks":
                run.update(guard_mm=data["guard_mm"], directional_rebuilds=data["directional_rebuilds"])
            runs[mode] = run
            series[mode] = trace
            time = np.arange(len(trace)) * data["step_s"]
            speed, torque = responses(trace, nominal)
            curves[f"{method}-{mode}"] = (time, speed[:, 1], torque[:, 2])
        first, second = runs["projection"], runs["blocks"]
        diff = series["blocks"] - series["projection"]
        diff_v, diff_t = responses(diff, np.zeros(3))
        switch = (time >= 1.251 - 1e-12) & (time <= 1.254 + 1e-12)
        historical = np.load(work / f"pair-{method}.npy")
        at = round(1.25275 / data["step_s"])
        metrics[method] = dict(
            runs=runs,
            response_shape=list(series["blocks"].shape),
            all_response_bits_identical=np.array_equal(
                series["blocks"].view(np.uint64), series["projection"].view(np.uint64)
            ),
            online_speedup=first["online_s"] / second["online_s"],
            reduction_in_visits=1 - second["visits_per_step"] / first["visits_per_step"],
            max_shaft_speed_change_rad_s=np.abs(diff_v).max(axis=0).tolist(),
            max_shaft_torque_change_Nm=np.abs(diff_t).max(axis=0).tolist(),
            max_normal_force_change_N=np.abs(diff[:, [6, 10]]).max(axis=0).tolist(),
            changed_counts_of_bearing_pairs=int(np.count_nonzero(diff[:, [9, 13]])),
            switch_max_speed_change_rad_s=np.abs(diff_v[switch]).max(axis=0).tolist(),
            switch_max_torque_change_Nm=np.abs(diff_t[switch]).max(axis=0).tolist(),
            switch_middle_speed_rad_s=float(speed[at, 1]),
            switch_output_torque_Nm=float(torque[at, 2]),
            max_shaft_speed_change_vs_archived_rad_s=float(
                np.abs(series["blocks"][:, 3:6] - historical[:, 3:6]).max()
            ),
        )
    at = int(np.argmin(abs(native_time - 1.25275)))
    np.testing.assert_allclose(native_time[at], 1.25275, rtol=0, atol=1e-10)
    pilot = {}
    for name in ("rebuild", "default", "wide", "projection"):
        run = read(work / f"pilot-{name}.json")
        pilot[name] = {
            k: run[k]
            for k in (
                "physical_s",
                "steps",
                "trials",
                "online_s",
                "guard_mm",
                "directional_probe_visits",
                "directional_rebuilds",
            )
            if k in run
        }
    summary = dict(
        physical_s=3.0,
        step_us=62.5,
        reducer=metrics,
        pilot=pilot,
        switch_reference=dict(
            time_s=float(native_time[at]),
            middle_speed_rad_s=float(native_speed[at, 1]),
            output_torque_Nm=float(native_torque[at, 2]),
        ),
        cpu_box=read(work / "blocks-coverage.json"),
        fixed_poses=read(work / "blocks-real-coverage.json"),
    )
    for case in ("sphere", "cam"):
        prefix = work / f"final-{case}"
        traces = [np.load(f"{prefix}-{m}.npy") for m in ("projection", "blocks")]
        runs = [read(Path(f"{prefix}-{m}.json")) for m in ("projection", "blocks")]
        assert all(r["terminated"] == "complete" for r in runs)
        summary[case] = dict(
            physical_s=runs[0]["physical_s"],
            steps=runs[0]["steps"],
            max_difference_by_column=np.abs(traces[0] - traces[1]).max(axis=0).tolist(),
            online_s_by_projection_blocks=[r["online_s"] for r in runs],
            gpu=runs[0]["gpu"],
        )
    (out / "block-results.json").write_text(json.dumps(summary, indent=2) + "\n")
    draw_switch(out, curves)


def draw_switch(out, curves):
    plt.rcParams.update(
        {"font.size": 8, "pdf.fonttype": 42, "axes.spines.right": False, "axes.spines.top": False}
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.4), sharex=True, sharey="row", layout="constrained")
    with (out / "block-switch.csv").open("w", newline="") as f:
        table = csv.writer(f)
        table.writerow(["method", "time_s", "middle_speed_rad_s", "output_contact_torque_Nm"])
        for name, (t, v, q) in curves.items():
            sel = (t >= 1.251 - 1e-12) & (t <= 1.254 + 1e-12)
            table.writerows((name, *row) for row in zip(t[sel], v[sel], q[sel], strict=True))
    for col, method in enumerate(("cad", "mesh")):
        axes[0, col].set_title(("STEP (ours)", "Pure mesh")[col], loc="left", fontweight="bold")
        for name, style in (
            ("refined RecurDyn", dict(color="#77726D", ls="--", lw=1.2, label="Refined RecurDyn")),
            (f"{method}-projection", dict(color="#263647", lw=1.5, label="Projection")),
            (
                f"{method}-blocks",
                dict(color="#007B80", lw=0, marker="o", mfc="none", ms=2.9, label="Reusable blocks"),
            ),
        ):
            t, v, q = curves[name]
            sel = (t >= 1.251 - 1e-12) & (t <= 1.254 + 1e-12)
            for ax, y in zip(axes[:, col], (v, q), strict=True):
                ax.plot((t[sel] - 1.25275) * 1000, y[sel], **style)
        for ax in axes[:, col]:
            ax.axvline(0, color="#B8B8B8", lw=0.65, zorder=0)
            ax.grid(axis="y", color="#EEEEEE", lw=0.6)
        axes[1, col].set_xlabel("Time from 1.25275 s (ms)")
    axes[0, 0].set_ylabel("Middle-shaft speed (rad/s)")
    axes[1, 0].set_ylabel("Output contact torque (N m)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncols=3, frameon=False)
    fig.savefig(out / "block-switch.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    compare(args.work, args.output)
