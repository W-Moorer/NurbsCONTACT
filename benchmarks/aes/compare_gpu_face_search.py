"""Compare full resident reducer runs before and after GPU geometry optimizations."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_region_reduction import check_regions, compare, read_run
from cpu_cad_pressure_report import history


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    base = args.runs
    variants = dict(before="workspace", jet="jet", bridge="bridge", slabs="slabs")
    folders = {v: base / f"search-{v}-full" for v in variants}
    folders["before_repeat"] = base / "search-before-repeat"
    folders["final"] = base / "search-links-final"
    folders["repeat"] = base / "search-winner-repeat"
    folders["audit"] = base / "search-links-audit"
    runs = {v: read_run(p) for v, p in folders.items()}
    for v, folder in folders.items():
        with (folder / "response.gpu.csv").open() as f:
            timing = next(csv.DictReader(f))
        runs[v]["step_ms"] = {
            key: 1000 * float(timing[f"{key}_step_s"]) for key in ("p50", "p99", "max")
        }
    original = runs["before"]["response_sha256"]
    for name in runs:
        if runs[name]["response_sha256"] != original:
            raise ValueError(f"{name} changed the full response")
        for key in ("queries", "evaluations", "subdivisions"):
            if runs[name][key] != runs["before"][key]:
                raise ValueError(f"{name} changed the {key} count")
    with (folders["audit"] / "response.gpu.csv").open() as f:
        links = int(next(csv.DictReader(f))["checked_links"])
    if links <= 0:
        raise ValueError("the scalar/warp connection comparison did not run")
    traces = {v: history(folders[v] / "response.csv", args.scene) for v in ("before", "final")}
    t = traces["before"]["time_s"]
    if len(t) != 50001 or not np.allclose(np.diff(t), 1e-4, rtol=0, atol=1e-12):
        raise ValueError("expected a five-second record at 0.1 ms")
    np.testing.assert_array_equal(t, traces["final"]["time_s"])
    audit_digest = digest(folders["audit"] / "response.regions.csv")
    if audit_digest != digest(args.before_audit / "response.regions.csv"):
        raise ValueError("per-step support changes, region counts or wrench errors changed")
    stages, events, _ = check_regions(folders["audit"], t)
    torque = traces["before"]["output_torque_Nm"]
    jumps = np.abs(np.diff(torque, prepend=torque[0]))
    boundary = np.argmax(np.where(events, jumps, -1.))
    peak = int(np.argmax(np.abs(torque)))
    report = {
        "scene_sha256": digest(args.scene),
        "executables_sha256": {
            k: digest(base / f"{v}-bin" / "nc_contact_runtime.exe")
            for k, v in {**variants, "final": "search-checked"}.items()
        },
        "scalar_warp_connections_checked": links,
        "runs": runs,
        "regions_by_stage": stages,
        "per_step_region_audit_sha256": audit_digest,
        "response_error_against_before": compare(traces["before"], traces["final"]),
        "all_equal_to_before": {k: v["response_sha256"] == original for k, v in runs.items()},
        "plotted_switch_s": float(t[boundary]),
        "output_torque_peak": {"time_s": float(t[peak]), "Nm": float(torque[peak])},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    fig, ax = plt.subplots(3, 2, figsize=(10, 7.4), layout="constrained")
    labels = ("Before", "Tile cache", "+ Link tests", "+ Ray slabs")
    x = np.arange(len(labels))
    for offset, phase, label, color in (
        (-0.18, "search_s", "Face confirmation", "#295c78"),
        (0.18, "group_s", "Region grouping", "#d58842"),
    ):
        ax[0, 0].bar(x + offset, [runs[k]["phase_s"][phase] for k in variants],
                     0.35, label=label, color=color)
    ax[0, 0].set_xticks(x, labels, fontsize=8)
    ax[0, 0].set_ylabel("Online stage time [s]")
    ax[0, 0].legend(fontsize=8)
    keys = ("output_torque_Nm", "normal_N", "normal_N", "speed", "output_torque_Nm")
    titles = ("Output torque [N m]", "Stage 1 load [N]", "Stage 2 load [N]",
              "Intermediate speed [rad/s]", "Torque at face switch [N m]")
    for i, (panel, key, title) in enumerate(zip(ax.flat[1:], keys, titles, strict=True)):
        mask = abs(t - t[boundary]) <= 0.003 if i == 4 else slice(None)
        for v, label, color, style in (
            ("before", "Before", "#295c78", "-"), ("final", "Optimized", "#ad5d32", "--"),
        ):
            y = traces[v][key]
            if y.ndim == 2:
                y = y[:, 0 if i == 1 else 1]
            panel.plot(t[mask], y[mask], style, color=color, lw=0.8, label=label)
        panel.set(xlabel="Time [s]", ylabel=title)
        panel.margins(x=0)
        panel.grid(alpha=.2)
    ax[0, 1].legend(fontsize=8)
    ax[2, 1].axvline(t[boundary], color="#637177", ls=":", lw=0.8)
    fig.savefig(args.out / "response.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("scene", "runs", "before_audit", "out"):
        parser.add_argument(arg, type=Path)
    main(parser.parse_args())
