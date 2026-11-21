"""Plot the revised NASA benchmark and export its measured LaTeX results."""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compare import NASA, read
from plots import BLUE, GRID, INK, ORANGE, save


def plots(root, summary, baseline):
    folder = root/"figures"
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.1), layout="constrained")
    for ax, (key, (rpm, torque)) in zip(axes.flat, NASA.items(), strict=True):
        wave = read(root/"curves"/f"nasa-{key}.csv")
        samples = read(baseline/"measurements"/f"nasa-7{key}.csv")
        x = wave["roll_deg"]
        keep = (samples["roll_deg"] >= x[0]) & (samples["roll_deg"] <= x[-1])
        ax.plot(x, wave["legacy_brep"], color="#84909d", lw=1, label="Original B-rep")
        ax.plot(x, wave["brep"], color=BLUE, lw=1.5, label="Revised B-rep")
        ax.plot(x, wave["mesh"], color=ORANGE, lw=1.1, ls="--", label="Revised triangle BVH")
        ax.plot(samples["roll_deg"][keep], samples["force_N"][keep], "o", color=INK,
                ms=2.8, label="NASA digitization", zorder=4)
        if key=="b":
            loss = summary["conditions"][key]["brep"]["loss"]["measured_zero_span_deg"]
            ax.axvspan(*loss, color="#dae4ed", zorder=0, alpha=.7)
        ax.set(xlim=(30, 11.8), ylim=(0, 1.14*max(wave["measured"].max(), wave["brep"].max(), wave["mesh"].max())),
               xlabel="Driven tooth roll angle (°)", ylabel="Single-tooth normal force (N)")
        ax.set_title(f"{key} · {rpm:,} rpm · {torque:.1f} N m", loc="left", weight="bold")
        ax.grid(axis="y", color=GRID, lw=.55)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=4, frameon=False)
    save(fig, folder, "nasa-revised-force")

    fig, axes = plt.subplots(3, 2, figsize=(7.8, 5.7), layout="constrained")
    for row, key in enumerate("bad"):
        native = read(root/"statics"/f"nasa-{key}-brep.csv")
        mesh = read(root/"statics"/f"nasa-{key}-mesh.csv")
        x = native["mesh_cycle"]
        phase = np.r_[x, 1.]
        forces = np.nan_to_num([native["force0_N"], native["force1_N"]])
        split = np.min(forces, axis=0)/np.sum(forces, axis=0)
        ax, bx = axes[row]
        ax.stackplot(phase, np.r_[1-split, 1-split[0]], np.r_[split, split[0]], colors=(BLUE, "#adc9d7"),
                     labels=("Larger share", "Smaller share"))
        for trace, color, ls, label in ((native, BLUE, "-", "B-rep"), (mesh, ORANGE, "--", "Triangle BVH")):
            te = trace["te_um"]-np.mean(trace["te_um"])
            bx.plot(phase, np.r_[te, te[0]], color=color, ls=ls, lw=1.1, label=label)
        ax.set_title(f"{NASA[key][1]:.1f} N m", loc="left", weight="bold")
        ax.set(xlim=(0, 1), ylim=(0, 1), ylabel="Load fraction")
        bx.set(xlim=(0, 1), ylabel="AC static TE (µm)")
        for axis in (ax, bx):
            axis.grid(axis="y", color=GRID, lw=.55)
            if row==2:
                axis.set_xlabel("Mesh cycle")
    axes[0, 0].legend(frameon=True, facecolor="white", framealpha=.98, edgecolor=GRID,
                      ncols=2, loc="lower center")
    axes[0, 1].legend(frameon=False, loc="upper left")
    save(fig, folder, "nasa-revised-statics")

    relief = read(root.parent/"models"/"rig28-profiles.csv")
    fig, ax = plt.subplots(figsize=(5.4, 2.6), layout="constrained")
    roll = relief["roll_deg"]
    for name, nominal, color in (("driver", 23., BLUE), ("driven", 25., ORANGE)):
        ax.plot(roll, relief[f"{name}_um"], color=color, lw=1.4, label=f"{name.title()} · digitized shape")
        ax.plot(roll, nominal*np.maximum(0, (roll-24)/(roll[-1]-24)), ls="--", color=color, lw=.9)
    ax.set(xlabel="Gear roll angle (°)", ylabel="Normal relief (µm)", xlim=(7, 31.5))
    ax.grid(axis="y", color=GRID, lw=.55)
    ax.legend(loc="upper left")
    ax.text(.02,.60,"Dashed · Table I linear model",transform=ax.transAxes,color=INK)
    save(fig, folder, "nasa-revised-profile")


def latex(root, summary, manuscript, beam):
    figure = Path(os.path.relpath(root/"figures", manuscript)).as_posix()
    stat = summary["statics"]["a"]["brep"]
    cases = summary["conditions"]
    low = cases["b"]["brep"]["loss"]
    selected = summary["static_calibration"]["selected"]
    timing = json.loads((root/"timing.json").read_text(encoding="utf-8"))
    step_audit = json.loads((root/"step-audit.json").read_text(encoding="utf-8"))
    def angle_text(pair):
        return ", ".join(f"{x:.2f}" for x in pair)
    attrs = {
        "NasaRevisedPath": rf"\detokenize{{{figure}/}}",
        "NasaRevisedStiffness": f"{stat['pitch_stiffness_N_m']/1e6:.1f}",
        "NasaRevisedDamping": f"{summary['settings']['tooth_damping_ratio']:.2f}",
        "NasaRevisedKBeam": f"{beam['statics']['a']['brep']['pitch_stiffness_N_m']/1e6:.1f}",
        "NasaRevisedFrequencies": angle_text(summary["modal_audit"]["single_pair_pitch_Hz"]),
        "NasaRevisedGearboxFrequencies": angle_text(summary["modal_audit"]["table_ii_gearbox_static_reconstruction_Hz"]),
        "NasaRevisedStaticSpan": angle_text(selected["plateau_deg"]),
        "NasaRevisedStaticTarget": angle_text(summary["static_calibration"]["target_plateau_deg"]),
        "NasaRevisedLowMin": f"{low['min_force_in_measured_zero_N']:.1f}",
        "NasaRevisedLowOverlap": f"{100*low['zero_overlap_fraction']:.0f}",
        "NasaRevisedObservedSpan": angle_text(low["measured_zero_span_deg"]),
        "NasaBeamInternalSpan": angle_text(beam["conditions"]["b"]["brep"]["loss"].get("model_internal_zero_span_deg",[])) or r"\varnothing",
        "NasaRevisedSmallestGap": f"{min(case[mode]['min_gap_um'] for case in cases.values() for mode in ('brep','mesh')):.2f}",
        "NasaRevisedStepB": f"{cases['b']['brep_fine_step']['delta_to_brep_N_rms']:.2f}",
        "NasaRevisedStepD": f"{cases['d']['brep_fine_step']['delta_to_brep_N_rms']:.2f}",
        "NasaQuarterStepDelta": f"{step_audit['rmse_to_original_1us_N']:.2f}",
        "NasaQuarterStepMin": f"{step_audit['raw_selected_min_force_N']:.2f}",
        "NasaQuarterStepSamples": f"{step_audit['raw_selected_tooth_samples']:,}",
        "NasaRevisedTimeBrep": f"{timing['brep_median_wall_s']:.2f}",
        "NasaRevisedTimeBvh": f"{timing['mesh_median_wall_s']:.2f}",
        "NasaRevisedSpeed": f"{timing['mesh_to_brep_wall_ratio']:.2f}",
        "NasaRevisedDualShares": ", ".join(f"{100*summary['statics'][key]['brep']['double_loaded_fraction']:.1f}"
                                             for key in "bad"),
    }
    lines = [r"% Generated by benchmarks/gears/experimental/nasa_revision_artifacts.py."]
    lines.extend(rf"\newcommand{{\{key}}}{{{value}}}" for key,value in attrs.items())
    lines.append(r"\newcommand{\NasaRevisedRows}{%")
    for key, case in cases.items():
        model = case["brep"]
        error = 100*(model["prediction_peak"]/model["test_peak"]-1)
        cells = [f"7({key})", *(f"{case[mode]['rms']:.1f}" for mode in ("legacy_brep","inertia_only")),
                 f"{beam['conditions'][key]['brep']['rms']:.1f}",
                 *(f"{case[mode]['rms']:.1f}" for mode in ("brep","mesh")),
                 f"{error:+.1f}"]
        lines.append(" & ".join(cells)+r" \\")
    lines.append("}")
    (manuscript/"nasa-revised-results.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--manuscript", type=Path, default=Path("manuscript"))
    parser.add_argument("--baseline",type=Path,default=Path("results/gears/experimental"))
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": INK, "text.color": INK})
    summary = json.loads((args.results/"summary.json").read_text(encoding="utf-8"))
    beam = json.loads((args.results.parent/"summary.json").read_text(encoding="utf-8"))
    if "model_internal_zero_span_deg" not in beam["conditions"]["b"]["brep"]["loss"]:
        from nasa_revision import loss_metrics
        wave = read(args.results.parent/"curves"/"nasa-b.csv")
        beam["conditions"]["b"]["brep"]["loss"] = loss_metrics(wave["roll_deg"],wave["measured"],wave["brep"])
    plots(args.results.resolve(), summary, args.baseline)
    latex(args.results.resolve(), summary, args.manuscript.resolve(), beam)
