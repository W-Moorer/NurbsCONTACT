"""Assess guide and support-plate dissipation on the published slider-crank rig."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from clearance import RigDissipation, compare, restitution_scale, simulate, write_deck


REFERENCE = "https://doi.org/10.1007/s11071-010-9899-8"
RIG_DETAIL = "https://doi.org/10.1007/s11071-013-0787-x"
CALIBRATION = (200, .25)
MOUNTED = RigDissipation(1.5, 1., 0., 0.)
DAMPED = RigDissipation(1.35, 5., 2., 1.5, .00533)
EVENT_TRIAL = RigDissipation(1.35, 5., 30., .5)


def measured(path):
    return np.loadtxt(path, delimiter=",", skiprows=1).T


def observed_peak(angle, acceleration, low, high):
    within = (angle >= low) & (angle <= high)
    a = acceleration[within]
    x = angle[within]
    location = int(np.argmin(a))
    return {"angle_deg": float(x[location]), "acceleration_m_s2": float(a[location]),
            "interior": 0 < location < len(x)-1}


def assess(phase, trace, reference, travel, rpm, clearance):
    scores, _, position = compare(phase, trace, reference, travel, clearance, rpm)
    phase_grid = np.arange(1441)*.5
    a_measured = np.interp(phase_grid, *reference)
    a_simulated = np.interp(phase_grid, phase, trace["a1"])
    recorded = (phase >= 0) & (phase <= 720)
    local_phase = phase % 360
    quiet = recorded & (local_phase >= 90) & (local_phase <= 180)
    raw = trace["a1"]
    raw_error = raw-np.interp(phase, *reference)
    scores["raw_sample_rmse_m_s2"] = float(np.sqrt(np.mean(raw_error[recorded]**2)))
    scores["quiet_window_rmse_m_s2"] = float(np.sqrt(np.mean(raw_error[quiet]**2)))
    scores["quiet_window_max_abs_acceleration_m_s2"] = float(np.max(np.abs(raw[quiet])))
    scores["position_rmse_mm"] = float(1000*np.sqrt(np.mean((position[:, 1]-position[:, 2])**2)))
    scores["cycles"] = []
    for start in (0, 360):
        window = (phase >= start+43) & (phase <= start+63)
        scores["cycles"].append({
            "full_half_cycle_peak": observed_peak(phase-start, trace["a1"], 0, 180),
            "experiment_window_peak": observed_peak(phase-start, trace["a1"], 36, 58),
            "fraction_open_in_published_window": float(np.mean(trace["p0_gap_m"][window] > 0)),
        })
    bins = np.clip(np.rint(2*phase[recorded]).astype(int), 0, len(phase_grid)-1)
    lows = np.full(len(phase_grid), np.inf)
    highs = np.full(len(phase_grid), -np.inf)
    np.minimum.at(lows, bins, raw[recorded])
    np.maximum.at(highs, bins, raw[recorded])
    return scores, np.column_stack((phase_grid, a_measured, a_simulated, lows, highs))


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def read(variant):
        return measured(results / f"200rpm-0.25mm-{variant}.csv")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), layout="constrained")
    for index, (axis, limits, title) in enumerate(zip(
        axes, ((0, 360), (25, 70), (90, 180)), ("First evaluated revolution", "Impact region", "Sustained response")
    )):
        x, y, *_ = read("revised")
        inside = (x >= limits[0]) & (x <= limits[1])
        axis.plot(x[inside], y[inside], c="#343d47", label="Experiment", linewidth=1.2)
        for name, color, label in (("mounted", "#a75e49", "Mounting corrected"),
                                   ("revised", "#007e92", "Calibrated dissipation"),
                                   ("revised-half", "#6b638b", "Calibrated, half step")):
            if index == 0 and name == "revised-half":
                continue
            x, _, a, lows, highs = read(name)
            if name != "revised-half":
                axis.fill_between(x[inside], lows[inside], highs[inside], color=color, alpha=.19, linewidth=0)
            axis.plot(x[inside], a[inside], c=color, label=label, lw=1,
                      linestyle="--" if name == "revised-half" else "-")
        axis.set(xlim=limits, xlabel="Crank angle (°)", title=title)
        axis.grid(axis="y", alpha=.22)
        axis.spines[["right","top"]].set_visible(False)
    axes[0].set_ylabel("Slider acceleration (m/s²)")
    axes[0].legend(frameon=False, loc="best", fontsize=8)
    fig.savefig(results / "dissipation-comparison.pdf")
    fig.savefig(results / "dissipation-comparison.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("source", type=Path, help="directory containing digitized Flores curves")
    parser.add_argument("results", type=Path)
    parser.add_argument("--reuse-traces", action="store_true")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    exe = str(args.exe.resolve())
    base = restitution_scale()
    travel = measured(args.source / "measured-position-fig5b.csv")
    variants = (
        ("mounted", MOUNTED, *CALIBRATION, 0., 10e-6),
        ("revised", DAMPED, *CALIBRATION, 0., 10e-6),
        ("revised-half", DAMPED, *CALIBRATION, 0., 5e-6),
        ("revised-mesh", DAMPED, *CALIBRATION, 5e-6, 10e-6),
        ("event-trial", EVENT_TRIAL, *CALIBRATION, 0., 10e-6),
        ("revised", DAMPED, 150, .25, 0., 10e-6),
        ("revised", DAMPED, 200, .50, 0., 10e-6),
        ("revised", DAMPED, 250, .25, 0., 10e-6),
    )

    def run(case):
        name, model, rpm, clearance, mesh, step = case
        work = args.work / name / f"{rpm}rpm-{clearance:.2f}mm"
        work.mkdir(parents=True, exist_ok=True)
        deck = write_deck(exe, work, rpm, clearance, base, model)
        ref = measured(args.source / f"measured-{deck.stem}.csv")
        phase, trace, stats = simulate(exe, deck, work, name, mesh, step, rpm, 6, args.reuse_traces)
        metrics, paired = assess(phase, trace, ref, travel, rpm, clearance)
        if (rpm, clearance) != CALIBRATION:
            del metrics["position_rmse_mm"]
        np.savetxt(args.results / f"{deck.stem}-{name}.csv", paired, delimiter=",",
                   header="angle_deg,measured_acceleration_m_s2,simulated_acceleration_m_s2,"
                   "sample_min_m_s2,sample_max_m_s2", comments="")
        print(deck.stem, name, metrics["uniform_phase_acceleration_rmse_m_s2"],
              metrics["quiet_window_rmse_m_s2"], metrics["cycles"], flush=True)
        # Concurrent runs assess accuracy. Their wall times are not speed benchmarks.
        return {"variant": name, "rpm": rpm, "clearance_mm": clearance, "mesh_deflection_m": mesh,
                "parameters": asdict(model)} | metrics | {key: stats[key] for key in ("step_s", "accepted", "subdivisions", "missing")}

    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(run, variants))
    fine = measured(args.results / "200rpm-0.25mm-revised-half.csv")[2]
    coarse = measured(args.results / "200rpm-0.25mm-revised.csv")[2]
    (args.results / "summary.json").write_text(json.dumps({
        "sources": [REFERENCE, RIG_DETAIL],
        "calibration": {"rpm": CALIBRATION[0], "clearance_mm": CALIBRATION[1],
                        "published_phase_shift_deg": 0, "output_filter": None},
        "half_step_waveform_delta_rmse_m_s2": float(np.sqrt(np.mean((fine-coarse)**2))),
        "runs": rows,
    }, indent=2)+"\n", encoding="utf-8")
    plot(args.results)


if __name__ == "__main__":
    main()
