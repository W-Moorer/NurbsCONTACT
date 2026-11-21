"""Reproduce the NASA profile, quasi-static checks and revised HHT comparison."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from compare import MESH, NASA, metrics, read, read_run, rms, tooth_wave


def invoke(exe, *args):
    result = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{args}: {result.stderr.strip()}")


def fit_static_compliance(exe, models, output):
    # The printed DANST static plateau in Fig. 7(c) extends approximately
    # from 25 to 16 degrees. Its shoulders overlap the other plotted curves.
    target = np.array([16., 25.])
    _, torque = NASA["c"]
    folder = output/"static-calibration"
    folder.mkdir(parents=True, exist_ok=True)
    trials = []
    for k in np.arange(50., 105.1, 2.5)*1e6:
        path = folder/f"pitch-{int(k)}.csv"
        invoke(exe, "statics", "rig28-flex", models, 0., torque, k, path)
        trace = read(path)
        forces = np.concatenate([trace["force0_N"], trace["force1_N"]])
        roll = np.concatenate([trace["roll0_deg"], trace["roll1_deg"]])
        plateau = roll[forces >= .99*np.nanmax(forces)]
        endpoints = np.array([np.min(plateau), np.max(plateau)])
        trials.append({"pitch_stiffness_N_m": float(k), "plateau_deg": endpoints.tolist(),
                       "endpoint_rms_deg": rms(endpoints-target)})
    best = min(trials, key=lambda trial: trial["endpoint_rms_deg"])
    evidence = {"target_plateau_deg": target.tolist(), "torque_N_m": torque,
                "source": "NASA TM-103232 Fig. 7(c) DANST static line, approximate endpoints",
                "threshold_fraction_of_static_load": .99, "trials": trials, "selected": best}
    (folder/"grid.json").write_text(json.dumps(evidence,indent=2)+"\n",encoding="utf-8")
    return best["pitch_stiffness_N_m"], evidence


def modal_audit(pitch_stiffness):
    # Frozen single loaded pair approximated by a torsional spring at the pitch point.
    j = .1129848290276167*np.array([2.10, .00322, .00322, 7.5])
    diff = np.diff(np.eye(4), axis=0)
    radius = .003175*14*np.cos(np.radians(20))

    def frequencies(k):
        matrix = diff.T@np.diag([341, k, 12700])@diff
        values = np.linalg.eigvalsh(matrix/np.sqrt(j[:, None]*j))
        return (np.sqrt(values[1:])/(2*np.pi)).tolist()

    return {"table_ii_gearbox_static_reconstruction_Hz": frequencies(6158),
            "table_ii_reported_Hz": [6.56, 52.5, 1220],
            "danst_reported_Hz": [6.73, 645, 5821],
            "single_pair_pitch_Hz": frequencies(pitch_stiffness*radius**2)}


def static_checks(exe, output, models, pitch_stiffness):
    results = {}
    folder = output/"statics"
    folder.mkdir(exist_ok=True)
    radius = .003175*14*np.cos(np.radians(20))
    for condition in ("b", "a", "d"):
        _, torque = NASA[condition]
        records = {}
        for label, deflection in (("brep", 0), ("mesh", MESH)):
            path = folder/f"nasa-{condition}-{label}.csv"
            invoke(exe, "statics", "rig28-flex", models, deflection, torque, pitch_stiffness, path)
            trace = read(path)
            total = np.nansum([trace["force0_N"], trace["force1_N"]], axis=0)
            record = {
                "max_torque_balance_error_Nm": float(np.max(np.abs(trace["output_torque_Nm"]-torque))),
                "max_sum_vs_nominal_force_N": float(np.max(np.abs(total-torque/radius))),
                "te_peak_to_peak_um": float(np.ptp(trace["te_um"])),
                "double_loaded_fraction": float(np.mean(trace["loaded"] == 2)),
                "min_gap_um": float(1000*np.min(trace["gap_mm"])),
                "pitch_stiffness_N_m": float(trace["pitch_stiffness_N_m"][0]),
            }
            if record["max_torque_balance_error_Nm"] > 1e-6 or record["min_gap_um"] < 2:
                raise RuntimeError(f"Failed static balance or gap check: {path}")
            if label == "brep":
                native = trace["te_um"]
            else:
                record["te_delta_to_brep_um_rms"] = rms(trace["te_um"]-native)
            records[label] = record
        results[condition] = records
    return results


def loss_metrics(x, actual, predicted):
    middle = (x >= 18) & (x <= 23)
    observed = (actual <= 1e-6) & middle
    released = (predicted <= 1e-6) & middle

    def span(mask):
        values = x[mask]
        return [float(values[0]), float(values[-1])] if len(values) else []

    internal = (predicted <= 1e-6) & (x >= 14.5) & (x <= 27.)
    indices = np.flatnonzero(internal)
    components = np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
    return {
        "measured_zero_span_deg": span(observed),
        "model_zero_span_deg": span(released),
        "model_internal_zero_span_deg": span(internal),
        "model_internal_zero_intervals_deg": [[float(x[block[0]]),float(x[block[-1]])]
                                               for block in components if len(block)],
        "zero_overlap_fraction": float(np.count_nonzero(observed & released)/np.count_nonzero(observed)),
        "min_force_in_measured_zero_N": float(np.min(predicted[observed])),
        "max_force_in_measured_zero_N": float(np.max(predicted[observed])),
    }


def measure_timings(exe, models, output):
    summary = json.loads((output/"summary.json").read_text(encoding="utf-8"))
    settings = summary["settings"]
    rpm, torque = NASA["b"]
    repetitions = 3
    times = {"brep":[],"mesh":[]}
    with TemporaryDirectory(prefix="nc-rig28-timing-") as run_folder:
        for repeat in range(repetitions):
            order = (("brep",0.),("mesh",MESH))
            for label, mesh in order if repeat%2==0 else order[::-1]:
                prefix = Path(run_folder)/f"run-{repeat}-{label}"
                invoke(exe,"experiment","rig28-flex",models,mesh,rpm,torque,
                       settings["nominal_step_s"],.16,settings["pitch_stiffness_override_N_m"],
                       settings["tooth_damping_ratio"],prefix,1.)
                with prefix.with_suffix(".stats.csv").open(encoding="utf-8",newline="") as file:
                    times[label].append(next(csv.DictReader(file)))
    result = {"condition":"7(b)","repetitions":repetitions,"integration_duration_s":.16,
              "raw_trials":times}
    for label, trials in times.items():
        result[label+"_median_wall_s"] = float(np.median([float(row["wall_s"]) for row in trials]))
        result[label+"_median_setup_s"] = float(np.median([float(row["setup_s"]) for row in trials]))
    result["mesh_to_brep_wall_ratio"] = result["mesh_median_wall_s"]/result["brep_median_wall_s"]
    (output/"timing.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print({key:value for key,value in result.items() if key!="raw_trials"},flush=True)


def modal_sensitivity(exe, baseline, models, output):
    left, right = 60e6, 172e6
    for _ in range(40):
        stiffness = (left+right)/2
        if modal_audit(stiffness)["single_pair_pitch_Hz"][-1]<5821:
            left = stiffness
        else:
            right = stiffness
    output.mkdir(parents=True, exist_ok=True)
    results = {"pitch_stiffness_N_m":stiffness, "tooth_damping_ratio":.10, "cases":{},
               "constraint":"NASA DANST 5821 Hz highest frequency in the frozen single-pair pitch linearization"}
    for panel in ("b", "c"):
        rpm, torque = NASA[panel]
        prefix = output/f"nasa-{panel}"
        invoke(exe,"experiment","rig28-flex",models,0.,rpm,torque,1e-6,.24,stiffness,.10,prefix,1.)
        datum = read(baseline/"curves"/f"nasa-{panel}.csv")
        trace, _ = read_run(prefix)
        wave = tooth_wave(trace,datum["roll_deg"])
        record = metrics(datum["measured"],wave,"roll")
        if panel=="b":
            record["loss"] = loss_metrics(datum["roll_deg"],datum["measured"],wave)
        results["cases"][panel] = record
    (output/"summary.json").write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")


def step_audit(exe, models, output):
    settings = json.loads((output/"summary.json").read_text(encoding="utf-8"))["settings"]
    table = read(output/"curves"/"nasa-b.csv")
    rpm, torque = NASA["b"]
    dt = settings["nominal_step_s"]/4
    prefix = output/"runs"/"nasa-b-brep-quarter-step"
    invoke(exe,"experiment","rig28-flex",models,0.,rpm,torque,dt,settings["duration_s"],
           settings["pitch_stiffness_override_N_m"],settings["tooth_damping_ratio"],prefix,1.)
    trace, work = read_run(prefix)
    x = table["roll_deg"]
    wave = tooth_wave(trace,x)
    loss = loss_metrics(x,table["measured"],wave)
    lo, hi = loss["measured_zero_span_deg"]
    roll = np.array([trace[f"roll{i}_deg"] for i in range(4)])
    force = np.array([trace[f"force{i}_N"] for i in range(4)])
    identity = np.array([trace[f"id{i}"] for i in range(4)])
    selected = (roll>=lo)&(roll<=hi)&np.isfinite(force)
    internal = (roll>=14.5)&(roll<=27.)&np.isfinite(force)
    result = {
        "case":"NASA Fig. 7(b), revised static-calibrated B-rep", "dt_s":dt,
        "median_output_interval_s":float(np.median(np.diff(trace["time"]))),
        "rmse_to_original_1us_N":rms(wave-table["brep"]),
        "rmse_to_0p5us_N":rms(wave-table["brep_fine_step"]),
        "rmse_to_nasa_N":rms(wave-table["measured"]), "loss":loss,
        "raw_selected_tooth_samples":int(np.count_nonzero(selected)),
        "raw_selected_unique_tooth_ids":int(len(np.unique(identity[selected]))),
        "raw_selected_min_force_N":float(np.min(force[selected])),
        "raw_selected_zero_count":int(np.count_nonzero(force[selected]<=1e-6)),
        "raw_internal_zero_count":int(np.count_nonzero(force[internal]<=1e-6)),
        "elapsed_wall_s":work["wall_s"], "accepted":int(work["accepted"]),
        "probe_misses":int(work["probe_misses"]),
    }
    np.savetxt(output/"curves"/"nasa-b-step-audit.csv",
               np.column_stack((x,table["measured"],table["brep"],table["brep_fine_step"],wave)),
               header="roll_deg,nasa,brep_1us,brep_0p5us,brep_0p25us",delimiter=",",comments="",fmt="%.10g")
    (output/"step-audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)


def compare(exe, baseline, models, output, pitch_stiffness, zeta, duration, legacy_models, fit_static):
    calibration = None
    if fit_static:
        pitch_stiffness, calibration = fit_static_compliance(exe, models, output)
    for folder in ("curves", "runs"):
        (output/folder).mkdir(parents=True, exist_ok=True)
    summary = {
        "settings": {"pitch_stiffness_override_N_m": pitch_stiffness,
                     "tooth_damping_ratio": zeta, "duration_s": duration, "nominal_step_s": 1e-6},
        "statics": static_checks(exe, output, models, pitch_stiffness),
        "conditions": {},
        "static_calibration": calibration,
    }
    summary["modal_audit"] = modal_audit(summary["statics"]["a"]["brep"]["pitch_stiffness_N_m"])
    for condition, (rpm, torque) in NASA.items():
        name = "nasa-"+condition
        old = read(baseline/"curves"/f"{name}.csv")
        x, measured = old["roll_deg"], old["measured"]
        labels = ["measured", "legacy_brep", "legacy_mesh"]
        waves = [measured, old["brep"], old["mesh"]]
        details = {"rpm": rpm, "input_torque_Nm": torque}
        for label, wave in zip(labels[1:], waves[1:], strict=True):
            details[label] = metrics(measured, wave, "roll")
            if condition == "b":
                details[label]["loss"] = loss_metrics(x, measured, wave)
        modes = [("inertia_only", "rig28", 0., 1e-6, legacy_models, 60e6, .20, 10.),
                 ("brep", "rig28-flex", 0., 1e-6, models, pitch_stiffness, zeta, 1.),
                 ("mesh", "rig28-flex", MESH, 1e-6, models, pitch_stiffness, zeta, 1.)]
        if condition in ("b", "d"):
            modes.append(("brep_fine_step", "rig28-flex", 0., .5e-6, models, pitch_stiffness, zeta, 1.))
        for label, case, deflection, dt, geometry, k, damping, inertia in modes:
            prefix = output/"runs"/f"{name}-{label}"
            invoke(exe, "experiment", case, geometry, deflection, rpm, torque, dt, duration, k, damping, prefix, inertia)
            trace, work = read_run(prefix)
            wave = tooth_wave(trace, x)
            record = metrics(measured, wave, "roll")
            record["min_gap_um"] = float(1000*np.min(trace["gap_mm"]))
            record["late_wave_delta_N_rms"] = rms(tooth_wave(trace[trace["time"]>=.75*duration], x)-wave)
            record["work"] = work
            if label in ("mesh", "brep_fine_step"):
                record["delta_to_brep_N_rms"] = rms(wave-waves[labels.index("brep")])
            if condition == "b":
                record["loss"] = loss_metrics(x, measured, wave)
            details[label] = record
            labels.append(label)
            waves.append(wave)
        np.savetxt(output/"curves"/f"{name}.csv", np.column_stack((x,*waves)),
                   header=",".join(("roll_deg",*labels)), delimiter=",", comments="", fmt="%.10g")
        summary["conditions"][condition] = details
        (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
        print(name, {key: round(details[key]["rms"],2) for key in labels[1:]}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe",type=Path)
    parser.add_argument("baseline",type=Path)
    parser.add_argument("--duration",default=.48,type=float)
    parser.add_argument("--pitch-stiffness",default=0.,type=float)
    parser.add_argument("--damping",default=.10,type=float)
    parser.add_argument("--fit-static",action="store_true")
    parser.add_argument("--output",type=Path)
    diagnostics = parser.add_mutually_exclusive_group()
    diagnostics.add_argument("--timing",action="store_true")
    diagnostics.add_argument("--modal-sensitivity",action="store_true")
    diagnostics.add_argument("--step-audit",action="store_true")
    parser.add_argument("--legacy-models",default=Path("results/gears/models"),type=Path)
    args=parser.parse_args()
    root=args.baseline/"revised"
    if args.timing:
        measure_timings(args.exe.resolve(),root/"models",args.output or root/"static-matched")
    elif args.modal_sensitivity:
        modal_sensitivity(args.exe.resolve(),args.baseline,root/"models",
                          args.output or root/"modal-sensitivity")
    elif args.step_audit:
        step_audit(args.exe.resolve(),root/"models",args.output or root/"static-matched")
    else:
        compare(args.exe.resolve(),args.baseline,root/"models",args.output or root,
                args.pitch_stiffness,args.damping,args.duration,args.legacy_models,args.fit_static)
