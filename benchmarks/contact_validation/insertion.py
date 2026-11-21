"""Replay independent copper-ring trials through a calibrated axial STEP interface."""

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
from scipy.optimize import least_squares


SOURCE = "https://zenodo.org/records/6364950"
DOWNLOAD = "https://zenodo.org/api/records/6364950/files/insertion-dataset.zip/content"
CHECKSUM = "61029a290d4512382cd6a11e53def18a"
WINDOW = np.arange(-28, 24)
CALIBRATION_SLOTS = (1, 2)
HOLDOUT_SLOTS = (3, 4)
PRECONTACT_ROWS = 35
NAME = re.compile(r"kd-feb4/(\d)/([^/]+)/(x_ref|x|ft_tool|ft|rs_front)_(.+)\.(csv|jpg)")


def read_trials(path):
    if not path.exists():
        request = Request(DOWNLOAD, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=60) as response:
            archive_bytes = response.read()
    else:
        archive_bytes = path.read_bytes()
    if hashlib.md5(archive_bytes).hexdigest() != CHECKSUM:
        raise ValueError("the archive differs from the version published on Zenodo")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archive_bytes)
    with ZipFile(io.BytesIO(archive_bytes)) as archive:
        by_id = defaultdict(dict)
        for name in archive.namelist():
            matched = NAME.fullmatch(name)
            if matched:
                slot, condition, kind, identifier, _ = matched.groups()
                by_id[int(slot), condition, identifier][kind] = name
        if len(by_id) != 300 or any(len(files) != 5 for files in by_id.values()):
            raise ValueError("expected 300 complete trials with five files each")
        for (slot, condition, identifier), files in sorted(by_id.items()):
            pose, command, force = (
                np.loadtxt(io.BytesIO(archive.read(files[kind])), delimiter=",")
                for kind in ("x", "x_ref", "ft_tool")
            )
            if len(pose) != len(command) or len(command) != len(force):
                raise ValueError(f"pose and force sample counts differ for {identifier}")
            # The first minimum in the reference Z stream is the command's
            # first lower turnaround. It fixes an index before looking at force.
            turnaround = int(np.argmin(command[:, 2]))
            index = turnaround + WINDOW
            if index[0] < PRECONTACT_ROWS or index[-1] >= len(pose):
                raise ValueError(f"insufficient samples about turnaround for {identifier}")
            target = float(command[turnaround, 2])
            z = pose[index, 2]-target
            f = force[index, 2] - np.median(force[:PRECONTACT_ROWS, 2])
            yield slot, condition, identifier, z, f, target


def calibrate(trials, conditions):
    fitting = [trial for trial in trials if trial[0] in CALIBRATION_SLOTS]
    z_mm = np.stack([1000*trial[3] for trial in fitting])
    target = np.stack([trial[4] for trial in fitting])
    indices = np.array([conditions.index(trial[1]) for trial in fitting])
    initial_stops = [np.median(np.min(z_mm[indices == i], axis=1))+.5
                     for i in range(len(conditions))]

    def residual(parameters):
        stiffness, *stops = parameters
        approach = np.array(stops)[indices, None] - z_mm
        return (stiffness*np.maximum(0, approach)-target).ravel()

    fit = least_squares(residual, (6., *initial_stops),
                        bounds=([.01]+[-5.]*len(conditions), [100.]+[12.]*len(conditions)))
    if not fit.success:
        raise RuntimeError(f"axial interface calibration failed: {fit.message}")
    return 1000*float(fit.x[0]), dict(zip(conditions, map(float, fit.x[1:]/1000)))


def run(exe, work, results, trials, stiffness, stops):
    models = results / "models"
    models.mkdir(exist_ok=True)
    for kind, options, name in (
        ("box", (70, 70, 4, -2), "receiver.step"),
        ("annulus", (10, 3, 8), "ring.step"),
    ):
        part = work / name
        subprocess.run([exe, "make", kind, *map(str, options), str(part)],
                       check=True, capture_output=True)
        shutil.copyfile(part, models / name)

    rows = []
    for condition, stop in stops.items():
        subset = [trial for trial in trials if trial[1] == condition]
        data = np.concatenate([trial[3] for trial in subset])
        name = f"offset-{condition}"
        deck = work / f"{name}.scene"
        # Axial reduced interface. Set the annulus lower face's displacement
        # to the recorded Z. Radii only provide a trimmed witness location.
        with deck.open("w", encoding="ascii", newline="\n") as stream:
            stream.write(
                f"""nc-contact-v2
gravity 0 0 0 0 2 1
body "receiver.step" 0 0 0 {stop:.17g} 0 0 1 0 0 0 0 -1 0 0 0 0
body "ring.step" 0 0 0 .0015 0 0 1 0 0 0 0 -1 0 0 0 0
probe 0 1 world 0 0 0 .009 0 0 0 0 1 {stiffness:.17g} 1 0 0 .001 .02
drives 1
1 table {len(data)}
""")
            for index, z in enumerate(data):
                stream.write(f"{index} 0 0 {z:.17g}\n")
        if condition == "x+3":
            shutil.copyfile(deck, models / deck.name)
        predicted = {}
        for label, mesh in (("brep", 0), ("mesh25um", 25e-6)):
            prefix = work / f"{name}-{label}"
            result = subprocess.run([exe, str(deck), str(mesh), "1", str(len(data)-1), "1", str(prefix)],
                                    check=False, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"{name} {label}: {result.stderr}")
            trace = np.genfromtxt(f"{prefix}.csv", names=True, delimiter=",")
            if not trace["p0_hit"].all():
                raise ValueError(f"axial probe did not hit both trims for {name} {label}")
            predicted[label] = trace["p0_normal_N"].reshape(len(subset), len(WINDOW))
        for i, (slot, _, identifier, z, force, target) in enumerate(subset):
            for j, sample in enumerate(WINDOW):
                rows.append((slot, condition, identifier, int(sample), z[j], force[j],
                             predicted["brep"][i, j], predicted["mesh25um"][i, j], target))
    with (results / "paired.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("slot", "offset", "trial", "sample_from_command_turnaround", "relative_z_m",
                         "measured_axial_N", "brep_axial_N", "mesh_axial_N", "command_target_z_m"))
        writer.writerows(rows)
    return rows


def summarize(rows):
    summary = {}
    for group, slots in (("calibration", CALIBRATION_SLOTS), ("holdout", HOLDOUT_SLOTS)):
        grouped = defaultdict(list)
        for slot, offset, trial, *measurements in rows:
            if slot in slots:
                grouped[slot, offset, trial].append(measurements)
        values = np.stack(list(grouped.values()))
        experiment, brep, mesh = values[:, :, 2:5].transpose(2, 0, 1)
        def rms(values, axis=None):
            return np.sqrt(np.mean(values**2, axis=axis))
        trial_errors = rms(brep-experiment, axis=1)
        peak_prediction = brep.max(axis=1)
        peak_measured = experiment.max(axis=1)
        peak_error = peak_prediction-peak_measured
        observed_spread = np.sqrt(np.mean(experiment**2))
        by_slot_offset = defaultdict(list)
        for i, (slot, offset, _) in enumerate(grouped):
            by_slot_offset[slot, offset].append(i)
        residual_repeat = np.concatenate([
            experiment[indexes]-np.median(experiment[indexes], axis=0)
            for indexes in by_slot_offset.values()
        ])
        summary[group] = {
            "slots": slots,
            "count": len(values),
            "rmse_N": float(rms(brep-experiment)),
            "median_trial_rmse_N": float(np.median(trial_errors)),
            "zero_force_baseline_rmse_N": float(observed_spread),
            "replicate_median_rmse_N": float(rms(residual_repeat)),
            "peak_MAE_N": float(np.mean(np.abs(peak_error))),
            "peak_bias_N": float(np.mean(peak_error)),
            "late_window_mean_force_MAE_N": float(np.mean(np.abs(
                np.mean(brep[:, -8:]-experiment[:, -8:], axis=1)))),
            "median_measured_peak_N": float(np.median(peak_measured)),
            "median_simulated_peak_N": float(np.median(peak_prediction)),
            "max_backend_delta_N": float(np.max(np.abs(brep-mesh))),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    trials = list(read_trials(args.archive))
    conditions = sorted({trial[1] for trial in trials})
    stiffness, stops = calibrate(trials, conditions)
    rows = run(str(args.exe.resolve()), args.work, args.results, trials, stiffness, stops)
    metrics = summarize(rows)
    (args.results / "summary.json").write_text(json.dumps({
        "source": SOURCE, "archive_md5": CHECKSUM, "axial_stiffness_N_m": stiffness,
        "identified_stop_relative_to_command_target_m": stops,
        "sample_window_inclusive": [int(WINDOW[0]), int(WINDOW[-1])],
        "reference": "tool-frame axial force, precontact median removed",
        "clock": "source sample index; no timestamp or sample-rate metadata",
        "metrics": metrics,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
