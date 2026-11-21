"""Compare the Flores slider-crank accelerometer traces with STEP/HHT runs."""

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import fitz
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


REFERENCE = "https://doi.org/10.1007/s11071-010-9899-8"
# Publisher PDF, zero-based pages and vector drawing sequence numbers.
# Left-to-right, top-to-bottom order within each figure.
PANELS = (
    (200, .10, 7, 584, 636),
    (200, .25, 7, 391, 443),
    (200, .50, 7, 196, 250),
    (200, 1.00, 7, 3, 55),
    (100, .25, 9, 634, 674),
    (150, .25, 9, 438, 492),
    (250, .25, 9, 3, 96),
)


def published_curve(page, frame_id, curve_id, x_max, y_min, y_max):
    drawings = {item["seqno"]: item for item in page.get_drawings()}
    frame = drawings[frame_id]["rect"]
    lines = drawings[curve_id]["items"]
    if len(lines) < 300:
        raise ValueError("unexpected publisher PDF plot paths")
    points = []
    for item in lines:
        if item[0] == "l":
            points.extend(item[1:])
        elif item[0] == "c":
            start, middle_a, middle_b, end = item[1:]
            for t in np.linspace(0, 1, 9):
                points.append(start*(1-t)**3 + middle_a*(3*t*(1-t)**2)
                              + middle_b*(3*t*t*(1-t)) + end*t**3)
        elif item[0] == "qu":
            points.extend(item[1])
        else:
            raise ValueError(f"unexpected plot segment {item[0]}")
    x = x_max * (np.array([p.x for p in points]) - frame.x0) / frame.width
    y = y_max - (y_max - y_min) * (np.array([p.y for p in points]) - frame.y0) / frame.height
    order = np.argsort(x, kind="stable")
    return x[order], y[order]


def digitize(path, directory):
    with fitz.open(path) as pdf:
        reference = {
            (rpm, clearance): published_curve(pdf[page], frame, curve, 720, -150, 200)
            for rpm, clearance, page, frame, curve in PANELS
        }
        travel = published_curve(pdf[6], 353, 450, 360, .20, .45)
    for (rpm, clearance), (phase, acceleration) in reference.items():
        np.savetxt(directory / f"measured-{rpm}rpm-{clearance:.2f}mm.csv",
                   np.column_stack((phase, acceleration)), delimiter=",",
                   header="angle_deg,acceleration_m_s2", comments="")
    np.savetxt(directory / "measured-position-fig5b.csv", np.column_stack(travel), delimiter=",",
               header="angle_deg,position_m", comments="")
    return reference, travel


def restitution_scale():
    # Dimensionless forward impact, with the same unilateral force as nc_contact.
    def residual(damping):
        def rhs(_, state):
            depth, speed = state
            depth = max(depth, 0)
            return speed, -max(0, depth**1.5 + damping*depth**.25*speed)

        def exit_contact(_, state):
            return state[0]

        exit_contact.direction = -1
        exit_contact.terminal = True
        result = solve_ivp(rhs, (0, 20), (0, 1), events=exit_contact,
                           rtol=1e-9, atol=1e-10, max_step=.02)
        return -result.y[1, -1] - .46

    return brentq(residual, 0, 3)


@dataclass(frozen=True)
class RigDissipation:
    contact_exponent: float
    normal_rate_multiplier: float
    guide_damping: float
    plate_friction: float
    stiffness_scale: float = 1.


def write_deck(exe, directory, rpm, clearance, damping_scale, dissipation=None):
    radius = 11.125
    pin = radius - clearance
    label = f"{rpm}rpm-{clearance:.2f}mm"
    journal_file = directory / f"pin-{clearance:.2f}mm.step"
    sleeve_file = directory / "sleeve.step"
    for kind, args, path in (
        ("annulus", (16, 12, radius), sleeve_file),
        ("cylinder", (pin, 12), journal_file),
    ):
        if not path.exists():
            subprocess.run([exe, "make", kind, *map(str, args), str(path)],
                           check=True, capture_output=True)
    omega = rpm * 2 * math.pi / 60
    stiffness = 4/3 * (207e9/(2*(1-.3**2))) * math.sqrt(1e-3 * radius*pin/clearance)
    # The rig lies horizontally. Use the reported inertia as centroidal and
    # place the rod center of mass halfway between its pivots.
    offset = .15
    rod_mass = 1.13
    slider_mass = 1.013
    inertia = .0153 + rod_mass*offset**2
    power = 1.5
    guide = 0
    plate = ""
    slider_fields = '0 0 0 0 0 0 1 0 0 0 0 -1 0 0 0 1 1 1 0 0'
    rod_fields = f'{rod_mass} .3 0 0 0 0 1 0 {offset-.3} 0 0 0 0 0 0 0'
    mount = (slider_fields, rod_fields)
    orbit_body = 1
    if dissipation is not None:
        mount = (rod_fields, slider_fields)
        orbit_body = 0
        power = dissipation.contact_exponent
        guide = dissipation.guide_damping
        # Express the curvature-dependent coefficient at a common reference
        # compression so the scale retains its meaning when power changes.
        stiffness *= dissipation.stiffness_scale*(.3e-6)**(1.5-power)
        damping_scale *= dissipation.normal_rate_multiplier
        if dissipation.plate_friction:
            plate = f'resistances 1\nslide 0 1 0 0 0 0 0 0 0 0 1 {dissipation.plate_friction} .001\n'
    damping = damping_scale*math.sqrt(stiffness*slider_mass)
    path = directory / f"{label}.scene"
    path.write_text(
        f"""nc-contact-v2
gravity 0 0 0 2 2 1
coordinate {inertia} 0 {-omega/6} 0 0 0 0
coordinate {slider_mass} .35 0 0 {guide} 0 0
body "{sleeve_file.name}" {mount[0]}
body "{journal_file.name}" {mount[1]}
probe 0 1 contained-radial 0 0 0 0 0 0 -1 0 0 {stiffness} {power} {damping} .01 .001 .0005
drives 1
{orbit_body} orbit .05 0 0 0 0 1 {omega}
{plate}""", encoding="ascii")
    return path


def simulate(exe, deck, directory, tag, deflection, requested_step, rpm, cycles, reuse, stride=1):
    cycle = 60/rpm
    per_cycle = round(cycle/requested_step)
    dt = cycle/per_cycle
    prefix = directory / f"{deck.stem}-{tag}"
    trace_path = Path(f"{prefix}.csv")
    stats_path = Path(f"{prefix}.stats.csv")
    if not (reuse and trace_path.is_file() and stats_path.is_file()):
        subprocess.run([exe, str(deck), str(deflection), str(dt), str(cycles*cycle), str(stride), str(prefix)],
                       check=True, capture_output=True, text=True)
    with trace_path.open() as stream:
        fields = stream.readline().strip().split(",")
        trace = np.loadtxt(stream, dtype=np.dtype([(field, "f8") for field in fields]), delimiter=",")
    with stats_path.open(newline="") as stream:
        stats = {key: float(value) for key, value in next(csv.DictReader(stream)).items()}
    selected = trace["time_s"] >= (cycles-2)*cycle - dt/2
    phase = (trace["time_s"][selected]/cycle - cycles + 2)*360
    return phase, trace[selected], stats | {"step_s": dt}


def ideal_slider_acceleration(phase, rpm):
    angle = np.deg2rad(phase)
    speed = rpm*2*np.pi/60
    radius, rod = .05, .30
    width = np.sqrt(rod**2-(radius*np.sin(angle))**2)
    return -speed**2 * (radius*np.cos(angle) + radius**2*np.cos(2*angle)/width
                         + (radius*np.sin(angle))**2*(radius*np.cos(angle))**2/width**3)


def compare(phase, trace, experiment, travel, clearance, rpm):
    x, acceleration = experiment
    keep = (x >= 0) & (x <= 720)
    x, acceleration = x[keep], acceleration[keep]
    predicted = np.interp(x, phase, trace["a1"])
    # Preserve plotted extrema and use a uniform phase grid for waveform RMSE.
    uniform = np.linspace(0, 720, 1441)
    measured_uniform = np.interp(uniform, x, acceleration)
    predicted_uniform = np.interp(uniform, phase, trace["a1"])
    ideal_uniform = ideal_slider_acceleration(uniform, rpm)
    xp, yp = travel
    centers = np.column_stack([trace[f"b1_{coordinate}_m"]-trace[f"b0_{coordinate}_m"]
                               for coordinate in "xyz"])
    analytical_gap = .001*clearance - np.linalg.norm(centers, axis=1)
    first_cycle = x < 180
    first_cycle_model = (phase >= 0) & (phase < 180)
    negative_phase = phase[first_cycle_model]
    negative_acceleration = trace["a1"][first_cycle_model]
    result = {
        "measured_positive_peak_m_s2": float(max(acceleration)),
        "simulated_positive_peak_m_s2": float(max(trace["a1"])),
        "measured_negative_peak_m_s2": float(min(acceleration)),
        "simulated_negative_peak_m_s2": float(min(trace["a1"])),
        "measured_first_negative_peak_angle_deg": float(x[first_cycle][np.argmin(acceleration[first_cycle])]),
        "simulated_first_negative_peak_angle_deg": float(negative_phase[np.argmin(negative_acceleration)]),
        "simulated_peak_at_measured_vertices_m_s2": float(max(predicted)),
        "uniform_phase_acceleration_rmse_m_s2": float(np.sqrt(np.mean((measured_uniform-predicted_uniform)**2))),
        "ideal_joint_acceleration_rmse_m_s2": float(np.sqrt(np.mean((measured_uniform-ideal_uniform)**2))),
        "minimum_gap_um": float(1e6*min(trace["p0_gap_m"])),
        "max_geometric_gap_error_um": float(1e6*max(abs(trace["p0_gap_m"]-analytical_gap))),
    }
    curves = np.column_stack((x, acceleration, predicted, ideal_slider_acceleration(x, rpm)))
    positions = np.column_stack((xp, yp, np.interp(xp, phase, trace["q1"])))
    return result, curves, positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe", type=Path)
    parser.add_argument("source_pdf", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--extended", action="store_true", help="also integrate the representative case for 30 revolutions")
    parser.add_argument("--reuse-traces", action="store_true",
                        help="regenerate results using traces from an unchanged prior run")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    reference, travel = digitize(args.source_pdf, args.results)
    scale = restitution_scale()
    models = args.results / "models"
    models.mkdir(exist_ok=True)
    jobs = []
    configurations = (("brep", 0, 10e-6), ("mesh25um", 25e-6, 10e-6),
                      ("mesh5um", 5e-6, 10e-6), ("brep-half-step", 0, 5e-6))
    for rpm, clearance in reference:
        if args.pilot and (rpm, clearance) != (200, .25):
            continue
        deck = write_deck(str(args.exe.resolve()), args.work, rpm, clearance, scale)
        shutil.copyfile(deck, models / deck.name)
        for label, mesh, step in configurations:
            if label == "brep-half-step" and (rpm, clearance) != (200, .25):
                continue
            jobs.append((rpm, clearance, deck, label, mesh, step))
    shape_names = {"sleeve.step"} | {f"pin-{clearance:.2f}mm.step" for _, clearance, *_ in jobs}
    for name in sorted(shape_names):
        shutil.copyfile(args.work / name, models / name)

    def one(job):
        rpm, clearance, deck, label, mesh, step = job
        try:
            phase, trace, stats = simulate(str(args.exe.resolve()), deck, args.work, label, mesh, step, rpm, 6,
                                           args.reuse_traces)
        except subprocess.CalledProcessError as error:
            print(deck.stem, label, error.stderr, flush=True)
            return {"rpm": rpm, "clearance_mm": clearance, "backend": label,
                    "deflection_m": mesh, "error": error.stderr.strip()}
        metrics, curves, positions = compare(phase, trace, reference[(rpm, clearance)], travel, clearance, rpm)
        filename = f"{deck.stem}-{label}"
        if (rpm, clearance) == (200, .25):
            metrics["position_rmse_mm"] = float(1000*np.sqrt(np.mean((positions[:, 1]-positions[:, 2])**2)))
            np.savetxt(args.results / f"position-{filename}.csv", positions, delimiter=",",
                       header="angle_deg,measured_position_m,simulated_position_m", comments="")
            # Keep a display trace for the last revolution. Scoring uses all
            # original integration steps in the final two revolutions.
            visual_stride = max(1, len(trace)//480)
            display = (phase >= 360) & (phase < 720)
            keys = ("q0", "q1", "a1", "b1_x_m", "b1_y_m", "p0_gap_m", "p0_normal_N")
            np.savetxt(args.results / f"display-{filename}.csv",
                       np.column_stack([phase[display]-360, *[trace[key][display] for key in keys]])[::visual_stride],
                       delimiter=",", header="angle_deg,"+",".join(keys), comments="")
        np.savetxt(args.results / f"paired-{filename}.csv", curves, delimiter=",",
                   header="angle_deg,measured_acceleration_m_s2,simulated_acceleration_m_s2,ideal_joint_m_s2",
                   comments="")
        print(filename, metrics, flush=True)
        return {"rpm": rpm, "clearance_mm": clearance, "backend": label,
                "deflection_m": mesh} | metrics | stats

    # Process cases serially so measured integration times do not include
    # competing solves and tessellations from this experiment.
    rows = [one(job) for job in jobs]
    summary = {
        "source": REFERENCE, "source_panels": PANELS,
        "source_pdf_sha256": hashlib.sha256(args.source_pdf.read_bytes()).hexdigest(),
        "restitution_scale": scale, "runs": rows,
    }
    if args.extended:
        deck = args.work / "200rpm-0.25mm.scene"
        phase, trace, stats = simulate(str(args.exe.resolve()), deck, args.work, "brep-30rev", 0, 10e-6,
                                       200, 30, args.reuse_traces, stride=4)
        metrics, _, _ = compare(phase, trace, reference[(200, .25)], travel, .25, 200)
        summary["extended_run"] = {"duration_s": 9.0, "revolutions": 30, "stride": 4} | metrics | stats
    (args.results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
