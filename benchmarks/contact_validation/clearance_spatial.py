"""Resolve slider pitch and finite pin/bore engagement in the shared HHT runner."""

import argparse
import csv
import gzip
import json
import math
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from clearance import restitution_scale, simulate
from clearance_flexibility import CONFIGURATIONS, flex_deck
from clearance_revision import measured


SLIDER_INERTIA = .000772
BORE_HALF_LENGTH = .006


@dataclass(frozen=True)
class Spatial:
    pitch_frequency: float = 0
    pitch_damping: float = .20
    pin_height: float = .020
    sensor_height: float = .015
    neutral_tilt: float = 0
    sleeve_rotation: bool = True
    sections: int = 3
    mount_frequency: float = 0
    mount_mass: float = .03
    mount_damping: float = .05


CASES = {
    "single-center": Spatial(sleeve_rotation=False, sections=1),
    "sleeve-spin": Spatial(sections=1),
    "sectional": Spatial(),
    "pitch-80hz": Spatial(80),
    "pitch-160hz": Spatial(160),
    "pitch-320hz": Spatial(320),
    "tilt-plus": Spatial(160, neutral_tilt=.005),
    "tilt-minus": Spatial(160, neutral_tilt=-.005),
    "long-lever": Spatial(160, pin_height=.035, sensor_height=.035, neutral_tilt=.005),
}
SPATIAL_CASES = tuple(CASES)
CASES.update({
    "mount-26hz": replace(CASES["long-lever"], mount_frequency=26.5),
    "mount-36hz": replace(CASES["long-lever"], mount_frequency=36, mount_damping=.015),
    "mount-70hz": replace(CASES["long-lever"], mount_frequency=70.5, mount_damping=.015),
    "mount-heavy-28hz": replace(CASES["long-lever"], mount_frequency=28, mount_mass=.15),
    "mount-heavy-45hz": replace(CASES["long-lever"], mount_frequency=45, mount_mass=.12,
                                mount_damping=.12),
})


def body_line(line, origin, axis, rest, hinge):
    fields = line.split()
    name, mass = fields[1:3]
    center = " ".join(fields[10:13])
    translations = " ".join(fields[17:])
    return (f"body {name} {mass} {origin} {axis} {rest:.17g} {center} "
            f"{hinge} {origin} {translations}")


def spatial_deck(exe, directory, rpm, clearance, damping, configuration):
    deck = flex_deck(exe, directory, rpm, clearance, damping, CONFIGURATIONS["compliant-guide"])
    lines = deck.read_text(encoding="ascii").splitlines()
    coords = [line for line in lines if line.startswith("coordinate ")]
    bodies = [line for line in lines if line.startswith("body ")]
    pitch = len(coords) if configuration.pitch_frequency else -1
    mount = configuration.mount_frequency > 0
    shift = 0.
    inertia = SLIDER_INERTIA
    if mount:
        # Repartition the baseline carriage mass and its nominal first moment
        # and pitch inertia between the core and the measurement point.
        original_mass = float(coords[1].split()[1])
        core_mass = original_mass-configuration.mount_mass
        shift = -configuration.mount_mass*configuration.sensor_height/core_mass
        inertia -= configuration.mount_mass*configuration.sensor_height**2 + core_mass*shift**2
        if min(core_mass,inertia) <= 0:
            raise ValueError("invalid measurement-mode mass repartition")
        for i in (1,6):
            fields = coords[i].split()
            fields[1] = f"{core_mass:.17g}"
            coords[i] = " ".join(fields)
    if pitch >= 0:
        omega = 2*math.pi*configuration.pitch_frequency
        stiffness = inertia*omega**2
        drag = 2*configuration.pitch_damping*inertia*omega
        coords.append(f"coordinate {inertia:.17g} 0 0 {stiffness:.17g} {drag:.17g} 0 0")
    height = configuration.pin_height-shift
    sensor_arm = configuration.sensor_height-shift
    bodies[0] = body_line(bodies[0], "0 0 0", "0 0 1", 0, 0 if configuration.sleeve_rotation else -1)
    bodies[1] = body_line(bodies[1], "0 0 0", "0 1 0", configuration.neutral_tilt, pitch)
    bodies[3] = body_line(bodies[3], f"0 0 {-height:.17g}", "0 1 0", 0, pitch)

    prototype = next(line.split() for line in lines if line.startswith("probe "))
    stiffness, exponent, rate, friction, slip_speed, limit = map(float, prototype[13:])
    if configuration.sections == 1:
        nodes = ((0., 1.),)
    else:
        station = BORE_HALF_LENGTH*math.sqrt(3/5)
        nodes = ((-station, 5/18), (0., 4/9), (station, 5/18))
    probes = [
        f"probe 0 1 contained 0 0 {z:.17g} 0 0 {z:.17g} -1 0 0 "
        f"{weight*stiffness:.17g} {exponent:.17g} {weight*rate:.17g} "
        f"{friction:.17g} {slip_speed:.17g} {limit:.17g}"
        for z, weight in nodes
    ]
    rest = lines[lines.index("drives 1"):]
    for i, line in enumerate(rest):
        if line.startswith("slide 0 3 "):
            fields = line.split()
            fields[8] = f"{height:.17g}"
            rest[i] = " ".join(fields)
        if line.startswith("bushing 3 1 "):
            fields = line.split()
            fields[5] = f"{height:.17g}"
            rest[i] = " ".join(fields)
    sensor = f"accelerometer 3 0 0 {sensor_arm:.17g} 1 0 0"
    if mount:
        index = len(coords)
        coords.append(f"coordinate {configuration.mount_mass:.17g} .35 0 0 0 0 0")
        z = configuration.sensor_height-configuration.pin_height
        pin_shape = bodies[1].split()[1]
        bodies.append(f"body {pin_shape} 0 0 0 {z:.17g} 0 0 1 0 0 0 0 -1 0 0 0 1 {index} 1 0 0")
        omega = 2*math.pi*configuration.mount_frequency
        stiffness = configuration.mount_mass*omega**2
        drag = 2*configuration.mount_mass*configuration.mount_damping*omega
        rest[rest.index("bushings 2")] = "bushings 3"
        rest.append(f"bushing 3 4 0 0 {sensor_arm:.17g} 0 0 0 0 0 0 {stiffness:.17g} 0 0 {drag:.17g} 0 0")
        sensor = "accelerometer 4 0 0 0 1 0 0"
    output = [lines[0], f"gravity 0 0 0 {len(coords)} {len(bodies)} {len(probes)}",
              *coords, *bodies, *probes, *rest, "accelerometers 2",
              "accelerometer 3 0 0 0 1 0 0", sensor]
    deck.write_text("\n".join(output)+"\n", encoding="ascii")
    return deck, len(probes)


def score(phase, trace, reference, count):
    sensor = trace["s1_a_m_s2"]
    carriage = trace["s0_a_m_s2"]
    gaps = np.column_stack([trace[f"p{i}_gap_m"] for i in range(count)])
    force = np.column_stack([trace[f"p{i}_normal_N"] for i in range(count)])
    included = (phase >= 0) & (phase <= 720)
    angle = phase % 360
    quiet = included & (angle >= 90) & (angle <= 180)
    experimental = np.interp(phase, *reference)
    a = reference[0]
    measured_early = (a >= 44) & (a <= 54)
    if not np.any(measured_early):
        raise ValueError("missing published window")
    measured_peak = int(np.argmin(np.where(measured_early, reference[1], np.inf)))
    target = float(a[measured_peak])
    observed = float(reference[1][measured_peak])
    metrics = {
        "measured_phase_deg": target, "measured_peak_m_s2": observed,
        "raw_rmse_m_s2": float(np.sqrt(np.mean((sensor[included]-experimental[included])**2))),
        "quiet_rmse_m_s2": float(np.sqrt(np.mean((sensor[quiet]-experimental[quiet])**2))),
        "quiet_max_abs_m_s2": float(np.max(np.abs(sensor[quiet]))),
        "revolutions": [],
    }
    for start in (0, 360):
        group = np.flatnonzero((phase >= start+42) & (phase <= start+56))
        impact = int(group[np.argmin(sensor[group])])
        section = (phase >= start+43) & (phase <= start+63)
        local_force = force[section]
        effective = section & (force.sum(axis=1) > 0)
        metrics["revolutions"].append({
            "window_min_phase_deg": float(phase[impact]-start),
            "window_min_m_s2": float(sensor[impact]),
            "minimum_inside_window": bool(group[0] < impact < group[-1]),
            "sensor_at_measured_phase_m_s2": float(np.interp(start+target, phase, sensor)),
            "carriage_at_measured_phase_m_s2": float(np.interp(start+target, phase, carriage)),
            "max_sensor_carriage_difference_m_s2": float(np.max(np.abs(sensor[section]-carriage[section]))),
            "min_load_N": float(np.min(local_force.sum(axis=1))),
            "all_sections_open_fraction": float(np.mean(np.min(gaps[section], axis=1) > 0)),
            "partially_loaded_fraction": float(np.count_nonzero(
                effective & np.any(force <= 0, axis=1))/np.count_nonzero(section)),
            "max_clearance_for_any_section_um": float(np.max(gaps[section])*1e6),
        })
    return metrics


def install_warm_state(deck, trace_path, rpm, turn):
    if turn < 1:
        raise ValueError("warm start requires a positive whole crank revolution")
    expected_time = turn*60/rpm
    with trace_path.open(newline="", encoding="utf-8") as source:
        row = next((row for row in csv.DictReader(source)
                    if abs(float(row["time_s"])-expected_time) < 1e-8), None)
    if row is None:
        raise ValueError("missing warm state at the whole-crank-revolution restart")
    lines = deck.read_text(encoding="ascii").splitlines()
    coordinate = [i for i,line in enumerate(lines) if line.startswith("coordinate ")]
    for j, i in enumerate(coordinate):
        fields = lines[i].split()
        fields[2:4] = row[f"q{j}"], row[f"v{j}"]
        lines[i] = " ".join(fields)
    deck.write_text("\n".join(lines)+"\n",encoding="ascii")


def archive(directory, deck, phase, trace, sections, report):
    directory.mkdir(parents=True,exist_ok=True)
    for source in (deck, deck.parent/"sleeve.step", *deck.parent.glob("pin-*.step")):
        shutil.copyfile(source, directory/source.name)
    fields = ["time_s", "b3_x_m", "b3_angle_rad", "s0_a_m_s2", "s1_a_m_s2"]
    for i in range(sections):
        fields.extend((f"p{i}_gap_m", f"p{i}_normal_N"))
    output = np.column_stack([phase, *(trace[key] for key in fields)])
    with gzip.open(directory/"raw.csv.gz","wt",encoding="ascii",newline="") as stream:
        np.savetxt(stream, output, fmt="%.17g", delimiter=",",
                   header=",".join(["phase_deg", *fields]), comments="")
    (directory/"summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")


def verify_sensor(exe, work):
    target = work/"sensor-check"
    target.mkdir(parents=True, exist_ok=True)
    shape = target/"sphere.step"
    if not shape.exists():
        subprocess.run([exe, "make", "sphere", "10", str(shape)], check=True, capture_output=True)
    scene = target/"sensor.scene"
    scene.write_text("""nc-contact-v2
gravity .1 -.3 0 2 2 1
coordinate 2 0 .7 0 0 0 4
coordinate 1 .4 .2 0 0 0 .3
body "sphere.step" 0 .21 0 0 0 0 1 .25 0 0 0 0 0 0 0 1 1 1 0 0
body "sphere.step" 0 3 0 0 0 0 1 0 0 0 0 -1 0 0 0 0
probe 0 1 world 0 0 0 0 0 0 1 0 0 1000 1 0 0 .001 .001
drives 1
0 orbit .03 -.02 0 0 0 1 .8
accelerometers 1
accelerometer 0 .11 .04 0 1 0 0
""", encoding="ascii")
    prefix = target/"trace"
    subprocess.run([exe, str(scene), "0", ".001", ".15", "1", str(prefix)], check=True, capture_output=True)
    trace = np.genfromtxt(prefix.with_suffix(".csv"), delimiter=",", names=True)
    time = trace["time_s"]
    q = .7*time + time**2
    theta = .25+q
    spin = .7+2*time
    orientation = np.exp(1j*theta)
    rotating_arm = .21*np.exp(1j*q)+(.11+.04j)*orientation
    prescribed = -.8**2*(.03-.02j)*np.exp(.8j*time)
    true_specific_force = .3-(.1-.3j)+prescribed+(2j-spin**2)*rotating_arm
    expected = np.real(true_specific_force*np.conj(orientation))
    error = float(np.max(np.abs(trace["s0_a_m_s2"]-expected)))
    if error > 2e-11:
        raise ValueError(f"accelerometer differs from analytic rigid point by {error:.6g} m/s^2")
    print(f"Rigid-point acceleration maximum error {error:.3g} m/s^2", flush=True)


def verify_section_torque(exe, work):
    target = work/"section-torque-check"
    target.mkdir(parents=True, exist_ok=True)
    for name,parameters in (("sleeve.step", ("annulus","16","12","11.125")),
                            ("pin.step", ("cylinder","10.875","12"))):
        shape = target/name
        if not shape.exists():
            subprocess.run([exe, "make", *parameters, str(shape)], check=True, capture_output=True)
    scene = target/"tilted.scene"
    prefix = target/"trace"
    stiffness = 1e6
    inertia = .001

    def inspect(angle):
        scene.write_text(f"""nc-contact-v2
gravity 0 0 0 1 2 2
coordinate {inertia} {angle} 0 0 0 0 0
body "sleeve.step" 0 0 0 0 0 0 1 0 0 0 0 -1 0 0 0 0
body "pin.step" 0 .000245 0 0 0 1 0 0 0 0 0 0 .000245 0 0 0
probe 0 1 contained 0 0 -.004 0 0 -.004 -1 0 0 {stiffness} 1 0 0 .001 .0005
probe 0 1 contained 0 0 .004 0 0 .004 -1 0 0 {stiffness} 1 0 0 .001 .0005
drives 0
""", encoding="ascii")
        subprocess.run([exe,str(scene),"0",".00001",".00001","1",str(prefix)],
                       check=True,capture_output=True)
        with prefix.with_suffix(".csv").open(newline="",encoding="ascii") as stream:
            row = next(csv.DictReader(stream))
        gaps = [float(row[f"p{i}_gap_m"]) for i in range(2)]
        if not gaps[0] > 0 > gaps[1]:
            raise ValueError(f"expected one active axial station, got {gaps}")
        return .5*stiffness*gaps[1]**2, inertia*float(row["a0"])

    angle = .005
    h = 1e-5
    low, _ = inspect(angle-h)
    high, _ = inspect(angle+h)
    _, torque = inspect(angle)
    reference = -(high-low)/(2*h)
    if not math.isclose(torque,reference,rel_tol=2e-6,abs_tol=1e-9):
        raise ValueError(f"sectional contact virtual work {torque:g} versus {reference:g} N m")
    print(f"Tilted sectional contact torque {torque:.6g} N m verified by energy",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=SPATIAL_CASES)
    parser.add_argument("--rpm", type=int, default=200)
    parser.add_argument("--clearance", type=float, default=.25)
    parser.add_argument("--step-us", type=float, default=20)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--verify-sensor", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--warm-start", type=Path, help="raw trace of the same case at the same rpm and clearance")
    parser.add_argument("--warm-turn", type=int, default=4)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    if args.cycles < (2 if args.warm_start else 3) or args.jobs < 1 or args.step_us <= 0:
        parser.error("positive step and workers and either three cycles or two from a warm start are required")
    if args.warm_start and len(args.cases)!=1:
        parser.error("a warm trace belongs to one case")
    args.work.mkdir(parents=True, exist_ok=True)
    exe = str(args.exe.resolve())
    if args.verify_sensor:
        verify_sensor(exe, args.work)
        verify_section_torque(exe, args.work)
    reference = measured(args.reference/f"measured-{args.rpm}rpm-{args.clearance:.2f}mm.csv")
    damping = restitution_scale()
    tag = f"{args.rpm}rpm-{args.clearance:.2f}mm-{args.step_us:g}us-{args.cycles}cycles"

    def run(label):
        work = args.work/tag/label
        work.mkdir(parents=True, exist_ok=True)
        configuration = CASES[label]
        deck, sections = spatial_deck(exe, work, args.rpm, args.clearance, damping, configuration)
        if args.warm_start:
            install_warm_state(deck, args.warm_start, args.rpm, args.warm_turn)
        phase, trace, stats = simulate(exe, deck, work, "spatial", 0., args.step_us*1e-6, args.rpm,
                                       args.cycles, args.reuse)
        report = {"case": label, "parameters": asdict(configuration), "rpm": args.rpm,
                  "radial_clearance_mm": args.clearance, "cycles": args.cycles,
                  "warm_phase_turn": args.warm_turn if args.warm_start else None,
                  "phase_shift_deg": 0, "output_filter": None, "integration": stats}
        report.update(score(phase, trace, reference, sections))
        (work/"summary.json").write_text(json.dumps(report,indent=2)+"\n", encoding="utf-8")
        if args.archive:
            archive(args.archive/tag/label, deck, phase, trace, sections, report)
        print(label, report["quiet_rmse_m_s2"], report["revolutions"], flush=True)
        return report

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        reports = list(pool.map(run, args.cases))
    (args.work/tag/"summary.json").write_text(json.dumps(reports,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
