"""Release the crank speed using the rig's flywheel inertia and encoder coordinate."""

import argparse
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from clearance import REFERENCE, restitution_scale, simulate
from clearance_bearings import SHAFT_INERTIA, body, release_bearings
from clearance_impact_report import trough_width
from clearance_revision import measured
from clearance_spatial import CASES as SPATIAL, score, spatial_deck


CASES = {"constant-torque": None, "slow": .30, "fast": .03}


def needle_torque(trace):
    dx = trace["b6_x_m"]-trace["b5_x_m"]
    dy = trace["b6_y_m"]-trace["b5_y_m"]
    ex = trace["b6_x_m"]-trace["b8_x_m"]
    ey = trace["b6_y_m"]-trace["b8_y_m"]
    arm = np.hypot(ex,ey)
    if np.max(np.abs(arm-.05))>1e-12:
        raise ValueError("the wrist arm no longer matches the crank radius")
    normal = trace["p3_normal_N"]
    numerator = normal*(ey*dx-ex*dy)
    distance = np.hypot(dx,dy)
    return np.divide(numerator,distance,out=np.zeros_like(distance),where=distance>0)


def reaction_bias(source, rpm):
    fields = ("time_s","b5_x_m","b5_y_m","b6_x_m","b6_y_m","b8_x_m","b8_y_m","p3_normal_N")
    with source.open(encoding="ascii") as stream:
        columns = stream.readline().strip().split(",")
        samples = np.loadtxt(stream,delimiter=",",usecols=[columns.index(f) for f in fields])
    cycle = 60/rpm
    phase = samples[:,0]/cycle*360
    if abs(phase[-1]-2160)>1e-6:
        raise ValueError("feed-forward control requires the six-revolution fixed-speed trace")
    selected = phase>=1440-1e-7
    trace = dict(zip(fields,samples[selected].T))
    torque = needle_torque(trace)
    time = trace["time_s"]
    return float(-trapezoid(torque,time)/(time[-1]-time[0]))


def install_drive(deck, omega, tau, bias):
    lines = deck.read_text(encoding="ascii").splitlines()
    header = lines[1].split()
    old_count = int(header[-3])
    header[-3] = str(old_count+1)
    lines[1] = " ".join(header)
    pin = next(i for i,line in enumerate(lines) if line.startswith('body "needle-journal.step"'))
    original = lines[pin].split()
    translation_ids = tuple(map(int,original[18::4]))
    lines[pin] = body("needle-journal.step",.05,old_count,translation_ids)
    drive = lines.index("drives 1")
    lines[drive:drive+2] = ["drives 0"]
    damping = SHAFT_INERTIA/tau if tau is not None else 0.
    lines.insert(2+old_count,
                 f"coordinate {SHAFT_INERTIA:.17g} 0 {omega:.17g} 0 {damping:.17g} 0 {bias+damping*omega:.17g}")
    deck.write_text("\n".join(lines)+"\n",encoding="ascii")
    return old_count,damping


def drive_metrics(trace, coordinate, omega, damping, bias, stats):
    velocity = trace[f"v{coordinate}"]
    acceleration = trace[f"a{coordinate}"]
    reaction = needle_torque(trace)
    motor = bias + damping*(omega-velocity)
    torque = reaction+motor
    # The standard HHT alpha is -0.1 and these consecutive outputs are at stride 1.
    residual = SHAFT_INERTIA*acceleration[1:]-(.9*torque[1:]+.1*torque[:-1])
    error = float(np.max(np.abs(residual)))
    if stats["subdivisions"]==0 and error>3e-5:
        raise ValueError(f"angular balance residual exceeds tolerance: {error:g} N m")
    rate = velocity*60/(2*np.pi)
    elapsed = float(trace["time_s"][-1]-trace["time_s"][0])
    integral = float(trace[f"q{coordinate}"][-1]-trace[f"q{coordinate}"][0])
    return {
        "encoder_speed_rpm": [float(np.min(rate)),float(np.max(rate))],
        "mean_encoder_rpm": integral/elapsed*60/(2*np.pi),
        "max_speed_deviation_fraction": float(np.max(np.abs(velocity/omega-1))),
        "max_acceleration_rad_s2": float(np.max(np.abs(acceleration))),
        "max_motor_torque_N_m": float(np.max(np.abs(motor))),
        "max_motor_power_W": float(np.max(np.abs(motor*velocity))),
        "max_needle_torque_N_m": float(np.max(np.abs(reaction))),
        "hht_angular_balance_error_N_m": error if stats["subdivisions"]==0 else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe",type=Path)
    parser.add_argument("work",type=Path)
    parser.add_argument("reference",type=Path)
    parser.add_argument("fixed",type=Path,help="fixed-speed trace from clearance_bearings.py")
    parser.add_argument("--cases",nargs="+",choices=tuple(CASES),default=tuple(CASES))
    parser.add_argument("--step-us",type=float,default=10)
    parser.add_argument("--rpm",type=int,default=200)
    parser.add_argument("--clearance",type=float,default=.25)
    parser.add_argument("--jobs",type=int,default=3)
    parser.add_argument("--reuse",action="store_true")
    args = parser.parse_args()
    if args.step_us<=0 or args.jobs<=0:
        parser.error("positive step and worker count are required")
    exe = str(args.exe.resolve())
    condition = f"{args.rpm}rpm-{args.clearance:.2f}mm"
    reference = measured(args.reference/f"measured-{condition}.csv")
    root = args.work/f"{condition}-{args.step_us:g}us"
    omega = args.rpm*2*math.pi/60
    damping_scale = restitution_scale()
    bias = reaction_bias(args.fixed,args.rpm)

    def run(case):
        directory = root/case
        directory.mkdir(parents=True,exist_ok=True)
        deck, _ = spatial_deck(exe,directory,args.rpm,args.clearance,damping_scale,SPATIAL["mount-26hz"])
        release_bearings(exe,deck,args.rpm,True,1.,1.)
        coordinate,damping = install_drive(deck,omega,CASES[case],bias)
        try:
            phase,trace,stats = simulate(exe,deck,directory,"drive",0.,args.step_us*1e-6,args.rpm,
                                         7,args.reuse,encoder_window=(coordinate,4))
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"{case}: {error.stderr}") from error
        result = score(phase,trace,reference,3)
        result.update(drive_metrics(trace,coordinate,omega,damping,bias,stats))
        result.update({
            "case":case, "step_us":args.step_us, "rpm":args.rpm, "radial_clearance_mm":args.clearance,
            "encoder_coordinate":coordinate, "encoder_turns":[4,6], "output_filter":None, "phase_shift_deg":0,
            "response_time_s":CASES[case],"bias_N_m":bias,"controller_N_m_s_rad":damping,
            "trough_widths_deg":[trough_width(phase-start,trace["s1_a_m_s2"]) for start in (0,360)],
            "stats":stats,
        })
        (directory/"summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(case, result["encoder_speed_rpm"],
              [(round(r["window_min_phase_deg"],3),round(r["window_min_m_s2"],3),
                round(r["min_load_N"],3)) for r in result["revolutions"]],
              "quiet",round(result["quiet_rmse_m_s2"],3),flush=True)
        return result

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        cases = list(pool.map(run,args.cases))
    report = {"source":REFERENCE,"flywheel_inertia_kg_m2":SHAFT_INERTIA,"cases":cases}
    (root/"summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
