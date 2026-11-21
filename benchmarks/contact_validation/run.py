"""Calibrate material parameters and compare the shared STEP contact runner to GAUGE."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from gauge import (BALL_TRIALS, CALIBRATION, G, MATERIALS, SLIDE_TRIALS, UNPUBLISHED_SLIDES,
                   bounce_case, model_inputs, slider_case)


DT = .0002
BOUNCE_DT = .0000125
MESH = .000025
BALL_K = 2e7
SLIDE_K = 1e7
EXPONENT = 1.5
SLIP = .0005
ZERO = np.zeros(3)


def values(*args):
    return " ".join(str(x) for x in args)


def coordinate(mass, q, v):
    return values("coordinate", mass, q, v, 0, 0, 0, 0)


def body(file, mass, origin, axis=(0,0,1), angle=0, directions=()):
    return values("body", json.dumps(file.as_posix()), mass, *origin, *axis, angle,
                  0,0,0, -1,0,0,0, len(directions),
                  *(c for i, vector in enumerate(directions) for c in (i,*vector)))


def contact(k, m, zeta, mu):
    return values("probe", 0,1,"body-a", *ZERO, *ZERO, 0,0,1, k,EXPONENT,
                  2*zeta*np.sqrt(k*m),mu,SLIP,.015)


def scene(kind, case, geometry, inputs, parameter):
    if kind=="bounce":
        m = inputs["ball_mass"]
        q = case["initial_height"]
        xy = case["positions"][0,:2]
        q_lines = [coordinate(m,q,case["initial_velocity"])]
        a = body(geometry/"floor.step",0,(0,0,inputs["floor_top"]))
        b = body(geometry/"ball.step",m,(*xy,0),directions=((0,0,1),))
        probe = contact(BALL_K,m,parameter,0.)
    else:
        item = inputs["slider_materials"][case["material"]]
        m = item["mass"]
        x,y,_ = case["initial_local"]
        normal = case["matrix"][:,2]
        depth = (m*G*normal[2]/SLIDE_K)**(1/EXPONENT)
        height = (inputs["board"][2]+inputs["cube"][2])/2-depth
        q_lines = [coordinate(m,x,case["initial_speed"]),coordinate(m,y,0.),coordinate(m,height,0.)]
        a = body(geometry/"board.step",0,case["board_origin"],case["axis"],case["angle"])
        b = body(geometry/"cube.step",m,case["board_origin"],case["axis"],case["angle"],
                 tuple(case["matrix"].T))
        probe = contact(SLIDE_K,m,.18,parameter)
    return "\n".join(("nc-contact-v1", values("gravity",0,0,-G,len(q_lines),2,1),*q_lines,a,b,probe))+"\n"


def read_csv(path):
    return np.genfromtxt(path,delimiter=",",names=True)


def run(exe, scene_file, prefix, duration, dt=DT, mesh=0., stride=1):
    completed = subprocess.run([str(exe),str(scene_file),str(mesh),str(dt),str(duration),
                                str(stride),str(prefix)],capture_output=True,text=True)
    if completed.returncode:
        raise RuntimeError(f"{scene_file.name} {mesh}: {completed.stderr.strip()}")
    trace = read_csv(prefix.with_suffix(".csv"))
    work = read_csv(prefix.with_suffix(".stats.csv"))
    if float(work["accepted"])<float(work["steps"]) or not np.isfinite(trace["q0"]).all():
        raise RuntimeError(f"incomplete trajectory {prefix}")
    return trace,{name:float(work[name]) for name in work.dtype.names}


def prediction(kind, case, trace, inputs):
    nominal = (trace["b1_z_m"]-inputs["floor_top"]-inputs["radius"]
               if kind=="bounce" else trace["q0"])
    return np.interp(case["time"],trace["time_s"],nominal)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def fit_bounce(exe, cases, models, inputs, dt):
    cals = [case for case in cases if case["id"] in CALIBRATION]
    if len(cals)<3:
        raise ValueError("fewer than three resolved calibration drops")
    grid = []
    # These bounds bracket mild through strong damping without using held-out observations.
    with TemporaryDirectory(prefix="nc-contact-cal-") as folder:
        scratch = Path(folder)
        link = models.resolve()
        config = scratch/"cal.scene"
        for zeta in np.linspace(.055,.185,27):
            errors = []
            for case in cals:
                config.write_text(scene("bounce",case,link,inputs,zeta),encoding="utf-8")
                trace,_ = run(exe,config,scratch/"trace",case["time"][-1],dt,stride=max(1,round(DT/dt)))
                y = prediction("bounce",case,trace,inputs)
                errors.append(rms(y-case["observed"]))
            grid.append({"zeta":float(zeta),"mean_rmse_m":float(np.mean(errors))})
    return min(grid,key=lambda x:x["mean_rmse_m"])["zeta"],grid


def fit_slider(cases):
    mu = {}
    for material in MATERIALS:
        calibration = [case for case in cases if case["material"]==material and case["id"] in CALIBRATION]
        if len(calibration)<3:
            raise ValueError(f"fewer than three resolved calibration slides for {material}")
        numerator = denominator = 0.
        for case in calibration:
            t = case["time"][2:]
            lever = (G*case["matrix"][2,2]/2)*t*t
            free = (case["observed"][0]+case["initial_speed"]*t
                    -.5*G*case["matrix"][2,0]*t*t)
            numerator += float(lever @ (free-case["observed"][2:]))
            denominator += float(lever @ lever)
        mu[material] = numerator/denominator
        if not 0<mu[material]<1:
            raise ValueError(f"unphysical calibration for {material}")
    return mu


def rebound_events(t, y):
    increasing = np.diff(y)>0
    troughs = np.flatnonzero(~increasing[:-1]&increasing[1:])+1
    if not len(troughs):
        return []
    peaks = np.flatnonzero(increasing[:-1]&~increasing[1:])+1
    return [(float(t[idx]),float(y[idx])) for idx in peaks[peaks>troughs[0]][:2]]


def scores(kind, case, y, trace):
    first = case.get("score_begin",1)
    error = rms(y[first:]-case["observed"][first:])
    result = {"rmse_m": error, "normalized_rmse": error/case["scale"],
              "frames": len(y)-first, "end_error_m": float(y[-1]-case["observed"][-1]),
              "max_depth_m": -float(np.nanmin(np.minimum(trace["p0_gap_m"],0.))),
              "peak_predicted_force_N": float(np.max(trace["p0_normal_N"]))}
    if kind=="bounce":
        t = case["time"]
        observed, computed = rebound_events(t,case["observed"]),rebound_events(t,y)
        result["observed_apices_s_m"] = observed
        result["predicted_apices_s_m"] = computed
        if observed and computed:
            result["first_apex_time_error_s"] = computed[0][0]-observed[0][0]
            result["first_apex_height_error_m"] = computed[0][1]-observed[0][1]
    return result


def summarize(records):
    summary = {}
    for family in ("bounce", *MATERIALS):
        rows = [r for r in records if r["split"]=="held-out" and r["family"]==family]
        if not rows:
            continue
        result = {"trials":len(rows)}
        for mode in ("brep","mesh"):
            trials = [row[mode] for row in rows]
            result[mode] = {}
            for field in ("rmse_m","normalized_rmse","first_apex_time_error_s","first_apex_height_error_m"):
                points = [abs(item[field]) for item in trials if field in item]
                if points:
                    result[mode][field] = {"median":float(np.median(points)), "p90":float(np.quantile(points,.90)),
                                           "count":len(points)}
        summary[family]=result
    return summary


def compare(exe, source, root, frozen_calibration=None, bounce_dt=BOUNCE_DT):
    models = root/"models"
    inputs = model_inputs(source)
    for key in ("scenes","runs","curves"):
        (root/key).mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source/"sources.json",root/"sources.json")
    ball_cases, slide_cases, omitted = [], [], []
    for kind, material_list, trials in (("bounce",("bounce",),BALL_TRIALS),
                                        ("slider",MATERIALS,SLIDE_TRIALS)):
        for material in material_list:
            for trial in trials:
                if (material,trial) in UNPUBLISHED_SLIDES:
                    omitted.append({"family":material,"id":trial,"reason":"absent in pinned publication"})
                    continue
                try:
                    case = (bounce_case(source,inputs,trial) if kind=="bounce"
                            else slider_case(source,inputs,material,trial))
                except ValueError as exc:
                    omitted.append({"family":material,"id":trial,"reason":str(exc)})
                    continue
                (ball_cases if kind=="bounce" else slide_cases).append(case)
    if frozen_calibration:
        prior = json.loads(frozen_calibration.read_text(encoding="utf-8"))
        zeta = prior["settings"]["ball_damping_ratio"]
        mu = prior["settings"]["friction"]
        calibration = prior["bounce_calibration"]
    else:
        zeta, calibration = fit_bounce(exe,ball_cases,models,inputs,bounce_dt)
        mu = fit_slider(slide_cases)
    report = {"settings":{"slider_dt_s":DT,"bounce_dt_s":bounce_dt,"mesh_deflection_m":MESH,"ball_stiffness":BALL_K,
                          "slider_stiffness":SLIDE_K,"normal_exponent":EXPONENT,
                          "ball_damping_ratio":zeta,"friction":mu,"calibration_ids":sorted(CALIBRATION)},
              "bounce_calibration":calibration,"omitted":omitted,"trials":[]}
    for kind,cases in (("bounce",ball_cases),("slider",slide_cases)):
        for case in cases:
            key = f"{case.get('material',kind)}-{case['id']:02}"
            config = root/"scenes"/(key+".scene")
            parameter = zeta if kind=="bounce" else mu[case["material"]]
            config.write_text(scene(kind,case,Path("../models"),inputs,parameter),encoding="utf-8")
            row = {"key":key,"family":case.get("material",kind),"id":case["id"],
                   "split":"calibration" if case["id"] in CALIBRATION else "held-out",
                   "initial_frame":case["initial_frame"],"duration_s":case["time"][-1]}
            waves = []
            double = [("brep",0.),("mesh",MESH)]
            if case["id"]==6 and (kind=="bounce" or case.get("material")=="plastic"):
                double += [("brep_half_step",0.)]
            for mode, deflection in double:
                normal_dt=bounce_dt if kind=="bounce" else DT
                dt=normal_dt/2 if mode=="brep_half_step" else normal_dt
                stride=max(1,round(DT/dt))
                trace,work = run(exe,config,root/"runs"/f"{key}-{mode}",case["time"][-1],dt,deflection,stride)
                wave = prediction(kind,case,trace,inputs)
                row[mode] = scores(kind,case,wave,trace)
                row[mode].update(work)
                if mode!="brep":
                    row[mode]["delta_to_brep_m"]=rms(wave-waves[0])
                waves.append(wave)
            np.savetxt(root/"curves"/f"{key}.csv",
                       np.column_stack((case["time"],case["positions"],case["observed"],*waves)),
                       delimiter=",",comments="",
                       header="time_s,observed_x_m,observed_y_m,observed_z_m,measured_m,"+
                       ",".join(mode+"_m" for mode,_ in double),fmt="%.10g")
            report["trials"].append(row)
        report["holdout"] = summarize(report["trials"])
        (root/"summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(report["holdout"],indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe",type=Path)
    parser.add_argument("source",type=Path)
    parser.add_argument("results",type=Path)
    parser.add_argument("--frozen-calibration",type=Path)
    parser.add_argument("--bounce-step",type=float,default=BOUNCE_DT)
    args=parser.parse_args()
    compare(args.exe.resolve(),args.source,args.results,args.frozen_calibration,args.bounce_step)
