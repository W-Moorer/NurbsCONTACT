"""Replay synchronized shaker and tip acceleration from the cantilever gap experiment."""

import argparse
import hashlib
import json
import math
import subprocess
import zipfile
from pathlib import Path

import numpy as np
from numba import njit
from scipy.optimize import differential_evolution, minimize

from run import run, values


DOI = "https://doi.org/10.5281/zenodo.19205934"
MD5 = "84b170221d9908fa430eeb5f64061b7d"
RATE = 1024
SENSOR_RATIO = (.150/.170)**2*(3-.150/.170)/2
MASS = .235*.170*(33/140)+.031+.005*SENSOR_RATIO**2
NAMES = ("frequency_hz","damping_ratio","participation","upper_gap_mm",
         "lower_gap_mm","log10_contact_rate_s2","contact_damping_s")
BOUNDS = ((7.,14.),(.003,.12),(.7,1.8),(.5,1.),(.5,1.),(3.8,6.4),(0,700))
CALIBRATION = ("01_random","02_random")
HELD_OUT = ("03_random","06_stepped_sine")


def read_archive(path):
    with path.open("rb") as stream:
        if hashlib.file_digest(stream,"md5").hexdigest()!=MD5:
            raise ValueError("Zenodo archive checksum differs from the published value")
    records = {}
    with zipfile.ZipFile(path) as bundle:
        for name in bundle.namelist():
            source = Path(name)
            if source.suffix.lower()!=".csv" or source.stem not in (*CALIBRATION,*HELD_OUT):
                continue
            with bundle.open(name) as stream:
                header = stream.readline().decode("utf-8-sig").strip().split(",")
                if header!=["time_s","acc_base_ms2","acc_tip_ms2"]:
                    raise ValueError(f"unexpected columns in {name}: {header}")
                signal = np.loadtxt(stream,delimiter=",")
            clock = np.arange(len(signal))/RATE
            if not np.allclose(signal[:,0],clock,rtol=0,atol=1e-6):
                raise ValueError(f"timestamps in {name} do not match the instrument clock")
            records[source.stem] = signal
    if set(records)!=set((*CALIBRATION,*HELD_OUT)):
        raise ValueError(f"expected four named full experimental records, got {list(records)}")
    return records


@njit
def acceleration(q,v,base,w,c,g_hi,g_lo,k_contact,d_contact,participation):
    value = -w*w*q-c*v-participation*base
    if q>g_hi:
        value -= max(0,k_contact*(q-g_hi)+d_contact*v)
    elif q<-g_lo:
        value += max(0,k_contact*(-q-g_lo)-d_contact*v)
    return value


@njit
def reduced_trial(base,parameters,substeps):
    # A small explicit reference model accelerates parameter identification.
    # The archived predictions come from the production STEP/HHT executable.
    frequency,zeta,participation,g_hi,g_lo,log_contact,d_contact = parameters
    omega = 2*np.pi*frequency
    drag = 2*zeta*omega
    stiffness = 10**log_contact
    g_hi *= .001
    g_lo *= .001
    dt = 1/(RATE*substeps)
    output = np.empty(len(base))
    q,v = 0.,0.
    for i in range(len(base)-1):
        first = base[i]
        slope = (base[i+1]-first)/substeps
        output[i] = first+SENSOR_RATIO*acceleration(q,v,first,omega,drag,g_hi,g_lo,
                                                   stiffness,d_contact,participation)
        for j in range(substeps):
            b0 = first+j*slope
            b_half = b0+slope/2
            a0 = acceleration(q,v,b0,omega,drag,g_hi,g_lo,stiffness,d_contact,participation)
            v1 = v+dt*a0/2
            a1 = acceleration(q+dt*v/2,v1,b_half,omega,drag,g_hi,g_lo,stiffness,d_contact,participation)
            v2 = v+dt*a1/2
            a2 = acceleration(q+dt*v1/2,v2,b_half,omega,drag,g_hi,g_lo,stiffness,d_contact,participation)
            v3 = v+dt*a2
            a3 = acceleration(q+dt*v2,v3,b0+slope,omega,drag,g_hi,g_lo,stiffness,d_contact,participation)
            q += dt*(v+2*v1+2*v2+v3)/6
            v += dt*(a0+2*a1+2*a2+a3)/6
    output[-1] = base[-1]+SENSOR_RATIO*acceleration(q,v,base[-1],omega,drag,g_hi,g_lo,
                                                   stiffness,d_contact,participation)
    return output


def calibrate(records):
    # Both calibration records use the same parameters. The first two seconds
    # allow the response to settle from the documented zero-state assumption.
    trials = [records[name][:RATE*22] for name in CALIBRATION]
    warm = 2*RATE
    inputs = [(np.ascontiguousarray(row[:,1]),row[warm:,2]) for row in trials]
    scales = [np.mean(y*y) for _,y in inputs]

    def score(params,substeps):
        return sum(np.mean((reduced_trial(base,params,substeps)[warm:]-target)**2)/scale
                   for (base,target),scale in zip(inputs,scales))/len(inputs)

    result = differential_evolution(lambda x:score(x,4),BOUNDS,popsize=8,maxiter=65,
                                    seed=304,polish=False,tol=.001)
    fine = minimize(lambda x:score(x,8),result.x,method="Powell",bounds=BOUNDS,
                    options={"maxiter":22,"xtol":1e-5,"ftol":1e-6})
    candidate = fine.x if fine.fun<=score(result.x,8) else result.x
    return dict(zip(NAMES,map(float,candidate))),float(score(candidate,8))


def geometry(exe, work):
    for name,dimensions in (("tip",(24,30,1,0)),("lower",(36,36,8,-4)),("upper",(36,36,8,4))):
        dest = work/f"{name}.step"
        if not dest.exists():
            subprocess.run([str(exe),"make","box",*map(str,dimensions),str(dest)],
                           check=True,capture_output=True)


def write_scene(directory,base,parameters,linear=False):
    f,zeta,participation,g_hi,g_lo,log_k,drag = (parameters[key] for key in NAMES)
    omega = 2*math.pi*f
    contact_k = MASS*10**log_k
    half_tip = .0005
    if linear:
        g_hi = g_lo = 100.
    rows = [
        "nc-contact-v2","gravity 0 0 0 1 3 2",
        values("coordinate",MASS,0,0,MASS*omega**2,2*MASS*zeta*omega,0,0),
        values("body",'"../../lower.step"',0,0,0,-half_tip-g_lo*.001,0,0,1,0,0,0,0,-1,0,0,0,0),
        values("body",'"../../tip.step"',0,0,0,0,0,0,1,0,0,0,0,-1,0,0,0,1,0,0,0,1),
        values("body",'"../../upper.step"',0,0,0,half_tip+g_hi*.001,0,0,1,0,0,0,0,-1,0,0,0,0),
        values("probe",0,1,"world",0,0,0,0,0,0,0,0,1,contact_k,1,MASS*drag,0,.001,.01),
        values("probe",1,2,"world",0,0,0,0,0,0,0,0,1,contact_k,1,MASS*drag,0,.001,.01),
        "drives 0","loads 1",f"load 0 {len(base)}",
    ]
    path = directory/"cantilever.scene"
    with path.open("w",encoding="ascii") as stream:
        stream.write("\n".join(rows)+"\n")
        load = -MASS*participation*base
        np.savetxt(stream,np.column_stack((np.arange(len(base))/RATE,load)),fmt="%.17g")
    return path


def evaluate(exe, work, name, data, parameters, kind, subdivisions, mesh):
    target = work/name/kind
    target.mkdir(parents=True,exist_ok=True)
    scene = write_scene(target,data[:,1],parameters,linear=kind=="linear")
    source = target/"raw-step"
    dt = 1/(RATE*subdivisions)
    trace,stats = run(exe,scene,source,(len(data)-1)/RATE,dt,mesh,stride=subdivisions)
    if len(trace)!=len(data) or np.max(np.abs(trace["time_s"]-np.arange(len(data))/RATE))>2e-9:
        raise ValueError("simulation samples are not on the instrument clock")
    predicted = data[:,1]+SENSOR_RATIO*trace["a0"]
    active = (trace["p0_normal_N"]>0)|(trace["p1_normal_N"]>0)
    used = slice(2*RATE,None)
    error = predicted[used]-data[used,2]
    data_rms = np.sqrt(np.mean(data[used,2]**2))
    metrics = {
        "record":name,"model":kind,"step_s":dt,"source_duration_s":(len(data)-1)/RATE,
        "scoring_start_s":2,"raw_rmse_m_s2":float(np.sqrt(np.mean(error**2))),
        "measured_rms_m_s2":float(data_rms),"modeled_rms_m_s2":float(np.sqrt(np.mean(predicted[used]**2))),
        "normalized_rmse":float(np.sqrt(np.mean(error**2))/data_rms),
        "correlation":float(np.corrcoef(predicted[used],data[used,2])[0,1]),
        "contact_active_observation_fraction":float(np.mean(active[used])),
        "stats":stats,
    }
    print(name,kind,"RMSE",metrics["raw_rmse_m_s2"],"relative",metrics["normalized_rmse"],
          "correlation",metrics["correlation"],flush=True)
    return predicted,trace,metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe",type=Path)
    parser.add_argument("data",type=Path)
    parser.add_argument("work",type=Path)
    parser.add_argument("--parameter-file",type=Path)
    parser.add_argument("--calibrate-only",action="store_true")
    args = parser.parse_args()
    records = read_archive(args.data)
    args.work.mkdir(parents=True,exist_ok=True)
    if args.parameter_file:
        fit = json.loads(args.parameter_file.read_text(encoding="utf-8"))
    else:
        parameters,error = calibrate(records)
        fit = {"parameters":parameters,"reference_mean_normalized_mse":error}
    (args.work/"fit.json").write_text(json.dumps(fit,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(fit),flush=True)
    if args.calibrate_only:
        return
    exe = args.exe.resolve()
    geometry(exe,args.work)
    rows = []
    for name in HELD_OUT:
        record = records[name]
        outcomes = [evaluate(exe,args.work,name,record,fit["parameters"],*branch)
                    for branch in (("brep",8,0.),("mesh",8,2.5e-5),("refined",16,0.),("linear",8,0.))]
        native,trace,_ = outcomes[0]
        prefix = args.work/name
        np.savetxt(prefix/"paired.csv",
                   np.column_stack((record,*(outcome[0] for outcome in outcomes),
                                    trace["q0"],trace["p0_normal_N"],trace["p1_normal_N"])),
                   fmt="%.12g",delimiter=",",comments="",
                   header="time_s,acc_base_ms2,acc_tip_ms2,brep_ms2,mesh_ms2,refined_ms2,linear_ms2,"
                          "relative_tip_m,lower_N,upper_N")
        rows.extend(outcome[2] for outcome in outcomes)
        rows[-4]["refinement_rmse_m_s2"] = float(np.sqrt(np.mean((outcomes[2][0]-native)**2)))
        rows[-4]["mesh_delta_rmse_m_s2"] = float(np.sqrt(np.mean((outcomes[1][0]-native)**2)))
    summary = {"dataset":DOI,"archive_md5":MD5,"sample_rate_hz":RATE,"equivalent_mass_kg":MASS,
               "sensor_mode_ratio":SENSOR_RATIO,"phase_shift_s":0,"output_filter":None,
               "calibration_records":CALIBRATION,"held_out_records":HELD_OUT,"fit":fit,"runs":rows}
    (args.work/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
