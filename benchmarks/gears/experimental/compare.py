"""Calibrate on one condition per rig and score the frozen model on held-out conditions."""

import argparse
import csv
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np


NASA = {
    "a": (2000, .47*71.77),
    "b": (4000, .16*71.77),
    "c": (6000, .47*71.77),
    "d": (6000, .94*71.77),
}
UNSW_CAL = "Data_05Hz_10Nm_H"
MESH = 0.0015625
PHASE = np.arange(128)/128


def read(file):
    return np.genfromtxt(file,delimiter=",",names=True)


def read_run(prefix):
    with prefix.with_suffix(".stats.csv").open(encoding="utf-8",newline="") as f:
        stats=next(csv.DictReader(f))
    trace=read(prefix.with_suffix(".csv"))
    if int(stats["accepted"]) < int(stats["steps"]) or np.min(trace["gap_mm"]) < .002:
        raise ValueError(f"Incomplete or invalid trajectory {prefix}")
    return trace,{k:float(v) for k,v in stats.items()}


def run(exe, models, prefix, case, motion, stiffness, damping, mesh=0., refine=1, inertia=1.):
    prefix.parent.mkdir(exist_ok=True,parents=True)
    duration,dt=(.16,1e-6) if case=="rig28" else (.72,1e-5)
    args=[exe,"experiment",case,models,mesh,*motion,dt/refine,duration,stiffness,damping,prefix,inertia]
    result=subprocess.run(list(map(str,args)),capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(f"{prefix.name}: {result.stderr.strip()}")
    return read_run(prefix)


def tooth_wave(trace, angle):
    parts=[]
    for i in range(4):
        idx=trace[f"id{i}"]
        valid=np.isfinite(idx)
        parts.append(np.column_stack((idx[valid],trace[f"roll{i}_deg"][valid],trace[f"force{i}_N"][valid])))
    data=np.concatenate(parts)
    ids=np.unique(data[:,0])[2:-2]
    samples=[]
    for index in ids:
        row=data[data[:,0]==index]
        order=np.argsort(row[:,1])
        samples.append(np.interp(angle,row[order,1],row[order,2],left=0,right=0))
    if len(samples)<12:
        raise ValueError("too few simulated single-tooth engagements")
    return np.mean(samples,axis=0)


def transmission(trace):
    angle=trace["pinion_angle_rad"]*27/(2*np.pi)
    cycles=np.arange(np.ceil(angle[0]),np.floor(angle[-1]))
    if np.any(np.diff(angle)<=0) or len(cycles)<20:
        raise ValueError("insufficient ordered model mesh cycles")
    samples=np.interp(cycles[:,None]+PHASE,angle,trace["te_um"]).mean(axis=0)
    return samples-samples.mean()


def rms(error):
    return float(np.sqrt(np.mean(np.square(error))))


def metrics(expected, actual, basis):
    error={"rms":rms(actual-expected),"data_rms":rms(expected),
           "test_peak":float(np.max(expected)),"prediction_peak":float(np.max(actual)),
           "test_peak_to_peak":float(np.ptp(expected)),"prediction_peak_to_peak":float(np.ptp(actual))}
    if basis == "roll":
        error.update(peak_angle_error_deg=float(basis_grid(expected, actual)))
    else:
        error["harmonic_error_um"]=(2*np.abs(np.fft.rfft(actual))/len(actual)
                                   -2*np.abs(np.fft.rfft(expected))/len(expected))[1:5].tolist()
    return error


def basis_grid(expected,actual):
    # The NASA reporting grid below uses exactly 0.125 degrees.
    return (np.argmax(actual)-np.argmax(expected))*.125


def optimize(exe, models, root, case, motion, reference, abscissa):
    ks = [6e7,8e7,1.1e8,1.6e8,2.2e8,3.2e8] if case=="rig28" else [3.5e7,5e7,7.5e7,1e8,1.5e8,2.2e8]
    candidates=[]
    pipe=root/"calibration"/case
    for k,damping in itertools.product(ks,(.05,.10,.20)):
        prefix=pipe/f"k-{int(k):d}-z{round(100*damping):02d}"
        try:
            trace,_=run(exe,models,prefix,case,motion,k,damping)
            if case=="rig28":
                wave=tooth_wave(trace,abscissa)
                shift,error=0,rms(wave-reference)
            else:
                wave=transmission(trace)
                errors=[rms(np.roll(wave,p)-reference) for p in range(len(wave))]
                shift=int(np.argmin(errors))
                error=errors[shift]
        except RuntimeError as failure:
            candidates.append({"stiffness_N_m":k,"damping_ratio":damping,"error":str(failure)})
            continue
        candidates.append({"stiffness_N_m":k,"damping_ratio":damping,"phase_shift_bins":shift,"calibration_rms":error})
    (pipe/"grid.json").write_text(json.dumps(candidates,indent=2)+"\n",encoding="utf-8")
    successful=[c for c in candidates if "calibration_rms" in c]
    if not successful:
        raise RuntimeError(f"No successful calibration runs for {case}")
    best=min(successful,key=lambda item:item["calibration_rms"])
    print(case,"calibration",best,flush=True)
    return best


def compare(exe,root,nasa_models):
    measurements=root/"measurements"
    raw=root/"runs"
    curves=root/"curves"
    curves.mkdir(exist_ok=True,parents=True)
    settings={}
    references={}
    for key,motion in NASA.items():
        q=read(measurements/f"nasa-7{key}.csv")
        x=np.arange(12 if key!="d" else 12.625,29.501,.125)
        references["nasa-"+key]=(motion,x,np.interp(x,q["roll_deg"],q["force_N"],right=0,left=0))
    unsw=json.loads((measurements/"unsw.json").read_text(encoding="utf-8"))
    for key,record in unsw.items():
        data=read(measurements/f"{key}.csv")
        references[key]=((record["input_rpm"],record["torque_Nm"]),data["mesh_cycle"],data["te_um"])
    for rig,case,models,key in (("nasa","rig28",nasa_models,"nasa-a"),
                               ("unsw","unsw",root/"models",UNSW_CAL)):
        motion,x,actual=references[key]
        settings[rig]=optimize(exe,models,root,case,motion,actual,x)

    summary={"calibration":settings,"comparisons":{}}
    for key,(motion,x,actual) in references.items():
        rig="nasa" if key.startswith("nasa-") else "unsw"
        case="rig28" if rig=="nasa" else "unsw"
        models=nasa_models if rig=="nasa" else root/"models"
        fit=settings[rig]
        kwargs={"stiffness":fit["stiffness_N_m"],"damping":fit["damping_ratio"]}
        modes=[("brep",0.,1,1.),("mesh",MESH,1,1.)]
        if key in ("nasa-d","Data_10Hz_15Nm_H"):
            modes.append(("brep_fine_step",0.,2,1.))
        if key=="nasa-c":
            modes.append(("nasa_inertia_times10",0.,1,10.))
        values=[x,actual]
        header=["roll_deg" if rig=="nasa" else "mesh_cycle","measured"]
        record={"rpm":motion[0],"input_torque_Nm":motion[1],"role":"calibration" if key in ("nasa-a",UNSW_CAL) else "held_out"}
        for label,deflection,refine,inertia in modes:
            trace,timing=run(exe,models,raw/f"{key}-{label}",case,motion,mesh=deflection,refine=refine,
                             inertia=inertia,**kwargs)
            if rig=="nasa":
                q=tooth_wave(trace,x)
                scores=metrics(actual,q,"roll")
            else:
                q=np.roll(transmission(trace),fit["phase_shift_bins"])
                scores=metrics(actual,q,"phase")
            values.append(q)
            header.append(label)
            record[label]={"error":scores,"timing":timing}
            if label=="brep":
                native=q
            else:
                record[label]["delta_to_nominal_cad_rms"]=rms(q-native)
        np.savetxt(curves/f"{key}.csv",np.column_stack(values),delimiter=",",comments="",
                   header=",".join(header),fmt="%.10g")
        summary["comparisons"][key]=record
        (root/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
        print(key,record["brep"]["error"]["rms"],record["mesh"]["error"]["rms"],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe",type=Path)
    parser.add_argument("results",type=Path)
    parser.add_argument("--nasa-models",type=Path,default=Path("results/gears/models"))
    args=parser.parse_args()
    compare(args.exe.resolve(),args.results,args.nasa_models)


if __name__=="__main__":
    main()
