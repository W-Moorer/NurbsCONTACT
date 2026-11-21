"""Run and validate the catalog-envelope cylindrical roller bearing study."""

import argparse
import json
import statistics
import subprocess
from pathlib import Path

import numpy as np

from run import csv_rows


DURATION=.36
ROLLERS=14
SHAFT_RPM=1800
CAGE_RPM=SHAFT_RPM*.5*(1-4/32)
PROTOCOL=(
    ("cad",0.,.00002,3),
    ("mesh-medium",.002,.00002,3),
    ("mesh-fine",.0005,.00002,3),
    ("mesh-coarse",.008,.00002,3),
    ("cad-dt10",0.,.00001,1),
)


def track(root,label):
    return np.genfromtxt(root/"runs"/f"{label}.csv",delimiter=",",names=True)


def run(exe,root):
    models,runs=root/"models",root/"runs"
    for folder in (models,runs):
        folder.mkdir(parents=True,exist_ok=True)
    if not all((models/model).is_file() for model in ("inner.step","outer.step","roller.step")):
        with (models/"models.log").open("w",encoding="utf-8") as log:
            subprocess.run([str(exe),"models",str(models)],check=True,stdout=log,stderr=subprocess.STDOUT)
    protocol=[]
    for label,deflection,dt,repeat in PROTOCOL:
        subprocess.run([str(exe),"run",str(models),str(deflection),str(dt),str(DURATION),
                        str(runs/label),str(repeat)],check=True)
        protocol.append({"label":label,"deflection_mm":deflection,"dt_s":dt,
                         "duration_s":DURATION,"repeats":repeat})
    (root/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n",encoding="utf-8")


def compare(trace,reference):
    after=trace["time"]>=.03
    time=trace["time"][after]
    drift=np.stack([trace[k][after]-np.interp(time,reference["time"],reference[k])
                    for k in ("x_um","y_um")])
    peak=trace["peak_roller_N"][after]-np.interp(time,reference["time"],reference["peak_roller_N"])
    displacement=np.linalg.norm(drift,axis=0)
    return {"orbit_rms_um":float(np.sqrt(np.mean(displacement**2))),
            "orbit_max_um":float(np.max(displacement)),
            "peak_roller_rms_N":float(np.sqrt(np.mean(peak**2)))}


def summarize(root):
    traces={}
    records={}
    for label,_,dt,repeat in PROTOCOL:
        result=track(root,label)
        rows=csv_rows(root/"runs"/f"{label}.stats.csv")
        nominal=round(DURATION/dt)
        if (len(result)!=nominal+1 or len(rows)!=repeat or abs(result["time"][-1]-DURATION)>1e-10
                or not np.all(np.diff(result["time"])>0)):
            raise ValueError(f"Short or nonmonotone bearing run {label}")
        if any(not np.all(np.isfinite(result[name])) for name in result.dtype.names):
            raise ValueError(f"Nonfinite bearing output {label}")
        guard=1000*min(float(row["min_interval_gap_mm"]) for row in rows)
        if guard<=2 or min(int(row["accepted"]) for row in rows)<nominal:
            raise ValueError(f"Insufficient checked clearance in {label}")
        if any(int(row["misses"]) for row in rows):
            raise ValueError(f"Missing bearing mesh intersections in {label}")
        timing=[float(row["wall_s"]) for row in rows]
        records[label]={"median_s":statistics.median(timing),"min_s":min(timing),"max_s":max(timing),
                        "min_native_interval_radial_gap_um":guard,"nominal_steps":nominal,
                        "peak_roller_N":float(np.max(result["peak_roller_N"])),
                        "mean_loaded_rollers":float(np.mean(result["active"])),
                        "triangles":int(rows[0]["triangles"]),"setup_s":float(rows[0]["setup_s"])}
        traces[label]=result
    for label,_d,_dt,_n in PROTOCOL:
        records[label]["relative_to_cad_dt10"]=compare(traces[label],traces["cad-dt10"])
        if label.startswith("mesh"):
            records[label]["relative_to_matched_cad"]=compare(traces[label],traces["cad"])
            records[label]["speed_ratio"]=records[label]["median_s"]/records["cad"]["median_s"]
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable",type=Path)
    parser.add_argument("results",type=Path)
    parser.add_argument("--analyze-only",action="store_true")
    args=parser.parse_args()
    if not args.analyze_only:
        run(args.executable.resolve(),args.results)
    evidence=summarize(args.results)
    (args.results/"summary.json").write_text(json.dumps(evidence,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(evidence["mesh-fine"],indent=2))


if __name__=="__main__":
    main()
