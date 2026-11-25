"""Repeat differing HHT decisions from both accepted reducer trajectories."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from _gpu_source import read_model
from cpu_backend_run import digest
from cpu_cad_pressure_report import smooth_loads
from cpu_cad_pruning import verified_dynamics


def replay(scene, previous, current, executable, output):
    bodies,pairs=read_model(scene)
    sign=bodies[pairs[1][1]][2][2]
    fields=["time_s",*[k+str(i) for k in ("q","v","a","generalized_force") for i in range(3)]]
    sources={}
    progress={}
    for name,path in (("previous",previous),("current",current)):
        verified_dynamics(scene,path,"cad-bezier")
        sources[name]=np.genfromtxt(path/"response.csv",delimiter=",",names=True,usecols=fields)
        progress[name]=np.genfromtxt(path/"progress.csv",delimiter=",",names=True,usecols=("cad_patch_batches",))
    def batches(name):
        return np.diff(progress[name]["cad_patch_batches"],prepend=0)
    events=np.flatnonzero(batches("previous")!=batches("current"))
    result=[]
    if not len(events):
        output.write_text(json.dumps(dict(executable_sha256=digest(executable),rows=[]),indent=2)+"\n")
        return
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)
        # Cover the decision step and the next two output frames, so a
        # delayed torque excursion is checked at its actual time.
        for width in (1,2,3):
            records,labels=[],[]
            for name,trace in sources.items():
                for step in events:
                    if step<1 or step+width-1>=len(trace):
                        continue
                    for refinement in (1,2,4,8,16):
                        # Warm neighbouring poses with non-overlapping intervals.
                        first=int(step)-1
                        for j in range(first-min(8//width,first//width)*width,int(step),width):
                            r=trace[j]
                            records.append([r["time_s"],width*refinement if j==step-1 else width,
                                            *[r[k] for k in fields[1:]]])
                            labels.append((name,int(step),refinement) if j==step-1 else None)
            if not records:
                continue
            np.savetxt(path/"inputs.txt",records,fmt=["%.17g","%d"]+["%.17g"]*(len(fields)-1))
            cmd=[executable,scene,"0",str(width*.0001),str(width*.0001),"1",path/"audit.csv",
                 "--method","cad-bezier","--solver","penalty","--penalty-stiffness","shared",
                 "--spectral-radius","0","--geometry-update","predictor","--hht-replay",path/"inputs.txt"]
            subprocess.run(cmd,check=True,timeout=1800,capture_output=True)
            with (path/"audit.csv").open(newline="") as stream:
                rows=list(csv.DictReader(stream))
            for row,label in zip(rows,labels,strict=True):
                if label is None: continue
                name,step,refinement=label
                end=step+width-1
                record=sources[name][end]
                t=np.array([float(row["time_s"]),record["time_s"]])
                q,v=(np.array([[float(row[f"{key}{i}"]) for i in range(3)],
                               [record[f"{key}{i}"] for i in range(3)]]) for key in ("q","v"))
                _,smooth=smooth_loads(scene,t,q,v)
                result.append(dict(starting_trace=name,first_step=step,span_steps=width,refinement=refinement,
                    forced_substeps=int(row["substeps"]),additional_splits=int(row["subdivisions"]),
                    rejected_trials=int(row["rejected_trials"]),
                    replay_torque_Nm=sign*(float(row["force2"])-smooth[0,2]),
                    original_torque_Nm=sign*(record["generalized_force2"]-smooth[1,2])))
    output.write_text(json.dumps(dict(executable_sha256=digest(executable),rows=result),indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("scene","previous","current","executable","output"): p.add_argument(name,type=Path)
    replay(**vars(p.parse_args()))
