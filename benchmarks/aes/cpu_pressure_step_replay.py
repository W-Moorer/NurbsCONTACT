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


def replay(scene, previous, current, executable, output, current_method="cad-bezier",window_steps=3,
           previous_method="cad-bezier",steps=None,final_width_only=False):
    if window_steps<1:
        raise ValueError("Replay window must include at least the decision step")
    bodies,pairs=read_model(scene)
    sign=bodies[pairs[1][1]][2][2]
    fields=["time_s",*[k+str(i) for k in ("q","v","a","generalized_force") for i in range(3)]]
    sources={}
    progress={}
    for name,path in (("previous",previous),("current",current)):
        verified_dynamics(scene,path,current_method if name=="current" else previous_method)
        sources[name]=np.genfromtxt(path/"response.csv",delimiter=",",names=True,usecols=fields)
        progress[name]=np.genfromtxt(path/"progress.csv",delimiter=",",names=True,usecols=("cad_patch_batches",))
    def batches(name):
        return np.diff(progress[name]["cad_patch_batches"],prepend=0)
    events=np.flatnonzero(batches("previous")!=batches("current")).tolist()
    if steps:
        if set(steps)-set(events):
            raise ValueError("Selected steps must have different HHT decisions")
        events=sorted(set(steps))
    result=[]
    if not len(events):
        output.write_text(json.dumps(dict(executable_sha256=digest(executable),rows=[]),indent=2)+"\n")
        return
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)
        # Include the output frames following the decision to check a delayed torque peak.
        for width in ((window_steps,) if final_width_only else range(1,window_steps+1)):
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
                 "--method",current_method,"--solver","penalty","--penalty-stiffness","shared",
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
    output.write_text(json.dumps(dict(executable_sha256=digest(executable),previous_method=previous_method,
                    current_method=current_method,previous_response_sha256=digest(previous/"response.csv"),
                    current_response_sha256=digest(current/"response.csv"),rows=result),indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("scene","previous","current","executable","output"): p.add_argument(name,type=Path)
    p.add_argument("--current-method",choices=("cad-bezier","cad-bezier-bounds","cad-bezier-relocate"),default="cad-bezier")
    p.add_argument("--previous-method",choices=("cad-bezier","cad-bezier-bounds"),default="cad-bezier")
    p.add_argument("--window-steps",type=int,default=3)
    p.add_argument("--steps",type=int,nargs="+",help="Subset of differing integration decisions")
    p.add_argument("--final-width-only",action="store_true",help="Replay only the requested window length")
    replay(**vars(p.parse_args()))
