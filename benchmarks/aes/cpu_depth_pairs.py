"""Alternate two resident CPU geometry queries at every archived reducer pose."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import tempfile
import threading

import numpy as np

from cpu_cad_pressure_report import wrench
from cpu_pressure_columns import digest, fixed_geometry_inputs, stage_poses, write


def paired(scene,exe,archive,output):
    trajectory,indices,warm,bodies,pairs,angles,origins=fixed_geometry_inputs(scene,archive)
    result=dict(scene_sha256=digest(scene),exe_sha256=digest(exe),trajectory_sha256=digest(trajectory),stages=[])
    for stage,(a,b,stiffness,limit) in enumerate(pairs,1):
        poses=stage_poses(bodies,angles,origins,indices,a,b)
        args=[exe,bodies[a][0],bodies[b][0],"6",str(2*limit),"--loaded","--bezier"]
        # Retain independent caches with identical pose histories. Only one
        # process receives a pose at a time. Native timers exclude communication.
        elapsed=np.empty((len(poses),2,2))
        largest=0.
        compared=0
        with ExitStack() as stack:
            procs={}
            def stop():
                for proc in list(procs.values()):
                    if proc.poll() is None: proc.kill()
            watchdog=threading.Timer(1800,stop)
            stack.callback(watchdog.cancel)
            watchdog.start()
            for i,pose in enumerate(poses):
                order=(0,1) if (i+stage)%2 else (1,0)
                line=" ".join(map("{:.17g}".format,pose))+"\n"
                rows={}
                for mode in order:
                    if mode not in procs:
                        err=stack.enter_context(tempfile.TemporaryFile(mode="w+t"))
                        procs[mode]=stack.enter_context(subprocess.Popen(args+(["--depth-bounds"] if mode else []),
                            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1))
                        stack.callback(stop)
                    proc=procs[mode]
                    proc.stdin.write(line)
                    proc.stdin.flush()
                    row=json.loads(proc.stdout.readline())
                    rows[mode]=row
                    elapsed[i,mode]=[row["loaded_s"]-row["query_s"],row["loaded_s"]]
                def faces(row):
                    return [(g,w[:2]) for g,w in zip(row["labels"],row["witnesses"],strict=True)]
                x,y=rows[0],rows[1]
                if x["resolved"]!=y["resolved"] or faces(x)!=faces(y):
                    raise ValueError(f"Changed contact grouping at stage {stage}, step {indices[i]}")
                if x["resolved"]:
                    moments=[wrench(r["regions"],origins[indices[i],b],bodies[b][2],stiffness,limit,True) for r in (x,y)]
                    if any(v is None for v in moments): raise ValueError("Invalid wrench")
                    largest=max(largest,abs(moments[0][-1]-moments[1][-1]))
                    compared+=1
            for proc in procs.values():
                proc.stdin.close()
                if proc.wait(timeout=30): raise RuntimeError("CPU query did not finish")
        def timings(selection):
            means=1000*elapsed[selection].mean(axis=0)
            return dict(previous_area_ms=means[0,0],bounded_area_ms=means[1,0],
                        previous_query_ms=means[0,1],bounded_query_ms=means[1,1])
        result["stages"].append(dict(stage=stage,compared=compared,timed=int(warm.sum()),max_torque_delta_Nm=largest,
            **timings(warm),windows=[dict(start=int(indices[j]),**timings(np.arange(j+20,j+100)))
                                     for j in range(0,len(poses),100)]))
    output.parent.mkdir(parents=True,exist_ok=True)
    write(output,result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("scene","exe","archive","output"): p.add_argument(key,type=Path)
    paired(**vars(p.parse_args()))
