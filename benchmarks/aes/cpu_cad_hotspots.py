"""Collect user-mode call stacks in a warm-cache five-second CPU reducer run."""
import argparse
from pathlib import Path
import csv
import json
import os
import subprocess
import time

from cpu_backend_run import digest, run


def progress(path):
    if not path.is_file():
        return -1
    with path.open("rb") as stream:
        stream.seek(max(0,stream.seek(0,2)-8192))
        for line in reversed(stream.read().split(b"\n")[:-1]):
            step=line.partition(b",")[0]
            if step.isdigit():
                return int(step)
    return -1


def capture(exe,scene,profiler,directory):
    destination=directory/"solve"
    checkpoints=[("setup",-1,14),("early",6000,18),("late",30000,18)]
    manifest=dict(exe_sha256=digest(exe),profiler_sha256=digest(profiler),
                  scene_sha256=digest(scene),omp_num_threads=os.environ.get("OMP_NUM_THREADS"),windows=[])

    def observe(args,stdout,stderr,timeout):
        start=time.monotonic()
        csv_path=destination/"progress.csv"
        info=subprocess.STARTUPINFO()
        info.dwFlags|=subprocess.STARTF_USESHOWWINDOW
        info.wShowWindow=subprocess.SW_HIDE
        with subprocess.Popen(args,stdout=stdout,stderr=stderr) as solver:
            try:
                for name,step,seconds in checkpoints:
                    while progress(csv_path)<step and solver.poll() is None:
                        if time.monotonic()-start>=timeout:
                            raise subprocess.TimeoutExpired(args,timeout)
                        time.sleep(.25)
                    if solver.poll() is not None:
                        break
                    first=progress(csv_path)
                    trace=directory/(name+".sleepy")
                    subprocess.run([profiler,"-q","-a",str(solver.pid),"-t",str(seconds),"-o",trace.resolve()],
                                   startupinfo=info,check=True,timeout=min(120,start+timeout-time.monotonic()))
                    last=progress(csv_path)
                    manifest["windows"].append(dict(name=name,seconds=seconds,first_step=first,last_step=last,
                                                     trace_sha256=digest(trace)))
                    (directory/"capture.json").write_text(json.dumps(manifest,indent=2)+"\n")
                    print(name,first,last,flush=True)
                return subprocess.CompletedProcess(args,solver.wait(timeout=max(.01,timeout+start-time.monotonic())))
            finally:
                if solver.poll() is None:
                    solver.kill()
                    solver.wait()

    run(exe,scene,destination,"cad-bezier-bounds","contour-ray","penalty","shared","predictor",0,launch=observe)
    report=json.loads((destination/"run.json").read_text())
    if report["status"]!="complete" or len(manifest["windows"])!=3:
        raise RuntimeError("Incomplete five-second profile")
    with (destination/"progress.csv").open(newline="") as stream:
        booleans={int(r["step"]):int(r["cad_booleans"]) for r in csv.DictReader(stream)}
    for window in manifest["windows"]:
        if window["name"]=="setup":
            continue
        lo,hi=window["first_step"],window["last_step"]
        window["boolean_queries"]=booleans[hi]-booleans[max(0,lo-1)]
    (directory/"capture.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("exe","scene","profiler","directory"):
        parser.add_argument(name,type=Path)
    capture(**vars(parser.parse_args()))
