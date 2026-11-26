"""Summarize native call stacks separately from unsampled five-second wall timings."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
from zipfile import ZipFile

import numpy as np

from cpu_backend_run import digest
from cpu_cad_pressure_report import history
from cpu_pressure_columns import differences


def full_run(directory,control):
    report=json.loads((directory/"run.json").read_text())
    archived=json.loads(control.read_text())
    if (report["response_sha256"]!=archived["response_sha256"] or report["scene_sha256"]!=archived["scene_sha256"]
            or report["executable_sha256"]!=archived["executable_sha256"]
            or report["status"]!="complete" or report["last_accepted"]["step"]!=50000):
        raise ValueError("Unmatched original five-second timing")
    phases=("broad","search","group","area")
    names=["integration_s","cad_booleans","cad_patch_batches","cad_patch_queries",
           *("cad_patch_"+phase+"_s" for phase in phases)]
    rows=np.genfromtxt(directory/"progress.csv",delimiter=",",names=True,usecols=names)
    delta={field:np.diff(rows[field]) for field in names}
    fallback=delta["cad_booleans"]>0
    work={p:float(delta["cad_patch_"+p+"_s"][~fallback].sum()) for p in phases}
    return dict(process_s=report["process_s"],ordinary_steps=int((~fallback).sum()),
                ordinary_integration_s=float(delta["integration_s"][~fallback].sum()),
                fallback_steps=int(fallback.sum()),fallback_step_wall_s=float(delta["integration_s"][fallback].sum()),
                ordinary_pair_seconds=work,ordinary_pair_pct={p:100*v/sum(work.values()) for p,v in work.items()})


def stack_rows(path):
    with ZipFile(path) as archive:
        symbols=csv.reader(archive.read("Symbols.txt").decode("utf-8-sig").splitlines(),delimiter=" ")
        frames={row[0]:row[1:] for row in symbols}
        stats=archive.read("Stats.txt").decode()
        count=int(next(row.split(":",1)[1] for row in stats.splitlines() if row.startswith("Samples:")))
        entries=(line.split() for line in archive.read("Callstacks.txt").decode().splitlines())
        return count,[(float(row[0]),[frames[addr] for addr in row[1:]]) for row in entries]


def summarize(path):
    count,stacks=stack_rows(path)
    population=Counter()
    kernels=Counter()
    native=Counter()
    functions=Counter()
    setup=Counter()
    for weight,frames in stacks:
        names=[row[1] for row in frames]
        if any("WaitFor" in f or "DelayExecution" in f or "SleepConditionVariable" in f for f in names[:3]):
            population["waiting"]+=weight
            continue
        if any("nc::read_step_shape" in f for f in names):
            setup["read_step_shape"]+=weight
        app="|".join(row[1] for row in frames if row[0].lower()=="nc_contact")
        if not any(k in app for k in ("PatchSearch::","PatchRegionQuery::","CadExitQuery::","integrate_patch",
                                      "collect_indexed","DepthPacket::","CompiledChart::","propose_depth","::Search::")):
            population["other"]+=weight
            continue
        population["geometry"]+=weight
        functions[frames[0][0]+"!"+names[0]]+=weight
        if any("IntCurvesFace_Intersector::Perform" in f for f in names):
            kernel="OCCT_ray_face"
            if "CadExitQuery::interior_segment" in app:
                caller="group_interior_segment"
            elif "CadExitQuery::first_exit" in app and "integrate_patch" in app:
                caller="integration_first_exit"
            elif "CadExitQuery::first_exit" in app and "::Search::" in app:
                caller="discovery_first_exit"
            else:
                caller="other"
            native[caller]+=weight
        elif "CompiledChart::inside" in app:
            kernel="trim_membership"
        elif "CompiledChart::block_spans" in app:
            kernel="integration_block_span_search"
        elif "DepthPacket::" in app:
            kernel="depth_packet"
        elif "BezierRayAtlas::" in app:
            kernel="other_bezier_ray_query"
        else:
            kernel="other_geometry"
        kernels[kernel]+=weight
    def percents(weights,denominator):
        return {k:round(100*v/denominator,3) for k,v in weights.most_common()} if denominator else {}
    geom=population["geometry"]
    return dict(successful_thread_samples=count,weighted_thread_seconds=dict(population),
                kernel_pct_of_geometry=percents(kernels,geom),occt_callers_pct_of_geometry=percents(native,geom),
                top_geometry_leaf_pct=dict(list(percents(functions,geom).items())[:12]),
                step_read_pct_of_nonwait=round(100*setup["read_step_shape"]/(geom+population["other"]),3))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("timing","control","profile","output"):
        p.add_argument(key,type=Path)
    p.add_argument("--scene",type=Path,required=True)
    args=p.parse_args()
    capture=json.loads((args.profile/"capture.json").read_text())
    measured=json.loads((args.profile/"solve/run.json").read_text())
    if measured["status"]!="complete" or measured["last_accepted"]["step"]!=50000:
        raise ValueError("Incomplete diagnostic run")
    if capture["scene_sha256"]!=measured["scene_sha256"] or capture["exe_sha256"]!=measured["executable_sha256"]:
        raise ValueError("Diagnostic executable or scene differs")
    archived=json.loads(args.control.read_text())
    if digest(args.scene)!=archived["scene_sha256"]:
        raise ValueError("Wrong scene for response comparison")
    current=history(args.profile/"solve/response.csv",args.scene)
    with np.load(args.control.parent/"responses.npz") as original:
        np.testing.assert_allclose(current["time_s"],original["time_s"],rtol=0,atol=1e-10)
        loads=original["cad-bezier-bounds_normal_N"]
        torque_delta=differences(current["output_torque_Nm"]-original["cad-bezier-bounds_output_torque_Nm"])
        switch_mismatches=((current["normal_N"]>0)!=(loads>0)).sum(axis=0).tolist()
    key_fields=("steps","cad_patch_batches","cad_boolean_queries","cad_pressure_full","cad_depth_tests")
    results=dict(profile_run=measured,archive_timing=full_run(args.timing,args.control),
                 response_hash_matches_control=measured["response_sha256"]==archived["response_sha256"],
                 output_torque_delta_Nm=torque_delta,loaded_mismatches=switch_mismatches,
                 work_control={k:[int(archived["statistics"][k]),int(measured["statistics"][k])]
                               for k in key_fields},windows=[])
    for window in capture["windows"]:
        path=args.profile/(window["name"]+".sleepy")
        if digest(path)!=window["trace_sha256"]:
            raise ValueError(f"Changed trace: {path}")
        with ZipFile(path) as bundle:
            if any(name not in ("Stats.txt","Symbols.txt","IPCounts.txt","Callstacks.txt")
                   and not name.startswith("Version ") for name in bundle.namelist()):
                raise ValueError("Expected a symbol-only stack capture")
        if window["name"]!="setup" and window["boolean_queries"]!=0:
            raise ValueError("Ordinary-step sample includes a Boolean event")
        results["windows"].append({**window,**summarize(path)})
        shutil.copyfile(path,args.output.parent/path.name)
    args.output.write_text(json.dumps(results,indent=2)+"\n")


if __name__=="__main__":
    main()
