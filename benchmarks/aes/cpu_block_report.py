"""Compare monitored reducer timings, contact switches and torque at saved CPU poses."""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cpu_cad_pruning import verified_dynamics
from cpu_cad_pressure_report import history
from cpu_pressure_columns import differences, report, write


def mesh_audit(scene, runs, crossed, out):
    keys=("wall_s","setup_s","force_evaluations","triangle_tests","mesh_detection_s","mesh_pair_bvh_builds","mesh_patch_builds")
    controls=[]
    for directory in runs:
        record,_=verified_dynamics(scene,directory,"mesh-obb")
        with (directory/"progress.csv").open() as stream:
            windows=[(int(row["step"]),float(row["integration_s"])) for row in csv.DictReader(stream) if int(row["step"])%5000==0]
        if [step for step,_ in windows]!=list(range(0,50001,5000)):
            raise ValueError(f"Incomplete half-second timing series in {directory}")
        times=[t for _,t in windows]
        controls.append(dict(exe_sha256=record["executable_sha256"], response_sha256=record["response_sha256"],
                         scene_sha256=record["scene_sha256"], process_s=record["process_s"],
                         statistics={k:float(record["statistics"][k]) for k in keys},
                         half_second_bins_s=np.diff(times).tolist()))
    if len({r["response_sha256"] for r in controls})!=1 or controls[1]["exe_sha256"]!=controls[2]["exe_sha256"]:
        raise ValueError("Unmatched program or response in the historical timing control")
    work=("force_evaluations","triangle_tests","mesh_pair_bvh_builds","mesh_patch_builds")
    if len({tuple(row["statistics"][key] for key in work) for row in controls})!=1:
        raise ValueError("Historical mesh geometry work changed")
    abba=json.loads(crossed.read_text())
    if [r["exe_sha256"] for r in abba]!=[controls[i]["exe_sha256"] for i in (0,1,1,0)] or len({(r["response"],r["triangle_tests"]) for r in abba})!=1:
        raise ValueError("Crossed timings used unmatched mesh controls")
    delta=controls[1]["process_s"]-controls[0]["process_s"]
    detection=controls[1]["statistics"]["mesh_detection_s"]-controls[0]["statistics"]["mesh_detection_s"]
    write(out/"mesh-timing.json",dict(half_second_windows=controls,delta_wall_s=delta,
          delta_detection_s=detection,detection_share_of_slowdown=detection/delta if delta else None,
          crossed_800_steps=abba,repeat_samples=json.loads((runs[-1]/"telemetry.json").read_text())))
    durations=np.asarray([r["half_second_bins_s"] for r in controls])
    times=np.arange(.5,5.1,.5)
    fig,axes=plt.subplots(1,2,figsize=(9,3),layout="constrained")
    colors=("#287974","#a35432","#455e91")
    for i,(row,marker) in enumerate(zip(durations,("o","s","^"))):
        label=f'{controls[i]["process_s"]:.0f} s full run'
        axes[0].plot(times,row,marker=marker,color=colors[i],lw=1.3,ms=4,label=label)
        if i:
            axes[1].plot(times,row/durations[0],marker=marker,color=colors[i],lw=1.3,ms=4,label=label)
    axes[0].set(ylabel="Wall time per 0.5 s window (s)",xlabel="Window end in simulation (s)")
    axes[1].axhline(1,color=colors[0],lw=.8,ls="--")
    axes[1].set(ylabel="Window time / original fast run",xlabel="Window end in simulation (s)",ylim=(.65,2.9))
    for ax in axes:
        ax.legend(frameon=False,fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
    fig.savefig(out/"mesh-timing.png",dpi=165)
    plt.close(fig)


def dynamics(scene, runs, out, serial_run=None):
    report(scene,runs,out,previous=out.parent/"columns")
    reference=out.parent/"recurring"
    prior=out.parent/"columns"
    with np.load(out/"responses.npz") as current,np.load(prior/"responses.npz") as previous,np.load(reference/"responses.npz") as original:
        t=current["time_s"]
        data=dict(response_frames=len(t),last_time_s=float(t[-1]))
        y=current["cad-bezier_output_torque_Nm"]
        for name,x in (("preceding",previous),("original",original)):
            np.testing.assert_allclose(t,x["time_s"],atol=1e-10,rtol=0)
            other=x["cad-bezier_output_torque_Nm"]
            data[name]=dict(torque_delta_Nm=differences(y-other),
                            changed_load_frames=np.sum((current["cad-bezier_normal_N"]>0)!=(x["cad-bezier_normal_N"]>0),axis=0).tolist())
        if serial_run is not None:
            serial_record,_=verified_dynamics(scene,serial_run,"cad-bezier")
            serial=history(serial_run/"response.csv",scene)
            np.testing.assert_allclose(t,serial["time_s"],atol=1e-10,rtol=0)
            delta=y-serial["output_torque_Nm"]
            peak=int(np.argmax(abs(delta)))
            data["serial_local_boolean"]=dict(process_s=serial_record["process_s"],
                response_sha256=serial_record["response_sha256"],torque_delta_Nm=differences(delta),
                peak_time_s=float(t[peak]),block_Nm=float(y[peak]),serial_Nm=float(serial["output_torque_Nm"][peak]),
                previous_parallel_Nm=float(previous["cad-bezier_output_torque_Nm"][peak]),
                changed_load_frames=((current["cad-bezier_normal_N"]>0)!=(serial["normal_N"]>0)).sum(axis=0).tolist())
            fig,axes=plt.subplots(1,2,figsize=(10,3.3),layout="constrained")
            axes[0].plot(t,delta,lw=.6,color="#267671")
            axes[0].plot(t[peak],delta[peak],"o",ms=4,color="#9b471f")
            axes[0].set(xlabel="Simulation time (s)",ylabel="Block minus serial CAD torque (N m)")
            window=slice(max(0,peak-15),min(len(t),peak+16))
            serial_label=f'{serial_record["process_s"]:.0f} s serial CAD'
            for trace,label,color,style in ((serial["output_torque_Nm"],serial_label,"#9b471f","--"),
                    (y,"Block CAD (ours)","#267671","-")):
                axes[1].plot(t[window],trace[window],style,color=color,lw=1.5,label=label)
            axes[1].set(xlabel="Simulation time (s)",ylabel="Output torque (N m)")
            axes[1].ticklabel_format(axis="x",useOffset=False)
            axes[1].legend(frameon=False)
            for ax in axes: ax.spines[["top","right"]].set_visible(False)
            fig.savefig(out/"serial-torque.png",dpi=165)
            plt.close(fig)
        data["tracked_peaks"]=[dict(step=i,block_Nm=float(y[i]),pointwise_Nm=float(previous["cad-bezier_output_torque_Nm"][i]),
                                original_Nm=float(original["cad-bezier_output_torque_Nm"][i]))
                            for i in sorted({25377,41709,43376,45681,data["preceding"]["torque_delta_Nm"]["peak_step"]})]
    write(out/"trajectory-check.json",data)
    for name in ("cad-bezier","mesh-obb"):
        shutil.copyfile(runs/name/"telemetry.json",out/(name+"-telemetry.json"))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("mesh","dynamics"))
    parser.add_argument("scene", type=Path)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--out",type=Path,default=Path("results/aes/cpu-bezier/blocks"))
    parser.add_argument("--serial-run",type=Path,help="Include the archived serial local-Boolean CAD trajectory")
    a=parser.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    if a.action=="mesh":
        if len(a.paths)!=4: parser.error("Give the two historical runs, same-binary repeat and crossed.json")
        mesh_audit(a.scene,a.paths[:3],a.paths[3],a.out)
    elif len(a.paths)==1:
        dynamics(a.scene,a.paths[0],a.out,a.serial_run)
    else:
        parser.error("Give the directory containing the CAD and mesh runs")
