"""Compare the three complete, independently timed CPU reducer trajectories."""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cpu_cad_pressure_report import history
from cpu_cad_pruning import verified_dynamics
from cpu_pressure_columns import differences, write


def compare(scene, runs, archive, out, relocate=False):
    out.mkdir(parents=True, exist_ok=True)
    methods=("cad-bezier-bounds","cad-bezier","mesh-obb")
    if relocate: methods=("cad-bezier-relocate","cad-bezier-bounds","mesh-obb")
    current_label="Relocated CAD (ours)" if relocate else "Bounded CAD (ours)"
    traces={}
    metrics={"runs":{}}
    for method in methods:
        path=runs/method
        record,_=verified_dynamics(scene,path,method)
        traces[method]=history(path/"response.csv",scene)
        stats=record["statistics"]
        boolean_counter="mesh_booleans" if method=="mesh-obb" else "cad_booleans"
        progress=np.genfromtxt(path/"progress.csv",delimiter=",",names=True,
                               usecols=("integration_s",boolean_counter))
        times=np.diff(progress["integration_s"])
        fallback=np.diff(progress[boolean_counter])>0
        ordinary=times[~fallback]
        samples=json.loads((path/"telemetry.json").read_text())
        performance=[s["performance_pct"] for s in samples if "performance_pct" in s]
        metrics["runs"][method]=dict(process_s=record["process_s"],exe_sha256=record["executable_sha256"],
                ordinary_mean_ms=1000*float(ordinary.mean()),ordinary_p95_ms=1000*float(np.percentile(ordinary,95)),
                boolean_steps=(1+np.flatnonzero(fallback)).tolist(),boolean_step_wall_s=float(times[fallback].sum()),
                half_second_integration_s=np.diff(progress["integration_s"][::5000]).tolist(),
                host_performance_median_pct=float(np.median(performance)) if performance else None,
                area_pair_s=float(stats["cad_patch_area_s"]),setup_s=float(stats["setup_s"]),
                loaded_switch_steps=[(1+np.flatnonzero(np.diff(x>0))).tolist() for x in traces[method]["normal_N"].T])
        shutil.copyfile(path/"run.json",out/(method+"-run.json"))
        shutil.copyfile(path/"telemetry.json",out/(method+"-telemetry.json"))
        original=archive/("depth-bounds" if relocate else "blocks")/(method+"-run.json")
        if original.is_file():
            archived=json.loads(original.read_text())
            metrics["runs"][method]["same_response_as_archived_control"]=record["response_sha256"]==archived["response_sha256"]
    if len({r["exe_sha256"] for r in metrics["runs"].values()})!=1:
        raise ValueError("The three runs used different executables")
    new,prior,mesh=traces.values()
    t=new["time_s"]
    np.testing.assert_allclose(t,np.arange(50001)*.0001,atol=1e-10,rtol=0)
    for trace in (prior,mesh): np.testing.assert_allclose(t,trace["time_s"],atol=1e-10,rtol=0)
    with np.load(archive/"responses.npz") as rd:
        reference=rd["RecurDyn_output_torque_Nm"]
        reference_speed=rd["RecurDyn_speed"][:,1]
        np.testing.assert_allclose(t,rd["time_s"],atol=1e-10,rtol=0)
    torque=new["output_torque_Nm"]
    residual=torque-prior["output_torque_Nm"]
    metrics["output_torque_delta_Nm"]=differences(residual)
    metrics["loaded_mismatches_vs_previous"]=((new["normal_N"]>0)!=(prior["normal_N"]>0)).sum(axis=0).tolist()
    metrics["normal_load_delta_N"]=[differences(x) for x in (new["normal_N"]-prior["normal_N"]).T]
    metrics["shaft_speed_delta_rad_s"]=[differences(x) for x in (new["speed"]-prior["speed"]).T]
    if relocate:
        for method,trace in traces.items():
            values=trace["output_torque_Nm"]
            metrics["runs"][method]["torque_range_Nm"]=[float(values.min()),float(values.max())]
            metrics["runs"][method]["max_consecutive_torque_change_Nm"]=float(np.max(abs(np.diff(values))))
        metrics["output_torque_delta_vs_mesh_Nm"]=differences(torque-mesh["output_torque_Nm"])
    for name,trace in traces.items():
        metrics["runs"][name]["torque_vs_recurdyn_Nm"]=differences(trace["output_torque_Nm"]-reference)
        metrics["runs"][name]["intermediate_speed_vs_recurdyn_rad_s"]=differences(trace["speed"][:,1]-reference_speed)
        metrics["runs"][name]["output_torque_at_0_5ms_Nm"]=float(trace["output_torque_Nm"][5])
    with np.load(archive/"recurring/responses.npz") as native:
        prefix=native["cad_patch_reconstructed_output_torque_Nm"]
        stop=slice(25365,25406)
        native_delta=torque[stop]-prefix[stop]
        native_window=prefix[stop].copy()
        if not np.isfinite(native_delta).all(): raise ValueError("Missing native spike-window samples")
        metrics["native_window"]=dict(at_2_5377_s_Nm=float(torque[25377]-prefix[25377]),
                                      max_abs_Nm=float(np.max(abs(native_delta))),frames=len(native_delta))
    metrics["frames"]=len(t)
    write(out/"comparison.json",metrics)
    np.savez_compressed(out/"responses.npz",time_s=t,**{name+"_"+k:v for name,h in traces.items()
                        for k,v in h.items() if k!="time_s"})
    fig,axes=plt.subplots(3,2,figsize=(11,8),layout="constrained")
    colors=("#187970","#656c7d","#ad6d2e")
    def overlay(ax,x,field,column=None,slice_=slice(None)):
        for name,label,color,style in reversed(list(zip(methods,(current_label,"Previous CAD","Mesh OBB"),
                                                       colors,("-","--",":")))):
            y=traces[name][field]
            if column is not None: y=y[:,column]
            ax.plot(x[slice_],y[slice_],style,color=color,lw=.85,label=label)
    axes[0,0].plot(t,reference,color="#999999",lw=.6,label="RecurDyn")
    axes[0,1].plot(t,reference_speed,color="#999999",lw=.6,label="RecurDyn")
    overlay(axes[0,0],t,"output_torque_Nm")
    overlay(axes[0,1],t,"speed",1)
    for i in (0,1): overlay(axes[1,i],t,"normal_N",i)
    axes[2,0].plot(t,residual,color=colors[0],lw=.7)
    peak=metrics["output_torque_delta_Nm"]["peak_step"]
    zoom=slice(max(0,peak-15),min(len(t),peak+16))
    overlay(axes[2,1],t,"output_torque_Nm",slice_=zoom)
    labels=("Output torque (N m)","Intermediate shaft (rad/s)","Stage 1 normal load (N)",
            "Stage 2 normal load (N)","Current minus previous torque (N m)","Torque near largest difference (N m)")
    for ax,label in zip(axes.flat,labels):
        ax.set(xlabel="Simulation time (s)",ylabel=label)
        if ax.get_legend_handles_labels()[0]: ax.legend(frameon=False,fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
    axes[2,1].ticklabel_format(axis="x",useOffset=False)
    fig.savefig(out/"reducer.png",dpi=165)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout="constrained")
    overlay(axes[0],t*1e3,"normal_N",1,slice(0,31))
    axes[0].set(xlabel="Simulation time (ms)",ylabel="Stage 2 normal load (N)")
    # The native CAD comparison has exactly 41 frames in the earlier spike window.
    elapsed_ms=(t-t[25377])*1e3
    axes[1].plot(elapsed_ms[stop],native_window,"o",mfc="none",ms=4,
                 lw=.8,color="#30353b",label="Native CAD reference")
    for name,label,style in ((methods[1],"Previous CAD","--"),(methods[0],current_label,"-")):
        axes[1].plot(elapsed_ms[stop],traces[name]["output_torque_Nm"][stop],style,
                     color=colors[methods.index(name)],lw=1.1,label=label)
    axes[1].set(xlabel="Time relative to 2.5377 s (ms)",ylabel="Output shaft torque (N m)")
    for ax in axes:
        ax.legend(frameon=False,fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
    fig.savefig(out/"switches.png",dpi=165)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("scene","runs","archive","out"): p.add_argument(key,type=Path)
    p.add_argument("--relocate",action="store_true")
    compare(**vars(p.parse_args()))
