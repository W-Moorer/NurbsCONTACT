"""Replay reducer Boolean events and compare two complete 0.1 ms CPU duty cycles."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _gpu_source import read_model
from cpu_backend_run import digest
from cpu_bezier_diagnosis import predictor_poses, query
from cpu_cad_pressure_report import history, wrench
from cpu_cad_pruning import verified_dynamics


def write(path, values):
    path.write_text(json.dumps(values, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def integrate_groups(cmd, rows, order):
    # Each connected region has a different number of discovery chords.
    data="".join(" ".join(format(v,".17g") for v in r)+"\n" for r in rows)
    process=subprocess.run([*cmd,"--patch-wrench",str(order),"--bezier"],input=data,text=True,
                           capture_output=True,timeout=1800,check=True)
    return [json.loads(s) for s in process.stdout.splitlines()]


def diagnose(scene, previous, probes, out, reference=None):
    manifest, _ = verified_dynamics(scene, previous, "cad-bezier")
    progress = np.genfromtxt(previous/"progress.csv", names=True, delimiter=",", usecols=("cad_booleans",))
    steps = np.flatnonzero(np.diff(progress["cad_booleans"], prepend=0))
    trace = np.genfromtxt(previous/"response.csv", names=True, delimiter=",",
                          usecols=[k+str(j) for k in "qva" for j in range(3)])
    bodies, pairs = read_model(scene)
    cases,refinement,controls=[],[],[]
    for stage, pair in enumerate(pairs, 1):
        a, b, _, limit = pair
        indices = np.concatenate([np.arange(max(0,s-10),s) for s in steps])
        ends = np.cumsum(np.minimum(steps,10))-1
        poses = predictor_poses(trace[indices], bodies, pair, manifest["dt_s"])
        cmd = [probes,bodies[a][0],bodies[b][0],"6",str(2*limit)]
        rows = query([*cmd,"--loaded","--bezier"],poses)
        inputs,groups=[],[]
        for step,end in zip(steps,ends,strict=True):
            r=rows[end]
            cases.append(dict(step=int(step),stage=stage,groups=len(set(r["labels"])),
                              witnesses=len(r["witnesses"]),failure=r["failure"],resolved=r["resolved"]))
            if r["failure"]=="no_columns":
                for label in sorted(set(r["labels"])):
                    w=[w for g,w in zip(r["labels"],r["witnesses"],strict=True) if g==label]
                    inputs.append([*poses[end],len(w),*[v for hit in w for v in (label,*hit[:8])]])
                    groups.append((int(step),label,max(hit[-1] for hit in w)*1e6))
        if inputs:
            coarse=integrate_groups(cmd,inputs,96)
            fine=integrate_groups(cmd,inputs,192)
            for (step,label,depth),x,y in zip(groups,coarse,fine,strict=True):
                if not x["resolved"]:
                    refinement.append(dict(stage=stage,step=step,label=label,witness_depth_um=depth,
                        resolved_at_192=y["resolved"],integrated_depth_um=y["patches"][0][-1]*1e6 if y["resolved"] else None))
            if reference:
                for step in (1860,2741,9317):
                    use=[i for i,row in enumerate(groups) if row[0]==step]
                    if not use: continue
                    regular=integrate_groups(cmd,[inputs[i] for i in use],0)
                    mixed=[x if x["resolved"] else fine[i] for i,x in zip(use,regular,strict=True)]
                    full=query([reference,bodies[a][0],bodies[b][0],"--pressure-patches",str(2*limit)],
                               [inputs[use[0]][:14]])[0]
                    if not full["resolved"] or not all(x["resolved"] for x in mixed):
                        raise ValueError("Unresolved same-pose force control")
                    def torque(regions):
                        return wrench(regions,bodies[b][1],bodies[b][2],pair[2],limit,True)[-1]
                    controls.append(dict(stage=stage,step=step,whole_Nm=torque(full["patches"]),
                                         mixed_Nm=torque([x["patches"][0] for x in mixed]),
                                         whole_groups=len(full["patches"]),mixed_groups=len(mixed)))
    write(out/"fallback-replay.json",dict(scene_sha256=digest(scene),old_trace_sha256=manifest["response_sha256"],
             probe_sha256=digest(probes),cases=cases,refinement=refinement,controls=controls,
             reference_sha256=digest(reference) if reference else None))


def differences(y):
    idx=int(np.argmax(abs(y)))
    return dict(rms=float(np.sqrt(np.mean(y*y))),max_abs=float(abs(y[idx])),peak_step=idx)


def fixed_geometry_inputs(scene,archive):
    replay=json.loads((archive/"replay.json").read_text())
    trajectory=archive.parent/"cpu-local-cad/trajectory.npz"
    if digest(trajectory)!=replay["trajectory_sha256"] or digest(scene)!=replay["scene_sha256"]:
        raise ValueError("fixed-pose inputs differ from the archived comparison")
    bodies,pairs=read_model(scene)
    indices=np.concatenate([np.arange(s,s+100) for s in replay["starts"]])
    warm=np.arange(len(indices))%100>=20
    with np.load(trajectory) as data:
        angles,origins=data["angles_rad"],data["origins_m"]
    return trajectory,indices,warm,bodies,pairs,angles,origins


def stage_poses(bodies,angles,origins,indices,a,b):
    columns=[col for i in (a,b) for col in (*origins[:,i].T,
             *(np.full(len(angles),v) for v in bodies[i][2]),angles[:,i])]
    return np.column_stack(columns)[indices]


def paired_geometry(scene, previous, current, out, archive=None, report_all=False, current_flags=(), reverse_order=False, previous_flags=()):
    archive=out.parent if archive is None else archive
    trajectory,indices,warm,bodies,pairs,angles,origins=fixed_geometry_inputs(scene,archive)
    result=dict(probes_sha256=[digest(previous),digest(current)],trajectory_sha256=digest(trajectory),stages=[])
    if current_flags:
        result.update(current_flags=list(current_flags),reverse_order=reverse_order,scene_sha256=digest(scene))
    if previous_flags:
        result["previous_flags"]=list(previous_flags)
    for stage,(a,b,stiffness,limit) in enumerate(pairs,1):
        poses=stage_poses(bodies,angles,origins,indices,a,b)
        runs={}
        order=(("previous",previous),("current",current))
        for name,exe in order if (stage==1)!=reverse_order else reversed(order):
            flags=current_flags if name=="current" else previous_flags
            runs[name]=query([exe,bodies[a][0],bodies[b][0],"6",str(2*limit),"--loaded","--bezier",*flags],poses)
        face_switches=0
        former=None
        torque_delta=[]
        wrench_error=np.zeros(3)
        changed_states=[]
        changed_geometry=[]
        max_error=np.zeros(4)
        def grouped_faces(row):
            groups={}
            for label,hit in zip(row["labels"],row["witnesses"],strict=True):
                groups.setdefault(label,set()).add(tuple(hit[:2]))
            return sorted(tuple(sorted(faces)) for faces in groups.values())
        for frame,(x,y) in enumerate(zip(runs["previous"],runs["current"],strict=True)):
            current_faces=grouped_faces(y)
            same=x["resolved"]==y["resolved"] and x["labels"]==y["labels"] and grouped_faces(x)==current_faces
            if not same:
                changed_states.append(int(indices[frame]))
                if not report_all:
                    raise ValueError("fixed-pose result or grouping changed")
            if frame%100 and current_faces!=former:
                face_switches+=1
            former=current_faces
            if same and x["regions"] and len(x["regions"])==len(y["regions"]):
                error=abs(np.asarray(x["regions"])-np.asarray(y["regions"]))
                max_error=np.maximum(max_error,[error[:,:3].max(),error[:,3:6].max(),error[:,6:9].max(),error[:,-1].max()])
            if same and (len(x["regions"])!=len(y["regions"]) or not np.allclose(x["regions"],y["regions"],rtol=1e-8,atol=1e-12)):
                changed_geometry.append(int(indices[frame]))
                if not report_all:
                    raise ValueError("fixed-pose pressure geometry changed")
            if x["resolved"] and y["resolved"]:
                moments=[wrench(row["regions"],origins[indices[frame],b],bodies[b][2],stiffness,limit,True)
                         for row in (x,y)]
                if all(m is not None for m in moments):
                    diff=moments[1]-moments[0]
                    torque_delta.append(diff[-1])
                    wrench_error=np.maximum(wrench_error,[abs(diff[0]),np.linalg.norm(diff[1:4]),np.linalg.norm(diff[4:7])])
        entry=dict(stage=stage,checked=len(poses),timed=int(warm.sum()),face_group_changes=face_switches,
                   compared_torques=len(torque_delta),max_torque_delta_Nm=float(max(map(abs,torque_delta),default=0.)),
                   changed_states=changed_states,changed_geometry=changed_geometry,
                   max_component_errors=dict(zip(("point_m","unit_normal","arm_m","depth_m"),max_error.tolist())))
        if current_flags:
            entry["max_wrench_errors"]=dict(zip(("total_normal_N","resultant_N","moment_Nm"),wrench_error.tolist()))
        for name,rows in runs.items():
            area=1e3*np.array([r["loaded_s"]-r["query_s"] for r in rows])[warm]
            total=1e3*np.array([r["loaded_s"] for r in rows])[warm]
            entry[name]=dict(area_mean_ms=float(area.mean()),area_median_ms=float(np.median(area)),
                             full_query_mean_ms=float(total.mean()))
            entry[name]["ray_counts"]=rows[-1]["ray_counts"]
            if "area_rays" in rows[-1]:
                entry[name]["area_ray_counts"]=np.sum([r["area_rays"] for r in rows],axis=0).tolist()
            if "depth_bounds" in rows[-1]:
                keys=("interval_tests","excluded_sites","root_refinements","accepted_depths","empty_blocks","interior_blocks")
                entry[name]["depth_bounds"]=dict(zip(keys,rows[-1]["depth_bounds"],strict=True))
                if "depth_site_box_tests" in rows[-1]:
                    entry[name]["depth_bounds"]["site_box_tests"]=rows[-1]["depth_site_box_tests"]
            if "block_counts" in rows[-1]:
                entry[name]["full_blocks"],entry[name]["partial_blocks"],entry[name]["shared_columns"]=rows[-1]["block_counts"]
            if "column_windows" in rows[-1]:
                entry[name]["column_windows_built"],entry[name]["column_windows_reused"]=rows[-1]["column_windows"]
            if "strip_checks" in rows[-1]:
                entry[name]["strip_checks"]=dict(zip(("eligible","parallel_source","invariant_target","matched_ends"),rows[-1]["strip_checks"],strict=True))
        result["stages"].append(entry)
    write(out/"fixed-timing.json",result)


def report(scene, runs, out, previous=None):
    native_dir=out.parent/"recurring"
    old_dir=previous if previous is not None else native_dir
    traces,metrics={},{}
    for method in ("cad-bezier","mesh-obb"):
        path=runs/method
        record,_=verified_dynamics(scene,path,method)
        shutil.copyfile(path/"run.json",out/(method+"-run.json"))
        traces[method]=history(path/"response.csv",scene)
        stats=np.genfromtxt(path/"progress.csv",names=True,delimiter=",",usecols=("cad_booleans","integration_s"))
        events=np.flatnonzero(np.diff(stats["cad_booleans"],prepend=0))
        elapsed=np.diff(stats["integration_s"],prepend=0)
        regular=np.ones(len(elapsed),dtype=bool)
        regular[0]=False
        regular[events]=False
        ms=elapsed[regular]*1000
        fixed=record["process_s"]-ms.sum()/1000
        metrics[method]=dict(process_s=record["process_s"],boolean_steps=events.tolist(),
                             wall_in_boolean_steps_s=float(sum(elapsed[events])),
                             ordinary_steps=int(regular.sum()),ordinary_mean_ms=float(ms.mean()),
                             ordinary_p50_ms=float(np.median(ms)),ordinary_p95_ms=float(np.percentile(ms,95)),
                             budget_per_ordinary_step_ms_for_300s=float((300-fixed)*1000/len(ms)),
                             unchanged_trace_sha256=record["response_sha256"]==json.loads(
                                 (old_dir/(method+"-run.json")).read_text())["response_sha256"])
    if json.loads((out/"cad-bezier-run.json").read_text())["executable_sha256"] != json.loads((out/"mesh-obb-run.json").read_text())["executable_sha256"]:
        raise ValueError("CAD and mesh used different binaries")
    cad,mesh=traces.values()
    t=cad["time_s"]
    torque=cad["output_torque_Nm"]
    with np.load(old_dir/"responses.npz") as old, np.load(out.parent/"responses.npz") as archive:
        np.testing.assert_allclose(t,old["time_s"],atol=1e-10,rtol=0)
        np.testing.assert_allclose(t,mesh["time_s"],atol=1e-10,rtol=0)
        before=old["cad-bezier_output_torque_Nm"]
        delta=torque-before
        metrics["torque_delta_from_previous_Nm"]=differences(delta)
        metrics["torque_delta_from_mesh_Nm"]=differences(torque-mesh["output_torque_Nm"])
        metrics["speed_delta_from_mesh_rad_s"]=[differences(a-b) for a,b in zip(cad["speed"].T,mesh["speed"].T,strict=True)]
        metrics["loads_delta_from_previous_N"]=[differences(a-b) for a,b in zip(cad["normal_N"].T,old["cad-bezier_normal_N"].T,strict=True)]
        metrics["speed_delta_from_previous_rad_s"]=[differences(a-b) for a,b in zip(cad["speed"].T,old["cad-bezier_speed"].T,strict=True)]
        metrics["switch_steps"]={}
        for key,loads in (("previous",old["cad-bezier_normal_N"]),("cad-bezier",cad["normal_N"]),("mesh-obb",mesh["normal_N"])):
            metrics["switch_steps"][key]=[(1+np.flatnonzero(np.diff(x>0))).tolist() for x in loads.T]
        metrics["changed_loaded_frames_vs_previous"]=(np.sum(
            (cad["normal_N"]>0)!=(old["cad-bezier_normal_N"]>0),axis=0)).tolist()
        trigger_steps=np.asarray(metrics["cad-bezier"]["boolean_steps"],dtype=int)
        metrics["fallback_torque_delta_from_previous_Nm"]=[
            dict(step=int(i),same_step_Nm=float(delta[i]),
                 max_over_next_10_steps_Nm=float(np.max(abs(delta[i:i+11])))) for i in trigger_steps]
        for step in (1860,25377,46015):
            metrics.setdefault("tracked_steps",{})[step]=dict(
                output_torque_Nm=float(torque[step]),previous_output_torque_Nm=float(before[step]),
                mesh_output_torque_Nm=float(mesh["output_torque_Nm"][step]))
        for key,series in traces.items():
            metrics[key]["torque_minus_recurdyn_Nm"]=differences(series["output_torque_Nm"]-archive["RecurDyn_output_torque_Nm"])
        with np.load(native_dir/"responses.npz") as reference:
            prefix=reference["cad_patch_reconstructed_output_torque_Nm"]
        window=slice(25365,25406)
        residual=torque[window]-prefix[window]
        metrics["original_spike_window"]=dict(max_abs_vs_native_Nm=float(np.nanmax(abs(residual))),
                at_2_5377_s_vs_native_Nm=float(torque[25377]-prefix[25377]), checked=int(np.isfinite(residual).sum()))
        fig,axes=plt.subplots(3,2,figsize=(12,8),layout="constrained")
        colors=("#197971","#9a642d","#4e5460")
        def plot(ax,x,y,label,color):
            ax.plot(x,y,label=label,color=color,lw=.7)
        for k,vals in enumerate(((torque,mesh["output_torque_Nm"],archive["RecurDyn_output_torque_Nm"]),
                                (cad["speed"][:,1],mesh["speed"][:,1],archive["RecurDyn_speed"][:,1]),
                                (cad["normal_N"][:,0],mesh["normal_N"][:,0]),
                                (cad["normal_N"][:,1],mesh["normal_N"][:,1]))):
            for y,label,color in reversed(list(zip(vals,("Bézier CAD (ours)","Mesh OBB","RecurDyn"),colors))):
                plot(axes.flat[k],t,y,label,color)
        plot(axes[2,0],t,delta,"Current minus previous CAD",colors[0])
        axes[2,0].scatter(t[trigger_steps],delta[trigger_steps],s=13,facecolors="none",
                          edgecolors=colors[1],linewidths=.7,label="Local Boolean steps")
        plot(axes[2,1],t[window],before[window]-prefix[window],"Previous minus native",colors[2])
        plot(axes[2,1],t[window],residual,"Current minus native",colors[0])
        for ax,label in zip(axes.flat,("Output torque (N m)","Intermediate speed (rad/s)","Stage 1 normal load (N)",
                            "Stage 2 normal load (N)","Torque difference (N m)","Torque difference (N m)")):
            ax.set(xlabel="Simulation time (s)",ylabel=label)
            ax.legend(frameon=False,fontsize=8)
            ax.spines[["top","right"]].set_visible(False)
        axes[2,1].ticklabel_format(axis="x",useOffset=False)
        fig.savefig(out/"reducer.png",dpi=165)
        plt.close(fig)
        # Separate the initial load switches and the largest torque deviation
        # from the dense 50,001-frame traces. All samples are raw.
        fig,ax=plt.subplots(1,3,figsize=(12.3,3.3),layout="constrained")
        start=t<=.00261
        for offset,series,label,color in ((1.5,cad,"CAD",colors[0]),(0.,mesh,"Mesh",colors[1])):
            active=(series["normal_N"][start,1]>0).astype(float)
            ax[0].step(t[start]*1000,offset+active,where="post",label=label,color=color,lw=1.6)
        ax[0].set(xlabel="Time from startup (ms)",title="Stage 2 load switches",
                  yticks=[0,1,1.5,2.5],yticklabels=["Mesh off","Mesh on","CAD off","CAD on"],ylim=(-.18,2.68))
        peak=metrics["torque_delta_from_previous_Nm"]["peak_step"]
        win=slice(max(0,peak-13),min(len(t),peak+14))
        for values,label,color,style in ((before,"Previous CAD",colors[2],"--"),
                    (mesh["output_torque_Nm"],"Mesh OBB",colors[1],":"),(torque,"Bézier CAD (ours)",colors[0],"-")):
            ax[1].plot(t[win],values[win],style,color=color,lw=1.2,label=label)
        ax[1].set(title="Output torque near largest change",xlabel="Simulation time (s)",ylabel="Torque (N m)")
        ax[1].legend(frameon=False,fontsize=8)
        ax[2].plot(t[win],delta[win],".-",color=colors[0],lw=1.2,ms=3)
        ax[2].axhline(0,lw=.6,color=colors[2])
        ax[2].set(title="Current minus previous CAD",xlabel="Simulation time (s)",ylabel="Torque difference (N m)")
        for a in ax:
            a.spines[["top","right"]].set_visible(False)
            a.ticklabel_format(axis="x",useOffset=False)
        fig.savefig(out/"switches.png",dpi=165)
        plt.close(fig)
    np.savez_compressed(out/"responses.npz",time_s=t,**{method+"_"+k:v for method,values in traces.items()
                        for k,v in values.items() if k!="time_s"})
    write(out/"comparison.json",metrics)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("report","diagnose","paired"))
    p.add_argument("scene",type=Path)
    p.add_argument("runs",type=Path)
    p.add_argument("--probes",type=Path)
    p.add_argument("--reference",type=Path,help="Independent whole-STEP pressure benchmark for three fallback poses")
    p.add_argument("--depth-bounds",action="store_true",help="Evaluate experimental bounded depth packets in the current probe")
    p.add_argument("--relocate",action="store_true",help="Compare bounded CAD with and without span relocation")
    p.add_argument("--archive",type=Path,help="Parent directory of the original fixed-pose replay manifest")
    p.add_argument("--reverse-order",action="store_true",help="Reverse the two programs in the fixed-pose timing control")
    p.add_argument("--out",type=Path,default=Path("results/aes/cpu-bezier/columns"))
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    if a.action=="diagnose":
        if a.probes is None: p.error("diagnose requires --probes")
        diagnose(a.scene,a.runs,a.probes,a.out,a.reference)
    elif a.action=="paired":
        if a.probes is None: p.error("paired requires the previous probe as runs and the current --probes")
        previous=("--depth-bounds",) if a.relocate else ()
        current=("--depth-bounds","--relocate") if a.relocate else ("--depth-bounds",) if a.depth_bounds else ()
        paired_geometry(a.scene,a.runs,a.probes,a.out,archive=a.archive,report_all=bool(current),
                        previous_flags=previous,current_flags=current,reverse_order=a.reverse_order)
    else:
        report(a.scene,a.runs,a.out)
