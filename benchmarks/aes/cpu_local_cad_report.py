"""Summarize geometry replay on the accepted CPU trajectory, including failures."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cpu_local_cad_replay import read_model, records, representatives, load_torque, corrected
from cpu_backend_run import digest


def figures(directory, trace, audit):
    t=trace["time_s"]
    # The overview retains extrema; the full-resolution signals are archived.
    width=max(1,int(round(.01/np.median(np.diff(t)))))
    blocks=[slice(j,min(j+width,len(t))) for j in range(0,len(t),width)]
    centers=np.asarray([t[b].mean() for b in blocks])
    fig,axs=plt.subplots(2,2,figsize=(12,6),layout="constrained")
    for i in range(2):
        for col,(field,units) in enumerate([("load_N","Normal load (kN)"),("torque_Nm","Driven-axis torque (N m)")]):
            ax=axs[i,col]
            for prefix,color,label in [("mesh","#8c5b2d","Mesh OBB"),("local","#167580","STEP, same mesh seeds"),
                                       ("witness","#615387","STEP, deepest verified witness per group")]:
                y=trace[f"{prefix}_{field}"][:,i]/(1000 if col==0 else 1)
                bands=[]
                for block in blocks:
                    local=y[block]
                    valid=local[np.isfinite(local)]
                    bands.append((valid.min(),np.median(valid),valid.max()) if len(valid) else (np.nan,)*3)
                lo,med,hi=np.asarray(bands).T
                ax.fill_between(centers,lo,hi,color=color,alpha=.09,linewidth=0)
                ax.plot(centers,med,color=color,lw=1.1,label=label)
            points=[(t[int(j)],r["native_response"][col]) for j,r in audit[i].items() if r["native_response"] is not None]
            if points:
                x,y=np.asarray(points).T
                ax.scatter(x,y/(1000 if col==0 else 1),s=18,facecolors="white",edgecolors="#20252b",
                           linewidths=.75,label="Native STEP component check",zorder=4)
            ax.set(xlim=(0,5),ylabel=units,title=f"Stage {i+1}",xlabel="Prescribed trajectory time (s)")
            ax.grid(alpha=.18)
    axs[0,0].legend(fontsize=8)
    fig.suptitle("10 ms median and range, with instantaneous STEP component checks",fontsize=12)
    fig.savefig(directory/"fixed-pose-loads.png",dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,3.5),layout="constrained")
    i=1
    for ax,(lo,hi) in zip(axes,[(0,.0032),(.0288,.0306)]):
        for key,label,color in [("mesh_count","Mesh regions","#8c5b2d"),
                                ("native_witnesses","CAD face pairs with a witness","#9689a6"),
                                ("native_groups","CAD groups with a verified interior path","#167580")]:
            v=trace[key][:,i].astype(float)
            v[v<0]=np.nan
            use=(t>=lo)&(t<=hi)
            ax.step(t[use]*1000,v[use],where="mid",label=label,color=color,lw=1.35)
        controls=[(t[int(j)]*1000,r["reference"]["count"]) for j,r in audit[i].items()
                  if r["reference"]["resolved"] and lo<=t[int(j)]<=hi]
        if controls:
            x,y=np.asarray(controls).T
            ax.scatter(x,y,marker="o",facecolor="none",edgecolor="black",s=28,label="Whole STEP reference",zorder=5)
        rejected=(trace["mesh_count"][:,i]<0)&(t>=lo)&(t<=hi)
        for index,time_s in enumerate(t[rejected]):
            ax.axvline(time_s*1000,color="#a74238",lw=1.2,ls="--",label="Mesh query unresolved" if index==0 else None)
        if rejected.any():
            ax.legend(fontsize=7)
        ax.set(xlabel="Prescribed trajectory time (ms)",ylabel="Count",ylim=(-.5,None))
        ax.grid(alpha=.18)
    axes[0].legend(fontsize=7)
    fig.savefig(directory/"stage-two-switches.png",dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(8.2,5.4),sharex=True,layout="constrained")
    startup=t<=.0032
    for col,(ax,field,ylabel,unit) in enumerate(zip(axes,("load_N","torque_Nm"),
            ("Normal load (kN)","Driven-axis torque (N m)"),(1000.,1.),strict=True)):
        for key,label,color in (("mesh","Mesh OBB","#8c5b2d"),("witness","Deepest verified CAD witness","#615387")):
            ax.plot(1000*t[startup],trace[f"{key}_{field}"][startup,1]/unit,color=color,lw=1.25,label=label)
        control=[(1000*t[int(j)],r["native_response"][col]/unit) for j,r in audit[1].items()
                 if startup[int(j)] and r["native_response"] is not None]
        if control:
            x,y=np.asarray(control).T
            ax.scatter(x,y,facecolor="white",edgecolor="#1c303c",s=26,zorder=4,label="Native STEP component check")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Prescribed trajectory time (ms)")
    fig.savefig(directory/"stage-two-loads-detail.png",dpi=160)
    plt.close(fig)


def summarize(args):
    bodies,pairs=read_model(args.scene)
    with np.load(args.directory/"trajectory.npz") as src:
        t=src["time_s"]
        origins=src["origins_m"]
    n=len(t)
    trace={"time_s":t}
    for field in ("mesh_count","native_witnesses","native_groups","tracked","local_failed","native_regroup_failed",
                  "mesh_query_s","local_query_s","native_query_s","group_query_s","mesh_load_N","local_load_N",
                  "mesh_torque_Nm","local_torque_Nm","witness_load_N","witness_torque_Nm"):
        trace[field]=np.full((n,len(pairs)),np.nan)
    summary=dict(frames=n,scene_sha256=digest(args.scene),trajectory_sha256=digest(args.directory/"trajectory.npz"),stages=[])
    audits=[]
    for i,(a,b,k,limit) in enumerate(pairs):
        file=lambda suffix:(args.work/str(i)).with_suffix(suffix)
        info=json.loads(file(".audit.json").read_text())
        audited=info["rows"]
        audits.append(audited)
        fixed=json.loads(file(".fixed.json").read_text())
        timings={}
        fixed_stats={}
        for source in sorted({r["seed"] for r in fixed}):
            rows=[r for r in fixed if r["seed"]==source]
            valid=[r for r in rows if r["local"]["depth_m"] is not None and r["whole"]["status"]=="resolved"]
            fixed_stats[source]=dict(total=len(rows),resolved=len(valid),
                max_depth_difference_um=max(abs(r["local"]["depth_m"]-r["whole"]["depth_m"])*1e6 for r in valid),
                max_exit_difference_um=max(max(abs(x-y)*1e6 for x,y in zip(r["local"]["offsets_m"],r["whole"]["offsets_m"],strict=True)) for r in valid),
                median_local_ms=float(np.median([r["local"]["query_s"] for r in rows])*1000),
                median_whole_ms=float(np.median([r["whole"]["query_s"] for r in rows])*1000))
        sources=[records(file(s)) for s in (".jsonl",".local",".integrated")]
        totals=0
        for j,(mesh,local,native) in enumerate(zip(*sources,strict=True)):
            totals+=1
            regs=mesh["regions"]
            revised=corrected(regs,local)
            labels=native["labels"]
            if len(labels)!=len(native["witnesses"]):
                raise ValueError("native witness order or length differs")
            for key,value in dict(mesh_count=-1 if regs is None else len(regs),native_witnesses=len(labels),
                native_groups=len(set(labels)-{-1}),tracked=native["tracked"],
                local_failed=sum(c["depth_m"] is None for c in local),native_regroup_failed=labels.count(-1),
                mesh_query_s=mesh["query_s"],local_query_s=sum(c["query_s"] for c in local),
                native_query_s=native["query_s"]-native["group_s"],group_query_s=native["group_s"]).items():
                trace[key][j,i]=value
            estimated=representatives([s[2:] for s in native["witnesses"]],labels)
            for name,r in (("mesh",regs),("local",revised),("witness",estimated)):
                load,torque=load_torque(r,k,origins[j,b],bodies[b][2],limit)
                trace[name+"_load_N"][j,i]=load
                trace[name+"_torque_Nm"][j,i]=torque
            if str(j) in audited:
                row=audited[str(j)]
                if row["native"] != [s[2:] for s in native["witnesses"]]:
                    raise ValueError(f"membership audit witnesses differ at stage {i+1}, frame {j}")
                row["native_group_labels"]=labels
                control=row["cad_regions"]
                response=load_torque(control["regions"] if control["resolved"] else None,k,origins[j,b],bodies[b][2],limit)
                row["native_response"]=list(response) if np.isfinite(response).all() else None
                proxy=load_torque(estimated,k,origins[j,b],bodies[b][2],limit)
                row["witness_response"]=list(proxy) if np.isfinite(proxy).all() else None
        if totals!=n:
            raise ValueError("incomplete replay")
        for field in ("mesh","local","native","group"):
            sec=trace[field+"_query_s"][:,i]
            timings[field]=dict(sum_s=float(sec.sum()),mean_ms=float(sec.mean()*1000),p95_ms=float(np.percentile(sec,95)*1000))
        checks=[]
        for step,row in audited.items():
            ref=row["reference"]
            if not ref["resolved"]:
                continue
            physical=set(range(ref["count"]))
            labels=row["native_labels"]
            mapping={}
            for group,component in zip(row["native_group_labels"],labels,strict=True):
                if group>=0:
                    mapping.setdefault(group,set()).add(component)
            exact=(len(mapping)==ref["count"] and all(len(v)==1 for v in mapping.values())
                   and set(labels)==physical and set().union(*mapping.values())==physical)
            checks.append(dict(step=int(step),count=ref["count"],mesh_count=None if row["mesh"] is None else len(row["mesh"]),
                mesh_covered=len(physical.intersection(row["mesh_labels"])),native_covered=len(physical.intersection(labels)),
                groups=len(mapping),exact_groups=exact,false_join=any(len(s & physical)>1 for s in mapping.values()),
                unclassified=sum(label<0 for label in labels)))
        # Compare the same loaded, independently audited poses for each strategy.
        differences={name:[] for name in ("mesh","local","witness")}
        for step,row in audited.items():
            ref=row["native_response"]
            if ref is None or ref[0]<=0:
                continue
            errors={name:np.asarray([trace[f"{name}_load_N"][int(step),i],trace[f"{name}_torque_Nm"][int(step),i]])-ref
                    for name in differences}
            if all(np.isfinite(d).all() for d in errors.values()):
                for name,d in errors.items():
                    differences[name].append(d)
        response_errors={name:dict(poses=len(values),rms_load_N=float(np.mean(np.square(values),axis=0)[0]**.5),
                                   rms_torque_Nm=float(np.mean(np.square(values),axis=0)[1]**.5))
                         for name,values in differences.items() if values}
        normal_results=json.loads(file(".normals.json").read_text())
        normal_errors={name:[] for name in ("deepest","opposed_source","opposed_bisector")}
        for row in normal_results:
            control=audited[str(row["step"])]
            responses={"deepest":control["witness_response"],"opposed_source":row["opposed_source"],
                       "opposed_bisector":row["opposed_bisector"]}
            ref=control["native_response"]
            if ref is not None and ref[0]>0 and all(v is not None for v in responses.values()):
                for name,v in responses.items():
                    normal_errors[name].append(np.asarray(v)-ref)
        normal_scores={name:dict(poses=len(values),rms_load_N=float(np.mean(np.square(values),axis=0)[0]**.5),
                                 rms_torque_Nm=float(np.mean(np.square(values),axis=0)[1]**.5))
                       for name,values in normal_errors.items() if values}
        swap=json.loads(file(".swap.json").read_text())
        swapped=list(zip(swap["forward"],swap["reverse"],strict=True))
        summary["stages"].append(dict(stage=i+1,geometry_sha256={p.name:digest(p) for j in (a,b)
            for p in (bodies[j][0],bodies[j][0].with_suffix(".msh"))},fixed=fixed_stats,geometry_times=timings,
            mesh_unresolved_frames=int(sum(trace["mesh_count"][:,i]<0)),local_unresolved_points=int(sum(trace["local_failed"][:,i])),
            native_regroup_unresolved_points=int(sum(trace["native_regroup_failed"][:,i])),
            native_witnesses=int(trace["native_witnesses"][:,i].sum()),previous_site_confirmed=int(trace["tracked"][:,i].sum()),
            process_wall_s=json.loads(file(".times").read_text()),audited_component_checks=checks,audited_response_errors=response_errors,
            representative_comparison=normal_scores,
            role_swap=dict(poses=len(swapped),group_mismatches=sum(x["groups"]!=y["groups"] for x,y in swapped),
                max_load_difference_N=max(abs(x["load_N"]-y["load_N"]) for x,y in swapped if None not in (x["load_N"],y["load_N"])),
                max_torque_difference_Nm=max(abs(x["torque_Nm"]-y["torque_Nm"]) for x,y in swapped if None not in (x["torque_Nm"],y["torque_Nm"])))))
        info["role_swap"]=swap
        info["representative_trials"]=normal_results
        summary["stages"][-1]["unresolved_seed_reference"]=json.loads(file(".failed.json").read_text())
        if file(".binaries").exists():
            summary["stages"][-1]["geometry_executable_sha256"]=json.loads(file(".binaries").read_text())
        (args.directory/f"stage-{i+1}-fixed.json").write_text(json.dumps(fixed,indent=1,allow_nan=False)+"\n")
        (args.directory/f"stage-{i+1}-audit.json").write_text(json.dumps(info,indent=1,allow_nan=False)+"\n")
    np.savez_compressed(args.directory/"replay.npz",**trace)
    summary["stage_pair_query_times_s"]={}
    for name,cost in (("mesh",trace["mesh_query_s"]),("mesh_plus_local",trace["mesh_query_s"]+trace["local_query_s"]),
                      ("native_with_grouping",trace["native_query_s"]+trace["group_query_s"])):
        parallel=cost.max(axis=1)
        summary["stage_pair_query_times_s"][name]=dict(sum_stage_times_s=float(cost.sum()),
            sum_per_frame_max_s=float(parallel.sum()),mean_per_frame_max_ms=float(parallel.mean()*1000),
            p95_per_frame_max_ms=float(np.percentile(parallel,95)*1000))
    figures(args.directory,trace,audits)
    (args.directory/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("scene","work","directory"):
        parser.add_argument(arg,type=Path)
    summarize(parser.parse_args())
