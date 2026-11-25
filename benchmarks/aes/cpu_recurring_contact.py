"""Fixed-pose and five-second checks of material-normal pressure and recurring CAD candidates."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _gpu_source import read_model
from cpu_backend_run import digest
from cpu_bezier_diagnosis import query, swap_witness
from cpu_cad_pressure_report import history, smooth_loads, wrench
from cpu_cad_pruning import verified_dynamics


def write(path, values):
    path.write_text(json.dumps(values, indent=2, allow_nan=False)+"\n")


def fixed(scene, probes, out):
    fixture = json.loads((out/"witness-input.json").read_text())
    archive = out.parent
    replay = json.loads((archive/"replay.json").read_text())
    trajectory = archive.parent/"cpu-local-cad/trajectory.npz"
    if digest(scene) != fixture["scene_sha256"] or digest(trajectory) != replay["trajectory_sha256"]:
        raise ValueError("scene or trajectory differs from the original test")
    bodies, pairs = read_model(scene)
    a, b, k, limit = pairs[fixture["stage"]-1]
    pose = [v for i, angle in zip((a, b), fixture["angles"], strict=True)
            for v in (*bodies[i][1], *bodies[i][2], angle)]
    axes = [probes, bodies[a][0], bodies[b][0], "6", str(2*limit), "--patch-wrench", "0"]
    change = swap_witness([*axes, "--bezier"], pose, *fixture["witnesses"], bodies[b], k, limit)
    if abs(np.diff(change["torque_Nm"])[0]) > 1e-6:
        raise ValueError("replacing the interior witness changed the pressure torque")
    reference = list(csv.DictReader((archive.parent/"cpu-cad-pressure/wrenches.csv").read_text().splitlines()))
    ids = np.concatenate([np.arange(i,i+100) for i in replay["starts"]])
    result = dict(probe_sha256=digest(probes), witness_swap=change, stages=[])
    ref_switch = json.loads((out/"switch-reference.json").read_text())
    a,b,k,limit = pairs[ref_switch["stage"]-1]
    cmd = [probes,bodies[a][0],bodies[b][0],"6",str(2*limit),"--loaded","--bezier"]
    result["switch"] = {}
    with np.load(out/"switch-poses.npz") as histories:
        for i, name in enumerate(("cad-patch","cad-bezier")):
            row=query(cmd,histories[name])[-1]
            if not row["resolved"] or len(row["regions"])!=ref_switch["regions"][i]:
                raise ValueError("lost a loaded region at the archived switch")
            result["switch"][name]=dict(regions=len(row["regions"]),
                                        torque_Nm=wrench(row["regions"], bodies[b][1], bodies[b][2], k, limit, True)[-1])
    with np.load(trajectory) as data:
        angles, origins = data["angles_rad"], data["origins_m"]
    for stage, (a, b, k, limit) in enumerate(pairs, 1):
        cmd = [probes, bodies[a][0], bodies[b][0], "6", str(2*limit), "--loaded", "--bezier"]
        columns = [col for i in (a, b) for col in (*origins[:,i].T,
                   *(np.full(len(angles), value) for value in bodies[i][2]), angles[:,i])]
        # Three identical traversals of the final window, following seven other windows.
        steps = np.r_[ids, ids[-100:], ids[-100:]]
        rows = query(cmd, np.column_stack(columns)[steps])
        torques = []
        for i, row in enumerate(rows):
            w = wrench(row["regions"] if row["resolved"] else None, origins[steps[i],b], bodies[b][2], k, limit, True)
            torques.append(None if w is None else w[-1])
        ref = {int(x["step"]):float(x["shaft_Nm"]) for x in reference
               if int(x["stage"]) == stage and x["method"]=="pressure" and x["resolved"]=="True"}
        errors = [torques[i]-ref[s] for i,s in enumerate(ids) if s in ref and torques[i] is not None]
        windows = []
        for start in (700,800,900):
            sub = rows[start+20:start+100]
            windows.append(dict(mean_ms=1000*np.mean([r["loaded_s"] for r in sub]),
                                broad_ms=1000*np.mean([r["broad_s"] for r in sub]),
                                window_builds=sum(r["window_builds"] for r in sub),
                                window_reuses=sum(r["window_reuses"] for r in sub)))
        # Compare the same 80 poses on each revisit. Include unresolved outcomes explicitly.
        for i in range(720,800):
            for j in (i+100,i+200):
                x,y=rows[i],rows[j]
                if x["resolved"]!=y["resolved"] or x["candidate_face_pairs"]!=y["candidate_face_pairs"] or x["labels"]!=y["labels"]:
                    raise ValueError("recurring candidate or region mismatch")
                np.testing.assert_allclose(x["regions"], y["regions"], rtol=1e-8, atol=1e-12)
        result["stages"].append(dict(stage=stage, windows=windows, step5_Nm=torques[5],
                                     unresolved=sum(t is None for t in torques[:800]), reference_poses=len(errors),
                                     reference_rms_Nm=np.sqrt(np.mean(np.square(errors)))))
    write(out/"fixed.json", result)


def accepted_cad_steps(scene, path, record):
    """Recover output torque from the accepted, unsubdivided rho=0 HHT steps."""
    if record["scene_sha256"] != digest(scene) or record["spectral_radius"] != 0 or record["geometry_update"] != "predictor":
        raise ValueError("reconstruction requires the same scene and rho=0 predictor run")
    count = int(record["last_accepted"]["step"]) + 1
    fields = ("step", "time_s", "cad_patch_batches", "q0", "q1", "q2", "v0", "v1", "v2")
    log = np.genfromtxt(path/"progress.csv", names=True, delimiter=",", usecols=fields, max_rows=count)
    np.testing.assert_array_equal(log["step"], np.arange(count))
    t = log["time_s"]
    h = record["dt_s"]
    np.testing.assert_allclose(np.diff(t), h, rtol=0, atol=1e-12)
    q, v = (np.column_stack([log[f"{key}{i}"] for i in range(3)]) for key in ("q", "v"))
    coordinates, smooth = smooth_loads(scene, t, q, v)
    bodies, pairs = read_model(scene)
    signs = np.array([bodies[i][2][2] for i in (pairs[0][0], pairs[0][1], pairs[1][1])])
    # Newmark beta=1 and alpha_m=-1 give 2*a[n]-a[n-1] directly from q and v.
    force = (2*coordinates[2, 0]/h**2) * (np.diff(q[:, 2]) - h*v[:-1, 2])
    torque = np.r_[np.nan, (force - smooth[1:, 2])*signs[2]]
    torque[np.r_[False, np.diff(log["cad_patch_batches"]) != 1]] = np.nan
    return t, v*signs, torque


def report(scene, runs, out):
    methods = ("cad-bezier","mesh-obb","cad-patch")
    traces, stats, partial = {}, {}, {}
    for m in methods:
        path=runs/m
        record=json.loads((path/"run.json").read_text())
        shutil.copyfile(path/"run.json", out/(m+"-run.json"))
        if record["status"]!="complete":
            stats[m]=dict(status=record["status"], process_s=record["process_s"], last_accepted=record["last_accepted"])
            if m=="cad-patch" and "cad-bezier" in traces and record["last_accepted"]:
                t, vel, torque = accepted_cad_steps(scene, path, record)
                n=len(t)
                np.testing.assert_allclose(t, traces["cad-bezier"]["time_s"][:n], rtol=0, atol=1e-10)
                delta=vel-traces["cad-bezier"]["speed"][:n]
                stats[m]["common_frames"]=n
                stats[m]["speed_vs_bezier_prefix_rms_rad_s"]=np.sqrt(np.mean(delta**2,axis=0)).tolist()
                partial.update(cad_patch_prefix_time_s=t, cad_patch_prefix_speed=vel,
                               cad_patch_reconstructed_output_torque_Nm=torque)
            continue
        record,_=verified_dynamics(scene,path,m)
        series=traces[m]=history(path/"response.csv",scene)
        stats[m]=dict(status="complete", process_s=record["process_s"],
                      first_loaded_ms=[], load_changes=[])
        if m=="cad-bezier":
            _, _, recovered=accepted_cad_steps(scene,path,record)
            discrepancy=float(np.nanmax(abs(recovered-series["output_torque_Nm"])))
            if discrepancy>1e-4:
                raise ValueError("HHT torque reconstruction differs from logged contact torque")
            stats[m]["torque_reconstruction_max_error_Nm"]=discrepancy
            stats[m]["torque_reconstruction_checked_steps"]=int(np.isfinite(recovered).sum())
        progress=np.genfromtxt(path/"progress.csv", names=True, delimiter=",",
                               usecols=("integration_s","cad_booleans"))
        elapsed=np.diff(progress["integration_s"],prepend=0)
        booleans=np.diff(progress["cad_booleans"],prepend=0)>0
        stats[m]["steps_with_solid_boolean"]=int(np.sum(booleans))
        stats[m]["wall_s_in_those_steps"]=float(np.sum(elapsed[booleans]))
        stats[m]["mean_step_ms_by_second"]=(np.diff(progress["integration_s"][::10000])*.1).tolist()
        for normal in series["normal_N"].T:
            loaded=normal>0
            stats[m]["first_loaded_ms"].append(1000*series["time_s"][np.flatnonzero(loaded)[0]])
            stats[m]["load_changes"].append(int(np.count_nonzero(np.diff(loaded))))
    if len({json.loads((runs/m/"run.json").read_text())["executable_sha256"] for m in methods})!=1:
        raise ValueError("closed-loop runs used different binaries")
    if "cad-bezier" in traces and "mesh-obb" in traces:
        cad,mesh=traces["cad-bezier"],traces["mesh-obb"]
        stats["cad_vs_mesh"]=dict(
            output_torque_rms_Nm=float(np.sqrt(np.mean((cad["output_torque_Nm"]-mesh["output_torque_Nm"])**2))),
            speed_rms_rad_s=np.sqrt(np.mean((cad["speed"]-mesh["speed"])**2,axis=0)).tolist())
    with np.load(out.parent/"responses.npz") as old:
        t=old["time_s"]
        old_delta=old["cad-bezier_output_torque_Nm"]-old["cad-patch_output_torque_Nm"]
        reference=old["RecurDyn_output_torque_Nm"]
        for m, series in traces.items():
            np.testing.assert_allclose(t,series["time_s"],atol=1e-10,rtol=0)
            stats[m]["torque_vs_recurdyn_rms_Nm"]=float(np.mean((series["output_torque_Nm"]-reference)**2)**.5)
            stats[m]["speed_vs_recurdyn_rms_rad_s"]=np.sqrt(np.mean((series["speed"]-old["RecurDyn_speed"])**2,axis=0)).tolist()
            stats[m]["te_vs_recurdyn_rms_mrad"]=np.sqrt(np.mean((series["te_mrad"]-old["RecurDyn_te_mrad"])**2,axis=0)).tolist()
    fig,axes=plt.subplots(3,2,figsize=(12,8),layout="constrained")
    colors={"mesh-obb":"#c06e26","cad-bezier":"#087e73","cad-patch":"#475770"}
    legends={"mesh-obb":"Mesh OBB","cad-bezier":"Bézier CAD (ours)","cad-patch":"Native CAD"}
    for m,series in traces.items():
        for ax,y in zip(axes.flat, (series["output_torque_Nm"],series["speed"][:,1],
                        series["normal_N"][:,0],series["normal_N"][:,1]), strict=False):
            ax.plot(t,y,label=legends[m],lw=.65,color=colors[m],alpha=.85)
    axes[0,0].plot(t,reference,label="RecurDyn",lw=.45,color="#252525",alpha=.6)
    detail=slice(25362,25406)
    bottom_labels=("Bézier minus CAD torque (N m)","Bézier minus CAD torque (N m)")
    if "cad-patch" in traces and "cad-bezier" in traces:
        axes[2,0].plot(t,old_delta,label="Previous difference",color="#999",lw=.7)
        new=traces["cad-bezier"]["output_torque_Nm"]-traces["cad-patch"]["output_torque_Nm"]
        peak=int(np.argmax(abs(new)))
        stats["paired_cad"]=dict(rms_Nm=np.sqrt(np.mean(new**2)), max_abs_Nm=abs(new[peak]), peak_time_s=t[peak],
                                frames_above_1_Nm=int(np.sum(abs(new)>1)),
                                original_window_max_Nm=np.max(abs(new[25365:25406])),
                                loaded_state_differences=(np.sum((traces["cad-bezier"]["normal_N"]>0)!=
                                (traces["cad-patch"]["normal_N"]>0),axis=0)).tolist())
        axes[2,0].plot(t,new,label="Revised difference",color=colors["cad-bezier"],lw=.7)
        axes[2,1].plot(t[detail],old_delta[detail],".-",color="#999",lw=1,label="Previous difference")
        axes[2,1].plot(t[detail],new[detail],".-",color=colors["cad-bezier"],lw=1,label="Revised difference")
    elif "cad-bezier" in traces and "mesh-obb" in traces:
        delta=traces["cad-bezier"]["output_torque_Nm"]-traces["mesh-obb"]["output_torque_Nm"]
        axes[2,0].plot(t,delta,color=colors["cad-bezier"],lw=.7,label="Full CPU runs")
        if partial and len(partial["cad_patch_prefix_time_s"])>=detail.stop:
            prefix=partial["cad_patch_prefix_time_s"]
            residual=traces["cad-bezier"]["output_torque_Nm"][:len(prefix)]-partial["cad_patch_reconstructed_output_torque_Nm"]
            peak=int(np.nanargmax(abs(residual)))
            original=residual[25365:25406]
            stats["paired_cad_prefix"]=dict(steps_with_torque=int(np.isfinite(residual).sum()),
                end_s=float(prefix[-1]), max_abs_Nm=abs(residual[peak]), peak_time_s=prefix[peak],
                rms_Nm=float(np.sqrt(np.nanmean(residual**2))),
                at_original_peak_Nm=float(residual[25377]) if np.isfinite(residual[25377]) else None,
                original_window_max_Nm=float(np.nanmax(abs(original))) if np.isfinite(original).any() else None,
                original_window_checked_steps=int(np.isfinite(original).sum()))
            axes[2,1].plot(t[detail],old_delta[detail],".-",color="#999",lw=1,label="Previous difference")
            axes[2,1].plot(prefix[detail],residual[detail],".-",color=colors["cad-bezier"],lw=1,label="Revised, reconstructed CAD")
            bottom_labels=("Bézier minus mesh torque (N m)","Bézier minus CAD torque (N m)")
        else:
            for m,series in traces.items():
                axes[2,1].plot(t[detail],series["output_torque_Nm"][detail],".-",color=colors[m],lw=1,label=m)
            bottom_labels=("Bézier minus mesh torque (N m)","Output torque (N m)")
    labels=("Output torque (N m)","Intermediate shaft speed (rad/s)","Stage 1 normal load (N)",
            "Stage 2 normal load (N)",*bottom_labels)
    for ax,label in zip(axes.flat,labels,strict=True):
        ax.set(xlabel="Simulation time (s)",ylabel=label)
        if ax.lines: ax.legend(frameon=False,fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
    axes[2,1].ticklabel_format(useOffset=False,axis="x")
    fig.savefig(out/"reducer.png",dpi=165)
    plt.close(fig)
    np.savez_compressed(out/"responses.npz",time_s=t,**partial,**{m+"_"+k:v for m,x in traces.items() for k,v in x.items() if k!="time_s"})
    write(out/"closed-loop.json",stats)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("fixed","report"))
    p.add_argument("scene",type=Path)
    p.add_argument("source",type=Path,help="Probe executable for fixed, run directory for report")
    p.add_argument("--out",type=Path,default=Path("results/aes/cpu-bezier/recurring"))
    args=p.parse_args()
    (fixed if args.action=="fixed" else report)(args.scene,args.source,args.out)
