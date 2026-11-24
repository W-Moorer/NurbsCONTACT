"""Compare independently integrated reducer outputs on their recorded physical times."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rms(x):
    return np.sqrt(np.mean(np.square(x),axis=0))


def read_run(csv, deck):
    x=np.genfromtxt(csv,names=True,delimiter=",")
    t=x["time_s"]
    with np.load(deck/"assembly.npz") as cfg:
        signs=cfg["bodies"][[0,1,3],3]
        nominal=cfg["bodies"][[0,1,3],8]
        inertia,stiffness,damping=cfg["coord"][:,:3].T
    coordinates=[]
    tables={}
    lines=iter((deck/"reducer.scene").read_text().splitlines())
    for line in lines:
        if line.startswith("coordinate "):
            coordinates.append(list(map(float,line.split()[1:])))
        if line.startswith("load "):
            _,key,count=line.split()
            tables[int(key)]=np.array([list(map(float,next(lines).split())) for _ in range(int(count))])
    q,v,g=(np.column_stack([x[f"{key}{i}"] for i in range(3)]) for key in ("q","v","generalized_force"))
    base=np.array(coordinates)
    smooth=base[:,6]-stiffness*(q-base[:,5])-damping*v
    for key,table in tables.items(): smooth[:,key]+=np.interp(t,table[:,0],table[:,1])
    ratio=nominal[1:]/nominal[:-1]
    change=q-q[0]
    te=1e3*(change[:,1:]-ratio*change[:,:-1])
    return dict(t=t,speed=v*signs,q=q,te=te,torque=(g-smooth)*signs,smooth=smooth*signs,inertia=inertia,
                normal=np.column_stack([x[f"p{i}_normal_N"] for i in range(2)]),ratio=ratio)


def compare(trace, deck, matched, original, out, animation=False):
    out.mkdir(parents=True,exist_ok=True)
    cpu=read_run(trace,deck)
    t=cpu["t"]
    with np.load(deck/"assembly.npz") as model: nominal=model["bodies"][0,8]
    def reference(file):
        with np.load(file) as d:
            times=d["reference_rad"][:,0]/nominal
            ids=np.rint(t/(times[1]-times[0])).astype(int)
            np.testing.assert_allclose(times[ids],t,atol=1e-9,rtol=0)
            q=d["angle_rad"][ids]-d["angle_rad"][0]
            return dict(speed=d["global_speed_rad_s"][ids],torque=d["contact_raw_TZ_Nm"][ids,1],
                        te=1000*(q[:,1:]-q[:,:-1]*cpu["ratio"]),normal=d["resultant_N"][ids],
                        support=d["support_raw_TZ_Nm"][ids,2],drive=d["load_raw_TZ_Nm"][ids,1])
    rd=reference(matched)
    original_rd=reference(original)
    brake=(t>=.102-1e-9)&(t<=.114+1e-9)
    def metrics(ref,mask):
        a=cpu["torque"][mask,2]; b=ref["torque"][mask]
        def impulse_balance(torque,speed):
            return float(np.trapezoid(torque,t[mask])-cpu["inertia"][2]*(speed[-1]-speed[0]))
        return dict(samples=int(mask.sum()),velocity_rms_rad_s=rms(cpu["speed"][mask]-ref["speed"][mask]).tolist(),
                    transmission_error_difference_rms_mrad=rms(cpu["te"][mask]-ref["te"][mask]).tolist(),
                    output_torque_difference_rms_Nm=float(rms(a-b)),cpu_mean_output_Nm=float(a.mean()),
                    reference_mean_output_Nm=float(b.mean()),cpu_peak_output_Nm=float(a.max()),
                    reference_peak_output_Nm=float(b.max()),cpu_angular_impulse_Nms=float(np.trapezoid(a,t[mask])),
                    reference_angular_impulse_Nms=float(np.trapezoid(b,t[mask])),
                    cpu_sampled_balance_residual_Nms=impulse_balance(a+cpu["smooth"][mask,2],cpu["speed"][mask,2]),
                    reference_sampled_balance_residual_Nms=impulse_balance(b+ref["support"][mask]+ref["drive"][mask],ref["speed"][mask,2]))
    stats={"physical_s":float(t[-1]),"samples":len(t),"step_s":float(t[1]-t[0])}
    for label,mask in [("full",np.ones(len(t),bool)),("brake",brake)]:
        stats[label]=metrics(rd,mask)
        stats[label]["native_mesh_change_torque_rms_Nm"]=float(rms(original_rd["torque"][mask]-rd["torque"][mask]))
        stats[label]["cpu_vs_original_native_mesh"]=metrics(original_rd,mask)
    (out/"closed-loop.json").write_text(json.dumps(stats,indent=2,allow_nan=False)+"\n")
    np.savez_compressed(out/"closed-loop.npz",time_s=t,cpu_speed=cpu["speed"],rd_speed=rd["speed"],
                        cpu_te_mrad=cpu["te"],rd_te_mrad=rd["te"],cpu_output_torque_Nm=cpu["torque"][:,2],
                        rd_output_torque_Nm=rd["torque"],original_rd_torque_Nm=original_rd["torque"],
                        original_rd_speed=original_rd["speed"],original_rd_te_mrad=original_rd["te"],
                        cpu_sum_normal_load_N=cpu["normal"],rd_resultant_force_N=rd["normal"],
                        original_rd_resultant_force_N=original_rd["normal"])
    colors=("#006d77","#c35a25")
    fig,axes=plt.subplots(3,2,figsize=(12,9),layout="constrained")
    take=brake
    for i,name in enumerate(("Input","Intermediate","Output")):
        ax=axes[i,0]
        for v,c,lab in [(rd["speed"],colors[1],"RecurDyn"),(cpu["speed"],colors[0],"CPU")]:
            ax.plot(t[take]*1000,v[take,i],color=c,label=lab,lw=1.2)
        ax.set(ylabel="Angular speed (rad/s)",title=name+" shaft")
    for i in (0,1):
        ax=axes[i,1]
        for v,c in [(rd["te"],colors[1]),(cpu["te"],colors[0])]:ax.plot(t[take]*1000,v[take,i],color=c,lw=1.2)
        ax.set(ylabel="Transmission error (mrad)",title=f"Stage {i+1}")
    axes[2,1].plot(t[take]*1000,rd["torque"][take],color=colors[1],lw=1.2)
    axes[2,1].plot(t[take]*1000,cpu["torque"][take,2],color=colors[0],lw=1.2)
    axes[2,1].set(ylabel="Contact torque (N m)",title="Output shaft contact torque")
    for ax in axes.flat:
        ax.grid(alpha=.23);ax.spines[["top","right"]].set_visible(False);ax.set_xlabel("Time (ms)")
    axes[0,0].legend(frameon=False,ncol=2)
    fig.suptitle("Brake engagement on identical triangular surfaces")
    fig.savefig(out/"brake-responses.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,6),layout="constrained",sharex=True)
    for i,(v,c,lab) in enumerate([(rd["torque"],colors[1],"RecurDyn"),(cpu["torque"][:,2],colors[0],"CPU")]):
        axes[0].plot(t,np.r_[0,np.cumsum((v[:-1]+v[1:])*.5*np.diff(t))],color=c,lw=1.5,label=lab)
    axes[1].plot(t,cpu["speed"][:,2]-rd["speed"][:,2],color=colors[0],rasterized=True,lw=.5)
    axes[0].set_ylabel("Output contact impulse (N m s)");axes[1].set_ylabel("Output speed difference (rad/s)")
    axes[1].set_xlabel("Time (s)");axes[0].legend(frameon=False)
    for ax in axes:ax.grid(alpha=.23);ax.spines[["top","right"]].set_visible(False)
    fig.savefig(out/"duty-responses.png",dpi=160);plt.close(fig)
    if animation:
        from long_mesh_figures import render_reducer_animation
        render_reducer_animation(deck,out/"reducer.mp4",cpu["q"],cpu["normal"],t,
                                 title="CPU mesh dynamics | Two-stage reducer")


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for key in ("trace","deck","matched","original","out"): cli.add_argument(key,type=Path)
    cli.add_argument("--animation",action="store_true")
    compare(**vars(cli.parse_args()))
