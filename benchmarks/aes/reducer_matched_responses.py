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


def plot_full_history(t, cpu, rd, out):
    """Show raw same-mesh responses on the full common time axis."""
    series = (("CPU", cpu, "#006d77"), ("RecurDyn", rd, "#bd542b"))
    with plt.rc_context({"font.size": 9, "pdf.fonttype": 42, "path.simplify": False,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(3, 1, figsize=(8.3, 6.4), sharex=True, layout="constrained",
                                 height_ratios=(1, 1, .76))
        axes[1].sharey(axes[0])
        # Use each recorded interval once, including the final sample at t[-1].
        dt = np.diff(t)
        stride = max(1, round(.010 / np.median(dt)))
        endpoints = np.r_[np.arange(0, len(t)-1, stride), len(t)-1]
        edges = t[endpoints]
        for ax, (name, run, color) in zip(axes[:2], series, strict=True):
            y = run["torque"]
            ax.plot(t, y, color=color, lw=.38)
            peak = np.argmax(y)
            ax.scatter(t[peak], y[peak], s=18, facecolor="white", edgecolor=color, zorder=3, clip_on=False)
            ax.annotate(f"{y[peak]:.2f} N·m at {t[peak]*1000:.1f} ms",
                        xy=(t[peak], y[peak]), xytext=(.12, .93), textcoords="axes fraction",
                        ha="left", va="top", color=color, fontsize=8,
                        arrowprops={"arrowstyle": "-", "color": color, "lw": .7})
            ax.set_title(f"{name}   ·   recorded torque", loc="left", fontsize=10)
            impulse = .5 * (y[:-1] + y[1:]) * dt
            means = np.add.reduceat(impulse, endpoints[:-1]) / np.diff(edges)
            axes[2].stairs(means, edges, baseline=None, color=color, lw=1.05,
                           linestyle="-" if name == "CPU" else "--", label=name)
        axes[2].set_title("Mean torque over consecutive 10 ms intervals", loc="left", fontsize=10)
        axes[2].legend(loc="upper right", ncol=2, frameon=False, fontsize=8)
        for ax in axes:
            ax.set_ylabel("Torque (N·m)")
            ax.grid(axis="y", alpha=.18)
            ax.set_axisbelow(True)
        axes[-1].set(xlim=(t[0], t[-1]), xlabel="Simulation time (s)")
        fig.suptitle("Output shaft contact torque over the full simulation", fontsize=13)
        for fmt in ("png", "pdf"):
            fig.savefig(out/f"full-output-torque.{fmt}", dpi=240)
        plt.close(fig)

        fig, axes = plt.subplots(5, 2, figsize=(8.3, 8.6), sharex=True, sharey="row",
                                 layout="constrained")
        rows = [("speed", 0, "Input shaft\n(rad/s)"), ("speed", 1, "Intermediate shaft\n(rad/s)"),
                ("speed", 2, "Output shaft\n(rad/s)"), ("te", 0, "Stage 1 error\n(mrad)"),
                ("te", 1, "Stage 2 error\n(mrad)")]
        for i, (key, index, label) in enumerate(rows):
            axes[i, 0].set_ylabel(label)
            for j, (name, run, color) in enumerate(series):
                ax = axes[i, j]
                ax.plot(t, run[key][:, index], color=color, lw=.38)
                ax.grid(axis="y", alpha=.18)
                ax.set_axisbelow(True)
                if i == 0:
                    ax.set_title(name, color=color, loc="left", weight="bold")
        for ax in axes[-1]:
            ax.set(xlim=(t[0], t[-1]), xlabel="Simulation time (s)")
        fig.suptitle("Shaft angular speeds and transmission errors", fontsize=13)
        for fmt in ("png", "pdf"):
            fig.savefig(out/f"full-shaft-responses.{fmt}", dpi=240)
        plt.close(fig)


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
    plot_full_history(t, dict(speed=cpu["speed"], te=cpu["te"], torque=cpu["torque"][:, 2]), rd, out)
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
