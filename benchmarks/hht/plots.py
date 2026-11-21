"""Generate the paper's model and evidence figures from the HHT result files."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from PIL import Image

from analyze import drop_reference, load_trace, read_rows


BLUE = "#215f9b"
BRONZE = "#b76532"
TEAL = "#14867d"
MUTED = "#6c8193"
DARK = "#1f3548"


def style():
    mpl.rcParams.update({
        "font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":8.5,"axes.spines.top":False,"axes.spines.right":False,
        "axes.labelcolor":DARK,"text.color":DARK,"axes.edgecolor":"#8295a4",
        "xtick.color":DARK,"ytick.color":DARK,"axes.linewidth":.7,
        "xtick.major.width":.7,"ytick.major.width":.7,"legend.frameon":False,
        "pdf.fonttype":42,"svg.fonttype":"none","savefig.dpi":260,
    })


def title(ax,letter,text):
    ax.text(-.04,1.1,letter,transform=ax.transAxes,weight="bold",fontsize=10,ha="right")
    ax.set_title(text,pad=8,loc="left",fontweight="bold",fontsize=8.5)


def save(fig,folder,name):
    fig.savefig(folder/f"{name}.pdf",bbox_inches="tight",pad_inches=.07)
    fig.savefig(folder/f"{name}.png",bbox_inches="tight",pad_inches=.07)
    plt.close(fig)


def models(root,folder):
    specs=[
        ("drop","Single-ball impact","1 translational DOF"),
        ("cam-array","Cam and follower array","6 hinges + 6 sliders"),
        ("ball-train","Driven ball train","12 coupled sliders"),
    ]
    fig,axes=plt.subplots(1,3,figsize=(7.1,1.95),layout="constrained")
    for ax,(scene,heading,sub),letter in zip(axes,specs,"abc",strict=True):
        with Image.open(root/"images"/f"{scene}-isometric.png") as f:
            ax.imshow(f)
        ax.set_title(f"{letter}   {heading}",loc="left",fontsize=8.7,fontweight="bold",pad=5)
        ax.text(.01,-.06,sub,ha="left",va="top",transform=ax.transAxes,color=MUTED,fontsize=7.3)
        ax.set_axis_off()
    save(fig,folder,"model-scenes")


def calibration(root,folder,summary):
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.45),layout="constrained")
    a,b,c=axes
    osc=read_rows(root/"metrics"/"oscillator.csv")
    h=np.array([float(r["dt"])*1e3 for r in osc])
    error=np.array([float(r["rms_displacement"]) for r in osc])
    a.loglog(h,error,"o-",color=BLUE,label="HHT-α",markersize=3.5)
    a.loglog(h,error[0]*(h/h[0])**2,"--",color=MUTED,label="Order 2")
    a.set_xticks(h[::-1],labels=[f"{v:g}" for v in h[::-1]])
    a.xaxis.set_minor_formatter(NullFormatter())
    a.set(xlabel="Step size (ms)",ylabel="Displacement RMS error")
    a.legend(fontsize=7,loc="upper left")
    title(a,"a","Oscillator convergence")
    dr=load_trace(root,"drop-brep")
    exact=drop_reference(dr["time"])[1]
    b.plot(dr["time"]*1e3,exact,color=MUTED,linestyle="--",lw=1.7,label="Analytic")
    b.plot(dr["time"]*1e3,dr["solver_peak_force"],color=BLUE,lw=1,label="B-rep")
    b.set(xlabel="Time (ms)",ylabel="Normal force (N)",xlim=(20,112))
    b.legend(fontsize=7,loc="upper right")
    title(b,"b","Compliant impact")
    points=[r for r in summary["cam_queries"] if r["backend"]=="mesh"]
    x=[r["parameter"] for r in points]
    y=[r["max_error_mm"] for r in points]
    c.loglog(x,y,"o-",color=BRONZE,lw=1.2,markersize=3.5,label="Triangle BVH")
    c.axhline(.1,color=MUTED,linestyle=":",lw=1,label="0.1 mm target")
    c.set_xticks(x,labels=[str(int(v)) for v in x])
    c.xaxis.set_minor_formatter(NullFormatter())
    c.set(xlabel="Cam azimuthal facets",ylabel="Maximum gap error (mm)")
    c.legend(fontsize=6.8)
    title(c,"c","Faceting error")
    for ax in axes:
        ax.grid(color="#dce4eb",linewidth=.5,alpha=.62)
    save(fig,folder,"calibration")


def cost(folder,summary):
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.58),layout="constrained")
    a,b=axes
    runs=summary["runs"]
    names=["train-brep",summary["speedups"]["train_mesh_setting"]]
    positions=np.arange(2)
    values=[runs[n]["median_s"] for n in names]
    deviations=np.array([[runs[n]["median_s"]-runs[n]["min_s"] for n in names],
                         [runs[n]["max_s"]-runs[n]["median_s"] for n in names]])
    a.bar(positions,values,yerr=deviations,capsize=3,color=[BLUE,BRONZE],width=.55)
    for i,v in enumerate(values):
        a.text(i,v*1.20,f"{v:.2f} s",ha="center",fontsize=8,weight="bold")
    a.set(yscale="log",ylabel="Simulation time (s)",ylim=(min(values)*.6,max(values)*2.2))
    a.set_xticks(positions,labels=["B-rep","Triangle BVH"])
    title(a,"a","Ball train at matched accuracy")

    ids=["cam-brep","cam-mesh-12","cam-mesh-24","cam-mesh-48","cam-mesh-96"]
    for label in ids:
        r=runs[label]
        x=r["fine_reference"]["rms_translation_mm"]
        y=r["median_s"]
        native=label=="cam-brep"
        b.scatter(x,y,c=BLUE if native else BRONZE,marker="D" if native else "o",s=31,zorder=5)
        b.annotate("B-rep" if native else label.rsplit("-",1)[1],(x,y),xytext=(4,3),
                   textcoords="offset points",fontsize=7.3,ha="left")
    b.set(xscale="log",yscale="log",xlabel="Follower RMS error (mm)",ylabel="Simulation time (s)")
    b.margins(x=.19,y=.14)
    title(b,"b","Cam-array cost and trajectory error")
    for ax in axes:
        ax.grid(axis="y",color="#dce4eb",linewidth=.55)
        ax.set_axisbelow(True)
    save(fig,folder,"cost-accuracy")


def bands(ax,trace,field,unit_scale,color,seconds=1.0):
    t=trace["time"]
    bins=np.arange(0,t[-1]+seconds,seconds)
    idx=np.searchsorted(t,bins)
    idx[-1]=len(t)
    low=[]; high=[]; mean=[]; centers=[]
    for i in range(len(idx)-1):
        block=trace[field][idx[i]:idx[i+1]]*unit_scale
        if len(block):
            low.append(np.min(block)); high.append(np.max(block)); mean.append(np.mean(block))
            centers.append((bins[i]+bins[i+1])/2)
    ax.fill_between(centers,low,high,color=color,alpha=.19,linewidth=0)
    ax.plot(centers,mean,color=color,lw=.9)


def long_runs(root,folder):
    cam=load_trace(root,"cam-stress")
    train=load_trace(root,"train-stress")
    fig,axes=plt.subplots(2,2,figsize=(7.1,4.22),layout="constrained")
    fields=(("cad_gap",1000,"Minimum pose gap (mm)"),("solver_peak_force",1,"Peak contact force (N)"))
    for row,(trace,color,scene) in enumerate([(cam,BLUE,"Cam array"),(train,TEAL,"Ball train")]):
        for column,(field,scale,unit) in enumerate(fields):
            ax=axes[row,column]
            bands(ax,trace,field,scale,color)
            ax.set(ylabel=unit,xlim=(0,300))
            if row==1:
                ax.set_xlabel("Simulated time (s)")
            title(ax,"abcd"[2*row+column],scene+" · "+("clearance" if column==0 else "force"))
            ax.grid(axis="y",color="#dce4eb",linewidth=.55)
    save(fig,folder,"long-run")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path)
    args=parser.parse_args()
    root=args.results
    summary=json.loads((root/"metrics"/"summary.json").read_text(encoding="utf-8"))
    folder=root/"figures"
    folder.mkdir(parents=True,exist_ok=True)
    style()
    models(root,folder)
    calibration(root,folder,summary)
    cost(folder,summary)
    long_runs(root,folder)


if __name__=="__main__":
    main()
