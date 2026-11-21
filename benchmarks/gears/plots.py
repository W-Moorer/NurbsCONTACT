"""Create the model plate and industrial HHT comparison figures from recorded data."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from PIL import Image

from analyze import track
from run import CASES, csv_rows


BLUE="#1f689b"
GOLD="#b36c38"
TEAL="#218780"
INK="#20364b"
MUTED="#627d91"
GRID="#d7e1e9"
TITLES=("Spur pair","Helical pair","NASA-rig geometry","Two-stage reducer")
DETAILS=("24/48 teeth · 1800 rpm · 56 N·m","25/45 teeth · 25° helix · 1500 rpm",
         "28/28 teeth · 2000 rpm · 71.77 N·m","24/48 → 22/44 teeth · 4:1 · 148 N·m")


def style():
    mpl.rcParams.update({
        "font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"font.size":8.5,
        "axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,
        "axes.edgecolor":"#8499aa","axes.linewidth":.7,"axes.spines.top":False,"axes.spines.right":False,
        "legend.frameon":False,"legend.fontsize":7.5,"pdf.fonttype":42,"savefig.dpi":255,
    })


def label(ax,letter,title):
    ax.set_title(f"{letter}   {title}",loc="left",fontweight="bold",fontsize=9,pad=7)


def save(fig,folder,name):
    fig.savefig(folder/f"{name}.pdf",bbox_inches="tight",pad_inches=.08)
    fig.savefig(folder/f"{name}.png",bbox_inches="tight",pad_inches=.08)
    plt.close(fig)


def model_plate(root,folder):
    fig,axes=plt.subplots(2,2,figsize=(7.2,4.75),layout="constrained")
    for i,(axis,case,title,details) in enumerate(zip(axes.flat,CASES,TITLES,DETAILS,strict=True)):
        with Image.open(root/"images"/f"{case}-isometric.png") as image:
            axis.imshow(image)
        label(axis,"abcd"[i],title)
        axis.text(.01,-.055,details,transform=axis.transAxes,va="top",color=MUTED,fontsize=8)
        axis.set_axis_off()
    save(fig,folder,"gear-models")


def time_histories(root,folder):
    fig,axes=plt.subplots(4,2,figsize=(7.2,7.2),layout="constrained")
    colors=(BLUE,"#a58369",GOLD)
    legends=("B-rep","Mesh 25 µm","Mesh 1.56 µm")
    for i,case in enumerate(CASES):
        stage=int(case=="reducer")
        timings=[track(root,f"{case}-{key}") for key in ("cad","mesh-coarse","mesh-fine")]
        pinion=csv_rows(root/"models"/f"{case}.csv")[2*stage]
        mesh_hz=float(pinion["start_rpm"])*int(pinion["teeth"])/60
        start=.155
        end=start+7/mesh_hz
        for j,(axis,key) in enumerate(zip(axes[i],(f"normal{stage}_N",f"te{stage}_um"),strict=True)):
            for k,t in enumerate(timings):
                shown=(t["time"]>=start)&(t["time"]<=end)
                x=(t["time"][shown]-start)*1e3
                axis.plot(x,t[key][shown],color=colors[k],label=legends[k],lw=(1.25,.85,.9)[k],
                          linestyle=("-","-.",":")[k],zorder=(5,3,6)[k])
            axis.grid(axis="y",linewidth=.55,color=GRID)
            axis.set_xlim(0,1e3*(end-start))
            label(axis,"abcdefgh"[2*i+j],TITLES[i]+(" · stage 2" if stage else ""))
            axis.set_ylabel("Normal force (N)" if j==0 else "Transmission error (µm)")
            if i==3:
                axis.set_xlabel("Elapsed time in window (ms)")
    axes[0,0].legend(loc="lower right",ncol=1)
    save(fig,folder,"gear-dynamics")


def performance(root,folder,data):
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.8),layout="constrained")
    a,b=axes
    stages=data["runs"]
    resolutions=("coarse","medium","fine")
    palette=("#83a9c4","#427ead","#244e76")
    filters=[f"Mesh {value} µm" for value in ("25","6.25","1.56")]
    indexes=np.arange(len(CASES))
    for j,resolution in enumerate(resolutions):
        timing=[stages[f"{case}-mesh-{resolution}"]["median_integration_s"]/
                stages[f"{case}-cad"]["median_integration_s"] for case in CASES]
        a.bar(indexes+.23*(j-1),timing,width=.20,color=palette[j],label=filters[j],zorder=3)
    a.axhline(1,linestyle="--",color=MUTED,linewidth=.8)
    a.set_ylabel("Mesh runtime / B-rep runtime")
    a.set_xticks(indexes,labels=("Spur","Helical","NASA","Reducer"))
    a.grid(axis="y",color=GRID,linewidth=.5)
    a.legend(loc="upper left",fontsize=7.0)
    label(a,"a","Measured integration cost")

    x=np.array([25,6.25,1.5625])
    for case,name,color in zip(CASES,("Spur","Helical","NASA","Reducer"),(BLUE,TEAL,GOLD,"#7a608c"),strict=True):
        y=[max(r["transmission_rms_um"] for r in stages[f"{case}-mesh-{s}"]["relative_to_matched_cad"])
           for s in resolutions]
        b.loglog(x,y,"o-",label=name,color=color,lw=1.2,markersize=3.7)
    b.set_xticks(x,labels=("25","6.25","1.56"))
    b.xaxis.set_minor_formatter(NullFormatter())
    b.set(xlabel="Requested mesh deflection (µm)",ylabel="RMS error versus B-rep (µm)")
    b.grid(axis="y",color=GRID,linewidth=.5)
    b.legend(fontsize=7)
    label(b,"b","Faceting and transmission error")
    save(fig,folder,"gear-cost-accuracy")


def duty(root,folder):
    cad=track(root,"reducer-cad-duty")
    mesh=track(root,"reducer-mesh-duty")
    columns=(("normal","N","Normal force (N)",1),("axial","N","Axial load magnitude (N)",1),
             ("native_gap","mm","Native sampled gap (µm)",1000))
    fig,axes=plt.subplots(2,3,figsize=(7.2,4.35),layout="constrained")
    # Use tooth-cycle envelopes so the long-run plots retain short peaks.
    edges=np.linspace(0,3,151)
    x=(edges[:-1]+edges[1:])/2
    for stage in (0,1):
        for column,(key,unit,axis_title,factor) in enumerate(columns):
            axis=axes[stage,column]
            label(axis,"abcdef"[3*stage+column],"Stage "+str(stage+1))
            if stage==0 and key=="axial":
                axis.text(.5,.56,"Straight teeth",transform=axis.transAxes,ha="center",color=MUTED)
            for trace,color,method in ((mesh,GOLD,"Fine mesh"),(cad,BLUE,"B-rep")):
                q=np.abs(trace[f"{key}{stage}_{unit}"]) if key=="axial" else trace[f"{key}{stage}_{unit}"]
                q=q*factor
                boundary=np.searchsorted(trace["time"],edges)
                boundary[-1]=len(q)
                bands=np.array([(np.min(q[a:b]),np.mean(q[a:b]),np.max(q[a:b]))
                                for a,b in zip(boundary[:-1],boundary[1:],strict=True)])
                axis.fill_between(x,bands[:,0],bands[:,2],color=color,alpha=.13,linewidth=0)
                axis.plot(x,bands[:,0 if key=="native_gap" else 1],color=color,lw=.75,label=method)
            if stage==1:
                axis.set_xlabel("Simulated time (s)")
            axis.set_ylabel(axis_title)
            axis.set_xlim(0,3)
            axis.grid(axis="y",linewidth=.5,color=GRID)
    axes[0,0].legend()
    save(fig,folder,"reducer-duty")


def axial_refinement(root,folder,data):
    fig,axes=plt.subplots(2,2,figsize=(7.2,4.3),layout="constrained")
    for row,case in enumerate(("helical","reducer")):
        stage=int(row==1)
        for key,color,legend in (
            ("cad-axial63",INK,"B-rep / 63"),
            ("cad-axial31",BLUE,"B-rep / 31"),
            ("mesh-axial31",GOLD,"Mesh / 31"),
        ):
            trace=track(root,f"{case}-{key}")
            chosen=(trace["time"]>=.155)&(trace["time"]<=.17)
            time=1e3*(trace["time"][chosen]-.155)
            axes[row,0].plot(time,trace[f"normal{stage}_N"][chosen],lw=1.1,color=color,label=legend)
            axes[row,1].plot(time,trace[f"te{stage}_um"][chosen],lw=1.1,color=color,label=legend)
        speed=data["dense_axial"][case]["speed_ratio"]
        title="Helical pair" if row==0 else "Reducer · stage 2"
        label(axes[row,0],"ac"[row],title)
        label(axes[row,1],"bd"[row],f"{speed:.2f}× at 31 stations")
        axes[row,0].set_ylabel("Normal force (N)")
        axes[row,1].set_ylabel("Transmission error (µm)")
        for axis in axes[row]:
            axis.set_xlim(0,15)
            axis.grid(axis="y",color=GRID,linewidth=.55)
            if row==1:
                axis.set_xlabel("Elapsed time in window (ms)")
    axes[0,0].legend(loc="lower right",ncol=3,fontsize=6.8)
    save(fig,folder,"gear-axial-refinement")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path)
    args=parser.parse_args()
    root=args.results
    folder=root/"figures"
    folder.mkdir(exist_ok=True)
    data=json.loads((root/"summary.json").read_text(encoding="utf-8"))
    style()
    model_plate(root,folder)
    time_histories(root,folder)
    performance(root,folder,data)
    duty(root,folder)
    axial_refinement(root,folder,data)


if __name__=="__main__":
    main()
