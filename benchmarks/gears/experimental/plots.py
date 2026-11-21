"""Draw the digitized tooth-force and encoder transmission-error comparisons."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare import NASA, UNSW_CAL, read


INK="#253b50"
BLUE="#146383"
ORANGE="#ad6334"
GRID="#dce6ed"
MODEL=(("brep","B-rep",BLUE,"-"),("mesh","Triangle BVH",ORANGE,"--"))


def save(figure,folder,name):
    figure.savefig(folder/f"{name}.pdf",bbox_inches="tight",pad_inches=.08)
    figure.savefig(folder/f"{name}.png",bbox_inches="tight",pad_inches=.08,dpi=235)
    plt.close(figure)


def force_plot(root,folder,summary,selected=False):
    keys="acd" if selected else "abcd"
    rows,columns=(1,3) if selected else (2,2)
    dimensions=(9.1,2.9) if selected else (7.15,4.95)
    fig,axes=plt.subplots(rows,columns,figsize=dimensions,layout="constrained")
    for ax,key in zip(axes.flat,keys,strict=True):
        rpm,torque=NASA[key]
        data=read(root/"curves"/f"nasa-{key}.csv")
        source=read(root/"measurements"/f"nasa-7{key}.csv")
        x=data["roll_deg"]
        scored=(source["roll_deg"]>=x[0])&(source["roll_deg"]<=x[-1])
        ax.plot(source["roll_deg"][scored],source["force_N"][scored],"o",color=INK,
                markersize=3,label="Digitized experiment",zorder=5)
        for name,label,color,stroke in MODEL:
            ax.plot(x,data[name],label=label,color=color,ls=stroke,lw=1.25)
        role="calibration" if key=="a" else "held out"
        ax.set_title(f"{key}   {rpm:,} rpm · {torque:.1f} N m · {role}",loc="left",weight="bold",fontsize=9)
        ids=summary["comparisons"][f"nasa-{key}"]
        ax.text(.98,.95,f"RMSE {ids['brep']['error']['rms']:.0f} / {ids['mesh']['error']['rms']:.0f} N",
                ha="right",va="top",transform=ax.transAxes,fontsize=7.2,color=BLUE)
        ax.set_xlim(30,11.8)
        ax.set_ylim(0,2450)
        ax.set_xlabel("Driven tooth roll angle (°)")
        ax.set_ylabel("Single-tooth normal force (N)")
        ax.grid(axis="y",color=GRID,linewidth=.55)
    axes.flat[0].legend(loc="upper left",fontsize=7.2)
    save(fig,folder,"nasa-single-tooth-selected" if selected else "nasa-single-tooth")


def encoder_plot(root,folder,summary):
    fig,axes=plt.subplots(3,2,figsize=(7.15,6.65),layout="constrained")
    keys=[UNSW_CAL,"Data_05Hz_05Nm_H","Data_10Hz_15Nm_H"]
    for row,(key,(a,b)) in enumerate(zip(keys,axes,strict=True)):
        data=read(root/"curves"/f"{key}.csv")
        traces=read(root/"measurements"/f"{key}.csv")
        row_stats=summary["comparisons"][key]
        x=data["mesh_cycle"]
        a.fill_between(x,traces["block_min_um"],traces["block_max_um"],color="#9bafc1",alpha=.35,
                       linewidth=0,label="Measured block range")
        a.plot(x,data["measured"],label="Encoder experiment",color=INK,lw=1.5)
        for name,label,color,stroke in MODEL:
            a.plot(x,data[name],label=label,color=color,ls=stroke,lw=1.2)
        for i,(name,label,color) in enumerate((("measured","Encoder experiment",INK),
                                               ("brep","B-rep",BLUE),("mesh","Triangle BVH",ORANGE))):
            amplitudes=2*np.abs(np.fft.rfft(data[name]))/len(x)
            orders=np.arange(1,7)
            b.bar(orders+(i-1)*.24,amplitudes[1:7],width=.22,color=color,label=label)
        role="calibration" if key==UNSW_CAL else "held out"
        a.set_title(f"{'ace'[row]}   {row_stats['rpm']:.1f} rpm · {row_stats['input_torque_Nm']:.2f} N m · {role}",
                    loc="left",weight="bold",fontsize=8.6)
        b.set_title(f"{'bdf'[row]}   Mesh harmonic amplitudes",loc="left",weight="bold",fontsize=8.6)
        a.set(xlim=(0,1),ylabel="AC transmission error (µm)")
        b.set(xticks=np.arange(1,7),ylabel="Amplitude (µm)")
        for ax in (a,b):
            ax.grid(axis="y",color=GRID,linewidth=.55)
            ax.set_axisbelow(True)
        if row==2:
            a.set_xlabel("Mesh cycle")
            b.set_xlabel("Multiple of mesh frequency")
    axes[0,1].legend(loc="upper right",fontsize=7.2)
    save(fig,folder,"unsw-encoder-dte")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path)
    parser.add_argument("--nasa-selected-only",action="store_true")
    args=parser.parse_args()
    root=args.results
    folder=root/"figures"
    folder.mkdir(exist_ok=True)
    mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":8.1,"axes.spines.top":False,"axes.spines.right":False,"text.color":INK,
        "axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,"legend.frameon":False,"pdf.fonttype":42})
    report=json.loads((root/"summary.json").read_text(encoding="utf-8"))
    force_plot(root,folder,report,selected=True)
    if not args.nasa_selected_only:
        force_plot(root,folder,report)
        encoder_plot(root,folder,report)


if __name__=="__main__":
    main()
