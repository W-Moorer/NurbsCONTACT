"""Make the AES response figures and assemble the Blender model panels."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from paper_diagrams import AMBER, BLUE, GRID, INK, save
from sphere_accuracy import reference


CASES = ("plane", "spheres", "cam", "spur", "reducer")


def series(path, fields):
    with path.open(encoding="utf-8") as stream:
        header = stream.readline().strip().split(",")
        return np.loadtxt(stream, delimiter=",", usecols=[header.index(name) for name in fields], unpack=True)


def style(axes, xlabel):
    for ax in np.atleast_1d(axes):
        ax.spines[["right","top"]].set_visible(False)
        ax.grid(axis="y",color=GRID,lw=.7)
        ax.tick_params(labelsize=8.5)
        ax.yaxis.set_major_locator(MaxNLocator(4))
    np.atleast_1d(axes)[-1].set_xlabel(xlabel)


def spheres(work, output):
    for scene, case in (("drop","plane"),("two-ball","spheres")):
        fields = ["time","solver_peak_force","q0"]+(["q1"] if scene == "two-ball" else [])
        time, force, *states = series(work/"linux-spheres"/f"{scene}-bounded-cache-125us.csv",fields)
        qref, fref = zip(*(reference(scene,t,[q[0] for q in states]) for t in time), strict=True)
        qref = np.asarray(qref).T
        p = (states[0]-states[0][0])*1000
        pr = (qref[0]-qref[0,0])*1000
        if scene == "two-ball":
            p -= (states[1]-states[1][0])*1000
            pr -= (qref[1]-qref[1,0])*1000
        fig,axes = plt.subplots(2,1,figsize=(7.12,3.6),sharex=True,layout="constrained")
        for ax,a,b,ylabel in zip(axes,(p,force),(pr,fref),
             ("Displacement (mm)" if scene=="drop" else "Relative closure (mm)","Normal force (N)"),strict=True):
            ax.plot(time*1000,a,color=BLUE,lw=1.8,label="Bounded CAD/HHT")
            ax.plot(time*1000,b,color=AMBER,lw=1.15,ls="--",label="Closed form")
            ax.set_ylabel(ylabel)
        axes[0].legend(ncol=2,frameon=False,loc="best",fontsize=8.6)
        style(axes,"Time (ms)")
        save(fig,output,case+"-response")


def industrial(work, output, cam_baselines):
    configs = {
        "cam": ((cam_baselines/"cam-cylinder-plane", "Analytic cylinder–plane"),
                (cam_baselines/"cam-triangle-bvh", "Triangle BVH, 2048 sides"),
                ("q1","solver_peak_force"), ("Follower travel (mm)", "Normal force (N)"),1000.,(0.,20.)),
        "spur": ((work/"industrial-pilot/spur-native-0nm-5us-31","CAD, 5 µs"),
                 (work/"spur-section-confirm/spur-section-mesh-1024segments-5us-31","Polyline, 1024 segments"),
                 ("te0_um","normal0_N"),("Transmission error (µm)","Mesh force (N)"),1.,(.109,.124)),
        "reducer": ((work/"industrial-pilot/reducer-native-0nm-5us-31","CAD, 31 slices"),
                    (work/"industrial-pilot/reducer-native-axial-check-0nm-5us-63","CAD, 63 slices"),
                    ("te1_um","normal1_N"),("Stage 2 transmission error (µm)","Stage 2 mesh force (N)"),1.,(.076,.186)),
    }
    for case,(native,baseline,fields,labels,scale,window) in configs.items():
        fig,axes = plt.subplots(2,1,figsize=(7.12,3.61),sharex=True,layout="constrained")
        for (location,label),color,linestyle in ((native,BLUE,"-"),(baseline,AMBER,"--")):
            t,q,f = series(location.with_suffix(".csv"),("time",*fields))
            take = (t>=window[0]) & (t<=window[1])
            for ax,y,ylabel in zip(axes,(q*scale,f),labels,strict=True):
                ax.plot(t[take],y[take],label=label,color=color,ls=linestyle,lw=1.30)
                ax.set_ylabel(ylabel)
        style(axes,"Time (s)")
        axes[-1].set_xlim(*window)
        axes[0].legend(ncol=2,frameon=False,loc="best",fontsize=8.6)
        save(fig,output,case+"-response")


def model_panels(work, output, cases=CASES):
    render = work/"paper/renders"
    font = ImageFont.truetype(findfont("DejaVu Sans"),24)
    for case in cases:
        full = Image.new("RGB",(1280,493),"white")
        for name,box,title in (("overview",(0,0,780,468),"(a) Assembly"),
                              ("detail",(812,0,1280,468),"(b) Contact region")):
            panel = Image.new("RGBA",(box[2]-box[0],box[3]-box[1]),(240,246,250,255))
            with Image.open(render/f"{case}-{name}.png") as source:
                rgba = source.convert("RGBA").resize(panel.size,Image.Resampling.LANCZOS)
                rgb = ImageEnhance.Color(rgba.convert("RGB")).enhance(1.18)
                graded = ImageEnhance.Brightness(rgb).enhance(.89)
                graded.putalpha(rgba.getchannel("A"))
                panel.alpha_composite(graded)
            full.paste(panel.convert("RGB"),box[:2])
            ImageDraw.Draw(full).text((box[0]+15,box[3]-32),title,fill=INK,font=font,
                                     stroke_width=4,stroke_fill="#f0f6fa")
        full.crop((0,0,1280,468)).save(output/(case+"-model.png"),dpi=(300,300),optimize=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--plots-only",action="store_true")
    parser.add_argument("--models-only",action="store_true")
    parser.add_argument("--cam-baselines",type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.models_only:
        spheres(args.work,args.output)
        industrial(args.work,args.output,args.cam_baselines or args.work.parent/"aes-cam-baselines")
    if not args.plots_only:
        model_panels(args.work,args.output)


if __name__ == "__main__":
    main()
