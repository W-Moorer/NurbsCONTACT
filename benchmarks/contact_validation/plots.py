"""Draw measured-observable comparisons and generate the manuscript result table."""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from run import read_csv


MEASURED="#212c42"
BREP="#006f86"
MESH="#bb6128"
GRID="#e3e9ef"


def finish(fig, path):
    for suffix in ("png","pdf"):
        fig.savefig(path.with_suffix("."+suffix),dpi=230,pad_inches=.06)
    plt.close(fig)


def trajectory(ax, file, label, ylabel):
    data=read_csv(file)
    x=data["time_s"]
    ax.plot(x,1000*data["brep_m"],color=BREP,label="STEP B-rep",lw=1.7)
    ax.plot(x,1000*data["mesh_m"],color=MESH,label="Triangle BVH",lw=1.1,ls="--")
    ax.plot(x,1000*data["measured_m"],"o",color=MEASURED,label="Measured",ms=2.9)
    ax.set(title=label,xlabel="Time from initial state (s)",ylabel=ylabel)


def figures(root, summary):
    folder=root/"figures"
    folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(7.8,5.45),layout="constrained")
    trajectory(axes[0,0],root/"curves"/"bounce-06.csv","a  Ball rebound","Height above rest (mm)")
    trajectory(axes[0,1],root/"curves"/"plastic-06.csv","b  Block sliding on a slope","Along-board position (mm)")
    a=axes[1,0]
    data=read_csv(root/"beam"/"comparison.csv")
    x=data["impact_velocity_m_s"]
    a.plot(x,1e6*data["brep_s"],color=BREP,lw=1.7)
    a.plot(x,1e6*data["mesh_s"],color=MESH,lw=1.1,ls="--")
    a.plot(x,1e6*data["measured_s"],"o",color=MEASURED,ms=2.9)
    a.set(title="c  First electrical contact pulse",xlabel="Measured impact speed (m/s)",
          ylabel="Contact duration (µs)")
    a=axes[1,1]
    groups=("bounce","wood","plastic","metal")
    for i,name in enumerate(groups):
        trials=[row for row in summary["trials"] if row["family"]==name and row["split"]=="held-out"]
        jitter=np.linspace(-.07,.07,len(trials))
        for side,method,color in ((-.16,"brep",BREP),(.16,"mesh",MESH)):
            errors=np.array([row[method]["rmse_m"]*1000 for row in trials])
            a.plot(i+side+jitter,errors,"o",mec="none",color=color,ms=3,alpha=.55)
            a.plot([i+side-.105,i+side+.105],[np.median(errors)]*2,color=color,lw=1.8)
    a.set_xticks(range(4),("Bounce","Wood","Plastic","Metal"))
    a.set(title="d  Error across held-out trajectories",ylabel="Position RMSE (mm)")
    for ax in axes.flat:
        ax.grid(axis="y",color=GRID,lw=.55)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(ax.xaxis.get_major_locator() if ax is a else MaxNLocator(6))
        title=ax.get_title()
        ax.set_title("")
        ax.set_title(title,loc="left",fontweight="bold")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside upper center",ncols=3,frameon=False)
    finish(fig,folder/"contact-validation")


def manuscript(root, summary, source):
    beam=json.loads((root/"beam"/"summary.json").read_text(encoding="utf-8"))
    timing=json.loads((root/"timing.json").read_text(encoding="utf-8"))
    holdout=summary["holdout"]
    bounce=[x for x in summary["trials"] if x["split"]=="held-out" and x["family"]=="bounce"]
    one_frame=sum(abs(x["brep"].get("first_apex_time_error_s",np.inf))<=1/30+1e-10 for x in bounce)
    stats = timing["trials"]
    ball_refinement=next(row["brep_half_step"]["delta_to_brep_m"] for row in bounce if row["id"]==6)
    slide_refinement=next(row["brep_half_step"]["delta_to_brep_m"] for row in summary["trials"] if row["key"]=="plastic-06")
    slide_coefficient,slide_exponent=f"{1e3*slide_refinement:.2e}".split("e")
    values={
        "ContactValidationPath":rf"\detokenize{{{Path(os.path.relpath(root/'figures',source)).as_posix()}/}}",
        "ContactMeasuredDrops":str(len(bounce)),
        "ContactApexWithinFrame":str(one_frame),
        "ContactApexHeight":f"{1000*holdout['bounce']['brep']['first_apex_height_error_m']['median']:.2f}",
        "ContactBeamCount":str(beam["trials"]),
        "ContactBeamError":f"{beam['duration_errors']['brep']['rmse_us']:.2f}",
        "ContactBeamHalfStep":f"{beam['half_step_to_brep_us_rms']:.3f}",
        "ContactBeamMeshDelta":f"{beam['mesh_to_brep_us_rms']:.3f}",
        "ContactBallStep":f"{1e6*summary['settings']['bounce_dt_s']:g}",
        "ContactBallRefine":f"{1e3*ball_refinement:.3f}",
        "ContactSlideRefine":rf"\ensuremath{{{slide_coefficient}\times 10^{{{int(slide_exponent)}}}}}",
        "ContactPlateSpeed":f"{stats['plastic-06']['mesh_to_brep_ratio']:.2f}",
        "ContactBallSpeed":f"{stats['bounce-06']['mesh_to_brep_ratio']:.2f}",
        "ContactBeamSpeed":f"{stats['beam-27']['mesh_to_brep_ratio']:.2f}",
    }
    if "brep-direct" in stats["plastic-06"]:
        values["ContactPlanarNativeSpeed"]=f"{stats['plastic-06']['direct_to_prepared_ratio']:.2f}"
    lines=[r"% Generated from the validated contact-comparison records."]
    lines.extend(rf"\newcommand{{\{key}}}{{{value}}}" for key,value in values.items())
    lines.append(r"\newcommand{\ContactValidationRows}{%")
    for label,key in (("Ball rebound","bounce"),("Wood block","wood"),("Plastic block","plastic"),
                      ("Metal block","metal")):
        a=holdout[key]
        fields=[label,str(a["trials"])]
        fields.extend(f"{1000*a[m]['rmse_m'][field]:.2f}" for m in ("brep","mesh")
                      for field in ("median","p90"))
        lines.append(" & ".join(fields)+r" \\")
    lines.append("}")
    (source/"contact-validation-results.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path)
    parser.add_argument("--manuscript",type=Path,default=Path("manuscript"))
    args=parser.parse_args()
    plt.rcParams.update({"font.size":8,"axes.spines.top":False,"axes.spines.right":False,
                         "axes.labelcolor":MEASURED,"text.color":MEASURED,"pdf.fonttype":42})
    summary=json.loads((args.results/"summary.json").read_text(encoding="utf-8"))
    figures(args.results,summary)
    manuscript(args.results.resolve(),summary,args.manuscript.resolve())
