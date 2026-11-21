"""Package raw-sample plots and numerical evidence for the spatial-clearance study."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def single(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != 1:
        raise ValueError(f"expected one selected spatial model in {path}")
    return rows[0]


def plot(root, original, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = np.genfromtxt(root/"raw.csv.gz",delimiter=",",names=True)
    control = np.genfromtxt(original/"flexibility"/"flex-raw.csv.gz",delimiter=",",names=True)
    reference = np.loadtxt(original/"measured-200rpm-0.25mm.csv",delimiter=",",skiprows=1).T
    fig, axes = plt.subplots(1,3,figsize=(11.6,3.05),layout="constrained")
    for ax,limits,title in ((axes[0],(35,68),"Impact window"),
                            (axes[2],(90,180),"Sustained loading")):
        exp_phase, exp_accel = reference
        mask = (exp_phase>=limits[0]) & (exp_phase<=limits[1])
        ax.plot(exp_phase[mask],exp_accel[mask],color="#28323c",label="Published experiment",lw=1.2)
        mask = (control["angle_deg"]>=limits[0]) & (control["angle_deg"]<=limits[1])
        ax.plot(control["angle_deg"][mask],control["slider_a_m_s2"][mask],
                color="#a36b53",label="Translational carriage",lw=1,ls="--")
        mask = (raw["phase_deg"]>=limits[0]) & (raw["phase_deg"]<=limits[1])
        ax.plot(raw["phase_deg"][mask],raw["s1_a_m_s2"][mask],
                color="#00788b",label="Spatial sensor point",lw=1.05)
        ax.set(xlim=limits,ylabel="Axial acceleration (m/s²)",title=title)
    middle = axes[1]
    mask = (raw["phase_deg"]>=35) & (raw["phase_deg"]<=68)
    phase = raw["phase_deg"][mask]
    for i,label,color in ((0,"Lower station","#006d84"),(1,"Mid-plane","#94775f"),(2,"Upper station","#746a9a")):
        middle.plot(phase,1e6*raw[f"p{i}_gap_m"][mask],label=label,color=color,lw=1)
    middle.axhline(0,color="#303940",lw=.55)
    middle.set(xlim=(35,68),ylabel="Signed clearance (µm)",title="Axial contact stations")
    for ax in axes:
        ax.set_xlabel("Crank angle (°)")
        ax.spines[["right","top"]].set_visible(False)
        ax.grid(axis="y",alpha=.2)
    axes[0].legend(frameon=False,fontsize=7.4)
    middle.legend(frameon=False,fontsize=7.4)
    fig.savefig(output/"spatial-response.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spatial",type=Path)
    parser.add_argument("mount",type=Path)
    parser.add_argument("original_results",type=Path)
    parser.add_argument("archive",type=Path)
    args = parser.parse_args()
    screen = "200rpm-0.25mm-20us-5cycles"
    base = "200rpm-0.25mm-10us-6cycles"
    refinements = {
        "10us": base,
        "5us": "200rpm-0.25mm-5us-6cycles",
        "1us": "200rpm-0.25mm-1us-2cycles",
    }
    transfer = {
        "150rpm-0.25mm": "150rpm-0.25mm-10us-6cycles",
        "200rpm-0.50mm": "200rpm-0.50mm-10us-6cycles",
    }
    load = lambda root,tag: json.loads((root/tag/"summary.json").read_text(encoding="utf-8"))
    summary = {
        "protocol": "Fixed encoder origin and raw integration samples throughout. "
        "The long-lever variant is selected on sustained-loading response. "
        "Mount cases use a dynamic lumped mode and preserve nominal mass properties.",
        "screening": load(args.spatial,screen),
        "measurement_mount_screening": load(args.mount,screen),
        "measurement_mount_refinement": single(args.mount/"200rpm-0.25mm-5us-2cycles"/"summary.json"),
        "refinements": {k:single(args.spatial/v/"summary.json") for k,v in refinements.items()},
        "held_out": {k:single(args.spatial/v/"summary.json") for k,v in transfer.items()},
    }
    args.archive.mkdir(parents=True,exist_ok=True)
    (args.archive/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    shutil.copyfile(args.spatial/"reference-audit.json",args.archive/"reference-audit.json")
    plot(args.archive/base/"long-lever",args.original_results,args.archive)


if __name__ == "__main__":
    main()
