"""Summarize the recorded rigid-body runs and generate the paper figure."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np


def plot(rows, folders, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    fig, axes = plt.subplots(1,3,figsize=(11.2,3.),layout="constrained")
    branches = (
        ("nurbs","HHT","STEP/HHT","#007b8c","o"),
        ("mujoco","RK4","MuJoCo RK4","#725b97","s"),
        ("mujoco","implicit","MuJoCo implicit","#a56a4f","D"),
        ("drake","runge_kutta3","Drake RK3","#416e52","^"),
    )
    for ax,case,title in zip(axes[:2],("plane","spheres"),("Sphere–plane","Two spheres")):
        for engine,method,label,color,mark in branches:
            part = [row for row in rows if (row["engine"],row["integrator"],row["case"])==(engine,method,case)]
            ax.loglog([1000*row["step_s"] for row in part],
                      [1000*row["position_rms_mm"] for row in part],
                      label=label,color=color,marker=mark,lw=1.15)
        ax.set(xlabel="Time step (ms)",ylabel="Position RMS error (µm)",title=title)
        ax.set_xticks([0.25,0.5,1],labels=["0.25","0.5","1.0"])
        ax.xaxis.set_minor_formatter(NullFormatter())
    axes[0].legend(frameon=False,fontsize=7.5)
    contact = axes[2]
    reference = np.genfromtxt(folders["nurbs"]/"plane-HHT-1000us.csv",delimiter=",",names=True)
    contact.plot(1000*reference["time_s"],reference["ref_force_N"],color=".25",
                 lw=1.4,label="Analytical",linestyle="--")
    for engine,method,label,color,_ in branches:
        if method == "implicit":
            continue
        data = np.genfromtxt(folders[engine]/f"plane-{method}-1000us.csv",delimiter=",",names=True)
        contact.plot(1000*data["time_s"],data["force_N"],label=label,color=color,lw=1.05,alpha=.78)
    contact.set(xlim=(33,94),xlabel="Time (ms)",ylabel="Normal force (N)",title="Common force pulse")
    contact.legend(frameon=False,fontsize=7.5,loc="upper right")
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.23)
    fig.savefig(output/"engine-comparison.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nurbs",type=Path)
    parser.add_argument("mujoco",type=Path)
    parser.add_argument("drake",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    folders = {name:getattr(args,name) for name in ("nurbs","mujoco","drake")}
    results = {name:json.loads((folder/"summary.json").read_text(encoding="utf-8"))
               for name,folder in folders.items()}
    if len({tuple(report[k] for k in ("layer_m","stiffness_N_m","mass_each_kg",
                                     "initial_speed_m_s","duration_s","stand_off_m","repetitions"))
            for report in results.values()})!=1:
        raise ValueError("the engine protocols differ")
    same_platform = len({(report["platform"],report["host"]) for report in results.values()})==1
    rows = [row for report in results.values() for row in report["runs"]]
    assert len(rows)==24 and all(len(row["times_s"])==7 for row in rows)
    args.output.mkdir(parents=True,exist_ok=True)
    for engine,folder in folders.items():
        destination = args.output/engine
        destination.mkdir(exist_ok=True)
        for source in (*folder.glob("*-*us.csv"),*(folder/k for k in ("plane.scene","spheres.scene")
                                                  if (folder/k).is_file()),*folder.glob("*.step")):
            shutil.copyfile(source,destination/source.name)
    for report in results.values():
        del report["host"]
    summary = {"timings_share_platform":same_platform,"engines":results}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    plot(rows,folders,args.output)


if __name__=="__main__":
    main()
