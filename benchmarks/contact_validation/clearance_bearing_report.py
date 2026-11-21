"""Archive the auxiliary-bearing comparison and plot the unfiltered samples."""

import argparse
import gzip
import json
import shutil
from pathlib import Path

import numpy as np

from clearance_revision import measured


COARSE = "200rpm-0.25mm-10us-6cycles"
FINE = "200rpm-0.25mm-5us-6cycles"
HELD_OUT = ("150rpm-0.25mm-10us-6cycles", "200rpm-0.50mm-10us-6cycles")


def summary(work, run):
    return json.loads((work/run/"summary.json").read_text(encoding="utf-8"))


def trace(directory, rpm, probe_count):
    columns = ["time_s","s0_a_m_s2","s1_a_m_s2"]
    for index in range(probe_count):
        columns.extend((f"p{index}_gap_m", f"p{index}_normal_N"))
    with next(directory.glob("*-bearings.csv")).open(encoding="ascii") as stream:
        header = stream.readline().strip().split(",")
        raw = np.loadtxt(stream,delimiter=",",usecols=[header.index(c) for c in columns])
    time = raw[:,0]
    phase = (time*rpm/60-4)*360
    keep = phase >= -180*rpm/60*(time[1]-time[0])
    return ["phase_deg",*columns], np.column_stack((phase[keep],raw[keep]))


def plot(reference, control, both, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1,3,figsize=(11.5,3.1),layout="constrained")
    for ax, limits, title in ((axes[0],(35,68),"Target trough"),
                              (axes[2],(90,180),"Sustained loading")):
        lines = (
            ("Published experiment", reference.T, "#28323c", "-"),
            ("Fixed auxiliary bearings", control[:,[0,3]], "#997356", "--"),
            ("Both auxiliary clearances", both[:,[0,3]], "#007d8e", "-"),
        )
        for label, data, color, style in lines:
            mask = (data[:,0]>=limits[0])&(data[:,0]<=limits[1])
            ax.plot(*data[mask].T,color=color,ls=style,lw=1,label=label)
        ax.set(xlim=limits,title=title,ylabel="Axial acceleration (m/s²)")
    load = axes[1]
    mask = (both[:,0]>=35)&(both[:,0]<=68)
    total = np.sum(both[:,[5,7,9]],axis=1)
    for name,values,color in (("Pin–bore",total,"#007d8e"),("Needle bearing",both[:,11],"#997356"),
                              ("Shaft bearings",both[:,13],"#776293")):
        load.plot(both[mask,0],values[mask],label=name,color=color,lw=1)
    load.set(xlim=(35,68),title="Contact loads",ylabel="Normal load (N)")
    for ax in axes:
        ax.set_xlabel("Crank angle (°)")
        ax.spines[["right","top"]].set_visible(False)
        ax.grid(axis="y",alpha=.20)
    axes[0].legend(frameon=False,fontsize=7.5)
    axes[1].legend(frameon=False,fontsize=7.5)
    fig.savefig(output/"auxiliary-bearing-response.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work",type=Path)
    parser.add_argument("reference",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    source = args.work/FINE/"both"
    args.output.mkdir(parents=True,exist_ok=True)
    coarse = summary(args.work,COARSE)
    fine = summary(args.work,FINE)["cases"][0]
    transfer = [summary(args.work,tag)["cases"][0] for tag in HELD_OUT]
    control = next(row for row in coarse["cases"] if row["case"]=="control")
    verified = json.loads((args.reference/"impact-laws"/"summary.json").read_text(encoding="utf-8"))
    prior = next(row for row in verified["screening"] if row["case"]=="mount-ln")
    for key in ("quiet_rmse_m_s2","raw_rmse_m_s2"):
        if abs(control[key]-prior[key]) > 1e-8:
            raise ValueError("fixed-bearing deck differs from the verified control")
    _, control_trace = trace(args.work/COARSE/"control",200,3)
    fields, both_trace = trace(source,200,5)
    with gzip.open(args.output/"both-5us-raw.csv.gz","wt",encoding="ascii") as file:
        np.savetxt(file,both_trace,fmt="%.17g",delimiter=",",header=",".join(fields),comments="")
    fixtures = args.output/"model"
    fixtures.mkdir(exist_ok=True)
    for scene in (*source.glob("*.scene"), *source.glob("*.step")):
        shutil.copyfile(scene,fixtures/scene.name)
    report = {
        "experiment_doi":coarse["source"],
        "raw_sample_contract":"Final two out of six revolutions; raw and encoder-phase aligned.",
        "screening":coarse["cases"],
        "refinement":fine,
        "held_out":transfer,
    }
    (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    reference = measured(args.reference/"measured-200rpm-0.25mm.csv")
    plot(reference,control_trace,both_trace,args.output)


if __name__=="__main__":
    main()
