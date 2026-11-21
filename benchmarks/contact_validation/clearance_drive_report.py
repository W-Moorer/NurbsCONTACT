"""Archive the resolved crank speed, impact window and raw HHT outputs."""

import argparse
import gzip
import json
import shutil
from pathlib import Path

import numpy as np

from clearance import encoder_interval
from clearance_drive import needle_torque
from clearance_revision import measured


TEST = "200rpm-0.25mm"
HELD_OUT = ("150rpm-0.25mm", "200rpm-0.50mm")


def summary(work, tag, step):
    return json.loads((work/f"{tag}-{step}us"/"summary.json").read_text(encoding="utf-8"))


def archive_trace(directory, report, output):
    encoder = report["encoder_coordinate"]
    encoder_fields = [f"{variable}{encoder}" for variable in "qva"]
    fields = [
        "time_s",*encoder_fields,"s0_a_m_s2","s1_a_m_s2",
        "b5_x_m","b5_y_m","b6_x_m","b6_y_m","b8_x_m","b8_y_m",
    ]
    for index in range(5):
        fields.extend((f"p{index}_gap_m",f"p{index}_normal_N"))
    with next(directory.glob("*-drive.csv")).open(encoding="ascii") as source:
        header = source.readline().strip().split(",")
        raw = np.loadtxt(source,delimiter=",",usecols=[header.index(f) for f in fields])
    phase,interval = encoder_interval(raw[:,1],report["encoder_turns"][0])
    trace = dict(zip(fields,raw[interval].T))
    trace["phase_deg"] = phase
    trace["encoder_rpm"] = trace[encoder_fields[1]]*60/(2*np.pi)
    trace["needle_torque_N_m"] = needle_torque(trace)
    omega = report["rpm"]*2*np.pi/60
    trace["motor_torque_N_m"] = report["bias_N_m"]+report["controller_N_m_s_rad"]*(omega-trace[encoder_fields[1]])
    columns = (
        "phase_deg","time_s","encoder_rpm",encoder_fields[2],"s0_a_m_s2","s1_a_m_s2",
        "needle_torque_N_m","motor_torque_N_m",
        *(f"p{i}_{key}" for i in range(5) for key in ("gap_m","normal_N")),
    )
    with gzip.open(output/"slow-5us-raw.csv.gz","wt",encoding="ascii") as stream:
        data = np.column_stack([trace[c] for c in columns])
        np.savetxt(stream,data,fmt="%.17g",delimiter=",",header=",".join(columns),comments="")
    model = output/"model"
    model.mkdir(exist_ok=True)
    for fixture in (*directory.glob("*.scene"),*directory.glob("*.step")):
        shutil.copyfile(fixture,model/fixture.name)
    return trace


def plot(output, fixed, measured_curve, trace):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1,3,figsize=(11.4,3.05),layout="constrained")
    phase = trace["phase_deg"]
    for ax,limits,title in ((axes[0],(35,68),"Target trough"),(axes[2],(90,180),"Sustained loading")):
        series = (
            ("Published experiment",*measured_curve,"#28323c","-"),
            ("Prescribed speed",fixed["phase_deg"],fixed["s1_a_m_s2"],"#9d684a","--"),
            ("Finite crank inertia",phase,trace["s1_a_m_s2"],"#007e90","-"),
        )
        for label,x,y,color,style in series:
            mask = (x>=limits[0])&(x<=limits[1])
            ax.plot(x[mask],y[mask],label=label,color=color,ls=style,lw=1)
        ax.set(xlim=limits,title=title,ylabel="Axial acceleration (m/s²)")
    speed = axes[1]
    mask = (phase>=0)&(phase<=360)
    speed.plot(phase[mask],trace["encoder_rpm"][mask],color="#007e90",lw=1,label="Finite crank inertia")
    speed.axhline(200,color="#9d684a",ls="--",lw=.7,label="Prescribed speed")
    speed.set(xlim=(0,360),title="Encoder speed",ylabel="Crank speed (rpm)")
    for ax in axes:
        ax.set_xlabel("Crank angle (°)")
        ax.spines[["right","top"]].set_visible(False)
        ax.grid(axis="y",alpha=.22)
    axes[0].legend(frameon=False,fontsize=7.5,loc="best")
    speed.legend(frameon=False,fontsize=7.5,loc="best")
    fig.savefig(output/"drive-response.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work",type=Path)
    parser.add_argument("reference",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    fine = summary(args.work,TEST,5)["cases"][0]
    coarse = summary(args.work,TEST,10)
    transfer = [summary(args.work,tag,10)["cases"][0] for tag in HELD_OUT]
    source = args.work/f"{TEST}-5us"/"slow"
    trace = archive_trace(source,fine,args.output)
    shutil.copyfile(args.work/"speed-audit.json",args.output/"speed-audit.json")
    report = {"source":coarse["source"],"flywheel_inertia_kg_m2":coarse["flywheel_inertia_kg_m2"],
              "refinement":fine,"screening":coarse["cases"],"held_out":transfer}
    (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    fixed = np.genfromtxt(args.reference/"auxiliary-bearings"/"both-5us-raw.csv.gz",delimiter=",",names=True)
    measured_curve = measured(args.reference/f"measured-{TEST}.csv")
    plot(args.output,fixed,measured_curve,trace)


if __name__=="__main__":
    main()
