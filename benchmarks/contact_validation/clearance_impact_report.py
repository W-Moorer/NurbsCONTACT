"""Archive the raw-sample comparison with both curves from Flores Fig. 11."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import fitz
import numpy as np

from clearance import REFERENCE, published_curve
from clearance_reference_audit import envelope
from clearance_revision import measured


MAIN = "200rpm-0.25mm-10us-6cycles"
REFINED_RIGID = "200rpm-0.25mm-1us-6cycles"
REFINED_MOUNT = "200rpm-0.25mm-5us-6cycles"
TRANSFER = ("150rpm-0.25mm-10us-6cycles", "200rpm-0.50mm-10us-6cycles")


def minimum(phase, acceleration, begin, end):
    inside = np.flatnonzero((phase>=begin)&(phase<=end))
    index = inside[np.argmin(acceleration[inside])]
    return {"phase_deg": float(phase[index]), "acceleration_m_s2": float(acceleration[index])}


def trough_width(phase, acceleration):
    """Half-prominence width against the two shoulders within 42–56 degrees."""
    keep = (phase>=42)&(phase<=56)
    x, y = phase[keep], acceleration[keep]
    trough = int(np.argmin(y))
    if trough in (0,len(x)-1):
        return None
    shoulder = min(np.max(y[:trough]),np.max(y[trough+1:]))
    if shoulder <= y[trough]:
        return None
    level = (y[trough]+shoulder)/2
    left = np.flatnonzero(y[:trough]>=level)[-1]
    right = trough+1+np.flatnonzero(y[trough+1:]>=level)[0]
    begin = np.interp(level,y[left:left+2][::-1],x[left:left+2][::-1])
    end = np.interp(level,y[right-1:right+1],x[right-1:right+1])
    return float(end-begin)


def curve_metrics(phase, acceleration):
    return {
        "first_half_minimum": minimum(phase,acceleration,0,180),
        "impact_window_minimum": minimum(phase,acceleration,42,56),
        "impact_half_prominence_width_deg": trough_width(phase,acceleration),
    }


def archive_trace(work, tag, case, output):
    directory = work/tag/case
    report = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    condition = f"{report['rpm']}rpm-{report['radial_clearance_mm']:.2f}mm"
    columns = ["time_s","s0_a_m_s2","s1_a_m_s2"]
    for probe in range(len(report["sampled_entries"])):
        columns.extend(f"p{probe}_{column}" for column in (
            "gap_m","normal_N","gap_rate_m_s","impact_speed_m_s"))
    with (directory/f"{condition}-impact.csv").open(encoding="ascii") as source:
        fields = source.readline().strip().split(",")
        samples = np.loadtxt(source,delimiter=",",usecols=[fields.index(c) for c in columns])
    period = 60/report["rpm"]
    phase = (samples[:,0]/period-report["cycles"]+2)*360
    offset = report["integration"]["step_s"]/period*180
    samples = np.column_stack((phase[phase>=-offset],samples[phase>=-offset]))
    report["trough_widths_deg"] = [
        trough_width(samples[:,0]-start,samples[:,3]) for start in (0,360)
    ]
    with gzip.open(output/f"{case}-{report['step_us']:g}us.csv.gz","wt",encoding="ascii") as stream:
        np.savetxt(stream,samples,fmt="%.17g",delimiter=",",
                   header=",".join(["phase_deg",*columns]),comments="")
    return report,samples[:,[0,3]].T


def plot(output, experimental, published_numeric, rigid, spatial, mount):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1,3,figsize=(11.4,3.12),layout="constrained")
    measured_line = ("Published experiment",experimental,"#28323c","-")
    spatial_line = ("Spatial, rebound ratio 0.46",spatial,"#00788b","--")
    mount_line = ("Sensor mount, Eq. 3",mount,"#8e5c9c","-")
    panels = (
        ((0,72),"Published rigid comparison",(
            measured_line,("Published rigid simulation",published_numeric,"#557885","--"),
            ("STEP rigid, Eq. 3",rigid,"#a56343","-"))),
        ((42,56),"Target trough",(measured_line,spatial_line,mount_line)),
        ((90,180),"Sustained loading",(measured_line,spatial_line,mount_line)),
    )
    for ax,(limits,title,lines) in zip(axes,panels):
        for label,(phase,acceleration),color,style in lines:
            keep = (phase>=limits[0])&(phase<=limits[1])
            ax.plot(phase[keep],acceleration[keep],color=color,ls=style,lw=1,label=label)
        ax.set(xlabel="Crank angle (°)",xlim=limits,title=title)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.22)
    axes[0].set_ylabel("Axial acceleration (m/s²)")
    for ax in axes[:2]:
        ax.legend(frameon=False,fontsize=7.4,loc="best")
    fig.savefig(output/"impact-law-response.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work",type=Path)
    parser.add_argument("reference_pdf",type=Path)
    parser.add_argument("original_results",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with fitz.open(args.reference_pdf) as pdf:
        fig11_exp = published_curve(pdf[11],197,249,720,-150,200)
        fig11_numeric = published_curve(pdf[11],3,60,720,-150,200)
    fig6_exp = measured(args.original_results/"measured-200rpm-0.25mm.csv")

    def load(tag):
        return json.loads((args.work/tag/"summary.json").read_text(encoding="utf-8"))

    reference = {}
    for label,curve in (("figure6b",fig6_exp),("figure11c",fig11_exp),("figure11d",fig11_numeric)):
        phase, lower, _ = envelope(curve)
        reference[label] = curve_metrics(phase, lower)
    rigid, rigid_trace = archive_trace(args.work,REFINED_RIGID,"rigid-ln",args.output)
    mount, mount_trace = archive_trace(args.work,REFINED_MOUNT,"mount-ln",args.output)
    spatial, spatial_trace = archive_trace(args.work,MAIN,"spatial-exact",args.output)
    verify = json.loads((args.work/"impact-check"/"verification.json").read_text(encoding="utf-8"))
    report = {
        "reference_doi": REFERENCE,
        "source_sha256": hashlib.sha256(args.reference_pdf.read_bytes()).hexdigest(),
        "published_curves": reference,
        "isolated_rebound": verify,
        "screening": load(MAIN),
        "refinement": [rigid,mount],
        "spatial_widths_deg": spatial["trough_widths_deg"],
        "held_out": [row for tag in TRANSFER for row in load(tag)],
    }
    (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    plot(args.output,fig6_exp,fig11_numeric,rigid_trace,spatial_trace,mount_trace)


if __name__=="__main__":
    main()
