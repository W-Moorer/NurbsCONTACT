"""Package held-out cantilever waveforms, error metrics and replay scenes."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

from cantilever_gap import HELD_OUT


def blocks(samples):
    rows = []
    time = samples["time_s"]
    for start in range(0,100,10):
        keep = (time>=max(start,2))&(time<start+10)
        t = time[keep]
        a = samples["acc_tip_ms2"][keep]
        b = samples["brep_ms2"][keep]
        c = samples["linear_ms2"][keep]
        active = (samples["lower_N"][keep]>0)|(samples["upper_N"][keep]>0)
        basis = np.exp(-2j*np.pi*13*t)
        measured = np.dot(a,basis)
        predicted = np.dot(b,basis)
        rows.append({
            "begin_s":float(t[0]),"end_s":float(t[-1]),
            "measured_rms_m_s2":float(np.sqrt(np.mean(a*a))),
            "brep_rms_m_s2":float(np.sqrt(np.mean(b*b))),
            "linear_rms_m_s2":float(np.sqrt(np.mean(c*c))),
            "raw_rmse_m_s2":float(np.sqrt(np.mean((b-a)**2))),
            "raw_correlation":float(np.corrcoef(a,b)[0,1]),
            "contact_sample_fraction":float(np.mean(active)),
            "fundamental_phase_error_deg":float(np.angle(predicted/measured,deg=True)),
            "fundamental_amplitude_ratio":float(np.abs(predicted/measured)),
        })
    return rows


def plot(records, levels, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig,axes = plt.subplots(1,3,figsize=(11.4,3.05),layout="constrained")
    curves = (
        ("acc_tip_ms2","Measured","#222e35","-"),
        ("brep_ms2","STEP contact","#007d8e","-"),
        ("linear_ms2","Contact disabled","#a06c57","--"),
    )
    for ax,name,limits,title in (
        (axes[0],"06_stepped_sine",(91,91.40),"Held-out 13 Hz excitation"),
        (axes[2],"03_random",(20,20.75),"Held-out stronger random input"),
    ):
        record = records[name]
        time = record["time_s"]
        keep = (time>=limits[0])&(time<=limits[1])
        for key,label,color,style in curves:
            ax.plot(time[keep],record[key][keep],label=label,color=color,ls=style,lw=1.1)
        ax.set(xlim=limits,xlabel="Recorded time (s)",ylabel="Tip acceleration (m/s²)",title=title)
    mid = axes[1]
    center = [(row["begin_s"]+row["end_s"])/2 for row in levels]
    for key,label,color,style in (("measured_rms_m_s2","Measured","#222e35","-"),
                                  ("brep_rms_m_s2","STEP contact","#007d8e","-"),
                                  ("linear_rms_m_s2","Contact disabled","#a06c57","--")):
        mid.plot(center,[row[key] for row in levels],color=color,ls=style,lw=1.1,marker="o",ms=3,label=label)
    mid.set(xlabel="10 s block center (s)",ylabel="Tip acceleration RMS (m/s²)",title="Stepped-sine amplitude")
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.22)
    mid.legend(frameon=False,fontsize=7.4,loc="best")
    fig.savefig(output/"cantilever-gap-response.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work",type=Path)
    parser.add_argument("results",type=Path)
    args = parser.parse_args()
    args.results.mkdir(parents=True,exist_ok=True)
    summary = json.loads((args.work/"summary.json").read_text(encoding="utf-8"))
    summary["source_license"] = "CC BY 4.0"
    summary["source_authors"] = ["Dario Anastasio","Stefano Marchesiello"]
    records = {}
    for name in HELD_OUT:
        path = args.work/name/"paired.csv"
        values = np.genfromtxt(path,delimiter=",",names=True)
        records[name] = values
        # Preserve the sampled values and their shared experiment clock.
        np.savetxt(args.results/f"{name}.csv.gz",values.view(np.float64).reshape(len(values),-1),
                   fmt="%.12g",delimiter=",",comments="",header=",".join(values.dtype.names))
    summary["stepped_sine_blocks"] = blocks(records["06_stepped_sine"])
    (args.results/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    shutil.copyfile(args.work/"fit.json",args.results/"fit.json")
    with zipfile.ZipFile(args.results/"step-replay.zip","w",compression=zipfile.ZIP_DEFLATED) as archive:
        for file in (*args.work.glob("*.step"),*(args.work/n/"brep"/"cantilever.scene" for n in HELD_OUT)):
            archive.write(file,file.relative_to(args.work))
    plot(records,summary["stepped_sine_blocks"],args.results)


if __name__=="__main__":
    main()
