"""Compare complete reducer simulations on the same recorded 0.1 ms grid."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cpu_backend_run import digest
from reducer_matched_responses import read_run, rms


def compare(previous, trace, deck, out, contact_only=None, refined=None):
    with np.load(previous) as d:
        t = d["time_s"]
        series = {
            "Previous CPU": (d["cpu_output_torque_Nm"], d["cpu_speed"], d["cpu_te_mrad"]),
            "RecurDyn": (d["rd_output_torque_Nm"], d["rd_speed"], d["rd_te_mrad"]),
        }
    current = read_run(trace, deck)
    np.testing.assert_allclose(t, np.arange(50001)*.0001, rtol=0, atol=1e-10)
    np.testing.assert_allclose(current["t"], t, rtol=0, atol=1e-10)
    series["Revised CPU"] = (current["torque"][:, 2], current["speed"], current["te"])
    if contact_only is not None:
        control=read_run(contact_only, deck)
        np.testing.assert_allclose(control["t"],t,rtol=0,atol=1e-10)
        series["Contact repair with previous HHT"]=(control["torque"][:,2],control["speed"],control["te"])
    if refined is not None:
        half=read_run(refined, deck)
        np.testing.assert_allclose(half["t"],t,rtol=0,atol=1e-10)
        series["Revised CPU at half step"]=(half["torque"][:,2],half["speed"],half["te"])
    if any(not np.isfinite(x).all() for channels in series.values() for x in channels):
        raise ValueError("Incomplete reducer response")
    order = ("Previous CPU", "Revised CPU", "RecurDyn")
    colors = ("#586677", "#087668", "#AE5126")
    ref, ref_v, ref_te = series["RecurDyn"]
    # Consecutive seconds own each output sample once. Assign 5 s to the final second.
    seconds = np.minimum(np.arange(len(t))//10000, 4)
    report = dict(samples=len(t), step_s=.0001, reference_and_previous_sha256=digest(previous),
                  revised_trace_sha256=digest(trace), method={})
    if contact_only is not None:
        report["contact_only_trace_sha256"]=digest(contact_only)
    if refined is not None:
        report["refined_trace_sha256"]=digest(refined)
    for name, (y, v, te) in series.items():
        windows = {}
        for label, keep in [("full", np.ones(len(t), bool)), *[(f"{i}-{i+1}s", seconds==i) for i in range(5)]]:
            yy, tt = y[keep], t[keep]
            windows[label] = dict(samples=len(yy), min_Nm=float(yy.min()), peak_Nm=float(yy.max()),
                peak_time_s=float(tt[np.argmax(yy)]), mean_Nm=float(yy.mean()),std_Nm=float(yy.std()),
                torque_rms_difference_Nm=float(rms(yy-ref[keep])),
                speed_rms_difference_rad_s=rms(v[keep]-ref_v[keep]).tolist(),
                transmission_error_rms_difference_mrad=rms(te[keep]-ref_te[keep]).tolist())
        report["method"][name] = windows
    out.mkdir(parents=True, exist_ok=True)
    (out/"comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    np.savez_compressed(out/"revised-response.npz", time_s=t, speed_rad_s=current["speed"],
                        transmission_error_mrad=current["te"], output_contact_torque_Nm=current["torque"][:,2],
                        stage_normal_load_N=current["normal"])
    if contact_only is not None:
        np.savez_compressed(out/"contact-only-response.npz",time_s=t, speed_rad_s=control["speed"],
                            transmission_error_mrad=control["te"],output_contact_torque_Nm=control["torque"][:,2])
    if refined is not None:
        np.savez_compressed(out/"half-step-response.npz",time_s=t,speed_rad_s=half["speed"],
                            transmission_error_mrad=half["te"],output_contact_torque_Nm=half["torque"][:,2])
    with plt.rc_context({"font.size":9, "path.simplify":False, "axes.spines.top":False,
                         "axes.spines.right":False}):
        fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(9.4,6.5), layout="constrained")
        for ax, name, color in zip(axes, order, colors, strict=True):
            y=series[name][0]
            ax.plot(t, y, lw=.38, color=color)
            peak=int(np.argmax(y))
            ax.plot(t[peak],y[peak],"o",ms=4,mfc="white",mec=color,clip_on=False)
            ax.set_title(name, loc="left", color=color)
            ax.set_title(f"Peak {y[peak]:.2f} N·m at {1000*t[peak]:.1f} ms",loc="right",fontsize=9)
            ax.set_ylabel("Torque (N·m)")
            ax.grid(axis="y", alpha=.2)
        axes[-1].set(xlabel="Time (s)", xlim=(0,5))
        fig.suptitle("Output shaft contact torque")
        fig.savefig(out/"output-torque.png", dpi=210)
        plt.close(fig)
        fig, axes = plt.subplots(5,3,sharex=True,sharey="row",figsize=(11,8.7),layout="constrained")
        labels = ("Input speed\n(rad/s)", "Intermediate speed\n(rad/s)", "Output speed\n(rad/s)",
                  "Stage 1 error\n(mrad)", "Stage 2 error\n(mrad)")
        for col,(name,color) in enumerate(zip(order,colors,strict=True)):
            _, v, te = series[name]
            axes[0,col].set_title(name,color=color,loc="left")
            for row, y in enumerate(np.column_stack([v, te]).T):
                ax=axes[row,col]
                ax.plot(t,y,lw=.38,color=color)
                ax.grid(axis="y",alpha=.2)
                if not col: ax.set_ylabel(labels[row])
            axes[-1,col].set(xlabel="Time (s)", xlim=(0,5))
        fig.savefig(out/"shaft-responses.png",dpi=210)
        plt.close(fig)
        if refined is not None:
            fig,axes=plt.subplots(3,1,sharex=True,sharey=True,figsize=(9.4,6.5),layout="constrained")
            for ax, (name, label, color) in zip(axes,(
                    ("Revised CPU","Revised CPU · 100 µs","#087668"),
                    ("Revised CPU at half step","Revised CPU · 50 µs","#285C91"),
                    ("RecurDyn","RecurDyn · 100 µs maximum","#AE5126")),strict=True):
                ax.plot(t,series[name][0],lw=.38,color=color)
                ax.set_title(label,loc="left",color=color)
                ax.set_ylabel("Torque (N·m)")
                ax.grid(axis="y",alpha=.2)
            axes[-1].set(xlabel="Time (s)",xlim=(0,5))
            fig.suptitle("Output shaft contact torque on the common 100 µs output grid")
            fig.savefig(out/"step-size.png",dpi=210)
            plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("previous","trace","deck","out"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--contact-only",type=Path)
    parser.add_argument("--refined",type=Path)
    compare(**vars(parser.parse_args()))
