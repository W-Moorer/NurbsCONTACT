"""Compare matched braking transients on independently refined time grids."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reducer_matched_responses import read_run,rms


def compare(deck, cpu_files, native_files, out):
    with np.load(deck/"assembly.npz") as cfg:
        nominal=cfg["bodies"][0,8]
    out.mkdir(parents=True,exist_ok=True)
    traces={}; metrics={}; reference_times=None
    fig,axes=plt.subplots(2,2,figsize=(11,6.8),layout="constrained")
    colors=["#bd602a","#5295a0","#194664","#93528e"]
    sizes=[]
    for index,(cpu_file,rd_file) in enumerate(zip(cpu_files,native_files,strict=True)):
        cpu=read_run(cpu_file,deck)
        with np.load(rd_file) as rd:
            t=rd["reference_rad"][:,0]/nominal
            np.testing.assert_allclose(cpu["t"],t,rtol=0,atol=1e-9)
            data={"CPU":(cpu["torque"][:,2],cpu["speed"][:,2]),"RecurDyn":(rd["contact_raw_TZ_Nm"][:,1],rd["global_speed_rad_s"][:,2])}
        dt=f"{1e6*(t[1]-t[0]):.6g}"
        sizes.append(dt)
        mask=(t>=.102-1e-9)&(t<=.114+1e-9)
        metrics[dt]={}
        for col,(name,(torque,speed)) in enumerate(data.items()):
            metrics[dt][name]=dict(mean_torque_Nm=float(np.trapezoid(torque[mask],t[mask])/(t[mask][-1]-t[mask][0])),
                angular_impulse_Nms=float(np.trapezoid(torque[mask],t[mask])),peak_Nm=float(torque[mask].max()),
                min_speed_rad_s=float(speed[mask].min()),max_speed_rad_s=float(speed[mask].max()))
            axes[0,col].plot(t[mask]*1e3,torque[mask],c=colors[index],lw=1,label=f"{dt} µs")
            axes[1,col].plot(t[mask]*1e3,speed[mask],c=colors[index],lw=1)
            axes[0,col].set_title(name)
            if reference_times is None: reference_times=t[mask].copy()
            ids=np.rint(reference_times/(t[1]-t[0])).astype(int)
            np.testing.assert_allclose(t[ids],reference_times,rtol=0,atol=1e-9)
            traces[f"{name}_{dt}us"]=np.column_stack((torque[ids],speed[ids]))
        metrics[dt]["between_solvers_torque_rms_Nm"]=float(rms(data["CPU"][0][mask]-data["RecurDyn"][0][mask]))
    metrics["within_solver_rms_on_100us_grid"]={}
    for name in ("CPU","RecurDyn"):
        metrics["within_solver_rms_on_100us_grid"][name]={
            f"{a}_to_{b}us":rms(traces[f"{name}_{a}us"]-traces[f"{name}_{b}us"]).tolist()
            for a,b in zip(sizes,sizes[1:])}
    for a in axes.flat:a.grid(alpha=.25);a.spines[["top","right"]].set_visible(False);a.set_xlabel("Time (ms)")
    axes[0,0].set_ylabel("Output contact torque (N m)");axes[1,0].set_ylabel("Output speed (rad/s)")
    axes[0,0].legend(frameon=False,ncol=3,fontsize=8)
    fig.suptitle("Time-step sensitivity during brake engagement")
    fig.savefig(out/"step-refinement.png",dpi=155);plt.close(fig)
    np.savez_compressed(out/"step-refinement.npz",time_s=reference_times,**traces)
    (out/"step-refinement.json").write_text(json.dumps(metrics,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("deck",type=Path);p.add_argument("out",type=Path)
    p.add_argument("--cpu",dest="cpu_files",type=Path,nargs="+",required=True)
    p.add_argument("--native",dest="native_files",type=Path,nargs="+",required=True)
    args=p.parse_args()
    if len(args.cpu_files)!=len(args.native_files) or not 2<=len(args.cpu_files)<=4:
        p.error("Supply two to four matched step sizes, coarsest first")
    compare(**vars(args))
