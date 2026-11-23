"""Plot loaded reducer dynamics and animate the measured poses of all four meshes."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def envelope(ax, t, y, color, label):
    # Equal 10 ms bins for the overview; the adjacent detail uses every sample.
    blocks = y[:-1].reshape(-1, 100)
    centers = t[:-1].reshape(-1, 100).mean(axis=1)
    ax.fill_between(centers, blocks.min(axis=1), blocks.max(axis=1), color=color, alpha=.17)
    ax.plot(centers, blocks.mean(axis=1), color=color, lw=1, label=label)


def assembly_figure(render_dir, output):
    fig,axes=plt.subplots(1,3,figsize=(11,3.2),gridspec_kw={"width_ratios":[1.65,1,1]},layout="constrained")
    views=(("overview","Shared-shaft reducer"),("spur","Stage 1 · spur mesh"),("helical","Stage 2 · helical mesh"))
    for ax,(file,title) in zip(axes,views):
        ax.imshow(plt.imread(render_dir/(file+".png")))
        ax.set_title(title,loc="left",fontsize=11)
        ax.axis("off")
    fig.savefig(output/"assembly.png",dpi=175)
    plt.close(fig)


def time_breakdown(work, out):
    dynamics=json.loads((out/"dynamics.json").read_text())
    runs=list(dynamics["gpu"].values())+[dynamics["contact_time_resolution"]["8"]["measurement"]]
    labels=("Projection · 16","Blocks 0.2 mm · 16","Blocks 0.8 mm · 16","Blocks 0.8 mm · 8")
    fig,(gpu,cpu)=plt.subplots(1,2,figsize=(10,3.3),layout="constrained")
    y=np.arange(len(runs))
    offset=np.zeros(len(runs))
    for key,color,label in (("prepare_s","#7c9ea6","Prepare candidates"),("search_s","#28717d","Search triangles"),
                            ("merge_and_hht_s","#cc8256","Force and HHT")):
        values=np.array([v["phase_profile"][key] for v in runs])
        gpu.barh(y,values,left=offset,color=color,label=label)
        offset+=values
    for j,run in enumerate(runs): gpu.text(offset[j]+2,j,f'{run["online_s"]:.1f}',va="center",fontsize=8)
    gpu.set(yticks=y,yticklabels=labels,xlim=(0,196),xlabel="Online time (s)",title="GPU · complete 5 s dynamics")
    gpu.invert_yaxis()
    gpu.legend(loc="upper left",bbox_to_anchor=(0,-.23),ncol=2,fontsize=8)
    settings=(("Historical OBB",work/"replay-before.json"),("Same face",work/"paired-same.json"),
              ("Same face + neighbors",work/"paired-neighbors.json"))
    data=np.array([[s["query_s"] for s in json.loads(file.read_text())["methods"]["mesh-obb"]] for _,file in settings])
    y=np.arange(len(data))
    cpu.barh(y,data[:,0],color="#527397",label="Spur stage")
    cpu.barh(y,data[:,1],left=data[:,0],color="#acb9c7",label="Helical stage")
    for j,v in enumerate(data.sum(axis=1)): cpu.text(v+.6,j,f"{v:.1f}",va="center",fontsize=8)
    cpu.set(yticks=y,yticklabels=[name for name,_ in settings],xlim=(0,data.sum(axis=1).max()*1.16),
            xlabel="Query time (s)",title="CPU · overlap at 100,002 poses")
    cpu.invert_yaxis()
    cpu.legend(loc="upper left",bbox_to_anchor=(0,-.23),ncol=2,fontsize=8)
    fig.savefig(out/"timings.png",dpi=175)
    plt.close(fig)


def switching(work, out):
    """Compare geometric overlap and normal load at every saved pose."""
    with np.load(out/"responses.npz") as result, np.load(work/"replay-before.npz") as mask:
        t=result["time_s"]
        loads=result["stage_load_N"]
        fig,axes=plt.subplots(2,2,figsize=(10,5),layout="constrained",sharex=True)
        show=t<=.010
        for stage in range(2):
            flags=mask[f"mesh-obb_{stage}"]
            if not all(np.array_equal(flags,mask[f"{method}_{stage}"]) for method in ("mesh-aabb","mesh-sap")):
                raise ValueError("CPU overlap flags disagree")
            ax,force=axes[:,stage]
            ax.step(t[show]*1000,flags[show].astype(float),where="post",color="#c56435",label="Triangle overlap (CPU)")
            ax.step(t[show]*1000,(loads[show,stage]>0).astype(float),where="post",ls="--",
                    color="#257383",label="Positive normal load (GPU, 16 divisions)")
            ax.set(yticks=[0,1],yticklabels=["Off","On"],ylim=(-.13,1.35),title=f"Stage {stage+1}")
            for samples,name,color in ((8,"samples-8","#835e9a"),(16,"wide-blocks","#257383"),(32,"samples-32","#c56435")):
                raw=np.load(work/(name+".npy"),mmap_mode="r")
                force.plot(t[show]*1000,raw[show,6+4*stage]/1000,color=color,lw=.95,label=f"{samples} divisions")
            force.set(xlabel="Simulated time (ms)",ylabel="Normal load (kN)",xlim=(0,10))
            for axis in (ax,force): axis.grid(axis="y",alpha=.16)
        axes[0,0].legend(fontsize=7,loc="upper right",framealpha=.85)
        axes[1,0].legend(fontsize=8)
        fig.savefig(out/"switching.png",dpi=170)
        plt.close(fig)


def independent_full_run(work, out):
    import re
    import shutil
    source=work/"recurdyn"
    with np.load(source/"physics.npz") as rd, np.load(out/"responses.npz") as own, np.load(work/"deck/assembly.npz") as deck:
        t=own["time_s"]
        axes=deck["bodies"][[0,1,3],3]
        rt=rd["reference_rad"][:,0]/deck["bodies"][0,8]
        if rt.shape!=t.shape or not np.allclose(rt,t,rtol=0,atol=1e-9):
            raise ValueError("expected 50,001 simultaneous RecurDyn and GPU frames")
        speed=rd["global_speed_rad_s"]*axes
        torque=rd["contact_raw_TZ_Nm"][:,1]
        q=rd["angle_rad"]
        twist=q[:,:2]-2*q[:,1:3]
        gpu_speed=own["shaft_speed_rad_s"]
        gpu_torque=own["contact_torques_Nm"][:,-1]
        gpu_twist=own["transmission_error_rad"]
        for values in (speed,torque,twist):
            if not np.isfinite(values).all(): raise ValueError("nonfinite reference channel")
        def rms(x): return np.sqrt(np.mean(x*x,axis=0)).tolist()
        info=json.loads((source/"run.json").read_text())
        logs=sorted(source.glob("rd-reducer-5s-100us_*/rd-reducer-5s-100us.msg"),key=lambda p:p.stat().st_mtime_ns)
        if logs:
            content=logs[-1].read_text(errors="replace")
            if "Analysis is accomplished successfully" not in content:
                raise ValueError("native solver log does not report a completed analysis")
            found=re.search(r"CPU Time\s*=\s*(\d+(?:\.\d+)?) Second",content)
            if found: info["solver_reported_cpu_s"]=float(found[1])
        info.update(shared_samples=len(t),shaft_speed_rms_difference_rad_s=rms(gpu_speed-speed),
                    transmission_error_rms_difference_mrad=rms(1000*(gpu_twist-twist)),
                    output_contact_torque_rms_difference_Nm=rms(gpu_torque-torque),
                    mean_output_contact_torque_Nm={"mesh_hht":float(gpu_torque.mean()),"RecurDyn":float(torque.mean())})
        info["time_refinements"]={}
        with np.load(out/"substeps.npz") as refinements:
            for n in (8,16,32):
                velocity=gpu_speed if n==16 else refinements[f"shaft_speed_{n}_rad_s"]
                moment=gpu_torque if n==16 else refinements[f"output_torque_{n}_Nm"]
                info["time_refinements"][str(n)]=dict(shaft_speed_rms_difference_rad_s=rms(velocity-speed),
                    output_torque_rms_difference_Nm=rms(moment-torque),mean_output_torque_Nm=float(moment.mean()))
        fig,plots=plt.subplots(3,2,figsize=(10,7),layout="constrained",width_ratios=[1.3,1])
        nominal=deck["bodies"][3,8]
        channels=((gpu_speed[:,2]-nominal,speed[:,2]-nominal,"Output speed ripple (rad/s)"),
                  (1000*gpu_twist[:,1],1000*twist[:,1],"Stage 2 transmission error (mrad)"),
                  (gpu_torque,torque,"Output contact torque (N m)"))
        for (ours,reference,label),(left,right) in zip(channels,plots):
            for values,color,name in ((ours,"#257383","Mesh / HHT"),(reference,"#c56435","RecurDyn")):
                envelope(left,t,values,color,name)
                select=(t>=.102)&(t<=.114)
                right.plot(t[select]*1000,values[select],lw=1,label=name,color=color)
            left.set(ylabel=label,xlim=(0,5))
            right.set(xlim=(102,114))
            for ax in (left,right): ax.grid(axis="y",alpha=.16)
        plots[0,0].set_title("Five seconds  ·  10 ms mean and range",loc="left")
        plots[0,1].set_title("Brake loading  ·  every 0.1 ms sample",loc="left")
        plots[0,0].legend(fontsize=8)
        plots[-1,0].set_xlabel("Simulated time (s)")
        plots[-1,1].set_xlabel("Simulated time (ms)")
        fig.savefig(out/"recurdyn-5s.png",dpi=180)
        plt.close(fig)
        (out/"recurdyn-5s.json").write_text(json.dumps(info,indent=2)+"\n")
    shutil.copyfile(source/"physics.npz",out/"recurdyn-5s.npz")


def plot(work, out):
    names=("loaded-projection","loaded-blocks","wide-blocks")
    runs={k:json.loads((work/(k+".json")).read_text()) for k in names}
    x=np.load(work/"wide-blocks.npy")
    for name, run in runs.items():
        other=np.load(work/(name+".npy"))
        if (run["terminated"]!="complete" or run["steps"]!=50000 or x.shape!=(50001,14) or
            not np.isfinite(other).all() or not np.array_equal(x, other)):
            raise ValueError(f"incomplete or changed mesh trajectory: {name}")
    with np.load(work/"deck"/"assembly.npz") as d:
        bodies, coord, loads, h = (d[k] for k in ("bodies","coord","loads","h"))
    t=np.arange(len(x))*h
    nominal=bodies[[0,1,3],8]
    speed=x[:,3:6]+nominal
    angle=x[:,:3]+t[:,None]*nominal
    twist=(x[:,:2]-x[0,:2])-2*(x[:,1:3]-x[0,1:3])
    force=x[:,[6,10]]
    torque=x[:,[7,8,11,12]]
    reference=Path("results/aes/resident-reducer/physical/native-exports/rd-physics.npz")
    with np.load(reference) as rd:
        rt=rd["reference_rad"][:,0]/nominal[0]
        rv=rd["global_speed_rad_s"]*bodies[[0,1,3],3]
        ra=rd["angle_rad"]
        rtorque=rd["contact_raw_TZ_Nm"][:,1]
    rr=np.arange(0,len(rt),4)
    ii=np.rint(rt[rr]/h).astype(int)
    if not np.allclose(t[ii],rt[rr],rtol=0,atol=1e-10):
        raise ValueError("reference time grids do not coincide")
    # Nominal ratios are used only to report transmission error.
    rtwist=ra[:,:2]-2*ra[:,1:3]
    def rms(y):
        return np.sqrt(np.mean(y*y,axis=0)).tolist()
    stats=dict(steps=50000, duration_s=float(t[-1]), dt_s=float(h),
               mesh_triangles=sum(p["triangles"] for p in json.loads((work/"deck/models.json").read_text())["parts"]),
               speed_mean_rad_s=speed.mean(axis=0).tolist(), stage_force_peak_N=force.max(axis=0).tolist(),
               stage_contact_torque_mean_Nm=torque.mean(axis=0).tolist(),
               output_resisting_torque_mean_Nm=float(-loads[:,-1].mean()),
               output_support_mean_Nm=float(np.mean(-coord[2,1]*x[:,2]-coord[2,2]*x[:,5])),
               reference_s=float(rt[-1]), reference_shared_samples=len(ii), speed_RMS_rad_s=rms(speed[ii]-rv[rr]),
               transmission_error_RMS_mrad=rms(1000*(twist[ii]-rtwist[rr])),
               common_output_torque_mean_Nm=[float(torque[ii,-1].mean()),float(rtorque[rr].mean())],
               common_output_torque_RMS_difference_Nm=rms(torque[ii,-1]-rtorque[rr]),
               raw_array_sha256=hashlib.sha256(x.tobytes()).hexdigest(), gpu=runs)
    fine=np.load(work/"samples-32.npy")
    stats["contact_time_resolution"]={}
    refinement={}
    for samples,name in ((8,"samples-8"),(16,"wide-blocks"),(32,"samples-32")):
        data=np.load(work/(name+".npy"))
        run=json.loads((work/(name+".json")).read_text())
        if run["terminated"]!="complete" or data.shape!=x.shape or not np.isfinite(data).all():
            raise ValueError(f"incomplete substep refinement at {samples}")
        stats["contact_time_resolution"][str(samples)]=dict(measurement=run,
            speed_RMS_from_RecurDyn_rad_s=rms(data[ii,3:6]+nominal-rv[rr]),
            load_RMS_from_32_N=rms(data[:,[6,10]]-fine[:,[6,10]]),
            speed_RMS_from_32_rad_s=rms(data[:,3:6]-fine[:,3:6]),
            switches_from_32=np.count_nonzero((data[:,[6,10]]>0)!=(fine[:,[6,10]]>0),axis=0).tolist(),
            mean_output_torque_Nm=float(np.mean(data[:,12])))
        if samples!=16:
            refinement[f"shaft_speed_{samples}_rad_s"]=data[:,3:6]+nominal
            refinement[f"stage_load_{samples}_N"]=data[:,[6,10]]
            refinement[f"output_torque_{samples}_Nm"]=data[:,12]
    out.mkdir(parents=True,exist_ok=True)
    (out/"dynamics.json").write_text(json.dumps(stats,indent=2)+"\n")
    np.savez_compressed(out/"substeps.npz",time_s=t,**refinement)
    np.savez_compressed(out/"responses.npz",time_s=t,shaft_angle_rad=angle,shaft_speed_rad_s=speed,
                        transmission_error_rad=twist,stage_load_N=force,contact_torques_Nm=torque,
                        applied_torque_Nm=loads)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,
                         "axes.spines.right":False,"axes.spines.top":False,"legend.frameon":False})
    colors=("#257383","#c56435","#60528d")
    with PdfPages(out/"responses.pdf") as pdf:
        fig, axes=plt.subplots(3,2,figsize=(10,7),layout="constrained",width_ratios=[1.3,1])
        detail=(t>=.102)&(t<=.114)
        quantities=((speed-nominal,"Speed ripple (rad/s)",("Input","Intermediate","Output")),
                    (force/1000,"Normal load (kN)",("Stage 1","Stage 2")),
                    (torque[:,[1,3]],"Driven torque (N m)",("Stage 1","Stage 2")))
        for row,(values,unit,labels) in enumerate(quantities):
            left,right=axes[row]
            for j,label in enumerate(labels):
                envelope(left,t,values[:,j],colors[j],label)
                right.plot(t[detail]*1000,values[detail,j],lw=1.1,color=colors[j])
            left.set(ylabel=unit,xlim=(0,5))
            right.set(xlim=(102,114))
            left.legend(ncol=len(labels),loc="upper right",fontsize=8)
            for ax in (left,right): ax.grid(axis="y",alpha=.15)
        axes[0,0].set_title("Full duty cycle  ·  10 ms mean and range",loc="left")
        axes[0,1].set_title("Brake loading  ·  every 0.1 ms sample",loc="left")
        axes[-1,0].set_xlabel("Simulated time (s)")
        axes[-1,1].set_xlabel("Simulated time (ms)")
        fig.savefig(out/"responses.png",dpi=180)
        pdf.savefig(fig)
        plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(10,5.2),layout="constrained")
        for ax, key, a, b in zip(axes.ravel(),("Input speed (rad/s)","Intermediate speed (rad/s)",
                                      "Stage 2 transmission error (mrad)","Output contact torque (N m)"),
                                  (speed[:,0],speed[:,1],twist[:,1]*1000,torque[:,-1]),
                                  (rv[:,0],rv[:,1],rtwist[:,1]*1000,rtorque)):
            sel=(t>=.102)&(t<=.114)
            refsel=(rt>=.102)&(rt<=.114)
            ax.plot(t[sel]*1000,a[sel],color=colors[0],label="Mesh / HHT",lw=1.35)
            ax.plot(rt[refsel]*1000,b[refsel],color=colors[1],ls="--",label="RecurDyn",lw=1.15)
            ax.set(xlabel="Simulated time (ms)",ylabel=key)
            ax.grid(axis="y",alpha=.15)
        axes[0,0].legend()
        fig.savefig(out/"reference.png",dpi=180)
        pdf.savefig(fig)
        plt.close(fig)


def animate(work,out):
    import cv2
    import imageio.v2 as io
    import trimesh
    with np.load(out/"responses.npz") as d:
        angle, force, t = (d[k] for k in ("shaft_angle_rad","stage_load_N","time_s"))
    with np.load(work/"deck/assembly.npz") as d:
        bodies,pairs,mapping=(d[k] for k in ("bodies","pairs","mapping"))
    parts=json.loads((work/"deck/models.json").read_text())["parts"]
    cuts=[]
    for part in parts:
        with np.load(work/"deck"/part["archive"]) as d:
            nodes,faces=np.unique(d["tri"].reshape(-1,3),axis=0,return_inverse=True)
        mesh=trimesh.Trimesh(nodes,faces.reshape(-1,3),process=False)
        plane=mesh.section(plane_normal=[0,0,1],plane_origin=mesh.bounds.mean(axis=0))
        loops=[p[:,:2] for p in plane.discrete]
        if not loops: raise ValueError("missing transverse mesh section")
        cuts.append(loops)
    fps=50
    overview=np.arange(0,len(t),50)
    detail=np.arange(1000,1201)
    panels=((0,530),(531,1060))
    dark=(33,52,66)
    def label(frame,text,x,y,size=.53):
        cv2.putText(frame,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,dark,1,cv2.LINE_AA)
    with io.get_writer(out/"reducer.mp4",fps=fps,codec="libx264",quality=8,macro_block_size=1,
                       ffmpeg_log_level="error",output_params=["-pix_fmt","yuv420p"]) as writer:
        for mode, ids in (("Full 5 s / 0.25x playback",overview),("Loading detail / 0.005x playback",detail)):
            for idx in ids:
                frame=np.full((460,1060,3),250,np.uint8)
                label(frame,"Two-stage reducer  |  transverse mesh sections",24,34,.77)
                label(frame,f"{mode}     t = {t[idx]:.4f} s",24,63,.53)
                for stage,(first,last) in enumerate(panels):
                    pair=pairs[stage]
                    origins=bodies[pair,4]
                    radii=[max(np.linalg.norm(p,axis=1).max() for p in cuts[i]) for i in pair]
                    lo,hi=origins[0]-radii[0],origins[1]+radii[1]
                    scale=min((last-first-54)/(hi-lo),270/(2*max(radii)))
                    y0=236
                    for side,i in enumerate(pair):
                        phi=bodies[i,3]*(angle[idx,mapping[i]]+bodies[i,7])
                        c,s=np.cos(phi),np.sin(phi)
                        rot=np.array([[c,-s],[-s,-c]])
                        x0=(first+last)/2+(origins[side]-(lo+hi)/2)*scale
                        xy=[np.rint(p@rot*scale+[x0,y0]).astype(np.int32) for p in cuts[i]]
                        cv2.fillPoly(frame,xy,((112,170,177),(224,164,112))[side],cv2.LINE_AA)
                        cv2.polylines(frame,xy,True,dark,1,cv2.LINE_AA)
                        end=(round(x0+.8*radii[side]*scale*c),round(y0-.8*radii[side]*scale*s))
                        cv2.line(frame,(round(x0),y0),end,dark,2,cv2.LINE_AA)
                    label(frame,f"Stage {stage+1}   |   normal load  {force[idx,stage]/1000:.2f} kN",first+30,405,.56)
                cv2.line(frame,(25,435),(1035,435),(208,216,221),2)
                cv2.circle(frame,(25+round(1010*t[idx]/5),435),5,dark,-1)
                writer.append_data(frame)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("work",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--animate",action="store_true")
    a=p.parse_args()
    plot(a.work,a.output)
    if (a.work/"render/helical.png").exists(): assembly_figure(a.work/"render",a.output)
    if (a.work/"paired-neighbors.json").exists(): time_breakdown(a.work,a.output)
    if (a.work/"replay-before.npz").exists(): switching(a.work,a.output)
    if (a.work/"recurdyn/physics.npz").exists(): independent_full_run(a.work,a.output)
    if a.animate: animate(a.work,a.output)
