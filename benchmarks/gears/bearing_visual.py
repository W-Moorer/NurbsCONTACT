"""Render the STEP bearing, roller load zone, and HHT shaft orbit."""

import argparse
import gc
import json
from pathlib import Path

import av
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image, ImageDraw
import pyvista as pv

from bearing_run import CAGE_RPM, ROLLERS, SHAFT_RPM, track
from plots import BLUE, GOLD, GRID, INK, MUTED, label, save, style
from render import SIZE, font, reserve_caption, triangulation


CAGE_SPEED=CAGE_RPM*2*np.pi/60
SHAFT_SPEED=SHAFT_RPM*2*np.pi/60


class BearingPainter:
    def __init__(self,root,view,maximum):
        self.maximum=maximum
        self.plotter=pv.Plotter(off_screen=True,window_size=SIZE)
        self.plotter.set_background("#eaf1f7")
        self.plotter.enable_anti_aliasing("fxaa")
        properties={"ambient":.30,"diffuse":.81,"specular":.30,"smooth_shading":True,"split_sharp_edges":True}
        folder=(root/"models").resolve()
        self.plotter.add_mesh(triangulation(folder/"outer.step"),color="#7d92a7",**properties)
        self.inner=self.plotter.add_mesh(triangulation(folder/"inner.step"),color="#326b9f",**properties)
        self.rollers=[
            self.plotter.add_mesh(triangulation(folder/"roller.step").copy(),color="#9ac2cb",**properties)
            for _ in range(ROLLERS)
        ]
        direction=np.array([.45,-.90,1.2] if view=="isometric" else [0.,0.,1.])
        up=(0,0,1) if view=="isometric" else (0,1,0)
        self.plotter.camera_position=(direction*175,(0,0,0),up)
        self.plotter.enable_parallel_projection()
        self.plotter.camera.parallel_scale=56

    def image(self,row):
        time=row["time"]
        self.inner.position=(row["x_um"]/1000,row["y_um"]/1000,0)
        self.inner.orientation=(0,0,np.degrees(np.remainder(SHAFT_SPEED*time,2*np.pi)))
        for i,actor in enumerate(self.rollers):
            theta=CAGE_SPEED*time+2*np.pi*i/ROLLERS
            radius=row[f"radius{i}_mm"]
            load=row[f"roller{i}_N"]/self.maximum
            actor.position=(radius*np.cos(theta),radius*np.sin(theta),0)
            actor.prop.color=tuple(
                np.array([.50,.72,.76])*(1-load)+np.array([.98,.41,.12])*load
            )
        self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.render()
        return Image.fromarray(self.plotter.screenshot(return_img=True)).convert("RGB")

    def close(self):
        self.plotter.close()


def caption(image,view,row):
    width,_=image.size
    d=ImageDraw.Draw(image)
    d.rectangle((0,0,width,71),fill="#f4f8fb")
    d.line((0,71,width,71),fill="#cbd7e3")
    d.text((21,12),"Cylindrical roller bearing | 45 × 85 × 19 mm",fill=INK,font=font(22,True))
    d.text((22,44),f"{view.capitalize()} view  |  Amber intensity indicates contact force",fill=MUTED,font=font(13))
    d.text((width-22,15),f"t = {row['time']:.4f} s",anchor="ra",fill=INK,font=font(18,True))
    d.text((width-22,45),f"Peak roller load {row['peak_roller_N']:.0f} N",anchor="ra",fill=MUTED,font=font(13))
    return np.asarray(image)


def video(filename,painter,view,trace):
    count=144
    first,last=.155,.205
    indices=np.searchsorted(trace["time"],np.linspace(first,last,count))
    with av.open(str(filename),"w",options={"movflags":"faststart"}) as output:
        stream=output.add_stream("libx264",rate=24)
        stream.width,stream.height=SIZE
        stream.pix_fmt="yuv420p"
        stream.options={"crf":"19","preset":"faster"}
        for index in indices:
            row=trace[index]
            frame=av.VideoFrame.from_ndarray(caption(painter.image(row),view,row),format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return {"file":filename.name,"case":"bearing","view":view,"physical_s":last-first,"starting_s":first,
            "video_s":6,"frames":count,"fps":24,"width":SIZE[0],"height":SIZE[1],"source":"runs/cad.csv"}


def figure(root,cad,mesh,row):
    folder=root/"figures"
    folder.mkdir(exist_ok=True)
    style()
    fig=plt.figure(figsize=(7.2,4.55),layout="constrained")
    layout=fig.add_gridspec(2,2)
    model=fig.add_subplot(layout[0,0])
    polar=fig.add_subplot(layout[0,1],projection="polar")
    orbit=fig.add_subplot(layout[1,0])
    force=fig.add_subplot(layout[1,1])
    with Image.open(root/"images"/"bearing-isometric.png") as img:
        model.imshow(img)
    model.set_axis_off()
    label(model,"a","STEP rolling elements and raceways")
    theta=CAGE_SPEED*row["time"]+2*np.pi*np.arange(ROLLERS)/ROLLERS
    values=np.array([row[f"roller{i}_N"] for i in range(ROLLERS)])
    polar.bar(theta,values,width=.16,color=GOLD,alpha=.85,zorder=3)
    polar.set_xticks(np.arange(4)*np.pi/2,labels=("0°","90°","180°","270°"))
    polar.set_rlabel_position(120)
    polar.yaxis.set_major_locator(MaxNLocator(4))
    polar.tick_params(labelsize=7)
    polar.grid(color=GRID,linewidth=.6)
    label(polar,"b","Roller load at one instant (N)")
    for history,col,name in ((mesh,GOLD,"Fine mesh"),(cad,BLUE,"B-rep")):
        use=(history["time"]>=.24)&(history["time"]<=.24+60/SHAFT_RPM)
        orbit.plot(history["x_um"][use],history["y_um"][use],color=col,lw=.95,label=name)
        use=(history["time"]>=.16)&(history["time"]<=.205)
        force.plot(history["time"][use]*1000,history["peak_roller_N"][use],color=col,lw=.95,label=name)
    orbit.set(xlabel="Shaft center X (µm)",ylabel="Shaft center Y (µm)")
    force.set(xlabel="Time (ms)",ylabel="Maximum roller load (N)")
    for axis in (orbit,force):
        axis.grid(color=GRID,linewidth=.55)
    orbit.legend()
    label(orbit,"c","Shaft orbit over one turn")
    label(force,"d","Load ramp and roller passage")
    save(fig,folder,"bearing-loads")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path)
    parser.add_argument("--pictures-only",action="store_true")
    args=parser.parse_args()
    root=args.results
    cad,mesh=track(root,"cad"),track(root,"mesh-fine")
    images,movies=root/"images",root/"animations"
    for folder in (images,movies):
        folder.mkdir(exist_ok=True)
    row=cad[np.searchsorted(cad["time"],.190)]
    entries=[]
    for view in ("isometric","face"):
        painter=BearingPainter(root,view,float(np.max(cad["peak_roller_N"])))
        try:
            painter.image(row).save(images/f"bearing-{view}.png")
            if not args.pictures_only:
                reserve_caption(painter.plotter)
                path=movies/f"bearing-{view}.mp4"
                entries.append(video(path,painter,view,cad))
                print(f"Saved {path}",flush=True)
        finally:
            painter.close()
    if entries:
        (movies/"index.json").write_text(json.dumps(entries,indent=2)+"\n",encoding="utf-8")
    figure(root,cad,mesh,row)
    triangulation.cache_clear()
    gc.collect()


if __name__=="__main__":
    main()
