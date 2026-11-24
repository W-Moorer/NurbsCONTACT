"""Render the STEP gear solids at their recorded HHT poses."""

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyvista as pv

from analyze import track
from run import CASES, csv_rows


LABELS = {
    "spur": "Spur pair | 24/48 teeth",
    "helical": "Helical pair | 25/45 teeth",
    "rig28": "Gear-noise test geometry | 28/28 teeth",
    "reducer": "Two-stage reducer | 4:1",
}
VIEWS = ("isometric","face")
SIZE = 960, 540


@lru_cache
def triangulation(path):
    script=Path(__file__).with_name("tessellate.py").resolve()
    mesh_key=hashlib.sha256(script.read_bytes()+path.read_bytes()).hexdigest()
    build=script.parents[2]/"build"/"gear-render-cache"
    build.mkdir(exist_ok=True,parents=True)
    cache=build/f"{mesh_key}.npz"
    if not cache.is_file():
        subprocess.run([sys.executable,str(script),str(path),str(cache)],check=True)
    with np.load(cache) as mesh:
        points=mesh["points"]
        triangles=mesh["triangles"]
    cells = np.column_stack((np.full(len(triangles),3),triangles))
    return pv.PolyData(points, cells.ravel())


@lru_cache
def font(size, bold=False):
    name="arialbd.ttf" if bold else "arial.ttf"
    windows=Path("C:/Windows/Fonts")/name
    if windows.is_file():
        return ImageFont.truetype(str(windows),size)
    name="DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name,size)
    except OSError:
        return ImageFont.load_default(size=size)


class Assembly:
    def __init__(self,root,case,view):
        self.layout=csv_rows(root/"models"/f"{case}.csv")
        p=pv.Plotter(off_screen=True,window_size=SIZE)
        p.set_background("#eaf1f7")
        p.enable_anti_aliasing("fxaa")
        self.plotter=p
        self.actors=[]
        mins=np.full(3,np.inf)
        maxs=-mins.copy()
        stage_colors={"1":("#3276ad","#849aaf"),"2":("#2a8c80","#8ab2a8")}
        for row in self.layout:
            location=np.array([float(row["center_x_mm"]),0.,float(row["z_mm"])])
            width=float(row["width_mm"])
            mesh=triangulation((root/"models"/row["step"]).resolve())
            actor=p.add_mesh(mesh,color=stage_colors[row["stage"]][row["side"]=="b"],
                             ambient=.28,diffuse=.82,specular=.32,specular_power=26,
                             smooth_shading=True,split_sharp_edges=True,show_edges=False)
            actor.position=tuple(location)
            self.actors.append(actor)
            radius=max(abs(np.array(mesh.bounds[:4])))
            dimension=np.array([radius,radius,width/2])
            mins=np.minimum(mins,location-dimension)
            maxs=np.maximum(maxs,location+dimension)
        self.centers=[]
        for row in self.layout:
            if row["side"]!="a":
                continue
            self.centers.append((
                float(row["center_x_mm"])+float(row["pitch_radius_mm"]),
                float(row["z_mm"]),float(row["direction"]),
                np.arccos(float(row["base_radius_mm"])/float(row["pitch_radius_mm"])),
            ))
        focus=(mins+maxs)/2
        direction=np.array([.42,-.83,1.24] if view=="isometric" else [0.,.03,1.])
        direction/=np.linalg.norm(direction)
        up=np.array([0.,0.,1.]) if view=="isometric" else np.array([0.,1.,0.])
        right=np.cross(up,direction)
        right/=np.linalg.norm(right)
        up=np.cross(direction,right)
        corners=np.array([[x,y,z] for x in (mins[0],maxs[0]) for y in (mins[1],maxs[1])
                          for z in (mins[2],maxs[2])])
        p.camera_position=(focus+direction*np.linalg.norm(maxs-mins)*2.5,focus,up)
        p.enable_parallel_projection()
        p.camera.parallel_scale=.59*max(np.ptp(corners@up),np.ptp(corners@right)*SIZE[1]/SIZE[0])

    def frame(self,time,values,contacts):
        for row,actor in zip(self.layout,self.actors,strict=True):
            speed=float(row["start_rpm"])*2*np.pi/60
            shaft=int(row["shaft"])
            angle=float(row["angle_zero_rad"])+float(row["direction"])*(speed*time+values[shaft])
            actor.orientation=(0,0,float(np.degrees(np.remainder(angle,2*np.pi))))
        self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.render()
        image=Image.fromarray(self.plotter.screenshot(return_img=True)).convert("RGB")
        draw=ImageDraw.Draw(image)
        for center,contact in zip(self.centers,contacts,strict=True):
            station,z=contact
            if np.isfinite(station):
                x,base_z,direction,alpha=center
                render=self.plotter.renderer
                render.SetWorldPoint(x+station*np.sin(alpha),station*np.cos(alpha)*direction,base_z+z,1)
                render.WorldToDisplay()
                px,py,_=render.GetDisplayPoint()
                py=image.height-1-py
                draw.ellipse((px-4,py-4,px+4,py+4),fill="#f08131",outline="#fcf4e2",width=2)
        return image

    def close(self):
        self.plotter.close()


def annotation(frame,title,view,physical,forces,load,start,final):
    draw=ImageDraw.Draw(frame)
    width,height=frame.size
    draw.rectangle((0,0,width,72),fill="#f4f8fb")
    draw.line((0,71,width,71),fill="#cbd7e3")
    draw.text((22,12),title,fill="#18334e",font=font(22,True))
    draw.text((23,43),f"{view.capitalize()} view  |  Native B-rep  |  Strongest contact station",
              fill="#456078",font=font(13))
    draw.text((width-22,14),f"t = {physical:.4f} s",anchor="ra",fill="#18334e",font=font(18,True))
    draw.text((width-22,43),f"Load {load:.1f} N m  |  "+" / ".join(f"{f:.0f} N" for f in forces),
              anchor="ra",fill="#456078",font=font(13))
    draw.rectangle((0,height-4,width,height),fill="#cad8e4")
    draw.rectangle((0,height-4,width*(physical-start)/(final-start),height),fill="#287e83")
    return np.asarray(frame)


def reserve_caption(plotter):
    camera=plotter.camera
    camera.parallel_scale*=1.12
    shift=np.asarray(camera.up)*camera.parallel_scale*72/SIZE[1]
    camera.position=tuple(np.asarray(camera.position)+shift)
    camera.focal_point=tuple(np.asarray(camera.focal_point)+shift)


def movie(path,assembly,label,view,trace,seconds,fps):
    n=len(assembly.centers)
    start,final=.115,.195
    count=round(seconds*fps)
    # A uniformly slowed playback keeps individual teeth visible.
    times=np.linspace(start,final,count)
    indices=np.searchsorted(trace["time"],times)
    values=[np.interp(times,trace["time"],trace[f"q{k}"]) for k in range(n+1)]
    loads=np.interp(times,trace["time"],trace["load_Nm"])
    with av.open(str(path),"w",options={"movflags":"faststart"}) as output:
        stream=output.add_stream("libx264",rate=fps)
        stream.width,stream.height=SIZE
        stream.pix_fmt="yuv420p"
        stream.options={"crf":"19","preset":"faster"}
        for k,(t,index) in enumerate(zip(times,indices,strict=True)):
            row=trace[index]
            contacts=[(row[f"station{i}_mm"],row[f"z{i}_mm"]) for i in range(n)]
            f=assembly.frame(t,[q[k] for q in values],contacts)
            args=(f,LABELS[label],view,t,[row[f"normal{i}_N"] for i in range(n)],loads[k],start,final)
            pixels=annotation(*args)
            picture=av.VideoFrame.from_ndarray(pixels,format="rgb24")
            for packet in stream.encode(picture):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return count,start,final


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path)
    parser.add_argument("--pictures-only",action="store_true")
    parser.add_argument("--case",choices=CASES)
    parser.add_argument("--fps",type=int,default=24)
    args=parser.parse_args()
    root=args.results
    png,mp4=root/"images",root/"animations"
    for folder in (png,mp4):
        folder.mkdir(exist_ok=True)
    entries=[]
    for case in ((args.case,) if args.case else CASES):
        trace=track(root,case+"-cad")
        row=trace[np.searchsorted(trace["time"],.133)]
        stages=2 if case=="reducer" else 1
        for view in VIEWS:
            assembly=Assembly(root,case,view)
            try:
                points=[(row[f"station{i}_mm"],row[f"z{i}_mm"]) for i in range(stages)]
                img=assembly.frame(row["time"],[row[f"q{i}"] for i in range(stages+1)],points)
                img.save(png/f"{case}-{view}.png")
                if not args.pictures_only and view=="isometric":
                    reserve_caption(assembly.plotter)
                    path=mp4/f"{case}-{view}.mp4"
                    frames,start,final=movie(path,assembly,case,view,trace,8,args.fps)
                    entries.append({"file":path.name,"case":case,"view":view,"physical_s":final-start,"starting_s":start,
                                    "video_s":8,"frames":frames,"fps":args.fps,"width":SIZE[0],"height":SIZE[1],
                                    "source":f"runs/{case}-cad.csv"})
                    print(f"Saved {path} ({frames} frames)",flush=True)
            finally:
                assembly.close()
    if entries and not args.case:
        (mp4/"index.json").write_text(json.dumps(entries,indent=2)+"\n",encoding="utf-8")
    triangulation.cache_clear()
    gc.collect()


if __name__=="__main__":
    main()
