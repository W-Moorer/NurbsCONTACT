"""Render the exported STEP parts at poses recorded by the shared HHT runner."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyvista as pv
from scipy.spatial.transform import Rotation

from run import read_csv


SIZE=(960,540)
VIEWS={"isometric":(1.,-1.6,.9)}


def read_geometry(scene):
    return [(scene.parent/tokens[1],np.array(tokens[6:9],dtype=float))
            for line in scene.read_text().splitlines() if (tokens:=shlex.split(line))[0]=="body"]


def mesh(path, cache):
    npz=cache/(path.stem+".npz")
    worker=Path(__file__).resolve().parents[1]/"gears"/"tessellate.py"
    result=subprocess.run([sys.executable,str(worker),str(path),str(npz)],capture_output=True,text=True)
    if result.returncode or not npz.exists():
        raise RuntimeError(f"STEP tessellation preview failed: {result.stderr.strip()}")
    with np.load(npz) as data:
        faces=np.column_stack((np.full(len(data["triangles"]),3),data["triangles"]))
        return pv.PolyData(data["points"],faces.ravel())


def font(size, bold=False):
    path=Path("C:/Windows/Fonts")/("arialbd.ttf" if bold else "arial.ttf")
    return ImageFont.truetype(str(path),size) if path.exists() else ImageFont.load_default(size=size)


def render(root, cache):
    cache.mkdir(parents=True,exist_ok=True)
    animations=root/"animations"
    thumbnails=root/"images"
    animations.mkdir(exist_ok=True)
    thumbnails.mkdir(exist_ok=True)
    index=[]
    for key,title in (("bounce-06","Ball rebound"),("plastic-06","Plastic block on a slope")):
        inputs=read_geometry(root/"scenes"/f"{key}.scene")
        surfaces=[mesh(path,cache) for path,_ in inputs]
        trace=read_csv(root/"runs"/f"{key}-brep.csv")
        observed=read_csv(root/"curves"/f"{key}.csv")
        t=np.linspace(0,observed["time_s"][-1],96)
        positions=[1000*np.column_stack([np.interp(t,trace["time_s"],trace[f"b{i}_{d}_m"]) for d in "xyz"])
                   for i in range(len(inputs))]
        angles=[np.interp(t,trace["time_s"],trace[f"b{i}_angle_rad"]) for i in range(len(inputs))]
        focus=(positions[1].min(axis=0)+positions[1].max(axis=0))/2
        if key.startswith("bounce"):
            focus[2]+=22
        for view,direction in VIEWS.items():
            p=pv.Plotter(off_screen=True,window_size=SIZE)
            p.set_background("#edf3f8")
            actors=[p.add_mesh(surface,color=color,ambient=.35,diffuse=.7,specular=.22,
                                smooth_shading=True,split_sharp_edges=True)
                    for surface,color in zip(surfaces,("#9bafbf","#126982"),strict=True)]
            p.camera_position=(focus+1500*np.asarray(direction),focus,(0,0,1))
            p.enable_parallel_projection()
            p.camera.parallel_scale=230 if key.startswith("bounce") else 245
            p.show(auto_close=False,interactive=False)
            target=animations/f"{key}-{view}.mp4"
            try:
                with av.open(str(target),"w",options={"movflags":"faststart"}) as output:
                    stream=output.add_stream("libx264",rate=24)
                    stream.width,stream.height=SIZE
                    stream.pix_fmt="yuv420p"
                    for i,time in enumerate(t):
                        for j,actor in enumerate(actors):
                            matrix=np.eye(4)
                            matrix[:3,:3]=Rotation.from_rotvec(angles[j][i]*inputs[j][1]).as_matrix()
                            matrix[:3,3]=positions[j][i]
                            actor.user_matrix=matrix
                        p.renderer.ResetCameraClippingRange()
                        p.render()
                        image=Image.fromarray(p.screenshot(return_img=True)).convert("RGB")
                        canvas=ImageDraw.Draw(image)
                        canvas.rectangle((0,0,SIZE[0],65),fill="#f7fafc")
                        canvas.text((20,10),title,fill="#1b324c",font=font(22,True))
                        canvas.text((20,38),f"{view.title()} view · STEP poses from the B-rep run",
                                    fill="#31536c",font=font(14))
                        canvas.text((SIZE[0]-20,20),f"{time:.3f} s",anchor="ra",fill="#1b324c",font=font(19,True))
                        if i==28:
                            image.save(thumbnails/f"{key}-{view}.png")
                        frame=av.VideoFrame.from_ndarray(np.asarray(image),format="rgb24")
                        for packet in stream.encode(frame):
                            output.mux(packet)
                    for packet in stream.encode():
                        output.mux(packet)
            finally:
                p.close()
            index.append({"scene":key,"view":view,"file":target.name,"frames":len(t),"fps":24,
                          "physical_duration_s":float(t[-1])})
    (animations/"index.json").write_text(json.dumps(index,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path)
    parser.add_argument("cache",type=Path)
    args=parser.parse_args()
    render(args.results,args.cache)
