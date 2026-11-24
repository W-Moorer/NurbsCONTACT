"""Render CAD-derived bodies and accepted HHT poses with Python."""

import argparse
import json
from functools import lru_cache
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyvista as pv

from analyze import load_trace


SCENES = (
    ("drop", "drop-brep", "Single-ball impact", 4, ("isometric", "side")),
    ("cam-array", "cam-brep", "Coupled cam and follower array", 24, ("isometric", "front")),
    ("ball-train", "train-brep", "Driven ball train", 20, ("isometric", "top")),
)
COLORS = {"plate": "#697d94", "sphere": "#27968b", "cylinder": "#4473b2"}


def geometry_mesh(description):
    center = np.array(description["center"])
    shape = description["shape"]
    if shape == "sphere":
        return pv.Sphere(radius=description["radius"], center=center, theta_resolution=40, phi_resolution=32)
    if shape == "cylinder":
        return pv.Cylinder(center=center, direction=description["axis"], radius=description["radius"],
                           height=2*description["half"], resolution=64)
    u = np.array(description["u"])*description["half_u"]
    v = np.array(description["v"])*description["half_v"]
    points = np.array([center-u-v,center+u-v,center+u+v,center-u+v])
    return pv.PolyData(points, np.array([4,0,1,2,3]))


def data(source, label):
    description = json.loads((source/"runs"/f"{label}.json").read_text(encoding="utf-8"))
    return description,load_trace(source,label)


def sample(trace, times):
    count = len([k for k in trace.dtype.names if k.startswith("q")])
    return np.stack([np.interp(times,trace["time"],trace[f"q{i}"]) for i in range(count)], axis=1)


def body_pose(body, q):
    pos = np.asarray(body["anchor"], dtype=float).copy()
    angle = 0
    if body["joint"] == "slide":
        pos += np.asarray(body["direction"])*q[body["dof"]]
    elif body["joint"] == "hinge":
        angle = q[body["dof"]]
    return pos,angle


def world_bounds(scene, q_paths):
    minimum = np.full(3,np.inf)
    maximum = np.full(3,-np.inf)
    shapes = [geometry_mesh(g).points for g in scene["geometry"]]
    for q in q_paths:
        for body in scene["bodies"]:
            points = shapes[body["geometry"]]
            pos,ang = body_pose(body,q)
            c,s = np.cos(ang),np.sin(ang)
            rotation = np.array([[c,-s,0],[s,c,0],[0,0,1]])
            world = points @ rotation.T + pos
            minimum=np.minimum(minimum,np.min(world,axis=0))
            maximum=np.maximum(maximum,np.max(world,axis=0))
    return minimum,maximum


def camera(scene, view):
    if scene == "cam-array":
        if view=="front":
            return np.array([0.,.07,1.]),np.array([0.,1.,0.])
        return np.array([.65,.76,1.4]),np.array([0.,1.,0.])
    if scene == "ball-train":
        if view=="top":
            return np.array([0.,0.,1.]),np.array([0.,1.,0.])
        return np.array([.38,-.90,1.18]),np.array([0.,0.,1.])
    if view=="side":
        return np.array([0.,-1.,.09]),np.array([0.,0.,1.])
    return np.array([.72,-1.,.70]),np.array([0.,0.,1.])


class Renderer:
    def __init__(self, scene, bounds, view, size):
        self.bodies=scene["bodies"]
        self.size=size
        plotter=pv.Plotter(off_screen=True,window_size=size)
        plotter.set_background("#f3f6fa")
        plotter.enable_anti_aliasing("fxaa")
        self.plotter=plotter
        self.actors=[]
        meshes=[geometry_mesh(g) for g in scene["geometry"]]
        for body in self.bodies:
            index=body["geometry"]
            kind=scene["geometry"][index]["shape"]
            color=COLORS[kind]
            if body["name"]=="follower":
                color="#b97145"
            self.actors.append(plotter.add_mesh(
                meshes[index].copy(), color=color, smooth_shading=kind!="plate",
                ambient=.27, diffuse=.79, specular=.27, specular_power=21,
                show_edges=kind=="plate", edge_color="#33465c", line_width=1,
                lighting=True,
            ))
        lo,hi=bounds
        focus=(lo+hi)/2
        direction,vertical=camera(scene["scene"],view)
        direction/=np.linalg.norm(direction)
        right=np.cross(vertical,direction)
        right/=np.linalg.norm(right)
        vertical=np.cross(direction,right)
        corners=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
        span_h=np.ptp(corners@right)
        span_v=np.ptp(corners@vertical)
        scale=max(span_h*size[1]/size[0],span_v)*.64
        plotter.camera_position=(focus+direction*max(np.linalg.norm(hi-lo)*2.25,1.),focus,vertical)
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale=scale

    def image(self,q):
        for body,actor in zip(self.bodies,self.actors,strict=True):
            pos,ang=body_pose(body,q)
            actor.position=tuple(pos)
            actor.orientation=(0,0,np.degrees(ang))
        self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.render()
        return self.plotter.screenshot(return_img=True)

    def close(self):
        self.plotter.close()


@lru_cache
def font(size, bold=False):
    candidates=[
        Path("C:/Windows/Fonts")/("arialbd.ttf" if bold else "arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu")/("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path),size)
    return ImageFont.load_default(size=size)


def caption(pixels,title,view,time,end,rate,force):
    out=Image.fromarray(pixels[:,:,:3]).convert("RGB")
    draw=ImageDraw.Draw(out)
    w,h=out.size
    draw.rectangle((0,0,w,74),fill="#f8fafc")
    draw.line((0,73,w,73),fill="#dfe6ef")
    draw.text((24,13),title,font=font(23,True),fill="#153248")
    draw.text((25,46),f"{view.capitalize()} view | HHT-alpha | CAD boundary contact",
              font=font(13),fill="#416079")
    draw.text((w-24,18),f"t = {time:07.3f} s",anchor="ra",font=font(17,True),fill="#153248")
    draw.text((w-24,46),f"{rate:.2g}x physical time | peak {force:.2f} N",
              anchor="ra",font=font(13),fill="#416079")
    draw.rectangle((0,h-4,w,h),fill="#d9e3ec")
    draw.rectangle((0,h-4,w*np.clip(time/end,0,1),h),fill="#27968b")
    return np.asarray(out)


def movie(path,renderer,trace,title,view,seconds,fps):
    count=round(seconds*fps)
    physical=np.linspace(0,float(trace["time"][-1]),count)
    values=sample(trace,physical)
    forces=np.interp(physical,trace["time"],trace["solver_peak_force"])
    width,height=renderer.size
    with av.open(str(path),"w",options={"movflags":"faststart"}) as file:
        stream=file.add_stream("libx264",rate=fps)
        stream.width=width
        stream.height=height
        stream.pix_fmt="yuv420p"
        stream.options={"crf":"19","preset":"faster"}
        for time,q,force in zip(physical,values,forces,strict=True):
            pixels=caption(renderer.image(q),title,view,float(time),physical[-1],physical[-1]/seconds,force)
            frame=av.VideoFrame.from_ndarray(pixels,format="rgb24")
            for packet in stream.encode(frame):
                file.mux(packet)
        for packet in stream.encode():
            file.mux(packet)
    return count


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path)
    parser.add_argument("--pictures-only",action="store_true")
    parser.add_argument("--scene",choices=[row[0] for row in SCENES])
    parser.add_argument("--fps",type=int,default=30)
    args=parser.parse_args()
    images=args.results/"images"
    animations=args.results/"animations"
    images.mkdir(parents=True,exist_ok=True)
    animations.mkdir(parents=True,exist_ok=True)
    index=[]
    for name,label,title,seconds,views in SCENES:
        if args.scene and name!=args.scene:
            continue
        definition,trace=data(args.results,label)
        times=np.linspace(0,trace["time"][-1],37)
        bounds=world_bounds(definition,sample(trace,times))
        time={"drop":.065,"cam-array":1.8,"ball-train":1.4}[name]
        for view in views:
            painter=Renderer(definition,bounds,view,(960,540))
            try:
                frame=painter.image(sample(trace,[time])[0])
                Image.fromarray(frame).save(images/f"{name}-{view}.png")
                if not args.pictures_only and view=="isometric":
                    path=animations/f"{name}-{view}.mp4"
                    count=movie(path,painter,trace,title,view,seconds,args.fps)
                    index.append({"file":path.name,"scene":name,"view":view,
                                  "physical_s":float(trace["time"][-1]),"video_s":seconds,
                                  "fps":args.fps,"frames":count,"width":960,"height":540,
                                  "source":f"runs/{label}.csv"})
                    print(f"Saved {path} ({count} frames)",flush=True)
            finally:
                painter.close()
    if not args.pictures_only and not args.scene:
        (animations/"index.json").write_text(json.dumps(index,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
