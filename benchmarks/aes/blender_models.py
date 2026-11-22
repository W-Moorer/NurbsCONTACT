"""Render the manuscript benchmark assemblies using Blender's Python runtime."""

import argparse
import csv
from math import cos, pi, sin
from pathlib import Path
import sys

import bpy
from mathutils import Vector
import numpy as np


BLUE = (.045, .177, .345, 1)
STEEL = (.41, .55, .65, 1)
TEAL = (.055, .335, .313, 1)
LIGHT_TEAL = (.33, .56, .54, 1)
ORANGE = (.86, .33, .055, 1)


def material(name, color, metallic=.18):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = .31
    return mat


def style(obj, name, mat, smooth=True):
    obj.name = name
    obj.data.materials.append(mat)
    if smooth:
        for poly in obj.data.polygons:
            poly.use_smooth = True
    return obj


def mesh(name, points, faces, mat, smooth=True):
    data = bpy.data.meshes.new(name)
    data.from_pydata(points, [], faces)
    data.update()
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    return style(obj, name, mat, smooth)


def sphere(name, center, radius, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=64, radius=radius, location=center)
    return style(bpy.context.object, name, mat)


def block(name, center, size, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.dimensions = size
    return style(obj, name, mat, smooth=False)


def cylinder(name, center, radius, depth, mat):
    bpy.ops.mesh.primitive_cylinder_add(vertices=192, radius=radius, depth=depth, location=center)
    obj = style(bpy.context.object, name, mat, smooth=False)
    for poly in obj.data.polygons:
        poly.use_smooth = len(poly.vertices) == 4
    return obj


def loading_frame(path, target):
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if float(row["time"]) >= target:
                return {k:float(v) for k,v in row.items()}
    raise ValueError(f"No frame at t={target:g}: {path}")


def mechanism(case, folder, mesh_dir, trace_dir, materials):
    blue, steel, teal, light_teal, orange = materials
    if case == "plane":
        center = .067-.45/(5000**.5)
        block("Stationary plate", (0.,0.,-.007), (.76,.54,.014), steel)
        sphere("Sphere", (0.,0.,center), .055, blue)
        return ((0.,0.,.029), .94, (1.1,-1.8,1.26), (0.,0.,.005), .103, (1.1,-1.8,.35))
    if case == "spheres":
        distance = .122-.45/100.
        sphere("Lower sphere", (-distance/2,0.,0.), .055, steel)
        sphere("Upper sphere", (distance/2,0.,0.), .055, blue)
        return ((0.,0.,0.), .272, (.28,-1.78,.91), (0.,0.,0.), .082, (.13,-1.8,.38))
    if case == "cam":
        q = loading_frame(trace_dir/"cam.csv", .44)
        cx, cy = .024*cos(q["q0"]), .024*sin(q["q0"])
        cylinder("Eccentric cam", (cx,cy,0.), .105,.06,blue)
        height = .115+q["q1"]
        block("Flat follower", (0.,height+.010,0.), (.344,.020,.100), steel)
        cylinder("Pivot marker", (0.,0.,.035), .013,.011,orange)
        return ((0.,.014,0.), .445, (.38,.21,1.96), (cx,.5*(height+cy+.105),.02), .145, (.28,.11,1.8))

    target = .12
    frame = loading_frame(trace_dir/f"{case}.csv", target)
    with (folder/f"{case}.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            shaft = int(row["shaft"])
            spin = float(row["start_rpm"])*2*pi/60
            angle = float(row["angle_zero_rad"]) + int(row["direction"])*(spin*frame["time"]+frame[f"q{shaft}"])
            stage = int(row["stage"])
            mats = (blue,steel) if stage == 1 else (teal,light_teal)
            with np.load(mesh_dir/(Path(row["step"]).stem+".npz")) as source:
                obj = mesh(Path(row["step"]).stem, source["points"].tolist(),source["triangles"].tolist(),
                           mats[row["side"] == "b"], smooth=False)
            obj.location = (float(row["center_x_mm"]),0.,float(row["z_mm"]))
            obj.rotation_euler.z = angle
    if case == "spur":
        return ((55.,0.,0.), 205., (.22,-.77,2.), (30.,0.,6.), 30., (.04,-.37,2.))
    return ((119.,0.,18.), 330., (.26,-.65,2.), (125.6,0.,47.), 49., (.04,-.44,2.))


def orient(obj, point):
    obj.rotation_euler = (Vector(point)-obj.location).to_track_quat("-Z","Y").to_euler()


def render(out, center, width, direction, close):
    scene = bpy.context.scene
    camera = scene.camera
    camera.location = Vector(center)+Vector(direction).normalized()*(3.2*width)
    orient(camera,center)
    camera.data.ortho_scale = width
    camera.data.clip_start = width*.001
    camera.data.clip_end = width*60
    scene.render.resolution_x = 468 if close else 780
    scene.render.resolution_y = 468
    scene.render.filepath = str(out.resolve())
    bpy.ops.render.render(write_still=True)


def setup(center, width):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 12
    scene.cycles.use_denoising = True
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 12
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "AgX"
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (.54,.66,.82,1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = .27
    for name, offset, strength in (("Key", (.14,-.75,1.1),125), ("Fill", (-.7,.28,.8),40),
                                   ("Rim", (.55,.7,.93),79)):
        light = bpy.data.lights.new(name,"AREA")
        obj = bpy.data.objects.new(name,light)
        scene.collection.objects.link(obj)
        obj.location = Vector(center)+Vector(offset)*width
        light.energy = strength*width*width
        light.shape = "DISK"
        light.size = width*.85
        orient(obj,center)
    data = bpy.data.cameras.new("Orthographic")
    data.type = "ORTHO"
    camera = bpy.data.objects.new("Orthographic",data)
    scene.collection.objects.link(camera)
    scene.camera = camera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models",type=Path)
    parser.add_argument("meshes",type=Path)
    parser.add_argument("traces",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    args.output.mkdir(parents=True,exist_ok=True)
    for case in ("plane","spheres","cam","spur","reducer"):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.context.scene.world = bpy.data.worlds.new("Studio")
        mats = [material(name,color) for name,color in (("Blue",BLUE),("Steel",STEEL),
                ("Teal",TEAL),("Light teal",LIGHT_TEAL),("Orange",ORANGE))]
        center,width,direction,detail,detail_width,detail_direction = mechanism(
            case,args.models,args.meshes,args.traces,mats)
        setup(center,width)
        render(args.output/f"{case}-overview.png",center,width,direction,False)
        render(args.output/f"{case}-detail.png",detail,detail_width,detail_direction,True)


if __name__ == "__main__":
    main()
