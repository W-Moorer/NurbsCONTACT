"""Render the measured gear meshes at the brake-loading pose with Blender Python."""
import json
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from blender_models import BLUE, LIGHT_TEAL, STEEL, TEAL, material, mesh, render, setup


def main(deck, responses, output):
    with np.load(deck/"assembly.npz") as d, np.load(responses) as result:
        bodies,mapping=d["bodies"],d["mapping"]
        frame=np.abs(result["time_s"]-.112).argmin()
        angles=result["shaft_angle_rad"][frame]
    parts=json.loads((deck/"models.json").read_text())["parts"]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.world=bpy.data.worlds.new("Studio")
    for i,(part,color) in enumerate(zip(parts,(BLUE,STEEL,TEAL,LIGHT_TEAL),strict=True)):
        with np.load(deck/part["archive"]) as d:
            vertices,index=np.unique(d["tri"].reshape(-1,3),axis=0,return_inverse=True)
        obj=mesh(part["mesh"],(vertices*.001).tolist(),index.reshape(-1,3).tolist(),
                 material(part["mesh"],color),smooth=False)
        obj.location=(.001*bodies[i,4:7]).tolist()
        obj.rotation_euler.z=bodies[i,3]*(angles[mapping[i]]+bodies[i,7])
    center=(.119,0,.017)
    setup(center,.34)
    bpy.context.scene.view_settings.exposure=-1.25
    bpy.context.scene.render.threads=3
    output.mkdir(parents=True,exist_ok=True)
    render(output/"overview.png",center,.335,(.24,-1.1,1.55),False)
    render(output/"spur.png",(.030,0,.008),.036,(-.50,-1.05,1.15),True)
    render(output/"helical.png",(.126,0,.042),.043,(.38,-1.05,1.05),True)


if __name__=="__main__":
    main(*(Path(p) for p in sys.argv[sys.argv.index("--")+1:]))
