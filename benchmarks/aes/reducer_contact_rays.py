"""Measure mesh penetration along each reported physical contact normal at fixed native poses."""
import argparse
import json
from pathlib import Path

import meshio
import numpy as np
from scipy.spatial.transform import Rotation

from cpu_backend_pose_replay import read_model


def surface_hit(triangles, point, normal, reach):
    """Signed line distance to the nearest positively oriented triangle, limited to the local contact."""
    e1 = triangles[:,1]-triangles[:,0]
    e2 = triangles[:,2]-triangles[:,0]
    n = np.cross(e1,e2)
    den = n@normal
    front = den>1e-12*np.linalg.norm(n,axis=1)
    e1,e2,n,v,den = e1[front],e2[front],n[front],triangles[front,0],den[front]
    t = np.einsum("ij,ij->i",n,v-point)/den
    near = abs(t)<reach
    e1,e2,n,v,t = e1[near],e2[near],n[near],v[near],t[near]
    q = point+t[:,None]*normal-v
    determinant = np.einsum("ij,ij->i",n,n)
    u = np.einsum("ij,ij->i",np.cross(q,e2),n)/determinant
    v = np.einsum("ij,ij->i",np.cross(e1,q),n)/determinant
    hits = t[(u>=-1e-10)&(v>=-1e-10)&(u+v<=1+1e-10)]
    return hits[np.abs(hits).argmin()] if len(hits) else None


def measure(deck, native_dir, reference, output):
    bodies,pairs = read_model(deck/"reducer.scene")
    with np.load(reference) as rd,np.load(deck/"assembly.npz") as model:
        times = rd["reference_rad"][:,0]/model["bodies"][0,8]
        angles = rd["angle_rad"]+model["coord"][:,3]
    dt = times[1]-times[0]
    rows=[]
    for stage,(a,b,_,_) in enumerate(pairs,1):
        meshes=[meshio.read(bodies[i][0].with_suffix(".msh")) for i in (a,b)]
        faces=[mesh.get_cells_type("triangle") for mesh in meshes]
        triangles=[]
        with next(native_dir.glob(f"*_Mesh{stage-1}@*.con")).open() as stream:
            next(stream)
            for line in stream:
                v=np.fromstring(line,sep=" ")
                if len(v) in (2,24):
                    step=round(v[0]/dt)
                    in_window=.102-1e-9<=times[step]<=.114+1e-9
                    v=v[2:]
                    if in_window:
                        triangles=[]
                        for i,mesh,f in zip((a,b),meshes,faces,strict=True):
                            _,center,axis,initial,coordinate=bodies[i]
                            rotation=Rotation.from_rotvec(axis*(initial+angles[step,coordinate]))
                            triangles.append((rotation.apply(mesh.points)+center)[f])
                if not in_window or not len(v) or v[20]<=0: continue
                point,normal=v[7:10]*.001,v[10:13]
                normal=normal/np.linalg.norm(normal)
                hits=[surface_hit(t,point,n,.0003) for t,n in zip(triangles,(normal,-normal),strict=True)]
                depth=np.nan if None in hits else sum(hits)*1e6
                rows.append((stage,step,int(v[0]),-v[16]*1000.,depth))
    data=np.array(rows)
    metrics={}
    for stage in (1,2):
        group=data[data[:,0]==stage]
        error=group[:,4]-group[:,3]
        valid=np.isfinite(error)
        metrics[stage]=dict(loaded_points=len(group),unresolved=int((~valid).sum()),
                            rms_depth_difference_um=float(np.mean(error[valid]**2)**.5),
                            max_depth_difference_um=float(np.max(abs(error[valid]))))
    output.write_text(json.dumps(metrics,indent=2,allow_nan=False)+"\n")
    np.savez_compressed(output.with_suffix(".npz"),stage=data[:,0],step=data[:,1],contact_index=data[:,2],
                        native_depth_um=data[:,3],two_surface_depth_um=data[:,4])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ("deck","native_dir","reference","output"): parser.add_argument(field,type=Path)
    measure(**vars(parser.parse_args()))
