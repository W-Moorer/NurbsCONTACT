"""Compare loaded regions at recorded RecurDyn poses and inspect extra CPU contacts."""
import argparse
import json
from pathlib import Path

import meshio
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from cpu_backend_pose_replay import read_model
from reducer_contact_rays import surface_hit
from reducer_zero_load_replay import regions_at


def native_points(file, steps, dt):
    points = {int(i): [] for i in steps}
    step = -1
    with file.open() as f:
        next(f)
        for line in f:
            row = np.fromstring(line, sep=" ")
            if len(row) in (2, 24):
                step = round(row[0]/dt)
                row = row[2:]
            if step > steps[-1]:
                break
            if step in points and len(row) and row[20] > 0:
                points[step].append(np.r_[row[7:10]*.001, row[10:13], -row[16]*.001])
    return [np.array(points[i]).reshape(-1,7) for i in steps]


def winding(tri, point):
    a, b, c = (tri-point).transpose(1,0,2)
    def dot(x,y):
        return np.einsum("ij,ij->i",x,y)
    la, lb, lc = (np.linalg.norm(x,axis=1) for x in (a,b,c))
    numerator = dot(a,np.cross(b,c))
    denominator = la*lb*lc + dot(a,b)*lc + dot(b,c)*la + dot(c,a)*lb
    return float(np.arctan2(numerator,denominator).sum()/(2*np.pi))


def audit(exe, scene, deck, native, out, full=False):
    bodies,pairs = read_model(scene)
    with np.load(deck/"assembly.npz") as cfg, np.load(native/"physics.npz") as rd:
        q = rd["angle_rad"] + cfg["coord"][:,3]
        t = rd["reference_rad"][:,0]/cfg["bodies"][0,8]
        dt = t[1]-t[0]
        reference_torque=rd["contact_raw_TZ_Nm"][:,1]
    peaks=[]
    if full:
        np.testing.assert_allclose(t,np.arange(50001)*.0001,atol=1e-10,rtol=0)
        # One reproducible sample per 5 ms interval avoids locking to a tooth phase.
        samples=np.arange(0,50000,50)+np.random.default_rng(0).integers(50,size=1000)
        for s in range(5):
            start,stop=10000*s,10000*(s+1)+(s==4)
            peaks.append(start+int(np.argmax(reference_torque[start:stop])))
        steps=np.unique(np.r_[samples,peaks])
    else:
        steps=samples=np.arange(1020,1141)
    sample_set=set(samples)
    result = {}
    for stage,pair in enumerate(pairs):
        a,b,k,_ = pair
        cpu = regions_at(exe,bodies,pair,q[steps])
        rd = native_points(next(native.glob(f"*_Mesh{stage}@*.con")),steps,dt)
        def torque(x):
            if not len(x):
                return 0.
            return float(np.cross(x[:,:3]-bodies[b][1],x[:,3:6]*k*x[:,6:7])[:,2].mean())
        records = []
        unresolved=[]
        for step,x,y in zip(steps,cpu,rd,strict=True):
            i=int(step)
            if x is None:
                unresolved.append(i)
                continue
            dist = np.linalg.norm(x[:,None,:3]-y[None,:,:3],axis=2)
            matches,_ = linear_sum_assignment(dist)
            extra = np.delete(x,matches,axis=0)
            records.append(dict(step=i, cpu_count=len(x), reference_count=len(y), cpu_Nm=torque(x),
                                reference_Nm=torque(y), extra_cpu_points=extra.tolist()))
        summary={}
        sampled=[r for r in records if r["step"] in sample_set]
        for key, rows in [("all",sampled),("equal_count",[r for r in sampled if r["cpu_count"]==r["reference_count"]]),
                          ("different_count",[r for r in sampled if r["cpu_count"]!=r["reference_count"]])]:
            diff=np.array([r["cpu_Nm"]-r["reference_Nm"] for r in rows])
            summary[key]=dict(samples=len(rows), sum_squared_difference_Nm2=float(diff@diff),
                              rms_Nm=float(np.sqrt(np.mean(diff**2))) if len(rows) else None)
        candidates=[r for r in records if r["extra_cpu_points"]]
        checks=[]
        if candidates:
            worst=max(candidates,key=lambda r:abs(r["cpu_Nm"]-r["reference_Nm"]))
            snapshots={r["step"]:r for r in candidates if r["step"] in peaks}
            snapshots[worst["step"]]=worst
            meshes=[]
            for body in (a,b):
                meshes.append(meshio.read(bodies[body][0].with_suffix(".msh")))
            for step, snapshot in sorted(snapshots.items()):
                triangles=[]
                for body,mesh in zip((a,b),meshes,strict=True):
                    _,origin,axis,initial,joint=bodies[body]
                    rotation=Rotation.from_rotvec(axis*(initial+q[step,joint]))
                    triangles.append((rotation.apply(mesh.points)+origin)[mesh.get_cells_type("triangle")])
                for values in snapshot["extra_cpu_points"]:
                    p,n=np.array(values[:3]),np.array(values[3:6])
                    offsets=[surface_hit(tri,p,direction,.0003) for tri,direction in zip(triangles,(n,-n),strict=True)]
                    checks.append(dict(step=step,cpu_depth_um=values[6]*1e6,
                        winding_at_cpu_point=[winding(tri,p) for tri in triangles],
                        independent_ray_offsets_um=[float(h*1e6) if h is not None else None for h in offsets]))
        result[stage+1]=dict(summary=summary,records=records,independent_checks=checks,
                             output_peak_steps=peaks,unresolved_steps=unresolved)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("exe","scene","deck","native","out"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--full",action="store_true")
    audit(**vars(parser.parse_args()))
