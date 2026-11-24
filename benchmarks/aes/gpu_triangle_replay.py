"""Compare device intersection IDs with CPU exact triangle predicates at identical relative poses."""
import argparse
import json
import subprocess
from pathlib import Path

import cupy as cp
import meshio
import numpy as np
from scipy.spatial.transform import Rotation
from cpu_backend_pose_replay import read_model
from gpu_triangle_bvh import IntersectionQuery, TriangleMesh


def validate(deck, physics, exe, out):
    bodies,pairs=read_model(deck/'reducer.scene')
    with np.load(deck/'assembly.npz') as cfg,np.load(physics) as reference:
        angles=reference['angle_rad']+cfg['coord'][:,3]
    steps=np.unique(np.r_[0,1,25,100,400,1019,1020,1021,1040,1090,1140,len(angles)-1])
    steps=steps[steps<len(angles)]
    stats={}
    for stage,(a,b,_,_) in enumerate(pairs,1):
        files=[bodies[i][0].with_suffix('.msh') for i in (a,b)]
        meshes=[meshio.read(f) for f in files]
        device_mesh=[TriangleMesh(m.points,m.get_cells_type('triangle')) for m in meshes]
        query=IntersectionQuery(*device_mesh)
        transforms=[];cpu_inputs=[]
        for idx in steps:
            ra,rb=[Rotation.from_rotvec(bodies[i][2]*(bodies[i][3]+angles[idx,bodies[i][4]])) for i in (a,b)]
            rot=rb.inv()*ra
            translation=rb.inv().apply(bodies[a][1]-bodies[b][1])
            x,y,z,w=rot.as_quat()
            matrix=np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                             [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                             [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
            transforms.append(cp.asarray(np.column_stack((matrix,translation))))
            xyzq=np.r_[translation,rot.as_quat()[[3,0,1,2]],0,0,0,1,0,0,0]
            cpu_inputs.append(' '.join(format(x,'.17g') for x in xyzq))
        cpu=subprocess.run([exe,'--method','mesh-obb',*files],input='\n'.join(cpu_inputs)+'\n',text=True,capture_output=True,check=True,timeout=1800)
        rows=[]
        for idx,transform,line in zip(steps,transforms,cpu.stdout.splitlines(),strict=True):
            reference_points={(ia,ib):np.asarray(points) for _,_,ia,ib,points in json.loads(line)['hits']}
            expected=set(reference_points)
            sizes,points,_=query.enqueue(transform,geometry=True)
            found=list(map(tuple,query.read().tolist()))
            if len(found)!=len(set(found)): raise ValueError('Repeated GPU triangle pair')
            sizes=sizes[:len(found)].get()
            points=points[:len(found)].get()
            unresolved=int(np.count_nonzero(sizes<1))
            max_vertex_distance=0.
            for pair,size,vertices in zip(found,sizes,points,strict=True):
                if pair not in expected or size<1: continue
                delta=reference_points[pair][:,None,:]-vertices[None,:size,:]
                distance=np.linalg.norm(delta,axis=2)
                max_vertex_distance=max(max_vertex_distance,float(distance.min(axis=1).max()),float(distance.min(axis=0).max()))
            found=set(found)
            rows.append(dict(step=int(idx),cpu=len(expected),gpu=len(found),missed=len(expected-found),extra=len(found-expected),
                             unresolved_constructions=unresolved,max_intersection_vertex_distance_m=max_vertex_distance))
        # GPU event times for one geometry query. No integration or force evaluation in this metric.
        events=[]
        for _ in range(4):
            for transform in transforms:
                start,end=cp.cuda.Event(),cp.cuda.Event()
                start.record();query.enqueue(transform);end.record();events.append((start,end))
        cp.cuda.get_current_stream().synchronize()
        times=np.array([1000*cp.cuda.get_elapsed_time(x,y) for x,y in events])
        # A graph supplies all independent test poses without a Python dispatch gap between kernels.
        stream=cp.cuda.Stream(non_blocking=True)
        start,end=cp.cuda.Event(),cp.cuda.Event()
        with stream:
            stream.begin_capture()
            for transform in transforms: query.enqueue(transform)
            graph=stream.end_capture()
            graph.launch(stream)
            start.record(stream)
            for _ in range(32): graph.launch(stream)
            end.record(stream)
        end.synchronize()
        graph_us=1000*cp.cuda.get_elapsed_time(start,end)/(32*len(transforms))
        stats[stage]=dict(poses=rows,dispatch_median_us=float(np.median(times)),dispatch_p95_us=float(np.percentile(times,95)),
                          graph_static_query_mean_us=float(graph_us))
    report=dict(device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),stages=stats)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print([(s['graph_static_query_mean_us'],sum(r['missed'] for r in s['poses']),sum(r['extra'] for r in s['poses'])) for s in stats.values()])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('deck','physics','exe','out'):p.add_argument(key,type=Path)
    validate(**vars(p.parse_args()))
