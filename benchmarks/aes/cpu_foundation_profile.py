"""Replay accepted reducer poses to measure first-hit and full contact-geometry query costs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from cpu_backend_pose_replay import read_model, world_poses
from cpu_backend_run import digest


def profile(exe,scene,progress):
    bodies,pairs=read_model(scene)
    states=np.genfromtxt(progress,names=True,delimiter=',')
    selected=states[(states['step']>=12400)&(states['step']<12600)]
    np.testing.assert_array_equal(selected['step'],np.arange(12400,12600))
    out=dict(source_sha256=digest(progress),scene_sha256=digest(scene),executable_sha256=digest(exe),
             first_s=float(selected['time_s'][0]),last_s=float(selected['time_s'][-1]),poses=len(selected),runs=[])
    for stage,(a,b,_,limit) in enumerate(pairs,1):
        poses=np.column_stack([world_poses(bodies[i],selected) for i in (a,b)])
        rows=''.join(' '.join(format(x,'.17g') for x in p)+'\n' for p in poses)
        meshes=[bodies[i][0].with_suffix('.msh') for i in (a,b)]
        for method in ('mesh-aabb','mesh-obb'):
            for geometry in ('any','finite'):
                opts=['--any'] if geometry=='any' else ['--regions','finite','--max-depth-m',str(limit)]
                if method=='mesh-aabb' and geometry=='any': opts.append('--candidate-window')
                task=subprocess.run([exe,'--method',method,*opts,*meshes],input=rows,capture_output=True,
                                    check=True,text=True,timeout=1800)
                data=[json.loads(line) for line in task.stdout.splitlines()]
                if len(data)!=len(selected): raise ValueError('incomplete query replay')
                timings=np.array([v['query_s'] for v in data])
                active=sum(bool(v['overlap'] if geometry=='any' else v['regions']) for v in data)
                out['runs'].append(dict(stage=stage,method=method,geometry=geometry,active=active,
                          total_query_s=timings.sum(),median_ms=np.median(timings)*1000,p95_ms=np.percentile(timings,95)*1000,
                          triangle_counter=sum(v['triangle_tests'] for v in data)))
    return out


def compare_temporal_mesh(exe, original, scene, progress):
    bodies,pairs=read_model(scene)
    states=np.genfromtxt(progress,names=True,delimiter=',',max_rows=12600)
    result=dict(scene_sha256=digest(scene), source_sha256=digest(progress),
                baseline_sha256=digest(original), current_sha256=digest(exe), windows=[])
    for start in (1020,12400):
        forward=np.arange(start,start+200)
        # Jump to the other window and return to earlier poses to exercise invalidation.
        jumps=np.array([12400 if start==1020 else 1020,start+12,start+6,start+199])
        ids=np.concatenate([forward,jumps])
        np.testing.assert_array_equal(states['step'][ids],ids)
        for stage,(a,b,_,limit) in enumerate(pairs,1):
            poses=np.column_stack([world_poses(bodies[i],states[ids]) for i in (a,b)])
            rows=''.join(' '.join(format(x,'.17g') for x in row)+'\n' for row in poses)
            meshes=[bodies[i][0].with_suffix('.msh') for i in (a,b)]
            baseline=None
            item=dict(stage=stage,start_step=start,forward_poses=len(forward),jump_poses=len(jumps),runs=[])
            for binary,method in [(original,'mesh-obb'),(exe,'mesh-obb'),(exe,'mesh-aabb')]:
                task=subprocess.run([binary,'--method',method,'--regions','finite','--max-depth-m',str(limit),*meshes],
                                    input=rows,text=True,capture_output=True,check=True,timeout=1800)
                queries=[json.loads(line) for line in task.stdout.splitlines()]
                if len(queries)!=len(ids): raise ValueError('incomplete pose replay')
                regions=[q['regions'] for q in queries]
                if baseline is None: baseline=regions
                for step,left,right in zip(ids,baseline,regions,strict=True):
                    if left!=right: raise ValueError(f'changed contact geometry at step {step}, stage {stage}')
                batch=queries[:len(forward)]
                entry=dict(method=method,original=binary==original,query_s=sum(q['query_s'] for q in batch),
                           regions=sum(len(q['regions']) for q in batch))
                for key in ('pair_bvh_builds','pair_bvh_reuses','patch_builds','patch_reuses'):
                    if key in batch[0]: entry[key]=sum(q[key] for q in batch)
                item['runs'].append(entry)
            result['windows'].append(item)
    return result


def compare_order(exe,scene,progress):
    bodies,[(a,b,_,limit)]=read_model(scene)
    states=np.genfromtxt(progress,names=True,delimiter=',')
    ids=np.unique(np.concatenate([np.arange(start,start+20) for start in (340,385,2105,3450,27840)]))
    poses=np.column_stack([world_poses(bodies[i],states[ids]) for i in (a,b)])
    rows=''.join(' '.join(format(x,'.17g') for x in row)+'\n' for row in poses)
    result={}
    for method in ('mesh-aabb','mesh-obb'):
        run=subprocess.run([exe,'--method',method,'--regions','finite','--max-depth-m',str(limit),
                            *[bodies[i][0].with_suffix('.msh') for i in (a,b)]],
                           input=rows,text=True,capture_output=True,timeout=1800,check=True)
        result[method]=[json.loads(line)['regions'] for line in run.stdout.splitlines()]
    left,right=result.values()
    if len(left)!=len(ids) or len(right)!=len(ids): raise ValueError('missing finite contact query')
    return dict(executable_sha256=digest(exe),scene_sha256=digest(scene),steps=ids.tolist(),
            poses_with_different_order_or_values=[int(i) for i,x,y in zip(ids,left,right,strict=True) if x!=y],
            poses_with_different_feature_sets=[int(i) for i,x,y in zip(ids,left,right,strict=True) if sorted(x)!=sorted(y)])


def brake_loads(exe,scene,progress,ref_file,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    states=np.genfromtxt(progress,names=True,delimiter=',',max_rows=1141)
    selected=states[(states['step']>=1020)&(states['step']<=1140)]
    ids=np.arange(1020,1141)
    np.testing.assert_array_equal(selected['step'],ids)
    bodies,pairs=read_model(scene)
    force=np.empty((len(ids),len(pairs)))
    torque=np.empty_like(force)
    hashes=[]
    for j,(a,b,k,limit) in enumerate(pairs):
        poses=np.column_stack([world_poses(bodies[i],selected) for i in (a,b)])
        rows=''.join(' '.join(format(x,'.17g') for x in row)+'\n' for row in poses)
        hashes.append(hashlib.sha256(rows.encode()).hexdigest())
        cmd=[exe,'--method','mesh-obb','--regions','finite','--max-depth-m',str(limit),
             *[bodies[i][0].with_suffix('.msh') for i in (a,b)]]
        query=subprocess.run(cmd,input=rows,capture_output=True,text=True,check=True,timeout=1800)
        lines=query.stdout.splitlines()
        if len(lines)!=len(ids): raise ValueError('incomplete wrench replay')
        for i,line in enumerate(lines):
            regions=np.array(json.loads(line)['regions']).reshape(-1,7)
            forces=(k*regions[:,6,None])*regions[:,3:6]
            force[i,j]=np.linalg.norm(forces.sum(axis=0))
            torque[i,j]=np.cross(regions[:,:3]-bodies[b][1],forces)[:,2].sum()
    with np.load(ref_file) as rd:
        rf=rd['resultant_N'][ids]
        rt=rd['contact_raw_TZ_Nm'][ids]
        nominal=float(next(s for s in scene.read_text().splitlines() if s.startswith('coordinate ')).split()[3])
        np.testing.assert_allclose(rd['reference_rad'][ids,0]/nominal,selected['time_s'],rtol=0,atol=1e-9)
    t=selected['time_s']
    np.savez_compressed(out.with_suffix('.npz'),time_s=t,cpu_force_N=force,recurdyn_force_N=rf,
                        cpu_driven_Tz_Nm=torque,recurdyn_Tz_Nm=rt)
    fig,ax=plt.subplots(3,1,figsize=(9,6.5),sharex=True,layout='constrained')
    for panel,left,right,label in zip(ax,[force[:,0],force[:,1],torque[:,1]],[rf[:,0],rf[:,1],rt[:,1]],
                                     ['Stage 1 resultant (N)','Stage 2 resultant (N)','Output contact torque (N·m)'],strict=True):
        panel.plot(t*1000,left,lw=1.25,color='#165c82',label='CPU mesh / HHT')
        panel.plot(t*1000,right,lw=1.1,color='#df8136',ls='--',label='RecurDyn')
        panel.set_ylabel(label)
        panel.grid(alpha=.17)
    ax[0].set_title('Two-stage reducer · brake loading')
    ax[0].legend(frameon=False,ncol=2)
    ax[-1].set_xlabel('Simulated time (ms)')
    fig.savefig(out.with_suffix('.png'),dpi=160)
    plt.close(fig)
    return dict(scene_sha256=digest(scene),reference_sha256=digest(ref_file),pose_sha256=hashes,
                start_s=t[0],end_s=t[-1],samples=len(ids),cpu_peak_resultant_N=force.max(axis=0).tolist(),
                recur_dyn_peak_resultant_N=rf.max(axis=0).tolist(),resultant_difference_rms_N=np.sqrt(np.mean((force-rf)**2,axis=0)).tolist(),
                output_torque_difference_rms_Nm=np.sqrt(np.mean((torque[:,1]-rt[:,1])**2)))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ('exe','scene','progress','out'): cli.add_argument(name,type=Path)
    cli.add_argument('--compare-order',action='store_true')
    cli.add_argument('--brake-reference',type=Path)
    cli.add_argument('--compare-temporal-mesh',type=Path,metavar='ORIGINAL_NC_MESH')
    a=cli.parse_args()
    check=compare_order if a.compare_order else profile
    if a.compare_temporal_mesh:
        result=compare_temporal_mesh(a.exe.resolve(),a.compare_temporal_mesh.resolve(),a.scene,a.progress)
    elif a.brake_reference:
        result=brake_loads(a.exe.resolve(),a.scene,a.progress,a.brake_reference,a.out)
    else:
        result=check(a.exe.resolve(),a.scene,a.progress)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
