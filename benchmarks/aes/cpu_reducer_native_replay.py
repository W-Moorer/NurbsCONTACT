"""Compare contact wrenches at the saved RecurDyn poses, on both sets of facets."""
import argparse
import json
from pathlib import Path
import subprocess
from time import monotonic

import numpy as np

from cpu_backend_pose_replay import read_model
from cpu_backend_run import digest
from reducer_native_patches import native_surfaces


def export_meshes(deck, rmd, destination):
    destination.mkdir(parents=True, exist_ok=True)
    counts = {}
    for index, part, nodes, faces in native_surfaces(deck, rmd):
        # RecurDyn MMKS to the metres used in the Gmsh simulation deck.
        points = nodes * .001
        v = points[faces]
        volume = np.sum(v[:, 0] * np.cross(v[:, 1], v[:, 2])) / 6
        if volume == 0:
            raise ValueError("Degenerate native surface")
        if volume < 0:
            faces = faces[:, [0, 2, 1]]
        filename = destination / part['mesh']
        with filename.open('w', encoding='ascii') as out:
            print('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes', len(points), sep='\n', file=out)
            for n, xyz in enumerate(points, 1):
                print(n, *map(lambda x: format(x,'.17g'), xyz), file=out)
            print('$EndNodes\n$Elements', len(faces), sep='\n', file=out)
            for n, tri in enumerate(faces, 1):
                print(n, 2, 0, *(tri+1), file=out)
            print('$EndElements', file=out)
        counts[index] = len(faces)
    return [counts[i] for i in range(len(counts))]


def replay(exe, deck, reference, rmd, mesh_dir, output, gmsh_only=False, scope="finite"):
    counts = None if gmsh_only else export_meshes(deck, rmd, mesh_dir)
    bodies, pairs = read_model(deck/'reducer.scene')
    with np.load(deck/'assembly.npz') as model, np.load(reference) as rd:
        n = len(rd['angle_rad'])
        step = np.unique(np.r_[np.arange(0,n,200),np.arange(1020,1141),np.arange(12400,12521),n-1])
        step = step[step<n]
        time = rd['reference_rad'][step,0] / model['bodies'][0,8]
        angles = rd['angle_rad'][step] + model['coord'][:,3]
        reference_force = rd['resultant_N'][step]
        reference_torque = rd['contact_raw_TZ_Nm'][step,1]
        balance = (rd['global_accel_rad_s2'][step,2] * model['coord'][2,0]
                   - rd['support_raw_TZ_Nm'][step,2] - rd['load_raw_TZ_Nm'][step,1])
    results = dict(executable_sha256=digest(exe),rmd_sha256=digest(rmd),native_triangles=counts,scope=scope,
                   reference_sha256=digest(reference),samples=len(step),methods={})
    series = dict(step=step,time_s=time,native_resultant_N=reference_force,native_output_Nm=reference_torque,
                  native_balance_output_Nm=balance)
    brake = (step>=1020)&(step<=1140)
    for name in (('gmsh',) if gmsh_only else ('gmsh','native')):
        forces = np.full((len(step),2),np.nan)
        torques = np.full_like(forces,np.nan)
        num = np.zeros_like(forces,dtype=int)
        unresolved = []
        invalid = {}
        for j,(a,b,k,limit) in enumerate(pairs):
            poses, files = [], []
            for i in (a,b):
                path, center, axis, initial, coordinate = bodies[i]
                poses.extend([np.broadcast_to(center,(len(step),3)),np.broadcast_to(axis,(len(step),3)),
                              initial+angles[:,coordinate]])
                files.append(path.with_suffix('.msh') if name=='gmsh' else mesh_dir/path.with_suffix('.msh').name)
            rows = [' '.join(format(x,'.17g') for x in row)+'\n' for row in np.column_stack(poses)]
            records=[]
            deadline=monotonic()+1800
            while len(records)<len(step):
                run = subprocess.run([exe,'--method','mesh-obb','--regions',scope,'--axis-angle',
                                      '--max-depth-m',str(limit),*files],input=''.join(rows[len(records):]),
                                     capture_output=True,text=True,timeout=max(.01,deadline-monotonic()))
                records.extend(run.stdout.splitlines())
                if run.returncode == 0:
                    if len(records)!=len(step): raise ValueError("Missing force replay rows")
                    break
                if not records and 'contact response requires a closed outward solid boundary' in run.stderr:
                    invalid[j+1]=run.stderr.strip()
                    break
                if 'intersecting patch has no finite feature witness' not in run.stderr or len(records)>=len(step):
                    raise RuntimeError(run.stderr)
                unresolved.append([j+1,int(step[len(records)])])
                records.append(None)
            for i,line in enumerate(records):
                if line is None: continue
                values=np.asarray(json.loads(line)['regions']).reshape(-1,7)
                load=values[:,3:6]*(k*values[:,6,None])
                num[i,j]=len(values)
                forces[i,j]=np.linalg.norm(load.sum(axis=0))
                torques[i,j]=np.cross(values[:,:3]-bodies[b][1],load)[:,2].sum()
        divisor=np.maximum(num,1)
        series.update({name+'_force_N':forces,name+'_driven_Nm':torques,name+'_regions':num,
                       name+'_shared_force_N':forces/divisor,name+'_shared_driven_Nm':torques/divisor})
        results['methods'][name]=dict(unresolved_stage_step=unresolved,invalid_surface=invalid)
        for label, mask in [('brake',brake),('uniform',step%200==0),('second_window',(step>=12400)&(step<=12520))]:
            stats={}
            for j in range(len(pairs)):
                used=mask & np.isfinite(forces[:,j])
                entry=dict(samples=int(used.sum()),requested=int(mask.sum()))
                if used.any():
                    entry['force_rms_difference_N']=float(np.sqrt(np.mean((forces[used,j]-reference_force[used,j])**2)))
                    entry['shared_force_rms_difference_N']=float(np.sqrt(np.mean((forces[used,j]/divisor[used,j]-reference_force[used,j])**2)))
                    if j==1:
                        entry.update(mean_output_Nm=float(torques[used,1].mean()),
                                     reference_mean_output_Nm=float(reference_torque[used].mean()),
                                     output_rms_difference_Nm=float(np.sqrt(np.mean((torques[used,1]-reference_torque[used])**2))),
                                     shared_output_rms_difference_Nm=float(np.sqrt(np.mean((torques[used,1]/divisor[used,1]-reference_torque[used])**2))),
                                     shared_output_mean_Nm=float(np.mean(torques[used,1]/divisor[used,1])))
                stats[j+1]=entry
            results['methods'][name][label]=stats
    results['reference_brake_torque_balance_rms_Nm']=float(np.sqrt(np.mean((balance[brake]-reference_torque[brake])**2)))
    np.savez_compressed(output.with_suffix('.npz'),**series)
    output.write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ('exe','deck','reference','rmd','mesh_dir','output'):
        cli.add_argument(name,type=Path)
    cli.add_argument('--scope',choices=('finite','contour-ray'),default='finite')
    cli.add_argument('--gmsh-only',action='store_true')
    args=cli.parse_args()
    replay(**vars(args))
