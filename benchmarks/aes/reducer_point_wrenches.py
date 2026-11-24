"""Reconstruct native force and moment channels from RecurDyn contact-point output."""
import argparse
import json
from pathlib import Path

import numpy as np

from cpu_backend_run import digest


def point_wrenches(path, center_mm, samples, dt):
    forces=np.zeros((samples,3))
    moments=np.zeros_like(forces)
    counts=np.zeros(samples,dtype=int)
    positive=np.zeros_like(counts)
    seen=np.zeros(samples,dtype=bool)
    last=-1
    error=0.
    depth=np.zeros(samples)
    total=0
    with path.open() as stream:
        header=stream.readline().split()
        if header[9:15]!=['X_ConPos','Y_ConPos','Z_ConPos','X_NorDir','Y_NorDir','Z_NorDir']:
            raise ValueError('Unknown native point format')
        for line in stream:
            values=np.fromstring(line,sep=' ')
            if not len(values): continue
            if len(values) in (2,24):
                index=round(values[0]/dt)
                if abs(values[0]-index*dt)>1e-9 or not 0<=index<samples:
                    raise ValueError('Unexpected contact time')
                if seen[index] and (index!=0 or counts[index] or values[1]): raise ValueError('Repeated point frame')
                seen[index]=True
                last=index
                counts[index]=int(values[1])
                values=values[2:]
                if not len(values): continue
            if len(values)!=22 or last<0: raise ValueError('Invalid continuation row')
            number=int(values[0])
            if not 1<=number<=counts[last]: raise ValueError('Invalid contact point index')
            total+=1
            point=values[7:10]
            normal=values[10:13]
            f=values[20]*normal+values[21]*values[13:16]
            forces[last]+=f
            moments[last]+=np.cross((point-center_mm)*.001,f)
            if values[20]>0:
                positive[last]+=1
                depth[last]=max(depth[last],-values[16])
                error=max(error,abs(values[20]-160000*max(0.,-values[16])))
    if not seen.all() or total!=counts.sum(): raise ValueError('Incomplete contact point file')
    return forces,moments,counts,positive,depth,error


def check(directory, reference, original, assembly, out):
    with np.load(assembly) as model,np.load(reference) as rd,np.load(original) as old:
        t=rd['reference_rad'][:,0]/model['bodies'][0,8]
        dt=t[1]
        arrays=dict(time_s=t)
        metrics={}
        for j,(_,b) in enumerate(model['pairs']):
            f=next(directory.rglob(f'*_Mesh{j}@*.con'))
            force,moment,counts,positive,depth,linear_error=point_wrenches(f,model['bodies'][b,4:7],len(t),dt)
            divisor=np.maximum(positive,1)[:,None]
            avg_force,avg_moment=force/divisor,moment/divisor
            arrays.update({f'stage{j+1}_sum_force_N':force,f'stage{j+1}_sum_moment_Nm':moment,
                           f'stage{j+1}_mean_force_N':avg_force,f'stage{j+1}_mean_moment_Nm':avg_moment,
                           f'stage{j+1}_positive':positive,f'stage{j+1}_depth_mm':depth})
            brake=(t>=.102-1e-9)&(t<=.114+1e-9)
            stats=dict(con_sha256=digest(f),max_point_law_difference_N=linear_error,
                       min_max_positive_points=[int(positive[brake].min()),int(positive[brake].max())],
                       peak_penetration_um=float(depth[brake].max()*1000),
                       all_peak_penetration_um=float(depth.max()*1000))
            # Native Tz is in its reported global frame, at the action shaft.
            for label,mask in [('brake',brake),('all',np.ones(len(t),bool))]:
                stats[label]={}
                for law,fv,tv in [('sum',force,moment),('active_mean',avg_force,avg_moment)]:
                    ferr=np.linalg.norm(fv[mask],axis=1)-rd['resultant_N'][mask,j]
                    terr=tv[mask,2]-rd['contact_raw_TZ_Nm'][mask,j]
                    stats[label][law]=dict(force_rms_N=float(np.mean(ferr**2)**.5),torque_rms_Nm=float(np.mean(terr**2)**.5),
                                           torque_max_abs_Nm=float(np.max(np.abs(terr))))
            if j==1:
                balance=(model['coord'][2,0]*rd['global_accel_rad_s2'][:,2]
                         -rd['support_raw_TZ_Nm'][:,2]-rd['load_raw_TZ_Nm'][:,1])
                arrays.update(raw_torque_Nm=rd['contact_raw_TZ_Nm'][:,j],balance_Nm=balance)
                stats.update(brake_native_mean_Nm=float(rd['contact_raw_TZ_Nm'][brake,j].mean()),
                    brake_active_mean_Nm=float(avg_moment[brake,2].mean()),
                    native_TZ_vs_instantaneous_balance_rms_Nm=float(np.mean((rd['contact_raw_TZ_Nm'][brake,j]-balance[brake])**2)**.5))
            metrics[j+1]=stats
        np.testing.assert_allclose(t,old['reference_rad'][:len(t),0]/model['bodies'][0,8],atol=1e-10,rtol=0)
        metrics['repeat_speed_rms_rad_s']=np.sqrt(np.mean((rd['global_speed_rad_s']-old['global_speed_rad_s'][:len(t)])**2,axis=0)).tolist()
        metrics['repeat_geometric_force_rms_N']=np.sqrt(np.mean((rd['resultant_N']-old['resultant_N'][:len(t)])**2,axis=0)).tolist()
        arrays['native_resultant_N']=rd['resultant_N']
        np.savez_compressed(out.with_suffix('.npz'),**arrays)
        out.write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for field in ('directory','reference','original','assembly','out'): cli.add_argument(field,type=Path)
    check(**vars(cli.parse_args()))
