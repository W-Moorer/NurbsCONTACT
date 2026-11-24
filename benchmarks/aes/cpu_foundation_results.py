"""Archive and plot complete CPU runs. Read response values directly from the simulator."""
import argparse
import csv
import json
from pathlib import Path
import shlex
from zipfile import ZipFile, ZIP_DEFLATED

from rigid_movie import read_triangles, render_solids
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import meshio
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import ConvexHull

from cpu_backend_run import digest
from literature_validation import oscillator_reference
from resident_cam import reference

def gear(root,data,out):
    from cpu_backend_pose_replay import read_model
    t=data['time_s']
    f=data['p0_normal_N']
    te=data['q0']-2*data['q1']
    fig,axes=plt.subplots(3,2,figsize=(11,7),layout='constrained')
    for i,(y,label) in enumerate([(data['v1'],'Driven speed (rad/s)'),(1e3*te,'Transmission error (mrad)'),(f,'Normal load (N)')]):
        cut=(len(y)-1)//50*50
        blocks=y[:cut].reshape(-1,50)
        tc=t[:cut].reshape(-1,50).mean(axis=1)
        axes[i,0].fill_between(tc,blocks.min(axis=1),blocks.max(axis=1),color='#adc6d3')
        axes[i,0].plot(tc,blocks.mean(axis=1),lw=.8,color='#165c82')
        detail=(t>=1.25)&(t<=1.27)
        axes[i,1].plot(1e3*t[detail],y[detail],lw=1.1,color='#165c82')
        axes[i,0].set_ylabel(label)
        for a in axes[i]: a.grid(alpha=.17)
    axes[0,0].set_title('Five-second response · 5 ms mean and range')
    axes[0,1].set_title('Individual time steps')
    axes[-1,0].set_xlabel('Time (s)')
    axes[-1,1].set_xlabel('Time (ms)')
    fig.savefig(out/'gear-response.png',dpi=160)
    plt.close(fig)
    np.savez_compressed(out/'gear.npz',time_s=t,driver_speed_rad_s=data['v0'],driven_speed_rad_s=data['v1'],
                        driver_angle_rad=data['q0'],driven_angle_rad=data['q1'],normal_load_N=f)
    bodies,_=read_model(root/'spur.scene')
    render_scene(bodies, data, out/'gear.mp4', 'Spur gears  |  CPU mesh dynamics')
    return dict(mean_ratio=float((data['q0'][-1]-data['q0'][0])/(data['q1'][-1]-data['q1'][0])),
                min_load_N=float(f.min()),max_load_N=float(f.max()),contact_fraction=float(np.mean(f>0)),
                body_pair_load_switches=int(np.count_nonzero(np.diff(f>0))),
                peak_to_peak_te_rad=float(np.ptp(te)))


def render_scene(bodies, trace, out, title):
    triangles = [read_triangles(body[0].with_suffix(".msh")) for body in bodies]
    angle = np.column_stack([axis[2]*(offset+trace[f"q{j}"]) for _, _, axis, offset, j in bodies])
    origins = np.array([body[1] for body in bodies])
    load = (np.column_stack([trace[f"p{i}_normal_N"] for i in range(len(bodies)//2)])
            if "p0_normal_N" in trace else None)
    render_solids(out, triangles, origins, angle, trace["time_s"], load, title)


def reducer_movie(scene, trace_data, out):
    from cpu_backend_pose_replay import read_model
    bodies, _ = read_model(scene)
    render_scene(bodies, trace_data, out, "Two-stage reducer  |  CPU mesh dynamics")


def reducer_prefix(root,run,out,save=True):
    deck=root.parent/'long-mesh'
    with np.load(deck/'deck/assembly.npz') as assembly,np.load(deck/'recurdyn/physics.npz') as rd:
        signs=assembly['bodies'][[0,1,3],3]
        ref_t=rd['reference_rad'][:,0]/assembly['bodies'][0,8]
        ref_speed=rd['global_speed_rad_s']*signs
    count=int(json.loads((root/run/'run.json').read_text())['last_accepted']['step'])+1
    raw=trace(root/run/'progress.csv',max_rows=count)
    t=raw['time_s']
    ids=np.rint(t/.0001).astype(int)
    np.testing.assert_allclose(t,ref_t[ids],rtol=0,atol=1e-9)
    w=np.column_stack([raw['v0'],raw['v1'],raw['v2']])
    ref=ref_speed[ids]
    if save:
        fig,axs=plt.subplots(2,2,figsize=(11,5.4),layout='constrained')
        starts=np.arange(0,len(t),50)
        counts=np.diff(np.append(starts,len(t)))
        block_time=np.add.reduceat(t,starts)/counts
        detail=(t>=.102-1e-10)&(t<=.114+1e-10)
        for i,(row,title) in enumerate(zip(axs,['Intermediate','Output'],strict=True),1):
            overview,brake=row
            for y,color,style,name in [(w[:,i],'#165c82','-','CPU mesh / HHT'),(ref[:,i],'#c17531','--','RecurDyn')]:
                overview.plot(block_time,np.add.reduceat(y,starts)/counts,style,lw=.95,color=color,label=name)
                brake.plot(t[detail]*1e3,y[detail],style,lw=1.1,color=color)
            overview.set(ylabel=f'{title} (rad/s)',xlim=(0,t[-1]))
            for a in row: a.grid(alpha=.17)
        axs[0,0].set_title(f'Accepted interval (0–{t[-1]:.3f} s) · 5 ms mean')
        axs[0,1].set_title('Brake loading · individual time steps')
        axs[0,0].legend(frameon=False,ncol=2)
        axs[1,0].set_xlabel('Time (s)')
        axs[1,1].set_xlabel('Time (ms)')
        fig.savefig(out/'reducer-prefix.png',dpi=160)
        plt.close(fig)
        np.savez_compressed(out/'reducer-prefix.npz',time_s=t,cpu_speed_rad_s=w,recurdyn_speed_rad_s=ref)
        reducer_movie(deck/'deck/reducer.scene',raw,out/'reducer-prefix.mp4')
    return dict(shared_samples=len(t),shared_duration_s=float(t[-1]),
                speed_rms_difference_rad_s=np.sqrt(np.mean((w-ref)**2,axis=0)).tolist(),
                peak_abs_speed_difference_rad_s=np.max(abs(w-ref),axis=0).tolist(),
                reference_file_sha256=digest(deck/'recurdyn/physics.npz'),
                reference_run=json.loads((deck/'recurdyn/run.json').read_text()))

def polygon_reference(sectors):
    delta=2*np.pi/sectors
    def support(theta):
        phi=theta+delta*np.rint((np.pi/2-theta)/delta)
        return .024*np.sin(theta)+.105*np.sin(phi),.024*np.cos(theta)+.105*np.cos(phi)
    def rhs(t,y):
        theta,w,q,v=y
        height,arm=support(theta)
        f=5000*max(0.,height-.115-q)
        return [w,(1.8*(4*t-theta)+.1*(4-w)-f*arm)/.007,v,(f-3.2-70*q-.65*v)/.48]
    sol=solve_ivp(rhs,(0.,5.),[0,4,0,.096],method='DOP853',rtol=2e-12,atol=1e-14,
                  max_step=.0005,dense_output=True)
    if not sol.success: raise RuntimeError(sol.message)
    return support,sol.sol


def trace(path,max_rows=None):
    with path.open() as f:
        names=next(csv.reader(f))
    values=np.loadtxt(path,delimiter=',',skiprows=1,ndmin=2,max_rows=max_rows)
    return dict(zip(names,values.T,strict=True))


def write_summary(result,out):
    rows=['# CPU 刚体基线实测结果','']
    if (out/'temporal-reducer/summary.json').is_file():
        rows+=['两级减速器在相邻姿态间复用候选面片后的独立长时对照见[最新减速器实验](temporal-reducer/README.md)。以下保留原各版本计时。','']
    rows+=['主步长为 0.1 ms。耗时为单次独立进程墙钟时间，含初始化和轨迹输出。','',
          '| 模型 | 仿真时长 | AABB 耗时 | OBB 耗时 | 对照量 |',
          '|---|---:|---:|---:|---|']
    def row(name,duration,left,right,note):
        rows.append(f'| {name} | {duration:g} s | {left:.2f} s | {right:.2f} s | {note} |')
    for case,mesh,name in [('impact','coarse','双滑块正碰'),('oscillator','coarse','单边弹簧振子'),
                           ('cam','coarse','偏心凸轮，48 等分'),('cam','fine','偏心凸轮，96 等分')]:
        a,b=[next(x for x in result['runs'] if (x['case'],x['configuration'],x['method'])==(case,mesh,tree))
             for tree in ('mesh-aabb','mesh-obb')]
        if case=='impact': note=f"峰值 {b['peak_force_N']:.6f} N，解析值 {b['exact_peak_force_N']:.2f} N"
        elif case=='oscillator': note=f"{b['transitions']} 次切换，最大时刻偏差 {b['event_error_s']*1e9:.1f} ns"
        else: note=f"对光滑轮廓的位移 RMS {1e3*b['smooth_reference']['q_rms_m']:.3f} mm，对同一多边形的位移 RMS {1e6*b['polygon_reference']['integrated_travel_rms_m']:.3f} µm"
        row(name,b['requested_s'],a['wall_s'],b['wall_s'],note)
    for case,name in [('top','三维对称陀螺'),('head-on','自由双球正碰'),('eccentric','自由刚体偏心斜碰')]:
        batch={x['method']:x for x in result['spatial'] if x['case']==case and x['status']=='complete'}
        if len(batch)!=2: continue
        a,b=batch['mesh-aabb'],batch['mesh-obb']
        if case=='top': note=f"最大姿态误差 {b['max_attitude_error_rad']:.3g} rad"
        elif case=='head-on': note=f"峰值 {b['peak_normal_force_N']:.3f} N，光滑球线性罚力解析值 {b['reference_peak_N']:.1f} N"
        else: note=f"线动量最大漂移 {b['max_linear_momentum_drift_kg_m_s']:.3g} kg·m/s，峰值法向力 {b['peak_normal_force_N']:.3f} N"
        row(name,b['duration_s'],a['process_s'],b['process_s'],note)
    rows+=['','## 齿轮机构','','每个进程都请求 5 s / 50,000 个主步，单次计算预算为 30 分钟。','',
           '| 运行 | 几何索引 | 结束状态 | 已推进模型时间 | 进程耗时 |',
           '|---|---|---|---:|---:|']
    rd=None
    for name,item in sorted(result['industrial'].items()):
        last=item['last_accepted']
        case='两级减速器' if 'reducer' in name else '单级直齿轮'
        revision=('固定接触区顺序' if name.startswith('ordered-') else
                  '局部接触复算' if name.endswith(('window','release')) else '改进前对照')
        status={'complete':'完成','timeout':'超时','solver_failure':'求解失败'}.get(item['status'],item['status'])
        rows.append(f"| {case}，{revision} | {item['method']} | {status} | {last['time_s']:.4f} s | {item['process_s']:.2f} s |")
        if item.get('plotted_reference'): rd=item['recurdyn_comparison']
    if rd is not None:
        rms=', '.join(f'{x:.3f}' for x in rd['speed_rms_difference_rad_s'])
        reference_run=rd['reference_run']
        rows+=['',f"减速器在 OBB 与 RecurDyn 共同完成的 {rd['shared_duration_s']:.4f} s 内，输入、中间、输出轴的转速 RMS 偏差依次为 {rms} rad/s。",
               f"原生 RecurDyn 完成 {reference_run['physical_s']:g} s，初始化用时 {reference_run['setup_s']:.2f} s，动力学求解用时 {reference_run['online_s']:.2f} s。"]
    current=result['industrial'].get('ordered-spur-aabb',{})
    diff=current.get('difference_to_ordered-spur-obb')
    if diff:
        rows+=['',f"单级齿轮 50,001 个输出状态上，两种 BVH 的法向力最大差 {diff['p0_normal_N']:.6g} N，"
                  f"从动轴转速最大差 {diff['v1']:.6g} rad/s，接触开关相异的样本数为 {diff['load_state_mismatch_count']}。"]
        metrics=result['industrial']['ordered-spur-obb']['response_metrics']
        rows.append(f"全程平均传动比为 {metrics['mean_ratio']:.7f}，峰值法向合力为 {metrics['max_load_N']:.2f} N，"
                    f"整对刚体的承载状态切换 {metrics['body_pair_load_switches']} 次。")
        last=result['industrial']['ordered-spur-obb']['last_accepted']
        rows.append(f"OBB 运行的 {last['integration_s']:.2f} s 积分计时中，网格接触检测累计 {last['mesh_detection_s']:.2f} s，"
                    f"三角片候选测试 {int(last['mesh_triangle_tests']):,} 次，实体布尔运算 {int(last['mesh_booleans'])} 次。")
    rows+=['','## SAP 法向冲量','']
    sap=result['drake']
    rows.append(f"Drake 与本后端分别得到 {sap['drake_contact_impulse_Ns']:.16f} 和 {sap['native_contact_impulse_Ns']:.16f} N·s。共 {sap['updates']} 次离散更新。")
    rows+=['','每条运行的步数、误差、子步、几何计数和摘要保存在 [runs.json](runs.json)。']
    (out/'RESULTS.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')


def draw(arrays,out):
    colour='#165c82'
    fig,ax=plt.subplots(2,2,figsize=(10,6.1),layout='constrained')
    h=arrays['impact_t']
    keep=(h>=.036)&(h<=.077)
    ax[0,0].plot(h[keep]*1000,arrays['impact_force'][keep],color=colour,label='CPU mesh')
    ax[0,0].plot(h[keep][::12]*1000,arrays['impact_reference_force'][keep][::12],'.',color='#db7634',label='Closed form')
    ax[0,0].set(xlabel='Time (ms)',ylabel='Contact force (N)',title='Axial impact')
    ax[0,0].legend(frameon=False)
    ax[1,0].plot(h,arrays['impact_gap_error']*1e6,color=colour)
    ax[1,0].set(xlabel='Time (s)',ylabel='Gap error (µm)',title='Free flight after impact')
    t=arrays['oscillator_t']
    ax[0,1].plot(t,1e3*arrays['oscillator_q'],color=colour)
    ax[0,1].plot(t[::max(1,len(t)//36)],1e3*arrays['oscillator_reference_q'][::max(1,len(t)//36)],'.',color='#db7634')
    ax[0,1].set(xlabel='Time (s)',ylabel='Displacement (mm)',title='Unilateral oscillator')
    ax[1,1].plot(t,1e9*(arrays['oscillator_q']-arrays['oscillator_reference_q']),color=colour)
    ax[1,1].set(xlabel='Time (s)',ylabel='Displacement error (nm)',title='Phase accumulation over 20 s')
    for a in ax.flat: a.grid(alpha=.19)
    fig.savefig(out/'analytical.png',dpi=160)
    plt.close(fig)

    fig,axes=plt.subplots(3,1,figsize=(10,7.5),layout='constrained')
    t=arrays['cam_t']
    for key,label,color in [('coarse','48 sectors','#e58c44'),('fine','96 sectors',colour),('reference','Smooth cam','#272624')]:
        for a,suffix,scale in zip(axes[:2],['q','force'],[1e3,1],strict=True):
            a.plot(t,scale*arrays[f'cam_{key}_{suffix}'],color=color,lw=1 if key!='reference' else .85,
                   ls='--' if key=='reference' else '-',label=label)
        if key!='reference':
            axes[2].plot(t,1e3*(arrays[f'cam_{key}_q']-arrays['cam_reference_q']),color=color,lw=.9,label=label)
    axes[0].set_ylabel('Follower travel (mm)')
    axes[1].set_ylabel('Normal load (N)')
    axes[2].set_ylabel('Travel error (mm)')
    for a in axes:
        a.grid(alpha=.19)
        a.set_xlim(0,5)
    axes[0].legend(ncol=3,frameon=False)
    axes[0].set_title('Eccentric cam · coupled rotation and follower translation')
    axes[-1].set_xlabel('Time (s)')
    fig.savefig(out/'cam-response.png',dpi=160)
    plt.close(fig)


def movie(data,meshdir,out):
    triangles = [read_triangles(meshdir/(name+".msh")) for name in ("follower", "cam")]
    positions = np.stack([np.column_stack([data[f"b{j}_{k}_m"] for k in "xyz"]) for j in range(2)], axis=1)
    angles = np.column_stack([data[f"b{j}_angle_rad"] for j in range(2)])
    render_solids(out, triangles, positions, angles, data["time_s"], data["p0_normal_N"][:,None],
                  "Eccentric cam and follower  |  CPU mesh dynamics")


def free_impact(root,out):
    data=trace(root/'spatial/eccentric-mesh-obb/response.csv')
    t=data['time_s']
    impact=t<=.1
    moment=.4*.041**2
    energy=np.zeros_like(t)
    spin=[]
    for j in (0,1):
        v=np.stack([data[f'v{7*j+k}'] for k in range(3)])
        w=np.stack([data[f'b{j}_w{k}_rad_s'] for k in 'xyz'])
        omega=np.linalg.norm(w,axis=0)
        spin.append(omega)
        energy+=.5*((v*v).sum(axis=0)+moment*omega**2)
    fig,axes=plt.subplots(1,3,figsize=(12,3.4),layout='constrained')
    axes[0].plot(1000*t[impact],data['p0_normal_N'][impact],color='#165c82')
    axes[0].set(xlabel='Time (ms)',ylabel='Normal load (N)',title='Oblique collision')
    for j,y in enumerate(spin):
        axes[1].plot(1000*t[impact],y[impact],label=f'Body {j+1}')
    axes[1].set(xlabel='Time (ms)',ylabel='Angular speed (rad/s)',title='Spin transfer')
    axes[1].legend(frameon=False)
    axes[2].plot(t,(energy-energy[0])*1e3,color='#165c82')
    axes[2].set(xlabel='Time (s)',ylabel='Kinetic energy change (mJ)',title='Post-impact free motion')
    for a in axes: a.grid(alpha=.17)
    fig.savefig(out/'spatial-impact.png',dpi=160)
    plt.close(fig)


def main(root,cam,out):
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    arrays={}
    sol,enter=reference(5,0,.001)
    polygons={key:polygon_reference(n) for key,n in [('coarse',48),('fine',96)]}
    rims={}
    for key,n in [('coarse',48),('fine',96)]:
        nodes=meshio.read(cam/f'aligned-{n}'/'cam.msh').points[:,:2]
        rims[key]=nodes[ConvexHull(nodes).vertices]
    events=sol.t_events[0]
    for subdir,mesh in [('baseline','coarse'),('baseline-fine','fine'),('baseline-half','half')]:
        for case in ['impact','oscillator','cam']:
            pair={}
            for method in ['mesh-aabb','mesh-obb']:
                directory=root/subdir/(case+'-'+method)
                original=directory
                frame_run=None
                if method=='mesh-aabb' and mesh in ('coarse','fine'):
                    faster=root/('frame' if mesh=='coarse' else 'frame-fine')/(case+'-'+method)
                    if (faster/'report.json').exists():
                        directory=faster
                        frame_run=json.loads((faster/'report.json').read_text())
                if mesh in ('coarse','fine'):
                    release=root/('release' if mesh=='coarse' else 'release-fine')/(case+'-'+method)
                    if (release/'report.json').exists(): directory=release
                if not directory.exists(): continue
                item=json.loads((directory/'report.json').read_text())
                item['configuration']=mesh
                if frame_run is not None:
                    before=json.loads((original/'report.json').read_text())
                    item['before_frame_change_process_s']=before['wall_s']
                    item['before_frame_change_steps_s']=float(before['stats']['wall_s'])
                    item['frame_change_process_s']=frame_run['wall_s']
                    item['frame_change_steps_s']=float(frame_run['stats']['wall_s'])
                    item['frame_trial_counter']=[int(before['stats']['triangle_tests']),int(frame_run['stats']['triangle_tests'])]
                records.append(item)
                if item['status']!='complete': continue
                p=directory/'response.csv'
                item['response_sha256']=digest(p)
                d=trace(p)
                pair[method]=d
                t=d['time_s']
                take=np.arange(0,len(t),max(1,round(.001/item['step_s'])))
                if case=='cam':
                    angle,omega,q,v=sol.sol(t)
                    force=5000*np.maximum(.105+.024*np.sin(angle)-.115-q,0)
                    f=d['p0_normal_N']
                    switches=t[np.flatnonzero(np.diff(f>0))+1]
                    item['smooth_reference']=dict(q_rms_m=float(np.sqrt(np.mean((d['q1']-q)**2))),
                        q_max_m=float(max(abs(d['q1']-q))),force_rms_N=float(np.sqrt(np.mean((f-force)**2))),
                        force_max_N=float(max(abs(f-force))),cam_speed_rms_rad_s=float(np.sqrt(np.mean((d['v0']-omega)**2))),
                        sampled_switch_count=len(switches),reference_switch_count=len(events),
                        max_switch_difference_s=float(max(abs(switches-events))) if len(switches)==len(events) else None)
                    support,poly=polygons['coarse' if mesh=='half' else mesh]
                    height,_=support(d['q0'])
                    angle_p,_,q_p,_=poly(t)
                    f_p=5000*np.maximum(support(angle_p)[0]-.115-q_p,0)
                    item['polygon_reference']=dict(fixed_pose_force_max_N=float(np.max(abs(f-5000*np.maximum(height-.115-d['q1'],0)))),
                        integrated_travel_rms_m=float(np.sqrt(np.mean((d['q1']-q_p)**2))),
                        integrated_force_rms_N=float(np.sqrt(np.mean((f-f_p)**2))))
                    rim=rims['coarse' if mesh=='half' else mesh]
                    actual_height=(np.sin(d['q0'][:,None])*rim[:,0]+np.cos(d['q0'][:,None])*rim[:,1]).max(axis=1)
                    item['polygon_reference']['mesh_vertex_fixed_pose_force_max_N']=float(max(abs(f-5000*np.maximum(actual_height-.115-d['q1'],0))))
                    if method=='mesh-obb' and mesh in ('coarse','fine'):
                        arrays.update(cam_t=t[take],cam_reference_q=q[take],cam_reference_force=force[take])
                        for key,values in [('q',d['q1']),('force',f),('omega',d['v0']),('v',d['v1'])]:
                            arrays[f'cam_{mesh}_{key}']=values[take]
                        if mesh=='fine': movie(d,cam/'aligned-96',out/'cam.mp4')
                if case=='impact' and mesh=='coarse' and method=='mesh-obb':
                    take=np.union1d(take,np.flatnonzero(t<=.08))
                    t=t[take]
                    tau=np.maximum(0,t-.04)
                    exact=np.where(t<.04,.018-.45*t,np.where(tau<np.pi/100,-.0045*np.sin(100*tau),.45*(tau-np.pi/100)))
                    arrays.update(impact_t=t,impact_force=d['p0_normal_N'][take],impact_reference_force=np.where(tau<np.pi/100,22.5*np.sin(100*tau),0),
                                  impact_gap_error=.018+d['q1'][take]-d['q0'][take]-exact)
                if case=='oscillator' and mesh=='coarse' and method=='mesh-obb':
                    arrays.update(oscillator_t=t[take],oscillator_q=d['q0'][take],
                                  oscillator_reference_q=oscillator_reference(t[take],1.,.1,.001)[0])
            if len(pair)==2:
                left,right=pair.values()
                fields=[key for key in left if key.startswith(('q','v')) or key=='p0_normal_N']
                diff={key:float(max(abs(left[key]-right[key]))) for key in fields}
                for item in records[-2:]: item['tree_max_difference']=diff
    extra={p.parent.name:json.loads(p.read_text()) for pattern in ('bounded-*/run.json','ordered-*/run.json') for p in root.glob(pattern)}
    scenes={'spur':root/'spur.scene','reducer':root.parent/'long-mesh/deck/reducer.scene'}
    for name,item in extra.items():
        scene=scenes['spur' if 'spur' in name else 'reducer']
        bodies=[(scene.parent/shlex.split(line)[1]).resolve() for line in scene.read_text().splitlines() if line.startswith('body ')]
        item['mesh_sha256']={body.name:digest(body.with_suffix('.msh')) for body in bodies}
    preferred=next((name for name in ('ordered-spur-obb','ordered-spur-aabb','bounded-spur-release','bounded-spur')
                    if extra.get(name,{}).get('status')=='complete'),None)
    if extra.get(preferred,{}).get('status')=='complete':
        gear_data=trace(root/preferred/'response.csv')
        extra[preferred]['response_metrics']=gear(root,gear_data,out)
        for name,item in extra.items():
            if name==preferred or 'spur' not in name or item['status']!='complete': continue
            other=trace(root/name/'response.csv')
            item['difference_to_'+preferred]={c:float(max(abs(other[c]-gear_data[c]))) for c in ('q0','q1','v0','v1','p0_normal_N')}
            item['difference_to_'+preferred]['load_state_mismatch_count']=int(np.sum((other['p0_normal_N']>0)!=(gear_data['p0_normal_N']>0)))
    reducer=next((n for n in ('ordered-reducer-obb','bounded-reducer-release','bounded-reducer') if n in extra),None)
    for name,item in extra.items():
        if 'reducer' in name:
            item['recurdyn_comparison']=reducer_prefix(root,name,out,save=name==reducer)
            item['plotted_reference']=name==reducer
    drake=json.loads((root/'sap/rechecked.json').read_text())
    result=dict(runs=records,drake=dict(**drake['runs'][0],drake_version=drake['drake_version'],
                                        native_method=drake['native_method'],geometry=drake['geometry']),
                industrial=extra,
                smooth_cam_events_s=events.tolist(),smooth_cam_enter=enter.tolist())
    result['spatial']=[json.loads(p.read_text()) for batch in ('free','spatial') for p in (root/batch).glob('*/report.json')]
    for item in result['spatial']:
        if 'frames' in item: item['status']='complete'
    if (root/'spatial/eccentric-mesh-obb/response.csv').exists(): free_impact(root,out)
    legacy=root/'caps/cam-mesh-obb'
    if (legacy/'report.json').exists():
        old=trace(legacy/'response.csv')
        xy=meshio.read(cam/'compact-planar/cam.msh').points[:,:2]
        rim=xy[ConvexHull(xy).vertices]
        height=(np.sin(old['q0'][:,None])*rim[:,0]+np.cos(old['q0'][:,None])*rim[:,1]).max(axis=1)
        load=5000*np.maximum(height-.115-old['q1'],0)
        delta=old['p0_normal_N']-load
        result['unstructured_cam']=dict(actual_mesh_sha256=digest(cam/'compact-planar/cam.msh'),
            status='complete',max_excess_over_one_normal_spring_N=float(max(delta)),
            rms_difference_N=float(np.sqrt(np.mean(delta**2))),reference='one spring at the deepest mesh point; solver uses one spring per connected cap',
            run=json.loads((legacy/'report.json').read_text()))
    (out/'runs.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    write_summary(result,out)
    np.savez_compressed(out/'responses.npz',**arrays)
    draw(arrays,out)
    with ZipFile(out/'models.zip','w',ZIP_DEFLATED) as z:
        for p in (root/'analytic').iterdir():
            if p.suffix in ['.scene','.step','.msh']: z.write(p,'analytic/'+p.name)
        for name in ['model','aligned-48','aligned-96']:
            for p in (cam/name).iterdir():
                if p.suffix in ['.scene','.step','.msh','.json']: z.write(p,'cam/'+name+'/'+p.name)
        sphere=cam.parent/'offset-sphere.msh'
        for path in [sphere,sphere.with_suffix('.quality.json')]: z.write(path,'spatial/'+path.name)
        for case,batch in [('top','free'),('head-on','spatial'),('eccentric','spatial')]:
            model=root/batch/(case+'-mesh-obb')/'model.scene'
            if model.exists(): z.write(model,'spatial/'+case+'.scene')
        for name,scene in scenes.items():
            lines=[]
            for line in scene.read_text().splitlines():
                if line.startswith('body '):
                    words=shlex.split(line)
                    part=scene.parent/words[1]
                    line=line.replace(words[1],part.name,1)
                    for ext in ('.msh','.quality.json'):
                        source=part.with_suffix(ext)
                        z.write(source,name+'/'+source.name)
                lines.append(line)
            z.writestr(name+'/'+name+'.scene','\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['root','cam','out']: parser.add_argument(name,type=Path)
    args=parser.parse_args()
    main(args.root,args.cam,args.out)
