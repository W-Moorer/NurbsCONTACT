"""Compare the mechanism-specific spur CAD and triangle BVH with identical HHT settings."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import subprocess
from time import perf_counter

import numpy as np


def check_impulse(models, work, output):
    """Independently integrate the prescribed load and audit the driven-shaft equation."""
    with (models/'spur.csv').open(newline='') as stream:
        _, wheel = csv.DictReader(stream)
    radius = float(wheel['pitch_radius_mm'])*.001
    base = float(wheel['base_radius_mm'])*.001
    width = float(wheel['width_mm'])*.001
    inertia = .0003 + .5*math.pi*7850*width*(radius**4-(.32*radius)**4)
    damping = .09

    def load_integral(t):
        def ramp(mid, scale):
            return .5*(t+scale*(np.logaddexp((t-mid)/scale,(mid-t)/scale)
                                -np.logaddexp(mid/scale,-mid/scale)))
        freq = 2*np.pi*13
        return 56*(t+.26*ramp(.10,.018)-.34*ramp(.24,.028)+.06*(1-np.cos(freq*t))/freq)

    checks = {}
    for trace in sorted(work.glob('*us.csv')):
        hist = np.genfromtxt(trace,delimiter=',',names=True,usecols=('time','q1','v1','normal0_N'))
        t, q, v, f = (hist[k] for k in hist.dtype.names)
        dt = t[1]-t[0]
        ix = np.rint(np.arange(21)*.1/dt).astype(int)
        if len(t) != ix[-1]+1 or not np.allclose(t[ix],np.arange(21)*.1,rtol=0,atol=1e-9):
            raise ValueError(f'Incomplete 2 s trace in {trace}')
        shaft_impulse = (inertia*(v[ix]-v[0])+damping*(q[ix]-q[0])+load_integral(t[ix]))/base
        force_impulse = np.r_[0.,np.cumsum(.5*(f[1:]+f[:-1])*np.diff(t))][ix]
        window = np.diff(force_impulse-shaft_impulse)/np.diff(t[ix])
        checks[trace.stem] = dict(mean_normal_force_N=float(force_impulse[-1]/t[-1]),
                                  mean_from_shaft_balance_N=float(shaft_impulse[-1]/t[-1]),
                                  max_abs_100ms_mean_imbalance_N=float(np.max(np.abs(window))))
    if not checks:
        raise ValueError(f'No completed spur histories in {work}')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(window_s=.1,driven_inertia_kg_m2=inertia,
                                      driven_base_radius_m=base,shaft_damping_Nm_s=damping,
                                      configurations=checks),indent=2,allow_nan=False)+'\n')


def replay_inputs(models, work, destination):
    from cad_region_replay import write_poses
    with (models/'spur.csv').open(newline='') as stream:
        a, b = csv.DictReader(stream)
    for step in (50, 100):
        folder = destination/f'{step}us'
        folder.mkdir(parents=True, exist_ok=True)
        with (work/f'cad-{step}us.csv').open(newline='') as stream:
            rows = [r for r in csv.DictReader(stream) if .1 <= float(r['time']) < .11]
        counts = [round(float(r['active0'])) for r in rows]
        windows = {}
        for before, after in ((1,2), (2,1)):
            i = next(i for i in range(1, len(rows)-1) if counts[i-1:i+1] == [before,after])
            windows[(before,after)] = rows[i-1:i+2]
        write_poses(windows, a, b, folder/'poses.txt')


def run(executable, models, work, output, repeats, refinement=False):
    work.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    configs = [('cad', 0.), ('mesh-1563nm', 0.0015625), ('mesh-391nm', 0.000390625)]
    steps = (20, 50, 100)
    if refinement:
        configs = [configs[0], configs[2]]
        steps = (5, 10, 20, 50, 100)
    rows, signals = {}, {}
    for rep in range(repeats):
        order = configs[rep % len(configs):] + configs[:rep % len(configs)]
        for step in steps:
            for method, deflection in order:
                name = f'{method}-{step}us'
                prefix = work/name
                record = rows.setdefault(name, dict(step_us=step, deflection_mm=deflection, runs=[]))
                if 'failure' in record:
                    continue
                command = [str(executable), 'run', 'spur', str(models), str(deflection),
                           str(step*1e-6), '2', str(prefix), '1', '31', '1']
                start = perf_counter()
                try:
                    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                               text=True, timeout=1800)
                except subprocess.TimeoutExpired:
                    record['failure'] = '30 minute timeout'
                    continue
                elapsed = perf_counter()-start
                if completed.returncode:
                    record['failure'] = completed.stdout.strip() or f'process exit {completed.returncode}'
                    continue
                with prefix.with_suffix('.stats.csv').open(newline='') as stream:
                    stats, = csv.DictReader(stream)
                trace = prefix.with_suffix('.csv')
                digest = hashlib.sha256(trace.read_bytes()).hexdigest()
                if rep == 0:
                    data = np.genfromtxt(trace, delimiter=',', names=True,
                                         usecols=('time','te0_um','normal0_N','active0'))
                    if (len(data) != 2_000_000//step + 1 or not np.isfinite(data.tolist()).all()
                            or not np.allclose(data['time'], np.arange(len(data))*step*1e-6, atol=1e-9, rtol=0)):
                        raise ValueError(f'Incomplete or invalid history: {name}')
                    record.update(trace_sha256=digest, triangles=int(stats['triangles']),
                                  force_peak_N=float(data['normal0_N'].max()),
                                  newton=int(stats['newton']), subdivisions=int(stats['subdivisions']))
                    signals[name] = data[::100//step]
                if digest != record['trace_sha256']:
                    raise RuntimeError(f'Nonrepeatable trace: {name}')
                record['runs'].append(dict(integrator_s=float(stats['wall_s']), setup_s=float(stats['setup_s']),
                                           process_s=elapsed))
                print(f'{name} repeat {rep+1}: {elapsed:.2f} s', flush=True)
    for name, record in rows.items():
        if 'failure' in record:
            continue
        record['median_integrator_s'] = median(x['integrator_s'] for x in record['runs'])
        record['median_process_s'] = median(x['process_s'] for x in record['runs'])
        method = name.rsplit('-',1)[0]
        for label, reference in [('same_step_cad', f"cad-{record['step_us']}us"),
                                 (f'same_method_{min(steps)}us', f'{method}-{min(steps)}us')]:
            if reference not in signals:
                continue
            a, b = signals[name], signals[reference]
            record[label] = dict(te_rms_difference_um=float(np.sqrt(np.mean((a['te0_um']-b['te0_um'])**2))),
                                 force_rms_difference_N=float(np.sqrt(np.mean((a['normal0_N']-b['normal0_N'])**2))),
                                 peak_difference_N=record['force_peak_N']-rows[reference]['force_peak_N'])
    output.write_text(json.dumps(dict(scope='CPU mechanism-specific flank probes; 24/48 spur; 1800 rpm',
                                      duration_s=2, comparison_grid_us=100, requested_axial_stations=31,
                                      effective_axial_stations=1,
                                      repeats=repeats, configurations=rows),indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('executable','models','work','output'):
        parser.add_argument(key,type=Path)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--refinement',action='store_true',
                        help='Compare the fine mesh and CAD at 5, 10, 20, 50 and 100 microseconds')
    parser.add_argument('--balance',action='store_true', help='Check existing 2 s traces against angular impulse balance')
    parser.add_argument('--replay-poses',type=Path,
                        help='Extract the first entry and exit windows after 0.1 s from completed CAD histories')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    if args.balance:
        check_impulse(args.models,args.work,args.output)
    elif args.replay_poses:
        replay_inputs(args.models, args.work, args.replay_poses)
    else:
        run(args.executable.resolve(),args.models.resolve(),args.work.resolve(),args.output,args.repeats,args.refinement)
