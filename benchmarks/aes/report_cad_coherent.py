"""Report full-duration coherent CAD ablations; retain failed numerical comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from report_cad_batched import load
from report_cad_segments import differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    control, original = load(args.runs / 'control')
    names = [('control', 'Frozen control'), ('mask-0', 'New binary, off'),
             ('mask-1', '1: parameter cache'), ('mask-2', '2: rule blocks'),
             ('mask-4', '3: local links'), ('mask-8', '4: region queue'),
             ('mask-15', 'All four')]
    for folder, label in [('all-repeat', 'All four, repeat'), ('mask-9', '1 + 4'),
                          ('mask-9-repeat', '1 + 4, repeat')]:
        if (args.runs/folder/'run.json').exists():
            names.append((folder,label))
    results = []
    for folder, label in names:
        info, data = load(args.runs / folder)
        actual_hash = hashlib.sha256((args.runs/folder/'response.csv').read_bytes()).hexdigest()
        if actual_hash != info['response_sha256']:
            raise ValueError(f'Response file changed after timing: {folder}')
        if info['scene_sha256'] != control['scene_sha256']:
            raise ValueError('Changed scene')
        delta = differences(original, data)
        flags = [k for k in data.dtype.names if k.endswith('_hit')]
        events = {k: int(np.count_nonzero(data[k] != original[k])) for k in flags}
        results.append(dict(label=label, folder=folder, run=info, differences=delta,
                            exact_values=all(v['maximum'] == 0 and v['same_missing_pattern'] for v in delta.values()),
                            same_bytes=info['response_sha256'] == control['response_sha256'],
                            contact_mismatch_frames=events))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'ablations.json').write_text(json.dumps(results, indent=2)+'\n', encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5,.6*len(results))), layout='constrained')
    times = [float(r['run']['gpu']['online_s']) for r in results]
    colors = ['#68757d']*2+['#3682a0']*4+['#c5683a']*(len(results)-6)
    bars = axes[0].barh([r['label'] for r in results], times, color=colors)
    axes[0].bar_label(bars, fmt='%.2f s', padding=4)
    axes[0].invert_yaxis()
    axes[0].set(xlabel='Online simulation time [s]', xlim=(0, max(times)*1.22))
    stages = ['broad_s', 'sort_s', 'search_s', 'group_s', 'integral_s']
    y = np.arange(len(results)); left = np.zeros(len(results))
    for field, label in zip(stages, ['Broad', 'Sort', 'Confirm', 'Connect', 'Integrate']):
        values = np.array([float(r['run']['gpu'][field]) for r in results])
        axes[1].barh(y, values, left=left, label=label); left += values
    axes[1].set(yticks=y, yticklabels=[r['label'] for r in results], xlabel='Geometry stage time [s]')
    axes[1].invert_yaxis(); axes[1].legend(loc='upper center', bbox_to_anchor=(.5, -.09), ncol=5, fontsize=8)
    fig.suptitle('Complete reducer: 5 s, 0.1 ms; preprocessing excluded')
    fig.savefig(args.out/'ablations.png', dpi=170)
    plt.close(fig)
    cpu_path = Path(__file__).resolve().parents[2] / 'results/aes/cpu-bezier/standalone/simulation-timing/cpu-worker-scaling.json'
    cpu_record = json.loads(cpu_path.read_text(encoding='utf-8'))
    cpu = {r['method']: r for r in cpu_record['cpu_runs']
           if r['variant'] == 'fused' and r['threads'] == 4 and r['steps'] == 50000}
    mesh = json.loads((args.runs/'mesh/run.json').read_text(encoding='utf-8'))
    if not (mesh['completed'] and mesh['duration_s'] == 5 and mesh['step_s'] == .0001
            and mesh['steps'] == 50000 and mesh['scene_sha256'] == control['scene_sha256']):
        raise ValueError('GPU mesh timing is not a complete matched experiment')
    mesh_response = np.load(args.runs/'mesh/response.npy')
    coordinates = sum(k.startswith('q') and k[1:].isdigit() for k in original.dtype.names)
    pairs = sum(k.endswith('_hit') for k in original.dtype.names)
    gaps = {1 + 4*coordinates + 3*i for i in range(pairs)}
    required = [i for i in range(mesh_response.shape[1]) if i not in gaps]
    if (mesh_response.shape != (50001, 1 + 4*coordinates + 3*pairs)
            or np.isinf(mesh_response).any() or not np.isfinite(mesh_response[:, required]).all()):
        raise ValueError('Incomplete or invalid GPU mesh response')
    np.savez_compressed(args.out/'mesh-response.npz', response=mesh_response)
    lookup = {r['folder']: r['run'] for r in results}
    rows = []
    for key, label in [('mesh-obb', 'CPU mesh (archived)'), ('cad-compiled', 'CPU CAD (archived)')]:
        rows.append(dict(label=label, online_s=[cpu[key]['wall_s']], source=str(cpu_path), rerun=False))
    rows.append(dict(label='GPU mesh', online_s=[mesh['online_s']], source='runs/mesh/run.json', rerun=True))
    for folders, label in [(['control'], 'GPU CAD before'),
                           (['mask-15', 'all-repeat'], 'GPU CAD all four'),
                           (['mask-9', 'mask-9-repeat'], 'GPU CAD 1 + 4 (ours)')]:
        present = [f for f in folders if f in lookup]
        if present:
            rows.append(dict(label=label, online_s=[float(lookup[f]['gpu']['online_s']) for f in present],
                             source=[f'runs/{f}/run.json' for f in present], rerun=True))
    backend_report = dict(rows=rows, cpu_rerun=False, duration_s=5, step_s=.0001,
                          mesh_response_sha256=hashlib.sha256((args.runs/'mesh/response.npy').read_bytes()).hexdigest(),
                          mesh_response_shape=list(mesh_response.shape),
                          scope='Online timing only. Mesh/CAD timing comparison does not establish equal physical accuracy.')
    (args.out/'backend-comparison.json').write_text(json.dumps(backend_report, indent=2)+'\n', encoding='utf-8')
    fig, ax = plt.subplots(figsize=(10, 4.6), layout='constrained')
    means = [np.mean(r['online_s']) for r in rows]
    bars = ax.barh([r['label'] for r in rows], means,
                   color=['#8c969c', '#8c969c', '#3682a0', '#68757d', '#c5683a', '#a9552c'][:len(rows)])
    ax.bar_label(bars, labels=[f'{t:.2f} s' for t in means], padding=5)
    for y, row in enumerate(rows):
        if len(row['online_s']) > 1:
            ax.plot(row['online_s'], [y]*len(row['online_s']), '|', color='black', ms=12)
    ax.invert_yaxis()
    ax.set(xlabel='Online time [s]; bars average available repeats', xlim=(0, max(means)*1.18))
    fig.suptitle('Complete reducer: 5 s / 0.1 ms; preprocessing excluded')
    fig.savefig(args.out/'backends.png', dpi=170)
    plt.close(fig)
    for r, t in zip(results, times):
        print(r['folder'], t, 'exact=', r['exact_values'], 'events=', r['contact_mismatch_frames'])
    if any(not r['exact_values'] or not r['same_bytes'] or any(r['contact_mismatch_frames'].values())
           for r in results):
        raise SystemExit('Response changed; scheduling-only acceptance requires identical full histories')


if __name__ == '__main__':
    main()
