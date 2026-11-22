"""Plot the first 3 ms following the load-increase midpoint from computed HHT histories."""
import argparse
import csv
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def window(path):
    points = []
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            t = float(row['time'])
            if t > .103+1e-10:
                break
            if t >= .100-1e-10:
                points.append((1000*t, float(row['normal0_N'])))
    return np.array(points)


def main(work, dest):
    dest.mkdir(parents=True, exist_ok=True)
    series = {}
    for method, step in [('mesh-391nm',5),('cad',50),('mesh-391nm',50),('cad',100),('mesh-391nm',100)]:
        data = window(work/f'{method}-{step}us.csv')
        if data.shape!=(3000//step+1,2) or not np.isfinite(data).all():
            raise ValueError(f'Incomplete {method} history at {step} microseconds')
        series[method,step] = data
    with (dest/'spur-force-window.csv').open('w',newline='') as out:
        table = csv.writer(out)
        table.writerow(['method','step_us','time_ms','normal_force_N'])
        for (method,step), data in series.items():
            table.writerows((method,step,t,f) for t,f in data)

    mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
    fig, axes = plt.subplots(1,2,figsize=(7.1,2.72),sharex=True,sharey=True)
    fig.subplots_adjust(left=.09,right=.982,bottom=.20,top=.77,wspace=.13)
    for ax, step in zip(axes,(50,100)):
        for method, h, label, style in [
            ('mesh-391nm',5,'Fine mesh, 5 µs',dict(color='#797b80',lw=1.3)),
            ('cad',step,'CAD flank',dict(color='#225a91',lw=1.7)),
            ('mesh-391nm',step,'Fine mesh',dict(color='#ae531d',lw=1.0,ls='--',marker='o',
                                                ms=2.3,mfc='white',mew=.6))]:
            t,f = series[method,h].T
            ax.plot(t,f/1000, label=label, **style)
        ax.set(title=f'{step} µs step',xlabel='Time (ms)',xlim=(100,103),
               xticks=(100,101,102,103),ylim=(0,None))
        ax.grid(axis='y',color='.90',lw=.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Normal contact force (kN)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.52,.995))
    fig.savefig(dest/'spur-force-window.pdf')
    fig.savefig(dest/'spur-force-window.png',dpi=240)
    plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    main(args.work,args.output)
