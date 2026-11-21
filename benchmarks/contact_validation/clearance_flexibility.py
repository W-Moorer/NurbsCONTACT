"""Test explicit support and rod-tip flexibility with the shared STEP/HHT runner."""

import argparse
import gzip
import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from clearance import RigDissipation, restitution_scale, simulate, write_deck
from clearance_revision import CALIBRATION, DAMPED, assess, measured


@dataclass(frozen=True)
class Flexibility:
    support_x: float
    support_y: float
    rod_axial: float
    rod_bending: float
    damping_ratio: float
    pin_mass: float = .10
    tip_mass: float = .12
    guide_lateral: float = 0.


CONFIGURATIONS = {
    "reference": Flexibility(1.6e6, 8e6, 80e6, 1.4e6, .16),
    "soft": Flexibility(.45e6, 3e6, 50e6, .35e6, .22),
    "stiff": Flexibility(8e6, 28e6, 200e6, 5e6, .12),
    "light-damping": Flexibility(1.6e6, 8e6, 80e6, 1.4e6, .045),
    "damped": Flexibility(1.6e6, 8e6, 80e6, 1.4e6, .35),
    "lateral": Flexibility(1.6e6, 1.6e6, 80e6, .35e6, .16),
    "lateral-damped": Flexibility(1.6e6, 1.6e6, 80e6, .35e6, .36),
    "lateral-soft": Flexibility(.7e6, 1.6e6, 80e6, .35e6, .30),
    "compliant-guide": Flexibility(1.6e6, 1.6e6, 80e6, .35e6, .22, guide_lateral=.45e6),
}
LOCAL_CONTACT = RigDissipation(1.5, 1., DAMPED.guide_damping, DAMPED.plate_friction)


def flex_deck(exe, work, rpm, clearance, damping_scale, flex):
    path = write_deck(exe, work, rpm, clearance, damping_scale, LOCAL_CONTACT)
    rigid = path.read_text(encoding="ascii").splitlines()
    sleeve, pin = (line.split('"')[1] for line in rigid if line.startswith("body "))
    probe = next(line for line in rigid if line.startswith("probe "))
    omega = rpm*2*math.pi/60
    rod_mass, slider_mass, length, offset = 1.13, 1.013, .30, .15
    core_mass = rod_mass-flex.tip_mass
    core_center = (rod_mass*offset-flex.tip_mass*length)/core_mass
    core_inertia = .0153+rod_mass*offset**2-flex.tip_mass*length**2
    carriage_mass = slider_mass-flex.pin_mass
    assert min(core_mass, carriage_mass, core_inertia)>0
    lateral_coordinate = ""
    carriage_translations = "1 1 1 0 0"
    if flex.guide_lateral:
        guide_drag = 2*flex.damping_ratio*math.sqrt(flex.guide_lateral*carriage_mass)
        lateral_coordinate = f"coordinate {carriage_mass:.17g} 0 0 {flex.guide_lateral} {guide_drag:.17g} 0 0\n"
        carriage_translations = "2 1 1 0 0 6 0 1 0"

    def bushing(a, b, anchor_a, x, y, mass):
        cx = 2*flex.damping_ratio*math.sqrt(x*mass)
        cy = 2*flex.damping_ratio*math.sqrt(y*mass)
        return f"bushing {a} {b} {anchor_a} 0 0 0 0 0 0 0 0 {x:.17g} {y:.17g} 0 {cx:.17g} {cy:.17g} 0"

    path.write_text(f"""nc-contact-v2
gravity 0 0 0 {6+bool(flex.guide_lateral)} 4 1
coordinate {core_inertia:.17g} 0 {-omega/6:.17g} 0 0 0 0
coordinate {carriage_mass:.17g} .35 0 0 {DAMPED.guide_damping} 0 0
coordinate {flex.tip_mass} .35 0 0 0 0 0
coordinate {flex.tip_mass} 0 0 0 0 0 0
coordinate {flex.pin_mass} .35 0 0 0 0 0
coordinate {flex.pin_mass} 0 0 0 0 0 0
{lateral_coordinate}body "{sleeve}" 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 2 2 1 0 0 3 0 1 0
body "{pin}" 0 0 0 0 0 0 1 0 0 0 0 -1 0 0 0 2 4 1 0 0 5 0 1 0
body "{sleeve}" {core_mass:.17g} 0 0 0 0 0 1 0 {core_center:.17g} 0 0 0 0 0 0 0
body "{pin}" 0 0 0 0 0 0 1 0 0 0 0 -1 0 0 0 {carriage_translations}
{probe}
drives 1
2 orbit .05 0 0 0 0 1 {omega:.17g}
resistances 1
slide 0 3 0 0 0 0 0 0 0 0 1 {DAMPED.plate_friction} .001
bushings 2
{bushing(2,0,".3",flex.rod_axial,flex.rod_bending,flex.tip_mass)}
{bushing(3,1,"0",flex.support_x,flex.support_y,flex.pin_mass)}
""", encoding="ascii")
    return path


def impact_scores(phase, trace, experiment):
    enrolled = (phase>=0)&(phase<=720)
    local = phase % 360
    event = enrolled & (local>=42) & (local<=53)
    error = trace["a1"]-np.interp(phase,*experiment)
    measured_window = np.flatnonzero((experiment[0]>=42)&(experiment[0]<=53))
    measured_peak = measured_window[np.argmin(experiment[1][measured_window])]
    angle, center = experiment[:,measured_peak]
    local_peaks = []
    for offset in (0, 360):
        which = np.flatnonzero((phase>=offset+42)&(phase<=offset+53))
        k = which[np.argmin(trace["a1"][which])]
        local_peaks.append({
            "phase_deg": float(phase[k]-offset), "acceleration_m_s2": float(trace["a1"][k]),
            "interior": bool(k not in (which[0],which[-1])),
            "at_measured_phase_m_s2": float(np.interp(angle+offset,phase,trace["a1"])),
        })
    return {
        "measured_peak_phase_deg": float(angle), "measured_peak_m_s2": float(center),
        "impact_window_rmse_m_s2": float(np.sqrt(np.mean(error[event]**2))),
        "impact_peaks": local_peaks,
    }


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11.3,3.2), layout="constrained")
    traces = [(label, np.loadtxt(results/f"{name}.csv",delimiter=",",skiprows=1))
              for name,label in (("rigid","Equivalent compliance"),("flex","Structural flexibility"))]
    for ax,limits,title in zip(axes,((0,360),(35,67),(90,180)),
                               ("Full revolution","Impact window","Sustained loading")):
        for (label,data),color in zip(traces,("#ae674b","#007e92")):
            x, experiment, predicted, lo, hi = data.T
            keep = (x>=limits[0])&(x<=limits[1])
            ax.fill_between(x[keep],lo[keep],hi[keep],color=color,alpha=.20,linewidth=0)
            ax.plot(x[keep],predicted[keep],color=color,lw=1.1,label=label)
        ax.plot(x[keep],experiment[keep],color="#25303a",lw=1.25,label="Experiment")
        ax.set(xlabel="Crank angle (°)",xlim=limits,title=title)
        ax.grid(axis="y",alpha=.25)
        ax.spines[["right","top"]].set_visible(False)
    axes[0].set_ylabel("Slider acceleration (m/s²)")
    axes[0].legend(frameon=False,fontsize=8)
    fig.savefig(results/"structural-flexibility.pdf")
    fig.savefig(results/"structural-flexibility.png",dpi=165)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe",type=Path)
    parser.add_argument("work",type=Path)
    parser.add_argument("source",type=Path)
    parser.add_argument("results",type=Path)
    parser.add_argument("--reuse-traces",action="store_true")
    parser.add_argument("--selected",choices=tuple(CONFIGURATIONS),help="fixed physical variant for refinement")
    args = parser.parse_args()
    rpm, clearance = CALIBRATION
    name = f"{rpm}rpm-{clearance:.2f}mm"
    exp = measured(args.source/f"measured-{name}.csv")
    travel = measured(args.source/"measured-position-fig5b.csv")
    exe = str(args.exe.resolve())
    damping = restitution_scale()
    args.results.mkdir(parents=True,exist_ok=True)

    def run(label, flex, step, save=False):
        work = args.work/label
        work.mkdir(parents=True,exist_ok=True)
        deck = (write_deck(exe,work,rpm,clearance,damping,DAMPED) if flex is None else
                flex_deck(exe,work,rpm,clearance,damping,flex))
        phase,trace,stats = simulate(exe,deck,work,label,0.,step,rpm,6,args.reuse_traces)
        metrics,curves = assess(phase,trace,exp,travel,rpm,clearance)
        metrics |= impact_scores(phase,trace,exp)
        print(label,metrics["raw_sample_rmse_m_s2"],metrics["quiet_window_rmse_m_s2"],
              metrics["impact_window_rmse_m_s2"],metrics["impact_peaks"],flush=True)
        if save:
            np.savetxt(args.results/f"{label}.csv",curves,delimiter=",",
                       header="angle_deg,measured_m_s2,simulated_m_s2,sample_min_m_s2,sample_max_m_s2",comments="")
            raw = np.column_stack((phase, trace["time_s"], trace["q1"], trace["v1"], trace["a1"],
                                   trace["p0_gap_m"], trace["p0_normal_N"],
                                   trace["b0_x_m"], trace["b0_y_m"], trace["b1_x_m"], trace["b1_y_m"]))
            with gzip.open(args.results/f"{label}-raw.csv.gz","wt",encoding="ascii",newline="") as out:
                np.savetxt(out,raw,fmt="%.17g",delimiter=",",
                           header="angle_deg,time_s,slider_x_m,slider_v_m_s,slider_a_m_s2,gap_m,normal_N,"
                           "sleeve_x_m,sleeve_y_m,pin_x_m,pin_y_m",comments="")
        return {"case":label,"parameters":asdict(flex) if flex else None} | metrics | {
            key:stats[key] for key in ("step_s","steps","accepted","subdivisions","missing")}

    if args.selected is None:
        with ThreadPoolExecutor(max_workers=3) as pool:
            rows = list(pool.map(lambda item:run(*item,10e-6),CONFIGURATIONS.items()))
        (args.results/"screening.json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")
        return
    chosen = CONFIGURATIONS[args.selected]
    with ThreadPoolExecutor(max_workers=3) as pool:
        cases = (("rigid",None,10e-6),("flex",chosen,10e-6),("flex-half",chosen,5e-6))
        runs = list(pool.map(lambda item:run(*item,save=True),cases))
    shutil.copyfile(args.work/"flex"/f"{name}.scene",args.results/f"{name}.scene")
    for file in (args.work/"flex").glob("*.step"):
        shutil.copyfile(file,args.results/file.name)
    coarse = np.loadtxt(args.results/"flex.csv",delimiter=",",skiprows=1,usecols=2)
    fine = np.loadtxt(args.results/"flex-half.csv",delimiter=",",skiprows=1,usecols=2)
    (args.results/"summary.json").write_text(json.dumps({
        "rpm":rpm,"radial_clearance_mm":clearance,"selected":args.selected,"runs":runs,
        "phase_shift_deg":0,"output_filter":None,
        "half_step_common_grid_delta_m_s2":float(np.sqrt(np.mean((coarse-fine)**2))),
    },indent=2)+"\n",encoding="utf-8")
    plot(args.results)


if __name__=="__main__":
    main()
