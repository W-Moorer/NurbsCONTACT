"""Compare episode-dependent damping against the Flores slider-crank measurements."""

import argparse
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from clearance import restitution_scale, simulate, write_deck
from clearance_revision import MOUNTED, measured
from clearance_spatial import CASES as SPATIAL, score, spatial_deck


RESTITUTION = .46
LN = 3*(1-RESTITUTION**2)/4
APPROACH_FLOOR = 1e-4
CASES = {
    "rigid-ln": (None,"ln"),
    "spatial-ln": ("long-lever","ln"),
    "mount-ln": ("mount-26hz","ln"),
    "spatial-exact": ("long-lever","exact"),
    "mount-exact": ("mount-26hz","exact"),
}


def dissipation_integral(z):
    return z-math.log1p(z)


def restitution_for(coefficient):
    """Velocity ratio for isolated compression and rebound at zero depth."""
    initial = dissipation_integral(coefficient)
    return brentq(lambda e:dissipation_integral(-coefficient*e)-initial,
                  0,min(1,(1-1e-12)/coefficient))


def exact_coefficient():
    return brentq(lambda x:dissipation_integral(x)-dissipation_integral(-x*RESTITUTION),
                  1e-4,(1-1e-12)/RESTITUTION)


def impact_block(sections, coefficient):
    return "\n".join([f"impact-laws {sections}",
        *(f"impact {i} {coefficient:.17g} {APPROACH_FLOOR}" for i in range(sections))])+"\n"


def verify_impact(exe, target):
    target.mkdir(parents=True,exist_ok=True)
    for name,params in (("plate.step",("box","120","120","10","-5")),
                        ("sphere.step",("sphere","5"))):
        shape = target/name
        if not shape.exists():
            subprocess.run([exe,"make",*params,str(shape)],check=True,capture_output=True)
    outcomes = []
    for law,coefficient in (("ln",LN),("exact",exact_coefficient())):
        for speed in (.03,.15):
            scene = target/"drop.scene"
            scene.write_text(f"""nc-contact-v2
gravity 0 0 0 1 2 1
coordinate 1 .001357 {-speed} 0 0 0 0
body "plate.step" 0 0 0 0 0 0 1 0 0 0 0 -1 0 0 0 0
body "sphere.step" 0 0 0 .005 0 0 1 0 0 0 0 -1 0 0 0 1 0 0 0 1
probe 0 1 world 0 0 0 0 0 0 0 0 1 4e7 1.5 0 0 .001 .003
drives 0
"""+impact_block(1,coefficient),encoding="ascii")
            prefix = target/f"drop-{law}-{int(round(1000*speed))}mmps"
            subprocess.run([exe,str(scene),"0","1e-6",str(.001357/speed+.012),"1",str(prefix)],
                           check=True,capture_output=True)
            trace = np.genfromtxt(prefix.with_suffix(".csv"),delimiter=",",names=True)
            closed = trace["p0_gap_m"]<0
            entry, exit_ = np.flatnonzero(closed)[[0,-1]]
            assert exit_>entry and np.all(closed[entry:exit_+1]) and exit_<len(trace)-1
            incidence = trace["p0_impact_speed_m_s"]
            measured = float(trace["v0"][exit_+1]/speed)
            expected = restitution_for(coefficient)
            assert np.all(incidence[~closed]==0)
            assert np.allclose(incidence[closed],speed,rtol=5e-5,atol=1e-8)
            depth = -trace["p0_gap_m"][closed]
            force = 4e7*depth**1.5*np.maximum(0,1-coefficient*trace["p0_gap_rate_m_s"][closed]/incidence[closed])
            assert np.allclose(force,trace["p0_normal_N"][closed],rtol=1e-11,atol=1e-8)
            if abs(measured-expected)>3e-4:
                raise ValueError(f"{law} rebound ratio {measured:g} differs from analytic {expected:g}")
            outcomes.append({"law":law,"entry_m_s":speed,"expected_rebound_ratio":expected,
                             "measured_rebound_ratio":measured})
            print(f"{law} at {speed:g} m/s: rebound {measured:.6f} vs {expected:.6f}",flush=True)
    (target/"verification.json").write_text(json.dumps(outcomes,indent=2)+"\n",encoding="utf-8")


def entry_counts(phase, trace, sections):
    active = (phase>=0)&(phase<=720)
    gaps = np.column_stack([trace[f"p{i}_gap_m"] for i in range(sections)])
    impact = np.column_stack([trace[f"p{i}_impact_speed_m_s"] for i in range(sections)])
    entering = (gaps[1:]<0)&(gaps[:-1]>=0)
    first = entering & active[1:,None]
    in_contact = (gaps[1:]<0)&(gaps[:-1]<0)&active[1:,None]
    # An exit/re-entry between output samples can also change the recorded rate.
    speed_changes = in_contact & (impact[1:]!=impact[:-1])
    return {
        "sampled_entries": first.sum(axis=0).tolist(),
        "at_approach_floor": (first & (impact[1:]<=APPROACH_FLOOR*(1+1e-12))).sum(axis=0).tolist(),
        "within_sample_history_changes": speed_changes.sum(axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe",type=Path)
    parser.add_argument("work",type=Path)
    parser.add_argument("reference",type=Path)
    parser.add_argument("--cases",nargs="+",choices=tuple(CASES),default=tuple(CASES))
    parser.add_argument("--step-us",type=float,default=10)
    parser.add_argument("--cycles",type=int,default=6)
    parser.add_argument("--rpm",type=int,default=200)
    parser.add_argument("--clearance",type=float,default=.25)
    parser.add_argument("--jobs",type=int,default=3)
    parser.add_argument("--verify",action="store_true")
    parser.add_argument("--reuse",action="store_true")
    args = parser.parse_args()
    if args.cycles<3 or args.jobs<1 or args.step_us<=0:
        parser.error("positive step and workers and at least three cycles are required")
    exe = str(args.exe.resolve())
    if args.verify:
        verify_impact(exe,args.work/"impact-check")
    ref = measured(args.reference/f"measured-{args.rpm}rpm-{args.clearance:.2f}mm.csv")
    fixed_damping = restitution_scale()
    tag = f"{args.rpm}rpm-{args.clearance:.2f}mm-{args.step_us:g}us-{args.cycles}cycles"

    def run(case):
        base,law = CASES[case]
        target = args.work/tag/case
        target.mkdir(parents=True,exist_ok=True)
        if base is None:
            deck = write_deck(exe,target,args.rpm,args.clearance,fixed_damping,MOUNTED)
            with deck.open("a",encoding="ascii") as f:
                f.write("\naccelerometers 2\naccelerometer 1 0 0 0 1 0 0\naccelerometer 1 0 0 0 1 0 0\n")
            sections = 1
        else:
            deck,sections = spatial_deck(exe,target,args.rpm,args.clearance,fixed_damping,SPATIAL[base])
        coefficient = LN if law=="ln" else exact_coefficient()
        with deck.open("a",encoding="ascii") as f:
            f.write(impact_block(sections,coefficient))
        phase,trace,stats = simulate(exe,deck,target,"impact",0.,args.step_us*1e-6,args.rpm,args.cycles,args.reuse)
        result = score(phase,trace,ref,sections)
        early = (phase>=0)&(phase<180)
        index = np.flatnonzero(early)[np.argmin(trace["s1_a_m_s2"][early])]
        result.update({"case":case,"rpm":args.rpm,"radial_clearance_mm":args.clearance,"cycles":args.cycles,
                       "step_us":args.step_us,"phase_shift_deg":0,"output_filter":None,
                       "dissipation_coefficient":coefficient,"approach_floor_m_s":APPROACH_FLOOR,
                       "first_half_minimum_phase_deg":float(phase[index]),
                       "first_half_minimum_m_s2":float(trace["s1_a_m_s2"][index]),"integration":stats})
        result.update(entry_counts(phase,trace,sections))
        (target/"summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        compact = [(round(r["window_min_phase_deg"],3),round(r["window_min_m_s2"],3),
                    round(r["min_load_N"],3)) for r in result["revolutions"]]
        print(case,compact,"quiet RMSE",round(result["quiet_rmse_m_s2"],3),
              "entries",result["sampled_entries"],flush=True)
        return result

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(run,args.cases))
    (args.work/tag/"summary.json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
