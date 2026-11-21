"""Release the two auxiliary bearings listed in Table 1 of Flores et al."""

import argparse
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from clearance import REFERENCE, restitution_scale, simulate
from clearance_impact_report import trough_width
from clearance_impacts import LN, impact_block
from clearance_reference_audit import gap_displacement_bound
from clearance_revision import measured
from clearance_spatial import CASES as SPATIAL, score, spatial_deck


CASES = {
    "control": (False, 0., 1.),
    "needle": (False, 1., 1.),
    "both": (True, 1., 1.),
    "both-half": (True, .5, 1.),
    "both-compliant": (True, 1., .1),
}
SHAFT_MASS = 17.9
SHAFT_INERTIA = .460327


def bearing_shapes(exe, directory, name, radius_mm, gap_mm):
    sleeve = f"{name}-bore.step"
    pin = f"{name}-journal.step"
    for filename, arguments in (
        (sleeve, ("annulus", radius_mm+4, 12, radius_mm+gap_mm)),
        (pin, ("cylinder", radius_mm, 12)),
    ):
        path = directory/filename
        if not path.exists():
            subprocess.run([exe, "make", *map(str, arguments), str(path)], check=True, capture_output=True)
    return sleeve, pin


def body(shape, origin_x=0., hinge=-1, coordinates=()):
    translations = (f"{coord} {int(axis==0)} {int(axis==1)} 0"
                    for axis, coord in enumerate(coordinates))
    return " ".join((f'body "{shape}" 0 {origin_x:.17g} 0 0 0 0 1 0 0 0 0 {hinge} 0 0 0',
                     str(len(coordinates)), *translations))


def probe(first, second, radius_mm, gap_mm, scale):
    effective_radius_m = .001*radius_mm*(radius_mm+gap_mm)/gap_mm
    stiffness = scale*(4/3)*(207e9/(2*(1-.3**2)))*math.sqrt(effective_radius_m)
    return (f"probe {first} {second} contained-radial 0 0 0 0 0 0 1 0 0 "
            f"{stiffness:.17g} 1.5 0 0 .001 .0001")


def release_bearings(exe, deck, rpm, release_shaft, gap_scale, stiffness_scale):
    lines = deck.read_text(encoding="ascii").splitlines()
    coordinates = [line for line in lines if line.startswith("coordinate ")]
    bodies = [line for line in lines if line.startswith("body ")]
    probes = [line for line in lines if line.startswith("probe ")]
    attachments = lines[lines.index("resistances 1"):]
    assert len(coordinates)==9 and len(bodies)==5 and len(probes)==3
    omega = rpm*2*math.pi/60
    core = bodies[2].split()
    mass, center = float(core[2]), float(core[10])
    rotation = coordinates[0].split()
    pivot_inertia = float(rotation[1])
    centroidal = pivot_inertia-mass*center**2
    tip_mass = float(coordinates[2].split()[1])
    total_mass = mass+tip_mass
    total_center = (mass*center+tip_mass*.30)/total_mass
    total_centroidal = centroidal+mass*(center-total_center)**2+tip_mass*(.30-total_center)**2
    if not np.allclose([total_mass,total_center,total_centroidal],[1.13,.15,.0153],rtol=0,atol=1e-14):
        raise ValueError("rod mass, first moment or centroidal inertia differ from the baseline")
    mass_properties = {"total_mass_kg":total_mass, "center_from_wrist_m":total_center,
                       "first_moment_about_wrist_kg_m":total_mass*total_center,
                       "total_centroidal_inertia_kgm2":total_centroidal,
                       "core_centroidal_inertia_kgm2":centroidal}
    if not gap_scale:
        with deck.open("a",encoding="ascii") as file:
            file.write(impact_block(3,LN))
        return [],mass_properties

    # With the core center as a free coordinate, the parallel-axis contribution
    # is already in its translation. The hinge now uses the centroidal inertia.
    rotation[1] = f"{centroidal:.17g}"
    coordinates[0] = " ".join(rotation)
    wrist_coordinates = (len(coordinates),len(coordinates)+1)
    coordinates.extend((
        f"coordinate {mass:.17g} {.05+center:.17g} 0 0 0 0 0",
        f"coordinate {mass:.17g} 0 {(.05-center/6)*omega:.17g} 0 0 0 0",
    ))
    needle_radius, needle_gap = 5.,.005*gap_scale
    needle_sleeve, needle_pin = bearing_shapes(exe,deck.parent,"needle",needle_radius,needle_gap)
    bodies[2] = body(needle_sleeve, hinge=0, coordinates=wrist_coordinates)
    rod_bushing = next(i for i,line in enumerate(attachments) if line.startswith("bushing 2 0 "))
    fields = attachments[rod_bushing].split()
    fields[3] = f"{float(fields[3])-center:.17g}"
    attachments[rod_bushing] = " ".join(fields)
    needle_sleeve_id, needle_pin_id = len(bodies), len(bodies)+1
    bodies.append(body(needle_sleeve,-center,0,wrist_coordinates))
    shaft_coordinates = ()
    if release_shaft:
        shaft_coordinates = (len(coordinates),len(coordinates)+1)
        coordinates.extend([f"coordinate {SHAFT_MASS:.17g} 0 0 0 0 0 0"]*2)
    bodies.append(body(needle_pin,coordinates=shaft_coordinates))
    geometry = [(needle_sleeve_id,needle_pin_id,needle_radius,needle_gap)]
    if release_shaft:
        shaft_radius, shaft_gap = 8.5,.009*gap_scale
        shaft_sleeve, shaft_pin = bearing_shapes(exe,deck.parent,"shaft",shaft_radius,shaft_gap)
        geometry.append((len(bodies),len(bodies)+1,shaft_radius,shaft_gap))
        bodies.extend((body(shaft_sleeve),body(shaft_pin,coordinates=shaft_coordinates)))
    probes.extend(probe(a,b,r,g,stiffness_scale) for a,b,r,g in geometry)
    drive = f"{needle_pin_id} orbit .05 0 0 0 0 1 {omega:.17g}"
    header = f"gravity 0 0 0 {len(coordinates)} {len(bodies)} {len(probes)}"
    deck.write_text("\n".join((lines[0], header, *coordinates, *bodies, *probes,
                               "drives 1", drive, *attachments, impact_block(len(probes),LN))),
                    encoding="ascii")
    return geometry,mass_properties


def bearing_metrics(phase, trace, geometry):
    summaries = []
    for index,(first,second,_,gap_mm) in enumerate(geometry,start=3):
        delta = [trace[f"b{first}_{a}_m"]-trace[f"b{second}_{a}_m"] for a in "xy"]
        eccentricity = np.hypot(*delta)
        gap = trace[f"p{index}_gap_m"]
        error = float(np.max(np.abs(gap-(gap_mm*.001-eccentricity))))
        if error > 3e-9:
            raise ValueError(f"secondary probe {index} failed the center-distance check by {error:g} m")
        angular_window = (phase>=43)&(phase<=63)
        normal = trace[f"p{index}_normal_N"]
        summaries.append({
            "geometry_error_m": error,
            "radial_clearance_um": 1000*gap_mm,
            "maximum_load_N": float(np.max(normal)),
            "window_maximum_load_N": float(np.max(normal[angular_window])),
            "window_open_fraction": float(np.mean(gap[angular_window]>0)),
            "window_eccentricity_um": float(np.max(eccentricity[angular_window])*1e6),
        })
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe",type=Path)
    parser.add_argument("work",type=Path)
    parser.add_argument("reference",type=Path)
    parser.add_argument("--cases",nargs="+",choices=tuple(CASES),default=tuple(CASES))
    parser.add_argument("--step-us",type=float,default=10)
    parser.add_argument("--rpm",type=int,default=200)
    parser.add_argument("--clearance",type=float,default=.25)
    parser.add_argument("--cycles",type=int,default=6)
    parser.add_argument("--jobs",type=int,default=3)
    parser.add_argument("--reuse",action="store_true")
    args = parser.parse_args()
    if args.cycles<3 or args.step_us<=0 or args.jobs<=0:
        parser.error("a positive step and worker count and at least three cycles are required")
    exe = str(args.exe.resolve())
    condition = f"{args.rpm}rpm-{args.clearance:.2f}mm"
    reference = measured(args.reference/f"measured-{condition}.csv")
    root = args.work/f"{condition}-{args.step_us:g}us-{args.cycles}cycles"
    damping = restitution_scale()

    def run(case):
        directory = root/case
        directory.mkdir(parents=True,exist_ok=True)
        deck, _ = spatial_deck(exe,directory,args.rpm,args.clearance,damping,SPATIAL["mount-26hz"])
        release_shaft,gap_scale,stiffness_scale = CASES[case]
        geometry,mass_properties = release_bearings(exe,deck,args.rpm,release_shaft,gap_scale,stiffness_scale)
        try:
            phase,trace,stats = simulate(exe,deck,directory,"bearings",0.,
                                         args.step_us*1e-6,args.rpm,args.cycles,args.reuse)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"{case}: {error.stderr}") from error
        result = score(phase,trace,reference,3)
        clearance_sum = (args.clearance+sum(bearing[3] for bearing in geometry))*.001
        rigid_bound = np.max(gap_displacement_bound(np.deg2rad(np.linspace(25,85,601)),clearance_sum))
        result.update({
            "case": case, "step_us":args.step_us, "rpm":args.rpm, "clearance_mm":args.clearance,
            "cycles":args.cycles, "phase_shift_deg":0, "output_filter":None,
            "stiffness_scale": stiffness_scale,
            "rod_mass_properties": mass_properties,
            "rigid_planar_bound_25_to_85_deg_mm": float(1000*rigid_bound),
            "auxiliary_bearings": bearing_metrics(phase,trace,geometry),
            "trough_widths_deg": [trough_width(phase-start,trace["s1_a_m_s2"]) for start in (0,360)],
            "stats":stats,
        })
        (directory/"summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(case,[(round(r["window_min_phase_deg"],3),round(r["window_min_m_s2"],3),
                     round(r["min_load_N"],3)) for r in result["revolutions"]],
              "quiet",round(result["quiet_rmse_m_s2"],3),flush=True)
        return result

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        cases = list(pool.map(run,args.cases))
    report = {"source":REFERENCE, "crank_mass_kg":SHAFT_MASS, "prescribed_crank_inertia_kgm2":SHAFT_INERTIA,
              "cases":cases}
    (root/"summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
