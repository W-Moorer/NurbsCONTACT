"""Check compatibility of the published Flores acceleration with planar clearance."""

import argparse
import hashlib
import json
from pathlib import Path

import fitz
import numpy as np
from scipy.optimize import linprog

from clearance import published_curve


PANELS = {
    "5c-ideal-experiment": (6, 182, 219, 360),
    "5d-clearance-experiment": (6, 3, 40, 360),
    "6b-clearance-experiment": (7, 391, 443, 720),
}
CRANK = .05
ROD = .30
CLEARANCE = .00025
OMEGA = 200*2*np.pi/60


def ideal_position_and_speed(angle):
    sine, cosine = np.sin(angle), np.cos(angle)
    height = CRANK*sine
    projection = np.sqrt(ROD**2-height**2)
    position = CRANK*cosine+projection
    speed = -OMEGA*(height+height*CRANK*cosine/projection)
    return position, speed


def gap_displacement_bound(angle, clearance=CLEARANCE):
    height = CRANK*np.sin(angle)
    ideal = np.sqrt(ROD**2-height**2)
    return np.maximum(np.sqrt((ROD+clearance)**2-height**2)-ideal,
                      ideal-np.sqrt((ROD-clearance)**2-height**2))


def integrated_displacement(time, acceleration):
    step = np.diff(time)
    first, last = acceleration[:-1], acceleration[1:]
    speed = np.r_[0., np.cumsum((first+last)*step/2)]
    increments = speed[:-1]*step+(first/3+last/6)*step**2
    return np.r_[0., np.cumsum(increments)]


def envelope(samples):
    phase, values = samples
    knots, cluster = np.unique(phase, return_inverse=True)
    lower = np.full(len(knots), np.inf)
    upper = np.full(len(knots), -np.inf)
    np.minimum.at(lower, cluster, values)
    np.maximum.at(upper, cluster, values)
    return knots, lower, upper


def minimum_discrepancy(time, low, high):
    # Unknown initial position and speed contribute an arbitrary affine term.
    # These two inequalities relax consistency to pointwise overlap of intervals.
    inequalities = np.column_stack((np.ones(len(time)),time,-np.ones(len(time))))
    solved = linprog([0,0,1], A_ub=np.vstack((inequalities,inequalities*[-1,-1,1])),
                    b_ub=np.r_[-low,high], bounds=[(None,None),(None,None),(0,None)], method="highs")
    if not solved.success:
        raise RuntimeError(solved.message)
    return float(solved.x[2]*1000)


def evaluate_window(acceleration, begin, end, measured_position=None):
    knots, lower, upper = envelope(acceleration)
    relevant = [knots[(knots>begin)&(knots<end)]]
    if measured_position is not None:
        positions, p_low, p_high = envelope(measured_position)
        relevant.append(positions[(positions>begin)&(positions<end)])
    grid = np.unique(np.concatenate((np.linspace(begin,end,2001), *relevant)))
    time = np.deg2rad(grid-begin)/OMEGA
    radians = np.deg2rad(grid)
    position, speed = ideal_position_and_speed(radians)
    ideal = position-position[0]-speed[0]*time
    low = integrated_displacement(time, np.interp(grid,knots,lower))
    high = integrated_displacement(time, np.interp(grid,knots,upper))
    result = {
        "phase_window_deg": [begin,end],
        "necessary_peak_displacement_mm": minimum_discrepancy(time,low-ideal,high-ideal),
        "rigid_planar_clearance_bound_mm": float(np.max(gap_displacement_bound(radians))*1000),
    }
    if measured_position is not None:
        # The graphs can be compared at the same indicated angle even when
        # their raw record/channel synchrony is unknown.
        result["necessary_difference_from_published_position_mm"] = minimum_discrepancy(
            time, low-np.interp(grid,positions,p_high), high-np.interp(grid,positions,p_low))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_pdf",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    report = {
        "source_sha256": hashlib.sha256(args.reference_pdf.read_bytes()).hexdigest(),
        "assumptions": "Published vector paths with linear interpolation between per-angle extrema, "
        "constant 200 rpm, initial position and velocity freely chosen, rigid planar rail and links.",
        "panels": {},
    }
    with fitz.open(args.reference_pdf) as doc:
        travel = published_curve(doc[6],353,450,360,.20,.45)
        for label,(page,frame,path,period) in PANELS.items():
            phase,acceleration = published_curve(doc[page],frame,path,period,-150,200)
            rows = np.flatnonzero((phase>=44)&(phase<=54))
            peak = rows[np.argmin(acceleration[rows])]
            report["panels"][label] = {
                "window_minimum_phase_deg": float(phase[peak]),
                "window_minimum_m_s2": float(acceleration[peak]),
                "compatibility": [
                    evaluate_window((phase,acceleration),*window,
                                    travel if "clearance-experiment" in label else None)
                    for window in [(35,72),(25,85)]
                ],
            }
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
