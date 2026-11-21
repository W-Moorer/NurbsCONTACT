"""Bound speed ripple needed to interpret the published acceleration as center motion."""

import argparse
import hashlib
import json
from pathlib import Path

import fitz
import numpy as np
from scipy.optimize import linprog

from clearance import ideal_slider_acceleration, published_curve
from clearance_bearings import SHAFT_INERTIA
from clearance_reference_audit import (
    CRANK, OMEGA, PANELS, ROD, envelope, ideal_position_and_speed, integrated_displacement,
)


def signed_grid(knots, low, high, begin, end):
    # Split integration panels at all acceleration sign changes so that
    # integrating clipped positive and negative parts preserves the linear paths.
    points = [np.array([begin,end]),knots[(knots>begin)&(knots<end)]]
    for values in (low,high):
        edges = np.flatnonzero(values[1:]*values[:-1]<0)
        roots = knots[edges]-values[edges]*np.diff(knots)[edges]/np.diff(values)[edges]
        points.append(roots[(roots>begin)&(roots<end)])
    return np.unique(np.concatenate(points))


def minimum_bound(feasible):
    low,high = 0.,.9
    if feasible(low):
        return low
    if not feasible(high):
        raise ValueError("the necessary conditions failed up to 90% ripple")
    for _ in range(28):
        middle = (low+high)/2
        if feasible(middle):
            high = middle
        else:
            low = middle
    return low


def rigid_clearance_requirement(angle, lower, upper, radial_clearance):
    """Relax time and the acceleration integral independently at each phase."""
    nominal = (angle-angle[0])/OMEGA
    cosine, sine = np.cos(angle),np.sin(angle)
    x = CRANK*cosine+np.sqrt(ROD**2-(CRANK*sine)**2)
    x_lo = CRANK*cosine+np.sqrt((ROD-radial_clearance)**2-(CRANK*sine)**2)-x[0]
    x_hi = CRANK*cosine+np.sqrt((ROD+radial_clearance)**2-(CRANK*sine)**2)-x[0]
    low_pos = integrated_displacement(nominal,np.maximum(0,lower))
    low_neg = integrated_displacement(nominal,np.minimum(0,lower))
    high_pos = integrated_displacement(nominal,np.maximum(0,upper))
    high_neg = integrated_displacement(nominal,np.minimum(0,upper))
    one = np.ones_like(nominal)

    def feasible(ripple):
        h_lo,h_hi = 1/(1+ripple),1/(1-ripple)
        y_lo = h_lo**2*low_pos+h_hi**2*low_neg
        y_hi = h_hi**2*high_pos+h_lo**2*high_neg
        limits = np.r_[x_hi-y_lo,y_hi-x_lo]
        for sign,t_min,t_max in ((-1,h_hi*nominal,h_lo*nominal),(1,h_lo*nominal,h_hi*nominal)):
            # Unknown initial position and velocity are shared by every sample.
            # The sign determines which end of the time interval gives the lower product.
            rows = np.column_stack((np.r_[one,-one],np.r_[t_min,-t_max]))
            velocity = (None,0) if sign<0 else (0,None)
            result = linprog([0,0],A_ub=rows,b_ub=limits,bounds=[(None,None),velocity],method="highs",
                             options={"primal_feasibility_tolerance":1e-9})
            if result.success:
                return True
            if result.status!=2:
                raise RuntimeError(result.message)
        return False

    return minimum_bound(feasible)


def ideal_path_requirement(angle, lower, upper):
    """Use the first integral of the center acceleration along the ideal path."""
    knots, weights = np.polynomial.legendre.leggauss(8)
    middle = (angle[1:]+angle[:-1])/2
    half = np.diff(angle)/2
    sample_points = middle[:,None]+half[:,None]*knots
    _, speed = ideal_position_and_speed(sample_points)
    path_derivative = speed/OMEGA
    if np.any(path_derivative>=0):
        raise ValueError("the chosen window must lie on the inward stroke")

    def integral(samples):
        linear = (samples[:-1,None]*(1-knots)+samples[1:,None]*(1+knots))/2
        change = 2*half*np.sum(linear*path_derivative*weights,axis=1)
        return np.r_[0.,np.cumsum(change)]

    v2_lo,v2_hi = integral(upper),integral(lower)
    _,nominal_speed = ideal_position_and_speed(angle)
    nominal_squared = nominal_speed**2

    def feasible(ripple):
        lower_constant = np.max((1-ripple)**2*nominal_squared-v2_hi)
        upper_constant = np.min((1+ripple)**2*nominal_squared-v2_lo)
        return lower_constant<=upper_constant+1e-10

    return minimum_bound(feasible)


def evaluate(curve, begin=25, end=85):
    knots,low,high = envelope(curve)
    grid = signed_grid(knots,low,high,begin,end)
    angle = np.deg2rad(grid)
    lower,upper = np.interp(grid,knots,low),np.interp(grid,knots,high)
    return {
        "phase_window_deg":[begin,end],
        "knots":len(grid),
        "rigid_clearance_0p264mm_lower_bound_fraction":rigid_clearance_requirement(angle,lower,upper,.000264),
        "exact_ideal_path_lower_bound_fraction":ideal_path_requirement(angle,lower,upper),
    }


def verify():
    phase = np.linspace(25,85,2001)
    nominal_acceleration = ideal_slider_acceleration(phase,200)
    result = evaluate((phase,nominal_acceleration))
    assert result["rigid_clearance_0p264mm_lower_bound_fraction"]<1e-6
    assert result["exact_ideal_path_lower_bound_fraction"]<1e-6
    result = evaluate((phase,ideal_slider_acceleration(phase,214)))
    amplitude = result["exact_ideal_path_lower_bound_fraction"]
    assert 0<amplitude<=.07+1e-6
    angle = np.deg2rad(phase)
    _, nominal_speed = ideal_position_and_speed(angle)
    speed_scale = 1+.05*np.cos(3*angle)
    angular_acceleration = -.15*np.sin(3*angle)*speed_scale*OMEGA**2
    offset = .00014
    position_derivative = nominal_speed/OMEGA+2*offset*np.cos(2*angle)
    position_second_derivative = nominal_acceleration/OMEGA**2-4*offset*np.sin(2*angle)
    acceleration = position_second_derivative*(speed_scale*OMEGA)**2+position_derivative*angular_acceleration
    flexible_path = evaluate((phase,acceleration))
    clearance_result = flexible_path["rigid_clearance_0p264mm_lower_bound_fraction"]
    assert clearance_result<=.05+1e-6
    return {
        "exact_constant_200rpm": "feasible to within one part per million",
        "exact_constant_214rpm_bound": amplitude,
        "variable_speed_within_five_percent_and_position_within_0p14mm_bound": clearance_result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_pdf",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    report = {"source_sha256":hashlib.sha256(args.reference_pdf.read_bytes()).hexdigest(),
              "assumptions":"The plotted extrema are linearly interpolated in encoder angle. "
              "Angular speed is positive. Axial acceleration is assigned to slider-center motion. "
              "The first bound allows total rigid planar clearance of 0.264 mm. "
              "The second fixes the ideal center path. Both leave initial speed free and relax "
              "acceleration envelopes independently at each phase.",
              "analytic_controls":verify(), "panels":{}}
    with fitz.open(args.reference_pdf) as doc:
        for name,(page,frame,path,period) in PANELS.items():
            curve = published_curve(doc[page],frame,path,period,-150,200)
            report["panels"][name] = evaluate(curve)
            if name=="6b-clearance-experiment":
                phase,acceleration = early_peak(*curve)
    _,nominal_speed = ideal_position_and_speed(np.deg2rad(phase))
    x_derivative = nominal_speed/OMEGA
    nominal_acceleration = float(ideal_slider_acceleration(phase,200))
    speed_limits = [1-.01,1+.01]
    inertial = sorted(nominal_acceleration*rate**2 for rate in speed_limits)
    alpha_interval = sorted((acceleration-term)/x_derivative for term in inertial)
    report["ideal_path_within_one_percent_at_peak"] = {
        "phase_deg":phase,"target_acceleration_m_s2":acceleration,
        "required_angular_acceleration_rad_s2":alpha_interval,
        "required_crank_inertia_torque_N_m":[a*SHAFT_INERTIA for a in alpha_interval],
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    output = json.dumps(report,indent=2)+"\n"
    args.output.write_text(output,encoding="utf-8")
    print(output)


def early_peak(phase,acceleration):
    window = np.flatnonzero((phase>=44)&(phase<=54))
    minimum = window[np.argmin(acceleration[window])]
    return float(phase[minimum]),float(acceleration[minimum])


if __name__=="__main__":
    main()
