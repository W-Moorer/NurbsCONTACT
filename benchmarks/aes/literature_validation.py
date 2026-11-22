"""Reproduce the published unilateral oscillator and constant-breadth cam."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np


def rms(values):
    return float(np.sqrt(np.mean(values**2)))


def oscillator_reference(time, omega, epsilon, amplitude):
    loaded = np.sqrt(omega**2 + epsilon)
    half_contact = np.pi / (2 * loaded)
    free_duration = np.pi / omega
    period = 2 * half_contact + free_duration
    phase = np.remainder(time, period)
    displacement = np.where(phase < half_contact, amplitude * np.cos(loaded * phase),
        np.where(phase < half_contact + free_duration,
                 -amplitude * loaded / omega * np.sin(omega * (phase - half_contact)),
                 amplitude * np.sin(loaded * (phase - half_contact - free_duration))))
    return displacement, half_contact, free_duration, period


def oscillator(executable, output):
    # Section 2 gives free parameters and release from rest at a positive amplitude.
    omega, epsilon, amplitude = 1., .1, 1.
    cycles = 50
    _, half_contact, free_duration, period = oscillator_reference(0., omega, epsilon, amplitude)
    exact_events = np.ravel(np.arange(cycles)[:, None] * period +
                            [half_contact, half_contact + free_duration])
    sweeps = []
    for index, dt in enumerate((.01, .005, .0025)):
        trace = output / f"oscillator-{index}.csv"
        subprocess.run([str(executable), "unilateral-oscillator", str(omega), str(epsilon),
                        str(amplitude), str(dt), str(cycles * period), str(trace)], check=True)
        table = np.genfromtxt(trace, delimiter=",", names=True)
        t, q, f = (table[name] for name in ("time", "displacement", "contact_force"))
        exact_q, _, _, _ = oscillator_reference(t, omega, epsilon, amplitude)
        exact_f = epsilon * np.maximum(exact_q, 0)
        switched, = np.nonzero((q[1:] > 0) != (q[:-1] > 0))
        if len(switched) != len(exact_events) or np.any(q[switched[::2]] <= 0):
            raise ValueError("Unexpected number or order of oscillator transitions")
        events = t[switched] - q[switched] * (t[switched + 1] - t[switched]) / (q[switched + 1] - q[switched])
        event_error = events - exact_events
        sweeps.append({"step": float(t[1]), "steps": len(t) - 1, "displacement_rms": rms(q - exact_q),
                       "force_rms": rms(f - exact_f), "max_event_error": float(np.max(np.abs(event_error))),
                       "last_onset_error": float(event_error[-1]),
                       "max_contact_duration_error": float(np.max(np.abs(events[2::2] - events[1:-1:2] - 2 * half_contact))),
                       "max_detached_duration_error": float(np.max(np.abs(events[1::2] - events[::2] - free_duration))),
                       "subdivisions": int(table["subdivisions"][-1]),
                       "max_residual": float(table["max_residual"][-1])})
        if index == 2:
            np.savetxt(output / "oscillator-events.csv", np.column_stack((exact_events, events, event_error)),
                       delimiter=",", header="exact_time,hht_time,error", comments="")
    if not all(a["max_event_error"] > b["max_event_error"] and a["displacement_rms"] > b["displacement_rms"]
               for a, b in zip(sweeps, sweeps[1:])):
        raise ValueError("Oscillator errors do not decrease when refining the step")
    return {"omega_0": omega, "epsilon": epsilon, "amplitude": amplitude, "cycles": cycles,
            "period": period, "contact_duration": 2 * half_contact, "detached_duration": free_duration,
            "observed_transitions": len(exact_events), "steps": sweeps}


def cam_reference(angle, a, b, c):
    """Table 2 at x0=y0=0. Branches follow input angle, not the CAD representation."""
    alpha = np.arccos((a*a + c*c - b*b) / (2*a*c))
    beta = np.arccos((a*a + b*b - c*c) / (2*a*b))
    switches = np.array([0., beta, np.pi - alpha, np.pi, np.pi + beta, 2*np.pi - alpha, 2*np.pi])
    phase = np.remainder(angle, 2*np.pi)
    segment = np.searchsorted(switches[1:-1], phase, side="right")
    centers = np.array([0., b, b-c, a+b-c, a-c, a])
    cosine = np.array([0., -b, -a, 0., -b, -a])
    offset = np.array([0., beta, 0., 0., beta, 0.])
    p = phase - offset[segment]
    displacement = centers[segment] + cosine[segment] * np.cos(p)
    velocity = -cosine[segment] * np.sin(p)
    acceleration = -cosine[segment] * np.cos(p)
    return displacement, velocity, acceleration, switches


def cam_shapes(a, b, c, d):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.gp import gp_Ax2, gp_Ax3, gp_Circ, gp_Dir, gp_Pln, gp_Pnt, gp_Vec

    alpha = np.arccos((a*a + c*c - b*b) / (2*a*c))
    beta = np.arccos((a*a + b*b - c*c) / (2*a*b))
    center_a = (-a, 0.)
    center_b = (0., 0.)
    center_c = (-b * np.cos(beta), b * np.sin(beta))
    arcs = ((center_a, a+d, 0., alpha), (center_c, a+d-c, alpha, np.pi-beta),
            (center_b, a+b+d-c, np.pi-beta, np.pi), (center_a, b+d-c, np.pi, np.pi+alpha),
            (center_c, b+d, np.pi+alpha, 2*np.pi-beta), (center_b, d, 2*np.pi-beta, 2*np.pi))
    builder = BRepBuilderAPI_MakeWire()
    for (x, y), radius, start, end in arcs:
        circle = gp_Circ(gp_Ax2(gp_Pnt(x, y, 0), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0)), radius)
        builder.Add(BRepBuilderAPI_MakeEdge(circle, start, end).Edge())
    wire = builder.Wire()
    if not wire.Closed():
        raise ValueError("Table 1 does not close the cam profile")
    surface = BRepPrimAPI_MakePrism(wire, gp_Vec(0, 0, 1)).Shape()
    support_x = 9.
    plane = gp_Pln(gp_Ax3(gp_Pnt(support_x, 0, 0), gp_Dir(1, 0, 0), gp_Dir(0, 1, 0)))
    follower = BRepBuilderAPI_MakeFace(plane, -10, 10, -.5, 1.5).Face()
    return wire, surface, follower, support_x


def write_step(model, path):
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    writer = STEPControl_Writer()
    if writer.Transfer(model, STEPControl_AsIs) != IFSelect_RetDone or writer.Write(str(path)) != IFSelect_RetDone:
        raise RuntimeError(f"STEP export failed for {path}")


def cam_motion(wire, angle, initial_support):
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.TopLoc import TopLoc_Location
    from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf

    axis = gp_Ax1(gp_Pnt(), gp_Dir(0, 0, 1))
    right, width = [], []
    for value in angle:
        rotation = gp_Trsf()
        rotation.SetRotation(axis, value)
        box = Bnd_Box()
        BRepBndLib.AddOptimal_s(wire.Moved(TopLoc_Location(rotation)), box, False, False)
        lower, _, _, upper, _, _ = box.Get()
        right.append(upper - initial_support)
        width.append(upper - lower)
    return np.array(right), np.array(width)


def cam(executable, output):
    a, b, c, d = 6., 5., 7., 2.5
    wire, surface, follower, support_x = cam_shapes(a, b, c, d)
    cam_file, follower_file = output / "cam.step", output / "follower-plane.step"
    write_step(surface, cam_file)
    write_step(follower, follower_file)

    dt = np.deg2rad(.05)
    angle = dt * np.arange(-2, 7203)
    motion, width = cam_motion(wire, angle, d)
    position = motion[2:-2]
    velocity = (motion[:-4] - 8*motion[1:-3] + 8*motion[3:-1] - motion[4:]) / (12*dt)
    acceleration = (-motion[:-4] + 16*motion[1:-3] - 30*position + 16*motion[3:-1] - motion[4:]) / (12*dt**2)
    t = angle[2:-2]
    exact_position, exact_velocity, exact_acceleration, switches = cam_reference(t, a, b, c)
    smooth = np.min(np.abs(t[:, None] - switches), axis=1) > 2*dt
    np.savetxt(output / "cam-motion.csv", np.column_stack((t, position, velocity, acceleration,
                exact_position, exact_velocity, exact_acceleration, smooth)), delimiter=",",
               header="time,position,velocity,acceleration,exact_position,exact_velocity,exact_acceleration,smooth", comments="")
    # Use the original publication's 1-degree grid for a separate library STEP query check.
    path = output / "cam-native.csv"
    tolerance = 1e-4
    subprocess.run([str(executable), "constant-breadth-cam", str(cam_file), str(follower_file), "360",
                    str(tolerance), str(path)], check=True)
    native = np.genfromtxt(path, delimiter=",", names=True)
    if len(native) != 361 or not np.allclose(native["angle"], np.linspace(0, 2*np.pi, 361), rtol=0, atol=2e-15):
        raise ValueError("Missing or unexpected STEP query angles")
    native_ref, _, _, _ = cam_reference(native["angle"], a, b, c)
    lo = support_x - d - native["distance_upper"]
    hi = support_x - d - native["distance_lower"]
    bracket_width = hi - lo
    if np.any(~np.isfinite(bracket_width)) or np.any(bracket_width < 0):
        raise ValueError("Nonfinite or inverted CAD position interval")
    if np.any((native_ref < lo) | (native_ref > hi)):
        raise ValueError("STEP distance query missed an analytical follower position")
    return {"a": a, "b": b, "c": c, "d": d, "omega": 1., "step": float(dt), "samples": len(t),
            "derivative_samples": int(np.count_nonzero(smooth)),
            "position_max": float(np.max(np.abs(position - exact_position))),
            "velocity_max": float(np.max(np.abs(velocity[smooth] - exact_velocity[smooth]))),
            "acceleration_max": float(np.max(np.abs(acceleration[smooth] - exact_acceleration[smooth]))),
            "breadth": a+b+2*d-c, "breadth_max_error": float(np.max(np.abs(width - (a+b+2*d-c)))),
            "native_samples": len(native), "native_tolerance": tolerance,
            "native_within_tolerance": int(np.count_nonzero(bracket_width <= tolerance)),
            "native_max_bracket_width": float(np.max(bracket_width)),
            "native_max_midpoint_error": float(np.max(np.abs((lo + hi)/2 - native_ref))) }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    executable = args.executable.resolve()
    result = {"oscillator": oscillator(executable, args.output)}
    (args.output / "oscillator.json").write_text(json.dumps(result["oscillator"], indent=2) + "\n", encoding="utf-8")
    result["cam"] = cam(executable, args.output)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
