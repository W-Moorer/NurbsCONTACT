"""Compare discrete Drake SAP and the CPU STEP solver at the same update times."""

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from rigid import DURATION, LEGACY, SPEED, STANDOFF, drake_model, exact


def compare(trace, step):
    plant, sim, context, initial = drake_model("spheres", step, LEGACY)
    sim.Initialize()
    steps = round(DURATION / step)
    time = np.arange(steps + 1) * step
    q = np.empty((steps + 1, 2))
    v = np.empty_like(q)
    for i in range(steps + 1):
        if i:
            # Updates occur at ticks 0, h, ... . Sample after exactly i updates.
            sim.AdvanceTo((i - .5) * step)
        q[i] = plant.GetPositions(context) - initial
        v[i] = plant.GetVelocities(context)
    reference, _ = exact("spheres", time, LEGACY)
    free = time < STANDOFF / SPEED - step
    np.testing.assert_allclose(q[free], reference[free], rtol=0., atol=1e-12)
    np.testing.assert_allclose(v.sum(axis=1), -SPEED, rtol=0., atol=1e-12)
    native = np.genfromtxt(trace, names=True, delimiter=",")
    np.testing.assert_allclose(native["time_s"], time, rtol=0., atol=2e-14)
    with trace.with_suffix(".stats.csv").open() as stream:
        stats = next(csv.DictReader(stream))
    if stats["solver"] != "sap" or float(stats["event_tolerance_s"]) != 0 or int(stats["subdivisions"]):
        raise ValueError("expected fixed steps and SAP in the native trace")
    native_q = np.column_stack([native["q0"], native["q1"]])
    native_v = np.column_stack([native["v0"], native["v1"]])
    if not (np.isfinite(q).all() and np.isfinite(v).all() and v[-1, 0] < -.4):
        raise ValueError("incomplete Drake impact")
    return dict(step_s=step, updates=steps, final_drake_velocity_m_s=v[-1].tolist(),
                final_native_velocity_m_s=native_v[-1].tolist(),
                drake_contact_impulse_Ns=float(-v[-1, 0]),
                native_contact_impulse_Ns=float(step * native["p0_normal_N"][1:].sum()),
                drake_peak_force_N=float((-np.diff(v[:, 0]) / step).max()),
                native_peak_force_N=float(native["p0_normal_N"].max()),
                drake_max_displacement_error_m=float(np.abs(q-reference).max()),
                native_max_displacement_error_m=float(np.abs(native_q-reference).max()),
                max_interengine_displacement_difference_m=float(np.abs(q-native_q).max()),
                max_interengine_velocity_difference_m_s=float(np.abs(v-native_v).max()),
                native_trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--native-method",default="cad-occt")
    parser.add_argument("--native-geometry",default="STEP solid overlap")
    parser.add_argument("--run", nargs=2, action="append", required=True, metavar=("STEP_S", "NATIVE_TRACE"))
    args = parser.parse_args()
    result = dict(drake_version=version("drake"), duration_s=DURATION, sphere_radius_m=.041,
                  mass_per_body_kg=1., pair_stiffness_N_m=5000., relaxation_s=0., friction=0.,
                  geometry=["Drake Sphere", args.native_geometry], native_method=args.native_method, solver="sap",
                  runs=[compare(Path(trace), float(step)) for step, trace in args.run])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
