"""Compare saved closed-loop histories at their shared output times."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def history_metrics(data, h, radii):
    loaded = np.flatnonzero(data[:, 4] > 0)
    return dict(
        first_loaded_s=float(loaded[0] * h) if len(loaded) else None,
        peak_N=float(data[:, 4].max()),
        normal_impulse_Ns=float(np.trapezoid(data[:, 4], dx=h)),
        peak_abs_transmission_error_um=float(
            np.max(np.abs(1e3 * (radii[1] * data[:, 1] - radii[0] * data[:, 0])))
        ),
    )


def delta(a, b, radii):
    d = a - b
    te = 1e3 * (radii[1] * d[:, 1] - radii[0] * d[:, 0])
    return dict(
        states=len(d),
        max_shaft_angle_rad=float(np.max(np.abs(d[:, :2]))),
        max_shaft_speed_rad_s=float(np.max(np.abs(d[:, 2:4]))),
        max_abs_normal_force_N=float(np.max(np.abs(d[:, 4]))),
        rms_normal_force_N=float(np.sqrt(np.mean(d[:, 4] ** 2))),
        rms_transmission_error_um=float(np.sqrt(np.mean(te**2))),
        differing_face_pair_count_steps=int(np.count_nonzero(d[:, 5])),
    )


def compare(folder, cad):
    with Path("results/gears/models/spur.csv").open() as f:
        radii = [float(row["base_radius_mm"]) for row in csv.DictReader(f)]

    def read(prefix, step):
        return np.load(folder / f"spur-{prefix}-{step}-3070.npy")

    def comparison(prefix, control, step):
        a, b = read(prefix, step), read(control, step)
        if len(a) != len(b) or len(a) != round(2e6 / step) + 1:
            raise ValueError("Expected two complete gear histories over two seconds")
        return delta(a, b, radii)

    result = {"mesh_grid_vs_bvh": {s: comparison("grid", "bvh", s) for s in (100, 125)}}
    for step in (100, 125):
        if (folder / f"spur-cad-grid-{step}-3070.npy").exists():
            result.setdefault("cad_probes_grid_vs_bvh", {})[step] = comparison("cad-grid", "cad-bvh", step)
            result.setdefault("cad_probes_vs_mesh_probes", {})[step] = comparison("cad-grid", "grid", step)
    h_ref = 5
    ref = read("grid", h_ref)
    result["first_20ms"] = {}
    for s in (5, 10, 20, 50, 100, 125):
        tr = read("grid", s)[: round(20000 / s) + 1]
        stride = int(np.lcm(s, h_ref))
        result["first_20ms"][s] = dict(
            **history_metrics(tr, s * 1e-6, radii),
            vs_5us_common_outputs=delta(tr[:: stride // s], ref[:: stride // h_ref], radii),
        )
    for prefix in ("impact", "cad-impact"):
        if (folder / f"spur-{prefix}-grid-125-3070.npy").exists():
            result[prefix + "_grid_vs_bvh"] = comparison(prefix + "-grid", prefix + "-bvh", 125)
            tr = read(prefix + "-grid", 125)[:161]
            fine = ref if prefix == "impact" else read("cad-grid", 5)
            result[prefix + "_common_outputs_first_20ms"] = delta(tr, fine[::25], radii)
            if prefix == "cad-impact":
                result["cad_5us_first_20ms"] = history_metrics(fine, 5e-6, radii)
                result["cad_fixed_125us_vs_5us_first_20ms"] = delta(
                    read("cad-grid", 125)[:161], fine[::25], radii
                )
    if cad:
        with cad.open() as f:
            other = list(csv.DictReader(f))
        source = np.array(
            [[float(r[k]) for k in ("q0", "q1", "v0", "v1", "normal0_N", "active0")] for r in other]
        )
        warm = read("grid", 100)[:, :6]
        if source.shape != warm.shape or not np.allclose(
            [float(r["time"]) for r in other], np.arange(len(warm)) * 0.0001
        ):
            raise ValueError("Native trace times or number of states differ")
        result["specialized_cad_separate_force_model"] = dict(
            **history_metrics(source, 1e-4, radii), difference_from_mesh_grid=delta(warm, source, radii)
        )
    sphere = np.load(folder / "spheres-grid-100-3070.npy")
    enter = 0.03 / 0.45
    exit_time = enter + np.pi / 100
    loaded = np.flatnonzero(sphere[:, 4] > 0)
    result["two_spheres"] = dict(
        analytic_entry_s=enter,
        analytic_release_s=exit_time,
        analytic_peak_N=22.5,
        first_loaded_s=float(loaded[0] * 0.0001),
        first_released_sample_s=float((loaded[-1] + 1) * 0.0001),
        measured_peak_N=float(sphere[:, 4].max()),
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("traces", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--native-100us", type=Path)
    args = p.parse_args()
    args.output.write_text(json.dumps(compare(args.traces, args.native_100us), indent=2) + "\n")
