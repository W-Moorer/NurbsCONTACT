"""Validate gear traces and compare speed, transmission error, and tooth loads."""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from run import CASES, RUNS, completed, csv_rows


def track(root, label):
    return np.genfromtxt(root/"runs"/f"{label}.csv", delimiter=",", names=True)


def rms(value):
    return float(np.sqrt(np.mean(np.square(value))))


def difference(values, reference, stages, start=.02):
    t = values["time"]
    use = t >= start
    t = t[use]
    gear_results = []
    for stage in range(stages):
        te = f"te{stage}_um"
        force = f"normal{stage}_N"
        offset = values[te][use]-np.interp(t, reference["time"], reference[te])
        force_reference = np.interp(t, reference["time"], reference[force])
        force_delta = values[force][use]-force_reference
        gear_results.append({
            "transmission_rms_um": rms(offset),
            "transmission_max_um": float(np.max(np.abs(offset))),
            "transmission_mean_bias_um": float(np.mean(offset)),
            "normal_force_rms_N": rms(force_delta),
            "normal_force_rms_pct": 100*rms(force_delta)/float(np.mean(force_reference)),
        })
    return gear_results


def summarize(root):
    full = {}
    values = {}
    longest = max(spec.duration_s for spec in RUNS)
    for spec in RUNS:
        prefix = root/"runs"/spec.stem
        if not completed(prefix, spec):
            raise ValueError(f"Incomplete run {spec.stem}")
        trace = track(root,spec.stem)
        if not np.all(np.diff(trace["time"]) > 0) or abs(trace["time"][-1]-spec.duration_s) > 2e-10:
            raise ValueError(f"Invalid timeline {spec.stem}")
        fields = [field for field in trace.dtype.names if not field.startswith("station")]
        if any(not np.all(np.isfinite(trace[field])) for field in fields):
            raise ValueError(f"Invalid numeric values in {spec.stem}")
        repeat = csv_rows(prefix.with_suffix(".stats.csv"))
        times = [float(r["wall_s"]) for r in repeat]
        stages = 2 if spec.case == "reducer" else 1
        diagnostics = []
        for i in range(stages):
            native_gap = trace[f"native_gap{i}_mm"]
            available_gap = trace[f"mesh_gap{i}_mm"]
            if np.min(available_gap) < .002-1e-10:
                raise ValueError(f"Accepted geometry gap below floor in {spec.stem}")
            diagnostics.append({
                "native_minimum_sample_um": float(1000*np.min(native_gap)),
                "backend_minimum_sample_um": float(1000*np.min(available_gap)),
                "overlapping_native_samples": int(np.count_nonzero(native_gap <= 0)),
                "peak_normal_N": float(np.max(trace[f"normal{i}_N"])),
                "normal_rms_N": rms(trace[f"normal{i}_N"]),
                "peak_axial_N": float(np.max(np.abs(trace[f"axial{i}_N"]))),
                "mean_active_weight": float(np.mean(trace[f"active{i}"])),
            })
        full[spec.stem] = {
            "nominal_steps": round(spec.duration_s/spec.dt_s),
            "duration_s": spec.duration_s,
            "median_integration_s": statistics.median(times),
            "min_s": min(times), "max_s": max(times),
            "setup_s": float(repeat[0]["setup_s"]),
            "median_including_setup_s": statistics.median(times)+float(repeat[0]["setup_s"]),
            "repeats": len(repeat),
            "retries": int(repeat[0]["subdivisions"]),
            "solver_evaluations": int(repeat[0]["evaluations"]),
            "probe_calls": int(repeat[0]["probe_calls"]),
            "probe_misses": int(repeat[0]["probe_misses"]),
            "triangle_tests": int(repeat[0]["triangle_tests"]),
            "unique_flank_triangles": int(repeat[0]["triangles"]),
            "stage": diagnostics,
        }
        if spec.duration_s < longest:
            values[spec.stem] = trace
    comparison = {}
    for case in CASES:
        cad = f"{case}-cad"
        ref = values[cad+"-dt10"]
        stages = 2 if case == "reducer" else 1
        use = (cad, *[f"{case}-mesh-{name}" for name in ("coarse","medium","fine")])
        for label in use:
            full[label]["relative_to_10us_cad"] = difference(values[label], ref, stages)
            if label != cad:
                full[label]["relative_to_matched_cad"] = difference(values[label],values[cad],stages)
        finest = full[f"{case}-mesh-fine"]
        comparison[case] = {
            "fine_mesh_speed_ratio": finest["median_integration_s"]/full[cad]["median_integration_s"],
            "fine_mesh_ratio_with_setup": finest["median_including_setup_s"]/full[cad]["median_including_setup_s"],
            "cad_s": full[cad]["median_integration_s"], "fine_mesh_s": finest["median_integration_s"],
            "cad_nominal_step_rms_um": max(r["transmission_rms_um"] for r in full[cad]["relative_to_10us_cad"]),
            "mesh_geometry_rms_um": max(r["transmission_rms_um"] for r in finest["relative_to_matched_cad"]),
            "mesh_force_rms_pct": max(r["normal_force_rms_pct"] for r in finest["relative_to_matched_cad"]),
        }
    section = difference(values["helical-cad-section"],values["helical-cad-direct"],1,start=0)[0]
    section["speed_ratio"] = (
        full["helical-cad-direct"]["median_integration_s"]/full["helical-cad-section"]["median_integration_s"]
    )
    if section["transmission_max_um"] > 1e-5 or section["normal_force_rms_N"] > 1e-4:
        raise ValueError(f"Extracted CAD sections changed the reference result: {section}")
    axial = {
        case: difference(values[f"{case}-cad"],values[f"{case}-cad-axial15"], 2 if case=="reducer" else 1)
        for case in ("helical","reducer")
    }
    dense = {}
    for case in ("helical","reducer"):
        cad,mesh,reference=(f"{case}-{tag}" for tag in ("cad-axial31","mesh-axial31","cad-axial63"))
        n=2 if case=="reducer" else 1
        speed=full[mesh]["median_integration_s"]/full[cad]["median_integration_s"]
        dense[case]={
            "cad_s":full[cad]["median_integration_s"],"mesh_s":full[mesh]["median_integration_s"],
            "speed_ratio":speed,
            "matched_geometry":difference(values[mesh],values[cad],n),
            "cad_31_vs_63":difference(values[cad],values[reference],n),
            "fine_mesh_31_vs_cad_63":difference(values[mesh],values[reference],n),
        }
    return {"comparison": comparison, "section_ablation": section, "axial_sensitivity": axial,
            "dense_axial": dense, "runs": full}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path)
    args = parser.parse_args()
    data = summarize(args.results)
    output = args.results/"summary.json"
    output.write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(data["comparison"],indent=2))
    print("CAD section ablation:",data["section_ablation"])


if __name__ == "__main__":
    main()
