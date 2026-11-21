"""Build the LaTeX paper with result macros generated from the validated data."""

import argparse
import json
import os
import shutil
import subprocess
from statistics import median
from pathlib import Path


def scientific(value):
    if value == 0:
        return "0"
    coefficient, exponent = f"{value:.2e}".split("e")
    return rf"\ensuremath{{{coefficient}\times10^{{{int(exponent)}}}}}"


def macro_values(summary):
    runs = summary["runs"]
    speedups = summary["speedups"]
    cad = runs["train-brep"]
    mesh = runs[speedups["train_mesh_setting"]]
    cam_queries = {r["parameter"]: r for r in summary["cam_queries"] if r["backend"] == "mesh"}
    trim = summary["trimmed_queries"]
    fine_trim = min((r for r in trim if r["backend"] == "mesh"), key=lambda r: r["parameter"])
    cad_trim = next(r for r in trim if r["backend"] == "brep")
    temporal = summary["temporal_refinement"]
    train_refinement = temporal["ball-train"]
    cam_refinement = temporal["cam-array"]
    production_cad = [row for row in summary["production_queries"] if row["backend"] == "brep"]
    oracle_poses = sum(row["poses"] for row in production_cad)
    oracle_agreement = sum(row["poses"]-row["oracle_outside_interval"] for row in production_cad)
    return {
        "TrainMeshTime": f"{mesh['median_s']:.2f}",
        "TrainCadTime": f"{cad['median_s']:.3f}",
        "TrainGain": f"{speedups['train_speedup']:.1f}",
        "CamFineGain": f"{speedups['cam_96_speedup']:.1f}",
        "OscillatorOrders": ", ".join(f"{order:.3f}" for order in summary["oscillator_order"]),
        "DropError": f"{runs['drop-brep']['rms_to_exact_mm']:.4f}",
        "DropFineError": f"{runs['drop-brep-fine']['rms_to_exact_mm']:.4f}",
        "CamCoarseGapError": f"{cam_queries[12]['max_error_mm']:.3f}",
        "CamFineGapError": f"{cam_queries[96]['max_error_mm']:.4f}",
        "TrainMatchError": scientific(mesh["matched_reference"]["max_translation_mm"]),
        "FineCamMeshError": f"{runs['cam-mesh-96']['fine_reference']['rms_translation_mm']:.4f}",
        "CamCadTimeError": f"{runs['cam-brep']['fine_reference']['rms_translation_mm']:.4f}",
        "GuardOff": f"{runs['cache-off']['guard_endpoints']:,}",
        "GuardOn": f"{runs['cache-on']['guard_endpoints']:,}",
        "GuardGain": f"{speedups['guard_cache_speedup']:.2f}",
        "StressCamGuard": f"{runs['cam-stress']['min_guard_mm']:.3f}",
        "StressTrainGuard": f"{runs['train-stress']['min_guard_mm']:.3f}",
        "TrimFinestError": scientific(fine_trim["max_error_mm"]),
        "TrimFinestMesh": f"{fine_trim['median_us']/1000:.3f}",
        "TrimCadTime": f"{cad_trim['median_us']/1000:.3f}",
        "CamTemporalFine": f"{cam_refinement['2.5ms']['rms_translation_mm']:.4f}",
        "TrainEarlyCoarse": f"{train_refinement['train-brep_first_5s']['rms_translation_mm']:.3f}",
        "TrainEarlyFine": f"{train_refinement['train-brep-fine_first_5s']['rms_translation_mm']:.3f}",
        "TrainLongCoarse": f"{train_refinement['5ms']['rms_translation_mm']:.3f}",
        "TrainLongFine": f"{train_refinement['2.5ms']['rms_translation_mm']:.3f}",
        "ProductionOraclePoses": str(oracle_poses),
        "ProductionOracleAgreement": str(oracle_agreement),
    }


def table_rows(summary):
    configurations = [
        ("Cam array", "B-rep", "cam-brep"),
        *[("", f"Mesh ({facets})", f"cam-mesh-{facets}") for facets in (12, 24, 48, 96)],
        ("Ball train", "B-rep", "train-brep"),
        ("", "Mesh (24)", summary["speedups"]["train_mesh_setting"]),
    ]
    rows = []
    for scene, method, label in configurations:
        run = summary["runs"][label]
        error = run["fine_reference"]["rms_translation_mm"]
        rows.append(
            f"  {scene} & {method} & {run['median_s']:.3f} & {error:.4f} & "
            rf"{run['min_guard_mm']:.3f} & {run['substeps']} \\"
        )
    return rows


def production_rows(summary):
    groups = {}
    for row in summary["production_queries"]:
        key = row["scene"], row["backend"], row["deflection_mm"]
        groups[key] = row
    labels = {
        "wheel-washer-stack": "Wheel washer stack",
        "wheel-eccentric-bolt": "Wheel screw 1",
        "wheel-eccentric-bolt2": "Wheel screw 2",
        "lidar-nut-preseat": "Lidar pre-seat",
    }
    rows = []
    for key, name in labels.items():
        cad = groups[(key, "brep", 0)]
        mesh = groups[(key, "mesh", 0.0125)]
        rows.append(
            f"  {name} & {cad['resolved']}/{cad['poses']} & {cad['median_us']/1000:.3f} & "
            rf"{mesh['median_us']/1000:.3f} & {scientific(mesh['max_error_mm'])} \\"
        )
    return rows


def external_validation_macros(root):
    slider_file = root / "clearance" / "summary.json"
    if not slider_file.exists():
        return {}
    slider = json.loads(slider_file.read_text(encoding="utf-8"))
    comparison = {run["backend"]: run for run in slider["runs"]
                  if run["rpm"] == 200 and run["clearance_mm"] == .25}
    brep = comparison["brep"]
    half = comparison["brep-half-step"]
    mesh = comparison["mesh5um"]
    matched = {}
    for run in slider["runs"]:
        if "wall_s" in run:
            matched.setdefault((run["rpm"], run["clearance_mm"]), {})[run["backend"]] = run["wall_s"]
    time_ratios = [group["brep"]/group["mesh5um"] for group in matched.values()
                   if {"brep", "mesh5um"} <= group.keys()]
    macros = {
        "ExternalSliderPosition": f"{brep['position_rmse_mm']:.2f}",
        "ExternalSliderRmse": f"{brep['uniform_phase_acceleration_rmse_m_s2']:.2f}",
        "ExternalSliderIdealRmse": f"{brep['ideal_joint_acceleration_rmse_m_s2']:.2f}",
        "ExternalSliderHalfRmse": f"{half['uniform_phase_acceleration_rmse_m_s2']:.2f}",
        "ExternalSliderMeasuredPhase": f"{brep['measured_first_negative_peak_angle_deg']:.1f}",
        "ExternalSliderPredictedPhase": f"{brep['simulated_first_negative_peak_angle_deg']:.1f}",
        "ExternalSliderHalfPhase": f"{half['simulated_first_negative_peak_angle_deg']:.1f}",
        "ExternalSliderBrepGeometry": scientific(brep["max_geometric_gap_error_um"]),
        "ExternalSliderMeshGeometry": f"{mesh['max_geometric_gap_error_um']:.2f}",
        "ExternalSliderSpeedRatio": f"{mesh['wall_s']/brep['wall_s']:.2f}",
        "ExternalSliderMedianTimeRatio": f"{median(time_ratios):.2f}",
        "ExternalSliderTimingCases": str(len(time_ratios)),
    }
    if "extended_run" in slider:
        extended = slider["extended_run"]
        macros["ExternalSliderExtendedSteps"] = f"{int(extended['accepted']):,}"
        macros["ExternalSliderExtendedPhase"] = f"{extended['simulated_first_negative_peak_angle_deg']:.1f}"
    revised_file = slider_file.parent / "revised" / "summary.json"
    if revised_file.is_file():
        revision = json.loads(revised_file.read_text(encoding="utf-8"))
        comparison = {(run["rpm"], run["clearance_mm"], run["variant"]): run for run in revision["runs"]}
        mounted = comparison[200, .25, "mounted"]
        candidate = comparison[200, .25, "revised"]
        transferred = comparison[150, .25, "revised"]
        macros.update({
            "ExternalDampedBefore": f"{mounted['raw_sample_rmse_m_s2']:.2f}",
            "ExternalDampedAfter": f"{candidate['raw_sample_rmse_m_s2']:.2f}",
            "ExternalDampedQuietBefore": f"{mounted['quiet_window_rmse_m_s2']:.2f}",
            "ExternalDampedQuietAfter": f"{candidate['quiet_window_rmse_m_s2']:.2f}",
            "ExternalDampedPeakBefore": f"{mounted['quiet_window_max_abs_acceleration_m_s2']:.1f}",
            "ExternalDampedPeakAfter": f"{candidate['quiet_window_max_abs_acceleration_m_s2']:.1f}",
            "ExternalDampedDepthBefore": f"{-mounted['minimum_gap_um']:.2f}",
            "ExternalDampedDepthAfter": f"{-candidate['minimum_gap_um']:.2f}",
            "ExternalDampedHalfStep": f"{revision['half_step_waveform_delta_rmse_m_s2']:.4f}",
            "ExternalDampedIndependent": f"{transferred['raw_sample_rmse_m_s2']:.2f}",
        })
    flex_file = slider_file.parent / "flexibility" / "summary.json"
    if flex_file.is_file():
        report = json.loads(flex_file.read_text(encoding="utf-8"))
        by_case = {row["case"]:row for row in report["runs"]}
        control, flexible = by_case["rigid"], by_case["flex"]
        macros.update({
            "FlexQuietControl": f"{control['quiet_window_rmse_m_s2']:.2f}",
            "FlexQuietModel": f"{flexible['quiet_window_rmse_m_s2']:.2f}",
            "FlexPeakControl": f"{control['quiet_window_max_abs_acceleration_m_s2']:.2f}",
            "FlexPeakModel": f"{flexible['quiet_window_max_abs_acceleration_m_s2']:.2f}",
            "FlexDepth": f"{-flexible['minimum_gap_um']:.2f}",
            "FlexObserved": f"{flexible['measured_peak_m_s2']:.1f}",
            "FlexPredicted": f"{flexible['impact_peaks'][0]['at_measured_phase_m_s2']:.1f}",
            "FlexRefinement": f"{report['half_step_common_grid_delta_m_s2']:.4f}",
        })
    spatial_file = slider_file.parent / "spatial" / "summary.json"
    if spatial_file.is_file():
        report = json.loads(spatial_file.read_text(encoding="utf-8"))
        spatial = report["refinements"]
        a10 = spatial["10us"]["revolutions"][0]["sensor_at_measured_phase_m_s2"]
        a1 = spatial["1us"]["revolutions"][0]["sensor_at_measured_phase_m_s2"]
        audit = json.loads(spatial_file.with_name("reference-audit.json").read_text(encoding="utf-8"))["panels"]
        mounts = {r["case"]:r for r in report["measurement_mount_screening"]}
        near_phase = report["measurement_mount_refinement"]["revolutions"][0]
        larger_peak = mounts["mount-70hz"]["revolutions"][0]
        macros.update({
            "SpatialMeasured": f"{spatial['10us']['measured_peak_m_s2']:.2f}",
            "SpatialPredicted": f"{a10:.2f}",
            "SpatialQuiet": f"{spatial['10us']['quiet_rmse_m_s2']:.2f}",
            "SpatialOneMicrosecondDelta": f"{abs(a10-a1):.4f}",
            "SpatialFigFiveBound": f"{audit['5d-clearance-experiment']['compatibility'][1]['necessary_peak_displacement_mm']:.3f}",
            "SpatialFigSixBound": f"{audit['6b-clearance-experiment']['compatibility'][1]['necessary_peak_displacement_mm']:.3f}",
            "SpatialPlanarBound": f"{audit['5d-clearance-experiment']['compatibility'][1]['rigid_planar_clearance_bound_mm']:.3f}",
            "SpatialPositionBound": f"{audit['5d-clearance-experiment']['compatibility'][1]['necessary_difference_from_published_position_mm']:.3f}",
            "SpatialMountPhase": f"{near_phase['window_min_phase_deg']:.2f}",
            "SpatialMountPeak": f"{near_phase['window_min_m_s2']:.2f}",
            "SpatialLargeMountPeak": f"{larger_peak['window_min_m_s2']:.2f}",
            "SpatialLargeMountQuiet": f"{mounts['mount-70hz']['quiet_rmse_m_s2']:.2f}",
            "SpatialTransferLow": f"{report['held_out']['150rpm-0.25mm']['raw_rmse_m_s2']:.2f}",
            "SpatialTransferBig": f"{report['held_out']['200rpm-0.50mm']['raw_rmse_m_s2']:.2f}",
        })
    impact_file = slider_file.parent / "impact-laws" / "summary.json"
    if impact_file.is_file():
        report = json.loads(impact_file.read_text(encoding="utf-8"))
        measured = report["published_curves"]["figure6b"]
        original = report["published_curves"]["figure11d"]["first_half_minimum"]
        rigid, mount = report["refinement"]
        event = mount["revolutions"][0]
        screening = {row["case"]: row for row in report["screening"]}
        spatial_ln, spatial_exact = screening["spatial-ln"], screening["spatial-exact"]
        low, large = report["held_out"]
        macros.update({
            "IncidentLNBounce": f"{report['isolated_rebound'][0]['measured_rebound_ratio']:.3f}",
            "IncidentRigidPhase": f"{rigid['first_half_minimum_phase_deg']:.2f}",
            "IncidentRigidPeak": f"{rigid['first_half_minimum_m_s2']:.2f}",
            "IncidentRigidQuietCoarse": f"{screening['rigid-ln']['quiet_rmse_m_s2']:.2f}",
            "IncidentRigidQuietFine": f"{rigid['quiet_rmse_m_s2']:.2f}",
            "IncidentOriginalPhase": f"{original['phase_deg']:.2f}",
            "IncidentOriginalPeak": f"{original['acceleration_m_s2']:.2f}",
            "IncidentMountPhase": f"{event['window_min_phase_deg']:.2f}",
            "IncidentMountPeak": f"{event['window_min_m_s2']:.2f}",
            "IncidentMountLoad": f"{event['min_load_N']:.2f}",
            "IncidentMeasuredWidth": f"{measured['impact_half_prominence_width_deg']:.2f}",
            "IncidentMountWidth": f"{mount['trough_widths_deg'][0]:.2f}",
            "IncidentSpatialQuiet": f"{spatial_ln['quiet_rmse_m_s2']:.2f}",
            "IncidentExactQuiet": f"{spatial_exact['quiet_rmse_m_s2']:.2f}",
            "IncidentTransferLow": f"{low['raw_rmse_m_s2']:.2f}",
            "IncidentTransferLarge": f"{large['raw_rmse_m_s2']:.2f}",
        })
    bearing_file = slider_file.parent / "auxiliary-bearings" / "summary.json"
    if bearing_file.is_file():
        report = json.loads(bearing_file.read_text(encoding="utf-8"))
        coarse = {row["case"]: row for row in report["screening"]}
        fine = report["refinement"]
        event = fine["revolutions"][0]
        low, large = report["held_out"]
        open_fraction = max(row["all_sections_open_fraction"] for row in fine["revolutions"])
        macros.update({
            "BearingFinePhase": f"{event['window_min_phase_deg']:.2f}",
            "BearingFinePeak": f"{event['window_min_m_s2']:.2f}",
            "BearingFineWidth": f"{fine['trough_widths_deg'][0]:.2f}" if fine["trough_widths_deg"][0] else r"\textemdash",
            "BearingFineLoad": f"{event['min_load_N']:.2f}",
            "BearingFineOpen": f"{100*open_fraction:.2f}",
            "BearingFineQuiet": f"{fine['quiet_rmse_m_s2']:.2f}",
            "BearingQuietDelta": f"{abs(coarse['both']['quiet_rmse_m_s2']-fine['quiet_rmse_m_s2']):.5f}",
            "BearingRigidBound": f"{fine['rigid_planar_bound_25_to_85_deg_mm']:.3f}",
            "BearingTransferLow": f"{low['raw_rmse_m_s2']:.2f}",
            "BearingTransferLarge": f"{large['raw_rmse_m_s2']:.2f}",
        })
    return macros


def paper_output_path(results: Path) -> Path:
    project = Path(__file__).resolve().parents[2]
    if results.resolve() == project / "results" / "hht":
        return project / "docs" / "manuscript.pdf"
    return results / "paper" / "manuscript.pdf"


def nasa_selected_macros(root):
    report_file = root / "summary.json"
    if not report_file.is_file() or not (root / "figures" / "nasa-single-tooth-selected.pdf").is_file():
        return []
    report = json.loads(report_file.read_text(encoding="utf-8"))
    comparisons = report["comparisons"]
    rows = [r"\newcommand{\NasaSelectedRows}{%"]
    for panel in "acd":
        case = comparisons[f"nasa-{panel}"]
        cad = case["brep"]["error"]
        mesh = case["mesh"]["error"]
        role = "Calibration" if panel == "a" else "Held out"
        rows.append(f"7({panel}) & {case['rpm']:.0f} & {case['input_torque_Nm']:.2f} & "
                    f"{cad['rms']:.1f} & {mesh['rms']:.1f} & {role} " + r"\\")
    rows.append("}")
    return rows


def clearance_followup_macros(root):
    report_file = root / "summary.json"
    if not report_file.is_file():
        return []
    report = json.loads(report_file.read_text(encoding="utf-8"))

    def value(x, digits=3):
        if x is None:
            return "--"
        parts = f"{x:.{digits}g}".split("e")
        if len(parts) == 1:
            return parts[0]
        mantissa, exponent = parts
        return rf"\({mantissa}\times10^{{{int(exponent)}}}\)"

    def end(row):
        return " & ".join(row) + r" \\"

    def table(name, rows):
        return [rf"\newcommand{{\{name}}}{{%", *(end(row) for row in rows), "}"]

    rotation_labels = {
        "planar-tilt": "Planar tilt", "curved-tilt": "Curved tilt",
        "interior-entry": "Interior entry", "rotating-near-miss": "Rotating near miss",
    }
    rotation = []
    for case, label in rotation_labels.items():
        baseline = report["rotation"][case]["chen-obb-chord"]
        native = report["rotation"][case]["native"]
        ref = native["reference_lower"]
        rotation.append((label, f"{float(ref):.6f}" if ref else "Clear",
            f"{baseline['median_us']/1000:.3f}", f"{native['median_us']/1000:.3f}",
            value(baseline["maximum_width"]), value(native["maximum_width"])))

    hole_labels = {
        "throat-pass": "Counterbore passage", "shoulder-entry": "Counterbore shoulder",
        "mouth-entry": "Counterbore mouth", "voron-spacer": "Voron M5 spacer",
        "voron-washer": "Voron M3 washer", "voron-bearing": "Voron F695 bearing",
    }
    holes = []
    for case, label in hole_labels.items():
        group = "assembly" if case.startswith("voron-") else "counterbore"
        row = report[group][case]
        source = row["axial"]
        ref = source["reference_lower"]
        timings = []
        for method in ("swept-box", "directional", "axial"):
            item = row[method]
            tag = r"${}^{*}$" if item["status"] == "unresolved" or item["runs_above_width_target"] else ""
            timings.append(f"{item['median_us']/1000:.3f}" + tag)
        reference = "Unresolved" if source["reference_status"] == "unresolved" else (
            f"{float(ref):.6f}" if ref else "Clear")
        holes.append((label, reference, *timings,
                      f"{row['axial']['median_nodes']:.0f}"))

    audit_labels = {
        "trim-edge-entry": "Trim-edge entry", "narrow-edge-entry": "Narrow entry",
        "exact-edge-graze": "Exact grazing", "near-edge-miss": "Near miss",
    }
    audit = []
    for case, label in audit_labels.items():
        row = report["audit"][case]["axial"]
        ref = row["reference_lower"]
        audit.append((label, f"{float(ref):.9f}" if ref else "Clear", row["status"],
                      f"{row['minimum_safe_prefix']:.9f}", value(row["maximum_width"]),
                      value(row["maximum_witness_distance"], 10)))
    return (table("RotationFollowupRows", rotation) + table("HoleFollowupRows", holes) +
            table("WitnessFollowupRows", audit))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--macros-only", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    source = project / "manuscript"
    summary = json.loads((args.results / "metrics" / "summary.json").read_text(encoding="utf-8"))
    figures = (args.results / "figures").resolve()
    try:
        figures = Path(os.path.relpath(figures, source))
    except ValueError:
        # Windows paths on different drives have no relative representation.
        pass
    lines = [
        "% Generated from the validated summary by benchmarks/hht/paper.py.",
        rf"\graphicspath{{{{\detokenize{{{figures.as_posix()}/}}}}}}",
    ]
    for name, value in macro_values(summary).items():
        lines.append(rf"\newcommand{{\{name}}}{{{value}}}")
    for name, value in external_validation_macros(args.results.parent / "contact_validation").items():
        lines.append(rf"\newcommand{{\{name}}}{{{value}}}")
    lines.extend(nasa_selected_macros(project / "results" / "gears" / "experimental"))
    lines.extend(clearance_followup_macros(project / "benchmarks" / "clearance-followup"))
    lines.extend([r"\newcommand{\PerformanceRows}{%", *table_rows(summary), "}"])
    lines.extend([r"\newcommand{\ProductionRows}{%", *production_rows(summary), "}"])
    (source / "manuscript-results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.macros_only:
        return
    build = project / "build" / "manuscript-paper"
    build.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
         f"-outdir={build.as_posix()}", "manuscript.tex"],
        cwd=source, check=True,
    )
    output = paper_output_path(args.results)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(build / "manuscript.pdf", output)
    print(output)


if __name__ == "__main__":
    main()
