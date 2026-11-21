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
    return macros


def paper_output_path(results: Path) -> Path:
    project = Path(__file__).resolve().parents[2]
    if results.resolve() == project / "results" / "hht":
        return project / "docs" / "hht.pdf"
    return results / "paper" / "hht.pdf"


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
    lines.extend([r"\newcommand{\PerformanceRows}{%", *table_rows(summary), "}"])
    lines.extend([r"\newcommand{\ProductionRows}{%", *production_rows(summary), "}"])
    (source / "hht-results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.macros_only:
        return
    build = project / "build" / "hht-paper"
    build.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
         f"-outdir={build.as_posix()}", "hht.tex"],
        cwd=source, check=True,
    )
    output = paper_output_path(args.results)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(build / "hht.pdf", output)
    print(output)


if __name__ == "__main__":
    main()
