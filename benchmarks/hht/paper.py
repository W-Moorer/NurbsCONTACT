"""Build the LaTeX paper with result macros generated from the validated data."""

import argparse
import json
import os
import shutil
import subprocess
import sys
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


def engine_benchmark_macros(root, source):
    summary_file = root / "summary.json"
    if not summary_file.is_file():
        return []
    report = json.loads(summary_file.read_text(encoding="utf-8"))
    engines = report["engines"]
    figure = Path(os.path.relpath((root/"engine-comparison.pdf").resolve(),source))
    mappings = [
        ("nurbs", "HHT", "STEP/HHT"),
        ("mujoco", "implicit", "MuJoCo implicit"),
        ("mujoco", "RK4", "MuJoCo RK4"),
        ("drake", "runge_kutta3", "Drake RK3"),
    ]
    peak = max(row["maximum_force_law_residual_N"] for engine in engines.values()
               for row in engine["runs"])
    commands = [
        rf"\newcommand{{\EngineFigurePath}}{{\detokenize{{{figure.as_posix()}}}}}",
        rf"\newcommand{{\EngineMuJoCoVersion}}{{{engines['mujoco']['version']}}}",
        rf"\newcommand{{\EngineDrakeVersion}}{{{engines['drake']['version']}}}",
        rf"\newcommand{{\EngineForceResidual}}{{{scientific(peak)}}}",
    ]
    errors = []
    times = []
    for engine, method, name in mappings:
        by_case = {row["case"]:row for row in engines[engine]["runs"]
                   if row["integrator"]==method and row["step_s"]==.001}
        plane, spheres = by_case["plane"], by_case["spheres"]
        errors.append(
            f"  {name} & {1000*plane['position_rms_mm']:.3f} & {1000*spheres['position_rms_mm']:.3f} & "
            rf"{plane['force_rms_N']:.6f} & {spheres['force_rms_N']:.6f} \\"
        )
        times.append(rf"  {name} & {1e6*plane['median_integrate_s']:.1f} & {1e6*spheres['median_integrate_s']:.1f} \\")
    commands.extend([r"\newcommand{\EngineBenchmarkRows}{%", *errors, "}"])
    if report["timings_share_platform"]:
        commands.extend([r"\newcommand{\EngineTimingRows}{%", *times, "}"])
    return commands


def paper_output_path(results: Path) -> Path:
    project = Path(__file__).resolve().parents[2]
    if results.resolve() == project / "results" / "hht":
        return project / "docs" / "manuscript.pdf"
    return results / "paper" / "manuscript.pdf"


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
    parser.add_argument("--legacy", action="store_true", help="Build the original HHT benchmark paper")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    if args.results.resolve() == project / "results" / "hht" and not (args.macros_only or args.legacy):
        subprocess.run([sys.executable, str(project / "docs" / "AES" / "build.py")], check=True)
        return
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
    lines.extend(engine_benchmark_macros(args.results.parent / "engine-comparison", source))
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
    output = build / "legacy-manuscript.pdf" if args.results.resolve() == project / "results" / "hht" else paper_output_path(args.results)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(build / "manuscript.pdf", output)
    print(output)


if __name__ == "__main__":
    main()
