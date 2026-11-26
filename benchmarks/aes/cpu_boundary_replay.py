"""Replay recorded CPU reducer poses; sum elastic forces and moments about the driven shaft."""

import argparse
import json
import subprocess
from pathlib import Path

from cpu_backend_run import digest
from cpu_cad_pressure_report import wrench


def replay(config, exe, assets, method, reverse=False, force_n=10.0, moment_nm=0.5):
    reference = method.startswith("step")
    mesh = method == "mesh-obb"
    suffix = ".msh" if mesh else ".step" if reference else ".ncpart"
    rows = []
    fingerprints = {}
    for stage in config["stages"]:
        files = [assets / (name + suffix) for name in stage["parts"]]
        for path in files:
            fingerprints[path.name] = digest(path)
        limit = stage["limit_m"]
        stiffness = stage["stiffness_N_m"]
        if mesh:
            args = [
                "--method",
                "mesh-obb",
                "--regions",
                "contour-ray",
                "--axis-angle",
                "--max-depth-m",
                str(limit),
                "--keep-unresolved",
                *files,
            ]
        elif reference:
            setting = "--pressure-patches" if method == "step" else "--refined-pressure-patches"
            args = [*files, setting, str(2 * limit)]
        else:
            args = [*files, str(limit)]
            if method != "cad-compiled":
                args += ["--trim-boundary"]
            if method in ("adaptive", "fixed96"):
                args += [method, str(stiffness), "1", str(force_n), str(moment_nm)]
        records = list(zip(stage["steps"], stage["poses"], strict=True))
        if reverse:
            records.reverse()
        payload = "".join(" ".join(format(x, ".17g") for x in pose) + "\n" for _, pose in records)
        proc = subprocess.run([exe, *args], input=payload, text=True, capture_output=True, timeout=1800)
        if proc.returncode:
            raise RuntimeError(proc.stderr)
        output = [json.loads(line) for line in proc.stdout.splitlines()]
        for (step, pose), result in zip(records, output, strict=True):
            regions = result["patches" if reference else "regions"]
            resolved = result.get("resolved", True)
            load = wrench(regions, pose[7:10], pose[10:13], stiffness, limit, not mesh) if resolved else None
            rows.append(
                dict(
                    stage=stage["stage"],
                    step=step,
                    resolved=resolved,
                    regions=len(regions) if resolved else None,
                    wrench=load.tolist() if load is not None else None,
                    rules=result.get("rules"),
                    converged=result.get("converged"),
                )
            )
    return dict(
        method=method,
        reverse_sequence=reverse,
        executable_sha256=digest(exe),
        assets=fingerprints,
        quadrature_tolerances=[force_n, moment_nm] if method == "adaptive" else None,
        rows=rows,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("poses", type=Path)
    p.add_argument("exe", type=Path)
    p.add_argument("assets", type=Path)
    p.add_argument(
        "method",
        choices=("mesh-obb", "cad-compiled", "cad-boundary", "adaptive", "fixed96", "step", "step-refined"),
    )
    p.add_argument("out", type=Path)
    p.add_argument("--reverse", action="store_true", help="Process the same poses in reverse order")
    p.add_argument("--force-tol-n", type=float, default=10.0)
    p.add_argument("--moment-tol-nm", type=float, default=0.5)
    a = p.parse_args()
    data = replay(
        json.loads(a.poses.read_text()),
        a.exe.resolve(),
        a.assets,
        a.method,
        a.reverse,
        a.force_tol_n,
        a.moment_tol_nm,
    )
    a.out.write_text(json.dumps(data, indent=1, allow_nan=False) + "\n")
