"""Compare capped STEP feature searches to OCCT shell distances at two archived poses."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/aes/general-contact/spur-static-cad-cost.json"


def run(exe, local=False):
    reference = json.loads(SOURCE.read_text())
    assets = [ROOT / "results/gears/models" / item["asset"] for item in reference["step_assets"]]
    for path, item in zip(assets, reference["step_assets"]):
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["step_sha256"]:
            raise ValueError("STEP model differs from the recorded benchmark")
    lines = []
    for sample in reference["samples"]:
        values = [x for pose in sample["poses"] for x in (*pose["origin_m"], *pose["axis"], pose["angle_rad"])]
        lines.append(" ".join(format(x, ".17g") for x in values))
    child = subprocess.run([exe, *assets, "1", *(["--local"] if local else [])], input="\n".join(lines)+"\n", text=True,
                           capture_output=True, timeout=1800, check=True)
    measurements = list(csv.DictReader(io.StringIO(child.stdout)))
    if len(measurements) != len(lines):
        raise ValueError("incomplete static comparison")
    for sample, row in zip(reference["samples"], measurements):
        for key in list(row):
            row[key] = int(row[key]) if key in ("extrema", "boxes", "classifications", "clipped_bodies") else float(row[key])
        if abs(row["distance_mm"] - row["shell_distance_mm"]) > 2e-6:
            raise ValueError("indexed distance differs from complete OCCT boundary search")
        if (row["distance_mm"] <= 1e-6) != sample["solid_intersection"]["has_volume"]:
            raise ValueError("contact status differs from original Boolean at this pose")
        row["time_s"] = sample["time_s"]
    return dict(source=SOURCE.name, source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                distance_cap_mm=1., contact_tolerance_mm=1e-6, local=local, samples=measurements)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    result = run(args.exe.resolve(), args.local)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
