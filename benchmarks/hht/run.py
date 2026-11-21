"""Run the recorded HHT experiment protocol on the local Release build."""

import argparse
import csv
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    label: str
    scene: str
    backend: str
    facets: int
    dt: float
    seconds: float
    repeats: int = 1
    variant: str = ""


CASES = (
    Case("drop-brep-4ms", "drop", "brep", 24, .004, .20),
    Case("drop-brep-2ms", "drop", "brep", 24, .002, .20),
    Case("drop-brep", "drop", "brep", 24, .001, .20, 5),
    Case("drop-brep-fine", "drop", "brep", 24, .0005, .20),
    Case("drop-mesh", "drop", "mesh", 24, .001, .20, 5),
    Case("cam-brep", "cam-array", "brep", 24, .005, 120, 3),
    Case("cam-mesh-12", "cam-array", "mesh", 12, .005, 120, 3),
    Case("cam-mesh-24", "cam-array", "mesh", 24, .005, 120, 3),
    Case("cam-mesh-48", "cam-array", "mesh", 48, .005, 120, 3),
    Case("cam-mesh-96", "cam-array", "mesh", 96, .005, 120, 3),
    Case("cam-brep-fine", "cam-array", "brep", 24, .0025, 120),
    Case("cam-brep-finest", "cam-array", "brep", 24, .00125, 120),
    Case("train-brep", "ball-train", "brep", 24, .005, 60, 3),
    Case("train-mesh-24", "ball-train", "mesh", 24, .005, 60, 3),
    Case("train-mesh-24-cold", "ball-train", "mesh", 24, .005, 60, 3, "cold"),
    Case("train-brep-fine", "ball-train", "brep", 24, .0025, 60),
    Case("train-brep-finest", "ball-train", "brep", 24, .00125, 60),
    Case("cam-stress", "cam-array", "brep", 24, .005, 300),
    Case("train-stress", "ball-train", "brep", 24, .005, 300),
    Case("cache-on", "cam-array", "brep", 24, .005, 18, 5),
    Case("cache-off", "cam-array", "brep", 24, .005, 18, 5, "uncached"),
    Case("warm-mesh", "ball-train", "mesh", 24, .005, 6, 3),
    Case("cold-mesh", "ball-train", "mesh", 24, .005, 6, 3, "cold"),
)


def call(exe, *args):
    subprocess.run([str(exe), *map(str, args)], check=True)


def valid(prefix, case):
    trace = prefix.with_suffix(".csv")
    if not trace.is_file() or not prefix.with_suffix(".json").is_file():
        return False
    stats = prefix.with_suffix(".stats.csv")
    if not stats.is_file():
        return False
    with stats.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return len(rows) == case.repeats and all(
        r["scene"] == case.scene and r["backend"] == case.backend and int(r["facets"]) == case.facets
        and float(r["duration"]) == case.seconds and abs(float(r["dt"])-case.dt) < 1e-12
        and bool(int(r["warm"])) == (case.variant != "cold")
        and bool(int(r["cache_guard"])) == (case.variant != "uncached")
        for r in rows
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trim-fixtures", type=Path, default=Path("build/occt/step_fixtures"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.output
    models = root / "models"
    raw = root / "runs"
    metrics = root / "metrics"
    for folder in (models, raw, metrics):
        folder.mkdir(parents=True, exist_ok=True)
    if not all((models/name).is_file() for name in (
        "floor.step", "follower.step", "left-stop.step", "right-stop.step", "ball.step", "cam.step"
    )):
        call(args.exe, "models", models)
    for name in ("bore.step", "pin.step"):
        source = args.trim_fixtures / name
        if not (models/name).exists():
            shutil.copy2(source, models/name)

    for action, row_count in (("oscillator",4), ("cam-queries",1812), ("trimmed-queries",125)):
        target = metrics / f"{action}.csv"
        if args.resume and target.is_file():
            with target.open(encoding="utf-8",newline="") as f:
                if sum(1 for _ in csv.DictReader(f)) == row_count:
                    continue
        print(f"Running {action}", flush=True)
        call(args.exe, action, *([] if action == "oscillator" else [models]), target)

    for case in CASES:
        prefix = raw / case.label
        if args.resume and valid(prefix, case):
            continue
        print(f"Running {case.label}", flush=True)
        call(args.exe, "run", case.scene, case.backend, case.facets, case.dt, case.seconds,
             models, prefix, case.repeats, *([case.variant] if case.variant else []))
        if not valid(prefix, case):
            raise RuntimeError(f"incomplete run {case.label}")
    (root/"protocol.json").write_text(json.dumps([asdict(case) for case in CASES], indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
