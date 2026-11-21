"""Run the STEP gear contact matrix with the project HHT solver."""

import argparse
import csv
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


CASES = ("spur", "helical", "rig28", "reducer")


@dataclass(frozen=True)
class Run:
    case: str
    label: str
    deflection_mm: float = 0
    dt_s: float = .00002
    duration_s: float = .60
    repeats: int = 3
    stations: int = 7

    @property
    def stem(self):
        return f"{self.case}-{self.label}"


RUNS = [
    run
    for case in CASES
    for run in (
        Run(case, "mesh-medium", .00625),
        Run(case, "cad"),
        Run(case, "mesh-fine", .0015625),
        Run(case, "mesh-coarse", .025),
        Run(case, "cad-dt10", dt_s=.00001, repeats=1),
    )
] + [
    Run("helical", "cad-section", duration_s=.12),
    Run("helical", "cad-direct", -1, duration_s=.12),
    Run("helical", "cad-axial15", repeats=1, stations=15),
    Run("reducer", "cad-axial15", repeats=1, stations=15),
    Run("reducer", "cad-duty", duration_s=3, repeats=1),
    Run("reducer", "mesh-duty", .0015625, duration_s=3, repeats=1),
] + [
    run
    for case in ("helical","reducer")
    for run in (
        Run(case,"cad-axial31",duration_s=.36,stations=31),
        Run(case,"mesh-axial31",.0015625,duration_s=.36,stations=31),
        Run(case,"cad-axial63",duration_s=.36,repeats=1,stations=63),
    )
]


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def completed(prefix, spec):
    trajectory = prefix.with_suffix(".csv")
    timing = prefix.with_suffix(".stats.csv")
    if not trajectory.is_file() or not timing.is_file():
        return False
    rows = csv_rows(timing)
    steps = round(spec.duration_s/spec.dt_s)
    if len(rows) != spec.repeats or not all(
        int(r["steps"]) == steps
        and int(r["accepted"]) >= steps
        and float(r["deflection_mm"]) == spec.deflection_mm
        and float(r["dt_s"]) == spec.dt_s
        for r in rows
    ):
        return False
    with trajectory.open(encoding="utf-8") as file:
        return sum(1 for _ in file) == steps+2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    executable = args.executable.resolve()
    models, raw = args.output/"models", args.output/"runs"
    for folder in (models, raw):
        folder.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        layout = models/f"{case}.csv"
        rows = csv_rows(layout) if layout.exists() else []
        model_count = 4 if case == "reducer" else 2
        if args.resume and len(rows) == model_count and all(
            (models/row["step"]).is_file() and (models/(Path(row["step"]).stem+"-flank.step")).is_file()
            for row in rows
        ):
            continue
        with (models/f"{case}.log").open("w", encoding="utf-8") as log:
            subprocess.run([str(executable), "models", case, str(models)], stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    protocol = [asdict(run) for run in RUNS]
    (args.output/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n", encoding="utf-8")
    for spec in RUNS:
        prefix = raw/spec.stem
        if args.resume and completed(prefix, spec):
            continue
        subprocess.run(
            [str(executable), "run", spec.case, str(models), str(spec.deflection_mm), str(spec.dt_s),
             str(spec.duration_s), str(prefix), str(spec.repeats), str(spec.stations)],
            check=True,
        )
        if not completed(prefix, spec):
            raise RuntimeError(f"short trajectory for {spec.stem}")


if __name__ == "__main__":
    main()
