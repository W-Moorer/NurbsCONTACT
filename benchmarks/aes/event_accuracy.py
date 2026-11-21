"""Measure guided versus ordinary STEP event witnesses against analytic times."""

import argparse
import csv
from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assemblies"))
from audit_comparison import exact
from threshold_scan import DELTA, DIAGONAL_GAPS_NM, FIXTURES, GAPS_NM, label, placement, reference_cases


TIME_TOLERANCE = Decimal("0.0005")


def measure(executable, fixtures, repeats):
    references, _ = reference_cases(include_diagonal=True)
    for geometry in FIXTURES:
        a, b = (("edge-plate.step", "edge-disk.step") if geometry.name == "trim-edge"
                else ("counterbore.step", "pin.step"))
        for diagonal, gaps in ((False, GAPS_NM), (True, DIAGONAL_GAPS_NM)):
            for gap in gaps:
                scene = label(geometry, gap, diagonal)
                x, y = placement(geometry, gap, diagonal)
                truth, first, _ = references[scene][1]
                args = [str(executable), "sweep", str(fixtures / a), str(fixtures / b),
                        "--translation", str(x), str(y), str(geometry.z_start),
                        "--velocity", "0", "0", str(geometry.vz), "--clearance", str(DELTA),
                        "--distance-tol", "0.0002", "--time-tol", str(TIME_TOLERANCE),
                        "--max-nodes", "200000", "--query-time"]
                for rep in range(repeats):
                    for guided in (bool(rep % 2), not bool(rep % 2)):
                        command = args + (["--guided-witness"] if guided else [])
                        result = json.loads(subprocess.run(command, check=True, capture_output=True,
                                                           text=True, timeout=90).stdout)
                        status = result["status"]
                        lower = exact(result["safe_prefix"])
                        upper = result["witness_time"]
                        assert 0 <= lower <= 1, scene
                        if truth == "hit":
                            assert status != "clear" and lower <= first, (scene, status, lower, first)
                        else:
                            assert status != "hit" and (status != "clear" or lower == 1), scene
                        width = None
                        residual_squared = None
                        if upper is not None:
                            assert status == "hit" and lower <= first <= exact(upper) <= 1, scene
                            width = exact(upper) - lower
                            witness = result["witness"]
                            assert witness["time"] == upper, scene
                            residual_squared = sum(
                                max(abs(exact(p[0]) - exact(q[1])), abs(exact(p[1]) - exact(q[0])))**2
                                for p, q in zip(witness["a"]["xyz"], witness["b"]["xyz"], strict=True)
                            )
                            assert residual_squared <= DELTA**2, (scene, residual_squared)
                        else:
                            assert status != "hit" and result["witness"] is None, scene
                        yield {
                            "scene": scene, "fixture": geometry.name, "direction": "3:4" if diagonal else "x",
                            "gap_from_threshold_nm": gap, "method": "guided" if guided else "ordinary",
                            "repeat": rep, "analytic_status": truth,
                            "analytic_time": "" if truth == "clear" else format(first, ".17g"),
                            "status": status, "safe_prefix": result["safe_prefix"], "witness_time": upper,
                            "width": "" if width is None else float(width),
                            "within_target": status == "clear" or (width is not None and width <= TIME_TOLERANCE),
                            "witness_gap_bound_mm": "" if residual_squared is None else float(residual_squared.sqrt()),
                            "visited": result["visited"], "guided_attempts": result["guided_attempts"],
                            "query_us": result["query_us"],
                        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    rows = measure(args.executable.resolve(), args.fixtures.resolve(), args.repeats)
    first = next(rows)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=first, lineterminator="\n")
        writer.writeheader()
        writer.writerow(first)
        for row in rows:
            writer.writerow(row)
            stream.flush()


if __name__ == "__main__":
    main()
