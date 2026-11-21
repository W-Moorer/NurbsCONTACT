"""Generate STEP threshold-scan motions and audit the timed answers analytically."""

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from statistics import median

from audit_comparison import exact, summarize


DELTA = Decimal(".03125")
GAPS_NM = (-10000, -1000, -300, -100, -30, -10, 0, 10, 30, 100, 300, 1000, 10000)
DIAGONAL_GAPS_NM = (-100, -10, 10, 100)


@dataclass(frozen=True)
class Fixture:
    name: str
    fixed_radius: Decimal
    moving_radius: Decimal
    z_start: Decimal
    vz: Decimal


FIXTURES = (
    Fixture("trim-edge", Decimal(1), Decimal(".25"), Decimal(".375"), Decimal(-1)),
    Fixture("counterbore", Decimal(".8125"), Decimal(".375"), Decimal(3), Decimal(-4)),
)


def offset(fixture, gap_nm):
    return fixture.fixed_radius - fixture.moving_radius - DELTA - Decimal(gap_nm).scaleb(-6)


def placement(fixture, gap_nm, diagonal):
    shift = offset(fixture, gap_nm)
    return (Decimal(".6") * shift, Decimal(".8") * shift) if diagonal else (shift, Decimal(0))


def label(fixture, gap_nm, diagonal):
    stem = f"{fixture.name}-diagonal" if diagonal else fixture.name
    return stem if gap_nm is None else f"{stem}-{gap_nm:+d}nm"


def write_cases(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for fixture in FIXTURES:
        for diagonal, gaps in ((False, GAPS_NM), (True, DIAGONAL_GAPS_NM)):
            with (directory / f"{label(fixture, None, diagonal)}.txt").open(
                    "w", encoding="utf-8", newline="\n") as stream:
                stream.write("# name A B x y z vx vy vz clearance (mm)\n")
                for gap_nm in gaps:
                    x, y = placement(fixture, gap_nm, diagonal)
                    stream.write(f"{label(fixture, gap_nm, diagonal)} 1 2 {x} {y} "
                                 f"{fixture.z_start} 0 0 {fixture.vz} {DELTA}\n")


def reference_cases(include_diagonal=False):
    references, metadata = {}, {}
    for fixture in FIXTURES:
        directions = ((False, GAPS_NM), (True, DIAGONAL_GAPS_NM)) if include_diagonal else ((False, GAPS_NM),)
        for diagonal, gaps in directions:
            for gap_nm in gaps:
                scene = label(fixture, gap_nm, diagonal)
                x, y = map(exact, placement(fixture, gap_nm, diagonal))
                eccentricity = (x**2 + y**2).sqrt() if diagonal else x
                gap = fixture.fixed_radius - fixture.moving_radius - eccentricity
                # The disk crosses the plate's inner rim. The pin enters the
                # smaller bore at z=0 and stays vertically within it through
                # t=1. Separation from the wider rim always exceeds DELTA.
                if gap <= DELTA:
                    height = (DELTA**2 - gap**2).sqrt()
                    first = (height - fixture.z_start) / fixture.vz
                    assert 0 < first < 1
                    answer = ("hit", first, first)
                else:
                    first = None
                    answer = ("clear", Decimal(1), None)
                references[scene] = (True, answer)
                axis = "3:4" if diagonal else "x"
                metadata[scene] = (fixture.name, gap_nm, (gap - DELTA).scaleb(6), first, axis)
    return references, metadata


def precision_status(row, truth):
    if row["agreement"] == "compatible":
        return row["completion"]
    if truth is None:
        return "false-event"
    if row["status"] == "clear":
        return "false-clear"
    if exact(row["lower_time"]) > truth:
        return "late-bound"
    assert row["upper_time"] and exact(row["upper_time"]) < truth
    return "early-bracket"


def cad_certificate(row):
    if row["authority"] != "cad-bounds":
        return "none"
    if row["status"] == "clear":
        return "clear-interval"
    if row["status"] == "hit":
        return "event-bracket"
    return "safe-prefix" if float(row["lower_time"]) > 0 else "none"


def audit_scan(paths, repeats, include_diagonal):
    reference, metadata = reference_cases(include_diagonal)
    rows = list(summarize(paths, repeats, reference))
    assert {row["scene"] for row in rows} == set(reference), "Incomplete threshold scan"
    prepare = defaultdict(list)
    for row in rows:
        fixture = metadata[row["scene"]][0]
        prepare[fixture, row["method"], row["deflection_mm"]].append(row["prepare_ms"])
    rows.sort(key=lambda row: (metadata[row["scene"]][0], metadata[row["scene"]][4] != "x",
                              metadata[row["scene"]][1], row["method"], row["deflection_mm"]))
    for row in rows:
        fixture, nominal, actual, first, axis = metadata[row["scene"]]
        prep = median(prepare[fixture, row["method"], row["deflection_mm"]])
        yield {
            "fixture": fixture, "offset_direction": axis, "gap_from_threshold_nm": nominal,
            "actual_gap_from_threshold_nm": format(actual, ".12g") if actual else "0",
            "analytic_first_time": "" if first is None else format(first, ".17g"),
            "accuracy": precision_status(row, first),
            "cad_certificate": cad_certificate(row),
            **row,
            "pooled_prepare_ms": round(prep, 6),
            "first_query_ms": round(prep + row["median_ms"], 6),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True, dest="command")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("directory", type=Path)
    audit = commands.add_parser("audit")
    audit.add_argument("output", type=Path)
    audit.add_argument("inputs", type=Path, nargs="+")
    audit.add_argument("--repeats", type=int, required=True)
    audit.add_argument("--include-diagonal", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        write_cases(args.directory)
        return
    rows = list(audit_scan(args.inputs, args.repeats, args.include_diagonal))
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Validated {len(rows)} configurations across {len({row['scene'] for row in rows})} motions")


if __name__ == "__main__":
    main()
