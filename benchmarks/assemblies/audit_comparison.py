"""Validate and summarize the continuous clearance baseline CSVs."""

import argparse
import csv
from collections import defaultdict
from decimal import Decimal, getcontext
from pathlib import Path
from statistics import median


HERE = Path(__file__).resolve().parent
CERTIFIED = {"native-sweep", "cad-ca-bounds", "mesh-ccd-cad-bounds"}
METHODS = {(name, 0.0) for name in CERTIFIED - {"mesh-ccd-cad-bounds"}} | {
    ("occt-ca", 0.0),
    ("mesh-ccd-cad-bounds", .003125),
    *(("mesh-bvh-ccd", h) for h in (.05, .0125, .003125)),
}
REFINED_MESHES = {("mesh-bvh-ccd", h) for h in (.000390625, .000048828125, .000006103515625)}
getcontext().prec = 70


def exact(value):
    return Decimal.from_float(float(value))


def analytic_references():
    """Use the geometry of the generated fixtures, independently of either solver."""
    references = {}
    for file in ("edge-proximity.txt", "counterbore-proximity.txt"):
        for line in (HERE / file).read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            scene, _, _, x, _, z, _, _, speed, delta = line.split()
            x, z, speed, delta = map(exact, (x, z, speed, delta))
            if file.startswith("edge"):
                rims = [(Decimal(1), Decimal(".25"), Decimal(0))]
            else:
                rims = [(Decimal("1.375"), Decimal(".375"), Decimal(2)),
                        (Decimal(".8125"), Decimal(".375"), Decimal(0))]
            roots = []
            for radius, moving_radius, height in rims:
                radial_gap = max(Decimal(0), radius - x - moving_radius)
                if radial_gap <= delta:
                    vertical_gap = (delta**2 - radial_gap**2).sqrt()
                    roots.append((height + vertical_gap - z) / speed)
            first = min(roots, default=None)
            reference = ("clear", Decimal(1), None) if first is None else ("hit", first, first)
            references[scene] = (file.startswith("edge"), reference)
    return references


def optional(value):
    return exact(value) if value else None


def outcome(row, analytic):
    status = row["status"]
    lo, hi = exact(row["lower_time"]), optional(row["upper_time"])
    assert status in {"clear", "hit", "unresolved"}
    assert 0 <= lo <= 1
    assert (hi is not None) == (status == "hit")
    assert hi is None or lo <= hi <= 1
    assert status != "clear" or lo == 1

    scene = row["scene"]
    ref_kind = "analytic" if scene in analytic else "occt"
    if scene in analytic:
        _, (ref_status, ref_lo, ref_hi) = analytic[scene]
    else:
        ref_status = row["reference_status"]
        ref_lo, ref_hi = exact(row["reference_lower"]), optional(row["reference_upper"])
    if ref_status == "hit":
        conflict = status == "clear" or lo > ref_hi or (hi is not None and hi < ref_lo)
    elif ref_status == "clear":
        conflict = status == "hit"
    else:
        conflict = hi is not None and hi < ref_lo
    agreement = "conflict" if conflict else "unknown" if ref_status == "unresolved" else "compatible"
    width = hi - lo if hi is not None else None
    meets = status == "clear" or (width is not None and width <= exact(row["time_tolerance"]))
    result = "within-target" if meets else "wide" if hi is not None else "unresolved"
    if ref_kind == "analytic" and row["method"] in CERTIFIED:
        assert not conflict, f"{scene}: {row['method']} contradicts the analytic answer"
    return result, agreement, width, ref_kind, ref_status


def summarize(paths, repeats, analytic=None):
    if analytic is None:
        analytic = analytic_references()
    groups = defaultdict(list)
    all_methods = defaultdict(set)
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = row["scene"], row["method"], float(row["deflection_mm"])
                groups[key].append(row)
                all_methods[key[0]].add(key[1:])
    assert groups, "No comparison rows"
    for scene, methods in all_methods.items():
        expected = METHODS | REFINED_MESHES if scene in analytic and analytic[scene][0] else METHODS
        assert methods == expected, (scene, "Incomplete methods")
        bounded = [groups[scene, method, h][0] for method, h in methods if method in CERTIFIED]
        witnesses = [exact(row["upper_time"]) for row in bounded if row["upper_time"]]
        if witnesses:
            assert not any(row["status"] == "clear" for row in bounded), (scene, "Conflicting certificates")
            assert max(exact(row["lower_time"]) for row in bounded) <= min(witnesses), (
                scene, "Disjoint certified brackets")

    for (scene, method, deflection), rows in sorted(groups.items()):
        assert sorted(int(row["repeat"]) for row in rows) == list(range(repeats)), (scene, method, deflection)
        results = {outcome(row, analytic) for row in rows}
        assert len(results) == 1, f"{scene}: {method} produced inconsistent outcomes"
        assert len({(row["status"], row["lower_time"], row["upper_time"]) for row in rows}) == 1
        result, agreement, width, ref_kind, ref_status = results.pop()
        first = rows[0]
        timings = [float(row["query_us"]) / 1000 for row in rows]
        assert all(t > 0 for t in timings)
        assert float(first["prepare_us"]) >= 0
        yield {
            "scene": scene, "method": method, "deflection_mm": deflection,
            "authority": "cad-bounds" if method in CERTIFIED else "numerical",
            "status": first["status"], "completion": result, "reference": ref_kind,
            "reference_status": ref_status, "agreement": agreement,
            "lower_time": first["lower_time"], "upper_time": first["upper_time"],
            "width": "" if width is None else float(width),
            "median_ms": round(median(timings), 6), "max_ms": round(max(timings), 6),
            "prepare_ms": round(float(first["prepare_us"]) / 1000, 6),
            "triangles": int(first["triangles"]),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--repeats", type=int, required=True)
    args = parser.parse_args()
    rows = list(summarize(args.inputs, args.repeats))
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Validated {len(rows)} method/scene configurations across {args.repeats} repetitions")


if __name__ == "__main__":
    main()
