"""Check the CCD follow-up against independent analytic and OCCT reference data."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

import mpmath as mp


mp.mp.dps = 80
DELTA = mp.mpf("0.03125")
TIME_TARGET = mp.mpf("0.0005")
NODE_CAP = 200_000
ABLATIONS = {"swept-box", "directional", "axial"}
EXPECTED = {
    "rotation": ({"planar-tilt", "curved-tilt", "interior-entry", "rotating-near-miss"},
                 {"chen-obb-chord", "native"}),
    "counterbore": ({"throat-pass", "shoulder-entry", "mouth-entry"}, ABLATIONS),
    "assembly": ({"voron-spacer", "voron-washer", "voron-bearing"}, ABLATIONS),
    "audit": ({"trim-edge-entry", "narrow-edge-entry", "exact-edge-graze", "near-edge-miss"},
              {"axial"}),
}


def number(row, field):
    """Interpret an exported round-trip decimal as the original binary64."""
    return mp.mpf(float(row[field]))


def coordinate_box(row, body):
    bounds = []
    for axis in "xyz":
        lo, hi = (number(row, f"{body}_{axis}_{side}") for side in ("lo", "hi"))
        assert mp.isfinite(lo) and mp.isfinite(hi) and lo <= hi
        bounds.append((lo, hi))
    return bounds


def spans(interval, x):
    return interval[0] <= x <= interval[1]


def overlaps(interval, lo, hi):
    return interval[0] <= hi and lo <= interval[1]


def squared(interval):
    lo, hi = interval
    return 0 if spans(interval, 0) else min(lo**2, hi**2), max(lo**2, hi**2)


def radial_squared(box, offset=0):
    sx = squared((box[0][0]-offset, box[0][1]-offset))
    sy = squared(box[1])
    return sx[0]+sy[0], sx[1]+sy[1]


def check_witness(row, time, group):
    a, b = (coordinate_box(row, body) for body in ("a", "b"))
    offset = number(row, "offset")
    z = number(row, "z_start") + number(row, "z_rate")*time
    ra, rb = radial_squared(a), radial_squared(b, offset)
    if group == "audit":
        # The trimmed plate is the square minus the open unit disk.
        # The independently closest part of the plate is its inner edge.
        assert spans(a[2], 0) and spans(b[2], z)
        assert all(overlaps(i, -3, 3) for i in a[:2])
        assert ra[1] >= 1 and rb[0] <= mp.mpf(".25")**2
        if row["scene"] in {"trim-edge-entry", "narrow-edge-entry"}:
            assert row["a_kind"] == "edge"
        if row["a_kind"] in {"edge", "vertex"}:
            assert spans(ra, 1)
        if row["b_kind"] in {"edge", "vertex"}:
            assert spans(rb, mp.mpf(".25")**2)
    else:
        r1, r2, rout = map(mp.mpf, (".8125", "1.375", "2.5"))
        cylinders = ((r1, -3, 0), (r2, 0, 2), (rout, -3, 2))
        rims = ((-3, r1, rout), (0, r1, r2), (2, r2, rout))
        assert any(spans(ra, r**2) and overlaps(a[2], lo, hi) for r, lo, hi in cylinders) or any(
            spans(a[2], height) and overlaps(ra, lo**2, hi**2) for height, lo, hi in rims
        )
        pin = mp.mpf(".375")
        top = z+mp.mpf(".5")
        assert (spans(rb, pin**2) and overlaps(b[2], z, top)) or (
            rb[0] <= pin**2 and (spans(b[2], z) or spans(b[2], top))
        )
    upper = mp.sqrt(sum(squared((p[0]-q[1], p[1]-q[0]))[1] for p, q in zip(a, b, strict=True)))
    assert upper <= DELTA
    return upper


def reference(row, group):
    if group == "rotation":
        height, angle = number(row, "height"), number(row, "angle")
        # For the curved fixture cos(theta)>0. Its minimum is on v=1/2;
        # for all fixtures the minimizing x is the rightmost pole x=1.
        gap = height-1
        if height-DELTA > 1:
            assert angle >= mp.pi/2
            return None, None, gap
        root = mp.asin(height-DELTA)/angle
        assert 0 < root < 1
        if row["scene"] == "interior-entry":
            assert height-mp.sin(angle) > DELTA
        return root, root, None
    if group == "assembly":
        assert row["reference_status"] in {"hit", "clear", "unresolved"}
        if row["reference_status"] == "clear":
            return None, None, None
        if row["reference_status"] == "unresolved":
            return number(row, "reference_lower"), None, None
        lower, upper = (number(row, f"reference_{end}") for end in ("lower", "upper"))
        assert 0 <= lower <= upper <= 1 and upper-lower <= TIME_TARGET
        return lower, upper, None
    offset, z, velocity = (number(row, field) for field in ("offset", "z_start", "z_rate"))
    if group == "counterbore":
        pin = mp.mpf(".375")
        eligible = []
        for radius, end in ((mp.mpf("1.375"), 2), (mp.mpf(".8125"), 0)):
            gap = radius-(offset+pin)
            if gap <= DELTA:
                entry_height = end + mp.sqrt(DELTA**2-max(0, gap)**2)
                eligible.append((entry_height-z)/velocity)
        root = min(eligible, default=None)
        return root, root, None
    gap = 1-(offset+mp.mpf(".25"))
    if gap > DELTA:
        return None, None, gap
    root = (mp.sqrt(DELTA**2-gap**2)-z)/velocity
    return root, root, gap


def audit(row, group):
    status = row["status"]
    assert status in {"clear", "hit", "unresolved"}
    lower = number(row, "safe_prefix")
    assert 0 <= lower <= 1 and int(row["nodes"]) <= NODE_CAP
    ref_lo, ref_hi, min_distance = reference(row, group)
    if status == "clear":
        assert ref_lo is None and lower == 1
    elif ref_hi is not None:
        assert lower <= ref_hi, "prefix overran the reference event"
    elif ref_lo is not None:
        assert status == "unresolved" and lower <= ref_lo
    upper = residual = None
    if status == "hit":
        assert ref_lo is not None
        upper = number(row, "witness_time")
        assert lower <= upper <= 1 and upper >= ref_lo
        if group in {"counterbore", "audit"}:
            residual = check_witness(row, upper, group)
            if group == "audit":
                assert min_distance <= DELTA
                z = number(row, "z_start") + number(row, "z_rate")*upper
                assert min_distance**2+z**2 <= DELTA**2
        if group == "rotation":
            residual = number(row, "point_residual")
            assert 0 <= residual <= DELTA
    elif group == "rotation":
        assert not math.isfinite(float(row["point_residual"]))
    if group == "rotation" and row["method"] == "chen-obb-chord":
        assert 0 < number(row, "max_chord_padding") < mp.mpf(".000201")
    return status, lower, upper, residual, ref_lo, ref_hi, min_distance


def summarize(path, group, repeats):
    raw = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            raw[row["scene"], row.get("method") or row["variant"]].append(row)
    cases, methods = EXPECTED[group]
    assert set(raw) == {(case, method) for case in cases for method in methods}, path
    output = {}
    for (scene, method), rows in raw.items():
        assert sorted(int(row["repeat"]) for row in rows) == list(range(repeats))
        records = [audit(row, group) for row in rows]
        statuses, prefixes, upper, distances, ref_lo, ref_hi, gaps = zip(*records, strict=True)
        assert len(set(statuses)) == len(set(ref_lo)) == len(set(ref_hi)) == 1
        widths = [up-lo for lo, up in zip(prefixes, upper, strict=True) if up is not None]
        measured = sorted(float(r["query_us"]) for r in rows)
        statuses = statuses[0]
        expected = "hit" if ref_hi[0] is not None else "unresolved" if ref_lo[0] is not None else "clear"
        if group in {"rotation", "counterbore"}:
            assert statuses == expected and all(w <= TIME_TARGET for w in widths)
        if group == "audit" and scene in {"trim-edge-entry", "near-edge-miss"}:
            assert statuses == expected and all(w <= TIME_TARGET for w in widths)
        output.setdefault(scene, {})[method] = {
            "status": statuses,
            "minimum_safe_prefix": float(min(prefixes)),
            "reference_status": expected,
            "reference_lower": mp.nstr(ref_lo[0], 45) if ref_lo[0] is not None else None,
            "reference_upper": mp.nstr(ref_hi[0], 45) if expected == "hit" else None,
            "reference_minimum_distance": mp.nstr(gaps[0], 45) if gaps[0] is not None else None,
            "maximum_width": float(max(widths)) if widths else None,
            "minimum_witness_time": min(float(x) for x in upper if x is not None) if widths else None,
            "maximum_witness_distance": max((float(x) for x in distances if x is not None), default=None),
            "witness_a_features": sorted({r["a_kind"] for r in rows if r.get("a_kind")}),
            "runs_above_width_target": sum(w > TIME_TARGET for w in widths),
            "median_us": median(measured),
            "p95_us": measured[math.ceil(.95*len(measured))-1],
            "median_nodes": median(int(r["nodes"]) for r in rows),
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="directory with the four result CSVs")
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--partial", action="store_true", help="allow missing groups during execution")
    args = parser.parse_args()
    report = {}
    for group in ("rotation", "counterbore", "assembly", "audit"):
        path = args.directory/f"{group}.csv"
        if path.exists() or not args.partial:
            report[group] = summarize(path, group, args.repeats)
    assert report
    (args.directory/"summary.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(", ".join(f"{k}: {len(v)} cases" for k, v in report.items()))


if __name__ == "__main__":
    main()
