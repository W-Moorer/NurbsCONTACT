"""Run the upstream or safeguarded TDIB solver on queries with exact TDIB-E labels."""

import argparse
from collections import Counter
import csv
from fractions import Fraction
import hashlib
from itertools import islice
import json
from pathlib import Path
import subprocess
import time


def replay(args):
    started = time.monotonic()
    results = []
    files = {}
    for contact in ("face-face", "vertex-vertex"):
        for category, truth in (("rp", 1), ("rn", 0), ("nhp", 1), ("nmn", 0)):
            path = args.dataset / contact / (category + ".csv")
            key = f"{contact}/{category}"
            if args.only and key != args.only:
                continue
            hasher = hashlib.sha256()
            row_count = 0
            with path.open("rb") as stream:
                for line in stream:
                    hasher.update(line)
                    row_count += 1
            if not row_count or row_count % 24:
                raise ValueError(f"{path} has incomplete query blocks")
            total = row_count // 24
            count = min(args.per_file, total)
            # Midpoint of each equal-sized index stratum, without label-dependent selection.
            indices = [(2 * i + 1) * total // (2 * count) for i in range(count)]
            if args.index:
                indices = sorted(set(args.index))
                if indices[0] < 0 or indices[-1] >= total:
                    raise ValueError(f"query index outside {key}")
            with path.open(newline="") as stream:
                blocks = {}
                cursor = 0
                for index in indices:
                    blocks[index] = list(
                        csv.reader(islice(stream, 24 * index - cursor, 24 * (index + 1) - cursor))
                    )
                    cursor = 24 * (index + 1)
            outcomes = Counter()
            for index in indices:
                points = []
                for row in blocks[index]:
                    if len(row) != 7 or int(row[6]) != truth:
                        raise ValueError(f"unexpected label or record width in {key} query {index}")
                    points.append([Fraction(int(row[k]), int(row[k + 1])) for k in (0, 2, 4)])
                values = [[float(x) for x in p] for p in points]
                # Bound the input change for the actual affine motion, including
                # the subtraction of rounded endpoints used to prepare the velocity.
                error = Fraction()
                for j in range(12):
                    for k in range(3):
                        start = Fraction.from_float(values[j][k])
                        delta = Fraction.from_float(values[j + 12][k] - values[j][k])
                        error = max(
                            error, abs(start - points[j][k]), abs(start + delta - points[j + 12][k])
                        )
                payload = "".join(" ".join(format(x, ".17g") for x in p) + "\n" for p in values)
                if args.method == "chen-2026" and any(Fraction.from_float(y) != x
                        for p, q in zip(points, values) for x, y in zip(p, q)):
                    raise ValueError("the binary64 endpoints changed the rational query")
                entry = dict(file=key, index=index, label=truth, max_coordinate_perturbation=float(error))
                remaining = 1800 - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("dataset replay reached the 30-minute budget")
                try:
                    run = subprocess.run(
                        [args.exe, str(args.width), *(["--robust"] if args.method == "chen-2026" else [])],
                        input=payload,
                        text=True,
                        capture_output=True,
                        timeout=min(remaining, args.query_timeout),
                    )
                except subprocess.TimeoutExpired:
                    entry["outcome"] = "timeout"
                else:
                    if run.returncode:
                        raise RuntimeError(f"{key} query {index}: {run.stderr.strip()}")
                    reply = json.loads(run.stdout)
                    entry.update(reply)
                    if "outcome" not in reply:
                        entry["outcome"] = "candidate" if reply["candidate"] else "clear"
                outcomes[entry["outcome"]] += 1
                results.append(entry)
            files[key] = dict(
                available=total, tested=len(indices), sha256=hasher.hexdigest(), outcomes=dict(outcomes)
            )
    return dict(
        solver=("Chen et al. 2024 official TDIB-OBB" if args.method == "chen-2024"
                else "Independent TDIB-E variant with outward-rounded line envelopes"),
        dataset="Chen et al. 2026 TDIB-E",
        width=args.width,
        query_timeout_s=args.query_timeout,
        exe_sha256=hashlib.sha256(args.exe.read_bytes()).hexdigest(),
        elapsed_s=time.monotonic() - started,
        files=files,
        queries=results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("exe", "dataset", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--per-file", type=int, default=16)
    parser.add_argument("--method", choices=["chen-2024", "chen-2026"], default="chen-2024")
    parser.add_argument("--width", type=float, default=0.0005)
    parser.add_argument("--query-timeout", type=float, default=10.0)
    parser.add_argument(
        "--only",
        choices=[f"{c}/{s}" for c in ("face-face", "vertex-vertex") for s in ("rp", "rn", "nhp", "nmn")],
    )
    parser.add_argument("--index", action="append", type=int, help="zero-based query in the selected CSV")
    args = parser.parse_args()
    if args.per_file <= 0 or not 0 < args.width <= 1 or not 0 < args.query_timeout <= 1800:
        parser.error("expected a positive sample count, valid cell width and query budget")
    if args.index and not args.only:
        parser.error("--index requires --only")
    report = replay(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
