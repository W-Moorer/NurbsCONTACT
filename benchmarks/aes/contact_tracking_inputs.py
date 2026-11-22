"""Export CAD face IDs and rigid poses for a full-mesh contact replay."""

import argparse
import csv
from hashlib import sha256
from pathlib import Path

import numpy as np

from gear_contact_paths import angle
from gear_precision import mesh_step


def owners(args):
    tags = []
    v, f = mesh_step(args.deflection_um, False, step=args.step, face_ids=tags)
    with np.load(args.mesh) as mesh:
        np.testing.assert_array_equal(mesh["v"], v)
        np.testing.assert_array_equal(mesh["f"], f)
    tags = np.concatenate(tags)
    if tags.min() < 1:
        raise ValueError("Unmapped STEP face")
    np.savez_compressed(args.output, face=tags, mesh_sha256=sha256(args.mesh.read_bytes()).hexdigest())


def poses(args):
    with args.parts.open(newline="") as stream:
        a, b = csv.DictReader(stream)
    dx = float(b["center_x_mm"]) - float(a["center_x_mm"])
    dz = float(b["z_mm"]) - float(a["z_mm"])
    rows = []
    with args.trace.open(newline="") as stream:
        for row in csv.DictReader(stream):
            t = float(row["time"])
            if t < args.start - 1e-12:
                continue
            if t > args.end + 1e-12:
                break
            aa, ab = angle(a, row), angle(b, row)
            c, s = np.cos(ab - aa), np.sin(ab - aa)
            rows.append([t, c, -s, 0, dx * np.cos(aa), s, c, 0, -dx * np.sin(aa), 0, 0, 1, dz])
    if len(rows) < 2:
        raise ValueError("The window needs at least two recorded poses.")
    np.savetxt(args.output, rows, fmt="%.17g", header="time_s and row-major 3x4 relative transform in mm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    m, p = sub.add_parser("owners"), sub.add_parser("poses")
    for cmd, names in [(m, ("step", "mesh", "output")), (p, ("parts", "trace", "output"))]:
        for name in names:
            cmd.add_argument(name, type=Path)
    m.add_argument("--deflection-um", type=float, required=True)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (owners if args.command == "owners" else poses)(args)
