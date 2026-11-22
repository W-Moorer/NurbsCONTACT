"""Comparable pure mesh vertex, field and CAD sampling controls on the cam."""

import argparse
import json
from pathlib import Path

from cam_accuracy_cost import reference_data, run_case


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--base", type=Path, default=Path("build/resident-cam"))
    args = p.parse_args()
    ref = reference_data(args.work, 160001, 0.000125)
    configs = []
    for level in (1.0, 0.25, 0.125, 0.0625):
        configs.append(("vertices", level, "grid", 4, 1))
    for pitch in (2, 4, 8):
        configs.extend((("cad", 1.0, "sdf", pitch, 1), ("mesh", 0.0625, "sdf", pitch, 1)))
    configs.extend([("cad", 1.0, "grid", 4, 1), ("mesh", 0.0625, "grid", 4, 1)])
    rows = []
    for method, level, backend, pitch, lanes in configs:
        name = f"{method}-{level:g}-{backend}-{pitch}-w{lanes}"
        row = run_case(
            args.work,
            name,
            args.work / f"deck-{level:g}.npz",
            ref,
            probes=args.base / "cam-probes.npz" if method == "cad" else None,
            pitch=pitch,
            backend=backend,
            repeats=1,
            extra=["--lanes", str(lanes)] + (["--vertex-probes"] if method == "vertices" else []),
        )
        row.update(method=method, deflection_um=level, backend=backend, pitch_mm=pitch, lanes=lanes)
        rows.append(row)
        args.output.write_text(json.dumps(rows, indent=2) + "\n")
        print(name, row["status"], row.get("rms_um"), row.get("online_s"), flush=True)


if __name__ == "__main__":
    main()
