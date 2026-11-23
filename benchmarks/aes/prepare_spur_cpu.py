"""Build the default CPU contact mesh pair and shared dynamics scene for the archived spur gears."""

import argparse
from pathlib import Path
import shutil
import sys

from pair_dynamics import gears

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "apps"))
from prepare_contact_mesh import prepare


def main(directory):
    source = ROOT / "results/gears/models"
    directory.mkdir(parents=True, exist_ok=True)
    for filename in ("spur.csv", "spur-1a.step", "spur-1b.step"):
        shutil.copyfile(source / filename, directory / filename)
    for part in ("spur-1a", "spur-1b"):
        quality = prepare(
            directory / f"{part}.step",
            directory / f"{part}.msh",
            0.03,
            max_error_m=0.00005,
            max_normal_deg=8,
            extend_boundary=False,
            mapped_edge_error_m=0.000002,
            curvature_samples=64,
        )
        print(f"{part}: {quality['triangles']} triangles", flush=True)
    gears(directory, directory / "spur.scene", 0.36, 56, surface=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
