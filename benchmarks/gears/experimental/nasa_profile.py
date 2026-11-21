"""Digitize the four profile traces per gear in NASA TM-103232 Fig. 3."""

import argparse
from pathlib import Path

import fitz
import numpy as np


def extract(pdf, out):
    # Full-page pixels at 3 px/pt. The two panels have separate x calibrations.
    panels = ((730, 1087, (1568, 1600, 1631, 1663), 23.),
              (733, 1090, (1809, 1840, 1871, 1902), 25.))
    with fitz.open(pdf) as paper:
        pix = paper[2].get_pixmap(matrix=fitz.Matrix(3, 3), colorspace=fitz.csGRAY)
    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    degrees = np.arange(7., 32.)
    curves, raw = [], []
    for side, (x6, x33, seeds, tip_um) in enumerate(panels):
        x = np.rint(x6 + (x33-x6)*(degrees-6)/27).astype(int)
        profiles = []
        for tooth, seed in zip((22, 15, 8, 1), seeds, strict=True):
            y = []
            ypos = float(seed)
            for column in x:
                rows = np.arange(round(ypos)-7, round(ypos)+13)
                ink = np.maximum(150-gray[np.ix_(rows, np.arange(column-2, column+3))].astype(float), 0)
                weights = ink.sum(axis=1)
                if not weights.any():
                    raise ValueError(f"Profile trace missing for gear {side}, tooth {tooth}, pixel {column}")
                ypos = float(weights @ rows / weights.sum())
                y.append(ypos)
            profile = np.array(y)-np.mean(y[5:14])
            profiles.append(profile)
            raw.extend(zip(np.full(len(x), side+1), np.full(len(x), tooth), degrees, x, y))
        average = np.mean(profiles, axis=0)
        # Use printed Table I tip magnitude and the relative measured trace
        # shape. A linear extrapolation of the last degree reaches the CAD tip.
        pitch_radius = 3.175*28/2
        tip_roll = np.degrees(np.sqrt(((pitch_radius+3.175)/(pitch_radius*np.cos(np.radians(20))))**2-1))
        end = average[-1] + (tip_roll-degrees[-1])*(average[-1]-average[-2])
        curves.append(np.r_[average*tip_um/end, tip_um])
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out/"rig28-profiles.csv", np.column_stack((np.r_[degrees, tip_roll], *curves)),
               header="roll_deg,driver_um,driven_um", delimiter=",", comments="", fmt="%.9f")
    np.savetxt(out/"rig28-profile-pixels.csv", raw,
               header="gear,tooth,roll_deg,source_x_px,source_y_px", delimiter=",", comments="", fmt="%.6f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    extract(args.pdf, args.output)
