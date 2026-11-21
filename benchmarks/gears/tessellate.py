"""Tessellate an Open CASCADE STEP solid for the separate VTK renderer."""

import argparse
import os
from pathlib import Path

import cadquery as cq
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    model=cq.importers.importStep(str(args.step)).val()
    if len(model.Solids())!=1 or not model.isValid():
        raise ValueError(f"Invalid STEP solid {args.step}")
    coordinates,triangles=model.tessellate(.13,.19)
    points=np.array([vertex.toTuple() for vertex in coordinates])
    np.savez(args.output,points=points,triangles=np.asarray(triangles))


if __name__=="__main__":
    main()
    if os.name=="nt":
        # This isolated worker has closed the NPZ file. Skip the CAD bindings'
        # native teardown, which can corrupt the Windows process heap.
        os._exit(0)
