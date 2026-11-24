"""Prepare and run fixed-hinge mesh contact on a CUDA GPU."""

import argparse
from pathlib import Path


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("deck", type=Path)
    cli.add_argument("work", type=Path)
    cli.add_argument("--prepare", action="store_true")
    cli.add_argument(
        "--scene",
        type=Path,
        help="Build the device deck from an nc-contact CPU scene and its Gmsh meshes",
    )
    cli.add_argument("--duration", type=float, default=5.0)
    cli.add_argument("--step", type=float, default=0.0001)
    cli.add_argument("--bins", type=int, default=4096)
    cli.add_argument("--angle-band", type=float, default=0.0025)
    cli.add_argument("--method", choices=("mesh-anchors", "mesh-wrench", "wrench-only"), default="mesh-wrench")
    cli.add_argument("--output", type=Path)
    cli.add_argument("--warps", type=int, choices=(1, 2, 4),
                     help="Warps per contact pair (default: 2 for mapped exchange, otherwise 4)")
    cli.add_argument("--geometry", type=Path)
    transfer = cli.add_mutually_exclusive_group()
    transfer.add_argument("--stream-io", action="store_true")
    transfer.add_argument("--exchange", action="store_true", help="Use the packaged synchronous exchange library")
    transfer.add_argument("--exchange-lib", type=Path, help="Use an external exchange library")
    cli.add_argument("--steps", type=int)
    arg = cli.parse_args()
    from .backend import export_scene, prepare, run

    if arg.scene:
        export_scene(arg.scene, arg.deck, arg.duration, arg.step)
    if arg.prepare:
        prepare(arg.deck, arg.work, arg.bins, arg.angle_band)
    else:
        run(
            arg.deck,
            arg.work,
            arg.output or arg.work / arg.method,
            arg.method,
            arg.steps,
            arg.geometry,
            arg.warps,
            arg.stream_io,
            arg.exchange_lib,
            exchange=arg.exchange,
        )
