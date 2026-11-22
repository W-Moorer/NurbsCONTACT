"""Build the two published-case response figures and the Blender cam model panel."""

import argparse
import json
from pathlib import Path
import subprocess

from paper_diagrams import AMBER, BLUE, GREEN, GRID, save
from paper_responses import model_panels, style
from literature_validation import cam_shapes, oscillator_reference
import matplotlib.pyplot as plt
import numpy as np


def oscillator_plot(work, output, params):
    trace = np.genfromtxt(work / "oscillator-2.csv", delimiter=",", names=True)
    events = np.genfromtxt(work / "oscillator-events.csv", delimiter=",", names=True)
    t = trace["time"]
    q = trace["displacement"]
    exact, _, _, period = oscillator_reference(t, params["omega_0"], params["epsilon"], params["amplitude"])
    early = t <= 2*period
    fig, (response, phase) = plt.subplots(1, 2, figsize=(7.12, 2.5), layout="constrained")
    response.fill_between(t[early], -1.12, 1.12, where=exact[early] > 0, color=GRID, alpha=.53,
                          label="Contact active")
    response.plot(t[early], q[early], color=BLUE, lw=1.8, label="HHT")
    response.plot(t[early], exact[early], color=AMBER, lw=1.1, ls="--", label="Closed form")
    response.set(xlabel="Normalized time", ylabel="Displacement", ylim=(-1.14, 1.14))
    response.legend(frameon=False, fontsize=7.8, ncol=3, loc="lower center", bbox_to_anchor=(.5, 1.01))
    cycle = np.arange(1, params["cycles"] + 1)
    phase.plot(cycle, events["error"][1::2]*1e6, color=BLUE, lw=1.55, label="Contact onset")
    phase.plot(cycle, events["error"][::2]*1e6, color=GREEN, lw=1.25, ls="--", label="Release")
    phase.set(xlabel="Cycle", ylabel=r"Timing error ($10^{-6}$ time units)")
    phase.legend(frameon=False, fontsize=7.8, loc="upper left")
    for ax in (response, phase):
        style(ax, ax.get_xlabel())
    save(fig, output, "unilateral-response")


def cam_plot(work, output):
    trace = np.genfromtxt(work / "cam-motion.csv", delimiter=",", names=True)
    angle = np.rad2deg(trace["time"])
    smooth = trace["smooth"].astype(bool)
    fig, axes = plt.subplots(3, 1, figsize=(7.12, 4.02), sharex=True, layout="constrained")
    labels = ("Displacement", "Velocity", "Acceleration")
    for ax, field, label in zip(axes, ("position", "velocity", "acceleration"), labels, strict=True):
        numerical, reference = trace[field].copy(), trace["exact_"+field].copy()
        if field == "acceleration":
            numerical[~smooth] = np.nan
            reference[~smooth] = np.nan
        ax.plot(angle, numerical, color=BLUE, lw=1.8, label="CAD reconstruction")
        ax.plot(angle, reference, color=AMBER, ls="--", lw=1.1, label="Dhande and Rajaram")
        ax.set_ylabel(label, fontsize=8.6)
        ax.set_xlim(0, 360)
    axes[0].legend(frameon=False, fontsize=8., loc="upper right", ncol=2)
    axes[-1].set_xticks(np.arange(0, 361, 60))
    style(axes, "Cam rotation (degrees)")
    save(fig, output, "constant-breadth-response")


def cam_model(work, output, blender, params):
    from OCP.BRep import BRep_Tool
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.TopLoc import TopLoc_Location
    from OCP.gp import gp_Vec

    wire, _, _, _ = cam_shapes(*(params[k] for k in ("a", "b", "c", "d")))
    filled = BRepPrimAPI_MakePrism(BRepBuilderAPI_MakeFace(wire).Face(), gp_Vec(0, 0, 1)).Shape()
    BRepMesh_IncrementalMesh(filled, .008, False, .075).Perform()
    vertices, triangles = [], []
    faces = TopExp_Explorer(filled, TopAbs_FACE)
    while faces.More():
        face = TopoDS.Face_s(faces.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is None:
            raise ValueError("Cam face has no render triangulation")
        start = len(vertices) - 1  # OCCT node indices start at one.
        for i in range(1, triangulation.NbNodes() + 1):
            point = triangulation.Node(i).Transformed(location.Transformation())
            vertices.append((point.X(), point.Y(), point.Z()))
        for i in range(1, triangulation.NbTriangles() + 1):
            a, b, c = triangulation.Triangle(i).Get()
            if face.Orientation() == TopAbs_REVERSED:
                b, c = c, b
            triangles.append((start + a, start + b, start + c))
        faces.Next()
    points = np.asarray(vertices)
    # The 90-degree pose is on the numerical motion grid. Rotate local points by pi/2.
    points[:, :2] = np.column_stack((-points[:, 1], points[:, 0]))
    trace = np.genfromtxt(work / "cam-motion.csv", delimiter=",", names=True)
    frame = np.argmin(np.abs(trace["time"] - np.pi/2))
    right = trace["position"][frame] + params["d"]
    np.savez(work / "constant-breadth.npz", points=points, triangles=np.asarray(triangles),
             right=right, left=right-params["breadth"])
    blender_script = Path(__file__).resolve().with_name("blender_models.py")
    subprocess.run([str(blender), "-b", "--factory-startup", "--python-exit-code", "1", "--python",
                    str(blender_script), "--", str(work), str(work), str(work), str(work / "paper/renders"),
                    "--constant-breadth-only"], check=True)
    model_panels(work, output, ("constant-breadth",))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("blender", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.work / "summary.json").open(encoding="utf-8") as stream:
        results = json.load(stream)
    oscillator_plot(args.work, args.output, results["oscillator"])
    cam_plot(args.work, args.output)
    cam_model(args.work, args.output, args.blender, results["cam"])


if __name__ == "__main__":
    main()
