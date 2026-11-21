"""Compare the single cam under Drake's native point contact law."""

import argparse
import csv
from importlib.metadata import version
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np


DURATION = 20.
TRACE_DT = .001
STEPS = (.001, .0005, .00025, .000125, .0000625)
STIFFNESS = 5000.
OFFSET = .024
PADDED_RADIUS = .117
FOLLOWER_HEIGHT = .115


def build(step):
    from pydrake.geometry import AddContactMaterial, Cylinder, HalfSpace, ProximityProperties
    from pydrake.math import RigidTransform, RollPitchYaw
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, ContactModel, CoulombFriction
    from pydrake.multibody.tree import PrismaticJoint, RevoluteJoint, SpatialInertia, UnitInertia
    from pydrake.systems.analysis import ResetIntegratorFromFlags, Simulator
    from pydrake.systems.framework import DiagramBuilder
    from pydrake.systems.primitives import Adder, MatrixGain, TrajectorySource
    from pydrake.trajectories import PiecewisePolynomial

    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, 0.)
    plant.mutable_gravity_field().set_gravity_vector([0., 0., 0.])
    plant.set_contact_model(ContactModel.kPoint)
    cam = plant.AddRigidBody("cam", SpatialInertia(1., [0., 0., 0.], UnitInertia(.007, .007, .007)))
    follower = plant.AddRigidBody("follower", SpatialInertia(.48, [0., 0., 0.], UnitInertia(.01, .01, .01)))
    turn = plant.AddJoint(RevoluteJoint("pivot", plant.world_frame(), cam.body_frame(), [0., 0., 1.]))
    slide = plant.AddJoint(PrismaticJoint("slide", plant.world_frame(), follower.body_frame(), [0., 1., 0.]))
    props = ProximityProperties()
    AddContactMaterial(dissipation=0., point_stiffness=2*STIFFNESS,
                       friction=CoulombFriction(0., 0.), properties=props)
    plant.RegisterCollisionGeometry(cam, RigidTransform([OFFSET, 0., 0.]),
                                    Cylinder(PADDED_RADIUS, .060), "cam-cylinder", props)
    plant.RegisterCollisionGeometry(follower,
                                    RigidTransform(RollPitchYaw(math.pi/2, 0., 0.), [0., FOLLOWER_HEIGHT, 0.]),
                                    HalfSpace(), "follower-plane", props)
    plant.Finalize()
    if (turn.position_start(), slide.position_start(), turn.velocity_start(), slide.velocity_start()) != (0, 1, 0, 1):
        raise RuntimeError("Unexpected Drake coordinate order")
    spring = builder.AddSystem(MatrixGain([[-1.8, 0., -.1, 0.], [0., -70., 0., -.65]]))
    drive = builder.AddSystem(TrajectorySource(PiecewisePolynomial.FirstOrderHold(
        [0., DURATION], np.array([[.4, .4 + 7.2*DURATION], [-3.2, -3.2]]))))
    add = builder.AddSystem(Adder(2, 2))
    builder.Connect(plant.get_state_output_port(), spring.get_input_port())
    builder.Connect(spring.get_output_port(), add.get_input_port(0))
    builder.Connect(drive.get_output_port(), add.get_input_port(1))
    builder.Connect(add.get_output_port(), plant.get_applied_generalized_force_input_port())
    diagram = builder.Build()
    sim = Simulator(diagram)
    context = plant.GetMyMutableContextFromRoot(sim.get_mutable_context())
    plant.SetVelocities(context, [4., OFFSET*4])
    solver = ResetIntegratorFromFlags(sim, "runge_kutta2", step)
    solver.set_fixed_step_mode(True)
    sim.Initialize()
    return diagram, sim, plant, context


def record(step, duration):
    diagram, sim, plant, context = build(step)
    series = []
    for i in range(round(duration/TRACE_DT) + 1):
        t = i*TRACE_DT
        if i:
            sim.AdvanceTo(t)
        angle, follower = plant.GetPositions(context)
        contact = plant.get_contact_results_output_port().Eval(context)
        if contact.num_point_pair_contacts() > 1:
            raise RuntimeError("More than one active contact in the cam proxy")
        force = (np.linalg.norm(contact.point_pair_contact_info(0).contact_force())
                 if contact.num_point_pair_contacts() else 0.)
        reference = STIFFNESS * max(0., PADDED_RADIUS + OFFSET*math.sin(angle) - FOLLOWER_HEIGHT - follower)
        series.append((t, angle, follower, force, reference))
    return np.array(series)


def run(output, repeats, duration):
    rows = []
    output.mkdir(parents=True, exist_ok=True)
    for step in STEPS:
        data = record(step, duration)
        law_error = float(np.max(np.abs(data[:, 3] - data[:, 4])))
        if law_error > 1e-8:
            raise ValueError(f"Drake cam force law mismatch at {step:g} s: {law_error:g} N")
        path = output / f"drake-cam-{round(step*1e6)}us.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("time", "q0", "q1", "solver_peak_force", "force_law_N"))
            writer.writerows(data)
        prep, integration = [], []
        for _ in range(repeats):
            start = perf_counter()
            diagram, sim, plant, context = build(step)
            ready = perf_counter()
            sim.AdvanceTo(duration)
            prep.append(ready-start)
            integration.append(perf_counter()-ready)
        rows.append({"step_s": step, "setup_s": prep, "online_s": integration,
                     "median_setup_s": median(prep), "median_online_s": median(integration),
                     "max_force_law_error_N": law_error, "trace": path.name})
        print(path.stem, "force law error", law_error, flush=True)
    (output / "summary.json").write_text(json.dumps({"version": version("drake"),
        "duration_s": duration, "repeats": repeats, "runs": rows}, indent=2)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--duration", type=float, default=DURATION)
    args = parser.parse_args()
    if args.repeats < 1 or not 0. < args.duration <= DURATION:
        parser.error("invalid repeats or duration")
    run(args.output, args.repeats, args.duration)


if __name__ == "__main__":
    main()
