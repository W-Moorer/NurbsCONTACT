"""Rigid-body comparison using native MuJoCo, Drake and STEP/HHT contact."""

import argparse
import csv
from dataclasses import dataclass
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
from statistics import median
import subprocess
from time import perf_counter

import numpy as np


LAYER = .012
STIFFNESS = 5000.
STANDOFF = .018
SPEED = .45
DURATION = .16
CASES = ("plane", "spheres")
REPEATS = 7


@dataclass(frozen=True)
class Fixtures:
    radius: float
    steps: tuple[float, ...]
    centered_pair: bool = False


LEGACY = Fixtures(.035, (.001, .0005, .00025))
AES = Fixtures(.055, (.0005, .00025, .000125), centered_pair=True)


def case_values(case, fixtures):
    if case=="plane":
        return fixtures.radius+LAYER+STANDOFF,1.,1
    return 2*fixtures.radius+LAYER+STANDOFF,.5,2


def initial_velocity(count, fixtures):
    if count == 2 and fixtures.centered_pair:
        return np.array([SPEED / 2, -SPEED / 2])
    return np.array([0.] * (count - 1) + [-SPEED])


def exact(case,time,fixtures):
    _,mass,count = case_values(case,fixtures)
    omega = math.sqrt(STIFFNESS/mass)
    entry = STANDOFF/SPEED
    exit_ = entry+math.pi/omega
    during = -STANDOFF-SPEED/omega*np.sin(omega*(time-entry))
    displacement = np.where(time<=entry,-SPEED*time,
                            np.where(time<=exit_,during,-STANDOFF+SPEED*(time-exit_)))
    force = STIFFNESS*np.maximum(0,-STANDOFF-displacement)
    if count==1:
        q = displacement[:,None]
    else:
        center = np.mean(initial_velocity(count, fixtures))*time
        q = np.column_stack((center-.5*displacement,center+.5*displacement))
    return q,force


def mujoco_runner(case,step,record,integrator,fixtures):
    import mujoco

    setup_start = perf_counter()
    z,mass,count = case_values(case,fixtures)
    padding = LAYER/2 if count==2 else 0.
    geom_radius = fixtures.radius+padding

    def body(name,height):
        return (f'<body name="{name}" pos="0 0 {height:.17g}"><joint type="slide" axis="0 0 1"/>'
                f'<geom name="{name}" type="sphere" size="{geom_radius:.17g}" mass="1"/>'
                '</body>')

    lower = body("lower",0) if count==2 else f'<geom name="lower" type="plane" size="1 1 .1" pos="0 0 {LAYER}"/>'
    xml = f"""<mujoco>
<option timestep="{step:.17g}" gravity="0 0 0" integrator="{integrator}" solver="Newton" tolerance="1e-12"/>
<worldbody>{lower}{body("upper",z)}</worldbody>
<contact><pair geom1="lower" geom2="upper" condim="1" margin="0" gap="0"
 solref="{-STIFFNESS/mass:.17g} 0" solimp="0.9 0.9 0.001 0.5 2"/></contact>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    steps = round(DURATION/step)
    data.qvel[:] = initial_velocity(count, fixtures)
    mujoco.mj_forward(model,data)
    if not record:
        t0 = perf_counter()
        mujoco.mj_step(model,data,nstep=steps)
        return t0-setup_start,perf_counter()-t0
    positions = np.empty((steps+1,count))
    forces = np.empty(steps+1)
    sample_force = np.zeros(6)
    for i in range(steps+1):
        mujoco.mj_forward(model,data)
        positions[i] = data.qpos
        forces[i] = 0.
        if data.ncon:
            if data.ncon!=1:
                raise ValueError("expected at most one active MuJoCo contact")
            mujoco.mj_contactForce(model,data,0,sample_force)
            forces[i] = sample_force[0]
        if i<steps:
            mujoco.mj_step(model,data)
    return positions,forces


def drake_model(case,step,fixtures):
    from pydrake.geometry import AddContactMaterial, HalfSpace, ProximityProperties, Sphere
    from pydrake.math import RigidTransform
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, ContactModel, CoulombFriction, DiscreteContactApproximation
    from pydrake.multibody.tree import PrismaticJoint, SpatialInertia, UnitInertia
    from pydrake.systems.analysis import Simulator
    from pydrake.systems.framework import DiagramBuilder

    z,_,count = case_values(case,fixtures)
    padding = LAYER/2 if count==2 else 0.
    builder = DiagramBuilder()
    plant,_ = AddMultibodyPlantSceneGraph(builder,step)
    plant.mutable_gravity_field().set_gravity_vector([0,0,0])
    plant.set_contact_model(ContactModel.kPoint)
    props = ProximityProperties()
    AddContactMaterial(dissipation=0.,point_stiffness=2*STIFFNESS,friction=CoulombFriction(0.,0.),properties=props)
    if step > 0:
        plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSap)
        props.AddProperty("material", "relaxation_time", 0.)

    def add_ball(name):
        body = plant.AddRigidBody(name,SpatialInertia(1.,[0,0,0],UnitInertia.SolidSphere(fixtures.radius)))
        joint = plant.AddJoint(PrismaticJoint(name,plant.world_frame(),body.body_frame(),[0,0,1]))
        plant.RegisterCollisionGeometry(body,RigidTransform(),Sphere(fixtures.radius+padding),name,props)
        return joint

    if count==1:
        plant.RegisterCollisionGeometry(plant.world_body(),RigidTransform([0,0,LAYER]),HalfSpace(),"lower",props)
    else:
        add_ball("lower")
    add_ball("upper")
    plant.Finalize()
    diagram = builder.Build()
    sim = Simulator(diagram)
    root = sim.get_mutable_context()
    pc = plant.GetMyMutableContextFromRoot(root)
    reference_position = np.zeros(count)
    reference_position[-1] = z
    plant.SetPositions(pc,reference_position)
    plant.SetVelocities(pc,initial_velocity(count, fixtures))
    return plant, sim, pc, reference_position


def drake_runner(case,step,record,integrator,fixtures):
    from pydrake.systems.analysis import ResetIntegratorFromFlags

    setup_start = perf_counter()
    plant,sim,pc,reference_position = drake_model(case,0.,fixtures)
    count = len(reference_position)
    solver = ResetIntegratorFromFlags(sim,integrator,step)
    solver.set_fixed_step_mode(True)
    sim.Initialize()
    if not record:
        t0 = perf_counter()
        sim.AdvanceTo(DURATION)
        return t0-setup_start,perf_counter()-t0
    steps = round(DURATION/step)
    positions = np.empty((steps+1,count))
    forces = np.empty(steps+1)
    for i in range(steps+1):
        if i:
            sim.AdvanceTo(i*step)
        positions[i] = plant.GetPositions(pc)-reference_position
        contact = plant.get_contact_results_output_port().Eval(pc)
        if contact.num_point_pair_contacts()>1:
            raise ValueError("expected at most one active Drake contact")
        forces[i] = (np.linalg.norm(contact.point_pair_contact_info(0).contact_force())
                     if contact.num_point_pair_contacts() else 0.)
    return positions,forces


def nc_scene(folder,case,fixtures):
    z,_,count = case_values(case,fixtures)
    if count==1:
        bodies = [
            f'body "floor.step" 0 0 0 {LAYER} 0 0 1 0 0 0 0 -1 0 0 0 0',
            f'body "sphere.step" 0 0 0 {z} 0 0 1 0 0 0 0 -1 0 0 0 1 0 0 0 1',
        ]
    else:
        bodies = [
            'body "offset-sphere.step" 0 0 0 0 0 0 1 0 0 0 0 -1 0 0 0 1 0 0 0 1',
            f'body "offset-sphere.step" 0 0 0 {z} 0 0 1 0 0 0 0 -1 0 0 0 1 1 0 0 1',
        ]
    coord = [f"coordinate 1 0 {speed} 0 0 0 0" for speed in initial_velocity(count, fixtures)]
    path = folder/f"{case}.scene"
    path.write_text("\n".join(("nc-contact-v1",f"gravity 0 0 0 {count} 2 1",*coord,*bodies,
                               f"probe 0 1 world 0 0 0 0 0 0 0 0 1 {STIFFNESS} 1 0 0 .001 .020"))+"\n",
                    encoding="ascii")
    return path


def nc_models(exe,folder,fixtures):
    for name,args in (("floor",("box",1000,1000,10,-5)),
                      ("sphere",("sphere",fixtures.radius*1000)),
                      ("offset-sphere",("sphere",(fixtures.radius+LAYER/2)*1000))):
        subprocess.run([str(exe),"make",*map(str,args),str(folder/f"{name}.step")],
                       check=True,capture_output=True)


def nc_runner(exe,folder,case,step,record,_,fixtures,deflection=0.):
    scene = nc_scene(folder,case,fixtures)
    prefix = folder/"nc"
    steps = round(DURATION/step)
    completed = subprocess.run([str(exe),str(scene),str(deflection),str(step),str(DURATION),
                                "1" if record else str(steps),str(prefix)],capture_output=True,text=True)
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    with prefix.with_suffix(".stats.csv").open(newline="") as file:
        stats = next(csv.DictReader(file))
    if not record:
        return float(stats["setup_s"]),float(stats["wall_s"])
    data = np.genfromtxt(prefix.with_suffix(".csv"),delimiter=",",names=True)
    count = case_values(case,fixtures)[2]
    return np.column_stack([data[f"q{i}"] for i in range(count)]),data["p0_normal_N"]


def statistics(case,step,positions,forces,timings,fixtures):
    time = step*np.arange(round(DURATION/step)+1)
    reference,expected_force = exact(case,time,fixtures)
    coordinate_error = positions-reference
    spring_force = STIFFNESS*np.maximum(0,-STANDOFF-(positions[:,-1]-(positions[:,0] if positions.shape[1]>1 else 0)))
    law_error = float(np.max(np.abs(spring_force-forces)))
    if not np.isfinite(positions).all() or not math.isfinite(law_error) or law_error>2e-6:
        raise ValueError(f"engine does not reproduce the specified one-contact law: {law_error:g} N")
    setup, integrate = (list(values) for values in zip(*timings, strict=True))
    return {
        "case":case,"step_s":step,"median_integrate_s":median(integrate),"times_s":integrate,
        "median_setup_s":median(setup),"setup_times_s":setup,
        "position_rms_mm":float(1000*np.sqrt(np.mean(coordinate_error**2))),
        "maximum_position_error_mm":float(1000*np.max(np.abs(coordinate_error))),
        "force_rms_N":float(np.sqrt(np.mean((forces-expected_force)**2))),
        "maximum_force_law_residual_N":law_error,
    }, (time,positions,forces,reference,expected_force)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--backend",choices=("mujoco","drake","nurbs"),required=True)
    parser.add_argument("--nc-contact",type=Path)
    parser.add_argument("--aes-fixtures",action="store_true",help="use 55 mm spheres and a stationary pair center of mass")
    args = parser.parse_args()
    backend = args.backend
    fixtures = AES if args.aes_fixtures else LEGACY
    args.output.mkdir(parents=True,exist_ok=True)
    if backend=="mujoco":
        runner,integrators,release = mujoco_runner,("implicit","RK4"),version("mujoco")
    elif backend=="drake":
        runner,integrators,release = drake_runner,("runge_kutta3",),version("drake")
    else:
        if args.nc_contact is None:
            parser.error("--nc-contact is required for the native backend")
        exe = args.nc_contact.resolve()
        nc_models(exe,args.output,fixtures)
        runner = lambda *params:nc_runner(exe,args.output,*params)
        integrators,release = ("HHT",),"workspace build"
    results = []
    for case in CASES:
        for integrator in integrators:
            for step in fixtures.steps:
                positions,forces = runner(case,step,True,integrator,fixtures)
                timings = [runner(case,step,False,integrator,fixtures) for i in range(REPEATS)]
                result,trace = statistics(case,step,positions,forces,timings,fixtures)
                result.update({"engine":backend,"integrator":integrator})
                results.append(result)
                time,positions,forces,reference,expected_force = trace
                count = positions.shape[1]
                columns = ["time_s",*(f"q{i}_m" for i in range(count)),"force_N",
                           *(f"ref_q{i}_m" for i in range(count)),"ref_force_N"]
                np.savetxt(args.output/f"{case}-{integrator}-{step*1e6:.0f}us.csv",
                           np.column_stack((time,positions,forces,reference,expected_force)),
                           delimiter=",",comments="",header=",".join(columns),fmt="%.17g")
                print(backend,integrator,case,step,result["position_rms_mm"],flush=True)
    report = {
        "engine":backend,"version":release,"platform":platform.platform(),"host":platform.node(),
        "layer_m":LAYER,"stiffness_N_m":STIFFNESS,"mass_each_kg":1.,"initial_speed_m_s":SPEED,
        "radius_m":fixtures.radius,"centered_pair":fixtures.centered_pair,
        "duration_s":DURATION,"stand_off_m":STANDOFF,"repetitions":REPEATS,"runs":results,
    }
    (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
