"""Check full spatial rigid-body motion through the nc_contact scene interface."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np
from scipy.spatial.transform import Rotation


SPHERE = Path(__file__).resolve().parents[2] / "results/engine-comparison/nurbs/offset-sphere.step"
MASS = 1.0
RADIUS = 0.041
STIFFNESS = 5000.0
SPEED = 0.45


def fields(values):
    return " ".join(format(float(v), ".17g") for v in values)


def body(origin, axis=(0, 0, 1), angle=0.0, com=(0, 0, 0), mass=MASS):
    return f'body "{SPHERE.name}" {mass} {fields(origin)} {fields(axis)} {angle} {fields(com)} -1 0 0 0 0'


def free(index, inertia, velocity, omega):
    off_diagonal = inertia[[0, 0, 1], [1, 2, 2]]
    return (
        f"free {index} {fields(inertia.diagonal())} {fields(off_diagonal)} {fields(velocity)} {fields(omega)}"
    )


def vec(trace, names):
    return np.column_stack([trace[name] for name in names])


def attitude(trace, index):
    return Rotation.from_quat(vec(trace, [f"b{index}_{k}" for k in ("qx", "qy", "qz", "qw")]))


def largest_norm(array):
    return float(np.linalg.norm(array, axis=1).max())


def top():
    axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14)
    transverse, parallel = 0.00065, 0.00086
    inertia = transverse * np.eye(3) + (parallel - transverse) * np.outer(axis, axis)
    omega = np.array([2.1, -1.4, 4.3])
    velocity = np.array([0.21, 0.17, -0.09])
    gravity = np.array([0.5, -1.0, -9.81])
    com = np.array([0.008, -0.003, 0.004])
    sensor = np.array([0.02, 0.01, -0.01])
    rotation_axis = np.array([2.0, -1.0, 0.5])
    rotation_axis /= np.linalg.norm(rotation_axis)
    initial = Rotation.from_rotvec(0.7 * rotation_axis)
    scene = [
        "nc-contact-v2",
        f"gravity {fields(gravity)} 0 2 1",
        body((0, 0, 0), rotation_axis, 0.7, com),
        body((20, 0, 0), mass=0.0),
        "surface 0 1 5000 1 0 0 .001 .02",
        "drives 0",
        "free-bodies 1",
        free(0, inertia, velocity, omega),
        "accelerometers 1",
        f"accelerometer 0 {fields(sensor)} 1 0 0",
    ]

    def compare(trace):
        t = trace["time_s"][:, None]
        rotation = attitude(trace, 0)
        momentum = initial.apply(inertia @ omega)
        rate = (parallel - transverse) / transverse * (axis @ omega)
        # Euler's symmetric-top solution in arbitrary initial orientation.
        reference = (
            Rotation.from_rotvec(t * momentum / transverse) * initial * Rotation.from_rotvec(-t * rate * axis)
        )
        omega_body = Rotation.from_rotvec(t * rate * axis).apply(omega)
        w = vec(trace, [f"b0_w{k}_rad_s" for k in "xyz"])
        measured_l = rotation.apply(rotation.inv().apply(w) @ inertia.T)
        center = initial.apply(com) + t * velocity + 0.5 * t**2 * gravity
        q = vec(trace, ["q0", "q1", "q2"])
        lever = sensor - com
        alpha = rate * np.cross(axis, omega_body)
        sensor_ref = (np.cross(alpha, lever) + np.cross(omega_body, np.cross(omega_body, lever)))[:, 0]
        return dict(
            max_attitude_error_rad=float((reference.inv() * rotation).magnitude().max()),
            max_world_omega_error_rad_s=largest_norm(w - reference.apply(omega_body)),
            max_world_angular_momentum_drift=largest_norm(measured_l - momentum),
            max_center_position_error_m=largest_norm(q - center),
            max_specific_force_error_m_s2=float(np.abs(trace["s0_a_m_s2"] - sensor_ref).max()),
        )

    return scene, 2.4, compare


def collision(eccentric, solver="penalty"):
    moment = 0.4 * MASS * RADIUS**2
    inertia = moment * np.eye(3)
    offset = (0.004, 0, 0) if eccentric else (0, 0, 0)
    lateral = 0.012 if eccentric else 0.0
    scene = [
        "nc-contact-v2",
        "gravity 0 0 0 0 2 1",
        body((0, 0, 0), com=offset),
        body((lateral, 0, 0.1), axis=(0, 1, 0), angle=0.33 if eccentric else 0.0),
        f"surface 0 1 {STIFFNESS} 1 0 {.18 if eccentric else 0.} .001 .020",
        "drives 0",
        "free-bodies 2",
        free(0, inertia, (0, 0, 0), (0, 0, 0)),
        free(1, inertia, (0, 0, -SPEED), (0, 0, 0)),
    ]

    def compare(trace):
        t, force = trace["time_s"], trace["p0_normal_N"]
        compliant = solver in ("penalty", "sap")
        if force[0] or (compliant and force[-1]) or not force.max() > 0:
            raise ValueError("expected a completed impact")
        p, l, energy = np.zeros((len(t), 3)), np.zeros((len(t), 3)), np.zeros(len(t))
        spin, vel = [], []
        for j in (0, 1):
            center = vec(trace, [f"q{7*j+k}" for k in range(3)])
            v = vec(trace, [f"v{7*j+k}" for k in range(3)])
            w = vec(trace, [f"b{j}_w{k}_rad_s" for k in "xyz"])
            p += MASS * v
            l += np.cross(center, MASS * v) + moment * w
            energy += 0.5 * (MASS * (v * v).sum(axis=1) + moment * (w * w).sum(axis=1))
            spin.append(w[-1].tolist())
            vel.append(v[-1].tolist())
        result = dict(
            max_linear_momentum_drift_kg_m_s=largest_norm(p - p[0]),
            max_angular_momentum_drift_kg_m2_s=largest_norm(l - l[0]),
            initial_kinetic_energy_J=energy[0],
            final_kinetic_energy_J=energy[-1],
            final_velocities_m_s=vel,
            final_angular_velocities_rad_s=spin,
            peak_normal_force_N=float(force.max()),
            normal_impulse_Ns=float(np.trapezoid(force, t) if solver=="penalty" else np.diff(t) @ force[1:]),
        )
        if not eccentric and not compliant:
            result.update(reference_impulse_Ns=MASS*SPEED/2,
                          reference_final_axial_velocities_m_s=[-SPEED/2, -SPEED/2])
        if not eccentric and compliant:
            omega = np.sqrt(STIFFNESS / (MASS / 2))
            entry = (0.1 - 2 * RADIUS) / SPEED
            phase = np.clip(omega * (t - entry), 0.0, np.pi)
            force_ref = STIFFNESS * SPEED / omega * np.sin(phase)
            result.update(
                reference_entry_s=entry,
                reference_peak_N=STIFFNESS * SPEED / omega,
                reference_impulse_Ns=MASS * SPEED,
                reference_final_axial_velocities_m_s=[-SPEED, 0.0],
                force_rmse_N=float(np.sqrt(np.mean((force - force_ref) ** 2))),
            )
        return result

    return scene, 0.16, compare


def run(args):
    scene, duration, compare = top() if args.case == "top" else collision(args.case == "eccentric", args.solver)
    if getattr(args,"duration",None) is not None:
        duration=args.duration
    args.work.mkdir(parents=True, exist_ok=True)
    geometry = {"step_sha256": hashlib.sha256(SPHERE.read_bytes()).hexdigest()}
    mesh_args = []
    if args.method.startswith("mesh-"):
        if args.mesh_dir is None:
            raise ValueError("provide the Gmsh directory for a mesh method")
        mesh = (args.mesh_dir / SPHERE.with_suffix(".msh").name).resolve()
        quality = json.loads(mesh.with_suffix(".quality.json").read_text())
        geometry["mesh_sha256"] = hashlib.sha256(mesh.read_bytes()).hexdigest()
        if any(quality[k] != v for k,v in geometry.items()):
            raise ValueError("the geometry differs from the mesh quality report")
        mesh_args = ["--mesh-dir", mesh.parent]
        if getattr(args,"mesh_region_scope",None):
            mesh_args+=["--mesh-region-scope",args.mesh_region_scope]
    else:
        destination = args.work / SPHERE.name
        if destination.resolve() != SPHERE:
            shutil.copyfile(SPHERE, destination)
    model = args.work / "model.scene"
    model.write_text("\n".join(scene) + "\n", encoding="utf-8")
    prefix = args.work / "response"
    cmd = [
        args.exe.resolve(),
        model,
        "0",
        str(args.dt),
        str(duration),
        "1",
        prefix,
        "--method",
        args.method,
        *mesh_args,
        "--solver",
        args.solver,
    ]
    if args.event_tol:
        cmd.extend(["--event-tol-s", str(args.event_tol)])
    if getattr(args,"geometry_update",None):
        cmd.extend(["--geometry-update",args.geometry_update])
    start = time.perf_counter()
    subprocess.run(cmd, check=True, timeout=1800, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    trace = np.genfromtxt(prefix.with_suffix(".csv"), delimiter=",", names=True)
    if abs(trace["time_s"][-1] - duration) > 1e-9 or len(trace) != round(duration / args.dt) + 1:
        raise ValueError("incomplete trace")
    with prefix.with_suffix(".stats.csv").open() as file:
        stats = next(csv.DictReader(file))
    return dict(
        case=args.case,
        method=args.method,
        solver=args.solver,
        dt_s=args.dt,
        duration_s=duration,
        mesh_region_scope=getattr(args,"mesh_region_scope",None),
        geometry_update=getattr(args,"geometry_update",None),
        event_tolerance_s=float(stats.get("event_tolerance_s",args.event_tol)),
        **geometry,
        scene_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        trace_sha256=hashlib.sha256(prefix.with_suffix(".csv").read_bytes()).hexdigest(),
        frames=len(trace),
        process_s=elapsed,
        integration_s=float(stats["wall_s"]),
        subdivisions=int(stats["subdivisions"]),
        evaluations=int(stats["force_evaluations"]),
        step_checks=int(stats["ccd_checks"]),
        event_rejections=int(stats["ccd_rejected"]),
        **{k:int(stats[k]) for k in ("cad_boolean_queries","cad_event_distances","chen_patch_queries",
                                     "chen_confirmed_candidates","tdib_cells","tdib_candidates","tdib_unresolved") if k in stats},
        **{k:float(stats[k]) for k in ("impulse_residual_m_s",) if k in stats},
        **{k:int(stats[k]) for k in ("impulse_solves","impulse_regions","impulse_iterations","ccd_refinements",
                                     "ccd_refined_clear","ccd_unresolved") if k in stats},
        **compare(trace),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe", type=Path)
    parser.add_argument("mesh_dir", nargs="?", type=Path, help="required for a mesh method")
    for name in ("work", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--case", choices=["top", "head-on", "eccentric"], required=True)
    parser.add_argument("--method", choices=["mesh-aabb", "mesh-obb", "mesh-sap","cad-occt","chen-2024","chen-2026"], default="mesh-obb")
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--duration",type=float)
    parser.add_argument("--mesh-region-scope",choices=("local","full","fcl","finite","contour-ray"))
    parser.add_argument("--geometry-update",choices=("newton","predictor"))
    parser.add_argument("--solver", choices=("penalty","sap","lagrange","augmented-lagrange","lcp","ncp","soccp"),
                        default="penalty")
    parser.add_argument("--event-tol", type=float, default=0.0)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started=time.perf_counter()
    try:
        report = run(args)
    except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
        message=error.stderr or error.stdout or str(error)
        if isinstance(message,bytes): message=message.decode(errors='replace')
        match=re.search(r'\bt=([.0-9eE+-]+)',message)
        failure=dict(case=args.case,method=args.method,mesh_region_scope=args.mesh_region_scope,
                     solver=args.solver,dt_s=args.dt,requested_s=args.duration,
                     status="timeout" if isinstance(error,subprocess.TimeoutExpired) else "failed",
                     runner_elapsed_s=time.perf_counter()-started,message=message.strip(),
                     failure_time_s=float(match[1]) if match else None)
        args.output.write_text(json.dumps(failure,indent=2)+'\n',encoding='utf-8')
        raise
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
