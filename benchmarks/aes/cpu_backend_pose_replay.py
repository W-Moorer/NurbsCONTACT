"""Replay the CPU reducer's accepted hinge states through its two mesh contact outputs."""

import argparse
import csv
import json
from pathlib import Path
import shlex
import subprocess

import numpy as np

from cpu_backend_run import digest


def read_model(scene):
    bodies, pairs = [], []
    for line in scene.read_text().splitlines():
        if line.startswith("body "):
            fields = shlex.split(line)
            source = (scene.parent / fields[1]).resolve()
            origin = np.array(fields[3:6], dtype=float)
            axis = np.array(fields[6:9], dtype=float)
            if (len(fields) != 18 or fields[17] != "0" or not np.array_equal(origin, np.array(fields[14:17], dtype=float))
                    or np.linalg.norm(axis) != 1 or int(fields[13]) < 0):
                raise ValueError("This diagnostic expects a single centered revolute joint per body")
            bodies.append((source, origin, axis, float(fields[9]), int(fields[13])))
        if line.startswith("surface "):
            _, a, b, k, exponent, damping, friction, _, limit = line.split()
            if (float(exponent), float(damping), float(friction)) != (1., 0., 0.):
                raise ValueError("Force replay requires linear undamped frictionless contact")
            pairs.append((int(a), int(b), float(k), float(limit)))
    return bodies, pairs


def world_poses(body, progress):
    _, position, axis, initial, coordinate = body
    half = (initial + progress[f"q{coordinate}"]) / 2
    return np.column_stack((np.broadcast_to(position, (len(half), 3)), np.cos(half), np.sin(half)[:, None]*axis))


def plane_witness(exe, meshes, pose, target_depth):
    """Locate the FCL candidate and test its orthogonal foot on the opposite finite facet."""
    import meshio
    from scipy.spatial.transform import Rotation

    vertices, faces = [], []
    for i,path in enumerate(meshes):
        frame = pose[i*7:i*7+7]
        mesh = meshio.read(path)
        rotation = Rotation.from_quat(frame[[4, 5, 6, 3]])
        vertices.append(rotation.apply(mesh.points) + frame[:3])
        faces.append(mesh.get_cells_type("triangle"))
    run = subprocess.run([exe, "--method", "mesh-obb", *meshes], text=True, check=True, capture_output=True,
                         input=" ".join(f"{x:.17g}" for x in pose)+"\n", timeout=1800)
    candidates = []
    for _, _, i, j, _ in json.loads(run.stdout)["hits"]:
        triangles = [vertices[0][faces[0][i]], vertices[1][faces[1][j]]]
        sides = []
        for triangle, other in (triangles, triangles[::-1]):
            edges = triangle[1:] - triangle[0]
            normal = np.cross(*edges)
            normal /= np.linalg.norm(normal)
            depths = (triangle[0] - other) @ normal
            k = depths.argmax()
            depth = depths[k]
            foot = other[k] + depth*normal
            u,v = np.linalg.lstsq(edges.T, foot - triangle[0], rcond=None)[0]
            bary = np.array([1-u-v, u, v])
            gaps = []
            for n in range(3):
                a, b = triangle[n], triangle[(n+1)%3]
                length = b-a
                fraction = np.clip(np.dot(foot-a,length)/np.dot(length,length),0,1)
                gaps.append(np.linalg.norm(foot-a-fraction*length))
            sides.append(dict(depth_um=depth*1e6, foot_barycentric=bary.tolist(),
                              foot_outside_um=0. if min(bary)>=0 else min(gaps)*1e6))
        selected = min(range(2), key=lambda n: sides[n]["depth_um"])
        candidates.append(dict(triangles_zero_based=[i,j], opposite_surface=selected, **sides[selected]))
    result = min(candidates, key=lambda c: abs(c["depth_um"]-target_depth))
    if abs(result["depth_um"]-target_depth)>1e-5:
        raise ValueError("Could not match the native maximum depth at the diagnostic pose")
    return result


def replay(exe, scene, trace, directory):
    directory.mkdir(parents=True, exist_ok=True)
    bodies, pairs = read_model(scene)
    report = dict(scene_sha256=digest(scene), progress_sha256=digest(trace), mesh_executable_sha256=digest(exe),
                  parts=[], queries=[])
    for step, *_ in bodies:
        mesh = step.with_suffix(".msh")
        quality = json.loads(step.with_suffix(".quality.json").read_text())
        step_hash, mesh_hash = digest(step), digest(mesh)
        if mesh_hash != quality["mesh_sha256"] or step_hash != quality["step_sha256"]:
            raise ValueError(f"Mesh audit does not match {step.stem}")
        report["parts"].append(dict(asset=step.stem, step_sha256=step_hash, mesh_sha256=mesh_hash,
                                    triangles=quality["triangles"]))
    states = np.genfromtxt(trace, delimiter=",", names=True, ndmin=1)
    output = []
    for stage, (a,b,k,limit) in enumerate(pairs, start=1):
        data = np.column_stack([world_poses(bodies[i], states) for i in (a,b)])
        lines = "".join(" ".join(f"{x:.17g}" for x in pose) + "\n" for pose in data)
        meshes = [bodies[i][0].with_suffix(".msh") for i in (a,b)]
        for scope in ("local", "fcl"):
            process = subprocess.run([exe, "--method", "mesh-obb", "--regions", scope, *meshes], input=lines,
                                     text=True, capture_output=True, timeout=1800)
            values = [json.loads(line) for line in process.stdout.splitlines()]
            if process.returncode == 0 and len(values) != len(states):
                raise ValueError("incomplete pose replay")
            report["queries"].append(dict(stage=stage, scope=scope, accepted_poses=len(values),
                                          error=process.stderr.strip() if process.returncode else None))
            for state, response in zip(states, values):
                contacts = np.asarray(response["regions"]).reshape(-1, 7)
                forces = k * contacts[:, 6:7] * contacts[:, 3:6]
                moments = np.cross(contacts[:, :3] - bodies[b][1], forces)
                depths = contacts[:, 6]
                maximum = max(depths, default=0)
                force = k * sum(depths)
                output.append(dict(step=int(state["step"]), time_s=state["time_s"], stage=stage, scope=scope,
                                   regions=len(depths), booleans=response["boolean_queries"], max_depth_um=1e6*maximum,
                                   exceeds_model_limit=int(maximum>limit), sum_normal_force_N=force,
                                   resultant_N=np.linalg.norm(forces.sum(axis=0)),
                                   driven_global_Tz_Nm=moments[:, 2].sum()))
        fcl_rows = [r for r in output if r["stage"]==stage and r["scope"]=="fcl" and r["exceeds_model_limit"]]
        if fcl_rows:
            first = fcl_rows[0]
            pos = int(np.flatnonzero(states["step"] == first["step"])[0])
            example = plane_witness(exe, meshes, data[pos], first["max_depth_um"])
            report.setdefault("first_depth_exceedance", []).append(dict(stage=stage, step=first["step"], **example))
    with (directory / "pose-replay.csv").open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)
    (directory / "pose-replay.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("exe", "scene", "progress", "out"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    replay(a.exe.resolve(), a.scene.resolve(), a.progress.resolve(), a.out.resolve())
