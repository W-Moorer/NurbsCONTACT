"""Independent rigid triangle mesh dynamics with MuJoCo's rigid flex elements."""
import argparse
import json
import multiprocessing as mp
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco as mj
import numpy as np


def numbers(row):
    return " ".join(map(str, np.asarray(row).ravel()))


def make_model(root, deck, parts, radius):
    coord, bodies, mapping = (deck[k] for k in ("coord", "bodies", "mapping"))
    xml = ET.Element("mujoco", model="shared shaft reducer")
    ET.SubElement(xml, "option", timestep=str(float(deck["h"])), gravity="0 0 0", integrator="implicitfast")
    world = ET.SubElement(xml, "worldbody")
    deform = ET.SubElement(xml, "deformable")
    for j, (inertia, spring, damping, _) in enumerate(coord):
        shaft = bodies[np.flatnonzero(mapping == j)[0]]
        center = shaft[4:7] / 1000
        body = ET.SubElement(world, "body", name=f"shaft{j}", pos=numbers(center))
        ET.SubElement(body, "inertial", pos="0 0 0", mass="1", diaginertia=numbers([inertia] * 3))
        ET.SubElement(body, "joint", type="hinge", axis=numbers(shaft[1:4]), stiffness=str(spring),
                      damping=str(damping))
        for i in np.flatnonzero(mapping == j):
            member = bodies[i]
            if not np.array_equal(member[1:4], shaft[1:4]):
                raise ValueError("inconsistent shaft axes")
            stage = np.flatnonzero(np.any(deck["pairs"] == i, axis=1)).item()
            from scipy.spatial.transform import Rotation
            offset = Rotation.from_rotvec(member[1:4] * member[7])
            with np.load(root / parts[i]["archive"]) as f:
                vertices, elements = np.unique(f["tri"].reshape(-1, 3), axis=0, return_inverse=True)
            vertices = (offset.apply(vertices) + member[4:7]) / 1000 - center
            flex = ET.SubElement(deform, "flex", name=f"part{i}", dim="2", radius=str(radius),
                                 body=f"shaft{j}", vertex=numbers(vertices), element=numbers(elements))
            ET.SubElement(flex, "contact", contype=str(1 << stage), conaffinity=str(1 << stage),
                          selfcollide="none", internal="false", condim="1", solref="0.0002 1")
    return mj.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))


def solve(root, output, radius, progress):
    start = time.perf_counter()
    with np.load(root / "assembly.npz") as file:
        deck = dict(file)
    parts = json.loads((root / "models.json").read_text())["parts"]
    model = make_model(root, deck, parts, radius)
    data = mj.MjData(model)
    n = model.nv
    coord, loads, mapping = (deck[k] for k in ("coord", "loads", "mapping"))
    h = float(deck["h"])
    count = len(loads)
    if count != 50001 or abs(h - .0001) > 1e-12:
        raise ValueError("expected a five-second deck at 0.1 ms")
    nominal = [deck["bodies"][np.flatnonzero(mapping == i)[0], 8] for i in range(n)]
    data.qpos[:] = coord[:, 3]
    data.qvel[:] = nominal
    pairs = deck["pairs"]
    flex_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_FLEX, f"part{i}") for i in range(len(parts))]
    stages = {tuple(sorted((flex_ids[a], flex_ids[b]))): i for i, (a, b) in enumerate(pairs)}
    trace = np.empty((count, 2*n + 2*len(pairs)))
    wrench = np.empty(6)
    mj.set_mjcb_time(time.perf_counter)
    tic = time.perf_counter()
    for i in range(count):
        if abs(data.time - i*h) > 1e-9:
            raise RuntimeError(f"MuJoCo reset at step {i}")
        trace[i, :n], trace[i, n:2*n] = data.qpos, data.qvel
        data.qfrc_applied[:] = loads[i] + (coord[:, 2] + i*h*coord[:, 1]) * nominal
        if i+1 < count:
            mj.mj_step(model, data)
        else:
            mj.mj_forward(model, data)
        row = trace[i, 2*n:]
        row.fill(0)
        for k, contact in enumerate(data.contact):
            if contact.efc_address < 0:
                continue
            stage = stages[tuple(sorted(contact.flex))]
            mj.mj_contactForce(model, data, k, wrench)
            row[2*stage] += wrench[0]
            row[2*stage+1] += 1
        if i and i%1000 == 0:
            progress.value=i
            print(i, round(time.perf_counter()-tic, 2), "contacts",data.ncon,"shaft speeds",data.qvel,flush=True)
    online = time.perf_counter()-tic
    if not np.isfinite(trace).all() or not np.all(trace[:, 2*n::2].max(axis=0)>0):
        raise ValueError("invalid or unloaded response")
    np.save(output.with_suffix(".npy"), trace)
    timers = {name: float(data.timer[value].duration) for name, value in mj.mjtTimer.__members__.items()
              if name.startswith("mjTIMER_") and data.timer[value].number}
    summary = dict(engine="MuJoCo", version=mj.__version__, method="rigid flex triangle mesh",status="complete",
                   timestep_s=h, steps=count-1, physical_s=(count-1)*h, online_s=online, setup_s=tic-start,
                   triangle_radius_m=radius, pairs=pairs.tolist(), rigid_flex=model.flex_rigid.tolist(),
                   normal_force_peak_N=trace[:, 2*n::2].max(axis=0).tolist(),
                   warnings=[int(w.number) for w in data.warning], cumulative_timers_s=timers)
    output.write_text(json.dumps(summary, indent=2)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("deck", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--radius-m", type=float, default=2e-6)
    args = p.parse_args()
    context = mp.get_context("spawn")
    progress = context.Value("i", 0)
    job = context.Process(target=solve, args=(args.deck, args.output, args.radius_m, progress))
    start = time.perf_counter()
    job.start()
    job.join(1800)
    if job.is_alive():
        job.terminate()
        job.join()
        status = "timeout"
    elif job.exitcode:
        status = "solver_failure"
    else:
        raise SystemExit(0)
    args.output.write_text(json.dumps(dict(status=status, engine="MuJoCo", version=mj.__version__,
        method="rigid flex triangle mesh", requested_s=5., step_s=.0001, cap_s=1800,
        reported_steps=progress.value, process_s=time.perf_counter()-start), indent=2)+"\n")
    raise SystemExit(124 if status=="timeout" else 1)
