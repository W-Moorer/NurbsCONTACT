"""Read fixed-hinge CPU scenes and convert loads for the GPU integrator."""

import shlex

import numpy as np


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


def equations(work, h=0.000125, duration=3.0, scene=None):
    """Convert the existing mechanism's loads to corrections about nominal joint motion."""
    lines = iter((scene or work / "reducer.scene").read_text().splitlines())
    next(lines)
    n, nb, npairs = map(int, next(lines).split()[-3:])
    coords = np.array([list(map(float, next(lines).split()[1:])) for _ in range(n)])
    bodies, mapping = [], []
    for _ in range(nb):
        body = shlex.split(next(lines))
        j = int(body[13])
        origin = 1000 * np.array(body[3:6], float)
        bodies.append(
            [
                0,
                *map(float, body[6:9]),
                *origin,
                float(body[9]),
                coords[j, 2],
                coords[j, 0],
                coords[j, 3],
                coords[j, 4],
                coords[j, 1],
            ]
        )
        mapping.append(j)
    pairs, stiffness = [], []
    for _ in range(npairs):
        row = next(lines).split()
        pairs.append([int(row[1]), int(row[2])])
        if row[0] not in ("closest", "surface"):
            raise ValueError(f"unsupported contact pair {row[0]}")
        stiffness.append(float(row[3] if row[0] == "surface" else row[4]))
    if next(lines) != "drives 0":
        raise ValueError("Expected free shafts")
    t = np.arange(round(duration / h) + 1) * h
    loads = coords[:, 6] + coords[:, 3] * coords[:, 5] - coords[:, 4] * coords[:, 2] - t[:, None] * coords[:, 2] * coords[:, 3]
    for _ in range(int(next(lines).split()[1])):
        _, j, count = next(lines).split()
        samples = np.array([list(map(float, next(lines).split())) for _ in range(int(count))])
        if samples[0, 0] > 0 or samples[-1, 0] < t[-1] - 1e-12:
            raise ValueError("load history does not span the integration interval")
        loads[:, int(j)] += np.interp(t, samples[:, 0], samples[:, 1])
    np.savez_compressed(
        work / "assembly.npz",
        coord=coords[:, [0, 3, 4, 1]],
        bodies=bodies,
        mapping=mapping,
        pairs=pairs,
        stiffness=stiffness,
        loads=loads,
        h=h,
    )
