"""Time live STEP line discovery, GPU wrenches and implicit HHT on the two-shaft spur deck."""

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import shlex
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

import cupy as cp
import numpy as np

from gpu_face_candidates import KERNEL as BOX_KERNEL
from gpu_line_wrench import CUDA, POINT_WRENCH
from gpu_mesh_line import CUDA as MESH_CUDA, device_bvhs, read_full_mesh
from pair_dynamics import gears


def scene(models, duration):
    # Use the same prescribed loads, initial state and bearing parameters as nc_contact.
    with TemporaryDirectory(dir=models.parent) as folder:
        path = Path(folder) / "spur.scene"
        gears(models, path, duration, 56.0)
        rows = iter(path.read_text().splitlines())
        if next(rows) != "nc-contact-v2" or next(rows).split() != [
            "gravity",
            "0",
            "0",
            "0",
            "2",
            "2",
            "1",
        ]:
            raise ValueError("Expected the two-shaft spur deck")
        coordinates = np.array([list(map(float, next(rows).split()[1:])) for _ in range(2)])
        bodies = [shlex.split(next(rows)) for _ in range(2)]
        files = [(path.parent / body[1]).resolve() for body in bodies]
        centers = np.array([list(map(float, b[3:6])) for b in bodies])
        axes = np.array([float(b[8]) for b in bodies])
        zero = axes * np.array([float(b[9]) for b in bodies])
        if any(b[6:8] != ["0", "0"] for b in bodies) or not np.all(np.abs(axes) == 1):
            raise ValueError("This scene driver requires parallel z hinges")
        cutoff = float(next(rows).split()[3])
        if next(rows) != "drives 0" or next(rows) != "loads 2":
            raise ValueError("Unexpected drive layout")
        loads = []
        for i in range(2):
            tag, idx, count = next(rows).split()
            if tag != "load" or int(idx) != i:
                raise ValueError("Unexpected load order")
            loads.append(np.array([list(map(float, next(rows).split())) for _ in range(int(count))]))
    return files, coordinates, centers[1] - centers[0], axes, zero, cutoff, loads


@contextmanager
def worker(executable, files, cutoff, gpu_screen=False):
    options = ["--external-mask"] if gpu_screen else []
    p = subprocess.Popen(
        [str(executable.resolve()), *map(str, files), str(cutoff), *options],
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        bufsize=1,
    )
    try:
        header = p.stdout.readline().split()
        if len(header) != 4 or header[0] != "READY":
            raise RuntimeError(f"Cannot initialize OCCT region worker: {header}")
        yield p, header[1], list(map(int, header[2:]))
    finally:
        p.stdin.close()
        if p.poll() is None:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        p.stdout.close()


class DeviceScreen:
    def __init__(self, p, counts):
        def read_boxes(count):
            bounds = np.array(
                [list(map(float, p.stdout.readline().split())) for _ in range(count)]
            ).reshape(count, 2, 3)
            return cp.asarray(
                np.column_stack(
                    ((bounds[:, 0] + bounds[:, 1]) * 0.5, (bounds[:, 1] - bounds[:, 0]) * 0.5)
                )
            )

        self.a, self.b = map(read_boxes, counts)
        self.scale = max(1.0, float(cp.max(cp.abs(self.a)).get()), float(cp.max(cp.abs(self.b)).get()))
        self.mask = cp.empty((len(self.b), len(self.a)), dtype=cp.uint8)
        self.kernel = cp.RawKernel(BOX_KERNEL, "candidates")
        self.kernel.compile()

    def candidates(self, center, angle, cutoff):
        c, s = np.cos(angle), np.sin(angle)
        matrix = cp.asarray(
            [[c, -s, 0.0, center[0]], [s, c, 0.0, center[1]], [0.0, 0.0, 1.0, center[2]]]
        )
        tolerance = 64 * np.finfo(float).eps * max(self.scale, np.max(np.abs(center)), cutoff)
        self.kernel(
            (len(self.b),),
            (128,),
            (self.a, np.int32(len(self.a)), self.b, matrix, np.float64(cutoff + tolerance), self.mask),
        )
        return np.flatnonzero(self.mask.get())


def discover(p, center, angle, candidates=None):
    fields = () if candidates is None else (len(candidates), *candidates)
    print(*center, 0, 0, 1, angle, *fields, file=p.stdin, flush=True)
    header = p.stdout.readline().split()
    if len(header) != 8 or header[0] != "LINES":
        raise RuntimeError(f"Invalid region response: {header}")
    count, unresolved, tested, reused = map(int, header[1:5])
    keys, samples, offsets = [], [], [0]
    for _ in range(count):
        a, b, n = map(int, p.stdout.readline().split())
        keys.append((a, b))
        for _ in range(n):
            samples.append(list(map(float, p.stdout.readline().split())))
        offsets.append(len(samples))
    samples = np.asarray(samples, dtype=float).reshape(-1, 7)
    if not np.isfinite(samples).all() or np.any(samples[:, 6] <= 0):
        raise ValueError("Invalid live line quadrature")
    return (
        keys,
        samples,
        np.asarray(offsets, dtype=np.int32),
        dict(
            tested=tested,
            reused=reused,
            region_s=float(header[5]),
            assembly_s=float(header[6]),
            unresolved=unresolved,
            touching_witnesses=int(header[7]),
        ),
    )


class DeviceWrenches:
    def __init__(self, method, source):
        self.method = method
        self.previous = {}
        if method == "cad":
            self.charts, coefficients = {}, []
            size = 0
            with np.load(source) as archive:
                for key in archive.files:
                    if key.startswith("coef_"):
                        coef = archive[key]
                        self.charts[int(key[5:])] = [size, *coef.shape[:2]]
                        coefficients.append(coef.reshape(-1, 3))
                        size += len(coefficients[-1])
            self.geometry = cp.asarray(np.concatenate(coefficients))
            self.kernel = cp.RawKernel(CUDA, "parallel_line")
        else:
            _, self.geometry = device_bvhs({0: read_full_mesh(source)})
            self.kernel = cp.RawKernel(POINT_WRENCH + MESH_CUDA, "mesh_line")
        self.kernel.compile()
        self.last = None

    def __call__(self, keys, samples, offsets, center, cutoff, stiffness):
        n, lines = len(samples), len(keys)
        if not lines:
            self.previous, self.last = {}, None
            return cp.empty((0, 6))
        points = np.column_stack((samples[:, 3:6] * 1000, samples[:, 6]))
        interval = cp.asarray([0, lines], dtype=cp.int32)
        history = [(a, b, j) for (a, b), count in zip(keys, np.diff(offsets)) for j in range(count)]
        prior = cp.asarray(
            [n + self.previous[k] if k in self.previous else -1 for k in history], dtype=cp.int32
        )
        centers = cp.asarray(np.tile(center, (lines, 1)))
        common = (cp.asarray(points), centers, np.float64(cutoff * 1000), np.float64(stiffness))
        w = cp.empty((lines, 6))
        if self.method == "cad":
            uv = cp.empty((n, 3))
            buf = uv if self.last is None else cp.concatenate((uv, self.last))
            params = (
                interval,
                cp.asarray(offsets),
                cp.asarray([self.charts[k[0]] for k in keys], dtype=cp.int32),
                self.geometry,
                *common,
                prior,
                buf,
                w,
            )
            self.kernel((lines,), (128,), params, shared_mem=4 * 6 * 8)
        else:
            ids = cp.empty(n, dtype=cp.int32)
            buf = ids if self.last is None else cp.concatenate((ids, self.last))
            params = (
                interval,
                cp.asarray(offsets),
                cp.zeros(lines, cp.int32),
                *self.geometry,
                common[0],
                centers,
                prior,
                *common[2:],
                buf,
                cp.empty((n, 3)),
                w,
            )
            self.kernel((lines,), (128,), params)
        self.last = buf[:n].copy()
        self.previous = {key: i for i, key in enumerate(history)}
        return w


def reference(samples, offsets, center, cutoff, stiffness):
    """Direct CPU accumulation on the original STEP witnesses in SI."""
    delta = samples[:, 3:6] - samples[:, :3]
    gap = np.linalg.norm(delta, axis=1)
    if np.any(gap <= 0):
        raise ValueError("Undefined line normal")
    f = (stiffness * samples[:, 6] * np.maximum(cutoff - gap, 0) / gap)[:, None] * delta
    point_w = np.column_stack((f, np.cross(samples[:, 3:6] - center, f)))
    return np.asarray([point_w[a:b].sum(axis=0) for a, b in zip(offsets[:-1], offsets[1:])]).reshape(
        -1, 6
    )


def relative_pose(q, zeros, signs, separation):
    angles = zeros + signs * q
    c, s = np.cos(angles[0]), np.sin(angles[0])
    x, y, z = separation
    return np.array([c * x + s * y, -s * x + c * y, z]), angles[1] - angles[0]


def cpu_reference(args):
    """Independent one-step root solve using native witnesses and NumPy force assembly."""
    from scipy.optimize import root

    files, coords, sep, signs, zeros, cutoff, loads = scene(args.models.resolve(), args.duration)
    mass, q, v, spring, damp, rest, bias = coords.T
    h = args.step_us * 1e-6
    alpha = -0.1
    beta, gamma = (1 - alpha) ** 2 / 4, 0.5 - alpha
    with worker(args.worker, files, cutoff) as (p, version, counts):
        calls = []

        def force(t, pos, vel):
            center, angle = relative_pose(pos, zeros, signs, sep)
            keys, samples, offsets, info = discover(p, center, angle)
            if info["unresolved"] or info["touching_witnesses"]:
                raise ArithmeticError("CPU reference geometry unresolved")
            calls.append(dict(time_s=t, faces=keys, **info))
            w = reference(samples, offsets, center, cutoff, args.stiffness).sum(axis=0)
            moments = np.array([-w[5] - np.cross(center, w[:3])[2], w[5]])
            external = [np.interp(t, table[:, 0], table[:, 1]) for table in loads]
            return bias + external - spring * (pos - rest) - damp * vel + signs * moments

        f0 = force(0.0, q, v)
        a0 = f0 / mass

        def state(acc):
            return (
                q + h * v + h * h * ((0.5 - beta) * a0 + beta * acc),
                v + h * ((1 - gamma) * a0 + gamma * acc),
            )

        cache = {}

        def residual(acc):
            key = tuple(acc)
            if key not in cache:
                cache[key] = mass * acc - (1 + alpha) * force(h, *state(acc)) + alpha * f0
            return cache[key]

        solved = root(residual, a0, options={"eps": 1e-10, "xtol": 1e-8})
        acc = solved.x
        error = float(np.max(np.abs(residual(acc))))
        pos, vel = state(acc)
        if not np.isfinite(error) or error > 1e-7 * (1 + np.max(np.abs(f0))):
            raise ArithmeticError(f"CPU reference residual {error}: {solved.message}")
    return dict(
        method="CPU OCCT line forces and SciPy HHT residual root",
        occt=version,
        face_counts=counts,
        dt_s=h,
        q=pos.tolist(),
        v=vel.tolist(),
        a=acc.tolist(),
        residual_Nm=error,
        calls=calls,
        stiffness_N_per_m2=args.stiffness,
    )


def run(args):
    start = perf_counter()
    files, coords, separation, signs, zeros, cutoff, loads = scene(args.models.resolve(), args.duration)
    mass, q0, v0, spring, damp, rest, bias = cp.asarray(coords.T.copy())
    drive_signs = cp.asarray(signs)
    projector = DeviceWrenches(args.method, args.geometry)
    cp.linalg.solve(cp.eye(2), cp.ones(2)).get()
    report = dict(
        method=args.method,
        gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        duration_target_s=args.duration,
        dt_s=args.step_us * 1e-6,
        stiffness_N_per_m2=args.stiffness,
        backend="CPU OCCT live regions and line points; GPU geometry projection, wrenches and HHT algebra",
        geometry_input=args.geometry.name,
        loads="nc_contact spur deck",
        calls=[],
        steps=[],
        diagnostic_window_s=args.observe_ms * 0.001,
        mass_kg_m2=mass.get().tolist(),
    )
    # Record the OCCT force cross-check outside the online wall timer.
    audits = []
    with worker(args.worker, files, cutoff, args.gpu_screen) as (p, version, face_counts):
        screen = DeviceScreen(p, face_counts) if args.gpu_screen else None
        report["broad_phase"] = "GPU face boxes" if screen else "CPU face tree"
        report.update(occt=version, face_counts=face_counts, setup_s=perf_counter() - start)

        def force(t, q, v):
            center, angle = relative_pose(q.get(), zeros, signs, separation)
            candidates = screen.candidates(center, angle, cutoff) if screen else None
            keys, samples, offsets, info = discover(p, center, angle, candidates)
            if candidates is not None:
                info["gpu_box_candidates"] = len(candidates)
            report["calls"].append(dict(t=t, faces=keys, samples=len(samples), **info))
            if info["unresolved"] or info["touching_witnesses"]:
                raise ArithmeticError(
                    "Unresolved contact region or zero-distance witness without a line normal"
                )
            w = projector(keys, samples, offsets, center, cutoff, args.stiffness)
            torque = (
                cp.column_stack((-(w[:, 5] + center[0] * w[:, 1] - center[1] * w[:, 0]), w[:, 5]))
                * drive_signs
            )
            magnitude = cp.linalg.norm(w[:, :3], axis=1)
            jac = torque / cp.where(magnitude > 0, magnitude, 1.0)[:, None]
            lengths = cp.asarray([samples[a:b, 6].sum() for a, b in zip(offsets[:-1], offsets[1:])])
            stiff = cp.diag(spring) + args.stiffness * (jac.T * lengths) @ jac
            applied = cp.asarray([np.interp(t, table[:, 0], table[:, 1]) for table in loads])
            f = bias + applied - spring * (q - rest) - damp * v + torque.sum(axis=0)
            audits.append((samples, offsets, center, w))
            return f, stiff

        q, v = q0.copy(), v0.copy()
        initial = perf_counter()
        f, _ = force(0.0, q, v)
        a = f / mass
        cp.cuda.get_current_stream().synchronize()
        report.update(
            initial_force_s=perf_counter() - initial,
            initial_force_Nm=f.get().tolist(),
            initial_q=q.get().tolist(),
            initial_v=v.get().tolist(),
            initial_a=a.get().tolist(),
        )
        wall = perf_counter()
        h = report["dt_s"]
        goal_steps = round(args.duration / h)
        alpha = -0.1
        beta, gamma = (1 - alpha) ** 2 / 4, 0.5 - alpha
        try:
            for step in range(1, goal_steps + 1):
                t = step * h
                begin = perf_counter()
                q_pred, v_pred = q + h * v + h * h * (0.5 - beta) * a, v + h * (1 - gamma) * a

                def trial(acc):
                    pos, vel = q_pred + beta * h * h * acc, v_pred + gamma * h * acc
                    new_f, k = force(t, pos, vel)
                    residual = mass * acc - (1 + alpha) * new_f + alpha * f
                    err = float(cp.max(cp.abs(residual)).get())
                    return pos, vel, new_f, k, residual, err

                acc = a.copy()
                current = trial(acc)
                for iteration in range(19):
                    pos, vel, new_f, k, residual, err = current
                    tol = 1e-7 * (1 + float(cp.maximum(cp.max(cp.abs(f)), cp.max(cp.abs(new_f))).get()))
                    if not np.isfinite(err):
                        raise ArithmeticError("Nonfinite HHT residual")
                    if err <= tol:
                        break
                    if iteration == 18:
                        raise ArithmeticError("HHT iteration budget exhausted")
                    matrix = cp.diag(mass) + (1 + alpha) * (beta * h * h * k + gamma * h * cp.diag(damp))
                    correction = cp.linalg.solve(matrix, -residual)
                    for ls in range(9):
                        candidate = acc + correction * (0.5**ls)
                        proposed = trial(candidate)
                        if proposed[-1] < err:
                            acc, current = candidate, proposed
                            break
                    else:
                        raise ArithmeticError("HHT line search failed")
                q, v, a, f = pos, vel, acc, new_f
                cp.cuda.get_current_stream().synchronize()
                elapsed = perf_counter() - begin
                report["steps"].append(
                    dict(
                        time_s=t,
                        wall_s=elapsed,
                        newton=iteration,
                        residual_Nm=err,
                        tolerance_Nm=tol,
                        q=q.get().tolist(),
                        v=v.get().tolist(),
                        a=a.get().tolist(),
                        force_Nm=f.get().tolist(),
                    )
                )
                if perf_counter() - wall > args.duration and t + 1e-15 >= args.observe_ms * 0.001:
                    break
        except ArithmeticError as error:
            report["failure"] = str(error)
        online = perf_counter() - wall
        report.update(
            wall_after_initialization_s=online,
            physical_s=len(report["steps"]) * h,
            completed=len(report["steps"]) == goal_steps,
            realtime=len(report["steps"]) == goal_steps and online <= args.duration,
        )
        errors = []
        for samples, offsets, center, w in audits:
            e = w.get() - reference(samples, offsets, center, cutoff, args.stiffness)
            errors.extend(np.linalg.norm(e[:, :3], axis=1))
        report["max_line_force_difference_N"] = float(max(errors, default=0.0))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("worker", "models", "geometry", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--method", choices=("cad", "mesh", "cpu-reference"), required=True)
    parser.add_argument("--step-us", type=float, required=True)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--stiffness", type=float, default=1e9)
    parser.add_argument(
        "--gpu-screen", action="store_true", help="Supply live GPU face-box candidates to OCCT"
    )
    parser.add_argument(
        "--observe-ms",
        type=float,
        default=0.0,
        help="After missing the wall budget, continue to this physical time for diagnostics",
    )
    args = parser.parse_args()
    if not all(np.isfinite(x) and x > 0 for x in (args.step_us, args.duration, args.stiffness)):
        parser.error("Expected finite positive time parameters and stiffness")
    steps = args.duration / (args.step_us * 1e-6)
    if round(steps) < 1 or not math.isclose(steps, round(steps), rel_tol=0, abs_tol=1e-7):
        parser.error("Duration must be a whole number of steps")
    if not 0 <= args.observe_ms <= 1000 * args.duration:
        parser.error("Diagnostic window must fit inside the simulation horizon")
    report = cpu_reference(args) if args.method == "cpu-reference" else run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
