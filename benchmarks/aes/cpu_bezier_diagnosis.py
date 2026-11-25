"""Diagnose cache warmup and torque spikes in the five-second CPU reducer runs."""

import argparse
import io
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _gpu_source import read_model
from cpu_backend_run import digest
from cpu_cad_pressure_report import wrench


def read_csv(path):
    return np.genfromtxt(path, names=True, delimiter=",")


def query(command, poses):
    stream = io.StringIO()
    np.savetxt(stream, poses, fmt="%.17g")
    run = subprocess.run(command, input=stream.getvalue(), capture_output=True, text=True,
                         check=True, timeout=1800)
    rows = list(map(json.loads, run.stdout.splitlines()))
    if len(rows) != len(poses):
        raise ValueError("incomplete geometry replay")
    return rows


def predictor_poses(previous, bodies, pair, dt):
    columns = []
    for body in pair[:2]:
        _, origin, axis, angle, j = bodies[body]
        q, v, acc = (previous[f"{k}{j}"] for k in "qva")
        # Same arithmetic order as Hht::solve with spectral radius zero.
        q_pred = q + dt*v + dt*dt*(-0.5)*acc
        columns.extend(np.full(len(q), x) for x in (*origin, *axis))
        columns.append(angle + (q_pred + dt*dt*acc))
    return np.column_stack(columns)


def contact_record(row, body, stiffness, limit):
    if not row["resolved"]:
        raise ValueError("unresolved predictor geometry")
    regions = row["regions"]
    load = wrench(regions, body[1], body[2], stiffness, limit, True)
    if load is None:
        raise ValueError("invalid predictor geometry")
    return dict(torque_Nm=load[-1], region_depth_um=[r[-1]*1e6 for r in regions],
                witnesses=[dict(faces=w[:2], group=g, point_m=w[2:5], depth_um=w[-1]*1e6)
                           for w, g in zip(row["witnesses"], row["labels"], strict=True)])


def swap_witness(command, pose, first, second, body, stiffness, limit):
    """Keep the pose and groups fixed; swap only the most displaced witness."""
    def keyed(row):
        return {tuple(w[:2]): (g, w) for g, w in zip(row["labels"], row["witnesses"], strict=True)}
    left, right = keyed(first), keyed(second)
    if {k: g for k, (g, _) in left.items()} != {k: g for k, (g, _) in right.items()}:
        raise ValueError("witness substitution requires matching groups")
    def movement(k):
        return np.linalg.norm(np.array(left[k][1][2:5])-right[k][1][2:5])
    target = max(left, key=movement)
    rows = []
    for substitute in (False, True):
        row = [*pose, len(left)]
        for key, (group, witness) in left.items():
            if substitute and key == target:
                witness = right[key][1]
            row.extend((group, *witness[:8]))
        rows.append(row)
    result = query(command, rows)
    if not all(r["resolved"] for r in result):
        raise ValueError("unresolved controlled patch integration")
    return dict(faces=target, witness_displacement_mm=1000*movement(target),
                torque_Nm=[wrench(r["patches"], body[1], body[2], stiffness, limit, True)[-1] for r in result],
                region_depth_um=[[p[-1]*1e6 for p in r["patches"]] for r in result])


def whole_reference(records, exe, body, stiffness, limit):
    if len(records) != 2 or not all(r["resolved"] for r in records):
        raise ValueError("incomplete whole-STEP pressure reference")
    return dict(executable_sha256=digest(exe), regions=[len(r["patches"]) for r in records],
                torque_Nm=[wrench(r["patches"], body[1], body[2], stiffness, limit, True)[-1] for r in records])


def step_queries(runs, indices):
    result = {}
    for name in ("cad-patch", "cad-bezier"):
        work = read_csv(runs / name / "progress.csv")
        result[name] = {s: {key: int(work[key][s]-work[key][s-1])
                            for key in ("cad_patch_batches", "cad_patch_queries")} for s in indices}
    return result


def plot(output, t, old, new, peak, onset, timing):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), layout="constrained")
    colors = ["#246762", "#b45930"]
    axes[0, 0].plot(t, new-old, color=colors[1], lw=.8)
    axes[0, 0].set(xlabel="Time (s)", ylabel="Bézier minus CAD (N m)", xlim=(0, 5))
    steps = np.arange(onset-6, peak+15)
    for data, label, marker, color in zip((old, new), ("CAD", "Bézier CAD"), ("o", "x"), colors):
        axes[0, 1].plot(t[steps], data[steps], marker=marker, markersize=4, lw=1.1, label=label, color=color)
    axes[0, 1].set(xlabel="Time (s)", ylabel="Output contact torque (N m)")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].plot(t[steps], (new-old)[steps], "o-", ms=3, lw=1.2, color=colors[1])
    axes[1, 0].axhline(0, lw=.6, color="gray")
    axes[1, 0].set(xlabel="Time (s)", ylabel="Bézier minus CAD (N m)")
    for ax in (axes[0, 1], axes[1, 0]):
        ax.axvline(t[onset], ls=":", lw=1, color="#4b5660")
        ax.ticklabel_format(useOffset=False, axis="x")
        ax.tick_params(axis="x", labelrotation=15)
    for (name, times), color in zip(timing.items(), colors):
        axes[1, 1].plot(np.arange(1, 6), times["mean_step_ms_by_second"], "o-", color=color, label=name)
    axes[1, 1].set(xlabel="Simulation time interval ending at (s)", ylabel="Integration wall time (ms/step)",
                   xticks=range(1, 6), ylim=(0, None))
    axes[1, 1].legend(frameon=False)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=175)
    plt.close(fig)


def run(args):
    out, runs = args.results, args.runs
    reference = json.loads((out / "replay.json").read_text())
    if digest(args.scene) != reference["scene_sha256"] or digest(args.trajectory) != reference["trajectory_sha256"]:
        raise ValueError("expected the recorded reducer scene and mesh pose history")
    for file, checksum in reference["binaries"].items():
        if digest(args.binaries / file) != checksum:
            raise ValueError(f"unexpected replay executable: {file}")
    bodies, pairs = read_model(args.scene)
    dt = 0.0001
    traces, timing = {}, {}
    for name in ("mesh-obb", "cad-bezier", "cad-patch"):
        record = json.loads((out / f"{name}-run.json").read_text())
        path = runs / name / "response.csv"
        if digest(path) != record["response_sha256"] or record["dt_s"] != dt or record["spectral_radius"] != 0:
            raise ValueError(f"unexpected integration input: {name}")
        traces[name] = read_csv(path)
        if name == "cad-patch":
            continue
        progress = read_csv(runs / name / "progress.csv")
        end, stats = record["last_accepted"], record["statistics"]
        seconds = np.arange(6) * 10000
        np.testing.assert_array_equal(progress["step"][seconds], seconds)
        mesh = name == "mesh-obb"
        batch = float(stats["mesh_detection_s" if mesh else "cad_patch_batch_s"])
        count = int(stats["mesh_parallel_batches" if mesh else "cad_patch_batches"])
        setup, total = float(stats["setup_s"]), record["process_s"]
        integration = end["integration_s"]
        timing[name] = dict(setup_s=setup, batch_s=batch, batches=count, mean_batch_ms=1000*batch/count,
                            other_integration_s=integration-batch, outside_s=total-setup-integration,
                            process_s=total, mean_step_ms_by_second=(np.diff(progress["integration_s"][seconds])*.1).tolist())
    with np.load(args.trajectory) as archive:
        angles, origins = archive["angles_rad"], archive["origins_m"]
    for i in range(len(bodies)):
        np.testing.assert_array_equal(angles[:, i], traces["mesh-obb"][f"b{i}_angle_rad"])
    with np.load(out / "responses.npz") as obs:
        t, old, new = (obs[k] for k in ("time_s", "cad-patch_output_torque_Nm", "cad-bezier_output_torque_Nm"))
    delta = new-old
    large = np.flatnonzero(abs(delta) > 1)
    clusters = np.split(large, np.flatnonzero(np.diff(large) > 20)+1)
    peak = int(np.argmax(abs(delta)))
    starts = [int(g[0]) for g in clusters]
    onset = next(int(g[0]) for g in clusters if peak in g)
    # At the single-frame excursion the raw curves already diverge one step earlier.
    earlier = [int(g[0])-1 for g in clusters if len(g) == 1]
    spike = dict(threshold_Nm=1, grouping_gap_steps=20, rms_Nm=np.sqrt(np.mean(delta**2)), peak_step=peak,
                 trial_queries=step_queries(runs, sorted(starts+earlier)),
                 episodes=[dict(first_step=int(g[0]), last_step=int(g[-1]), above_threshold=len(g),
                                peak_delta_Nm=delta[g[np.argmax(abs(delta[g]))]]) for g in clusters])
    repeated, crossover, controls = {}, {}, {}
    for stage, pair in enumerate(pairs, 1):
        a, b, stiffness, limit = pair
        p = [bodies[i][0] for i in (a, b)]
        cad = [args.binaries / "nc_cad_probes.exe", *p, "6", str(2*limit), "--loaded"]
        mesh = [args.binaries / "nc_mesh.exe", "--method", "mesh-obb", "--regions", "contour-ray",
                "--axis-angle", "--keep-unresolved", "--max-depth-m", str(limit), *[f.with_suffix(".msh") for f in p]]
        columns = [field for i in (a, b) for field in (*origins[:, i].T,
                   *(np.full(len(t), x) for x in bodies[i][2]), angles[:, i])]
        window = np.column_stack(columns)[49900:50000]
        for name, command, counter, timer in (("mesh", mesh, "triangle_tests", "query_s"),
                                             ("bezier", cad+["--bezier"], "candidate_face_pairs", "loaded_s")):
            records = query(command, np.tile(window, (3, 1)))
            repeated[f"stage_{stage}_{name}"] = [
                dict(ms=1000*np.mean([r[timer] for r in records[i+20:i+100]]),
                     **{counter: np.mean([r[counter] for r in records[i+20:i+100]])}) for i in (0, 100, 200)]
        predictors = {k: predictor_poses(traces[k], bodies, pair, dt) for k in ("cad-patch", "cad-bezier")}
        # Input row j predicts output j+1. Each window has 101 prior queries and the target.
        def prefix(poses, step):
            return poses[step-102:step]
        for trajectory, poses in predictors.items():
            for method, command in (("cad-patch", cad), ("cad-bezier", cad+["--bezier"])):
                records = query(command, np.concatenate([prefix(poses, s) for s in starts]))
                entry = {s: contact_record(records[102*j+101], bodies[b], stiffness, limit)
                         for j, s in enumerate(starts)}
                for s in earlier:
                    entry[s] = contact_record(query(command, prefix(poses, s))[-1], bodies[b], stiffness, limit)
                crossover[f"stage_{stage}_{trajectory}_with_{method}"] = entry
        if stage == 2:
            witnesses = []
            for mask in range(4):
                poses = prefix(predictors["cad-patch"], onset).copy()
                for bit, col in enumerate((6, 13)):
                    if mask & (1 << bit):
                        poses[-1, col] = predictors["cad-bezier"][onset-1, col]
                record = query(cad+["--bezier"], poses)[-1]
                controls[mask] = dict(angles_rad=poses[-1, [6, 13]].tolist(),
                                      **contact_record(record, bodies[b], stiffness, limit))
                witnesses.append(record)
            substitution = swap_witness([*cad[:-1], "--patch-wrench", "0", "--bezier"],
                                        predictors["cad-patch"][onset-1], witnesses[0], witnesses[3],
                                        bodies[b], stiffness, limit)
            exe = args.binaries / "nc_cad_distance_benchmark.exe"
            records = query([exe, *p, "--pressure-patches", str(2*limit)],
                            [predictors[k][onset-1] for k in ("cad-patch", "cad-bezier")])
            whole = whole_reference(records, exe, bodies[b], stiffness, limit)
    summary = dict(timing=timing, repeat_identical_window=repeated, spikes=spike,
                   predictor_crossover=crossover, final_angle_controls=controls, single_witness_swap=substitution,
                   whole_step_at_onset=whole)
    (out / "diagnosis.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    plot(out / "torque-diagnosis.png", t, old, new, peak, onset, timing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("scene", "binaries", "runs"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--results", type=Path, default=Path("results/aes/cpu-bezier"))
    parser.add_argument("--trajectory", type=Path, default=Path("results/aes/cpu-local-cad/trajectory.npz"))
    run(parser.parse_args())
