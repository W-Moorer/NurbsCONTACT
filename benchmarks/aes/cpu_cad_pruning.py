"""Audit CPU CAD pruning at fixed poses and through independent 5 s trajectories."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _gpu_source import read_model
from cpu_backend_run import digest, run
from cpu_cad_pressure import read_rows
from cpu_cad_pressure_report import history, timing, wrench
from cpu_free_body import collision, SPHERE


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def static_checks(args):
    bodies, pairs = read_model(args.scene)
    results = []
    binaries = {"previous": args.previous, "new": args.new}
    for stage, (a, b, _, _) in enumerate(pairs):
        with (args.poses / f"stage-{stage}.poses").open() as stream:
            initial = next(stream)
        for name in ("previous", "new", "new", "previous"):
            exe = binaries[name]
            cmd = [exe, bodies[a][0], bodies[b][0], ".00001"]
            result = subprocess.run(
                cmd, input=initial, capture_output=True, text=True, check=True, timeout=1800
            )
            (row,) = csv.DictReader(result.stdout.splitlines())
            results.append(
                dict(
                    stage=stage + 1,
                    method=name,
                    executable_sha256=digest(exe),
                    **{k: float(v) for k, v in row.items()},
                )
            )
    save(args.out / "startup-static.json", results)


def spheres(args):
    scene, _, compare = collision(False)
    shutil.copyfile(SPHERE, args.work / SPHERE.name)
    path = args.work / "impact.scene"
    path.write_text("\n".join(scene) + "\n")
    results = {}
    for name, exe in (("previous", args.previous), ("new", args.new)):
        directory = args.work / name
        run(
            exe.resolve(),
            path.resolve(),
            directory,
            "cad-patch",
            "contour-ray",
            "penalty",
            "shared",
            "predictor",
            0.0,
        )
        manifest = json.loads((directory / "run.json").read_text())
        if manifest["status"] != "complete":
            raise RuntimeError(f"Sphere impact {name}: {manifest['status']}")
        trace = np.genfromtxt(directory / "response.csv", names=True, delimiter=",")
        t = trace["time_s"] - 0.04
        exact = np.where((t >= 0) & (t < np.pi / 100), 22.5 * np.sin(100 * t), 0)
        results[name] = dict(
            process_s=manifest["process_s"],
            response_sha256=manifest["response_sha256"],
            executable_sha256=manifest["executable_sha256"],
            frames=len(t),
            peak_N=float(trace["p0_normal_N"].max()),
            max_force_error_N=float(abs(trace["p0_normal_N"] - exact).max()),
            **compare(trace),
        )
    save(args.out / "sphere.json", results)


def fixed_loads(args):
    bodies, pairs = read_model(args.scene)
    archive = Path("results/aes/cpu-cad-pressure/wrenches.csv")
    with archive.open() as stream:
        previous = {
            (int(r["stage"]), int(r["step"])): r for r in csv.DictReader(stream) if r["method"] == "area-0"
        }
    fields = ("sum_normal_N", "Fx_N", "Fy_N", "Fz_N", "Mx_Nm", "My_Nm", "Mz_Nm", "shaft_Nm")
    rows = []
    with np.load("results/aes/cpu-local-cad/trajectory.npz") as source:
        origins = source["origins_m"]
    for stage, (_, driven, stiffness, limit) in enumerate(pairs):
        stem = args.current_replay / f"area-0-{stage}"
        ids = json.loads(stem.with_suffix(".ids").read_text())
        for step, entry in zip(ids, read_rows(stem.with_suffix(".jsonl")), strict=True):
            old = previous[stage + 1, step]
            w = wrench(
                entry["patches"] if entry["resolved"] else None,
                origins[step, driven],
                bodies[driven][2],
                stiffness,
                limit,
                True,
            )
            row = dict(
                stage=stage + 1,
                step=step,
                resolved=w is not None,
                previous_resolved=old["resolved"] == "True",
            )
            if w is not None and row["previous_resolved"]:
                delta = w - np.array([float(old[k]) for k in fields])
                row.update(
                    shaft_Nm=w[-1],
                    shaft_change_Nm=delta[-1],
                    force_vector_change_N=float(np.linalg.norm(delta[1:4])),
                    torque_vector_change_Nm=float(np.linalg.norm(delta[4:7])),
                )
            rows.append(row)
    if {(r["stage"], r["step"]) for r in rows} != set(previous):
        raise ValueError("Missing fixed-pose load audit")
    save(args.out / "fixed-loads.json", rows)


def peak_poses(args):
    """Recompute the largest closed-loop discrepancy on each accepted pose with both builds."""
    bodies, pairs = read_model(args.scene)
    a, b, k, limit = pairs[1]
    histories = {name: history(args.work / name / "response.csv", args.scene) for name in ("old", "new")}
    step = int(np.argmax(abs(histories["old"]["output_torque_Nm"] - histories["new"]["output_torque_Nm"])))
    rows = []

    def query(exe, options, source):
        cmd = [exe, bodies[a][0], bodies[b][0], *options]
        ret = subprocess.run(cmd, input=source, text=True, capture_output=True, timeout=1800, check=True)
        return [json.loads(line) for line in ret.stdout.splitlines()]

    for path, series in histories.items():
        with (args.work / path / "response.csv").open() as stream:
            poses = [r for i, r in enumerate(csv.DictReader(stream)) if step - 20 <= i <= step]
        inputs = [
            " ".join(
                str(x)
                for i in (a, b)
                for x in [*[r[f"b{i}_{c}_m"] for c in "xyz"], *bodies[i][2], r[f"b{i}_angle_rad"]]
            )
            for r in poses
        ]
        origin = np.array([float(poses[-1][f"b{b}_{c}_m"]) for c in "xyz"])
        record = dict(
            trajectory=path,
            step=step,
            time_s=series["time_s"][step],
            integrated_shaft_Nm=series["output_torque_Nm"][step],
        )
        for tag, exe in (("old", args.previous), ("new", args.new)):
            hits = query(exe, ["6", str(2 * limit), "--indexed-groups"], "\n".join(inputs) + "\n")[-1]
            values = [
                v for label, w in zip(hits["labels"], hits["witnesses"], strict=True) for v in (label, *w[:8])
            ]
            line = " ".join([inputs[-1], str(len(hits["labels"])), *map(str, values)])
            (result,) = query(exe, ["6", str(2 * limit), "--patch-wrench", "0"], line + "\n")
            w = wrench(
                result["patches"] if result["resolved"] else None, origin, bodies[b][2], k, limit, True
            )
            record[tag] = None if w is None else dict(regions=len(result["patches"]), shaft_Nm=w[-1])
        rows.append(record)
    save(args.out / "peak-poses.json", rows)


def fixed_replay(previous, current):
    summaries = []
    for stage in range(2):
        name = f"indexed-{stage}.jsonl"
        differences = dict(face_pairs=0, partitions=0, counts=0)
        max_error = np.zeros(3)
        first = []

        def partition(row):
            groups = {}
            for label, w in zip(row["labels"], row["witnesses"], strict=True):
                groups.setdefault(label, []).append(tuple(w[:2]))
            return sorted(sorted(g) for g in groups.values())

        for step, (old, new) in enumerate(
            zip(read_rows(previous / name), read_rows(current / name), strict=True)
        ):
            a, b = old["witnesses"], new["witnesses"]
            if [w[:2] for w in a] != [w[:2] for w in b]:
                differences["face_pairs"] += 1
            elif a:
                delta = np.asarray(a)[:, 2:] - np.asarray(b)[:, 2:]
                error = [
                    np.linalg.norm(delta[:, :3], axis=1).max(),
                    np.linalg.norm(delta[:, 3:6], axis=1).max(),
                    np.abs(delta[:, 6]).max(),
                ]
                max_error = np.maximum(max_error, error)
            g, h = partition(old), partition(new)
            if g != h:
                differences["partitions"] += 1
                if len(first) < 20:
                    first.append(step)
            differences["counts"] += len(g) != len(h)
        if step != 50000:
            raise ValueError("Incomplete pose replay")
        summaries.append(
            dict(
                stage=stage + 1,
                poses=step + 1,
                differences=differences,
                first_different_partitions=first,
                max_point_normal_depth_difference=max_error.tolist(),
            )
        )
    return summaries


def verified_dynamics(scene, directory, method):
    manifest = json.loads((directory / "run.json").read_text())
    expected = dict(
        method=method,
        solver="penalty",
        penalty_stiffness="shared",
        geometry_update="predictor",
        spectral_radius=0,
        scope="contour-ray" if method == "mesh-obb" else None,
        requested_s=5,
        dt_s=0.0001,
        output_dt_s=0.0001,
        status="complete",
        scene_sha256=digest(scene),
        response_sha256=digest(directory / "response.csv"),
    )
    if (
        any(manifest[key] != value for key, value in expected.items())
        or manifest["last_accepted"]["step"] != 50000
    ):
        raise ValueError(f"Incompatible 5 s dynamics run: {directory}")
    with (directory / "progress.csv").open() as stream:
        startup = next(row for row in csv.DictReader(stream) if int(row["step"]) == 20)
    return manifest, startup


def repeat_report(args):
    results = {}
    torque = {}
    for name in ("new", "old"):
        directory = args.work / f"repeat-{name}"
        manifest, start = verified_dynamics(args.scene, directory, "cad-patch")
        control = json.loads((args.work / name / "run.json").read_text())
        if manifest["executable_sha256"] != control["executable_sha256"]:
            raise ValueError(f"Incompatible repeat: {name}")
        first = float(start["integration_s"])
        results[name] = dict(
            process_s=manifest["process_s"],
            first_2ms_s=first,
            later_integration_s=float(manifest["statistics"]["wall_s"]) - first,
            setup_s=float(manifest["statistics"]["setup_s"]),
            response_sha256=manifest["response_sha256"],
            executable_sha256=manifest["executable_sha256"],
        )
        shutil.copyfile(directory / "run.json", args.out / f"repeat-{name}-run.json")
        torque[name] = history(directory / "response.csv", args.scene)["output_torque_Nm"]
    delta = torque["new"] - torque["old"]
    results["response"] = dict(
        torque_rms_Nm=float(np.mean(delta**2) ** 0.5),
        torque_max_abs_Nm=float(np.abs(delta).max()),
        new_0p5ms_Nm=torque["new"][5],
        old_0p5ms_Nm=torque["old"][5],
    )
    save(args.out / "isolated-repeat.json", results)
    folder = args.work / "distance-only"
    if (folder / "run.json").exists():
        doc, early = verified_dynamics(args.scene, folder, "cad-patch")
        start = float(early["integration_s"])
        shaft = history(folder / "response.csv", args.scene)["output_torque_Nm"]
        diff = shaft - torque["old"]
        save(
            args.out / "distance-only.json",
            dict(
                process_s=doc["process_s"],
                first_2ms_s=start,
                later_integration_s=float(doc["statistics"]["wall_s"]) - start,
                setup_s=float(doc["statistics"]["setup_s"]),
                torque_at_0p5ms_Nm=shaft[5],
                torque_rms_vs_repeat_old_Nm=float(np.mean(diff**2) ** 0.5),
                torque_max_abs_vs_repeat_old_Nm=float(np.abs(diff).max()),
            ),
        )
        shutil.copyfile(folder / "run.json", args.out / "distance-only-run.json")


def report(args):
    previous = None
    stats = {}
    rd = np.load("results/aes/cpu-foundation/reducer-physics/closed-loop.npz")
    traces = {}
    for name in ("old", "new", "mesh"):
        directory = args.work / name
        manifest, milestone = verified_dynamics(
            args.scene, directory, "mesh-obb" if name == "mesh" else "cad-patch"
        )
        path = directory / "response.csv"
        values = history(path, args.scene)
        np.testing.assert_allclose(values["time_s"], rd["time_s"], atol=1e-10, rtol=0)
        metrics = dict(
            process_s=manifest["process_s"],
            setup_s=float(manifest["statistics"]["setup_s"]),
            start_2ms_wall_s=float(milestone["integration_s"]),
            steps=manifest["last_accepted"]["step"],
            torque_rms_vs_RecurDyn_Nm=float(
                np.mean((values["output_torque_Nm"] - rd["rd_output_torque_Nm"]) ** 2) ** 0.5
            ),
            torque_at_0p5ms_Nm=float(values["output_torque_Nm"][5]),
        )
        metrics["end_to_end_mean_ms_per_step"] = metrics["process_s"] / 50.0
        metrics["integration_after_2ms_s"] = (
            float(manifest["statistics"]["wall_s"]) - metrics["start_2ms_wall_s"]
        )
        if name != "mesh":
            metrics["startup_local_search_s"] = float(milestone["cad_patch_search_s"])
        if name == "old":
            previous = values
            archive = Path("results/aes/cpu-cad-pressure")
            earlier = json.loads((archive / "cad-run.json").read_text())
            if earlier["executable_sha256"] == manifest["executable_sha256"]:
                with np.load(archive / "closed-loop-cad.npz") as reference:
                    delta = values["output_torque_Nm"] - reference["output_torque_Nm"]
                    metrics["same_executable_repeat_torque_Nm"] = dict(
                        max_abs=float(np.max(abs(delta))), rms=float(np.mean(delta**2) ** 0.5)
                    )
        if name == "new":
            metrics["difference_from_previous"] = {}
            for key in ("output_torque_Nm", "speed", "normal_N", "te_mrad"):
                delta = values[key] - previous[key]
                metrics["difference_from_previous"][key] = dict(
                    max_abs=np.max(abs(delta), axis=0).tolist(),
                    rms=np.sqrt(np.mean(delta**2, axis=0)).tolist(),
                )
        stats[name] = metrics
        traces[name] = values
        shutil.copyfile(directory / "run.json", args.out / f"{name}-run.json")
    save(args.out / "dynamics.json", stats)
    save(args.out / "same-poses.json", fixed_replay(args.previous_replay, args.current_replay))
    if (args.current_replay / "paired-old-0.json").exists():
        timing(argparse.Namespace(work=args.current_replay, out=args.out))
    t = rd["time_s"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2), layout="constrained")
    ax = axes[0]
    labels = ["Previous CAD", "CAD (ours)", "CPU mesh OBB"]
    repeat = args.out / "isolated-repeat.json"
    rounds = [("First run", [s["process_s"] for s in stats.values()], "#778491", -0.17)]
    if repeat.exists():
        second = json.loads(repeat.read_text())
        rounds.append(
            ("Reverse order", [second[name]["process_s"] for name in ("old", "new")], "#167d75", 0.17)
        )
    for label, costs, color, offset in rounds:
        rows = np.arange(len(costs)) + offset
        ax[0].barh(rows, costs, height=0.32, color=color, label=label)
        for row, cost in zip(rows, costs, strict=True):
            ax[0].text(cost - 4, row, f"{cost:.1f} s", ha="right", va="center", color="white", fontsize=8)
    ax[0].set_yticks(range(3), labels)
    ax[0].legend(loc="lower right", fontsize=8)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("5 s simulation, wall time (s)")
    ax[1].plot(
        t, traces["new"]["output_torque_Nm"] - traces["old"]["output_torque_Nm"], lw=0.6, color="#167d75"
    )
    ax[1].set(xlim=(0, 5), xlabel="Time (s)", ylabel="New minus previous output torque (N m)")
    quantities = {
        "output_torque_Nm": "Output shaft torque (N m)",
        "speed": "Intermediate shaft speed (rad/s)",
    }
    for panel, (key, label) in zip(axes[1], quantities.items(), strict=True):
        for name, color, tag in (
            ("rd", "#596479", "RecurDyn"),
            ("mesh", "#bd7139", "CPU mesh OBB"),
            ("new", "#167d75", "CAD (ours)"),
        ):
            y = rd[f"rd_{key}"] if name == "rd" else traces[name][key]
            panel.plot(t, y if y.ndim == 1 else y[:, 1], lw=0.6, alpha=0.8, color=color, label=tag)
        panel.set(xlim=(0, 5), xlabel="Time (s)", ylabel=label)
    axes[1, 1].legend(ncol=3, fontsize=8, loc="lower center")
    np.savez_compressed(
        args.out / "output-torque.npz",
        time_s=t,
        **{name: tr["output_torque_Nm"] for name, tr in traces.items()},
        recurdyn=rd["rd_output_torque_Nm"],
    )
    fig.savefig(args.out / "cpu-pruning.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="action", required=True)
    for action, keys in (
        ("static", ("scene", "poses", "previous", "new", "out")),
        ("spheres", ("work", "previous", "new", "out")),
        ("loads", ("scene", "current_replay", "out")),
        ("peaks", ("scene", "work", "previous", "new", "out")),
        ("repeat-report", ("scene", "work", "out")),
        ("report", ("scene", "work", "previous_replay", "current_replay", "out")),
    ):
        cmd = subs.add_parser(action)
        for key in keys:
            cmd.add_argument(key, type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.action == "spheres":
        args.work.mkdir(parents=True, exist_ok=True)
        os.environ["OMP_NUM_THREADS"] = "2"
        spheres(args)
    elif args.action == "loads":
        fixed_loads(args)
    elif args.action == "peaks":
        peak_poses(args)
    elif args.action == "static":
        static_checks(args)
    elif args.action == "repeat-report":
        repeat_report(args)
    else:
        report(args)
