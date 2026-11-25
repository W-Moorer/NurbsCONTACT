"""Publish the measured CPU Bézier replay and the three independent reducer runs."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _gpu_source import read_model
from cpu_cad_pressure_report import history, wrench
from cpu_cad_pruning import verified_dynamics


METHODS = ("mesh-obb", "cad-patch", "cad-bezier")
LABELS = {"mesh-obb": "Mesh OBB", "cad-patch": "CAD", "cad-bezier": "Bézier CAD (ours)", "rd": "RecurDyn"}
COLORS = {"mesh-obb": "#b2642b", "cad-patch": "#67768a", "cad-bezier": "#00796b", "rd": "#28232e"}


def error(series, ref):
    delta = series - ref
    return dict(
        rms=np.sqrt(np.mean(delta**2, axis=0)).tolist(), max_abs=np.max(np.abs(delta), axis=0).tolist()
    )


def cross_check_peaks(scene, runs, probes, traces):
    bodies, pairs = read_model(scene)
    old, new = traces["cad-patch"], traces["cad-bezier"]
    torque_step = int(np.argmax(abs(old["output_torque_Nm"] - new["output_torque_Nm"])))
    output = []
    for stage, (a, b, stiff, limit) in enumerate(pairs):
        force_step = int(np.argmax(abs(old["normal_N"][:, stage] - new["normal_N"][:, stage])))
        targets = sorted({force_step, torque_step} if stage == 1 else {force_step})
        windows = []
        for name in ("cad-patch", "cad-bezier"):
            with (runs / name / "response.csv").open() as stream:
                rows = {
                    i: r
                    for i, r in enumerate(csv.DictReader(stream))
                    if any(step - 20 <= i <= step for step in targets)
                }
            for step in targets:
                poses = []
                for i in range(max(0, step - 20), step + 1):
                    row = rows[i]
                    poses.append(
                        " ".join(
                            str(x)
                            for body in (a, b)
                            for x in [
                                *[row[f"b{body}_{axis}_m"] for axis in "xyz"],
                                *bodies[body][2],
                                row[f"b{body}_angle_rad"],
                            ]
                        )
                    )
                origin = np.array([float(rows[step][f"b{b}_{axis}_m"]) for axis in "xyz"])
                windows.append((name, step, origin, poses))
        answers = {}
        for method in ("cad-patch", "cad-bezier"):
            cmd = [probes, bodies[a][0], bodies[b][0], "6", str(2 * limit), "--loaded"]
            if method == "cad-bezier":
                cmd.append("--bezier")
            source = "".join(line + "\n" for _, _, _, poses in windows for line in poses)
            proc = subprocess.run(
                cmd, input=source, capture_output=True, text=True, check=True, timeout=1800
            )
            values = list(map(json.loads, proc.stdout.splitlines()))
            if len(values) != sum(len(poses) for _, _, _, poses in windows):
                raise ValueError("incomplete peak-pose replay")
            offset = 0
            for name, step, origin, poses in windows:
                offset += len(poses)
                row = values[offset - 1]
                w = wrench(
                    row["regions"] if row["resolved"] else None, origin, bodies[b][2], stiff, limit, True
                )
                rec = answers.setdefault(
                    (name, step),
                    dict(
                        stage=stage + 1,
                        step=step,
                        trajectory=name,
                        closed_loop_normal_N=traces[name]["normal_N"][step, stage],
                        closed_loop_output_torque_Nm=traces[name]["output_torque_Nm"][step],
                    ),
                )
                rec[method] = (
                    dict(regions=len(row["regions"]), normal_N=w[0], shaft_Nm=w[-1])
                    if w is not None
                    else None
                )
        output.extend(answers.values())
    return output


def report(scene, replay, runs, out, probes=None, repeat=None):
    out.mkdir(parents=True, exist_ok=True)
    for src, name in (("summary.json", "replay.json"), ("poses.csv", "poses.csv")):
        shutil.copyfile(replay / src, out / name)
    with np.load("results/aes/cpu-foundation/reducer-physics/closed-loop.npz") as rd:
        time = rd["time_s"]
        commercial = {key: rd["rd_" + key] for key in ("output_torque_Nm", "speed", "te_mrad")}
    stats, traces = {}, {}
    for name in METHODS:
        path = runs / name
        manifest, _ = verified_dynamics(scene, path, name)
        shutil.copyfile(path / "run.json", out / (name + "-run.json"))
        obs = history(path / "response.csv", scene)
        np.testing.assert_allclose(obs["time_s"], time, rtol=0, atol=1e-10)
        trace = traces[name] = obs
        s = stats[name] = dict(
            process_s=manifest["process_s"],
            setup_s=float(manifest["statistics"]["setup_s"]),
            executable_sha256=manifest["executable_sha256"],
            torque_vs_recurdyn=error(trace["output_torque_Nm"], commercial["output_torque_Nm"]),
            speed_vs_recurdyn=error(trace["speed"], commercial["speed"]),
            te_vs_recurdyn=error(trace["te_mrad"], commercial["te_mrad"]),
        )
        loaded = obs["normal_N"] > 0
        s["load_changes"] = [int(np.count_nonzero(np.diff(x))) for x in loaded.T]
        s["first_loaded_ms"] = [float(1000 * time[np.flatnonzero(x)[0]]) for x in loaded.T]
        s["peak_normal_N"] = np.max(obs["normal_N"], axis=0).tolist()
    if len({s["executable_sha256"] for s in stats.values()}) != 1:
        raise ValueError("different CPU programs used")
    if repeat:
        repeated, _ = verified_dynamics(scene, repeat, "cad-bezier")
        shutil.copyfile(repeat / "run.json", out / "cad-bezier-repeat-run.json")
        other = history(repeat / "response.csv", scene)
        np.testing.assert_allclose(other["time_s"], time, atol=1e-10, rtol=0)
        stats["cad-bezier"]["repeat"] = dict(
            process_s=repeated["process_s"],
            executable_sha256=repeated["executable_sha256"],
            response_change={
                k: error(other[k], traces["cad-bezier"][k])
                for k in ("normal_N", "output_torque_Nm", "speed", "te_mrad")
            },
            load_state_differences=np.count_nonzero(
                (other["normal_N"] > 0) != (traces["cad-bezier"]["normal_N"] > 0), axis=0
            ).tolist(),
        )
    for name in ("cad-bezier", "mesh-obb"):
        x, y = traces[name], traces["cad-patch"]
        stats[name]["vs_cad"] = {
            k: error(x[k], y[k]) for k in ("output_torque_Nm", "normal_N", "speed", "te_mrad")
        }
        stats[name]["load_state_differences"] = np.count_nonzero(
            (x["normal_N"] > 0) != (y["normal_N"] > 0), axis=0
        ).tolist()
        stats[name]["max_torque_difference_time_s"] = float(
            time[np.argmax(abs(x["output_torque_Nm"] - y["output_torque_Nm"]))]
        )
    (out / "closed-loop.json").write_text(json.dumps(stats, indent=2, allow_nan=False) + "\n")
    if probes:
        results = cross_check_peaks(scene, runs, probes, traces)
        (out / "peak-poses.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(
        out / "responses.npz",
        time_s=time,
        **{"RecurDyn_" + k: series for k, series in commercial.items()},
        **{
            name + "_" + k: values[k]
            for name, values in traces.items()
            for k in ("output_torque_Nm", "speed", "te_mrad", "normal_N")
        },
    )

    fig, axes = plt.subplots(3, 2, figsize=(12, 10.1), layout="constrained")
    for name, values in {"rd": commercial, **traces}.items():
        style = dict(color=COLORS[name], lw=0.7, alpha=0.83, label=LABELS[name])
        axes[0, 0].plot(time, values["output_torque_Nm"], **style)
        axes[0, 1].plot(time, values["speed"][:, 1], **style)
    axes[0, 0].set(ylabel="Output contact torque (N m)")
    axes[0, 1].set(ylabel="Intermediate shaft speed (rad/s)")
    for ax in axes[0]:
        ax.set(xlim=(0, 5), xlabel="Time (s)")
        ax.legend(frameon=False, fontsize=8, ncol=2)
    torque = traces["cad-bezier"]["output_torque_Nm"] - traces["cad-patch"]["output_torque_Nm"]
    axes[1, 0].plot(time, torque, lw=0.65, color=COLORS["cad-bezier"])
    axes[1, 0].set(xlim=(0, 5), xlabel="Time (s)", ylabel="Bézier minus CAD output torque (N m)")
    with (replay / "poses.csv").open() as f:
        poses = list(csv.DictReader(f))
    for name in METHODS:
        points = [p for p in poses if p["method"] == name and int(p["stage"]) == 2 and int(p["step"]) <= 24]
        axes[1, 1].plot(
            [int(p["step"]) * 0.1 for p in points],
            [float(p["shaft_Nm"]) if p["shaft_Nm"] else np.nan for p in points],
            color=COLORS[name],
            label=LABELS[name],
            lw=1.2,
            ls="--" if name == "cad-patch" else "-",
        )
    axes[1, 1].set(xlabel="Time on common pose history (ms)", ylabel="Stage 2 contact torque (N m)")
    axes[1, 1].legend(frameon=False, fontsize=8)
    for stage, ax in enumerate(axes[2]):
        for name in METHODS:
            ax.plot(
                time,
                traces[name]["normal_N"][:, stage],
                color=COLORS[name],
                lw=0.65,
                alpha=0.8,
                label=LABELS[name],
            )
        ax.set(xlim=(0, 5), xlabel="Time (s)", ylabel=f"Stage {stage + 1} total normal load (N)")
        ax.legend(frameon=False, fontsize=8, ncol=3)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
    fig.savefig(out / "reducer-cpu.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("scene", "replay", "runs", "out"):
        p.add_argument(name, type=Path)
    p.add_argument(
        "--probes",
        type=Path,
        help="Cross-check the largest closed-loop load differences on common output poses.",
    )
    p.add_argument("--repeat", type=Path, help="Independent five-second CAD Bézier rerun")
    report(**vars(p.parse_args()))
