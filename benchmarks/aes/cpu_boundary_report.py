"""Compare complete CPU reducer histories and the eccentric cam against their independent references."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cpu_backend_run import digest
from cpu_cad_pressure_report import history
from resident_cam import reference


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def changes(t, active):
    return t[1:][active[:-1] != active[1:]]


def cam(folder, out):
    solution, entering = reference(3.2, 0, 0.001)
    result = dict(
        reference="DOP853",
        reference_events_s=solution.t_events[0].tolist(),
        reference_entering=entering.tolist(),
        methods={},
    )
    fig, axes = plt.subplots(2, figsize=(9, 5.2), sharex=True, layout="constrained")
    traces = {}
    for label, filename in (
        ("mesh-obb", "final-mesh-obb.csv"),
        ("cad-compiled", "cad-compiled.csv"),
        ("cad-boundary", "final-cad-boundary.csv"),
        ("cad-boundary fixed96", "freeze-fixed96.csv"),
    ):
        path = folder / filename
        x = np.genfromtxt(path, delimiter=",", names=True)
        t = x["time_s"]
        np.testing.assert_allclose(t, np.arange(32001) * 1e-4, atol=1e-10, rtol=0)
        state = solution.sol(t)
        load = x["p0_normal_N"]
        # The exact circle / flat geometry gives a separate check at the CPU's own posture.
        same_pose_load = 5000 * np.maximum(0, 0.105 + 0.024 * np.sin(x["q0"]) - 0.115 - x["q1"])
        error = x["q1"] - state[2]
        event_times = changes(t, load > 0)
        stats = next(csv.DictReader(path.with_suffix(".stats.csv").read_text().splitlines()))
        result["methods"][label] = dict(
            response_sha256=digest(path),
            integration_s=float(stats["wall_s"]),
            follower_rms_mm=1000 * rms(error),
            follower_max_abs_mm=float(1000 * np.max(abs(error))),
            same_pose_max_load_error_N=float(np.max(abs(load - same_pose_load))),
            events_s=event_times.tolist(),
        )
        detected_entering = (load[1:] > 0)[(load[1:] > 0) != (load[:-1] > 0)]
        if np.array_equal(detected_entering, entering):
            dt_ms = 1000 * (event_times - solution.t_events[0])
            result["methods"][label].update(
                event_time_rms_ms=rms(dt_ms), event_time_max_abs_ms=float(np.max(abs(dt_ms)))
            )
        traces[label + "_follower_m"] = x["q1"]
        traces[label + "_normal_N"] = load
        axes[0].plot(t, 1e3 * x["q1"], lw=0.8, label=label)
        axes[1].plot(t, 1e3 * error, lw=0.8, label=label)
    axes[0].plot(t, 1e3 * state[2], color="#252525", lw=0.9, ls="--", label="independent ODE")
    axes[0].set_ylabel("Follower travel (mm)")
    axes[1].set_ylabel("Travel error (mm)")
    axes[1].set_xlabel("Time (s)")
    axes[0].legend(ncol=3, fontsize=8)
    for ax in axes:
        ax.set_xlim(0, 3.2)
        ax.grid(alpha=0.2)
    fig.savefig(out / "cam.png", dpi=170)
    plt.close(fig)
    np.savez_compressed(out / "cam.npz", time_s=t, reference_follower_m=state[2], **traces)
    (out / "cam.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def reducer(scene, folders, archive, out):
    labels = ("mesh-obb", "cad-compiled", "cad-boundary")
    data = {m: history(path / "response.csv", scene) for m, path in zip(labels, folders, strict=True)}
    t = data[labels[0]]["time_s"]
    reference_run = {}
    with np.load(archive, allow_pickle=False) as stored:
        np.testing.assert_allclose(t, stored["time_s"], atol=1e-10, rtol=0)
        for name in ("output_torque_Nm", "speed", "te_mrad"):
            reference_run[name] = stored["recurdyn_" + name]
    result = {}
    for name, folder in zip(labels, folders, strict=True):
        d = data[name]
        np.testing.assert_allclose(d["time_s"], np.arange(50001) * 0.0001, atol=1e-10, rtol=0)
        run = json.loads((folder / "run.json").read_text())
        if run["status"] != "complete" or digest(folder / "response.csv") != run["response_sha256"]:
            raise ValueError("Incomplete or changed simulation")
        torque = d["output_torque_Nm"]
        i = int(np.argmax(abs(torque)))
        j = int(np.argmax(abs(np.diff(torque)))) + 1
        result[name] = dict(
            integration_s=float(run["statistics"]["wall_s"]),
            steps=len(t) - 1,
            scene_sha256=run["scene_sha256"],
            executable_sha256=run["executable_sha256"],
            response_sha256=run["response_sha256"],
            peak_abs_torque_Nm=float(abs(torque[i])),
            time_at_peak_s=float(t[i]),
            largest_adjacent_torque_jump_Nm=float(abs(torque[j] - torque[j - 1])),
            time_after_jump_s=float(t[j]),
            torque_rms_vs_RecurDyn_Nm=rms(torque - reference_run["output_torque_Nm"]),
            intermediate_speed_rms_vs_RecurDyn_rad_s=rms(d["speed"][:, 1] - reference_run["speed"][:, 1]),
            active_switches_s=[changes(t, n > 0).tolist() for n in d["normal_N"].T],
        )
    i = int(np.argmax(abs(reference_run["output_torque_Nm"])))
    result["RecurDyn"] = dict(
        peak_abs_torque_Nm=float(abs(reference_run["output_torque_Nm"][i])), time_at_peak_s=float(t[i])
    )
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), layout="constrained")
    styles = {
        "mesh-obb": "#2776A6",
        "cad-compiled": "#B15C22",
        "cad-boundary": "#227851",
        "RecurDyn": "#404049",
    }
    for label, d in [("RecurDyn", reference_run), *data.items()]:
        for ax, values in zip(
            axes, (d["output_torque_Nm"], d["speed"][:, 1], d["output_torque_Nm"]), strict=True
        ):
            ax.plot(t, values, color=styles[label], lw=0.72, alpha=0.85, label=label)
    for ax in axes:
        ax.grid(alpha=0.22)
        ax.set_xlim(0, 5)
    center = result["cad-boundary"]["time_at_peak_s"]
    axes[-1].set_xlim(max(0, center - 0.012), min(5, center + 0.012))
    axes[0].set_ylabel("Output torque (N m)")
    axes[1].set_ylabel("Intermediate speed (rad/s)")
    axes[2].set_ylabel("Output torque (N m)")
    axes[2].set_xlabel("Time (s)")
    axes[0].legend(ncol=4, fontsize=8)
    fig.savefig(out / "reducer.png", dpi=170)
    plt.close(fig)
    data.pop("mesh-obb")
    data.pop("cad-compiled")
    # Frozen control and RecurDyn arrays are already distributed in the supplied archive.
    np.savez_compressed(out / "reducer.npz", **data["cad-boundary"])
    (out / "reducer.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def pose_table(config_path, reference_path, files, out):
    config = json.loads(config_path.read_text())
    poses = {(s["stage"], step) for s in config["stages"] for step in s["steps"]}
    fresh = {(s["stage"], step) for s in config["stages"] for step in s["held_out_steps"]}
    ref = {(r["stage"], r["step"]): r for r in json.loads(reference_path.read_text())["rows"]}
    if poses != ref.keys():
        raise ValueError("STEP reference and recorded poses differ")
    fingerprints = {}
    with (out / "pose-loads.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            (
                "method",
                "stage",
                "step",
                "held_out",
                "resolved",
                "regions",
                "step_regions",
                "sum_normal_N",
                "Fx_N",
                "Fy_N",
                "Fz_N",
                "Mx_Nm",
                "My_Nm",
                "Mz_Nm",
                "shaft_Nm",
                "force_error_N",
                "moment_error_Nm",
                "shaft_error_Nm",
                "rules",
                "converged",
            )
        )
        for path in files:
            run = json.loads(path.read_text())
            name = run["method"] + ("-reverse" if run["reverse_sequence"] else "")
            fingerprints[name] = {k: v for k, v in run.items() if k != "rows"}
            rows = run["rows"]
            if len(rows) != len(poses) or {(r["stage"], r["step"]) for r in rows} != poses:
                raise ValueError(f"Incomplete pose set in {path}")
            for r in rows:
                key = (r["stage"], r["step"])
                a = r["wrench"]
                expected = ref[key]
                err = np.asarray(a) - expected["wrench"] if a is not None else None
                w.writerow(
                    [
                        name,
                        *key,
                        key in fresh,
                        r["resolved"],
                        r["regions"],
                        expected["regions"],
                        *(a if a is not None else [""] * 8),
                        *(
                            [""] * 3
                            if err is None
                            else [np.linalg.norm(err[1:4]), np.linalg.norm(err[4:7]), err[-1]]
                        ),
                        r["rules"],
                        r["converged"],
                    ]
                )
    (out / "probe-builds.json").write_text(json.dumps(fingerprints, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="case", required=True)
    c = sub.add_parser("cam")
    c.add_argument("folder", type=Path)
    r = sub.add_parser("reducer")
    for arg in ("scene", "mesh", "frozen", "boundary", "archive"):
        r.add_argument(arg, type=Path)
    t = sub.add_parser("poses")
    t.add_argument("poses", type=Path)
    t.add_argument("reference", type=Path)
    t.add_argument("files", type=Path, nargs="+")
    for cmd in (r, c, t):
        cmd.add_argument("out", type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.case == "cam":
        cam(a.folder, a.out)
    elif a.case == "poses":
        pose_table(a.poses, a.reference, a.files, a.out)
    else:
        reducer(a.scene, (a.mesh, a.frozen, a.boundary), a.archive, a.out)
