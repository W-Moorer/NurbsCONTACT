"""Compare serial, parallel and same-pose diagnostic runs of the five-second reducer."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cpu_cad_pressure_report import history


def read_run(directory):
    def row(name):
        with (directory / name).open(newline="") as stream:
            return next(csv.DictReader(stream))

    run = row("response.stats.csv")
    times = row("response.gpu.csv")
    if int(run["steps"]) != 50000:
        raise ValueError(f"incomplete five-second run in {directory}")
    return {
        "wall_s": float(times["online_s"]),
        "phase_s": {
            k: float(times[k])
            for k in (
                "broad_s",
                "sort_s",
                "search_s",
                "group_s",
                "integral_s",
                "reduction_s",
            )
            if k in times
        },
        "reductions": int(times["reductions"]) if "reductions" in times else None,
        "queries": int(run["cad_patch_queries"]),
        "evaluations": int(run["force_evaluations"]),
        "subdivisions": int(run["subdivisions"]),
        "missed_deadlines": int(times["missed_deadlines"]),
        "response_sha256": hashlib.sha256(
            (directory / "response.csv").read_bytes()
        ).hexdigest(),
    }


def peak(error, mask):
    return np.max(np.abs(error[mask]), axis=0).tolist() if np.any(mask) else None


def check_regions(directory, times):
    rows = np.genfromtxt(directory / "response.regions.csv", delimiter=",", names=True)
    stages = []
    event = np.zeros(len(times), dtype=bool)
    for index in np.unique(rows["pair"]):
        r = rows[rows["pair"] == index]
        np.testing.assert_array_equal(r["step"], np.arange(len(times)))
        changed = r["entered_face_pairs"] + r["exited_face_pairs"] > 0
        event |= changed
        near = changed.copy()
        near[1:] |= changed[:-1]
        near[:-1] |= changed[1:]
        fields = {}
        for key, prefix, unit in (
            ("force_N", "F", "N"),
            ("torque_A_Nm", "A", "Nm"),
            ("torque_B_Nm", "B", "Nm"),
        ):
            data = np.column_stack(
                [r[f"{prefix}{axis}_error_{unit}"] for axis in "xyz"]
            )
            if not np.isfinite(data).all() or np.any(data < 0):
                raise ValueError("invalid per-region wrench error")
            fields[key] = {
                "all": peak(data, r["regions"] > 0),
                "switch_plus_minus_one_step": peak(data, near & (r["regions"] > 0)),
            }
        stages.append(
            {
                "pair": int(index),
                "fresh_queries": int(r["queries"].sum()),
                "regions_compared": int(r["regions"].sum()),
                "entering_face_pairs": int(r["entered_face_pairs"].sum()),
                "exiting_face_pairs": int(r["exited_face_pairs"].sum()),
                "switch_steps": int(changed.sum()),
                "region_count_changes": int(r["region_count_changes"].sum()),
                "compared_regions_in_switch_windows": int(r["regions"][near].sum()),
                "max_component_errors": fields,
            }
        )
    if not all(
        s["regions_compared"] and s["compared_regions_in_switch_windows"]
        for s in stages
    ):
        raise ValueError("missing checks at contact transitions")
    return stages, event, rows


def compare(a, b):
    return {
        key: {
            "max_abs": np.max(np.abs(b[key] - a[key]), axis=0).tolist(),
            "rms": np.sqrt(np.mean((b[key] - a[key]) ** 2, axis=0)).tolist(),
        }
        for key in ("speed", "output_torque_Nm", "normal_N")
    }


def split_steps(response, names, dt):
    """Find steps whose saved velocity is incompatible with a single full HHT update."""
    v = response[:, [names.index(f"v{j}") for j in range(3)]]
    a = response[:, [names.index(f"a{j}") for j in range(3)]]
    left = (dt * -0.5) * a[:-1]
    right = (1.5 * dt) * a[1:]
    defect = np.abs(v[1:] - ((v[:-1] + left) + right))
    rounding = (
        32 * np.finfo(float).eps * (np.abs(v[:-1]) + np.abs(left) + np.abs(right))
    )
    return np.flatnonzero(np.any(defect > rounding, axis=1)) + 1


def main(args):
    paths = {
        name: getattr(args, name)
        for name in ("original", "serial", "parallel", "audit")
    }
    info = {name: read_run(path) for name, path in paths.items()}
    traces = {
        name: history(path / "response.csv", args.scene)
        for name, path in paths.items()
        if name != "audit"
    }
    traces["cpu"] = history(args.cpu, args.scene)
    t = traces["serial"]["time_s"]
    for tr in traces.values():
        np.testing.assert_allclose(t, tr["time_s"], rtol=0, atol=1e-11)
    if len(t) != 50001 or abs(t[-1] - 5) > 1e-10:
        raise ValueError("require all original time steps")
    stages, event, audit_rows = check_regions(paths["audit"], t)
    if sum(s["fresh_queries"] for s in stages) != info["audit"]["queries"]:
        raise ValueError("unaccounted contact geometry queries")
    data = {
        name: np.atleast_2d(
            np.loadtxt(path / "response.csv", delimiter=",", skiprows=1)
        )
        for name, path in paths.items()
    }
    audit_same = np.array_equal(data["parallel"], data["audit"], equal_nan=True)
    if not audit_same:
        raise ValueError("instrumented and production responses differ")
    # Body-pair hits and loaded states are distinct from changes of constituent CAD face pairs.
    with (paths["serial"] / "response.csv").open() as stream:
        names = stream.readline().strip().split(",")
    mismatches = {}
    switches = {}
    for k in range(len(stages)):
        key = f"p{k}_hit"
        col = names.index(key)
        mismatches[key] = int(
            np.count_nonzero(data["serial"][:, col] != data["parallel"][:, col])
        )
        switches[key] = {}
        for baseline in ("original", "serial"):
            before = np.flatnonzero(np.diff(data[baseline][:, col])) + 1
            after = np.flatnonzero(np.diff(data["parallel"][:, col])) + 1
            switches[key][baseline] = {
                "baseline_switches": len(before),
                "parallel_switches": len(after),
                "unmatched_baseline_steps": np.setdiff1d(before, after).tolist(),
                "unmatched_parallel_steps": np.setdiff1d(after, before).tolist(),
            }
    delta = (
        traces["parallel"]["output_torque_Nm"] - traces["serial"]["output_torque_Nm"]
    )
    near = event.copy()
    near[1:] |= event[:-1]
    near[:-1] |= event[1:]
    metric = {
        "runs": info,
        "audit_matches_production": audit_same,
        "all_response_values_match": {
            name: bool(np.array_equal(data[name], data["parallel"], equal_nan=True))
            for name in ("original", "serial")
        },
        "regions": stages,
        "body_pair_hit_mismatches": mismatches,
        "body_pair_switches": switches,
        "parallel_minus_serial": compare(traces["serial"], traces["parallel"]),
        "parallel_minus_original": compare(traces["original"], traces["parallel"]),
        "parallel_minus_cpu": compare(traces["cpu"], traces["parallel"]),
        "max_output_torque_error_in_switch_windows_Nm": peak(delta, near),
        "scene_sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest(),
    }
    metric["output_contact_torque_peaks"] = {}
    worst = int(np.argmax(np.abs(delta)))
    split = {k: split_steps(data[k], names, 1e-4) for k in ("serial", "parallel")}
    metric["substep_history"] = {
        "serial_steps": split["serial"].tolist(),
        "parallel_steps": split["parallel"].tolist(),
        "parallel_only": np.setdiff1d(split["parallel"], split["serial"]).tolist(),
        "serial_only": np.setdiff1d(split["serial"], split["parallel"]).tolist(),
        "all_subdivisions_accounted_for": all(
            len(split[k]) == info[k]["subdivisions"] for k in split
        ),
    }
    metric["largest_closed_loop_torque_difference"] = {
        "step": worst,
        "time_s": float(t[worst]),
        "difference_Nm": float(delta[worst]),
        "serial_Nm": float(traces["serial"]["output_torque_Nm"][worst]),
        "parallel_Nm": float(traces["parallel"]["output_torque_Nm"][worst]),
        "within_one_step_of_face_change": bool(near[worst]),
    }
    metric["binaries_sha256"] = {
        k: hashlib.sha256(exe.read_bytes()).hexdigest()
        for k, exe in (("original", args.original_binary), ("modified", args.binary))
        if exe is not None
    }
    metric["cpu_response_sha256"] = hashlib.sha256(args.cpu.read_bytes()).hexdigest()
    if (args.parallel_repeat is None) != (args.serial_repeat is None):
        raise ValueError("supply both timings for a reverse-order repeat")
    if args.parallel_repeat:
        repeat = {
            "serial": read_run(args.serial_repeat),
            "parallel": read_run(args.parallel_repeat),
        }
        metric["reverse_order_repeat"] = repeat
        for mode in repeat:
            if repeat[mode]["response_sha256"] != info[mode]["response_sha256"]:
                raise ValueError(f"{mode} repeat produced a different trajectory")
        metric["matched_savings_s"] = [
            {
                "wall": batch["serial"]["wall_s"] - batch["parallel"]["wall_s"],
                "reduction": batch["serial"]["phase_s"]["reduction_s"]
                - batch["parallel"]["phase_s"]["reduction_s"],
                "integral": batch["serial"]["phase_s"]["integral_s"]
                - batch["parallel"]["phase_s"]["integral_s"],
            }
            for batch in (info, repeat)
        ]
    for name, trace in traces.items():
        y = trace["output_torque_Nm"]
        idx = int(np.argmax(np.abs(y)))
        metric["output_contact_torque_peaks"][name] = {
            "time_s": float(t[idx]),
            "torque_Nm": float(y[idx]),
        }
    args.out.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(args.out / "region-audit.npz", rows=audit_rows)
    (args.out / "comparison.json").write_text(
        json.dumps(metric, indent=2, allow_nan=False) + "\n"
    )
    fig, ax = plt.subplots(2, 2, figsize=(10, 5.5), layout="constrained")
    for key, color, style in (
        ("serial", "#2c6681", "-"),
        ("parallel", "#be6d32", "--"),
    ):
        y = traces[key]
        ax[0, 0].plot(t, y["output_torque_Nm"], style, color=color, lw=0.8, label=key)
        ax[0, 1].plot(t, y["speed"][:, 1], style, color=color, lw=0.8)
    ax[1, 0].plot(t, delta, color="#216c65", lw=0.65)
    # Zoom on the largest absolute output-torque jump coinciding with a change in confirmed faces.
    jumps = np.abs(np.diff(traces["serial"]["output_torque_Nm"], prepend=0))
    at = worst if np.any(delta) else int(np.argmax(np.where(event, jumps, -1)))
    zoom_label = (
        "Torque at largest difference [N m]"
        if np.any(delta)
        else "Torque at face switch [N m]"
    )
    window = np.abs(t - t[at]) <= 0.003
    for key, color, style in (
        ("serial", "#2c6681", "-"),
        ("parallel", "#be6d32", "--"),
    ):
        ax[1, 1].plot(
            t[window],
            traces[key]["output_torque_Nm"][window],
            style,
            color=color,
            lw=1.3,
        )
    ax[1, 1].axvline(t[at], color="#68737a", lw=0.8, ls=":")
    ax[0, 0].legend()
    for panel, label in zip(
        ax.flat,
        (
            "Output contact torque [N m]",
            "Intermediate speed [rad/s]",
            "Parallel minus serial [N m]",
            zoom_label,
        ),
    ):
        panel.set_ylabel(label)
        panel.set_xlabel("Time [s]")
        panel.margins(x=0)
        panel.grid(alpha=0.2)
    fig.savefig(args.out / "response.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel-repeat", type=Path)
    parser.add_argument("--serial-repeat", type=Path)
    parser.add_argument("--original-binary", type=Path)
    parser.add_argument("--binary", type=Path)
    for name in ("scene", "cpu", "original", "serial", "parallel", "audit", "out"):
        parser.add_argument(name, type=Path)
    main(parser.parse_args())
