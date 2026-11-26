"""Compare two nc_contact_runtime responses on the original time grid."""
import argparse
import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np


def read(path, max_rows=None):
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, max_rows=max_rows))


def stats(path):
    with path.with_suffix(".stats.csv").open(newline="") as stream:
        return next(csv.DictReader(stream))


def audit_transfers(path, expected_steps):
    with sqlite3.connect(path) as db:
        start, end, count = db.execute("""
            SELECT min(k.start), max(k.end), count(*)
            FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id
            WHERE s.value LIKE '%resident_steps%'
        """).fetchone()
        if count != expected_steps:
            raise ValueError(f"expected {expected_steps} resident kernels, captured {count}")
        copies = list(db.execute("""
            SELECT e.label, count(*), sum(m.bytes)
            FROM CUPTI_ACTIVITY_KIND_MEMCPY m JOIN ENUM_CUDA_MEMCPY_OPER e ON m.copyKind=e.id
            WHERE m.start < ? AND m.end > ? GROUP BY e.label
        """, (end, start)))
        launches = db.execute("""
            SELECT count(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id
            WHERE s.value LIKE 'cudaGraphLaunch%'
        """).fetchone()[0]
        if copies or launches != 1:
            raise ValueError(f"unexpected online transfers {copies} or graph launches {launches}")
        return {"resident_kernels": count, "graph_launches": launches, "online_copies": copies,
                "device_interval_s": (end-start)*1e-9}


def run(args):
    gpu = read(args.gpu)
    cpu = read(args.cpu, len(gpu) if args.prefix else None)
    if cpu.dtype.names != gpu.dtype.names:
        raise ValueError("response columns differ")
    if len(cpu) != len(gpu) or not np.allclose(cpu["time_s"], gpu["time_s"], rtol=0, atol=1e-10):
        raise ValueError("missing or misaligned output frames")
    report = {"frames": len(gpu), "final_time_s": float(gpu["time_s"][-1]), "prefix": args.prefix,
              "gpu_wall_s": float(stats(args.gpu)["wall_s"])}
    report["cpu_full_run_wall_s" if args.prefix else "cpu_wall_s"] = float(stats(args.cpu)["wall_s"])
    timings_path = args.gpu.with_suffix(".gpu.csv")
    if timings_path.is_file():
        with timings_path.open(newline="") as stream:
            timings = next(csv.DictReader(stream))
        steps = int(timings["steps"])
        if steps < 1 or steps != len(gpu) - 1:
            raise ValueError("step timings and response cover different numbers of steps")
        duration = float(gpu["time_s"][-1] - gpu["time_s"][0])
        dt = duration / steps
        if not np.allclose(np.diff(gpu["time_s"]), dt, atol=1e-12, rtol=1e-8):
            raise ValueError("step deadline comparison requires uniform frame spacing")
        missed = int(timings["missed_deadlines"])
        wall = float(timings["online_s"])
        report["realtime"] = {
            "steps": steps, "deadline_s": dt, "duration_s": duration, "online_s": wall,
            "wall_over_simulated": wall / duration, "missed_deadlines": missed,
            "p50_s": float(timings["p50_step_s"]), "p99_s": float(timings["p99_step_s"]),
            "max_s": float(timings["max_step_s"]),
            "kept_up": wall <= duration, "all_steps_on_time": missed == 0,
        }
    for name, path in (("cpu", args.cpu), ("gpu", args.gpu)):
        report[name + "_response_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    columns = report["columns"] = {}
    if args.profile:
        report["cuda_trace"] = audit_transfers(args.profile, int(stats(args.gpu)["steps"]))
    for name in cpu.dtype.names:
        x, y = cpu[name], gpu[name]
        good = np.isfinite(x) & np.isfinite(y)
        delta = y[good] - x[good]
        if np.any(np.isinf(x)) or np.any(np.isinf(y)):
            raise ValueError(f"infinite value in {name}")
        entry = columns[name] = {
            "finite_mask_mismatches": int(np.count_nonzero(np.isfinite(x) != np.isfinite(y))),
            "max_abs": float(np.max(np.abs(delta))) if delta.size else None,
            "rms": float(np.sqrt(np.mean(delta**2))) if delta.size else None,
        }
        if name.endswith("_hit"):
            entry.update(disagreeing_frames=int(np.count_nonzero(x != y)),
                         cpu_switches=int(np.count_nonzero(np.diff(x))),
                         gpu_switches=int(np.count_nonzero(np.diff(y))))
        if name.endswith("_normal_N") or name.startswith("generalized_force"):
            ix, iy = np.argmax(np.abs(x)), np.argmax(np.abs(y))
            entry.update(cpu_peak=float(x[ix]), gpu_peak=float(y[iy]),
                         cpu_peak_s=float(cpu["time_s"][ix]), gpu_peak_s=float(gpu["time_s"][iy]))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.plot:
        plot_responses(args, cpu, gpu)
    if args.require_realtime:
        timing = report.get("realtime", {})
        if args.prefix or not timing.get("kept_up") or not timing.get("all_steps_on_time"):
            raise SystemExit("real-time acceptance failed; see comparison.json")


def plot_responses(args, cpu, gpu):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = cpu.dtype.names
    rows = [
        ("Shaft speed [rad/s]", [k for k in names if k.startswith("v") and k[1:].isdigit()]),
        ("Normal force [N]", [k for k in names if k.endswith("_normal_N")]),
        ("Generalized load", [k for k in names if k.startswith("generalized_force")]),
    ]
    rows = [(label, keys) for label, keys in rows if keys]
    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 2.6*len(rows)), sharex=True, squeeze=False,
                             layout="constrained")
    for (label, keys), (overlay, error) in zip(rows, axes):
        for index, key in enumerate(keys):
            color = f"C{index%10}"
            overlay.plot(cpu["time_s"], cpu[key], color=color, alpha=.6, lw=1.3, label=f"{key} CPU")
            overlay.plot(gpu["time_s"], gpu[key], color=color, lw=.85, ls="--", label=f"{key} GPU")
            error.plot(gpu["time_s"], gpu[key]-cpu[key], color=color, lw=.8)
        overlay.set_ylabel(label)
        overlay.legend(fontsize=7, ncol=2)
        for ax in (overlay,error):
            ax.grid(alpha=.2)
            ax.margins(x=0)
    axes[0,0].set_title("Frozen CPU and resident GPU")
    axes[0,1].set_title("GPU minus CPU")
    for ax in axes[-1]:
        ax.set_xlabel("Simulation time [s]")
    fig.savefig(args.output / "responses.png", dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("cpu", "gpu", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--prefix", action="store_true", help="compare a diagnostic prefix only")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--require-realtime", action="store_true",
                        help="require a full response, wall time within the simulated duration and zero missed steps")
    parser.add_argument("--profile", type=Path, help="Nsight Systems SQLite export with CUDA graph node tracing")
    run(parser.parse_args())
