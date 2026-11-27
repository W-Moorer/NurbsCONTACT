"""Compare complete CPU/GPU response histories and render the measured phase costs."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu", type=Path)
    parser.add_argument("gpu", type=Path)
    parser.add_argument("timing", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--execution-reference", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cpu = np.genfromtxt(args.cpu, delimiter=",", names=True)
    gpu = np.load(args.gpu)
    if len(cpu) != len(gpu) or np.max(abs(cpu["time_s"] - gpu[:, 0])) > 1e-8:
        raise ValueError("Full histories must use the same time grid")
    n = sum(x.startswith("generalized_force") for x in cpu.dtype.names)
    pairs = (gpu.shape[1] - 1 - 4 * n) // 3
    report = {
        "cpu_sha256": hashlib.sha256(args.cpu.read_bytes()).hexdigest(),
        "gpu_sha256": hashlib.sha256(args.gpu.read_bytes()).hexdigest(),
        "frames": len(cpu),
        "duration_s": float(gpu[-1, 0]),
        "errors": {},
    }
    reference = np.empty_like(gpu)
    reference[:, 0] = cpu["time_s"]
    for i in range(n):
        for j, key in enumerate((f"q{i}", f"v{i}", f"a{i}", f"generalized_force{i}")):
            reference[:, 1 + 4 * i + j] = cpu[key]
    for i in range(pairs):
        for j, key in enumerate((f"p{i}_gap_m", f"p{i}_normal_N", f"p{i}_hit")):
            reference[:, 1 + 4 * n + 3 * i + j] = cpu[key]
    labels = (
        ["time"]
        + [f"{key}{i}" for i in range(n) for key in ("q", "v", "a", "force")]
        + [
            f"{key}{i}"
            for i in range(pairs)
            for key in ("gap", "normal_load", "contact")
        ]
    )
    for col, key in enumerate(labels[1:], 1):
        a, b = reference[:, col], gpu[:, col]
        finite = np.isfinite(a) & np.isfinite(b)
        delta = b[finite] - a[finite]
        report["errors"][key] = {
            "max_abs": float(np.max(abs(delta))) if len(delta) else None,
            "rms": float(np.sqrt(np.mean(delta**2))) if len(delta) else None,
            "reference_rms": (
                float(np.sqrt(np.mean(a[finite] ** 2))) if finite.any() else None
            ),
            "finite_mask_mismatches": int(
                np.count_nonzero(np.isfinite(a) != np.isfinite(b))
            ),
        }
    if args.execution_reference:
        old = np.load(args.execution_reference)
        report["same_response_as_execution_reference"] = bool(
            np.array_equal(old, gpu, equal_nan=True)
        )
        report["execution_reference_sha256"] = hashlib.sha256(
            args.execution_reference.read_bytes()
        ).hexdigest()
    report["cpu_equivalence_accepted"] = False
    report["reason"] = (
        "Near-coplanar triangle classifications and resulting load peaks remain under investigation."
    )
    np.savez_compressed(
        args.out / "responses.npz", cpu=reference, gpu=gpu, columns=labels
    )
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(
        n + 1, 2, figsize=(13, 3 * (n + 1)), sharex=True, layout="constrained"
    )
    t = gpu[:, 0]
    for i in range(n):
        for column, title in (
            (2 + 4 * i, "Angular velocity (rad/s)"),
            (4 + 4 * i, "Generalized torque (N m)"),
        ):
            ax = axes[i, 0 if column == 2 + 4 * i else 1]
            ax.plot(
                t, reference[:, column], color="#214f79", lw=0.85, label="CPU reference"
            )
            ax.plot(
                t,
                gpu[:, column],
                color="#dd7134",
                lw=0.65,
                alpha=0.8,
                label="GPU graph",
            )
            ax.set_ylabel(f"Coordinate {i+1}\n{title}")
            ax.grid(alpha=0.18)
    for i in range(min(pairs, 2)):
        ax = axes[-1, i]
        column = 2 + 4 * n + 3 * i
        ax.plot(
            t, reference[:, column], color="#214f79", lw=0.85, label="CPU reference"
        )
        ax.plot(
            t, gpu[:, column], color="#dd7134", lw=0.65, alpha=0.8, label="GPU graph"
        )
        ax.set_ylabel(f"Contact pair {i+1}\nNormal load (N)")
        ax.grid(alpha=0.18)
    for ax in axes.flat:
        ax.set_xlim(0, t[-1])
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Complete 0–5 s responses — peak-load equivalence has not passed")
    fig.savefig(args.out / "responses.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(
        n, 2, figsize=(13, 2.7 * n), sharex=True, layout="constrained"
    )
    axes = np.asarray(axes).reshape(n, 2)
    for i in range(n):
        for j, col in enumerate((2 + 4 * i, 4 + 4 * i)):
            ax = axes[i, j]
            ax.axhline(0, color="#999999", lw=0.5)
            ax.plot(t, gpu[:, col] - reference[:, col], color="#214f79", lw=0.7)
            ax.set_ylabel(
                f"Coordinate {i+1}\n"
                + (
                    "Velocity difference (rad/s)"
                    if j == 0
                    else "Torque difference (N m)"
                )
            )
            ax.set_xlim(0, t[-1])
            ax.grid(alpha=0.18)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle("GPU minus CPU — complete response differences, without smoothing")
    fig.savefig(args.out / "response-differences.png", dpi=160)
    plt.close(fig)
    timing = json.loads(args.timing.read_text())
    s = timing["status"]
    parts = [s[5], s[6], s[7], s[8], timing["online_s"] - sum(s[5:9])]
    fig, ax = plt.subplots(figsize=(10, 3.5), layout="constrained")
    names = [
        "Broad phase",
        "Triangle narrow phase",
        "Refinement / grouping / placement",
        "Contact information",
        "Solver / scheduling / other",
    ]
    ax.barh(names, parts, color=["#497799", "#679fb4", "#db9c55", "#b6614d", "#89888d"])
    for i, v in enumerate(parts):
        ax.text(v + 0.3, i, f"{v:.2f} s", va="center")
    ax.set_xlim(0, max(parts) * 1.22)
    ax.set_xlabel("Online time for 5 s simulation (s)")
    ax.invert_yaxis()
    ax.set_title("RTX 3070 · 50,000 steps · 0.1 ms · current-pose triangle contacts")
    fig.savefig(args.out / "phase-times.png", dpi=180)
    plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k != "errors"}, indent=2))


if __name__ == "__main__":
    main()
