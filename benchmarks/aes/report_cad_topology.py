"""Archive progressive-CAD trials, including rejected experiments and full histories."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("runs", "control", "selected", "cpu", "scene", "out"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    experiments = {}
    for folder in sorted(a.runs.iterdir()):
        source = folder / "run.json"
        if not source.is_file():
            continue
        info = json.loads(source.read_text())
        if info.get("completed") and "response_sha256" in info:
            if digest(folder / "response.csv") != info["response_sha256"]:
                raise ValueError(f"Response changed after timing: {folder}")
        experiments[folder.name] = info
        dest = a.out / "runs" / folder.name
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("run.json", "console.log", "response.gpu.csv", "response.stats.csv"):
            if (folder / name).is_file():
                shutil.copy2(folder / name, dest / name)
    control = json.loads((a.control / "run.json").read_text())
    selected = json.loads((a.selected / "run.json").read_text())
    mesh = json.loads((a.runs / "mesh/run.json").read_text())
    for info in (control, selected, mesh):
        if not info["completed"] or info["step_s"] != .0001 or info["scene_sha256"] != digest(a.scene):
            raise ValueError("Expected completed same-scene 0.1 ms comparisons")
        if info.get("requested_duration_s", info.get("duration_s")) != 5:
            raise ValueError("Full five-second comparison required")
    subprocess.run([sys.executable, str(Path(__file__).with_name("report_cad_segments.py")),
                    str(a.control), str(a.selected), str(a.cpu), str(a.scene),
                    str(a.out / "selected"), "--require-identical"], check=True)
    rows = []
    for name, info in experiments.items():
        if (info.get("completed") and info.get("requested_duration_s") == 5
                and info.get("step_s") == .0001 and not info.get("profiled_links")):
            rows.append((name, float(info["gpu"]["online_s"])))
    fig, ax = plt.subplots(figsize=(10, max(4, .47 * len(rows))), layout="constrained")
    names, times = zip(*rows)
    colors = ["#28787c" if name == a.selected.name else "#a8b5bc" for name in names]
    bars = ax.barh(names, times, color=colors)
    ax.bar_label(bars, fmt="%.2f s", padding=4)
    ax.axvline(mesh["online_s"], color="#cf6433", ls="--", label="GPU Mesh (inclusive timing)")
    ax.set(xlabel="Online time [s]", xlim=(0, max(times) * 1.16),
           title="Complete 5 s / 0.1 ms trials, including rejected variants")
    ax.invert_yaxis()
    ax.legend()
    fig.savefig(a.out / "trials.png", dpi=180)
    plt.close(fig)
    archive = Path(__file__).resolve().parents[2] / "results/aes/cuda-coherent-execution/backend-comparison.json"
    cpu_rows = [r for r in json.loads(archive.read_text())["rows"] if r["label"].startswith("CPU ")]
    cpu_by_label = {r["label"]: r["online_s"][0] for r in cpu_rows}
    values = [cpu_by_label["CPU mesh (archived)"], cpu_by_label["CPU CAD (archived)"], mesh["online_s"],
              float(control["gpu"]["online_s"]), float(selected["gpu"]["online_s"])]
    labels = ["CPU Mesh\narchived", "CPU CAD\narchived", "GPU Mesh", "GPU CAD\npreceding control", "GPU CAD\nretained (ours)"]
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    bars = ax.bar(labels, values, color=["#a8b5bc", "#81959e", "#cf6433", "#255e83", "#28787c"])
    ax.bar_label(bars, fmt="%.2f s", padding=4)
    ax.set(ylabel="Online time [s]", ylim=(0, max(values) * 1.14),
           title="Same reducer and main step; CPU records reused")
    fig.savefig(a.out / "backends.png", dpi=180)
    plt.close(fig)
    response = np.load(a.runs / "mesh/response.npy")
    # Three coordinates (q/v/a/Q) and two contact triples (gap/load/hit).
    required = [i for i in range(19) if i not in (13, 16)]
    if (response.shape != (50001, 19) or np.isinf(response).any()
            or not np.isfinite(response[:, required]).all()):
        raise ValueError("Incomplete or nonfinite Mesh history")
    np.savez_compressed(a.out / "mesh-response.npz", response=response)
    report = dict(experiments=experiments, selected=a.selected.name, control=a.control.name,
                  cpu_rerun=False, cpu_records=cpu_rows, cpu_archive_sha256=digest(archive), scene_sha256=digest(a.scene),
                  online_boundary="graph launch through final status/response downloads; setup and file export excluded",
                  selected_vs_control=float(control["gpu"]["online_s"]) / float(selected["gpu"]["online_s"]),
                  selected_vs_mesh=float(selected["gpu"]["online_s"]) / mesh["online_s"],
                  mesh_response_sha256=digest(a.runs / "mesh/response.npy"),
                  scope="Numerical regression and execution cost; Mesh equality of physical accuracy is not established")
    (a.out / "experiments.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
