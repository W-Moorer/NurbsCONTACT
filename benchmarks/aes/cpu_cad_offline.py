"""Report native CPU cache/reuse measurements on the five-second reducer benchmark."""

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cpu_cad_pressure import read_rows
from cpu_cad_pressure_report import history, timing
from cpu_cad_pruning import fixed_loads, fixed_replay, save, verified_dynamics


def report(scene, work, reference, out, cache):
    out.mkdir(exist_ok=True, parents=True)
    stats = {}
    traces = {}
    for name in ("before", "after", "mesh", "repeat-after", "repeat-before", "all-orders"):
        folder = work / name
        doc, startup = verified_dynamics(scene, folder, "mesh-obb" if name == "mesh" else "cad-patch")
        traces[name] = history(folder / "response.csv", scene)
        s = doc["statistics"]
        stats[name] = dict(
            process_s=doc["process_s"],
            setup_s=float(s["setup_s"]),
            first_20_steps_s=float(startup["integration_s"]),
            steps=doc["last_accepted"]["step"],
        )
        stats[name].update(
            {
                k: float(v)
                for k, v in s.items()
                if k.startswith(("cad_patch_", "cad_window_", "cad_skipped_", "cad_path_"))
            }
        )
        shutil.copyfile(folder / "run.json", out / (name + "-run.json"))
    with np.load("results/aes/cpu-foundation/reducer-physics/closed-loop.npz") as rd:
        t = rd["time_s"]
        rd_torque = rd["rd_output_torque_Nm"]
    for name, values in traces.items():
        np.testing.assert_allclose(t, values["time_s"], rtol=0, atol=1e-10)
        shaft = values["output_torque_Nm"]
        stats[name]["torque_0p5ms_Nm"] = shaft[5]
        stats[name]["torque_rms_vs_recurdyn_Nm"] = np.mean((shaft - rd_torque) ** 2) ** 0.5
    for new, old in (("after", "before"), ("repeat-after", "repeat-before"), ("all-orders", "repeat-before")):
        errors = {}
        for key in ("output_torque_Nm", "speed", "te_mrad", "normal_N"):
            delta = traces[new][key] - traces[old][key]
            errors[key] = {
                "rms": np.sqrt(np.mean(delta**2, axis=0)).tolist(),
                "max": np.max(np.abs(delta), axis=0).tolist(),
            }
        stats[new]["response_change"] = errors
    save(out / "dynamics.json", stats)
    np.savez_compressed(
        out / "responses.npz",
        time_s=t,
        recurdyn_torque_Nm=rd_torque,
        **{name: values["output_torque_Nm"] for name, values in traces.items()},
        **{key: traces["all-orders"][key] for key in ("speed", "te_mrad", "normal_N")},
    )

    prep = {}
    for key in ("cold", "warm"):
        raw = json.loads((work / (key + ".json")).read_text())
        prep[key] = [
            dict(
                process_s=r["process_s"],
                native_setup_s=float(r["prepare"].split()[1]),
                **json.loads(r["report"]),
            )
            for r in raw
        ]
    files = list(cache.glob("*.bin"))
    prep["files"] = len(files)
    prep["bytes"] = sum(p.stat().st_size for p in files)
    save(out / "preparation.json", prep)
    replay = work / "replay"
    groups = fixed_replay(reference, replay)
    for stage, row in enumerate(groups):
        builds = paths = 0
        for frame in read_rows(replay / f"indexed-{stage}.jsonl"):
            builds += frame["window_builds"]
            paths += frame["reused_paths"]
        row.update(window_builds=builds, window_reuses=2 * row["poses"] - builds, reused_paths=paths)
    save(out / "same-poses.json", groups)
    fixed_loads(SimpleNamespace(scene=scene, current_replay=replay, out=out))
    shutil.copyfile(work / "sphere-report" / "sphere.json", out / "sphere.json")
    timing(SimpleNamespace(work=replay, out=out))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), layout="constrained")
    for i, name in enumerate(("before", "after", "all-orders", "mesh")):
        names = [name, "repeat-" + name] if name in ("before", "after") else [name]
        for j, k in enumerate(names):
            v = stats[k]["process_s"]
            y = i - 0.12 + 0.24 * j
            axes[0].scatter(v, y, color=("#555f70", "#548485", "#147465", "#b86d39")[i])
            axes[0].annotate(
                f"{v:.0f}", (v, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8
            )
    axes[0].set(
        yticks=range(4),
        yticklabels=["Previous CAD", "Base tables", "All orders (ours)", "CPU OBB mesh"],
        xlabel="5 s run, wall time (s)",
        xlim=(0, max(s["process_s"] for s in stats.values()) * 1.18),
    )
    axes[0].invert_yaxis()
    for name, values, color in [
        ("RecurDyn", rd_torque, "#576170"),
        ("CPU mesh", traces["mesh"]["output_torque_Nm"], "#b86d39"),
        ("CAD (ours)", traces["all-orders"]["output_torque_Nm"], "#147465"),
    ]:
        axes[1].plot(t, values, lw=0.6, color=color, alpha=0.78, label=name)
    axes[1].set(xlim=(0, 5), xlabel="Time (s)", ylabel="Output torque (N m)")
    axes[1].legend(frameon=False, fontsize=8, ncol=3)
    axes[2].plot(
        t,
        traces["all-orders"]["output_torque_Nm"] - traces["repeat-before"]["output_torque_Nm"],
        lw=0.65,
        color="#147465",
    )
    axes[2].set(xlim=(0, 5), xlabel="Time (s)", ylabel="New minus previous torque (N m)")
    fig.savefig(out / "cpu-response.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("scene", "work", "reference", "out", "cache"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    report(**vars(args))
