"""Plot independently checked displacement errors and full-run times for the cam."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator


def import_fine(work, root):
    rows = []
    prep = json.loads((work / "fine-prep.json").read_text())
    for line in (work / "remote-fine.txt").read_text().splitlines():
        name, count, error, wall, events, event_error = line.split()
        method, level = name.split("-")
        data = json.loads((work / f"3090-{name}.json").read_text())
        (trial,) = data["runs"]
        if int(count) != 160001 or trial["steps"] != 160000 or trial["newton_failure"]:
            raise ValueError(f"Incomplete fine control {name}")
        rows.append(
            dict(
                name=name,
                method=method,
                deflection_um=float(level),
                gpu=data["gpu"],
                rms_um=float(error),
                online_s=float(wall),
                events=int(events),
                max_event_us=float(event_error),
                triangles=data["triangles"],
                samples=data["samples"],
                reserved_MiB=trial["cupy_pool_reserved_bytes"] / 2**20,
                setup_s=data["setup_s"],
                geometry_preparation_s=prep[f"deck-{level}.npz"],
                runs=[trial],
            )
        )
    (root / "fine-mesh-3090.json").write_text(json.dumps(rows, indent=2) + "\n")


def plot(root):
    def read(name):
        return json.loads((root / f"{name}.json").read_text())

    main, tuned, step = (read(s) for s in ("accuracy-cost", "search-controls", "step-controls"))
    fine, temporal, symmetric = (read(s) for s in ("fine-mesh-3090", "time-error", "symmetric-controls"))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), layout="constrained")
    a, b = axes
    palette = {"cad": "#00726b", "mesh": "#314b73", "vertices": "#aa5028"}

    def curve(rows, label, color, marker):
        rows = sorted(rows, key=lambda r: -r["deflection_um"])
        a.plot(
            [r["deflection_um"] for r in rows],
            [r["rms_um"] for r in rows],
            marker=marker,
            lw=1.3,
            ms=4,
            color=color,
            label=label,
        )

    curve(
        [r for r in main + fine if r["method"] == "mesh"], "Flat face and edge probes", palette["mesh"], "o"
    )
    curve([r for r in tuned + fine if r["method"] == "vertices"], "Mesh vertices", palette["vertices"], "s")
    curve(
        [r for r in step if r["step_us"] == 125 and r["method"] == "cad" and r["backend"] == "grid"],
        "STEP face and edge probes",
        palette["cad"],
        "D",
    )
    a.axhline(temporal[0]["rms_um"], color=".4", ls="--", lw=1.2, label="Analytic geometry + HHT")
    a.set(
        xscale="log",
        xlim=(1.35, 0.003),
        xlabel="Requested source deflection (µm)",
        ylabel="Follower displacement RMS error (µm)",
        title="Geometry refinement at 125 µs",
    )
    a.legend(loc="upper right", fontsize=8, frameon=False)
    symmetric += read("symmetric-mesh-125")
    for method in ("cad", "vertices"):
        same = sorted(
            (r for r in symmetric if r["method"] == method and r["backend"] == "sdf"),
            key=lambda r: r["step_us"],
        )
        b.plot(
            [r["online_s"] for r in same],
            [r["rms_um"] for r in same],
            color=palette[method],
            lw=0.9,
            ls="--",
            alpha=0.7,
            zorder=1,
        )
    screening = read("mesh-density")
    confirmed = read("density-confirmation")
    screening = [r for r in screening if r["name"] not in {c["name"] for c in confirmed}]
    for step_size in (62.5, 125):
        rows = {
            r["triangles"][-1]: r
            for r in screening + symmetric + confirmed
            if r["step_us"] == step_size and r["method"] == "vertices" and r["backend"] == "sdf"
        }
        ordered = [r for _, r in sorted(rows.items())]
        b.plot(
            [r["online_s"] for r in ordered],
            [r["rms_um"] for r in ordered],
            ":",
            lw=0.8,
            color=palette["vertices"],
            alpha=0.55,
            zorder=1,
        )
    for r in screening + symmetric + confirmed:
        if r["status"] != "complete":
            continue
        marker = "s" if r["backend"] == "sdf" else "o"
        b.scatter(
            r["online_s"],
            r["rms_um"],
            marker=marker,
            s=58,
            edgecolor=palette[r["method"]],
            facecolor=palette[r["method"]] if r["step_us"] == 125 else "white",
            linewidth=1.5,
        )
    b.set(
        xlabel="Online time for 20 s of motion (s)",
        ylabel="Follower displacement RMS error (µm)",
        title="Bidirectional contact on RTX 3070",
    )
    b.set_yscale("log")
    b.set_ylim(0.071, 1.10)
    ticks = [0.08, 0.10, 0.15, 0.20, 0.40, 0.80, 1.0]
    b.yaxis.set_major_locator(FixedLocator(ticks))
    b.yaxis.set_major_formatter(FixedFormatter([f"{x:g}" for x in ticks]))
    b.margins(x=0.11)
    handles = [
        Line2D([], [], lw=2, color=palette[key], label=lab)
        for key, lab in (
            ("cad", "STEP probes (ours)"),
            ("vertices", "Mesh vertices"),
            ("mesh", "Flat probes"),
        )
    ]
    handles.extend(
        Line2D([], [], marker=m, lw=0, color=".25", label=lab, ms=6)
        for m, lab in (("o", "Triangle index"), ("s", "SDF"))
    )
    handles.append(Line2D([], [], marker="o", lw=0, color=".25", mfc="white", label="Open = 62.5 µs"))
    b.legend(
        handles=handles, fontsize=8, frameon=True, fancybox=False, edgecolor="none", ncol=2, loc="upper right"
    )
    for ax in axes:
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(axis="y", lw=0.5, color=".87")
        ax.set_axisbelow(True)
    for extension in ("pdf", "png"):
        fig.savefig(root / f"accuracy-cost.{extension}", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--import-fine", type=Path)
    args = p.parse_args()
    if args.import_fine:
        import_fine(args.import_fine, args.root)
    plot(args.root)
