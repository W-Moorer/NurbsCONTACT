"""Collect verified CPU histories and render the November 2026 response comparison."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AES = ROOT / "results/aes"
OUT = AES / "cpu-responses"
BUILD = ROOT / "build"
METHODS = ("mesh-obb", "cad-occt", "chen-2024", "chen-2026")
BLUE, ORANGE, INK = "#17678A", "#AB4C2F", "#27313C"


def report(path):
    return json.loads(path.read_text(encoding="utf-8"))


def trace(path, metadata):
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["trace_sha256"]:
        raise ValueError(f"Trace differs from the recorded run: {path}")
    return np.genfromtxt(path, delimiter=",", names=True)


def add_channels(columns, data, label, channels):
    t = columns["time_s"]
    if len(t) != len(data) or not np.allclose(t, data["time_s"], rtol=0, atol=1e-13):
        raise ValueError("Incompatible output grids")
    for target, source in channels.items():
        columns[f"{target}_{label}"] = data[source]


def save_columns(name, columns):
    values = np.column_stack(list(columns.values()))
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite figure data in {name}")
    np.savetxt(OUT / f"{name}.csv", values, delimiter=",", header=",".join(columns), comments="", fmt="%.17g")


def collect():
    sys.path.insert(0, str(ROOT / "benchmarks/engines"))
    from rigid import LEGACY, exact
    from resident_cam import reference

    contact = AES / "general-contact"
    impact = report(contact / "surface-methods-impact.json")
    for run in impact["runs"]:
        method = run["method"]
        stem = method
        if method == "mesh-obb":
            mesh = run["mesh_per_sphere"]
            stem = f"{method}-{mesh['triangles']}-{mesh['mesh_sha256'][:10]}"
        data = trace(BUILD / "chen26/impact-final" / f"{stem}.csv", run)
        if method == impact["runs"][0]["method"]:
            columns = {"time_s": data["time_s"]}
            q, f = exact("spheres", data["time_s"], LEGACY)
            columns.update(q_m_ref=q[:, 1], force_N_ref=f)
        add_channels(columns, data, method.replace("-", "_"), {"q_m": "q1", "force_N": "p0_normal_N"})
    save_columns("spheres", columns)

    for name, methods in (("cam-cycle", METHODS), ("cam-long", ("mesh-obb",))):
        for method in methods:
            if name == "cam-cycle":
                rec = report(contact / f"cam-contact-cycle-{method}.json")
                path = BUILD / f"chen26/cam-{method}/response.csv"
            else:
                rec = report(AES / "cpu-mesh/cam/compact-planar.json")
                path = BUILD / "cpu-contact/cam/compact-planar-dynamics-matched/response.csv"
            data = trace(path, rec)
            t = data["time_s"]
            if method == methods[0]:
                sol, _ = reference(rec["duration_s"], 0., .001)
                theta, _, q, _ = sol.sol(t)
                f = 5000 * np.maximum(0., .105 + .024 * np.sin(theta) - .115 - q)
                columns = dict(time_s=t, q_m_ref=q, force_N_ref=f)
            add_channels(columns, data, method.replace("-", "_"), {"q_m": "q1", "force_N": "p0_normal_N"})
            np.testing.assert_allclose(np.sqrt(np.mean((data["q1"]-columns["q_m_ref"])**2)),
                                       rec["follower_position_rmse_m"], rtol=1e-8)
        save_columns(name, columns)

    sweep = report(AES / "cpu-mesh/memory/light-mesh-sweep.json")
    work = []
    for case, cases in sweep["cases"].items():
        channels = dict(input_rad_s="v0", stage1_N="p0_normal_N")
        if case == "reducer":
            channels.update(middle_rad_s="v1", output_rad_s="v2", stage2_N="p1_normal_N")
        else:
            channels["output_rad_s"] = "v1"
        for profile, directory in (("light", "sparse"), ("preview", "preview")):
            rec = cases[profile]
            path = BUILD / f"cpu-contact/{case}-{directory}/run/response.csv"
            data = trace(path, rec)
            if profile == "light":
                if rec["trace_sha256"] != rec["compared_with_compact"]["reference_trace_sha256"]:
                    raise ValueError(f"{case} compact and light histories differ")
                columns = {"time_s": data["time_s"]}
            add_channels(columns, data, profile, channels)
            with path.with_suffix(".stats.csv").open(newline="") as stream:
                stats = next(csv.DictReader(stream))
            np.testing.assert_allclose(float(stats["wall_s"]), rec["integration_s"], rtol=1e-12)
            work.append(dict(case=case, profile=profile, trace_sha256=rec["trace_sha256"],
                             **{k: stats[k] for k in ("wall_s", "setup_s", "force_evaluations",
                                                      "mesh_boolean_queries", "mesh_boolean_s",
                                                      "mesh_detection_s", "mesh_placement_s")}))
        save_columns(case, columns)
    with (OUT / "gear-work.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=work[0].keys())
        writer.writeheader()
        writer.writerows(work)

    solvers = report(AES / "cpu-solvers/impact-chen-2026.json")
    for run in solvers["runs"]:
        law = run["solver"]
        data = trace(BUILD / f"chen26/spheres/{law}.csv", run)
        t, f = data["time_s"], data["p0_normal_N"]
        if law == solvers["runs"][0]["solver"]:
            columns = {"time_s": t}
        label = law.replace("-", "_")
        add_channels(columns, data, label, {"v_m_s": "v1"})
        step_load = .5*(f[:-1]+f[1:]) if law == "penalty" else f[1:]
        impulse = np.r_[0., np.cumsum(np.diff(t)*step_load)]
        np.testing.assert_allclose(impulse[-1], run["normal_impulse_Ns"], atol=1e-12)
        columns[f"impulse_Ns_{label}"] = impulse
    save_columns("response-laws", columns)


def read(name):
    return np.genfromtxt(OUT / f"{name}.csv", delimiter=",", names=True)


def axes_grid(rows, cols, width, height):
    return plt.subplots(rows, cols, figsize=(width, height), layout="constrained", squeeze=False)


def panel(ax, title, ylabel, xlabel):
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", color="#E2E6E9", lw=.55)
    ax.margins(x=0, y=.10)


def geometry_lines(ax, t, data, quantity, factor, residual=False):
    ref = data[quantity + "_ref"]
    for tag, name, color, style in (("mesh_obb", "三角网格 OBB", BLUE, "-"),
                                    ("cad_occt", "STEP", ORANGE, "--")):
        y = data[quantity + "_" + tag] - (ref if residual else 0.)
        ax.plot(t, y*factor, color=color, ls=style, lw=1.55, label=name)
    if not residual:
        ax.plot(t[::max(1, len(t)//18)], factor*ref[::max(1, len(t)//18)], ls="none", marker="o",
                ms=3.4, mfc="white", mec=INK, mew=.85, label="参考解")
    else:
        ax.axhline(0, color=INK, lw=.6, zorder=0)


def validation():
    fig, axes = axes_grid(3, 2, 8.3, 6.7)
    for j, (name, body, title) in enumerate((("spheres", "入射球", "双球碰撞"),
                                             ("cam-cycle", "从动件", "偏心凸轮"))):
        data = read(name)
        t = data["time_s"] * 1000
        for i, (quantity, factor, residual, label) in enumerate((
                ("q_m", 1000, False, body+"位移 (mm)"),
                ("force_N", 1, False, "法向接触力 (N)"),
                ("q_m", 1e6, True, body+"位移差 (µm)"))):
            ax = axes[i, j]
            geometry_lines(ax, t, data, quantity, factor, residual)
            desc = ("位移", "接触力", "仿真位移减参考位移")[i]
            panel(ax, f"{chr(97+2*i+j)}    {title} · {desc}", label, "时间 (ms)" if i == 2 else "")
            if i == 0:
                ax.legend(loc="best")
    return fig


def cam_history():
    fig, axes = axes_grid(3, 1, 8.3, 6.4)
    data = read("cam-long")
    t = data["time_s"]
    for i, (col, unit, title, factor) in enumerate((("q_m", "位移 (mm)", "从动件位移", 1000),
                                                    ("force_N", "力 (N)", "法向接触力", 1))):
        ax = axes[i, 0]
        ax.plot(t, data[col+"_mesh_obb"]*factor, color=BLUE, lw=1.2, label="三角网格 OBB")
        ax.plot(t, data[col+"_ref"]*factor, color=ORANGE, ls="--", lw=1.0, label="光滑凸轮参考解")
        panel(ax, f"{chr(97+i)}    {title}", unit, "时间 (s)" if i else "")
        if not i:
            ax.legend(ncol=2, loc="upper center")
    ax = axes[2, 0]
    rec = report(AES / "cpu-mesh/cam/compact-planar.json")
    ref = np.asarray(rec["reference_event_s"])
    on = data["force_N_mesh_obb"] > 0
    idx = np.flatnonzero(np.diff(on))
    brackets = np.column_stack([t[idx], t[idx+1]])
    np.testing.assert_allclose(brackets, rec["mesh_event_brackets_s"], atol=1e-12)
    delta = (brackets.mean(axis=1)-ref)*1000
    ax.axhline(0, lw=.7, color=INK)
    for enter, name, marker, color in ((True, "接触", "o", BLUE), (False, "脱开", "^", ORANGE)):
        mask = on[idx+1] == enter
        ax.scatter(ref[mask], delta[mask], s=13, marker=marker, color=color, label=name)
    ax.set_xlim(0, t[-1])
    panel(ax, "c    按发生顺序配对的切换时差", "网格时刻减参考时刻 (ms)", "参考切换时刻 (s)")
    ax.legend(ncol=2, loc="upper left")
    return fig


def mechanisms():
    fig, axes = axes_grid(2, 3, 10.5, 5.2)
    cases = (("spur", "input_rad_s", "stage1_N", "直齿轮副", "输入轴"),
             ("reducer", "middle_rad_s", "stage1_N", "减速器一级", "中间轴"),
             ("reducer", "output_rad_s", "stage2_N", "减速器二级", "输出轴"))
    for j, (name, shaft, stage, title, body) in enumerate(cases):
        data = read(name)
        for i, (col, scale, quantity, label) in enumerate(((shaft, 1., body+"转速", "角速度 (rad/s)"),
                                                          (stage, .001, "法向接触力", "力 (kN)"))):
            ax = axes[i, j]
            for profile, tag, c, ls in (("light", "默认网格", BLUE, "-"), ("preview", "预览网格", ORANGE, "--")):
                ax.plot(data["time_s"]*1000, data[col+"_"+profile]*scale, lw=1.25, color=c, ls=ls, label=tag)
            panel(ax, f"{chr(97+3*i+j)}    {title} · {quantity}", label, "时间 (ms)" if i else "")
            if i == j == 0:
                ax.legend(ncol=2, loc="best")
    return fig


def response_laws():
    data = read("response-laws")
    fig, axes = axes_grid(1, 2, 8.3, 2.95)
    for i, (col, title, unit) in enumerate((("v_m_s", "入射球速度", "速度 (m/s)"),
                                           ("impulse_Ns", "累计法向冲量", "冲量 (N·s)"))):
        ax = axes[0, i]
        for law, label, color, style in (("penalty", "罚法 / HHT", BLUE, "-"), ("sap", "SAP", ORANGE, "--"),
                                         ("ncp", "硬接触（NCP）", INK, ":")):
            ax.plot(data["time_s"]*1000, data[col+"_"+law], label=label, color=color, ls=style, lw=1.6)
        panel(ax, f"{chr(97+i)}    {title}", unit, "时间 (ms)")
        if not i:
            ax.legend(loc="lower right")
    return fig


def render():
    candidates = ["Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC"]
    available = set(font_manager.get_font_names())
    family = next((font for font in candidates if font in available), None)
    if family is None:
        raise RuntimeError("Install a Simplified Chinese font for the response figures")
    mpl.rcParams.update({"font.family": family, "font.size": 8.5, "axes.titlesize": 10,
                         "legend.fontsize": 8, "legend.frameon": False, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": .7, "axes.labelcolor": INK,
                         "text.color": INK, "xtick.color": INK, "ytick.color": INK,
                         "pdf.fonttype": 42, "path.simplify": False})
    with PdfPages(OUT / "response-curves.pdf", metadata={"CreationDate": None}) as pdf:
        for stem, build in (("contact-validation", validation), ("cam-history", cam_history),
                            ("gear-response", mechanisms), ("impact-laws", response_laws)):
            fig = build()
            fig.savefig(OUT / (stem + ".png"), dpi=200)
            pdf.savefig(fig)
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true", help="refresh figure data from the local benchmark traces")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.collect:
        collect()
    render()
