"""Compare joint-axis response, angular impulse and power at common physical times."""

import argparse
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.signal import welch


H = 0.000125
GROUPS = (
    ("speed", ("Input speed", "Middle speed", "Output speed"), "rad/s"),
    ("twist", ("Stage 1 transmission error", "Stage 2 transmission error"), "mrad"),
    ("torque", ("Input contact torque", "Middle net contact torque", "Output contact torque"), "N·m"),
    ("power", ("Input drive power", "Power absorbed by brake"), "W"),
    ("energy", ("Rotor kinetic energy change",), "J"),
)
STYLES = {
    "native": ("RecurDyn, default", "#343f52", "-"),
    "mesh": ("Mesh", "#c27534", "--"),
    "cad": ("STEP samples (ours)", "#00817e", "-"),
}


def rms(x):
    return np.sqrt(np.mean(x**2, axis=0))


def response(omega, angle_change, inertia, load, support):
    # Interval mean torque via angular impulse. Smooth external loads use endpoint quadrature.
    external = load + support
    torque = np.diff(omega, axis=0) * inertia / H - 0.5 * (external[1:] + external[:-1])
    return dict(
        speed=omega,
        twist=(angle_change[:, :2] - 2 * angle_change[:, 1:]) * 1000,
        torque=torque,
        power=np.column_stack((load[:, 0] * omega[:, 0], -load[:, 2] * omega[:, 2])),
        energy=(0.5 * ((omega**2 - omega[0] ** 2) * inertia).sum(axis=1))[:, None],
    )


def trajectories(work):
    with np.load(work / "assembly.npz") as deck:
        inertia, spring, damping, q0 = deck["coord"].T
        nominal = deck["bodies"][[0, 1, 3], 8]
        signs = deck["bodies"][[0, 1, 3], 3]
        loads = deck["loads"]
        np.testing.assert_allclose(deck["h"], H)
    runs = {}
    gpu_files = [
        ("old", "cad-h62"),
        ("cad", "full-cad"),
        ("mesh", "full-mesh"),
        ("cad125", "local-cad125"),
        ("fine16", "fine-16"),
        ("fine32", "fine-32"),
    ]
    for name in ("cad", "mesh"):
        if (work / f"vertices-{name}.json").is_file():
            gpu_files.append((name + "_vertices", "vertices-" + name))
        pair_report = work / f"pair-{name}.json"
        if pair_report.is_file() and json.loads(pair_report.read_text())["terminated"] == "complete":
            gpu_files.append((name + "_pair", "pair-" + name))
    for key, name in gpu_files:
        report = json.loads((work / f"{name}.json").read_text())
        assert report["terminated"] == "complete" and report["setup_s"] + report["online_s"] < 1800
        stride = round(H / report["step_s"])
        np.testing.assert_allclose(stride * report["step_s"], H, rtol=0, atol=1e-14)
        data = np.load(work / f"{name}.npy")[::stride]
        assert len(data) == 1 + round(report["physical_s"] / H) and np.isfinite(data).all()
        v = data[:, 3:6]
        np.testing.assert_allclose(data[0, :3], q0)
        runs[key] = response(
            v + nominal, data[:, :3] - q0, inertia, loads[: len(v)], -spring * data[:, :3] - damping * v
        )
    checks = {}
    native_files = [
        ("native", "rd-physics", "rd-reducer-3s-125us"),
        ("native31", "rd-refined-physics", "rd-reducer-0.32s-31.25us"),
        ("native_dense_short", "rd-geometric-physics", "rd-reducer-0.32s-125us-patch0.0625"),
    ]
    if (work / "rd-switch-geometric-physics.npz").exists():
        native_files.append(
            ("native_dense", "rd-switch-geometric-physics", "rd-reducer-1.26s-125us-patch0.0625")
        )
    for key, source, motion in native_files:
        with np.load(work / f"{source}.npz") as d, np.load(work / f"{motion}.npz") as geom:
            stride = round(H / np.diff(geom["t"][:2])[0])
            t = geom["t"][::stride]
            np.testing.assert_allclose(t, np.arange(len(t)) * H, atol=1e-11, rtol=0)
            angle = d["angle_rad"][::stride]
            omega = d["global_speed_rad_s"][::stride] * signs
            np.testing.assert_allclose(omega, geom["speeds"][::stride] * signs, atol=1e-8, rtol=0)
            np.testing.assert_allclose(d["reference_rad"][::stride], t[:, None] * nominal, atol=1e-9, rtol=0)
            support = d["support_raw_TZ_Nm"][::stride]
            predicted = -spring * (angle - t[:, None] * nominal + q0) - damping * (omega - nominal)
            np.testing.assert_allclose(support, predicted, atol=3e-8, rtol=0)
            applied = np.zeros_like(omega)
            applied[:, [0, 2]] = d["load_raw_TZ_Nm"][::stride]
            checks[key] = dict(
                max_support_identity_difference_Nm=abs(predicted - support).max(axis=0).tolist(),
                max_applied_load_difference_Nm=abs(loads[: len(t)] - applied).max(axis=0).tolist(),
            )
            runs[key] = response(omega, angle - angle[:1], inertia, applied, support)
    if "native_dense" in runs:
        checks["native_dense_repeat_prefix"] = {}
        for key, _, _ in GROUPS:
            short = runs["native_dense_short"][key]
            repeat = runs["native_dense"][key][: len(short)]
            checks["native_dense_repeat_prefix"][key + "_max_difference"] = (
                abs(repeat - short).max(axis=0).tolist()
            )
    return runs, checks


def time_axis(key, n):
    return (np.arange(n) + (0.5 if key == "torque" else 0.0)) * H


def compare(runs):
    metrics = []
    pairs = (
        ("old", "native"),
        ("mesh", "native"),
        ("cad", "native"),
        ("cad125", "native"),
        ("cad125", "cad"),
        ("cad", "fine16"),
        ("fine16", "fine32"),
        ("native", "native31"),
        ("cad", "native31"),
        ("fine16", "native31"),
        ("fine32", "native31"),
    )
    scored = [(method, ref, None) for method, ref in pairs]
    for ref in ("native_dense_short", "native_dense"):
        if ref in runs:
            scored.extend((method, ref, None) for method in ("cad", "mesh", "native"))
            # Use the same duration when changing the independent solver's surface resolution.
            scored.extend((m, "native", len(runs[ref]["speed"])) for m in ("cad", "mesh"))
    for name in ("cad_vertices", "mesh_vertices"):
        if name in runs:
            scored.append((name, "native_dense_short", None))
    for name in ("cad_pair", "mesh_pair"):
        if name in runs:
            scored.extend(
                (name, ref, None) for ref in ("native", "native_dense_short", "native_dense") if ref in runs
            )
    for method, ref, limit in scored:
        for key, labels, unit in GROUPS:
            n = min(len(runs[method][key]), len(runs[ref][key]))
            if limit is not None:
                n = min(n, limit - (key == "torque"))
            ts = time_axis(key, n)
            for window, mask in (("full", np.ones(n, bool)), ("steady", ts >= 0.5)):
                if not mask.any():
                    continue
                a, b = runs[method][key][:n][mask], runs[ref][key][:n][mask]
                diff = a - b
                for j, label in enumerate(labels):
                    metrics.append(
                        dict(
                            method=method,
                            reference=ref,
                            window=window,
                            duration_s=round(n * H if key == "torque" else (n - 1) * H, 8),
                            channel=label,
                            unit=unit,
                            mean=a[:, j].mean(),
                            reference_mean=b[:, j].mean(),
                            bias=diff[:, j].mean(),
                            rms_difference=rms(diff[:, j]),
                            max_abs_difference=abs(diff[:, j]).max(),
                            min_value=a[:, j].min(),
                            max_value=a[:, j].max(),
                            max_time_s=ts[mask][np.argmax(a[:, j])],
                            reference_max=b[:, j].max(),
                            reference_max_time_s=ts[mask][np.argmax(b[:, j])],
                            step_change_rms=rms(np.diff(a[:, j])),
                            reference_step_change_rms=rms(np.diff(b[:, j])),
                        )
                    )
    return pd.DataFrame(metrics)


def frozen_pose_torques(work, out, replay_file, channels_file, output_file):
    """Instantaneous generalized contact torque evaluated at archived native configurations."""
    replay = pd.read_csv(work / replay_file)
    replay = replay[replay.trajectory == "recurdyn"]
    steps = np.sort(replay.step.unique())
    t = np.sort(replay.t.unique())
    ids = np.rint(t / H).astype(int)
    np.testing.assert_allclose(ids * H, t, rtol=0, atol=1e-12)
    with np.load(work / "assembly.npz") as model, np.load(work / channels_file) as d:
        signs = model["bodies"][[0, 1, 3], 3]
        native = d["global_accel_rad_s2"][ids] * signs * model["coord"][:, 0] - d["support_raw_TZ_Nm"][ids]
        native[:, [0, 2]] -= d["load_raw_TZ_Nm"][ids]
    results = {"native": native}
    for name in ("cad", "mesh"):
        subset = replay[replay.method == name].sort_values(["step", "stage", "face_a", "face_b"])
        group = subset.groupby(["step", "stage"])
        summed = group[["torque_a_Nm", "torque_b_Nm"]].sum()
        deepest = subset.loc[group.depth_mm.idxmax()].set_index(["step", "stage"])
        for model, stage in ((name, summed), (name + "_pair", deepest)):
            a, b = (stage.xs(i, level="stage").loc[steps] for i in (1, 2))
            results[model] = np.column_stack((a.torque_a_Nm, a.torque_b_Nm + b.torque_a_Nm, b.torque_b_Nm))
    frame = pd.DataFrame(
        {
            "time_s": t,
            **{
                f"{method}_{shaft}_Nm": result[:, i]
                for method, result in results.items()
                for i, shaft in enumerate(("input", "middle", "output"))
            },
        }
    )
    frame.to_csv(out / output_file, index=False)
    return frame


def line(ax, t, series, appearance, mask, dense=False):
    label, color, style = appearance
    ax.plot(
        t[mask],
        series[mask],
        color=color,
        ls=style,
        lw=0.56 if dense else 1.15,
        alpha=0.76 if dense else 0.96,
        rasterized=dense,
        label=label,
    )


def style(axes):
    for ax in np.ravel(axes):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)


def refined_spectrum(pdf, runs, out):
    dense = runs["native_dense"]
    stop = len(dense["speed"])
    start = round(0.5 / H)
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 7.4), layout="constrained")
    rows = []
    for j, ax in enumerate(axes):
        for method, label, color, ls in (
            ("cad", "STEP, region stiffness", "#00817e", "-"),
            ("cad_pair", "STEP, body-pair stiffness", "#b46636", "--"),
            ("native", "RecurDyn, default", "#343f52", ":"),
            ("native_dense", "RecurDyn, refined", "#825d9c", "-."),
        ):
            if method not in runs:
                continue
            f, p = welch(runs[method]["speed"][start:stop, j], fs=1 / H, nperseg=4000, noverlap=2000)
            mask = (f >= 20) & (f <= 1600)
            ax.semilogy(f[mask], p[mask], color=color, ls=ls, lw=1, label=label)
            for tone in (330, 720):
                idx = np.argmin(abs(f - tone))
                rows.append(
                    dict(
                        method=method,
                        shaft=GROUPS[0][1][j],
                        frequency_Hz=f[idx],
                        duration_s=(stop - 1 - start) * H,
                        PSD_rad2_s2_per_Hz=p[idx],
                    )
                )
        ax.set_ylabel(f"{GROUPS[0][1][j]} PSD\n((rad/s)²/Hz)")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[0].set_title(f"Shaft-speed ripple, identical 0.50–{(stop-1)*H:.2f} s interval")
    axes[-1].set_xlabel("Frequency (Hz)")
    style(axes)
    pdf.savefig(fig)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "native-refinement-spectrum.csv", index=False)


def figures(runs, out, frozen, dense_frozen=None, sphere_file=None):
    dense = runs.get("native_dense", runs["native_dense_short"])
    response_styles = STYLES
    if "cad_pair" in runs and "mesh_pair" in runs:
        response_styles = {
            "native": STYLES["native"],
            "cad": ("STEP, region stiffness", "#688198", ":"),
            "mesh_pair": ("Mesh, pair stiffness", "#b46636", "--"),
            "cad_pair": ("STEP, pair stiffness (ours)", "#00817e", "-"),
        }

    def refined(ax, key, j, mask, dense_line=False):
        y = dense[key][:, j]
        t = time_axis(key, len(y))
        subset = mask[: len(t)]
        ax.plot(
            t[subset],
            y[subset],
            label="RecurDyn, refined",
            color="#825d9c",
            ls="-.",
            lw=0.6 if dense_line else 1.15,
            rasterized=dense_line,
        )

    with PdfPages(out / "mechanism-responses.pdf") as pdf:
        for key, labels, unit in GROUPS:
            t = time_axis(key, len(runs["cad"][key]))
            fig, axes = plt.subplots(
                len(labels), 2, squeeze=False, figsize=(10.2, 1.95 * len(labels) + 1.0), layout="constrained"
            )
            for j, label in enumerate(labels):
                for col, mask in enumerate((np.ones(len(t), bool), (t >= 0.097) & (t <= 0.110))):
                    ax = axes[j, col]
                    for method, appearance in response_styles.items():
                        line(ax, t, runs[method][key][:, j], appearance, mask, col == 0)
                    refined(ax, key, j, mask, col == 0)
                    ax.set_ylabel(f"{label}\n({unit})" if col == 0 else unit)
            for ax, title in zip(axes[0], ("Complete load history", "Detail near load engagement")):
                ax.set_title(title)
            axes[0, 0].legend(loc="upper right", ncol=2, fontsize=7, framealpha=0.94)
            for ax in axes[-1]:
                ax.set_xlabel("Time (s)")
            style(axes)
            pdf.savefig(fig, dpi=180)
            plt.close(fig)
        fig, axes = plt.subplots(3, 1, figsize=(8.4, 7.4), layout="constrained")
        peaks = []
        for j, ax in enumerate(axes):
            for method, (label, color, ls) in STYLES.items():
                f, psd = welch(runs[method]["speed"][4000:, j], fs=1 / H, nperseg=4000, noverlap=2000)
                mask = (f >= 20) & (f <= 1700)
                ax.semilogy(f[mask], psd[mask], label=label, color=color, ls=ls, lw=1)
                for nominal in (330, 720):
                    near = np.where(abs(f - nominal) <= 4)[0]
                    i = near[np.argmax(psd[near])]
                    peaks.append(
                        dict(
                            method=method,
                            shaft=GROUPS[0][1][j],
                            nominal_Hz=nominal,
                            peak_Hz=f[i],
                            PSD_rad2_s2_per_Hz=psd[i],
                        )
                    )
            for tone in (330, 720):
                ax.axvline(tone, color="#687885", ls=":", lw=0.8)
            ax.set_ylabel(f"{GROUPS[0][1][j]} PSD\n((rad/s)²/Hz)")
        axes[0].set_title("Shaft-speed ripple from 0.5 to 3 s • nominal mesh frequencies 330 and 720 Hz")
        axes[0].legend(frameon=False, ncol=3)
        axes[-1].set_xlabel("Frequency (Hz)")
        style(axes)
        pdf.savefig(fig)
        plt.close(fig)
        pd.DataFrame(peaks).to_csv(out / "mesh-frequency.csv", index=False)
        if "native_dense" in runs:
            refined_spectrum(pdf, runs, out)
        if "cad_pair" in runs:
            fig, axes = plt.subplots(3, 2, figsize=(10.2, 6.1), layout="constrained")
            for col, (left, right) in enumerate(((0.097, 0.110), (1.250, 1.256))):
                for j, (key, ch, label) in enumerate(
                    (
                        ("speed", 1, "Middle speed (rad/s)"),
                        ("twist", 1, "Stage 2 error (mrad)"),
                        ("torque", 2, "Output torque (N·m)"),
                    )
                ):
                    ax = axes[j, col]
                    for name, description, color, ls in (
                        ("native", "Native, default", "#343f52", ":"),
                        (
                            "native_dense" if "native_dense" in runs else "native_dense_short",
                            "Native, refined",
                            "#825d9c",
                            "-.",
                        ),
                        ("cad", "STEP, region stiffness", "#00817e", "-"),
                        ("cad_pair", "STEP, body-pair stiffness", "#b46636", "--"),
                    ):
                        y = runs[name][key][:, ch]
                        t = time_axis(key, len(y))
                        mask = (t >= left) & (t <= right)
                        if mask.any():
                            ax.plot(t[mask], y[mask], label=description, lw=1.1, color=color, ls=ls)
                    if col == 0:
                        ax.set_ylabel(label)
                axes[-1, col].set_xlabel("Time (s)")
            axes[0, 0].set_title("Brake engagement")
            axes[0, 1].set_title("Helical-stage switch")
            axes[0, 0].legend(ncol=2, fontsize=7, frameon=False)
            style(axes)
            pdf.savefig(fig)
            plt.close(fig)
        fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.5), layout="constrained")
        for ax, (key, j, label) in zip(
            axes, (("speed", 1, "Middle axle speed (rad/s)"), ("twist", 1, "Stage 2 error (mrad)"))
        ):
            for name, description, color, ls in (
                ("native", "RecurDyn • 125 µs", "#343f52", "--"),
                ("native31", "RecurDyn • 31.25 µs", "#343f52", "-"),
                ("cad", "STEP • 62.5 µs", "#00817e", "--"),
                ("fine32", "STEP • 31.25 µs, 32 local samples", "#00817e", "-"),
            ):
                t = time_axis(key, len(runs[name][key]))
                mask = (t >= 0.097) & (t <= 0.110)
                ax.plot(t[mask], runs[name][key][mask, j], ls=ls, color=color, lw=1, label=description)
            ax.set_ylabel(label)
        axes[0].legend(frameon=False, ncol=2, fontsize=8)
        axes[0].set_title("Time-step refinement during brake engagement")
        axes[-1].set_xlabel("Time (s)")
        style(axes)
        pdf.savefig(fig)
        plt.close(fig)
        poses = [("Default native facets", frozen)]
        if dense_frozen is not None:
            poses.append(("Refined native facets", dense_frozen))
        fig, axes = plt.subplots(
            3, len(poses), squeeze=False, figsize=(5.5 * len(poses), 6.5), layout="constrained"
        )
        for col, (title, at_pose) in enumerate(poses):
            for ax, shaft in zip(axes[:, col], ("input", "middle", "output")):
                for method, (label, color, ls) in STYLES.items():
                    if method == "native":
                        label = "Native reaction"
                    ax.plot(
                        at_pose.time_s,
                        at_pose[f"{method}_{shaft}_Nm"],
                        color=color,
                        ls=ls,
                        lw=1,
                        marker=".",
                        label=label,
                    )
                if "cad_pair_output_Nm" in at_pose:
                    ax.plot(
                        at_pose.time_s,
                        at_pose[f"cad_pair_{shaft}_Nm"],
                        ls=":",
                        color="#925335",
                        marker="x",
                        ms=3,
                        lw=1,
                        label="STEP, body-pair stiffness",
                    )
                ax.set_ylabel(f"{shaft.capitalize()} torque (N·m)")
            axes[0, col].set_title(title)
            axes[-1, col].set_xlabel("RecurDyn pose time (s)")
        axes[0, 0].legend(ncol=2, frameon=False, fontsize=8)
        fig.suptitle("Instantaneous contact torque at the native shaft poses in each run")
        style(axes)
        pdf.savefig(fig)
        plt.close(fig)
        with np.load(sphere_file or out / "sphere-exact.npz") as impact:
            fig, axes = plt.subplots(3, 1, figsize=(8.4, 6.7), layout="constrained")
            t = impact["time_s"]
            for name, label, color, ls in (
                ("exact", "Closed form", "#343f52", "-"),
                ("STEP2", "STEP • 2 µm deflection", "#00817e", "-"),
                ("RecurDyn2", "RecurDyn", "#c27534", "--"),
            ):
                y = impact[name]
                axes[0].plot(t, y[:, 2], color=color, ls=ls, lw=1.1, label=label)
                if name != "exact":
                    axes[1].plot(t, (y[:, 2] - impact["exact"][:, 2]) * 1000, color=color, ls=ls, lw=1)
                    axes[2].plot(t, (y[:, 0] - impact["exact"][:, 0]) * 1e6, color=color, ls=ls, lw=1)
            axes[0].set_title("Equal-mass sphere impact at 125 µs")
            axes[0].set_ylabel("Normal force (N)")
            axes[0].legend(frameon=False, ncol=3)
            axes[1].set_ylabel("Force error (mN)")
            axes[2].set_ylabel("Body 1 displacement error (µm)")
            axes[2].set_xlabel("Time (s)")
            style(axes)
            pdf.savefig(fig)
            plt.close(fig)
        coverage_file = (sphere_file or out / "sphere-exact.npz").with_name("sphere-coverage-traces.npz")
        if coverage_file.is_file():
            with np.load(coverage_file) as cv, np.load(sphere_file or out / "sphere-exact.npz") as sph:
                fig, axes = plt.subplots(2, 2, figsize=(10, 5.1), layout="constrained")
                for col, orientation in enumerate(("original", "rotated")):
                    for y, label, color, ls in (
                        (cv[f"{orientation}_v0"], "Surface samples", "#707d96", "--"),
                        (cv[f"{orientation}_v1"], "Surface + vertex samples", "#00817e", "-"),
                        (sph["RecurDyn2"], "RecurDyn sphere pair", "#c27534", ":"),
                    ):
                        for ax, j, scale in ((axes[0, col], 0, 1e6), (axes[1, col], 2, 1e3)):
                            ax.plot(
                                cv["t"],
                                (y[:, j] - cv["exact"][:, j]) * scale,
                                color=color,
                                ls=ls,
                                lw=1,
                                label=label,
                            )
                    axes[0, col].set_title(
                        "Original tessellation" if col == 0 else "Independently rotated tessellations"
                    )
                    axes[1, col].set_xlabel("Time (s)")
                axes[0, 0].set_ylabel("Body 1 displacement error (µm)")
                axes[1, 0].set_ylabel("Normal force error (mN)")
                axes[0, 1].legend(fontsize=7, frameon=False)
                fig.suptitle("Sphere impact • identical solids and loading • 2 µm tessellation • 125 µs step")
                style(axes)
                pdf.savefig(fig)
                plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 5.15), layout="constrained")
    for ax, (key, j, label) in zip(
        axes,
        (
            ("speed", 1, "Middle shaft (rad/s)"),
            ("twist", 1, "Stage 2 error (mrad)"),
            ("torque", 2, "Output torque (N·m)"),
        ),
    ):
        t = time_axis(key, len(runs["cad"][key]))
        mask = (t >= 0.097) & (t <= 0.110)
        for method, description, color, ls in (
            ("cad", "STEP, region stiffness", "#476b84", ":"),
            ("mesh_pair", "Mesh, pair stiffness", "#b46636", "--"),
            ("cad_pair", "STEP, pair stiffness (ours)", "#00817e", "-"),
        ):
            if method in runs:
                ax.plot(t[mask], runs[method][key][mask, j], color=color, ls=ls, lw=1.1, label=description)
        refined(ax, key, j, mask)
        ax.set_ylabel(label)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[-1].set_xlabel("Time (s)")
    style(axes)
    fig.savefig(out / "shaft-response.pdf")
    fig.savefig(out / "shaft-response.png", dpi=175)
    plt.close(fig)


def main(work, out):
    out.mkdir(parents=True, exist_ok=True)
    runs, checks = trajectories(work)
    exports = out / "native-exports"
    exports.mkdir(exist_ok=True)
    for name in ("rd-physics", "rd-refined-physics", "rd-switch-geometric-physics"):
        source = work / f"{name}.npz"
        if source.exists():
            shutil.copyfile(source, exports / source.name)
    timing = {}
    for method in ("cad", "mesh"):
        instrumented = np.load(work / f"full-{method}.npy", mmap_mode="r")
        for tag in ("production", "release"):
            name = f"{tag}-{method}"
            if not (work / f"{name}.json").exists():
                continue
            record = json.loads((work / f"{name}.json").read_text())
            if record["terminated"] != "complete" or record["setup_s"] + record["online_s"] > 1800:
                raise ValueError(f"Incomplete timed run {name}")
            np.testing.assert_array_equal(instrumented, np.load(work / f"{name}.npy", mmap_mode="r"))
            timing[name] = dict(**record, identical_to_instrumented=True)
    for name in (
        "rd-full",
        "rd-refined",
        "rd-geometric",
        "rd-switch-geometric",
        "vertices-cad",
        "vertices-mesh",
        "pair-cad",
        "pair-mesh",
    ):
        if (work / f"{name}.json").exists():
            timing[name] = json.loads((work / f"{name}.json").read_text())
    (out / "timings.json").write_text(json.dumps(timing, indent=2) + "\n")
    compare(runs).to_csv(out / "response-metrics.csv", index=False)
    (out / "channel-checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    np.savez_compressed(
        out / "physical-responses.npz", h=H, **{f"{m}_{k}": a for m, r in runs.items() for k, a in r.items()}
    )
    frozen = frozen_pose_torques(work, out, "switch-detail.csv", "rd-physics.npz", "fixed-native-torques.csv")
    dense_frozen = None
    if (work / "switch-dense-detail.csv").exists():
        dense_frozen = frozen_pose_torques(
            work,
            out,
            "switch-dense-detail.csv",
            "rd-switch-geometric-physics.npz",
            "fixed-dense-native-torques.csv",
        )
    for tag, native in (("default", "rd-physics.npz"), ("dense", "rd-geometric-physics.npz")):
        if (work / f"brake-{tag}.csv").exists():
            frozen_pose_torques(work, out, f"brake-{tag}.csv", native, f"brake-{tag}-torques.csv")
    figures(runs, out, frozen, dense_frozen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Regenerate metrics and plots from the published arrays on the CPU",
    )
    args = parser.parse_args()
    if args.replot:
        args.output.mkdir(exist_ok=True, parents=True)
        runs = {}
        with np.load(args.work / "physical-responses.npz") as data:
            np.testing.assert_allclose(float(data["h"]), H)
            for field in data.files:
                if field == "h":
                    continue
                method, key = field.rsplit("_", 1)
                runs.setdefault(method, {})[key] = data[field]
        compare(runs).to_csv(args.output / "response-metrics.csv", index=False)
        dense_path = args.work / "fixed-dense-native-torques.csv"
        figures(
            runs,
            args.output,
            pd.read_csv(args.work / "fixed-native-torques.csv"),
            pd.read_csv(dense_path) if dense_path.exists() else None,
            args.work / "sphere-exact.npz",
        )
    else:
        main(args.work, args.output)
