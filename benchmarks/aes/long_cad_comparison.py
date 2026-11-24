"""Compare completed reducer runs on their common native output grid."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def rms(values):
    return np.sqrt(np.mean(values**2, axis=0)).tolist()


def switches(faces):
    indices = np.flatnonzero(np.any(faces[1:] != faces[:-1], axis=1)) + 1
    result = defaultdict(list)
    for i in indices:
        result[(*faces[i - 1], *faces[i])].append(i)
    return result


def paired_switches(a, b, window):
    events_a, events_b = switches(a), switches(b)
    delta = []
    for signature, left in events_a.items():
        right = events_b.get(signature, [])
        i = j = 0
        while i < len(left) and j < len(right):
            diff = left[i] - right[j]
            if diff < -window:
                i += 1
            elif diff > window:
                j += 1
            else:
                delta.append(abs(diff))
                i += 1
                j += 1
    return dict(
        first_events=sum(map(len, events_a.values())),
        second_events=sum(map(len, events_b.values())),
        matched=len(delta),
        max_allowed_steps=window,
    )


def physical(raw, nominal):
    delta = raw[:, :3] - raw[0, :3]
    return dict(
        speed=raw[:, 3:6] + nominal,
        twist=(delta[:, :2] - 2 * delta[:, 1:3]) * 1000,
        load=raw[:, [6, 10]],
        torque=raw[:, 12],
    )


def timing_figure(out, report, cad_name, mesh_name):
    keys = [
        (mesh_name, "Mesh"),
        ("source-floor16", "CAD probes, flat target"),
        (
            ("width16", "CAD bounded candidate search")
            if cad_name == "direct16"
            else ("direct16", "CAD direct candidate search")
        ),
        (cad_name, "CAD patches (ours)"),
    ]
    keys = [(k, title) for k, title in keys if k in report]
    labels = [name for _, name in keys]
    fig, ax = plt.subplots(figsize=(8.4, 2.7), layout="constrained")
    totals = np.zeros(len(keys))
    y = np.arange(len(keys))
    for field, label, color in (
        ("prepare_s", "Prepare trial", "#99b9c4"),
        ("search_s", "Search for contact", "#d77b51"),
        ("merge_and_hht_s", "Assemble and integrate", "#64788f"),
    ):
        values = np.array([report[k]["timing"]["phase_profile"][field] for k, _ in keys])
        ax.barh(y, values, left=totals, color=color, label=label, height=0.6)
        totals += values
    for i, (k, _) in enumerate(keys):
        ax.text(
            totals[i] + max(totals) * 0.012,
            i,
            f'{report[k]["timing"]["online_s"]:.1f} s',
            va="center",
            fontsize=8,
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(totals) * 1.16)
    ax.set_xlabel("Wall time for 5 s of dynamics (s)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, fontsize=8)
    fig.savefig(out / "timings.png", dpi=200)
    plt.close(fig)


def event_figure(work, out, t, mesh, cad_name):
    if not (work / "obb.json").is_file():
        return
    reference = json.loads((work / "obb.json").read_text())
    anchor = work / "fast16.npy"
    if hashlib.sha256(anchor.read_bytes()).hexdigest() != reference["trace_sha256"] or not np.array_equal(
        np.load(anchor, mmap_mode="r"), np.load(work / (cad_name + ".npy"), mmap_mode="r")
    ):
        raise ValueError("OBB replay and plotted CAD poses differ")
    force = np.load(anchor, mmap_mode="r")[:, 10]
    with np.load(work / "obb.npz") as f:
        overlap = f["mesh-obb_1"]
    disagree = np.flatnonzero(overlap != (force > 0))
    if not len(disagree):
        return
    k = disagree[0]
    keep = np.abs(t - t[k]) <= 0.0009
    tm = t[keep] * 1000
    fig, (upper, lower) = plt.subplots(
        2, 1, figsize=(7, 4.3), sharex=True, layout="constrained", gridspec_kw={"height_ratios": [2, 1]}
    )
    upper.plot(tm, mesh["load"][keep, 1], "o--", ms=2.8, lw=0.9, color="#33779a", label="Mesh dynamics")
    upper.plot(tm, force[keep], "o-", ms=2.8, lw=1, color="#bf5a2a", label="CAD dynamics (ours)")
    upper.set_ylabel("Stage 2 normal load (N)")
    upper.set_title("Stage 2 engagement · one marker per 0.1 ms output", loc="left")
    lower.step(
        tm, (force[keep] > 0).astype(int), where="mid", color="#bf5a2a", lw=1.6, label="CAD load present"
    )
    lower.step(
        tm,
        overlap[keep].astype(int),
        where="mid",
        color="#43444b",
        lw=1.2,
        ls="--",
        label="Triangle overlap at CAD poses",
    )
    lower.scatter(
        [t[k] * 1000] * 2,
        [int(force[k] > 0), int(overlap[k])],
        s=17,
        facecolor="white",
        edgecolor="#43444b",
        zorder=5,
    )
    lower.set(yticks=[0, 1], yticklabels=["Off", "On"], ylim=(-0.18, 1.22), xlabel="Time (ms)")
    fig.legend(ncols=2, fontsize=8, loc="outside lower center", frameon=False)
    for ax in (upper, lower):
        ax.axvline(t[k] * 1000, color="#7b8085", lw=0.7, ls=":")
        ax.margins(x=0)
    fig.savefig(out / "engagement.png", dpi=200)
    plt.close(fig)


def figure(out, t, mesh, patch, reference):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(10, 7), layout="constrained")
    colors = {"Mesh": "#33779a", "CAD patches (ours)": "#bf5a2a", "RecurDyn": "#43444b"}
    lines = (("RecurDyn", reference), ("Mesh", mesh), ("CAD patches (ours)", patch))
    zoom = (t >= 1.245) & (t <= 1.265)
    for row, (field, idx, ylabel) in enumerate(
        (
            ("speed", 1, "Intermediate speed (rad/s)"),
            ("twist", 1, "Stage 2 transmission error (mrad)"),
            ("torque", None, "Output contact torque (N·m)"),
        )
    ):
        for name, data in lines:
            y = data[field] if idx is None else data[field][:, idx]
            # Show each time bin's mean and full range on the long record.
            b = y[:-1].reshape(250, -1)
            centers = t[:-1].reshape(250, -1).mean(axis=1)
            ax = axes[row, 0]
            ax.fill_between(
                centers, b.min(axis=1), b.max(axis=1), color=colors[name], alpha=0.12, linewidth=0
            )
            ax.plot(centers, b.mean(axis=1), color=colors[name], lw=1, label=name)
            axes[row, 1].plot(
                t[zoom], y[zoom], color=colors[name], lw=1, ls="--" if name == "Mesh" else "-", label=name
            )
        axes[row, 0].set_ylabel(ylabel)
    axes[0, 0].set_title("Full 5 s · 20 ms mean and range", loc="left")
    axes[0, 1].set_title("Contact-switch window · every output", loc="left")
    axes[0, 0].legend(ncol=3, fontsize=7, loc="lower left")
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    for ax in axes.ravel():
        ax.margins(x=0)
    fig.savefig(out / "responses.png", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 4.6), layout="constrained")
    for stage in range(2):
        for name, data in lines:
            y = data["load"][:, stage]
            b = y[:-1].reshape(250, -1)
            axes[stage, 0].fill_between(
                centers, b.min(axis=1), b.max(axis=1), color=colors[name], alpha=0.16, linewidth=0
            )
            axes[stage, 0].plot(
                centers,
                b.mean(axis=1),
                color=colors[name],
                lw=1,
                label="RecurDyn resultant" if name == "RecurDyn" else name,
            )
            if name != "RecurDyn":
                axes[stage, 1].plot(
                    t[zoom], y[zoom], color=colors[name], lw=1, ls="--" if name == "Mesh" else "-"
                )
        axes[stage, 0].set_ylabel(f"Stage {stage+1} contact load (N)")
    axes[0, 0].set_title("5 s · 20 ms mean and range", loc="left")
    axes[0, 1].set_title("Mesh and CAD patches · every output", loc="left")
    axes[0, 0].legend(ncol=3, fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    for ax in axes.ravel():
        ax.margins(x=0)
    fig.savefig(out / "loads.png", dpi=200)
    plt.close(fig)


def compare(work, out, cad_name, mesh_name, controls, animate=False):
    out.mkdir(parents=True, exist_ok=True)
    with np.load(work / "deck-v2" / "assembly.npz") as deck:
        nominal = deck["bodies"][[0, 1, 3], 8]
        axes = deck["bodies"][[0, 1, 3], 3]
    ref_path = Path("results/aes/long-mesh/recurdyn-5s.npz")
    with np.load(ref_path) as rd:
        t = rd["reference_rad"][:, 0] / nominal[0]
        reference = dict(
            load=rd["resultant_N"],
            speed=rd["global_speed_rad_s"] * axes,
            torque=rd["contact_raw_TZ_Nm"][:, 1],
            twist=1000 * (rd["angle_rad"][:, :2] - 2 * rd["angle_rad"][:, 1:3]),
        )
    np.testing.assert_allclose(t, np.arange(50001) * 1e-4, rtol=0, atol=1e-9)
    data = {}
    faces = {}
    report = {}
    for name in dict.fromkeys([cad_name, mesh_name, *controls]):
        info = json.loads((work / (name + ".json")).read_text())
        raw = np.load(work / (name + ".npy"))
        if (
            info["terminated"] != "complete"
            or info["steps"] != 50000
            or abs(info["physical_s"] - 5) > 1.0e-12
            or abs(info["step_s"] - 1.0e-4) > 1.0e-15
            or info["stiffness_scope"] != "pair"
            or raw.shape != (50001, 14)
            or not np.isfinite(raw).all()
        ):
            raise ValueError(f"Incomplete run {name}")
        data[name] = physical(raw, nominal)
        history = work / (name + "-faces.npz")
        if history.exists():
            with np.load(history) as f:
                faces[name] = f["faces"]
            np.testing.assert_array_equal(faces[name][:, :, 0] >= 0, raw[:, [6, 10]] > 0)
        fields = data[name]
        against = {key: rms(fields[key] - reference[key]) for key in ("speed", "twist", "torque")}
        report[name] = dict(
            raw_trace_sha256=hashlib.sha256(np.ascontiguousarray(raw).view(np.uint8)).hexdigest(),
            unloaded_output_frames=(fields["load"] == 0).sum(axis=0).tolist(),
            on_off_switches=np.count_nonzero(np.diff(fields["load"] > 0, axis=0), axis=0).tolist(),
            timing=info,
            rms_vs_recurdyn=against,
            mean_output_torque_Nm=float(fields["torque"].mean()),
            stage_peak_N=fields["load"].max(axis=0).tolist(),
            rms_stage_load_difference_vs_cad_N=None,
        )
    cad = data[cad_name]
    cad_run, mesh_run = report[cad_name]["timing"], report[mesh_name]["timing"]
    for key in (
        "gpu",
        "contact_samples",
        "contact_floor_nm",
        "coordinates",
        "pairs",
        "stiffness_scope",
        "broad_phase",
        "explicit_crossing_enabled",
    ):
        if cad_run[key] != mesh_run[key]:
            raise ValueError(f"Different primary comparison setting: {key}")
    for name, fields in data.items():
        report[name]["rms_stage_load_difference_vs_cad_N"] = rms(fields["load"] - cad["load"])
        if name in faces and cad_name in faces:
            report[name]["face_agreement_vs_cad"] = (
                np.all(faces[name] == faces[cad_name], axis=2).mean(axis=0).tolist()
            )
            report[name]["switches_vs_cad"] = [
                [paired_switches(faces[name][:, s], faces[cad_name][:, s], w) for w in (1, 5)]
                for s in range(2)
            ]
    (out / "comparison.json").write_text(
        json.dumps(
            dict(
                cad=cad_name,
                mesh=mesh_name,
                reference_mean_output_torque_Nm=float(reference["torque"].mean()),
                reference_unloaded_output_frames=(reference["load"] == 0).sum(axis=0).tolist(),
                window_s=[1.245, 1.265],
                window_output_torque_range_Nm={
                    key: [
                        float(x[(t >= 1.245) & (t <= 1.265)].min()),
                        float(x[(t >= 1.245) & (t <= 1.265)].max()),
                    ]
                    for key, x in [
                        ("RecurDyn", reference["torque"]),
                        *((k, v["torque"]) for k, v in data.items()),
                    ]
                },
                runs=report,
            ),
            indent=2,
        )
        + "\n"
    )
    # Keep both matched temporal resolutions alongside the independently measured timings.
    archive_names = [cad_name, mesh_name]
    if "mesh8" in data:
        for name in ("winner8", "patch8"):
            if name in data:
                for key in ("contact_samples", "contact_floor_nm", "broad_phase", "stiffness_scope"):
                    if report["mesh8"]["timing"][key] != report[name]["timing"][key]:
                        raise ValueError(f"Different 8-sample comparison setting: {key}")
                archive_names.extend(("mesh8", name))
                break
    np.savez_compressed(
        out / "responses.npz",
        time_s=t,
        **{f"{name}_{k}": v for name in archive_names for k, v in data[name].items()},
        **{f"{name}_faces": faces[name] for name in archive_names if name in faces},
    )
    figure(out, t, data[mesh_name], cad, reference)
    timing_figure(out, report, cad_name, mesh_name)
    event_figure(work, out, t, data[mesh_name], cad_name)
    if animate:
        from long_mesh_figures import render_reducer_animation

        angle = np.load(work / (cad_name + ".npy"), mmap_mode="r")[:, :3] + t[:, None] * nominal
        render_reducer_animation(
            work / "deck-v2",
            out / "reducer.mp4",
            angle,
            cad["load"],
            t,
            "CAD patch dynamics  |  Two-stage reducer",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("work", "out"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--cad", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--controls", nargs="*", default=[])
    parser.add_argument("--animate", action="store_true")
    args = parser.parse_args()
    compare(args.work, args.out, args.cad, args.mesh, args.controls, args.animate)
