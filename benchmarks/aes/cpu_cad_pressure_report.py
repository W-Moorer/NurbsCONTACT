"""Compare paired pressure-wrench audits and CPU reducer response histories."""

import argparse
import csv
from itertools import islice
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _gpu_source import read_model
from cpu_backend_run import digest
from cpu_cad_pressure import checked_run, read_rows
from cpu_local_cad_replay import representatives


def wrench(regions, center, axis, stiffness, limit, distributed=False):
    if regions is None:
        return None
    if not regions:
        return np.zeros(8)
    r = np.asarray(regions, dtype=float)
    depth = r[:, -1]
    if not np.isfinite(r).all() or np.any((depth <= 0) | (depth > limit)):
        return None
    magnitudes = stiffness * depth / len(depth)
    force = magnitudes[:, None] * r[:, 3:6]
    moment = np.cross(r[:, :3] - center, force)
    if distributed:
        moment += magnitudes[:, None] * r[:, 6:9]
    return np.r_[magnitudes.sum(), force.sum(axis=0), moment.sum(axis=0), moment.sum(axis=0) @ axis]


def audit(args):
    model, pairs = read_model(args.scene)
    fields = ("sum_normal_N", "Fx_N", "Fy_N", "Fz_N", "Mx_Nm", "My_Nm", "Mz_Nm", "shaft_Nm")
    entries, errors = [], []
    with np.load(args.archive / "trajectory.npz") as trace:
        origin = trace["origins_m"]
    for stage, (_, driven, k, limit) in enumerate(pairs):
        refs = json.loads((args.archive / f"stage-{stage+1}-audit.json").read_text())["rows"]
        traces = {}
        for key in ("pressure", "area-0", "area-24", "area-48", "area-96"):
            prefix = args.work / f"{key}-{stage}"
            ids = json.loads(prefix.with_suffix(".ids").read_text())
            traces[key] = dict(zip(ids, read_rows(prefix.with_suffix(".jsonl")), strict=True))
        for step in sorted(map(int, refs)):
            r = refs[str(step)]
            values = {
                "mesh": r["mesh"],
                "deepest": representatives(r["native"], r["native_group_labels"]),
                "centroid": r["cad_regions"]["regions"] if r["cad_regions"]["resolved"] else None,
            }
            for key, log in traces.items():
                value = log[step]
                values[key] = value["patches"] if value["resolved"] else None
            for key, data in values.items():
                w = wrench(data, origin[step, driven], model[driven][2], k, limit, key in traces)
                entries.append(
                    dict(
                        stage=stage + 1,
                        step=step,
                        time_s=step * 0.0001,
                        method=key,
                        resolved=w is not None,
                        **dict(zip(fields, w if w is not None else [None] * 8, strict=True)),
                    )
                )
        rows = [r for r in entries if r["stage"] == stage + 1]
        full = {
            r["step"]: r
            for r in rows
            if r["method"] == "pressure" and r["resolved"] and r["sum_normal_N"] > 0
        }
        common = set(full)
        for key in ("mesh", "deepest", "area-0"):
            common.intersection_update(r["step"] for r in rows if r["method"] == key and r["resolved"])
        for key in values:
            eligible = [r for r in rows if r["method"] == key and r["step"] in full]
            matched = [r for r in eligible if r["resolved"]]

            def rms(field):
                return float(np.mean([(r[field] - full[r["step"]][field]) ** 2 for r in matched]) ** 0.5)

            vector_rmse = float(
                np.mean([sum((r[t] - full[r["step"]][t]) ** 2 for t in fields[1:4]) for r in matched]) ** 0.5
            )
            paired = [r for r in matched if r["step"] in common]
            paired_error = float(
                np.sqrt(np.mean([(r["shaft_Nm"] - full[r["step"]]["shaft_Nm"]) ** 2 for r in paired]))
            )
            errors.append(
                dict(
                    stage=stage + 1,
                    method=key,
                    reference_loaded=len(full),
                    resolved=len(matched),
                    paired_poses=len(paired),
                    paired_shaft_torque_rms_Nm=paired_error,
                    load_rms_N=rms(fields[0]),
                    force_vector_rms_N=vector_rmse,
                    shaft_torque_rms_Nm=rms(fields[-1]),
                )
            )
    with (args.out / "wrenches.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=entries[0])
        writer.writeheader()
        writer.writerows(entries)
    (args.out / "wrench-errors.json").write_text(json.dumps(errors, indent=2))
    return entries


def plot_audit(entries, out):
    styles = {
        "mesh": ("CPU mesh OBB", "#ae591e", "s"),
        "deepest": ("CAD deepest point", "#697781", "^"),
        "area-0": ("CAD patch (ours)", "#00786f", "o"),
        "pressure": ("Whole STEP pressure", "#282936", "x"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.7), layout="constrained")
    for name, (label, color, marker) in styles.items():
        rows = [r for r in entries if r["stage"] == 2 and r["method"] == name and r["resolved"]]
        for i, unit in enumerate(("Resultant force (kN)", "Output torque (N m)")):
            for j in range(2):
                selected = [r for r in rows if (j == 1 or r["step"] <= 10)]
                x = np.array([r["time_s"] for r in selected])
                if i == 0:
                    y = np.array([np.linalg.norm([r["Fx_N"], r["Fy_N"], r["Fz_N"]]) for r in selected]) / 1000
                else:
                    y = np.array([r["shaft_Nm"] for r in selected])
                axes[i, j].plot(
                    x * (1000 if j == 0 else 1),
                    y,
                    color=color,
                    marker=marker,
                    ms=3.8,
                    lw=0.95,
                    linestyle="-" if j == 0 else "none",
                    label=label,
                )
                axes[i, 0].set_ylabel(unit)
    axes[0, 0].set_title("Startup on the same poses")
    axes[0, 1].set_title("Audited poses over the duty cycle")
    axes[1, 0].set_xlabel("Time (ms)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[0, 1].legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(out / "same-pose-wrenches.png", dpi=180)
    plt.close(fig)


def timing(args):
    results = []
    for stage in range(2):
        control = None
        for key in ("old", "indexed", "mesh"):
            stem = args.work / f"paired-{key}-{stage}"
            meta = json.loads(stem.with_suffix(".json").read_text())
            raw = list(read_rows(stem.with_suffix(".jsonl")))
            if len(raw) != len(meta["starts"]) * meta["window_length"]:
                raise ValueError("incomplete timing windows")
            rows = [v for j, v in enumerate(raw) if j % meta["window_length"] >= meta["warmup"]]
            times = np.array([r["query_s"] for r in rows]) * 1000
            record = dict(
                stage=stage + 1,
                method=key,
                poses=len(rows),
                mean_ms=float(times.mean()),
                p95_ms=float(np.percentile(times, 95)),
                **meta,
            )
            if key != "mesh":
                broad = 1000 * np.mean([r["broad_s"] for r in rows])
                group = 1000 * np.mean([r["group_s"] for r in rows])
                record.update(broad_ms=broad, local_search_ms=times.mean() - broad - group, group_ms=group)
                if control is None:
                    control = rows
                else:
                    record["changed_face_pairs"] = sum(
                        [p[:2] for p in a["witnesses"]] != [p[:2] for p in b["witnesses"]]
                        for a, b in zip(rows, control, strict=True)
                    )
                    record["changed_group_labels"] = sum(
                        a["labels"] != b["labels"] for a, b in zip(rows, control, strict=True)
                    )
            results.append(record)
    (args.out / "paired-window-timing.json").write_text(json.dumps(results, indent=2))
    rows = [r for r in results if r["method"] != "mesh"]
    fig, ax = plt.subplots(figsize=(7.8, 3), layout="constrained")
    left = np.zeros(len(rows))
    y = np.array([0, 1, 2.5, 3.5])
    for key, label, color in (
        ("broad_ms", "Broad phase", "#87b3cd"),
        ("local_search_ms", "Local surface search", "#178278"),
        ("group_ms", "Region grouping", "#be7237"),
    ):
        values = np.array([r[key] for r in rows])
        ax.barh(y, values, left=left, height=0.63, label=label, color=color)
        left += values
    for pos, total in zip(y, left, strict=True):
        ax.text(total + 0.012 * max(left), pos, f"{total:.2f}", va="center", fontsize=9)
    ax.set_yticks(
        y, ["Stage 1 · previous", "Stage 1 · CAD (ours)", "Stage 2 · previous", "Stage 2 · CAD (ours)"]
    )
    ax.invert_yaxis()
    ax.set_xlim(0, max(left) * 1.12)
    ax.set_xlabel("Mean CAD search and grouping time per pose (ms)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=3, frameon=False, fontsize=9)
    ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(args.out / "search-and-grouping.png", dpi=180)
    plt.close(fig)


def full_replay(args):
    phases = []
    summaries = []
    for stage in range(2):
        old = args.old_replay / f"old-{stage}.jsonl"
        if not old.exists():
            old = args.old_replay / f"{stage}.integrated"
        new = args.work / f"indexed-{stage}.jsonl"
        mesh = args.work / f"mesh-{stage}.jsonl"
        profile = []
        different_faces = 0
        different_partition = 0
        different_count = 0
        missing_mesh = 0
        examples = []
        for j, (a, b, m) in enumerate(zip(read_rows(old), read_rows(new), read_rows(mesh), strict=True)):
            row = []
            for r in (a, b):
                row.extend([r["broad_s"], r["query_s"] - r["broad_s"] - r["group_s"], r["group_s"]])
            profile.append([*row, m["query_s"]])
            changes = [x[:2] for x in a["witnesses"]] != [x[:2] for x in b["witnesses"]]

            def partition(r):
                groups = {}
                for label, w in zip(r["labels"], r["witnesses"], strict=True):
                    groups.setdefault(label, []).append(tuple(w[:2]))
                return sorted(sorted(g) for g in groups.values())

            same = partition(a) == partition(b)
            different_faces += changes
            different_partition += not same
            different_count += len(set(a["labels"])) != len(set(b["labels"]))
            if not same and len(examples) < 32:
                examples.append(j)
            missing_mesh += m["regions"] is None
        if len(profile) != 50001:
            raise ValueError("incomplete geometry replay")
        phases.append(np.array(profile))
        total = phases[-1].sum(axis=0)
        summaries.append(
            dict(
                stage=stage + 1,
                poses=len(profile),
                old_broad_local_group_s=total[:3].tolist(),
                new_broad_local_group_s=total[3:6].tolist(),
                mesh_query_s=total[6],
                different_face_sets=different_faces,
                different_partitions=different_partition,
                different_region_counts=different_count,
                first_different_partition_steps=examples,
                mesh_unresolved=missing_mesh,
                indexed_process=json.loads(new.with_suffix(".json").read_text()),
                mesh_process=json.loads(mesh.with_suffix(".json").read_text()),
            )
        )
    (args.out / "full-replay.json").write_text(json.dumps(summaries, indent=2))
    data = np.stack(phases, axis=1)
    np.savez_compressed(
        args.out / "geometry-times.npz",
        time_s=np.arange(len(data)) * 0.0001,
        old_broad_search_group_s=data[:, :, :3].astype("f4"),
        new_broad_search_group_s=data[:, :, 3:6].astype("f4"),
        mesh_s=data[:, :, 6].astype("f4"),
    )


def smooth_loads(scene, t, q, v):
    lines = iter(scene.read_text().splitlines())
    coords = []
    external = np.zeros_like(q)
    for line in lines:
        if line.startswith("coordinate "):
            coords.append(list(map(float, line.split()[1:])))
        if line.startswith("load "):
            _, key, size = line.split()
            table = np.array([list(map(float, next(lines).split())) for _ in range(int(size))])
            external[:, int(key)] += np.interp(t, table[:, 0], table[:, 1])
    p = np.asarray(coords)
    smooth = p[:, 6] - p[:, 3] * (q - p[:, 5]) - p[:, 4] * v + external
    return p, smooth


def history(csv_file, scene):
    x = np.genfromtxt(csv_file, delimiter=",", names=True)
    q, v, f = (np.column_stack([x[f"{key}{i}"] for i in range(3)]) for key in ("q", "v", "generalized_force"))
    t = x["time_s"]
    p, smooth = smooth_loads(scene, t, q, v)
    change = q - q[0]
    bodies, pairs = read_model(scene)
    shaft_bodies = [pairs[0][0], pairs[0][1], pairs[1][1]]
    signs = np.array([bodies[i][2][2] for i in shaft_bodies])
    return dict(
        time_s=t,
        speed=v * signs,
        te_mrad=1000 * (change[:, 1:] - change[:, :-1] * p[1:, 2] / p[:-1, 2]),
        output_torque_Nm=(f - smooth)[:, 2] * signs[2],
        normal_N=np.column_stack([x[f"p{i}_normal_N"] for i in range(2)]),
    )


def dynamics(args):
    rd = np.load(args.archive.parent / "cpu-foundation/reducer-physics/closed-loop.npz")
    ref = {
        "speed": rd["rd_speed"],
        "te_mrad": rd["rd_te_mrad"],
        "output_torque_Nm": rd["rd_output_torque_Nm"],
    }
    fig, axes = plt.subplots(3, 2, figsize=(10, 7.3), sharex=True, layout="constrained")
    stats = {
        "RecurDyn": dict(
            mean_torque_Nm=float(ref["output_torque_Nm"].mean()),
            peak_abs_torque_Nm=float(abs(ref["output_torque_Nm"]).max()),
            torque_at_0p5ms_Nm=float(ref["output_torque_Nm"][5]),
        )
    }

    def draw(label, source, color):
        t = rd["time_s"]
        for ax, y, title in [
            (axes[0, 0], source["speed"][:, 1], "Intermediate speed (rad/s)"),
            (axes[0, 1], source["speed"][:, 2], "Output speed (rad/s)"),
            (axes[1, 0], source["te_mrad"][:, 0], "Stage 1 transmission error (mrad)"),
            (axes[1, 1], source["te_mrad"][:, 1], "Stage 2 transmission error (mrad)"),
            (axes[2, 0], source["output_torque_Nm"], "Output contact torque (N m)"),
        ]:
            ax.plot(t, y, color=color, alpha=0.72, lw=0.55, label=label)
            ax.set_title(title, loc="left", fontsize=10)
        if "normal_N" in source:
            axes[2, 1].plot(t, source["normal_N"][:, 1] / 1000, color=color, lw=0.65, alpha=0.75, label=label)

    draw("RecurDyn", ref, "#424653")
    histories = {}
    for key, label, color, path in [
        ("mesh", "CPU mesh OBB", "#c27434", args.mesh_csv),
        ("cad", "CAD patch (ours)", "#00786f", args.cad_csv),
    ]:
        result = history(path, args.scene)
        np.testing.assert_allclose(result["time_s"], rd["time_s"], rtol=0, atol=1e-10)
        if not all(np.isfinite(v).all() for v in result.values()):
            raise ValueError(f"Nonfinite {key} trace")
        stats[key] = {
            f"{field}_rms_vs_RecurDyn": np.sqrt(np.mean((result[field] - target) ** 2, axis=0)).tolist()
            for field, target in ref.items()
        }
        stats[key].update(
            peak_abs_torque_Nm=float(abs(result["output_torque_Nm"]).max()),
            mean_torque_Nm=float(np.mean(result["output_torque_Nm"])),
            source_sha256=digest(path),
            torque_at_0p5ms_Nm=float(result["output_torque_Nm"][5]),
            stage2_normal_at_0p5ms_N=float(result["normal_N"][5, 1]),
            torque_rms_vs_original_RecurDyn_Nm=float(
                np.mean((result["output_torque_Nm"] - rd["original_rd_torque_Nm"]) ** 2) ** 0.5
            ),
        )
        difference = result["output_torque_Nm"] - ref["output_torque_Nm"]
        stats[key]["output_torque_rms_by_second_Nm"] = [
            float(np.sqrt(np.mean(difference[10000 * j : 10000 * (j + 1) + (j == 4)] ** 2))) for j in range(5)
        ]
        histories[key] = result
        draw(label, result, color)
    axes[2, 1].set_title("Stage 2 normal load (kN)", loc="left", fontsize=10)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.17)
        ax.set_xlim(0, 5)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    axes[0, 1].legend(fontsize=8, frameon=False, ncol=1)
    fig.savefig(args.out / "closed-loop-5s.png", dpi=190)
    plt.close(fig)
    for name, series in histories.items():
        np.savez_compressed(args.out / f"closed-loop-{name}.npz", **series)
    (args.out / "closed-loop-errors.json").write_text(json.dumps(stats, indent=2))


def startup_check(args):
    """Re-query the accepted poses of each solver, including all startup warm-up poses."""
    probes, boolean, mesh = args.startup_binaries
    model, pairs = read_model(args.scene)
    a, b, stiffness, limit = pairs[1]
    solids = [model[i][0] for i in (a, b)]
    common = [probes, *solids, "6", str(2 * limit)]
    mesh_cmd = [
        mesh,
        "--method",
        "mesh-obb",
        "--regions",
        "contour-ray",
        "--axis-angle",
        "--keep-unresolved",
        "--max-depth-m",
        str(limit),
        *[p.with_suffix(".msh") for p in solids],
    ]
    report = {}
    for name, path in (("cad", args.cad_csv), ("mesh", args.mesh_csv)):
        with path.open(newline="") as stream:
            rows = list(islice(csv.DictReader(stream), 6))
        if len(rows) != 6 or abs(float(rows[-1]["time_s"]) - 0.0005) > 1e-12:
            raise ValueError("Missing accepted startup pose")
        poses = []
        for row in rows:
            poses.append(
                " ".join(
                    str(x)
                    for i in (a, b)
                    for x in [*[row[f"b{i}_{c}_m"] for c in "xyz"], *model[i][2], row[f"b{i}_angle_rad"]]
                )
            )

        def query(key, command, inputs):
            base = args.work / f"accepted-{name}-{key}"
            source, dest = base.with_suffix(".input"), base.with_suffix(".jsonl")
            source.write_text("\n".join(inputs) + "\n")
            checked_run(command, source, dest)
            result = list(read_rows(dest))
            if len(result) != len(inputs):
                raise ValueError(f"Incomplete startup {key}")
            return result[-1]

        native = query("search", [*common, "--indexed-groups"], poses)
        samples = [
            str(x)
            for label, hit in zip(native["labels"], native["witnesses"], strict=True)
            for x in (label, *hit[:8])
        ]
        local = query(
            "area",
            [*common, "--patch-wrench", "0"],
            [" ".join([poses[-1], str(len(native["labels"])), *samples])],
        )
        check = query("whole", [boolean, *solids, "--pressure-patches", str(2 * limit)], [poses[-1]])
        tri = query("mesh", mesh_cmd, poses)
        center = np.array([float(rows[-1][f"b{b}_{c}_m"]) for c in "xyz"])
        reference = {}
        for label, regions, distributed in (
            ("area", local["patches"] if local["resolved"] else None, True),
            ("whole_STEP", check["patches"] if check["resolved"] else None, True),
            ("mesh", tri["regions"], False),
        ):
            w = wrench(regions, center, model[b][2], stiffness, limit, distributed)
            reference[label] = (
                None
                if w is None
                else dict(
                    regions=len(regions),
                    normal_load_N=w[0],
                    resultant_N=float(np.linalg.norm(w[1:4])),
                    shaft_Nm=w[-1],
                )
            )
        trace = np.load(args.out / f"closed-loop-{name}.npz")
        report[name] = dict(
            response_sha256=digest(path),
            time_s=float(rows[-1]["time_s"]),
            accepted_pose=poses[-1],
            integrated_shaft_Nm=float(trace["output_torque_Nm"][5]),
            reevaluated=reference,
        )
        trace.close()
    (args.out / "accepted-0p5ms.json").write_text(json.dumps(report, indent=2))


def contact_profile(progress_path, out):
    """Cumulative per-pair work and real time at the same accepted output steps."""
    if not progress_path.exists():
        return
    milestones = {0, 5, 10, 20, 100, 1000, 5000, 10000, 25000, 50000}
    selected = []
    with progress_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            step = int(row["step"])
            if step in milestones:
                selected.append(
                    {
                        "step": step,
                        **{
                            key: float(value)
                            for key, value in row.items()
                            if key in ("time_s", "integration_s", "cad_booleans")
                            or key.startswith("cad_patch_")
                        },
                    }
                )
    if {row["step"] for row in selected} != milestones:
        raise ValueError("Incomplete CPU contact profile")
    (out / "closed-loop-profile.json").write_text(json.dumps(selected, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for n in ("scene", "work", "out"):
        parser.add_argument(n, type=Path)
    parser.add_argument("--archive", type=Path, default=Path("results/aes/cpu-local-cad"))
    parser.add_argument("--cad-csv", type=Path)
    parser.add_argument("--mesh-csv", type=Path)
    parser.add_argument(
        "--startup-binaries",
        type=Path,
        nargs=3,
        metavar=("PROBES", "BOOLEAN", "MESH"),
        help="audit each solver's own accepted 0.5 ms pose",
    )
    parser.add_argument(
        "--old-replay", type=Path, help="unmodified executable's complete geometry log directory"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plot_audit(audit(args), args.out)
    if (args.work / "paired-old-0.json").exists():
        timing(args)
    if args.old_replay:
        full_replay(args)
    if args.cad_csv and args.mesh_csv:
        dynamics(args)
        contact_profile(args.cad_csv.parent / "progress.csv", args.out)
        if args.startup_binaries:
            startup_check(args)
