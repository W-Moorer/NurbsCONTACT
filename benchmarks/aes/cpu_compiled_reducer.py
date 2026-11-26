"""Compile standalone reducer parts, run isolated CPU backends and audit the full trajectory."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _gpu_source import read_model
from cpu_backend_run import digest, run
from cpu_cad_pressure_report import history, wrench
from cpu_pressure_columns import differences, fixed_geometry_inputs, query, stage_poses


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=False)
    bodies, _ = read_model(args.scene)
    inputs = sorted({b[0].resolve() for b in bodies})
    if len({p.stem for p in inputs}) != len(inputs):
        raise ValueError("Ambiguous flat part names. Keep the scene's relative directory structure.")
    manifest = []
    for path in inputs:
        out = args.out / (path.stem + ".ncpart")
        start = perf_counter()
        result = subprocess.run([args.compiler, path, out], check=True, capture_output=True, text=True)
        manifest.append(
            dict(
                source=path.name,
                source_sha256=digest(path),
                part=out.name,
                sha256=digest(out),
                bytes=out.stat().st_size,
                compile_s=perf_counter() - start,
                geometry=result.stdout.strip(),
            )
        )
    write(args.out / "parts.json", manifest)


def module_audit(args):
    """Record the module list of the actual Windows process and verify its isolated inputs."""
    import pefile
    import psutil

    exe = args.exe.resolve()
    process = psutil.Process(args.pid)
    if not Path(process.exe()).samefile(exe):
        raise ValueError("PID does not run the specified executable")
    pe = pefile.PE(str(exe))
    imports = [
        d.dll.decode()
        for key in ("DIRECTORY_ENTRY_IMPORT", "DIRECTORY_ENTRY_DELAY_IMPORT")
        for d in getattr(pe, key, [])
    ]
    images = sorted(
        {Path(r.path).name for r in process.memory_maps() if r.path.lower().endswith((".dll", ".exe"))}
    )
    assets = [p for p in args.root.rglob("*") if p.suffix.lower() in (".scene", ".ncpart")]
    sources = [
        str(p) for p in args.root.rglob("*") if p.suffix.lower() in (".step", ".stp", ".brep", ".msh")
    ]
    if sources or any(
        name.lower().startswith("tk") and name.lower().endswith(".dll") for name in imports + images
    ):
        raise ValueError("OCCT dependency or source geometry in standalone run")
    if len([p for p in assets if p.suffix.lower() == ".scene"]) != 1 or not any(
        p.suffix.lower() == ".ncpart" for p in assets
    ):
        raise ValueError("Missing isolated scene or parts")
    write(
        args.out,
        dict(
            executable_sha256=digest(exe),
            imports=imports,
            loaded_images=images,
            source_geometry=sources,
            working_set_sample_bytes=process.memory_info().wset,
            assets={p.relative_to(args.root).as_posix(): digest(p) for p in sorted(assets)},
        ),
    )


def cycle(args):
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    args.out.mkdir(parents=True, exist_ok=False)
    scene = args.scene.resolve()
    if args.method == "cad-compiled":
        # A previous scene can retain its STEP body names. Only compiled parts
        # are delivered to the isolated directory; never resolve a STEP asset.
        from shlex import split
        from os.path import commonpath

        sources = [
            Path(split(line)[1]) for line in scene.read_text().splitlines() if line.startswith("body ")
        ]
        if any(p.is_absolute() for p in sources):
            raise ValueError("Use a scene with relative body filenames for the isolation check")
        originals = [(scene.parent / p).resolve() for p in sources]
        base = Path(commonpath([scene.parent, *[p.parent for p in originals]]))
        root = args.out / "assets"
        isolated = root / scene.relative_to(base)
        isolated.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(scene, isolated)
        for path in set(originals):
            dst = (root / path.relative_to(base)).with_suffix(".ncpart")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.parts / dst.name, dst)
        scene = isolated
    run(
        args.exe.resolve(),
        scene.resolve(),
        args.out / "run",
        args.method,
        "contour-ray",
        "penalty",
        "shared",
        "predictor",
        0,
    )


def fixed(args):
    """Evaluate identical accepted shaft poses, independently of the integrator."""
    args.out.mkdir(parents=True, exist_ok=True)
    model, pairs = read_model(args.scene)
    data = np.genfromtxt(args.reference / "response.csv", names=True, delimiter=",")
    ref = json.loads((args.reference / "run.json").read_text())
    if ref["scene_sha256"] != digest(args.scene) or ref["response_sha256"] != digest(
        args.reference / "response.csv"
    ):
        raise ValueError("CAD reference trajectory or scene has changed")
    ids = np.array([*range(31), 1859, 1860, 2741, 9317, 25377, 36010, 36021, 45043, 49781])
    origins = np.stack(
        [np.column_stack([data[f"b{i}_{k}_m"] for k in "xyz"]) for i in range(len(model))], axis=1
    )
    angles = np.column_stack([data[f"b{i}_angle_rad"] for i in range(len(model))])

    def evaluate(command, poses):
        payload = "".join(" ".join(format(x, ".17g") for x in row) + "\n" for row in poses)
        process = subprocess.run(
            command, input=payload, check=True, text=True, capture_output=True, timeout=1800
        )
        rows = [json.loads(line) for line in process.stdout.splitlines()]
        if len(rows) != len(poses) or any(not row["resolved"] for row in rows):
            raise ValueError("Incomplete fixed-pose result")
        return rows

    checks = []
    for stage, (a, b, stiffness, limit) in enumerate(pairs, 1):
        poses = stage_poses(model, angles, origins, ids, a, b)
        body_files = [model[i][0] for i in (a, b)]
        old = evaluate(
            [
                args.native,
                *body_files,
                "6",
                str(2 * limit),
                "--loaded",
                "--bezier",
                "--depth-bounds",
                "--relocate",
            ],
            poses,
        )
        new = evaluate(
            [args.probe, *[args.parts / (f.stem + ".ncpart") for f in body_files], str(limit)], poses
        )
        whole_steps = (13, 16, 2741, 9317) if stage == 2 and args.boolean else ()
        whole = (
            evaluate(
                [args.boolean, *body_files, "--pressure-patches", str(2 * limit)],
                poses[np.isin(ids, whole_steps)],
            )
            if whole_steps
            else []
        )
        edges = (
            evaluate(
                [
                    args.probe,
                    *[args.parts / (f.stem + ".ncpart") for f in body_files],
                    str(limit),
                    "--audit-chart-edges",
                ],
                poses[np.isin(ids, whole_steps)],
            )
            if whole_steps
            else []
        )
        for step, x, y in zip(ids, old, new, strict=True):
            row = dict(stage=stage, step=int(step))
            measures = [("original", x["regions"]), ("compiled", y["regions"])]
            if step in whole_steps:
                measures.append(("whole_solid", whole[whole_steps.index(step)]["patches"]))
                measures.append(("edge_audit", edges[whole_steps.index(step)]["regions"]))
            for key, regions in measures:
                w = wrench(regions, origins[step, b], model[b][2], stiffness, limit, True)
                if w is None:
                    raise ValueError("Region outside material-law depth range")
                row[key] = dict(count=len(regions), depth_m=[r[-1] for r in regions], wrench=w.tolist())
            checks.append(row)
    write(
        args.out / "same-poses.json",
        dict(
            scene_sha256=digest(args.scene),
            reference_sha256=ref["response_sha256"],
            probes={
                k: digest(p)
                for k, p in [("compiled", args.probe), ("original", args.native), ("whole", args.boolean)]
                if p
            },
            rows=checks,
        ),
    )


def switches(args):
    """Replay the previously published eight 100-pose windows with material face IDs."""
    trajectory, ids, warm, bodies, pairs, angles, origins = fixed_geometry_inputs(args.scene, args.archive)
    result = dict(scene_sha256=digest(args.scene), trajectory_sha256=digest(trajectory), stages=[])
    for stage, (a, b, k, limit) in enumerate(pairs, 1):
        poses = stage_poses(bodies, angles, origins, ids, a, b)
        files = [bodies[i][0] for i in (a, b)]
        native = query(
            [
                args.native,
                *files,
                "6",
                str(2 * limit),
                "--loaded",
                "--bezier",
                "--depth-bounds",
                "--relocate",
            ],
            poses,
        )
        compiled = query(
            [args.probe, *[args.parts / (p.stem + ".ncpart") for p in files], str(limit)], poses
        )
        register = dict(
            stage=stage,
            frames=len(ids),
            face_group_mismatches=[],
            unresolved=[],
            original_switches=[],
            compiled_switches=[],
        )
        last = [None, None]
        torque = []
        for i, (x, y) in enumerate(zip(native, compiled, strict=True)):
            step = int(ids[i])
            if not x["resolved"] or not y["resolved"]:
                register["unresolved"].append(step)
                last = [None, None]
                continue
            groups = {}
            for label, hit in zip(x["labels"], x["witnesses"], strict=True):
                groups.setdefault(label, []).append(hit[:2])
            normalize = lambda rows: sorted(tuple(sorted(map(tuple, g))) for g in rows)
            current = [normalize(groups.values()), normalize(y["group_faces"])]
            if current[0] != current[1] or len(x["regions"]) != len(y["regions"]):
                register["face_group_mismatches"].append(
                    dict(step=step, original=current[0], compiled=current[1])
                )
            for j, name in enumerate(("original_switches", "compiled_switches")):
                if i % 100 and last[j] is not None and current[j] != last[j]:
                    register[name].append(step)
            last = current
            wx = wrench(x["regions"], origins[step, b], bodies[b][2], k, limit, True)
            wy = wrench(y["regions"], origins[step, b], bodies[b][2], k, limit, True)
            if wx is not None and wy is not None:
                torque.append([step, wy[-1] - wx[-1]])
        register["comparable_torques"] = len(torque)
        if torque:
            register["max_torque_difference_Nm"] = max(abs(row[1]) for row in torque)
            register["rms_torque_difference_Nm"] = float(
                np.sqrt(np.mean(np.square(np.array(torque)[:, 1])))
            )
        result["stages"].append(register)
    write(args.out, result)


def compare(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    labels = ["Compiled CAD (ours)", "Original CAD", "Mesh OBB"]
    paths = [args.compiled, args.native, args.mesh]
    traces, records = [], []
    for path in paths:
        record = json.loads((path / "run.json").read_text())
        if (
            record["scene_sha256"] != digest(args.scene)
            or record["status"] != "complete"
            or record["dt_s"] != 0.0001
            or record["requested_s"] != 5
            or record["geometry_update"] != "predictor"
            or record["penalty_stiffness"] != "shared"
            or record["spectral_radius"] != 0
            or record["solver"] != "penalty"
            or record["response_sha256"] != digest(path / "response.csv")
        ):
            raise ValueError(f"Run settings or output mismatch: {path}")
        report = history(path / "response.csv", args.scene)
        np.testing.assert_allclose(report["time_s"], np.arange(50001) * 0.0001, rtol=0, atol=1e-10)
        records.append(record)
        traces.append(report)
    compiled, native, mesh = traces
    t = compiled["time_s"]
    torque = "output_torque_Nm"
    delta = compiled[torque] - native[torque]
    metrics = {
        "scene_sha256": records[0]["scene_sha256"],
        "steps": len(t) - 1,
        "process_s": {k: r["process_s"] for k, r in zip(labels, records)},
        "compiled_to_mesh_time_ratio": records[0]["process_s"] / records[2]["process_s"],
        "cad_torque_difference_Nm": differences(delta),
        "stage_loaded_mismatches": np.sum(
            (compiled["normal_N"] > 0) != (native["normal_N"] > 0), axis=0
        ).tolist(),
        "stage_normal_difference_N": [
            differences(a) for a in (compiled["normal_N"] - native["normal_N"]).T
        ],
        "shaft_speed_difference_rad_s": [differences(a) for a in (compiled["speed"] - native["speed"]).T],
        "switch_steps": {
            k: [(1 + np.flatnonzero(np.diff(v > 0))).tolist() for v in tr["normal_N"].T]
            for k, tr in zip(labels, traces)
        },
        "torque_at_0p5_ms_Nm": {k: tr[torque][5] for k, tr in zip(labels, traces)},
        "torque_at_2p5377_s_Nm": {k: tr[torque][25377] for k, tr in zip(labels, traces)},
        "historical_spike_window_max_difference_Nm": float(max(abs(delta[25357:25398]))),
        "largest_sample_to_sample_torque_change_Nm": {
            k: float(max(abs(np.diff(tr[torque])))) for k, tr in zip(labels, traces)
        },
    }
    rd = None
    if args.recurdyn:
        with np.load(args.recurdyn) as archive:
            np.testing.assert_allclose(t, archive["time_s"], atol=1e-10, rtol=0)
            rd = {key: archive["RecurDyn_" + key] for key in (torque, "speed", "te_mrad")}
        metrics["recurdyn_sha256"] = digest(args.recurdyn)
        metrics["output_torque_difference_vs_recurdyn_Nm"] = {
            k: differences(tr[torque] - rd[torque]) for k, tr in zip(labels, traces)
        }
    write(out / "comparison.json", metrics)
    for key, r in zip(("compiled", "original", "mesh"), records):
        write(out / (key + "-run.json"), r)
    series = {
        k + "_" + field: tr[field]
        for k, tr in zip(("compiled", "original", "mesh"), traces)
        for field in (torque, "speed", "normal_N", "te_mrad")
    }
    if rd:
        series.update({"recurdyn_" + field: data for field, data in rd.items()})
    np.savez_compressed(out / "responses.npz", time_s=t, **series)
    fig, axes = plt.subplots(3, 2, figsize=(12, 8.2), layout="constrained")
    for key, tr, color, style in zip(labels, traces, ("#075985", "#c2410c", "#636363"), ("-", "--", ":")):
        for ax, field, col in [
            (axes[0, 0], torque, None),
            (axes[0, 1], "speed", 1),
            (axes[1, 0], "normal_N", 0),
            (axes[1, 1], "normal_N", 1),
        ]:
            y = tr[field] if col is None else tr[field][:, col]
            ax.plot(t, y, style, color=color, label=key, lw=1)
        idx = slice(25357, 25398)
        axes[2, 1].plot(1000 * (t[idx] - 2.5377), tr[torque][idx], style, color=color, lw=1)
    axes[2, 0].plot(t, delta, color="#075985", lw=0.8)
    if rd:
        axes[0, 0].plot(t, rd[torque], color="#64748b", lw=0.65, alpha=0.85, label="RecurDyn", zorder=0)
        axes[0, 1].plot(t, rd["speed"][:, 1], color="#64748b", lw=0.65, alpha=0.85, zorder=0)
    for i, (ax, label) in enumerate(
        zip(
            axes.flat,
            (
                "Output torque (N m)",
                "Intermediate speed (rad/s)",
                "Stage 1 normal load (N)",
                "Stage 2 normal load (N)",
                "CAD torque difference (N m)",
                "Output torque at switch (N m)",
            ),
        )
    ):
        ax.set(xlabel="Time (s)" if i < 5 else "Time from 2.5377 s (ms)", ylabel=label)
        ax.grid(alpha=0.18)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(out / "responses.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for name in ("scene", "compiler", "out"):
        prep.add_argument(name, type=Path)
    runner = commands.add_parser("run")
    for name in ("scene", "exe", "out"):
        runner.add_argument(name, type=Path)
    runner.add_argument("--method", choices=("cad-compiled", "mesh-obb"), required=True)
    runner.add_argument("--parts", type=Path)
    runner.add_argument("--threads", type=int, default=2)
    comp = commands.add_parser("compare")
    for name in ("scene", "compiled", "native", "mesh", "out"):
        comp.add_argument(name, type=Path)
    comp.add_argument("--recurdyn", type=Path)
    audit = commands.add_parser("fixed")
    for name in ("scene", "reference", "probe", "native", "parts", "out"):
        audit.add_argument(name, type=Path)
    audit.add_argument("--boolean", type=Path)
    deps = commands.add_parser("modules")
    for name in ("exe", "root", "out"):
        deps.add_argument(name, type=Path)
    deps.add_argument("pid", type=int)
    face = commands.add_parser("switches")
    for name in ("scene", "archive", "probe", "native", "parts", "out"):
        face.add_argument(name, type=Path)
    args = p.parse_args()
    if args.command == "run" and args.method == "cad-compiled" and args.parts is None:
        p.error("cad-compiled requires --parts")
    {
        "prepare": prepare,
        "run": cycle,
        "compare": compare,
        "fixed": fixed,
        "modules": module_audit,
        "switches": switches,
    }[args.command](args)
