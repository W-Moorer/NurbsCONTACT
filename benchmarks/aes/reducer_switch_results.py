"""Score an instrumented contact switch at common poses and the corrected closed-loop runs."""

import argparse
import gzip
import json
from pathlib import Path
import shutil

import igl
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def stats(d, step, trace):
    tr = pd.DataFrame(d["trials"], columns=d["columns"])
    selected = tr[tr.step == step]
    accepted = selected[selected.decision == 0]
    dt = float(d["h"]) * accepted.span / float(d["ticks"])
    assert accepted.span.sum() == int(d["ticks"])
    dv = float(np.sum(dt * (0.4 * accepted.previous_a1 + 0.6 * accepted.trial_a1)))
    np.testing.assert_allclose(dv, trace[step, 4] - trace[step - 1, 4], atol=2e-12, rtol=0)
    return dict(
        evaluations=len(selected),
        accepted=len(accepted),
        newton_updates=int((selected.decision == 4).sum()),
        stiffness_subdivisions=int((selected.decision == 1).sum()),
        convergence_subdivisions=int((selected.decision == 2).sum()),
        line_searches=int((selected.decision == 3).sum()),
        explicit_retries=int((selected.decision == 6).sum()),
        middle_delta_rad_s=dv,
        minimum_accepted_stage2_middle_torque_Nm=float(accepted.torque_a1.min()),
        window_evaluations=len(tr),
        window_accepted=int((tr.decision == 0).sum()),
    )


def cpu_witness(work, d, row, hit):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.TopAbs import TopAbs_OUT
    from OCP.gp import gp_Pnt
    from nhrep_gear import read_occt_solid

    with np.load(work / "assembly.npz") as deck:
        joints = deck["bodies"][2:]
    t = (row.step - 1 + (row.tick + row.span) / int(d["ticks"])) * float(d["h"])
    angles = joints[:, 7] + t * joints[:, 8] + np.array([row.q1, row.q2])
    rot = Rotation.from_rotvec(joints[:, 1:4] * angles[:, None]).as_matrix()
    reverse = hit[4] >= int(d["forward_n1"])
    target = int(reverse)
    loc = rot[target].T @ (rot[0] @ hit[6:9] + joints[0, 4:7] - joints[target, 4:7])
    target_name = "reducer-2b" if reverse else "reducer-2a"
    solid = read_occt_solid(
        Path(__file__).resolve().parents[2] / "results/gears/models" / (target_name + ".step")
    )
    point = gp_Pnt(*loc)
    classifier = BRepClass3d_SolidClassifier(solid, point, 1e-7)
    gap = BRepExtrema_DistShapeShape(BRepBuilderAPI_MakeVertex(point).Vertex(), solid)
    if classifier.State() != TopAbs_OUT or not gap.IsDone():
        raise ValueError("The original STEP solid did not confirm the outside witness")
    with np.load(work / ("reducer-2b.npz" if reverse else "reducer-2a.npz")) as data:
        v, inv = np.unique(data["tri"].reshape(-1, 3), axis=0, return_inverse=True)
        sd, idx, _, _ = igl.signed_distance(
            loc[None, :], v, inv.reshape(-1, 3), igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
        )
        return dict(
            time_s=t,
            stage=2,
            face_a=int(d["tags_a1"][int(hit[2])]),
            face_b=int(d["tags_b1"][int(hit[3])]),
            gpu_depth_um=1000 * hit[5],
            target_point_mm=loc.tolist(),
            normal_norm=float(np.linalg.norm(hit[9:12])),
            signed_distance_cpu_um=float(sd[0] * 1000),
            original_STEP_outside_gap_um=gap.Value() * 1000,
            nearest_target_face=int(data["faces"][idx[0]]),
            middle_torque_Nm=hit[12],
            probe_on="pinion" if reverse else "wheel",
        )


def compare(work, root):
    root.mkdir(parents=True, exist_ok=True)
    step = 20044
    old = np.load(work / "switch-cad-16.npy")
    original = np.load(work / "cad-h62.npy")
    assert np.array_equal(old, original[: len(old)])
    checks = {}
    traces = {}
    for name in ("switch-cad-16", "switch-mesh-16", "switch-cad-32", "same-start", "full-cad", "full-mesh"):
        trace = np.load(work / (name + ".npy"))
        report = json.loads((work / (name + ".json")).read_text())
        if report["terminated"] != "complete" or report["setup_s"] + report["online_s"] >= 1800:
            raise ValueError("Incomplete or over-budget run: " + name)
        with np.load(work / (name + ".npz")) as d:
            h = float(d["h"])
            checks[name] = stats(d, step, trace)
            before = int(d["first"])
            if name in ("same-start", "switch-cad-32"):
                np.testing.assert_array_equal(trace[:before], old[:before])
            if name == "switch-mesh-16":
                np.testing.assert_array_equal(trace, np.load(work / "mesh-h62.npy")[: len(trace)])
            checks[name].update(
                speed_rad_s=float(trace[step, 4] + 30 * np.pi),
                setup_s=report["setup_s"],
                online_s=report["online_s"],
                steps=report["steps"],
                over_budget=report["over_budget"],
                total_evaluations=report["trials"],
                total_increments=report["accepted_increments"],
                explicit_updates=report["explicit_crossings"],
                pool_MiB=report["device_pool_MiB"],
            )
            traces[name] = trace
            if name in ("same-start", "full-cad", "full-mesh"):
                regions = d["regions"]
                norm = np.linalg.norm(regions[:, 9:12], axis=1)
                if not len(norm) or not np.allclose(norm, 1, atol=1e-10, rtol=0):
                    raise ValueError("Non-unit corrected witness at the replayed main step")
                checks[name]["max_unit_normal_deviation"] = float(np.max(abs(norm - 1)))
    replay = pd.read_csv(work / "switch-detail.csv")
    with np.load(work / "switch-region.npz") as d, np.load(work / "switch-cad-16.npz") as base:
        np.testing.assert_array_equal(d["trials"], base["trials"][: len(d["trials"])])
        tr = pd.DataFrame(d["trials"], columns=d["columns"])
        regs = d["regions"]
        accepted = tr.index[(tr.step == step) & (tr.decision == 0)]
        bad = regs[
            (regs[:, 1] == 1)
            & np.isin(regs[:, 0], accepted)
            & (np.linalg.norm(regs[:, 9:12], axis=1) > 1.001)
        ]
        assert len(bad) > 0
        hit = bad[0]
        idx = int(hit[0])
        row = tr.iloc[idx]
        witness = cpu_witness(work, d, row, hit)
        witness["accepted_substeps_with_nonunit_normal"] = len(np.unique(bad[:, 0]))
        witness["first_corrupt_accepted_trial"] = idx
        witness["cold_face_torques"] = []
        for record in replay[
            (replay.trial == idx) & (replay.trajectory == "switch-cad-16") & (replay.stage == 2)
        ].itertuples():
            witness["cold_face_torques"].append(
                [record.method, record.face_a, record.face_b, record.torque_a_Nm]
            )
        cold_hits = witness["cold_face_torques"]
        if {r[0] for r in cold_hits} != {"cad", "mesh"} or any(
            (r[1], r[2]) == (witness["face_a"], witness["face_b"]) for r in cold_hits
        ):
            raise ValueError("A cold-query control retained the false face pair or returned no witnesses")
        keys = []
        for i in np.unique(regs[:, 0].astype(int)):
            if i not in accepted:
                continue
            rs = regs[regs[:, 0] == i]
            for p in (0, 1):
                rr = rs[rs[:, 1] == p]
                np.testing.assert_allclose(
                    rr[:, 12:14].sum(0),
                    tr.loc[i, [f"torque_a{p}", f"torque_b{p}"]].to_numpy(),
                    rtol=1e-11,
                    atol=1e-8,
                )
                for z in rr:
                    keys.append(
                        [
                            int(i),
                            p + 1,
                            int(d[f"tags_a{p}"][int(z[2])]),
                            int(d[f"tags_b{p}"][int(z[3])]),
                            float(z[5]),
                            float(np.linalg.norm(z[9:12])),
                            z[12],
                            z[13],
                        ]
                    )
        pd.DataFrame(
            keys,
            columns=[
                "trial",
                "stage",
                "face_a",
                "face_b",
                "depth_mm",
                "normal_norm",
                "torque_a_Nm",
                "torque_b_Nm",
            ],
        ).to_csv(root / "accepted-face-torques.csv", index=False)
    rd = np.load(work / "rd-reducer-3s-125us.npz")
    measured = rd["speeds"] * np.array([1.0, -1.0, 1.0])
    nominal = measured[0]
    with np.load(work / "rd-joints.npz") as d:
        q = d["angle_rad"] - d["angle_rad"][:1]
        twist = q[:, :2] - 2 * q[:, 1:]
    for name in ("full-cad", "full-mesh"):
        data = traces[name][::2]
        velocity = data[:, 3:6] + nominal
        q = data[:, :3] - data[:1, :3]
        phi = q[:, :2] - 2 * q[:, 1:]
        speed_err = velocity - measured
        checks[name].update(
            speed_RMS_rad_s=np.sqrt(np.mean(speed_err**2, axis=0)).tolist(),
            speed_max_difference_rad_s=np.max(abs(speed_err), axis=0).tolist(),
            twist_RMS_mrad=(1000 * np.sqrt(np.mean((phi - twist) ** 2, axis=0))).tolist(),
        )
    summary = dict(
        step=step,
        step_end_s=step * h,
        window_first_end_s=1.251,
        window_last_end_s=1.254,
        native_middle_speed_rad_s=float(measured[step // 2, 1]),
        matched_prefix_samples=before,
        witness=witness,
        runs=checks,
    )
    dense_path = work / "rd-switch-geometric-physics.npz"
    dense = None
    if dense_path.exists():
        with np.load(dense_path) as d:
            dense = -d["global_speed_rad_s"][:, 1]
            summary["native_refined_middle_speed_rad_s"] = float(dense[step // 2])
    (root / "switch.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with gzip.GzipFile(filename=str(root / "fixed-poses.csv.gz"), mode="wb", mtime=0) as dest, (
        work / "switch-detail.csv"
    ).open("rb") as src:
        shutil.copyfileobj(src, dest)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.7), layout="constrained")
    for name, label, color in (
        ("switch-cad-16", "Archived STEP", "#a85131"),
        ("same-start", "Corrected STEP (ours)", "#00716b"),
    ):
        with np.load(work / (name + ".npz")) as d:
            st = pd.DataFrame(d["trials"], columns=d["columns"])
            a = st[(st.step == step) & (st.decision == 0)]
            tm = (a.tick + a.span) / int(d["ticks"]) * h * 1e6
            axes[0].plot(tm, a.torque_a1, lw=1.4, marker=".", ms=3, label=label, color=color)
        ids = np.arange(step - 8, step + 9)
        axes[1].plot(
            (ids - step) * h * 1e6,
            nominal[1] + traces[name][ids, 4],
            lw=1.5,
            marker=".",
            color=color,
            label=label,
        )
    ids = np.arange(step - 8, step + 9, 2)
    axes[1].plot(
        (ids - step) * h * 1e6,
        measured[ids // 2, 1],
        color="#464b58",
        ls="--",
        marker=".",
        label="RecurDyn, default",
    )
    if dense is not None:
        axes[1].plot(
            (ids - step) * h * 1e6,
            dense[ids // 2],
            ls="-.",
            marker=".",
            color="#825d9c",
            label="RecurDyn, refined",
        )
    axes[0].set(xlabel="Time within the 62.5 µs step (µs)", ylabel="Stage 2 torque on\nmiddle shaft (N·m)")
    axes[1].set(xlabel="Time relative to 1.25275 s (µs)", ylabel="Middle shaft speed\n(rad/s)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(root / "switch.pdf")
    fig.savefig(root / "switch.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    compare(args.work, args.output)
