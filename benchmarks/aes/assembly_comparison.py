"""Score completed shared-shaft runs on the common physical time grid."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid


def compare(work, root, cam_control):
    root.mkdir(parents=True, exist_ok=True)
    with np.load(work / "assembly.npz") as model:
        rows = model["bodies"]
        h = float(model["h"])
    first = [0, 1, 3]
    nominal = rows[first, 8]
    sign = rows[first, 3]
    summary = dict(
        parts=json.loads((work / "models.json").read_text()),
        trials={},
        comparison={},
    )
    if cam_control is not None:
        cam = np.load(work / "cam-audit.npy")
        control = np.load(cam_control)[: len(cam)]
        summary["cam_driver_check"] = dict(
            duration_s=(len(cam) - 1) * h,
            max_position_change_um=float(1e6 * np.max(abs(cam[:, 0] - control[:, 0]))),
            max_angle_change_rad=float(np.max(abs(cam[:, 1] - control[:, 1]))),
            max_force_change_N=float(np.max(abs(cam[:, 4] - control[:, 4]))),
        )
    curves = {}
    for name in (
        "cad",
        "mesh",
        "cad-active8",
        "mesh-active8",
        "cad-refined",
        "mesh-refined",
        "cad-deep",
        "mesh-deep",
        "cad-crossing",
        "mesh-crossing",
        "mesh-h62",
        "cad-h62",
    ):
        report = work / f"{name}.json"
        if not report.exists():
            continue
        result = json.loads(report.read_text())
        summary["trials"][name] = result
        if result["terminated"] != "complete":
            continue
        trace = np.load(work / f"{name}.npy")
        stride = round(h / result["step_s"])
        if (
            stride < 1
            or not np.isclose(h, stride * result["step_s"], atol=1e-14, rtol=0)
            or trace.shape != (24000 * stride + 1, 14)
            or not np.isfinite(trace).all()
            or result["physical_s"] != 3.0
        ):
            raise ValueError("Incomplete accepted trajectory")
        trace = trace[::stride]
        speed = trace[:, 3:6] + nominal
        angle = trace[:, :3] - trace[0, :3]
        twist = angle[:, :2] - 2 * angle[:, 1:3]
        rel = speed[:, :2] - 2 * speed[:, 1:3]
        reconstructed = cumulative_trapezoid(rel, dx=h, axis=0, initial=0)
        result["twist_quadrature_rms_rad"] = np.sqrt(np.mean((twist - reconstructed) ** 2, axis=0)).tolist()
        curves[name] = (speed, twist, trace[:, [6, 10]], trace[:, [7, 8, 11, 12]])
    extras = {}
    rd = work / "rd-reducer-3s-125us.npz"
    if rd.exists():
        with np.load(rd) as data:
            times, speed, normal = (data[k] for k in ("t", "speeds", "normal"))
        if len(times) != 24001 or not np.allclose(times, np.arange(len(times)) * h, atol=1e-10, rtol=0):
            raise ValueError("External reference grid mismatch")
        speed = speed * sign
        np.testing.assert_allclose(speed[0], nominal, atol=1e-8, rtol=0)
        with np.load(work / "rd-joints.npz") as channels:
            angle, reference = channels["angle_rad"], channels["reference_rad"]
            np.testing.assert_allclose(reference, times[:, None] * nominal, rtol=0, atol=1e-9)
            np.testing.assert_allclose(
                np.mean(np.diff(angle, axis=0), axis=0) / h, np.mean(speed, axis=0), rtol=0, atol=0.1
            )
            angle = angle - angle[:1]
            twist = angle[:, :2] - 2 * angle[:, 1:3]
            extras["RecurDyn_resultant_N"] = channels["resultant_N"]
            extras["RecurDyn_point1_normal_N"] = normal
            extras["RecurDyn_joint_rad"] = angle
            friction = np.max(abs(channels["point1_friction_N"]), axis=0)
        summary["recurdyn"] = json.loads((work / "rd-full.json").read_text())
        summary["recurdyn"]["peak_point1_friction_N"] = friction.tolist()
        summary["recurdyn"]["peak_resultant_N"] = np.max(extras["RecurDyn_resultant_N"], axis=0).tolist()
        curves["RecurDyn"] = (speed, twist, None, None)
        for name, (omega, delta, normal_sum, _) in curves.items():
            if name == "RecurDyn":
                continue
            idx = int(np.argmax(abs(omega[:, 1] - speed[:, 1])))
            summary["comparison"][name] = dict(
                speed_rms_rad_s=np.sqrt(np.mean((omega - speed) ** 2, axis=0)).tolist(),
                twist_rms_mrad=(1000 * np.sqrt(np.mean((delta - twist) ** 2, axis=0))).tolist(),
                speed_max_rad_s=np.max(abs(omega - speed), axis=0).tolist(),
                largest_middle_shaft_difference=dict(
                    time_s=float(times[idx]),
                    gpu_rad_s=float(omega[idx, 1]),
                    recurdyn_rad_s=float(speed[idx, 1]),
                    summed_region_normal_N=normal_sum[idx].tolist(),
                ),
            )
    data = {"time_s": np.arange(24001) * h, **extras}
    selected = [name for name in ("cad-refined", "cad-h62", "mesh-h62", "RecurDyn") if name in curves]
    for name in selected:
        omega, twist, force, torque = curves[name]
        data[name + "_rad_s"] = omega
        data[name + "_twist_rad"] = twist
        if force is not None:
            data[name + "_summed_normal_N"] = force
        if torque is not None:
            data[name + "_torque_Nm"] = torque
    np.savez_compressed(root / "responses.npz", **data)
    (root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    if {"cad-h62", "mesh-h62", "cad-refined"}.issubset(selected):
        selected.remove("cad-refined")
    if not selected:
        return
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.4, 4.5), layout="constrained")
    for name in selected:
        label = (
            "RecurDyn"
            if name == "RecurDyn"
            else ("STEP probes (ours)" if name.startswith("cad") else "Mesh probes")
        )
        omega, delta, _, _ = curves[name]
        axes[0].plot(data["time_s"], omega[:, 1], lw=0.65, alpha=0.8, label=label)
        axes[1].plot(data["time_s"], delta[:, 1] * 1000, lw=0.65, alpha=0.8)
    axes[0].set_ylabel("Middle shaft (rad/s)")
    axes[1].set_ylabel("Stage 2 relative twist (mrad)")
    axes[1].set_xlabel("Time (s)")
    axes[0].legend(frameon=False, ncol=3)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", lw=0.5, color=".85")
        ax.set_xlim(0, 3)
    fig.savefig(root / "shaft-response.pdf")
    fig.savefig(root / "shaft-response.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("root", type=Path)
    p.add_argument(
        "--cam-control", type=Path, help="Also compare cam-audit.npy against this two-coordinate trace"
    )
    args = p.parse_args()
    compare(args.work, args.root, args.cam_control)
