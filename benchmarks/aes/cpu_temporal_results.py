"""Compare the bounded CPU reducer runs and plot their accepted response."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cpu_foundation_results import reducer_prefix, trace


def collect(root, out):
    out.mkdir(parents=True, exist_ok=True)
    runs = {}
    progress = {}
    responses = {}
    for method in ("current-obb", "current-aabb", "original-obb"):
        directory = root / ("validated-reducer-" + method)
        record = json.loads((directory / "run.json").read_text())
        count = int(record["last_accepted"]["step"]) + 1
        progress[method] = trace(directory / "progress.csv", max_rows=count)
        if record["status"] == "complete":
            responses[method] = trace(directory / "response.csv")
        runs[method] = record

    result = dict(runs=runs, comparisons={})
    current = progress["current-obb"]
    for method in ("current-aabb", "original-obb"):
        other = progress[method]
        count = min(len(current["time_s"]), len(other["time_s"]))
        np.testing.assert_array_equal(current["time_s"][:count], other["time_s"][:count])
        difference = {key: float(max(abs(current[key][:count] - other[key][:count])))
                      for key in ("q0", "q1", "q2", "v0", "v1", "v2")}
        compared = dict(samples=count, shared_time_s=float(current["time_s"][count-1]),
                        max_abs_difference=difference,
                        current_integration_s=float(current["integration_s"][count-1]),
                        other_integration_s=float(other["integration_s"][count-1]))
        if method in responses and "current-obb" in responses:
            compared["identical_response_file"] = (
                runs[method]["response_sha256"] == runs["current-obb"]["response_sha256"])
            for k in ("p0_normal_N", "p1_normal_N", "p0_hit", "p1_hit"):
                compared["max_abs_difference"][k] = float(max(abs(responses[method][k] - responses["current-obb"][k])))
        result["comparisons"][method] = compared

    result["recurdyn"] = reducer_prefix(root, "validated-reducer-current-obb", out)
    if "current-obb" in responses:
        d = responses["current-obb"]
        t = d["time_s"]
        # The simulator reports a scalar sum of patch normal loads. RecurDyn's
        # vector resultant is compared separately by the fixed-pose wrench replay.
        loads = np.column_stack([d["p0_normal_N"], d["p1_normal_N"]])
        overview, axes = plt.subplots(2, 2, figsize=(10,5.7), layout="constrained")
        starts = np.arange(0, len(t), 50)
        counts = np.diff(np.append(starts, len(t)))
        block_time = np.add.reduceat(t, starts) / counts
        brake = (t >= .102 - 1e-10) & (t <= .114 + 1e-10)
        for stage, (full, detail) in enumerate(axes):
            f = loads[:,stage]*1e-3
            full.fill_between(block_time, np.minimum.reduceat(f,starts), np.maximum.reduceat(f,starts),
                              color="#acc9d8", label="5 ms range")
            full.plot(block_time, np.add.reduceat(f,starts)/counts, color="#215c78", lw=.85, label="5 ms mean")
            detail.plot(t[brake]*1e3, f[brake], color="#215c78", lw=1)
            full.set(ylabel=f"Stage {stage+1} normal load (kN)", xlim=(0,t[-1]))
            for axis in (full,detail): axis.grid(alpha=.16)
        axes[0,0].legend(frameon=False,ncol=2)
        axes[0,0].set_title("Full duty cycle")
        axes[0,1].set_title("Brake loading · individual time steps")
        axes[1,0].set_xlabel("Time (s)")
        axes[1,1].set_xlabel("Time (ms)")
        overview.savefig(out / "normal-loads.png", dpi=160)
        plt.close(overview)
        np.savez_compressed(out / "response.npz", time_s=t, normal_load_N=loads,
                            angle_rad=np.column_stack([d[f"q{i}"] for i in range(3)]),
                            speed_rad_s=np.column_stack([d[f"v{i}"] for i in range(3)]),
                            active=np.column_stack([d[f"p{i}_hit"].astype(bool) for i in range(2)]))
        result["peak_normal_load_N"] = np.max(loads,axis=0).tolist()
    (out / "summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("out",type=Path)
    args = parser.parse_args()
    collect(args.root,args.out)
