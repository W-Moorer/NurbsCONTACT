"""Replay both stages on every saved pose of a completed five-second reducer run."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np


def replay(deck, mesh_dir, trace_path, exe, work, output, candidate_window=False,
           methods=("mesh-obb","mesh-aabb","mesh-sap"), same_face_only=False):
    trace = np.load(trace_path)
    meta = json.loads(trace_path.with_suffix(".json").read_text())
    if meta["terminated"] != "complete" or meta["physical_s"] != 5 or meta["steps"] != 50000:
        raise ValueError("expected a completed five-second run at 0.1 ms")
    if trace.shape != (50001, 14) or not np.isfinite(trace).all():
        raise ValueError("incomplete state history")
    parts = json.loads((deck / "models.json").read_text())["parts"]
    with np.load(deck / "assembly.npz") as d:
        bodies, mapping, pairs = (d[k] for k in ("bodies", "mapping", "pairs"))
    t = np.arange(len(trace)) * meta["step_s"]
    work.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    inputs = []
    for p, pair in enumerate(pairs):
        columns = []
        for i in pair:
            body = bodies[i]
            half = (body[7] + t*body[8] + trace[:, mapping[i]])/2
            columns.extend((np.broadcast_to(body[4:7]/1000, (len(t), 3)),
                            np.cos(half)[:, None], np.sin(half)[:, None]*body[1:4]))
        inp = work / f"poses-{p}.txt"
        np.savetxt(inp, np.column_stack(columns), fmt="%.17g")
        inputs.append(inp)
    report = dict(trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                  mesh_sha256=[p["sha256"] for p in parts], physical_s=5., steps=50000,
                  candidate_window=candidate_window, same_face_only=same_face_only, methods={}, incomplete={})
    flags = {}
    for method in methods:
        stages = []
        remaining = 1800.
        options = ["--candidate-window"] if candidate_window and method!="mesh-obb" else []
        if same_face_only: options.append("--same-face-only")
        for p, pair in enumerate(pairs):
            dest = work / f"{method}-{p}.jsonl"
            start = time.perf_counter()
            with inputs[p].open("rb") as poses, dest.open("wb") as rows:
                try:
                    subprocess.run([exe, "--method", method, "--any", *options, *[mesh_dir/parts[i]["mesh"] for i in pair]],
                                   stdin=poses, stdout=rows, timeout=max(0., remaining), check=True)
                except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as exc:
                    report["incomplete"][method]=dict(status="timeout" if isinstance(exc,subprocess.TimeoutExpired) else "failed",
                        stage=p+1, completed_stages=len(stages), process_s=1800-remaining+time.perf_counter()-start)
                    break
            wall = time.perf_counter()-start
            remaining -= wall
            with dest.open() as stream:
                rows = [json.loads(line) for line in stream]
            if len(rows) != len(trace):
                raise ValueError("missing geometric query")
            hit = np.array([r["overlap"] for r in rows])
            flags[f"{method}_{p}"] = hit
            loaded = trace[:, 6+4*p] > 0
            secs = np.array([r["query_s"] for r in rows])
            stages.append(dict(queries=len(hit), overlap_frames=int(sum(hit)), process_s=wall,
                               query_s=float(secs.sum()), p99_us=float(1e6*np.percentile(secs, 99)),
                               node_pairs=sum(r["node_pairs"] for r in rows),
                               triangle_tests=sum(r["triangle_tests"] for r in rows),
                               witness_reuses=sum(r["reused_triangle_pair"] for r in rows),
                               adjacent_witnesses=sum(r.get("adjacent_triangle_pair",False) for r in rows),
                               overlap_without_sampled_force=int(np.count_nonzero(hit & ~loaded)),
                               sampled_force_without_overlap=int(np.count_nonzero(loaded & ~hit))))
        if method not in report["incomplete"]:
            for p in range(len(pairs)):
                if any(not np.array_equal(flags[f"{old}_{p}"],flags[f"{method}_{p}"]) for old in report["methods"]):
                    raise ValueError("broad phases disagree on the same poses")
            report["methods"][method] = stages
        else:
            for p in range(len(stages)): flags.pop(f"{method}_{p}")
        print(method, report["incomplete"].get(method,stages), flush=True)
        np.savez_compressed(output.with_suffix(".npz"), **flags)
        output.write_text(json.dumps(report, indent=2)+"\n")
    return not report["incomplete"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("deck", "mesh_dir", "trace", "exe", "work", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--candidate-window", action="store_true")
    parser.add_argument("--same-face-only", action="store_true")
    parser.add_argument("--methods",nargs="+",choices=("mesh-obb","mesh-aabb","mesh-sap"),
                        default=("mesh-obb","mesh-aabb","mesh-sap"))
    a = parser.parse_args()
    if not replay(a.deck, a.mesh_dir, a.trace, a.exe.resolve(), a.work, a.output, a.candidate_window, a.methods, a.same_face_only):
        raise SystemExit(1)
