"""Windows CPU process timings and crossed-binary controls for the reducer."""
import argparse
import csv
import ctypes as ct
import json
from pathlib import Path
import statistics
import subprocess
import time
from ctypes import wintypes as wt

from cpu_backend_run import digest, run


class PdhValue(ct.Structure):
    _fields_ = [("status", wt.DWORD), ("value", ct.c_double)]


class ProcessorCounters:
    def __init__(self):
        self.api = ct.WinDLL("pdh")
        self.handle = wt.HANDLE()
        self.counters = {}
        if self.api.PdhOpenQueryW(None, 0, ct.byref(self.handle)):
            return
        for name, entry in (("performance_pct", "% Processor Performance"), ("busy_pct", "% Processor Time"),
                            ("limit_pct", "% Performance Limit")):
            handle = wt.HANDLE()
            path = "\\Processor Information(_Total)\\" + entry
            if not self.api.PdhAddEnglishCounterW(self.handle, path, 0, ct.byref(handle)):
                self.counters[name] = handle
        self.api.PdhCollectQueryData(self.handle)

    def sample(self):
        values = {}
        if self.handle and not self.api.PdhCollectQueryData(self.handle):
            for name, handle in self.counters.items():
                result = PdhValue()
                if not self.api.PdhGetFormattedCounterValue(handle, 0x8200, None, ct.byref(result)) and result.status <= 1:
                    values[name] = round(result.value, 2)
        return values

    def close(self):
        if self.handle:
            self.api.PdhCloseQuery(self.handle)


def observer(output):
    def launch(args, stdout, stderr, timeout):
        counters = ProcessorCounters()
        kernel = ct.WinDLL("kernel32", use_last_error=True)
        started = time.perf_counter()
        samples = []
        progress = Path(args[args.index("--progress")+1]) if "--progress" in args else None
        try:
            with subprocess.Popen(args, stdout=stdout, stderr=stderr) as process:
                handle = wt.HANDLE(int(process._handle))
                while True:
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        if time.perf_counter() - started >= timeout:
                            process.kill()
                            process.wait()
                            raise subprocess.TimeoutExpired(args, timeout)
                    creation, finish, system, user = (ct.c_uint64() for _ in range(4))
                    cycles = ct.c_uint64()
                    if not kernel.GetProcessTimes(handle, ct.byref(creation), ct.byref(finish),
                                                  ct.byref(system), ct.byref(user)):
                        raise ct.WinError(ct.get_last_error())
                    if not kernel.QueryProcessCycleTime(handle, ct.byref(cycles)):
                        raise ct.WinError(ct.get_last_error())
                    row=dict(wall_s=round(time.perf_counter()-started, 4),user_s=user.value/1e7,
                             system_s=system.value/1e7,cycles=cycles.value,**counters.sample())
                    if progress and progress.is_file():
                        with progress.open("rb") as stream:
                            stream.seek(max(0,stream.seek(0,2)-4096))
                            lines=stream.read().split(b"\n")
                        if len(lines)>=2:
                            step=lines[-2].partition(b",")[0]
                            if step.isdigit():
                                row["step"]=int(step)
                    samples.append(row)
                    if process.returncode is not None:
                        return subprocess.CompletedProcess(args, process.returncode)
        finally:
            counters.close()
            output.write_text(json.dumps(samples, indent=1)+"\n")
    return launch


def crossed(scene, binaries, out):
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for index, exe in enumerate((binaries[0], binaries[1], binaries[1], binaries[0]), 1):
        prefix = out/f"control-{index}"
        args = [exe, scene, "0", ".0001", ".08", "1", prefix, "--method", "mesh-obb", "--solver", "penalty",
                "--mesh-region-scope", "contour-ray", "--penalty-stiffness", "shared", "--geometry-update", "predictor",
                "--spectral-radius", "0"]
        telemetry = prefix.with_suffix(".json")
        with prefix.with_suffix(".log").open("wb") as log:
            observer(telemetry)(args, stdout=log, stderr=subprocess.STDOUT, timeout=1800).check_returncode()
        with prefix.with_suffix(".stats.csv").open() as f:
            stats = next(csv.DictReader(f))
        samples = json.loads(telemetry.read_text())
        summary.append(dict(exe_sha256=digest(exe), response=digest(prefix.with_suffix(".csv")),
                            triangle_tests=int(stats["triangle_tests"]), detection_s=float(stats["mesh_detection_s"]),
                            wall_s=samples[-1]["wall_s"], user_s=samples[-1]["user_s"], system_s=samples[-1]["system_s"],
                            cycles=samples[-1]["cycles"], performance_median_pct=statistics.median(
                                [s["performance_pct"] for s in samples if "performance_pct" in s]) if any(
                                "performance_pct" in s for s in samples) else None))
        (out/"crossed.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(index, summary[-1], flush=True)
    if len({(r["response"],r["triangle_tests"]) for r in summary})!=1:
        raise ValueError("Mesh programs did different work")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("crossed","mesh-obb","cad-bezier"))
    p.add_argument("scene",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("binaries",nargs="+",type=Path)
    a=p.parse_args()
    if a.mode=="crossed":
        if len(a.binaries)!=2: p.error("crossed requires both archived executables")
        crossed(a.scene,a.binaries,a.out)
    else:
        run(a.binaries[0],a.scene,a.out,a.mode,"contour-ray","penalty","shared","predictor",0,
            launch=observer(a.out/"telemetry.json"))
