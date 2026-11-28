"""Run bounded, unfiltered production-graph diagnostics; not a timing benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("native_test", type=Path)
    parser.add_argument("scene", type=Path)
    parser.add_argument("sanitizer", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--racecheck", type=Path, help="Explicit separate racechecker executable")
    parser.add_argument("--dll-path", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if args.dll_path:
        env["PATH"] = str(args.dll_path) + os.pathsep + env["PATH"]
    racecheck = args.racecheck or args.sanitizer
    versions = {}
    for tool in {args.sanitizer, racecheck}:
        run = subprocess.run([str(tool), "--version"], env=env, capture_output=True,
                             text=True, check=True, timeout=60)
        versions[str(tool)] = run.stdout.strip()
    report = dict(runtime_sha256=digest(args.runtime), test_sha256=digest(args.native_test),
                  scene_sha256=digest(args.scene),
                  sanitizer_versions=versions, diagnostic_steps=50, step_s=.0001,
                  sanitizer_full_duration=False, checks=[])

    def check(name, command, diagnostic=None):
        command = [str(c) for c in command]
        if diagnostic:
            diagnostic.unlink(missing_ok=True)
        start = time.perf_counter()
        with (args.output / (name + "-console.log")).open("w", encoding="utf-8") as stream:
            try:
                process = subprocess.run(command, env=env, stdout=stream,
                                         stderr=subprocess.STDOUT, timeout=1800)
                code = process.returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        record = dict(name=name, command=command, exit_code=code,
                      process_s=time.perf_counter() - start)
        if diagnostic:
            log = diagnostic.read_text(encoding="utf-8", errors="replace") if diagnostic.exists() else ""
            record["zero_errors"] = ("ERROR SUMMARY: 0 errors" in log or
                                      "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)" in log)
            record["zero_errors"] &= "Internal Sanitizer Error" not in log
        report["checks"].append(record)
        (args.output / "checks.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: exit={code}, zero_errors={record.get('zero_errors')}", flush=True)
        if code != 0 or not record.get("zero_errors", True):
            raise SystemExit(1)

    check("native", [args.native_test])
    check("native-cuda", [args.native_test, "--cuda"])
    for kind, execution in [("synccheck", "wavefront"), ("synccheck", "batched"),
                            ("memcheck", "wavefront"), ("initcheck", "wavefront"),
                            ("racecheck", "wavefront")]:
        name = kind + "-" + execution
        dest = args.output / name
        dest.mkdir(exist_ok=True)
        log = args.output / (name + ".log")
        tool = racecheck if kind == "racecheck" else args.sanitizer
        check(name, [tool, "--tool", kind, "--error-exitcode", "97", "--log-file", log,
                     args.runtime, args.scene, "0", ".0001", ".005", "1", dest / "response",
                     "--method", "cad-compiled", "--device", "cuda-resident", "--solver", "penalty",
                     "--penalty-stiffness", "shared", "--geometry-update", "predictor",
                     "--spectral-radius", "0", "--region-reduction", "audit",
                     "--gpu-execution", execution], log)
    log = args.output / "synccheck-native.log"
    check("synccheck-native", [args.sanitizer, "--tool", "synccheck", "--error-exitcode", "97",
                               "--log-file", log, args.native_test, "--cuda"], log)


if __name__ == "__main__":
    main()
