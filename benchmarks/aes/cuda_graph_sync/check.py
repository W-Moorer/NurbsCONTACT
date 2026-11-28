"""Check warp synchronization in conditional graphs, including a negative control."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("sanitizer", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for graph, sync, threads in (("flat", "block", 128), ("if", "block", 128),
                                  ("while", "block", 128), ("while", "warp", 32),
                                  ("while", "warp", 128), ("while", "invalid", 32)):
        name = f"{graph}-{sync}-{threads}"
        log = args.output / f"{name}.log"
        command = [str(args.sanitizer), "--tool", "synccheck", "--error-exitcode", "97",
                   "--print-limit", "5", "--log-file", str(log), str(args.probe),
                   graph, sync, str(threads)]
        start = perf_counter()
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=1800)
            code, console = run.returncode, run.stdout + run.stderr
        except subprocess.TimeoutExpired:
            code, console = "timeout", ""
        (args.output / f"{name}-console.log").write_text(console, encoding="utf-8")
        diagnostic = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        clean = (code == 0 and "ERROR SUMMARY: 0 errors" in diagnostic
                 and "Internal Sanitizer Error" not in diagnostic and "bad=0" in console)
        detected = code == 97 and "Barrier error detected" in diagnostic
        required = sync != "block" or graph != "while"
        passed = detected if sync == "invalid" else clean
        reports.append(dict(command=command, exit_code=code, process_s=perf_counter()-start,
                            clean=clean, error_detected=detected, required=required, passed=passed))
        (args.output / "checks.json").write_text(json.dumps(dict(
            executable_sha256=hashlib.sha256(args.probe.read_bytes()).hexdigest(),
            checks=reports), indent=2) + "\n", encoding="utf-8")
        print(f"{name}: exit={code}, clean={clean}, detected={detected}", flush=True)
    if any(r["required"] and not r["passed"] for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
