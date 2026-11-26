"""Compare the archived runtime parts to their source STEP files and run negative controls."""

import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checker", "step_directory", "part_directory", "output"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    manifest = Path(__file__).resolve().parents[2] / "results/aes/cpu-bezier/standalone/offline-parts.json"
    results = []
    for item in json.loads(manifest.read_text()):
        source = args.step_directory / item["source"]
        part = args.part_directory / item["part"]
        if digest(source) != item["source_sha256"] or digest(part) != item["sha256"]:
            raise ValueError(f"Input digest mismatch for {source.name}")
        run = subprocess.run([args.checker, source, part], capture_output=True, text=True, timeout=1800)
        if run.returncode:
            raise RuntimeError(f"{source.name}: {run.stdout} {run.stderr}")
        results.append(dict(source=source.name, source_sha256=item["source_sha256"], part_sha256=item["sha256"],
                            **json.loads(run.stdout)))
    first = results[0]["source"]
    source = args.step_directory / first
    controls = {}
    with TemporaryDirectory() as temporary:
        original = (args.part_directory / Path(first).with_suffix(".ncpart")).read_bytes()
        # NCDPART1 header (12 bytes), face domain (32), two boxes (48), tolerance (8), orientation.
        changed = bytearray(original)
        offset = 12 + 32 + 48 + 8
        sign, = struct.unpack_from("<i", changed, offset)
        if sign not in (-1, 1):
            raise ValueError("Unexpected runtime part layout")
        struct.pack_into("<i", changed, offset, -sign)
        copy = Path(temporary) / "reversed.ncpart"
        copy.write_bytes(changed)
        for name, test_part, reason in (
            ("flipped_face", copy, "orientation differs"),
            ("wrong_part", args.part_directory / Path(results[1]["source"]).with_suffix(".ncpart"), "face count differs"),
        ):
            run = subprocess.run([args.checker, source, test_part], capture_output=True, text=True, timeout=1800)
            if run.returncode == 0 or reason not in run.stderr:
                raise ValueError(f"{name}: expected rejection for {reason}, got {run.stderr}")
            controls[name] = run.stderr.strip()
    args.output.write_text(json.dumps(dict(parts=results, negative_controls=controls), indent=2) + "\n")


if __name__ == "__main__":
    main()
