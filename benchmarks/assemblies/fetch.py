"""Fetch and verify the public OpenAMRobot assembly inputs."""

import argparse
import hashlib
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen


COMMIT = "8ed2b06a46115abe2dffadbf0d902ae731d278d1"
BASE = f"https://raw.githubusercontent.com/openAMRobot/openamr-platform-hw/{COMMIT}/mechanical/cad/production_files/"
ASSEMBLIES = (
    ("mmp-wheel.step", "MMP.03.00.00.000 Wheel assembly.STEP",
     "aa6e37556754cd7baec59cbacdc50c13becb1929b2e6dccc3a6efeade836d748"),
    ("mmp-lidar.step", "MMP.07.00.00.000 Lidar support assembly.STEP",
     "a3c20d014753bf26cdda6c26a917319edb9687dc929bb92de8e5926888423f27"),
)


def verified_download(folder, local_name, source_name, sha256):
    target = folder / local_name
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == sha256:
        return target
    with urlopen(BASE + quote(source_name), timeout=90) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != sha256:
        raise ValueError(f"SHA-256 mismatch for {source_name}")
    target.write_bytes(content)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for entry in ASSEMBLIES:
        print(verified_download(args.directory, *entry))


if __name__ == "__main__":
    main()
