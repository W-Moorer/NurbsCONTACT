"""Check persistent CAD table reuse and recovery using an isolated cache directory."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


def check(exe, shape):
    with TemporaryDirectory() as tmp:
        cache = Path(tmp)
        env = {**os.environ, "NC_CAD_CACHE_DIR": tmp}

        def prepare():
            run = subprocess.run(
                [exe, shape, shape, "6", ".001", "--prepare"],
                check=True,
                capture_output=True,
                text=True,
                env=env,
                timeout=1800,
            )
            return json.loads(run.stdout)

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda _: prepare(), range(2)))
        assert prepare() == {"tables_built": 0, "tables_loaded": 8}
        table = next(cache.glob("*-charts-6.bin"))
        with table.open("r+b") as stream:
            stream.seek(-1, os.SEEK_END)
            value = stream.read(1)
            stream.seek(-1, os.SEEK_END)
            stream.write(bytes([value[0] ^ 1]))
        repaired = prepare()
        assert repaired == {"tables_built": 1, "tables_loaded": 7}, repaired
        assert prepare() == {"tables_built": 0, "tables_loaded": 8}
        assert not list(cache.glob("*.tmp"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("exe", type=Path)
    p.add_argument("shape", type=Path)
    args = p.parse_args()
    check(args.exe.resolve(), args.shape.resolve())
