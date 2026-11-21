"""Fetch the NASA report and pinned healthy UNSW encoder records."""

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


NASA = "https://ntrs.nasa.gov/api/citations/19910013253/downloads/19910013253.pdf"
NASA_SHA256 = "171b3d29836d37624c004653f6911c65407993fb2b449cb281902933cff67ff2"
UNSW = "https://data.mendeley.com/public-api/datasets/p72x3m92cv/files?folder_id=root&version=1"
FILES = ("Data_05Hz_05Nm_H.mat", "Data_05Hz_10Nm_H.mat", "Data_10Hz_15Nm_H.mat")


def download(url, path, expected=None):
    def digest(file):
        with file.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    if path.exists() and (expected is None or digest(path) == expected):
        return digest(path)
    part = path.with_suffix(path.suffix + ".part")
    for attempt in range(5):
        size = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={size}-"} if size else {}
        try:
            with requests.get(url, headers=headers, timeout=(20,60), stream=True) as response:
                response.raise_for_status()
                resumed = response.status_code == 206 and response.headers.get(
                    "Content-Range", "").startswith(f"bytes {size}-")
                with part.open("ab" if resumed else "wb") as stream:
                    for block in response.iter_content(256*1024):
                        stream.write(block)
            break
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(1+attempt)
    checksum = digest(part)
    if expected is not None and checksum != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}")
    part.replace(path)
    print(f"Fetched {path.name}", flush=True)
    return checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    root = parser.parse_args().source
    root.mkdir(parents=True, exist_ok=True)
    nasa_sha = download(NASA, root / "nasa-tm-103232.pdf", NASA_SHA256)
    response = requests.get(UNSW, headers={"Accept": "application/vnd.mendeley-public-dataset.1+json"},
                            timeout=50)
    response.raise_for_status()
    index = {row["filename"]: row["content_details"] for row in response.json()}
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks = {
            name: pool.submit(download, index[name]["download_url"], root / name, index[name]["sha256_hash"])
            for name in FILES
        }
        unsw = {name: {"url": index[name]["download_url"], "sha256": job.result()}
                for name, job in tasks.items()}
    manifest = {"nasa": {"url": NASA, "sha256": nasa_sha}, "unsw": unsw,
                "unsw_dataset": "10.17632/p72x3m92cv.1", "unsw_license": "CC BY 4.0"}
    (root / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
