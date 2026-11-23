"""Persist immutable device query tables for a rigid part; pose histories stay per run."""

import hashlib
import argparse
import json
from pathlib import Path
import tempfile
from time import perf_counter

import cupy as cp
import numpy as np


def index_path(folder, tri, tags, shift, pitch):
    digest = hashlib.sha256()
    files = Path(__file__).parent
    for name in ("resident_contact.cu", "resident_index.py", "resident_contact.py", "gpu_mesh_line.py"):
        digest.update((files / name).read_bytes())
    digest.update(
        str((float(pitch), cp.cuda.Device().compute_capability, cp.cuda.runtime.runtimeGetVersion())).encode()
    )
    for a in (tri, tags, shift):
        a = np.ascontiguousarray(a)
        digest.update(str((a.shape, a.dtype.str)).encode())
        digest.update(a.view(np.uint8))
    return folder / (digest.hexdigest() + ".npz")


def load(path):
    with np.load(path, allow_pickle=False) as data:
        if str(data["identity"]) != path.stem:
            raise ValueError("Incorrect static patch index identity")
        geometry = tuple(cp.asarray(data[f"g{i}"]) for i in range(14))
        return geometry, data["tags"], (cp.asarray(data["controls"]),)


def save(path, geometry, tags, field):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as tmp:
        staged = Path(tmp.name)
    try:
        np.savez(
            staged,
            identity=path.stem,
            tags=tags,
            controls=field[0].get(),
            **{f"g{i}": a.get() for i, a in enumerate(geometry)},
        )
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def prepare(deck, directory, pitch):
    """Use the simulation's geometry constructor to prepare each immutable part index."""
    from resident_contact import ResidentMeshContact
    from resident_index import planar_regions

    started = perf_counter()
    parts = []
    for entry in json.loads((deck / "models.json").read_text())["parts"]:
        with np.load(deck / entry["archive"], allow_pickle=False) as z:
            tri, faces, shift, points = (z[k] for k in ("tri", "faces", "shift", "points"))
        tic = perf_counter()
        mapped = planar_regions(tri, faces)[np.unique(faces, return_inverse=True)[1]]
        model = ResidentMeshContact(
            tri,
            tri,
            mapped,
            faces,
            pitch=pitch,
            lanes=4,
            backend="patch",
            patch_shift=shift,
            source_points=points,
            index_cache=directory,
        )
        cp.cuda.get_current_stream().synchronize()
        elapsed = perf_counter() - tic
        path = index_path(directory, tri, mapped, shift, pitch)
        parts.append(
            dict(
                part=entry["step"],
                key=path.stem,
                bytes=path.stat().st_size,
                restored=model.index_hit,
                setup_s=elapsed,
            )
        )
        del model
    return dict(wall_s=perf_counter() - started, pitch_mm=pitch, parts=parts)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("deck", type=Path)
    p.add_argument("directory", type=Path, nargs="?", help="Defaults to patch-index inside the prepared deck")
    p.add_argument("--pitch", type=float, default=0.8)
    p.add_argument("--report", type=Path)
    a = p.parse_args()
    details = json.dumps(prepare(a.deck, a.directory or a.deck / "patch-index", a.pitch), indent=2)
    if a.report:
        a.report.write_text(details + "\n")
    print(details)
