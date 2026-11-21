"""Fetch the pinned GAUGE observations and recover input states from early frames."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
from scipy.spatial.transform import Rotation


REPOSITORY = "https://huggingface.co/datasets/InternRobotics/GAUGE-Dataset"
REVISION = "f95bb06e7d11142f88ffd3b6b56c7258deb8fff6"
MATERIALS = ("wood", "plastic", "metal")
BALL_TRIALS = range(1, 23)
SLIDE_TRIALS = range(1, 21)
UNPUBLISHED_SLIDES = frozenset((("metal",17),))
CALIBRATION = frozenset((1, 2, 3, 4, 5))
G = 9.81


def ball_path(trial):
    return f"data/rigid/bouncing ball/json/{trial}.json"


def slider_path(material, trial):
    return f"data/rigid/slope slider/json/{material}/task-2/{trial}.json"


def fetch(source):
    paths = ["metadata/rigid/bouncing ball.json", "metadata/rigid/slope slider.json",
             "assets/obj/ball-6.obj", "assets/obj/board.obj", "assets/obj/cube.obj"]
    paths += [ball_path(i) for i in BALL_TRIALS]
    paths += [slider_path(m, i) for m in MATERIALS for i in SLIDE_TRIALS
              if (m,i) not in UNPUBLISHED_SLIDES]

    def download(name):
        target = source/name
        url = f"{REPOSITORY}/resolve/{REVISION}/{quote(name)}"
        if not target.is_file():
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=40) as response:
                data = response.read()
            if name.endswith(".json"):
                json.loads(data)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return name, {"url": url, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}

    with ThreadPoolExecutor(max_workers=8) as workers:
        files = dict(workers.map(download, paths))
    manifest = {"revision": REVISION, "repository": REPOSITORY, "license": "MIT", "files": files}
    (source/"sources.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return manifest


def load(source, name):
    return json.loads((source/name).read_text(encoding="utf-8"))


def positions(record, body):
    if record["translation unit"] != "mm":
        raise ValueError("unexpected GAUGE translation unit")
    return .001*np.column_stack([record[body][column] for column in "xyz"])


def dimensions(source, name):
    points = np.array([list(map(float, line.split()[1:4]))
                       for line in (source/f"assets/obj/{name}.obj").read_text(encoding="utf-8",errors="replace").splitlines()
                       if line.startswith("v ")])
    return np.ptp(points, axis=0)


def model_inputs(source):
    bounce = load(source, "metadata/rigid/bouncing ball.json")
    slider = load(source, "metadata/rigid/slope slider.json")
    diameter = dimensions(source, "ball-6")
    cube = dimensions(source, "cube")
    board = dimensions(source, "board")
    if np.ptp(diameter)>5e-4 or not (.05<diameter.mean()<.07):
        raise ValueError("unexpected GAUGE ball dimensions")
    if np.ptp(cube)>5e-4 or not (.04<cube.mean()<.06) or not (.4<board[0]<.6):
        raise ValueError("unexpected GAUGE board dimensions")
    return {
        "radius": float(diameter.mean()/2), "cube": cube.tolist(), "board": board.tolist(),
        "ball_mass": bounce["assets"]["ball"]["material"]["soft"]["mass"],
        "floor_top": bounce["tasks"]["default"]["translation"]["ground"][2],
        "slider_materials": slider["assets"]["cube"]["material"],
    }


def bounce_case(source, inputs, trial):
    record = load(source, ball_path(trial))
    xyz = positions(record, "ball")
    fps = record["FPS"]
    floor_center = inputs["floor_top"]+inputs["radius"]
    z = xyz[:, 2]
    air = np.flatnonzero((z[:-1]-z[1:]>.020)&(z[1:]>floor_center+.12))
    if not len(air):
        raise ValueError("no resolved pre-impact fall")
    start = int(air[0])
    stop = start+4
    if z[stop]<=floor_center+.10 or np.any(np.diff(z[start:stop+1])>=0):
        raise ValueError("insufficient pre-impact flight samples")
    offsets = (np.arange(start,stop+1)-stop)/fps
    vz, z0 = np.polyfit(offsets,z[start:stop+1]+.5*G*offsets**2,1)
    count = min(len(z)-stop, round(1.0*fps)+1)
    if count<round(.90*fps):
        raise ValueError("short bounce trajectory")
    return {"id": trial, "initial_frame": stop, "initial_velocity": float(vz), "initial_height": float(z0),
            "initial_fit_frames": [start, stop], "time": np.arange(count)/fps, "positions": xyz[stop:stop+count],
            "observed": z[stop:stop+count]-floor_center, "scale": float(z0-floor_center)}


def slider_case(source, inputs, material, trial):
    record = load(source, slider_path(material, trial))
    xyz = positions(record, "cube")
    board = positions(record, "board")
    fps = record["FPS"]
    angles = np.column_stack([record["board"][key] for key in ("qx","qy","qz","qw")])
    # The board is fixed during task 2. Its first three frames measure its pose.
    frame = Rotation.from_quat(angles[:3]).mean()
    matrix = frame.as_matrix()
    origin = board[:3].mean(axis=0)
    points = (xyz-origin) @ matrix
    tangent = points[:, 0]
    candidates = np.flatnonzero(np.diff(tangent)>.004)
    if not len(candidates):
        raise ValueError("no resolved release")
    start = int(candidates[0])+1
    if start<1 or start+2>=len(xyz):
        raise ValueError("insufficient initial slider frames")
    speed = (tangent[start+1]-tangent[start-1])*fps/2
    # Score full support, based on the measured board and nominal cube geometry.
    limit = (inputs["board"][0]-inputs["cube"][0])/2
    supported = np.flatnonzero(tangent[start+2:] > limit)
    end = start+2+int(supported[0]) if len(supported) else len(xyz)
    if end-start<8:
        raise ValueError("short fully supported slide")
    t = (np.arange(start,end)-start)/fps
    rotvec = frame.as_rotvec()
    angle = float(np.linalg.norm(rotvec))
    return {"id": trial, "material": material, "initial_frame": start, "time": t,
            "positions": xyz[start:end], "observed": tangent[start:end], "matrix": matrix,
            "board_origin": origin, "axis": rotvec/angle, "angle": angle,
            "initial_speed": float(speed), "initial_local": points[start],
            "scale": float(tangent[end-1]-tangent[start]), "score_begin": 2}


def step_models(exe, source, output):
    m = model_inputs(source)
    output.mkdir(parents=True, exist_ok=True)
    recipes = {"ball":("sphere",1000*m["radius"]), "floor":("box",1200,1000,10,-5),
               "cube":("box",*(1000*np.asarray(m["cube"])),0),
               "board":("box",*(1000*np.asarray(m["board"])),0)}
    for name, recipe in recipes.items():
        result = subprocess.run([str(exe),"make",*map(str,recipe),str(output/f"{name}.step")],
                                capture_output=True,text=True)
        if result.returncode:
            raise RuntimeError(f"STEP export failed for {name}: {result.stderr}")
    (output/"geometry.json").write_text(json.dumps(m,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe",type=Path)
    parser.add_argument("source",type=Path)
    parser.add_argument("models",type=Path)
    args=parser.parse_args()
    fetch(args.source)
    step_models(args.exe,args.source,args.models)
