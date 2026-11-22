"""Write nc_contact decks for full STEP cam, spur, and reducer body pairs."""

import argparse
import csv
import math
import os
from pathlib import Path


def step_path(path, folder):
    if not path.is_file():
        raise FileNotFoundError(path)
    return Path(os.path.relpath(path, folder)).as_posix()


def cam(models, output, duration):
    follower = step_path(models / "follower.step", output.parent)
    cam = step_path(models / "cam.step", output.parent)
    output.write_text(
        f'nc-contact-v2\n'
        f'gravity 0 0 0 2 2 1\n'
        f'coordinate .007 0 4 1.8 .10 0 0\n'
        f'coordinate .48 0 .096 70 .65 0 -3.2\n'
        f'body "{follower}" 0 0 .115 0 0 0 1 0 0 0 0 -1 0 0 0 1 1 0 1 0\n'
        f'body "{cam}" 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0\n'
        f'closest 0 1 .012 5000 1 0 0 .0005 .012\n'
        f'drives 0\nloads 1\nload 0 2\n0 .4\n{duration:.17g} {.4 + 7.2 * duration:.17g}\n',
        encoding="ascii")


def gears(models, output, duration, brake):
    with (models / f"{output.stem}.csv").open(newline="", encoding="utf-8") as stream:
        parts = list(csv.DictReader(stream))
    stages = list(zip(parts[::2], parts[1::2], strict=True))
    speeds = [0.0] * (len(stages) + 1)
    inertia = [0.0003] * len(speeds)
    preload = [0.0] * len(speeds)
    ratios = [int(b["teeth"]) / int(a["teeth"]) for a, b in stages]
    input_torque = brake / math.prod(ratios)
    torque = input_torque
    for (a, b), ratio in zip(stages, ratios, strict=True):
        helix = math.radians(float(a["helix_deg"]))
        pressure = math.atan(math.tan(math.radians(20)) / math.cos(helix))
        c = math.cos(math.atan(math.tan(helix) * math.cos(pressure)))
        ra, rb = float(a["base_radius_mm"]), float(b["base_radius_mm"])
        force = torque / (ra * .001 * c)
        closed = .09 - (.055 - force / 1.6e5) / c
        i, j = int(a["shaft"]), int(b["shaft"])
        preload[j] = (ra * preload[i] - closed) / rb
        torque *= ratio

    for part in parts:
        shaft = int(part["shaft"])
        speeds[shaft] = math.tau * float(part["start_rpm"]) / 60
        radius = float(part["pitch_radius_mm"]) * .001
        bore = .32 * radius
        width = float(part["width_mm"]) * .001
        inertia[shaft] += .5 * math.pi * 7850 * width * (radius**4 - bore**4)

    deck = ["nc-contact-v2", f"gravity 0 0 0 {len(speeds)} {len(parts)} {len(stages)}"]
    for i, (mass, velocity, q) in enumerate(zip(inertia, speeds, preload, strict=True)):
        damping = .69 if i == 0 else .09
        stiffness = 1800 if i == 0 else 0
        deck.append(f"coordinate {mass:.17g} {q:.17g} {velocity:.17g} {stiffness} {damping} 0 {damping * velocity:.17g}")
    for part in parts:
        path = step_path(models / part["step"], output.parent)
        x, z = (float(part[key]) * .001 for key in ("center_x_mm", "z_mm"))
        sign = int(part["direction"])
        angle = sign * float(part["angle_zero_rad"])
        deck.append(f'body "{path}" 0 {x:.17g} 0 {z:.17g} 0 0 {sign} {angle:.17g} '
                    f'0 0 0 {part["shaft"]} {x:.17g} 0 {z:.17g} 0')
    for i in range(len(stages)):
        deck.append(f"closest {2*i} {2*i+1} .000055 1.6e8 1 0 0 .0005 .000053")
    count = math.ceil(duration / .0005)
    times = [duration * i / count for i in range(count + 1)]
    deck.extend(("drives 0", "loads 2", f"load 0 {len(times)}"))
    for time in times:
        torque = input_torque * (1 + .10 * math.sin(math.tau * 23 * time)) + 1800 * speeds[0] * time
        deck.append(f"{time:.17g} {torque:.17g}")
    deck.append(f"load {len(speeds)-1} {len(times)}")
    for time in times:
        rise = .5 * (1 + math.tanh((time - .10) / .018))
        release = .5 * (1 + math.tanh((time - .24) / .028))
        load = brake * (1 + .26 * rise - .34 * release + .06 * math.sin(math.tau * 13 * time))
        deck.append(f"{time:.17g} {-load:.17g}")
    output.write_text("\n".join(deck) + "\n", encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cam_models", type=Path)
    parser.add_argument("gear_models", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cam-duration", type=float, default=20)
    parser.add_argument("--gear-duration", type=float, default=.36)
    args = parser.parse_args()
    if not all(math.isfinite(t) and t > 0 for t in (args.cam_duration, args.gear_duration)):
        parser.error("durations must be positive and finite")
    args.output.mkdir(parents=True, exist_ok=True)
    cam(args.cam_models, args.output / "cam.scene", args.cam_duration)
    for case, torque in (("spur", 56), ("reducer", 148)):
        gears(args.gear_models, args.output / f"{case}.scene", args.gear_duration, torque)


if __name__ == "__main__":
    main()
