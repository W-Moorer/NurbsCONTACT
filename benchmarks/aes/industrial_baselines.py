"""Run the initial native and mesh refinement ladders for three mechanisms."""

import argparse
import csv
from pathlib import Path
from statistics import median
import subprocess


GEAR_STEPS = (20e-6, 10e-6, 5e-6)
CAM_STEPS = (.001, .0005, .00025, .000125, .0000625)
DURATIONS = {"spur": 2., "cam": 20., "reducer": 3.}


def run_case(command, prefix):
    with prefix.with_suffix(".log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        return {"status": f"exit_{result.returncode}", "setup_s": "", "online_s": ""}
    with prefix.with_suffix(".stats.csv").open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    setup = [float(row["prep_us"]) / 1e6 if "prep_us" in row else float(row["setup_s"]) for row in records]
    online = [float(row["wall_us"]) / 1e6 if "wall_us" in row else float(row["wall_s"]) for row in records]
    return {"status": "complete", "setup_s": median(setup), "online_s": median(online)}


def run(gears, hht, gear_models, folder, repeats, cases, series):
    for case in (name for name in cases if name != "cam"):
        duration = DURATIONS[case]
        if series == "time-check":
            specs = [("native-time-check", 0, GEAR_STEPS[-1]/2, 31)]
        elif series == "tight-mesh":
            specs = [("mesh", h, step, 31) for step in GEAR_STEPS[1:]
                     for h in (.00078125, .000390625, .0001953125)]
        elif series == "section-mesh":
            specs = [("section-mesh", -n, GEAR_STEPS[1], 31) for n in (64, 256, 1024)]
            specs.append(("section-mesh", -1024, GEAR_STEPS[-1], 31))
        elif series == "bounded":
            specs = [(method, 0, GEAR_STEPS[1], 31) for method in ("bounded-exact", "bounded-cache")]
        else:
            specs = [("native", 0, step, 31) for step in GEAR_STEPS]
            specs += [("mesh", h, GEAR_STEPS[1], 31) for h in (.00625, .003125, .0015625)]
            if case == "reducer":
                specs.append(("native-axial-check", 0, GEAR_STEPS[-1], 63))
        for method, deflection, step, stations in specs:
            resolution = f"{-deflection}segments" if method == "section-mesh" else f"{round(deflection*1e6)}nm"
            step_label = format(step*1e6, "g").replace(".", "p")
            tag = f"{case}-{method}-{resolution}-{step_label}us-{stations}"
            prefix = folder/tag
            command = [str(gears), "run", case, str(gear_models), str(deflection), str(step),
                       str(duration), str(prefix), str(repeats), str(stations), str(round(GEAR_STEPS[0]/step))]
            if method.startswith("bounded"):
                command.append(method)
            yield {"case": case, "method": method, "step_s": step, "duration_s": duration,
                   "deflection_mm": deflection if deflection >= 0 else "", "facets": -deflection if deflection < 0 else "",
                   "axial_stations": stations, "prefix": tag,
                   **run_case(command, prefix)}

    if "cam" not in cases or series in ("section-mesh", "time-check"):
        return
    models = folder/"cam-models"
    subprocess.run([str(hht), "models", str(models)], check=True, stdout=subprocess.DEVNULL)
    duration = DURATIONS["cam"]
    specs = [("native", "brep", 24, step, None) for step in CAM_STEPS]
    specs += [("mesh", "mesh", facets, CAM_STEPS[1], None) for facets in (24, 48, 96)]
    specs += [(mode, "brep", 24, CAM_STEPS[1], mode) for mode in ("bounded-exact", "bounded", "bounded-cache")]
    if series == "tight-mesh":
        specs = [("mesh-support", "mesh-support", facets, step, None)
                 for step in CAM_STEPS[2:4] for facets in (512, 2048, 8192)]
        specs += [("mesh", "mesh", 512, CAM_STEPS[2], None)]
    elif series == "bounded":
        specs = [spec for spec in specs if spec[-1] is not None]
    for method, backend, facets, step, mode in specs:
        tag = f"cam-{method}-{facets}-{round(step*1e6)}us"
        prefix = folder/tag
        command = [str(hht), "run", "cam", backend, str(facets), str(step), str(duration),
                   str(models), str(prefix), str(repeats)]
        if mode:
            command.append(mode)
        yield {"case": "cam", "method": method, "step_s": step, "duration_s": duration,
               "deflection_mm": "", "facets": facets, "axial_stations": "", "prefix": tag,
               **run_case(command, prefix)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gears", type=Path)
    parser.add_argument("hht", type=Path)
    parser.add_argument("gear_models", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cases", nargs="+", choices=tuple(DURATIONS), default=tuple(DURATIONS))
    parser.add_argument("--series", choices=("initial", "tight-mesh", "bounded", "section-mesh", "time-check"),
                        default="initial")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.series in ("section-mesh", "time-check") and args.cases == ["cam"]:
        parser.error(f"{args.series} is available for spur and reducer")
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    rows = run(args.gears.resolve(), args.hht.resolve(), args.gear_models.resolve(), folder,
               args.repeats, args.cases, args.series)
    with (folder/"observations.csv").open("w", newline="", encoding="utf-8") as stream:
        first = next(rows)
        writer = csv.DictWriter(stream, fieldnames=first, lineterminator="\n")
        writer.writeheader()
        writer.writerow(first)
        stream.flush()
        print(first["prefix"], first["status"], flush=True)
        for row in rows:
            writer.writerow(row)
            stream.flush()
            print(row["prefix"], row["status"], flush=True)


if __name__ == "__main__":
    main()
