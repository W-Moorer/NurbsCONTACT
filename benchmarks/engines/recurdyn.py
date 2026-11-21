"""Run matched analytic contact cases through an active RecurDyn ProcessNet session."""

import argparse
import json
from pathlib import Path
import re
import sys
from time import perf_counter

import numpy as np

from rigid import AES, CASES, DURATION, LAYER, STIFFNESS, case_values, initial_velocity, statistics


MM = 1000


def connect(api_dir):
    sys.path.insert(0, str(api_dir.resolve()))
    import recurdyn
    from win32com.client import GetActiveObject

    app = recurdyn.IApplication(GetActiveObject("RecurDyn.RDApplication").RecurDynApplication)
    return app, recurdyn


def solver_cpu(path):
    files = list(path.parent.glob(f"{path.stem}_*/{path.stem}.msg"))
    if not files:
        return None
    latest = max(files, key=lambda file: file.stat().st_mtime_ns)
    content = latest.read_text(encoding="ascii", errors="ignore")
    if "Analysis is accomplished successfully" not in content:
        raise RuntimeError(f"RecurDyn did not report success for {path.name}")
    found = re.search(r"CPU Time\s*=\s*(\d+(?:\.\d+)?) Second", content)
    return float(found.group(1)) if found else None


def simulate(app, rd, case, step, path):
    setup_start = perf_counter()
    doc = rd.IModelDocument(app.NewModelDocumentWithUnitSystem(path.stem, rd.UnitSystem.MMKS))
    try:
        model = rd.ISubSystem(doc.Model)
        ground = rd.IBody(model.Ground)
        doc.ModelProperty.UseGravity = False
        frame = doc.CreateReferenceFrame()
        height, _, count = case_values(case, AES)

        def ball(name, origin, proxy_radius, velocity):
            frame.SetOrigin(0, 0, origin*MM)
            body = rd.IBody(model.CreateBodyGeneral(name))
            shape = rd.IGeometrySphere(body.CreateSphereGeometry(name+"_solid", frame, proxy_radius*MM))
            body.MaterialInput = rd.MaterialInput.UserInput
            body.Mass.Value = 1
            for moment in (body.Ixx, body.Iyy, body.Izz):
                moment.Value = .4 * (AES.radius*MM)**2
            body.UseInitialTranslationalVelocityZ = True
            body.InitialTranslationalVelocityZ.Value = velocity*MM
            joint = rd.IJointTranslational(model.CreateJointTranslational(name+"_slide", ground, body, frame))
            joint.InitialVelocity.Value = velocity*MM
            return body, shape

        radii = AES.radius + (LAYER/2 if count == 2 else 0)
        velocity = initial_velocity(count, AES)
        upper, upper_shape = ball("Upper", height, radii, velocity[-1])
        if count == 1:
            # In this primitive constructor the reference is the lower face.
            frame.SetOrigin(0, 0, LAYER*MM - 10)
            floor = rd.IGeometryBox(ground.CreateBoxGeometry("Floor", frame, 1000, 1000, 10))
            contact = rd.IContactSphereToBox(model.CreateContactSphereToBox2("Contact", floor, upper_shape))
            bodies = (upper,)
        else:
            lower, lower_shape = ball("Lower", 0, radii, velocity[0])
            contact = rd.IContactSphereToSphere(model.CreateContactSphereToSphere2("Contact", lower_shape, upper_shape))
            bodies = (lower, upper)

        props = contact.ContactProperty
        props.StiffnessCoefficient.Value = STIFFNESS/MM
        props.UseStiffnessExponent = True
        props.StiffnessExponent.Value = 1
        props.DampingCoefficient.Value = 0
        dynamic = doc.ModelProperty.DynamicAnalysisProperty
        count_steps = round(DURATION/step)
        dynamic.SimulationTime.Value = DURATION
        dynamic.SimulationStep.Value = count_steps
        dynamic.MaxTimeStep.Value = step
        dynamic.MatchSolvingStepSize = True
        if not doc.Validate():
            raise RuntimeError(f"RecurDyn model validation failed for {path.name}")
        doc.FileSave(str(path.resolve()), True)
        setup = perf_counter() - setup_start
        start = perf_counter()
        doc.Analysis(rd.AnalysisMode.Dynamic)
        wall = perf_counter() - start

        time = np.asarray(doc.GetPlotDataTime(), dtype=float)
        if len(time) != count_steps + 1 or not np.allclose(time, np.arange(count_steps + 1)*step, rtol=0, atol=1e-11):
            raise RuntimeError(f"RecurDyn did not produce the requested output grid for {path.name}")
        position = np.column_stack([doc.GetPlotData(body, "Pos_TZ") for body in bodies])/MM
        position -= position[0].copy()
        force = np.asarray(doc.GetPlotData(contact, "NormalForce1"), dtype=float)
        return position, force, setup, wall, solver_cpu(path)
    finally:
        app.CloseModelDocument(doc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--api-dir", type=Path, required=True, help="RecurDyn Bin/Python/Lib")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    app, rd = connect(args.api_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    models = args.output/"models"
    models.mkdir(exist_ok=True)
    runs = []
    for case in CASES:
        for step in AES.steps:
            label = f"{case}-{round(step*1e6)}us"
            measurements = [simulate(app, rd, case, step, models/f"{label}-{rep}.rdyn") for rep in range(args.repeats)]
            position, force, _, _, _ = measurements[0]
            result, trace = statistics(case, step, position, force,
                                       [(setup, wall) for _, _, setup, wall, _ in measurements], AES)
            result["solver_cpu_s"] = [row[4] for row in measurements]
            result["engine"] = "recurdyn"
            runs.append(result)
            time, position, force, expected_position, expected_force = trace
            columns = ["time_s", *(f"q{i}_m" for i in range(position.shape[1])), "force_N",
                       *(f"ref_q{i}_m" for i in range(position.shape[1])), "ref_force_N"]
            np.savetxt(args.output/f"{label}.csv", np.column_stack((time, position, force, expected_position, expected_force)),
                       delimiter=",", comments="", header=",".join(columns), fmt="%.17g")
            print("recurdyn", case, step, result["position_rms_mm"], flush=True)
    report = {"engine": "recurdyn", "version": app.ProcessNetVersion, "radius_m": AES.radius,
              "centered_pair": AES.centered_pair, "repeats": args.repeats, "runs": runs}
    (args.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
