"""Build the independent, shared-shaft RecurDyn solid-contact reducer (Windows ProcessNet)."""

import argparse
import json
import os
from pathlib import Path
import sys
from threading import Event, Thread
from time import perf_counter

import numpy as np


def extract_plot(app, rd, plot_file, output):
    """Read cumulative joint angles and native contact channels from the saved analysis."""
    plot = rd.IPlotDocument(app.NewPlotDocument("ReducerContactChannels"))
    try:
        index = plot.ImportPlotDataFile(str(plot_file.resolve()))

        def columns(paths):
            values = np.column_stack([plot.GetPlotDataFromIndex(index, p) for p in paths])
            if not values.size or not np.isfinite(values).all():
                raise ValueError("Incomplete RecurDyn plot channel")
            return values

        angles = columns(f"Joints/Axis{i}/Pos1_Relative" for i in range(3))
        references = columns(f"Joints/ReferenceAxis{i}/Pos1_Relative" for i in range(3))
        velocity = columns(f"Bodies/Shaft{i}/Vel_RZ" for i in range(3))
        acceleration = columns(f"Bodies/Shaft{i}/Acc_RZ" for i in range(3))
        axial_torque = columns(f"Contact/Solid Contact/Mesh{i}/TZ_SolidContact" for i in range(2))
        support = columns(f"Force/Rotational Spring Damper/Bearing{i}/TZ_RSDA" for i in range(3))
        applied = columns(f"Force/Rotational Single(Axial) Force/Load{i}/TZ_Rotat_Axial" for i in (0, 2))
        resultant = columns(f"Contact/Solid Contact/Mesh{i}/FM_SolidContact" for i in range(2))
        friction = columns(
            f"Contact/Solid Contact/Mesh{i}/ContactPoints/ContactPoint1/FrictionForce1" for i in range(2)
        )
        np.savez_compressed(
            output,
            angle_rad=np.deg2rad(angles),
            reference_rad=np.deg2rad(references),
            global_speed_rad_s=velocity,
            global_accel_rad_s2=acceleration,
            contact_raw_TZ_Nm=axial_torque * 1e-3,
            support_raw_TZ_Nm=support * 1e-3,
            load_raw_TZ_Nm=applied * 1e-3,
            resultant_N=resultant,
            point1_friction_N=friction,
        )
    finally:
        app.ClosePlotDocument(plot)


def stop_at_deadline(doc, finished, remaining):
    import pythoncom
    import recurdyn as rd

    stream = pythoncom.CoMarshalInterThreadInterfaceInStream(pythoncom.IID_IDispatch, doc._oleobj_)

    def watch():
        pythoncom.CoInitialize()
        try:
            import win32com.client as win

            dispatch = pythoncom.CoGetInterfaceAndReleaseStream(stream, pythoncom.IID_IDispatch)
            owned = rd.IModelDocument(win.Dispatch(dispatch))
            if not finished.wait(max(0.0, remaining)):
                owned.AnalysisStop()
        finally:
            pythoncom.CoUninitialize()

    Thread(target=watch, daemon=True).start()


def placed_step(part, row, z_offset, output):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
    from literature_validation import write_step
    from nhrep_gear import read_occt_solid

    transform = gp_Trsf()
    transform.SetRotation(gp_Ax1(gp_Pnt(), gp_Dir(*row[1:4])), row[7] + row[12])
    transform.SetTranslationPart(gp_Vec(0, 0, z_offset))
    write_step(BRepBuilderAPI_Transform(read_occt_solid(part), transform, True).Shape(), output)


def connect():
    import win32com.client as win

    sys.path.insert(0, str(Path(os.environ["PATH_RECURDYN"]) / "Python/Lib"))
    import recurdyn as rd

    try:
        wrapper = win.GetActiveObject("RecurDyn.RDApplication")
    except Exception:
        wrapper = win.Dispatch("RecurDyn.RDApplication")
    return rd.IApplication(wrapper.RecurDynApplication), rd


def simulate(work, duration, h, output, patch_scale=None, contact_point_output=False):
    app, rd = connect()
    setup_start = perf_counter()
    with np.load(work / "assembly.npz") as d:
        rows, coords, mapping, pairs = (d[k] for k in ("bodies", "coord", "mapping", "pairs"))
    parts = json.loads((work / "models.json").read_text())["parts"]
    suffix = "" if patch_scale is None else f"-patch{patch_scale:g}"
    path = (work / f"rd-reducer-{duration:g}s-{h*1e6:g}us{suffix}.rdyn").resolve()
    doc = rd.IModelDocument(app.NewModelDocumentWithUnitSystem("ReducerContact", rd.UnitSystem.MMKS))
    try:
        model = rd.ISubSystem(doc.Model)
        ground = rd.IBody(model.Ground)
        doc.ModelProperty.UseGravity = False
        solids, shafts, contacts = {}, [], []
        for j in range(len(coords)):
            indices = np.flatnonzero(mapping == j)
            first = rows[indices[0]]
            sign = first[3]
            origin = first[4:7]
            body = rd.IBody(model.CreateBodyGeneral(f"Shaft{j}"))
            for i in indices:
                source = Path("results/gears/models") / parts[i]["step"]
                transformed = work / f"rd-{source.name}"
                placed_step(source, rows[i], rows[i, 6] - first[6], transformed)
                before = body.GeometrySolidCollection.Count
                body.FileImport(str(transformed.resolve()))
                if body.GeometrySolidCollection.Count != before + 1:
                    raise ValueError("STEP import must add one solid")
                solid = rd.IGeometrySolid(body.GeometrySolidCollection.Item(before))
                solid.Name = f"Part{i}"
                solids[i] = solid
            body.RefFrame.SetOrigin(*map(float, origin))
            body.MaterialInput = rd.MaterialInput.UserInput
            body.Mass.Value = 1.0
            for moment in (body.Ixx, body.Iyy, body.Izz):
                moment.Value = float(coords[j, 0]) * 1e6
            body.UseInitialRotationalVelocityZ = True
            body.InitialRotationalVelocityZ.Value = float(sign * first[8])
            frame = doc.CreateReferenceFrame()
            frame.SetOrigin(*map(float, origin))
            if sign < 0:
                frame.SetEulerAngleDegree(rd.EulerAngle.EulerAngle_ZXZ, 0, 180, 0)
            joint = rd.IJointRevolute(model.CreateJointRevolute(f"Axis{j}", ground, body, frame))
            # Ground-referenced servo rotor supplies the nominal motion; the
            # torque acts on the small relative angle and velocity of the shaft.
            rotor = rd.IBody(model.CreateBodyGeneral(f"Nominal{j}"))
            rotor.RefFrame.SetOrigin(*map(float, origin))
            rotor.MaterialInput = rd.MaterialInput.UserInput
            rotor.Mass.Value = 1
            for moment in (rotor.Ixx, rotor.Iyy, rotor.Izz):
                moment.Value = 1.0
            drive = rd.IJointRevolute(model.CreateJointRevolute(f"ReferenceAxis{j}", ground, rotor, frame))
            drive.UseMotion = True
            drive.Motion.Condition = rd.MotionCondition.MotionCondition_Displacement
            drive.Motion.Expression = model.CreateExpression(f"NominalAngle{j}", f"{first[8]:.17g}*TIME")
            spring = rd.IForceSpringRotational(
                model.CreateForceRotationalSpringWithBody(f"Bearing{j}", rotor, body, frame)
            )
            spring.Stiffness.Coefficient.Value = float(coords[j, 1]) * 1000
            spring.Damping.Coefficient.Value = float(coords[j, 2]) * 1000
            spring.FreeAngle.Value = 0.0
            expr = (
                None
                if j == 1
                else (
                    "37000*(1+.1*SIN(144.51326206513048*TIME))"
                    if j == 0
                    else "-148000*(1+.26/(1+EXP(-2*(TIME-.10)/.018))-.34/(1+EXP(-2*(TIME-.24)/.028))+.06*SIN(81.68140899333463*TIME))"
                )
            )
            if expr:
                torque = rd.IForceAxialRotational(
                    model.CreateForceRotationalAxialWithJoint(f"Load{j}", joint)
                )
                torque.Expression = model.CreateExpression(f"AppliedTorque{j}", expr)
            shafts.append(body)
        for i, (a, b) in enumerate(pairs):
            contact = rd.IContactSolidContact(
                model.CreateContactSolidContact(f"Mesh{i}", solids[a], solids[b])
            )
            prop = contact.ContactProperty
            prop.StiffnessCoefficient.Value = 1.6e5
            prop.UseStiffnessExponent = True
            prop.StiffnessExponent.Value = 1
            prop.DampingCoefficient.Value = 0
            if contact_point_output:
                contact.OutputMarker=ground.MarkerCollection.Item(0)
                contact.UseContactOutputFile=True
            if patch_scale is not None:
                for patch in (contact.BasePatchOption, contact.ActionPatchOption):
                    factor = patch.PlaneToleranceFactor.Value
                    if factor <= 0:
                        raise ValueError("Missing native plane tolerance factor")
                    patch.UsePlaneToleranceFactor = True
                    patch.PlaneToleranceFactor.Value = factor * patch_scale
            contacts.append(contact)
        dyn = doc.ModelProperty.DynamicAnalysisProperty
        dyn.SimulationTime.Value = duration
        dyn.SimulationStep.Value = round(duration / h)
        dyn.MaxTimeStep.Value = h
        dyn.MatchSolvingStepSize = True
        if not doc.Validate():
            raise ValueError("RecurDyn rejected the assembly")
        doc.FileSave(str(path), True)
        print("RecurDyn model validated", flush=True)
        start = perf_counter()
        finished = Event()
        stop_at_deadline(doc, finished, 1800 - (start - setup_start))
        try:
            doc.Analysis(rd.AnalysisMode.Dynamic)
        finally:
            finished.set()
        wall = perf_counter() - start
        t = np.asarray(doc.GetPlotDataTime(), float)
        if len(t) != round(duration / h) + 1 or not np.allclose(t, np.arange(len(t)) * h, rtol=0, atol=1e-10):
            raise ValueError("Incomplete RecurDyn trajectory")
        speeds = np.column_stack([doc.GetPlotData(b, "Vel_RZ") for b in shafts])
        force = np.column_stack([doc.GetPlotData(c, "NormalForce1") for c in contacts])
        if not np.isfinite(speeds).all() or not np.isfinite(force).all():
            raise ValueError("Nonfinite RecurDyn result")
        np.savez_compressed(work / f"{path.stem}.npz", t=t, speeds=speeds, normal=force)
        output.write_text(
            json.dumps(
                dict(
                    version=app.ProcessNetVersion,
                    steps=len(t) - 1,
                    h=h,
                    physical_s=t[-1],
                    setup_s=start - setup_start,
                    online_s=wall,
                    speed_initial=speeds[0].tolist(),
                    peak_force=abs(force).max(axis=0).tolist(),
                    patch_plane_tolerance_scale=patch_scale,
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        app.CloseModelDocument(doc)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--step-us", type=float, default=125.0)
    p.add_argument(
        "--patch-scale",
        type=float,
        help="Multiply the native contact patch plane tolerance factor on all four surfaces",
    )
    p.add_argument(
        "--plot", type=Path, help="Extract direct joint angle and contact channels from a completed .rplt"
    )
    p.add_argument("--contact-point-output",action="store_true",
                   help="Write native solid-contact points in the ground frame")
    args = p.parse_args()
    if args.plot:
        extract_plot(*connect(), args.plot, args.output)
    else:
        if args.patch_scale is not None and not (0 < args.patch_scale <= 1):
            p.error("Patch scale must be in (0, 1]")
        simulate(args.work, args.duration, args.step_us * 1e-6, args.output, args.patch_scale,args.contact_point_output)
