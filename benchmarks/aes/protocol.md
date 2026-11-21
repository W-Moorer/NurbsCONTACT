# AES numerical study protocol

The study compares end-to-end rigid contact simulation at matched trajectory and force accuracy. Continuous clearance on the nominal CAD is a separate requirement. Keep the numerical records, configurations, and proof status for every run. Select the paper title and write the manuscript under `docs/AES` after the numerical study. Use the single-column Elsevier CAS layout and build conventions of `GPU/gpu-contact/writing/CoFlexSim-AES` in the ICRL repository.

## Cases

| Role | Model | Reference and primary outcomes |
| --- | --- | --- |
| Analytic 1 | Frictionless sphere against a plane | Piecewise closed-form motion and compliant normal force. Compare peak force, force duration and sphere displacement. |
| Analytic 2 | Translating sphere pair | Closed-form relative response at reduced mass and conserved center-of-mass motion. Compare force and both translations. |
| Analytic 3 | Disk passing a circular trim at a diagonal offset | Analytic minimum gap and first threshold entry. Compare both ends of the time bracket, proof status and gap response. Include a near miss and equality. |
| Simple industrial 1 | Single-stage spur gear pair | Use the existing STEP assets and matched projected contact law. Compare transmission error and total mesh force over steady meshing and load changes. |
| Simple industrial 2 | Cam and single follower | Use the existing eccentric cam and follower geometry. Compare follower displacement and normal force over several complete rotations. |
| Complex industrial | Two-stage spur–helical reducer | Use the existing assembly and loading schedule. Compare output transmission error, the critical stage force, completion, and cost. |

The sphere cases use the existing half-sine response as the reference and include approach, loading, unloading and free motion after release. Begin time-refinement ladders at 500, 250 and 125 µs for the sphere cases, 1, 0.5 and 0.25 ms for the cam, and 20, 10 and 5 µs for the gear pairs. These are starting values. Resolve the most restrictive force or trajectory requirement on each model before selecting the production step. A gear run covers at least 60 input turns and, for the reducer, the full loaded schedule with its transients. A cam run covers at least 12 driven revolutions. Event-query time resolution is varied independently of the time-integration step.

## Comparisons

- Compare the existing CAD and triangle backends under the same contact law, time-discretization family, load and acceptance requirements. Refine mesh resolution as well as time step. Separate setup from online cost and expose both the first-run and repeated-run costs.
- Isolate the residual-driven query, guided event witness and exclusion reuse. Run each with independently enabled support for the remaining strategies, then run the combined method. Record force refinements, geometry work and rejected or subdivided steps. Count valid CAD brackets, clear intervals, safe prefixes and unresolved results.
- Use MuJoCo and Drake on an equivalent model where their contact law can be matched. RecurDyn uses the same dimensions, masses, generalized coordinates, initial conditions and load history where its model supports them. An imposed common projected force law and a native engine contact law are different protocols and have different result labels. Verify actual contact laws against the engines' traces before ranking costs. Run same-host comparisons and document solver and platform settings.
- Compare against analytic references where available. For industrial response, use a demonstrably refined reference and retain residual, time-step and mesh-sensitivity records. Compare competing methods at equivalent response tolerances rather than at a single step or tessellation setting.

Score the industrial trajectories over the full run on a common output grid of 20 µs for the gears and 1 ms for the cam. Use RMS follower displacement or stage transmission error and the RMS force difference divided by the RMS reference force. Report the sampled peak-force difference separately. Measure three joint tolerance levels (0.5 µm and 2%, 1 µm and 5%, 2 µm and 10%). Both reducer stages must satisfy the specified level. The cam and spur references use the finest native time step, and the reducer reference also uses the largest axial quadrature. Count a setting with a missing or incomplete run as ineligible for the accuracy comparison.

## Manuscript assets after verification

Allocate the three principle figures to residual-driven precision, lower and upper event-time bounds, and the validity region of reused exclusions. Add one workflow diagram and one algorithm listing. Render the STEP-derived model overviews and contact insets with Blender through Python, and use Python for the response plots and figure checks. Choose the fewest response curves that establish each case's result.

## Prototype evidence

The Windows Release pilot uses OCCT 7.9.3. The [sphere accuracy records](../../results/aes/sphere-accuracy-windows.csv) cover the two half-sine cases at the specified three time steps. `bounded-exact` uses the same interval force oracle with a zero requested tolerance, `bounded` lets HHT request precision, and `bounded-cache` also reuses a previously proved inactive body pair under the point-travel bound. The current interval-force implementation covers the analytic sphere and cylinder/plane fixture types, with an independently checked finite face. It has yet to be integrated with the industrial gear rays. The analytic fixed-force path is faster on the sphere pilot, despite close output agreement. Timing rankings for these fixture results do not yet include the external engines.

The [event accuracy records](../../results/aes/event-accuracy.csv) compare the opt-in planar witness projection against ordinary sampling for the 34 threshold trajectories, each repeated three times. Guided projection raises the number of cases with a target-width bracket or complete safe interval from 29 to 32. All three prior diagonal entries with missing witnesses have valid event brackets. Exact equality remains a safe-prefix result. The independent auditor checks event containment, the time width, and the distance between the two outward point enclosures. The proposal currently targets planar trimmed patches wherever the continuous search opens a promising interval.

```powershell
# With OCCT DLLs on PATH and the Release targets built
python benchmarks/aes/sphere_accuracy.py build/occt/Release/nc_hht.exe build/aes-benchmark/sphere-accuracy
python benchmarks/aes/event_accuracy.py build/occt/Release/nc_step.exe build/threshold-scan build/aes-benchmark/event-accuracy.csv
```

The event fixtures are generated by the `audit` and `counterbore` modes of `nc_clearance_followup`, as shown in the [threshold study instructions](../assemblies/threshold-scan/README.md#reproduce).

## Analytic external baselines

[MuJoCo 4.0.1](../../results/aes/mujoco-analytic-linux.json), [Drake 1.58.0](../../results/aes/drake-analytic-linux.json), and the [native Linux build](../../results/aes/sphere-accuracy-linux.csv) ran serially in the same Linux environment. They share the 55 mm sphere radius, 0.45 m/s relative entry speed, 1 kg per sphere, 12 mm contact layer, and 5,000 N/m stiffness. The sphere-pair center of mass starts at rest. The engines' recorded force agrees with the stated contact law along their own trajectories. Compare their reported errors against the common half-sine solution. The Windows Release pilot is a separate timing batch.

Use `--aes-fixtures` with the [engine runner](../engines/README.md#reproduce-on-a-common-linux-host) to select this matched configuration. The [RecurDyn 2023 report](../../results/aes/recurdyn-analytic-windows.json) contains three timing repeats per configuration. Use the [adapter](../engines/recurdyn.py) in an active ProcessNet session with `--api-dir` pointing to the installed `Bin/Python/Lib`. It uses primitive vendor contacts and extracts `NormalForce1` directly, with a separate force-law check. The maximum observed force-law discrepancy was below \(10^{-9}\) N. Its wall time includes the vendor analysis call and its result handling, and its solver CPU time is read separately when reported by the solver.
