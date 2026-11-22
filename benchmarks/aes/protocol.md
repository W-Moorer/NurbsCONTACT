# AES numerical study protocol

The study compares end-to-end rigid contact simulation at matched trajectory and force accuracy. Continuous clearance on the nominal CAD is a separate requirement. Keep the numerical records, configurations, and proof status for every run. Select the paper title and write the manuscript under `docs/AES` after the numerical study. Use the single-column Elsevier CAS layout and build conventions of `GPU/gpu-contact/writing/CoFlexSim-AES` in the ICRL repository.

## Cases

| Role | Model | Reference and primary outcomes |
| --- | --- | --- |
| Published dynamics reference | Junca and Rousselet's unilateral oscillator, Sec. 2 | Contact duration, detached duration, and onset and release timing over 50 periods. Compare the exact scalar-force HHT runs to the piecewise harmonic solution. |
| Published kinematics reference | Dhande and Rajaram's constant-breadth cam, Example 1 | Compare the six-arc STEP model and translating flat follower to the published displacement, velocity, and acceleration. |
| Simple industrial 1 | Single-stage spur gear pair | Use the existing STEP assets and matched projected contact law. Compare transmission error and total mesh force over steady meshing and load changes. |
| Simple industrial 2 | Cam and single follower | Use the existing eccentric cam and follower geometry. Compare follower displacement and normal force over several complete rotations. |
| Complex industrial | Two-stage spur–helical reducer | Use the existing assembly and loading schedule. Compare output transmission error, the critical stage force, completion, and cost. |
| Appendix cross-engine checks | Sphere against a plane and translating sphere pair | Compare the native engines to a shared compliant force law and closed-form response. |

The sphere cases use the existing half-sine response as the reference and include approach, loading, unloading and free motion after release. Begin time-refinement ladders at 500, 250 and 125 µs for the sphere cases, 1, 0.5 and 0.25 ms for the industrial cam, and 20, 10 and 5 µs for the gear pairs. These are starting values. Resolve the most restrictive force or trajectory requirement on each model before selecting the production step. A gear run covers at least 60 input turns and, for the reducer, the full loaded schedule with its transients. The industrial cam run covers at least 12 driven revolutions.

## Published reference reproductions

With the `nc_hht` Release target built, its OCCT DLLs on `PATH`, and NumPy and OCP available to Python, run from the repository root:

```powershell
python benchmarks/aes/literature_validation.py build/occt/Release/nc_hht.exe build/aes-literature
```

The script records the full trajectories, STEP files, and `summary.json`. The [measured summary](../../results/aes/literature-validation.json) lists the three oscillator steps and the two cam checks. Section 2 of [Junca and Rousselet](https://doi.org/10.1007/978-3-540-89105-5_15) gives a parameterized model. The reproduced setting has `omega_0 = a_0 = 1` and `epsilon = 0.1`. The cam follows the dimensions and 1 rad/s rotation of [Dhande and Rajaram's Example 1](https://doi.org/10.1115/1.3258582). Dense CAD extrema provide the positions for finite differences. The 1-degree grid independently exercises the native bounded distance implementation on the imported STEP surfaces. The arc profile uses the original paper's length units.

## Comparisons

- Compare the existing CAD and triangle backends under the same contact law, time-discretization family, load and acceptance requirements. Refine mesh resolution as well as time step. Separate setup from online cost and expose both the first-run and repeated-run costs.
- Compare the direct, tight bounded, and adaptive cached force paths at the same time step. Record refinements, reused enclosures, response errors, and cost. The adaptive cached mode combines force adaptation and reuse.
- Use MuJoCo and Drake on an equivalent model where their contact law can be matched. RecurDyn uses the same dimensions, masses, generalized coordinates, initial conditions and load history where its model supports them. An imposed common projected force law and a native engine contact law are different protocols and have different result labels. Verify actual contact laws against the engines' traces before ranking costs. Run same-host comparisons and document solver and platform settings.
- Compare against analytic references where available. For industrial response, use a demonstrably refined reference and retain residual, time-step and mesh-sensitivity records. Compare competing methods at equivalent response tolerances rather than at a single step or tessellation setting.

Score the industrial trajectories over the full run on a common output grid of 20 µs for the gears and 1 ms for the cam. Use RMS follower displacement or stage transmission error and the RMS force difference divided by the RMS reference force. Report the sampled peak-force difference separately. Measure three joint tolerance levels (0.5 µm and 2%, 1 µm and 5%, 2 µm and 10%). Both reducer stages must satisfy the specified level. The cam and spur references use the finest native time step, and the reducer reference also uses the largest axial quadrature. Count a setting with a missing or incomplete run as ineligible for the accuracy comparison.

## Manuscript assets after verification

Allocate the three principle figures to residual-driven precision, lower and upper event-time bounds, and the validity region of reused exclusions. Add one workflow diagram and one algorithm listing. Render the STEP-derived model overviews and contact insets with Blender through Python, and use Python for the response plots and figure checks. Choose the fewest response curves that establish each case's result.

## Prototype evidence

The Windows Release pilot uses OCCT 7.9.3. The [sphere accuracy records](../../results/aes/sphere-accuracy-windows.csv) cover the two half-sine cases at the specified three time steps. `bounded-exact` uses the interval force oracle at its highest available precision, `bounded` lets HHT request precision, and `bounded-cache` also reuses a previously proved inactive body pair under the point-travel bound. The interval-force implementation covers the analytic sphere and cylinder/plane fixture types, with an independently checked finite face. The gear implementation encloses the intersections of prescribed rays with the nonrational cubic CAD sections. It checks endpoint signs and monotonicity in the projected root interval. Cached points propose enclosures at later rays and each proposed interval is checked against the current force budget. For the two sphere fixtures, the closed-form fixed-force path costs less than the bounded oracle.

```powershell
# With OCCT DLLs on PATH and the Release targets built
python benchmarks/aes/sphere_accuracy.py build/occt/Release/nc_hht.exe build/aes-benchmark/sphere-accuracy
python benchmarks/aes/industrial_baselines.py build/occt/Release/nc_gears.exe build/occt/Release/nc_hht.exe results/gears/models build/aes-benchmark/industrial
```

## Analytic external baselines

[MuJoCo 4.0.1](../../results/aes/mujoco-analytic-linux.json), [Drake 1.58.0](../../results/aes/drake-analytic-linux.json), and the [native Linux build](../../results/aes/sphere-accuracy-linux.csv) ran serially in the same Linux environment. They share the 55 mm sphere radius, 0.45 m/s relative entry speed, 1 kg per sphere, 12 mm contact layer, and 5,000 N/m stiffness. The sphere-pair center of mass starts at rest. The engines' recorded force agrees with the stated contact law along their own trajectories. Compare their reported errors against the common half-sine solution. The Windows Release pilot is a separate timing batch.

Use `--aes-fixtures` with the [engine runner](../engines/README.md#reproduce-on-a-common-linux-host) to select this matched configuration. The [RecurDyn 2023 report](../../results/aes/recurdyn-analytic-windows.json) contains three timing repeats per configuration. Use the [adapter](../engines/recurdyn.py) in an active ProcessNet session with `--api-dir` pointing to the installed `Bin/Python/Lib`. It uses primitive vendor contacts and extracts `NormalForce1` directly, with a separate force-law check. The maximum observed force-law discrepancy was below \(10^{-9}\) N. Its wall time includes the vendor analysis call and its result handling, and its solver CPU time is read separately when reported by the solver.

## Industrial baselines

The eccentric cam compares the analytic cylinder–plane support calculation, OCCT shape-distance extrema, a general triangle BVH, and a structure-aware polygon support query under the same HHT integrator, normal force law, and step guard. Both discrete queries use the same 2048-sided geometry in the fixed-step comparison. Polygon support selects the extremal side vertex and uses a general mesh search if the projection misses the finite follower. OCCT `BRepExtrema_DistShapeShape` caches the reference shape, moves the original STEP geometry, and uses a deflection setting of 10⁻⁷ mm. The query supplies a numerical distance and a closest-point pair. The [cam comparison records](../../results/aes/cam-baselines.json) include one complete timing for each general search and three repeats for each support calculation.

Run the common 20 s, 125 µs cam comparison with the refined reference from the initial industrial run. To regenerate that reference, use the first two commands below:

```powershell
$hht = 'build/occt/Release/nc_hht.exe'
$reference = 'build/aes-benchmark/industrial-cam2'
& $hht models "$reference/cam-models"
& $hht run cam brep 24 0.0000625 20 "$reference/cam-models" "$reference/cam-native-24-62us" 1
python benchmarks/aes/cam_baselines.py $hht "$reference/cam-native-24-62us.csv" build/aes-cam-baselines
```

The gear comparisons include a stage-specific polygonal alternative. It discretizes each of the same CAD sections into a piecewise-linear curve and intersects a line-of-action ray by bracketing its transverse zero, without the overhead of traversing a 3D BVH. Its segment count is the number of intervals per axial section. `--series tight-mesh` refines the discrete geometry of both baselines, `--series section-mesh` runs the gear polygonal alternative, and `--series time-check` adds a 2.5 µs native run. `--series bounded` runs the nonlinear force-decision ablation. The case selector can limit a study to any mechanism. `industrial_accuracy.py` merges the resulting folders and audits all completed traces against the converged native references.

The [Drake cam adapter](../engines/drake_cam.py) uses a revolute cylinder and a prismatic half-space, with 12 mm added to the cylinder's contact radius. It preserves the mass, rotational inertia, driven spring-damper relation, follower preload and frictionless force law. It verifies the recorded contact force against the cylinder/plane law along each of Drake's computed trajectories. `engine_cam_accuracy.py` compares this output with the equivalent native cam runs from the same Linux environment.

## Manuscript timing batch

[Paper measurements](../../results/aes/paper-measurements.json) contain the individual times and accuracy audit of the serial batch. [The full industrial ladder](../../results/aes/industrial-windows.csv) retains the original configuration sweeps and their timings. The serial batch controls for other benchmark processes on the workstation. Its medians supply the relaxed cam comparison and the selected spur and reducer costs in the manuscript.

```powershell
$work = 'build/aes-benchmark'
$timings = "$work/final-timing"
$hht = 'build/occt/Release/nc_hht.exe'
$gears = 'build/occt/Release/nc_gears.exe'
$models = 'results/gears/models'
$cam = "$work/industrial-cam2/cam-models"
& $hht run cam brep 24 0.000125 20 $cam "$timings/cam-native" 3
& $hht run cam mesh-support 8192 0.000125 20 $cam "$timings/cam-polygon" 3
& $hht run cam mesh-support 512 0.00025 20 $cam "$timings/cam-polygon-relaxed" 3
& $gears run spur $models 0 5e-6 2 "$timings/spur-native" 3 31 4
& $gears run spur $models -1024 5e-6 2 "$timings/spur-polyline" 3 31 4
& $gears run reducer $models 0 1e-5 3 "$timings/reducer-native" 2 31 2
& $gears run reducer $models -256 1e-5 3 "$timings/reducer-polyline" 2 31 2
& $gears run reducer $models 0 2e-5 0.2 "$work/reducer-fixed-ablation" 1 31 1 fixed
& $gears run reducer $models 0 2e-5 0.2 "$work/reducer-bounded-exact-ablation" 1 31 1 bounded-exact
& $gears run reducer $models 0 2e-5 0.2 "$work/reducer-bounded-cache-guardfix" 1 31 1 bounded-cache
python benchmarks/aes/paper_measurements.py $work results/aes/paper-measurements.json
```

The bounded diagnostic covers the first load rise. It includes the rule that a candidate interval must be tightened if the force budget allows it but its gap interval overlaps the gear clearance floor.
