# Shared STEP contact backend and experimental validation

`nc_contact` runs the same executable, contact law, virtual-work projection, and implicit HHT integrator for every scene. The deck defines shapes, their poses and generalized coordinates, contact probes, and material constants. The native and triangle backends share these inputs, the step and the recorded observables. A positive tessellation deflection selects the triangle BVH.

## Contact model

Each probe intersects two facing, finite STEP surfaces along one line. It uses their signed separation as the gap. The line can be fixed in world space, rotate with body A, or follow the two declared anchors. The ray hits provide normal vectors. Differentiation of those intersections gives the gap Jacobian, including the motion of the probe direction and origin. Virtual work maps the unilateral nonlinear spring and dashpot, and regularized Coulomb friction, into generalized forces.

The native adapter compiles planar faces with linear trimming wires and uses the full set of trim loops, including holes. Curved boundaries and unsupported planar parameterizations use OCCT face intersection. The mesh adapter calls the project triangle BVH with the same rays. The signed-depth limit rejects an inadmissible Newton trial. This solver validates directional contact on declared pairs and probes. The certified whole-step clearance API is described in the [repository distance and sweep documentation](../../README.md).

## Cases and evidence

| Test | State and measured quantity | Calibration and evaluation |
|---|---|---|
| GAUGE rebound | 60 mm rubber ball, vertical reduced coordinate. Compare ball center height at original 30 Hz times. Initial height and speed follow a quadratic fit to five descending, pre-impact frames with fixed gravity. | Trial IDs 1–5 determine one damping ratio. IDs 6–22 are held out, with a fixed 1 s scoring horizon after the initial state. |
| GAUGE incline | 50 mm wood, plastic and metal cubes with three translational coordinates. The first three board poses set its fixed frame. Compare center position along the board while the cube remains fully supported. | IDs 1–5 per material determine one kinetic friction coefficient from the supported motion. The remaining records use those frozen values. Metal trial 17 is absent in the pinned publication. The initial projected speed uses the two adjacent early frames, which are excluded from scoring. |
| Steel ball striking a beam | 70 mm steel sphere and one vertical beam mode. The measured observable is the duration of the first electrical contact pulse at midspan. | All 27 measurements are predictions with stiffness from reported elastic constants and curvature. The beam modal frequency comes from its measured displacement spectrum and its modal mass is half the beam mass. The duration points come from Qi and Yin's Fig. 17. |
| Driven cantilever with two stops | Simultaneous 1024 Hz base and tip accelerations, with intermittent contact against two PLA and fabric limiters. | Two random excitation records determine one set of mode and stop parameters. The high-amplitude random test and the full stepped-sine test are held out. |
| Slider-crank with pin–bore clearance | Prescribed crank with dynamic rod rotation and slider translation. Compare slider travel and acceleration on the published encoder angle. | Seven clearance–speed combinations from the vector paths of the publisher PDF. Use published material and restitution parameters without fitting measured curves. |

The same nominal STEP parts are used by both backends. These STEP reconstructions use the dimensions of the published geometry, including GAUGE's released assets, and the reported steel specimens. GAUGE scoring uses neither a fitted time shift nor a fitted trajectory transform. The rebound score is for the vertical component. The slide score is for supported translation. The beam score covers the first pulse in every reported speed condition. The coarser camera data support rebound heights and trajectory timing at the recorded cadence. Electrical pulse duration provides the independent high-bandwidth observable.

[GAUGE data and license](https://huggingface.co/datasets/InternRobotics/GAUGE-Dataset) · [GAUGE paper](https://arxiv.org/abs/2608.05948) · [Qi and Yin's experiment](https://doi.org/10.1177/1687814016668089)

## Results and source files

Across 61 held-out optical trajectories, the B-rep median RMSE is 6.90 mm for rebound height and 3.16 to 4.05 mm for supported displacement. All 17 held-out rebound apices are within one 30 Hz frame of their observed times. The 27 independent electrical durations give a 6.56 µs RMSE against measured durations of 174 to 199 µs. The mean signed bias is +6.50 µs. Halving the step changes the representative B-rep rebound by 0.10 mm RMS and the steel-beam duration cohort by 0.005 µs RMS.

Seven serial timing repeats give triangle-BVH to B-rep integration time ratios of 0.93 for rebound, 1.21 for supported sliding, and 0.47 for steel-beam impact. Preparing planar trims accelerates the sliding deck by 12.67× against direct native OCCT intersections. The compared geometry preparation preserves the sampled trajectories.

- [GAUGE error distributions and per-trial results](summary.json) · [Original URLs, data revision and checksums](sources.json) · [Paired sampled curves](curves)
- [Electrical duration metrics](beam/summary.json) · [Paired durations](beam/comparison.csv) · [Digitization and source coordinates](beam/measured.csv)
- [Force, position and gap traces](runs) · [Scene decks](scenes) · [Performance repetitions](timing.json)
- [Measured-versus-simulated figure](figures/contact-validation.pdf) · [Final paper](../../docs/manuscript.pdf)
- [Multi-view animations](animations/index.json)
- [Cantilever report and reproduction](external-validation.md#driven-cantilever-with-two-stops) · [Flexible contact study](../../manuscript/flexible-contact.md)
- [Clearance results, models, evidence limits and reproduction](external-validation.md#clearance-slider-crank)

## Scene format

All quantities are SI, with STEP geometry imported in millimeters. Body and coordinate IDs are zero-based. A hinge of `-1` denotes a body without a hinge. Every translation entry gives a coordinate and its derivative vector in world space. The generalized mass includes the inertia associated with the chosen coordinate. The body mass supplies gravity through its center-of-mass Jacobian.

~~~text
nc-contact-v1
gravity gx gy gz number_of_coordinates number_of_bodies number_of_probes
coordinate mass q0 v0 spring_K spring_C rest_q applied_load
body "file.step" mass origin_xyz axis_xyz angle com_local_xyz hinge_index pivot_xyz ntranslations [coordinate_id direction_xyz]...
probe body_a body_b world|body-a|anchors|contained|contained-radial anchor_a_xyz anchor_b_xyz normal_xyz K exponent C mu slip_speed max_depth
~~~

`C` multiplies `depth^((exponent-1)/2)` and the gap-closing speed. The solver clips a tensile normal response to zero. The output includes the generalized states, body poses, signed probe gaps and predicted forces. A probe outside a trim is reported as unresolved at that location. Multiple probes can be specified for one pair.

For a contained probe, place A's anchor in the cavity and B's anchor inside the contained part. The ray leaves each anchor along the same centerline towards the near cavity wall. At concentric centers, `normal_xyz` in A's frame defines the fallback direction. A missed trimmed surface rejects a contained probe. `contained-radial` uses the centerline for the force differential in both backends and applies to circular sections. Each backend still computes the gap from its own geometry.

The `nc-contact-v2` header adds a drive block after all probes. It accepts a prescribed translation orbit or a tabulated translation. An orbit translates a body by rotating its arm around the given axis at constant angular velocity. The orbit acceleration supplies the base-inertia force through the body's mass and center-of-mass Jacobian. A tabulated drive linearly interpolates consecutive world-space offsets and requires a prescribed body. A v2 scene with zero dynamic coordinates evaluates the same contact law on prescribed poses at the requested sample times.

### Whole-body nearest-contact pairs

A v2 deck can declare a pair of STEP bodies in place of a probe. The pair searches the full shapes at each trial pose. It applies one compliant point force per declared pair using the direction of the closest points and `gap = distance - activation`.

```text
closest body_a body_b activation_m K exponent C mu slip_speed max_depth_m
```

Select `pair-mesh` with a positive deflection in meters, `pair-occt` for numerical STEP extrema, or `pair-brep` for interval B-rep distance. Native B-rep accepts a feasible witness within the activation distance if the distance width meets the configured tolerance. A verified lower bound can also exclude the pair. An undecided trial is rejected and can trigger HHT step subdivision. The prototype models exterior contact across a separation layer. A zero-length witness has no usable point-pair normal and is unresolved. Use directional probes for a prescribed signed-penetration or containment model.

Add `EVENT_TOL_S` after the native tolerance and node cap to check a step independently of its force backend. This check runs a B-rep sweep along the interpolation between the converged endpoint coordinates. It supports affine origin motion, constant orientation of A, and fixed-axis rotation of B with aligned axes or an unrotated A. An undecided sweep triggers subdivision. A new contact can use the witness time to position the split. Continued contact with a witness at the start can proceed. The report separates force queries, unresolved force queries, and the CCD work.

```powershell
build/occt/Release/nc_contact.exe SCENE DEFLECTION_M DT_S DURATION_S OUTPUT_STRIDE OUTPUT_PREFIX pair-mesh
build/occt/Release/nc_contact.exe SCENE 0 DT_S DURATION_S OUTPUT_STRIDE OUTPUT_PREFIX pair-brep 2e-6 30000
```

These modes use the fixed-evaluation HHT force callback. The native distance tolerance controls the scalar distance interval. Residual-driven force accuracy additionally needs bounds on the closest-point normal and moment arms.

[`pair_dynamics.py`](../../benchmarks/aes/pair_dynamics.py) generates a common eccentric-cam deck and two transmission decks from the existing STEP assets. The cam deck keeps the `nc_hht` benchmark's spring, preload and initial conditions. The transmission decks use one nearest point pair per gear mesh, with the original shaft inertias and a tabulated approximation to the driving and braking torques.

```powershell
$out = 'build/aes-pair-dynamics'
& build/occt/Release/nc_hht.exe models "$out/cam-models"
python benchmarks/aes/pair_dynamics.py "$out/cam-models" results/gears/models "$out/decks"
& build/occt/Release/nc_contact.exe "$out/decks/cam.scene" 2.5e-5 .000125 20 16 "$out/cam-mesh" pair-mesh
& build/occt/Release/nc_contact.exe "$out/decks/cam.scene" 0 .000125 20 16 "$out/cam-occt" pair-occt
```

[`pair_cam_accuracy.py`](../../benchmarks/aes/pair_cam_accuracy.py) scores the sampled follower displacement and normal force against the refined `nc_hht` cam trace. It checks the end time and accepted step count before calculating errors. The [measured report](../aes/pair-dynamics.json) records the two mesh resolutions, OCCT, the gear startup checks, and the native and step-CCD budget failures.

### CAD contact candidates by face pair

`CadContactIndex` prepares a hierarchy of bounds for the faces of a STEP shape. A static query finds nearby bounded face pairs and evaluates numerical OCCT extrema on their trimmed faces. The output retains each witness's one-based source face index, support type, separation, and parameters for face-interior witnesses. It flags opposing face-interior normals with a cosine test. The unflagged edge, vertex, and face witnesses remain in the report so a later contact law can account for boundary and extended contacts.

The `regions` command takes the relative pose of B in A's frame. Translation and cutoff are in meters, the rotation uses an axis and an angle in radians, and the witness coordinates are returned in A's frame.

```text
nc_contact regions A.step B.step CUTOFF_M TX TY TZ AX AY AZ ANGLE_RAD OUTPUT_PREFIX [SEEDS.txt]
```

Each row of an optional seed file stores `face_a face_b ua va ub vb`. The tracking query starts at these parameters, updates the surface extrema numerically, and rechecks both trims, the cutoff and the opposing normals. An unsuccessful seed is reported as unresolved. The whole-shape search finds new candidate faces. Tracking updates only the supplied interiors. Both entry points return numerical candidates for a future contact assembly. Initial interpenetration, enumeration of distinct extrema within a single face pair, and distributed force error bounds require additional geometry and contact modelling.

Use the existing spur trajectory to compare the whole-shape query and local tracking on consecutive double-tooth frames.

```powershell
python benchmarks/aes/cad_contact_regions.py build/occt/Release/nc_contact.exe results/gears/models build/aes-benchmark/industrial-pilot/spur-native-0nm-5us-31.csv build/aes-pair-dynamics/spur-contact-regions
```

The [measured report](../aes/cad-contact-regions.json) records the two saved spur frames and the initial eccentric-cam pose. The cold timings cover face-pair discovery. The tracked timing covers only the two supplied face pairs and excludes the separately reported STEP import and index preparation.

### Straight CAD contact line loads

`nc_contact line-load` reconstructs straight contact lines from the facing
interior witnesses and paired end witnesses of a CAD face pair. It intersects
the corresponding traces with the original face trims and evaluates the valid
subsegments on the STEP surfaces. Each two-point quadrature uses weights in
meters. For a sample at gap `d`, the linear frictionless law gives a force of
magnitude `k_line * weight * max(0, cutoff - d)`. The line stiffness
`k_line` is in N/m². Forces and moments are expressed in A's axes and the
moment on each body is reported about that body's own origin.

```text
nc_contact line-load A.step B.step CUTOFF_M STIFFNESS_N_PER_M2 TX TY TZ AX AY AZ ANGLE_RAD OUTPUT_PREFIX [SEEDS.txt]
```

Each row in the load CSV represents one face pair. The statistics report the
time for the full face search and the subsequent assembly separately. With
seeds, the command also reports loads after replacing the interior witnesses
with tracked ones while holding the boundary witnesses and the pose fixed.
Failed line reconstructions and seed updates are counted separately.

For a fresh build, generate the cam STEP files with `nc_hht models` and
the 5 µs spur trace with `nc_gears run spur` using the settings in the
[industrial timing protocol](../../benchmarks/aes/protocol.md#manuscript-timing-batch).
Pass the resulting trace path in the fourth argument below.

```powershell
python benchmarks/aes/cad_line_loads.py build/occt/Release/nc_contact.exe build/aes-benchmark/industrial-cam2/cam-models results/gears/models build/aes-benchmark/industrial-pilot/spur-native-0nm-5us-31.csv build/aes-pair-dynamics/cad-line-loads
```

The [force and moment verification](../aes/cad-line-loads.json) uses the
analytic initial cam pose and the two loaded face pairs on a saved spur pose.
The spur check prescribes the same line stiffness for both face pairs and
compares against a midpoint calculation from the CAD boundary witnesses.
It also reports the moment change from applying the same resultant at each
alternative interior witness. The spur trajectory supplies the poses. Its
recorded forces use the existing gear loading law.

This diagnostic covers separated surfaces with a valid straight line in the
contact band. Face pairs with only boundary witnesses stay in the candidate
report. Point, curved line, area and penetrating contact need corresponding
load representations. Time integration also needs a policy for updating
regions and handling unresolved geometry during an HHT step.

### Reusing CAD face exclusions along a trajectory

`CadContactCache` stores directional separation bounds for one pair of
prepared STEP shapes. A rejected face pair's numerical extrema supply a
direction. The stored gap comes from bounding the original trimmed faces
along that direction with OCCT's geometry boxes. At a subsequent pose the
query adjusts the gap by the projected motion of B's face-box center and
subtracts a rotation allowance. The allowance is the local box radius
multiplied by the change in the projection direction expressed in B's frame.
Only a bound that still exceeds the current cutoff skips an extrema query.
The face hierarchy runs at every pose. Active faces and potential new faces
go through the current-pose extrema and line assembly. The cache attempts
one projection per face pair. An absent or insufficient bound leaves the
pair to the numerical query. A cache is local to one query stream and
resets if either prepared shape changes.

```text
nc_contact regions-replay A.step B.step CUTOFF_M STIFFNESS_N_PER_M2 POSES.txt OUTPUT_PREFIX cold|reuse
```

The pose file holds `time_s tx ty tz ax ay az angle_rad` on each line.
Both replay modes prepare the shapes once. The benchmark checks identical
candidate distances and line loads for both modes on saved poses around
entry and exit. The gear trace selects the windows. The traces' gear
loading law and the CAD line assembly use separate contact definitions.
The optional cam replay also checks the force and moment against its
analytic cylinder–plane reference.

```powershell
$cam = 'build/aes-benchmark/industrial-cam2'
python benchmarks/aes/cad_region_replay.py build/occt/Release/nc_contact.exe results/gears/models build/aes-benchmark/industrial-pilot/spur-native-0nm-5us-31.csv build/aes-region-replay --cam-models "$cam/cam-models" --cam-trace "$cam/cam-native-24-125us.csv"
```

The [replay report](../aes/cad-region-replay.json) records each method's
extrema calls and times with preparation and load assembly measured
separately. The checks cover the sampled poses. Contact within an interval
between poses and updates during HHT iterations require separate validation.

### NH-Rep on the STEP spur pinion

The pilot uses [upstream NH-Rep](https://github.com/guohaoxiang/NH-Rep)
at commit `bf807e998e613ba17d03c1c8e160b7fe2262e734`. The upstream
`FeatureSample` program prepares the Boolean tree and 50,000 training
samples from the STEP-derived watertight mesh. The adapter retains the
original losses and shape network. It batches the per-patch index copies
and the Boolean reductions and checks the original and batched Boolean
values and gradients for equality before training. The batched evaluation
also supports CPU TorchScript export. The training loop requires CUDA.

Prepare the upstream checkout and build its `FeatureSample` target using
its pre-processing instructions. The Python adapter uses CadQuery,
Trimesh with Rtree, and the upstream training dependencies. With the
OCCT runtime available, generate the training mesh and the distance
reference from the repository root.

```powershell
$out = 'build/aes-nhrep'
$gear = 'results/gears/models/spur-1a.step'
python benchmarks/aes/nhrep_gear.py prepare $gear $out
python benchmarks/aes/nhrep_gear.py reference $gear $out --exe build/aes-occt/Release/nc_contact.exe --witnesses build/aes-pair-dynamics/spur-contact-regions-cold.csv
```

Run the upstream sampler on the normalized gear mesh. This example uses
its Linux executable with the output directory as the working directory.

```bash
../deps/nh-rep/code/pre_processing/Bin/FeatureSample -i spur.obj -o spur_50k.xyz -f spur.fea -k spur_50k_mask.txt -m 1 --ns 50000 --fs 0 -c 1 --csg 1 --convex 1 -r 0 --mp 6 --repairtree
```

```powershell
python benchmarks/aes/nhrep_gear.py train build/deps/nh-rep $out --epochs 15000 --batch 16384
python benchmarks/aes/nhrep_gear.py audit "$out/gear_trial_spur_50k_model_h.pt" $out
```

The [precision report](../aes/nhrep-spur.json) uses the signed distance
and well-defined distance gradients evaluated on the original STEP
solid through `nc_contact cad-point-distance`. The source-mesh distance
isolates the observed input tessellation error. The four additional
queries come from Gauss points on the two loaded contact lines of the
archived double-tooth frame. Their line-load comparison keeps the
integration points, weights and material fixed and replaces the gap
only. The reported gap budget follows from a 2% normal-load error at
that pose. The CPU timings measure field evaluations on prescribed
points. Contact-region discovery and the HHT solve are outside these
timings.

The parametric-sampling trial reuses the 50,000 sample locations and
the 54 branches from the mesh trial. The adapter recovers each sample's
original STEP face and interpolates its surface parameters using the
tessellation chart. `nc_contact cad-surface-samples` checks the trimmed
face and evaluates its CAD position and outward normal. For an out-of-trim
sample, the adapter uses parameters of a vertex on the same triangle and
records the count and position shifts. The branch labels, Boolean tree,
validation queries and line-load setup are shared with the mesh trial.

```powershell
$cad = 'build/aes-nhrep-parametric'
python benchmarks/aes/nhrep_gear.py prepare-parametric $gear $cad --mesh-trial $out --exe build/aes-occt/Release/nc_contact.exe
python benchmarks/aes/nhrep_gear.py train build/deps/nh-rep $cad --trial parametric --epochs 15000 --batch 16384
python benchmarks/aes/nhrep_gear.py audit "$cad/gear_parametric_trial_spur_50k_model_h.pt" $cad
```

The parametric model trained from epoch zero. The archived mesh trial
resumed from checkpoints. Both used the same batch size and final epoch.
Each report represents one trained model. The [parametric trial report](../aes/nhrep-spur-parametric.json)
records a P95 distance error of 40.10 µm on smooth faces and 48.98 µm
near sharp edges, against 41.03 µm and 61.21 µm in the mesh-sampled
trial. All 448 distance-check errors exceed 0.125 µm in both trials.
The parametric field gives 0 N and 711.67 N on the two contact lines,
against 149.35 N per line with CAD distances at the same pose.

The [50 mm cube control](../aes/cube-field/cube-field.json) uses analytic
signed distances and outward distance gradients for supervision. It tests
a six-branch NH-Rep with distance and gradient losses and a convex field
fitted from tangent planes. It checks analytic distances against the
archived STEP cube at 256 test points. Both TorchScript files beside the
report accept an `N × 3` coordinate tensor in millimetres and return one
signed distance in millimetres per point (negative inside).

```powershell
python benchmarks/aes/cube_field.py build/deps/nh-rep build/aes-occt/Release/nc_contact.exe build/aes-cube-field
```

The [spur pinion field trial](../aes/gear-field/field-report.json) compares the global
tangent field, the triangle mesh and a local distance query using fitted CAD faces.
The local model fits tensor-product Chebyshev polynomials to parameterized CAD surface
samples. A triangle BVH supplies initial features. Near a crease the query checks
projections onto both faces and their common edge. The saved guide and coefficients
are sufficient for inference. The benchmark uses the 452 archived queries, a crease
diagnostic (seed 903), and a separate 1,536-point final check (seed 1407).
To rebuild from the gear training data above, install the Python libigl bindings and run
the following command with the OCCT runtime available.

```powershell
python benchmarks/aes/gear_field.py build/aes-gear-field --prepare build/aes-occt/Release/nc_contact.exe
```

From the repository root, load the saved query using coordinates in millimetres.
It returns one signed distance per point, with negative values inside.

```python
from pathlib import Path
from sys import path
path.insert(0, 'benchmarks/aes')
from gear_field import FittedFaces

model = Path('results/aes/gear-field')
field = FittedFaces(model/'guide.npz', model/'faces.npz')
distance_mm = field.predict([[30.365967780378336, 1.8307461969484468, 0.0]])
```

~~~text
drives count
body_id orbit arm_xyz axis_xyz angular_speed
body_id table row_count
time dx dy dz
...
~~~

Optional v2 load tables apply time-dependent forces to dynamic coordinates.
Each table starts at zero and covers the integration horizon. The forces
are linearly interpolated between samples and added to the coordinate's
constant applied load.

~~~text
loads count
load coordinate_id row_count
time force
...
~~~

An optional resistance block follows the v2 drives. A sliding resistance projects the relative velocity at two material points into the specified world-space plane and applies regularized Coulomb friction with a prescribed preload. Its tangential force acts on the two bodies through their velocity Jacobians. The contact probes evaluate their own normal forces independently. The slider-crank deck uses this block for the support plate and sets the guide viscosity in its translational coordinate.

~~~text
resistances count
slide body_a body_b anchor_a_xyz anchor_b_xyz plane_normal_xyz force_limit_N slip_speed_m_s
~~~

A second optional block adds translational Kelvin–Voigt elements. Principal
axes, the rest separation and the per-axis coefficients are in body A's
local frame. The relative rate includes both anchor motion and transport of
that frame. The element contributes generalized forces and velocity and
restoring tangents to the same implicit solve as the contact probes.

~~~text
bushings count
bushing body_a body_b anchor_a_xyz anchor_b_xyz rest_xyz K_xyz C_xyz
~~~

An optional accelerometer block records signed specific force along a
body-fixed unit axis at the given local point. The output column
`sN_a_m_s2` includes the HHT generalized accelerations, angular
acceleration and centripetal terms, prescribed orbit acceleration, and
gravity projected into the sensor axis. The block is available in v2
scenes after the drives.

~~~text
accelerometers count
accelerometer body_id point_local_xyz axis_local_xyz
~~~

A v2 scene with dynamic coordinates can also set an incidence-dependent
normal law for individual probes. The compressive force is
`max(0, K * depth^n * (1 - kappa * gap_rate / approach_speed))`.
Positive gap rate means opening. The solver estimates the approach speed
from the previous accepted state and the entry step, holds it through
compression, and clears it upon reopening. The positive floor supplies
a finite speed for grazing and for initially compressed states. A probe
with this law records `pN_gap_rate_m_s` and `pN_impact_speed_m_s`.

~~~text
impact-laws count
impact probe_id kappa minimum_approach_speed_m_s
~~~

## Reproduce

Set the OCCT runtime path using the [project build instructions](../../README.md#step-queries-on-windows). Run the commands from the workspace root.

~~~powershell
cmake --build build/occt --config Release --target nc_contact
python benchmarks/contact_validation/gauge.py build/occt/Release/nc_contact.exe build/contact-validation/source results/contact_validation/models
python benchmarks/contact_validation/run.py build/occt/Release/nc_contact.exe build/contact-validation/source results/contact_validation
python benchmarks/contact_validation/beam.py build/occt/Release/nc_contact.exe results/contact_validation build/contact-validation/qi2016_figure17.jpg
python benchmarks/contact_validation/timing.py build/occt/Release/nc_contact.exe results/contact_validation
python benchmarks/contact_validation/plots.py results/contact_validation
python benchmarks/contact_validation/render.py results/contact_validation build/contact-validation/render
python benchmarks/hht/paper.py results/hht
ctest --test-dir build/occt -C Release --output-on-failure
~~~

To run an individual deck, pass the desired deflection, integration step, duration and output stride. The following replays the first held-out rebound.

~~~powershell
build/occt/Release/nc_contact.exe results/contact_validation/scenes/bounce-06.scene 0 0.0000125 1.0 16 build/contact-validation/replay
~~~

Integration timing includes the prepared geometry query, force evaluation and HHT updates. It excludes STEP setup and diagnostic output. The recorded timing also contains an ablation against the unprepared OCCT ray query. Its validation uses the same contact equations and measurements.
