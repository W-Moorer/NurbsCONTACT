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
| Slider-crank with pin–bore clearance | Prescribed crank with dynamic rod rotation and slider translation. Compare slider travel and acceleration on the published encoder angle. | Seven clearance–speed combinations from the vector paths of the publisher PDF. Use published material and restitution parameters without fitting measured curves. |

The same nominal STEP parts are used by both backends. These STEP reconstructions use the dimensions of the published geometry, including GAUGE's released assets, and the reported steel specimens. GAUGE scoring uses neither a fitted time shift nor a fitted trajectory transform. The rebound score is for the vertical component. The slide score is for supported translation. The beam score covers the first pulse in every reported speed condition. The coarser camera data support rebound heights and trajectory timing at the recorded cadence. Electrical pulse duration provides the independent high-bandwidth observable.

[GAUGE data and license](https://huggingface.co/datasets/InternRobotics/GAUGE-Dataset) · [GAUGE paper](https://arxiv.org/abs/2608.05948) · [Qi and Yin's experiment](https://doi.org/10.1177/1687814016668089)

## Results and source files

Across 61 held-out optical trajectories, the B-rep median RMSE is 6.90 mm for rebound height and 3.16 to 4.05 mm for supported displacement. All 17 held-out rebound apices are within one 30 Hz frame of their observed times. The 27 independent electrical durations give a 6.56 µs RMSE against measured durations of 174 to 199 µs. The mean signed bias is +6.50 µs. Halving the step changes the representative B-rep rebound by 0.10 mm RMS and the steel-beam duration cohort by 0.005 µs RMS.

Seven serial timing repeats give triangle-BVH to B-rep integration time ratios of 0.93 for rebound, 1.21 for supported sliding, and 0.47 for steel-beam impact. Preparing planar trims accelerates the sliding deck by 12.67× against direct native OCCT intersections. The compared geometry preparation preserves the sampled trajectories.

- [GAUGE error distributions and per-trial results](summary.json) · [Original URLs, data revision and checksums](sources.json) · [Paired sampled curves](curves)
- [Electrical duration metrics](beam/summary.json) · [Paired durations](beam/comparison.csv) · [Digitization and source coordinates](beam/measured.csv)
- [Force, position and gap traces](runs) · [Scene decks](scenes) · [Performance repetitions](timing.json)
- [Measured-versus-simulated figure](figures/contact-validation.pdf) · [Final paper](../../docs/hht.pdf)
- [Multi-view animations](animations/index.json)
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

~~~text
drives count
body_id orbit arm_xyz axis_xyz angular_speed
body_id table row_count
time dx dy dz
...
~~~

An optional resistance block follows the v2 drives. A sliding resistance projects the relative velocity at two material points into the specified world-space plane and applies regularized Coulomb friction with a prescribed preload. Its tangential force acts on the two bodies through their velocity Jacobians. The contact probes evaluate their own normal forces independently. The slider-crank deck uses this block for the support plate and sets the guide viscosity in its translational coordinate.

~~~text
resistances count
slide body_a body_b anchor_a_xyz anchor_b_xyz plane_normal_xyz force_limit_N slip_speed_m_s
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
