# NASA gear-noise rig model revision

The original runs and digitized NASA force points remain under [external gear-rig comparison](../README.md). This revision compares two approaches to the tooth-pair compliance of the 28/28 spur pair. Inputs trace to [NASA TM-103232](https://ntrs.nasa.gov/citations/19910013253) and the [DANST manual](https://ntrs.nasa.gov/citations/19930018861).

## Model and verification

- [Profile pixels](models/rig28-profile-pixels.csv) retain four traces per gear from Fig. 3. We remove each trace's mean baseline over 12–20 degrees, then use the average relative shape with the 23/25 µm magnitudes of Table I. The resulting [profile](models/rig28-profiles.csv) is the input to both STEP exports and both geometry backends.
- The model converts the printed customary inertias using 0.112984829 kg·m² per lbf·in·s². The 6158 N·m/rad gearbox stiffness is used for a separate modal audit. With the converted inertias and the reported springs, the calculated gearbox frequencies are 6.56, 524.90 and 1220.79 Hz. Table II prints 6.56, 52.5 and 1220 Hz. The second frequency warrants independent verification.
- The tooth-pair model adds bending, shear, radial compression, and local line-contact compliances. A separate series compliance is selected to reproduce the approximate static DANST plateau in Fig. 7(c). The selected pitch-point stiffness is 67.5 MN/m. The uncalibrated beam model gives 172.8 MN/m. All dynamic tests of the calibrated model use the same stiffness, profile, and damping ratio of 0.10.
- Quasi-static solves cover 224 mesh phases at 11.48, 33.73, and 67.46 N·m. The maximum output torque imbalance is below (10^{-6}) N·m for either backend. The native fraction of phases with two loaded tooth pairs increases from 10.3% to 29.9% and 60.7% with torque. The static target is a numerical reference from NASA's report. The dynamic targets are the digitized experiments.
- HHT runs cover 0.48 s per condition. We repeat 7(b) and 7(d) at 0.5 µs. The waveform changes are 0.42 and 0.16 N RMS, respectively. The minimum recorded gap across the calibrated native and mesh runs is 29.33 µm and the solver uses the 2 µm working-flank gate.

## Dynamic results

| NASA panel | Original B-rep RMSE (N) | Calibrated B-rep RMSE (N) | Calibrated BVH RMSE (N) | Calibrated B-rep peak error |
|---|---:|---:|---:|---:|
| 7(a) | 63.06 | 63.18 | 63.88 | −3.5% |
| 7(b) | 146.71 | 203.17 | 201.85 | +0.6% |
| 7(c) | 148.79 | 174.03 | 176.04 | −17.6% |
| 7(d) | 90.77 | 75.69 | 81.63 | −4.7% |

The digitized 7(b) zero-force bins span 19.25–22.00 degrees on the scoring grid. The calibrated B-rep force stays positive and reaches a minimum of 125.88 N on those bins. The independent beam-based solution unloads earlier in the cycle. The 5821 Hz sensitivity that matches DANST's highest quoted mode in a frozen-pitch linearization also unloads earlier than the measured interval. These variants establish sensitivity to stiffness and inertias. A measured or independently identified dynamic transfer model of the shafts and supports would constrain the phase more directly.

Three alternating 0.16 s timing repeats of 7(b) give a median integration time of 1.53 s for B-rep and 6.94 s for the 1.5625 µm triangle BVH, a factor of 4.54. The timing excludes geometry setup and output. [Raw timing trials](static-matched/timing.json)

The 0.25 µs check of 7(b) changes the nominal B-rep trace by 1.33 N RMS. The audit uses 1 µs recorded outputs. Across 51,409 individual samples inside the measured zero-force interval, the minimum simulated force is 125.21 N and the zero-count is zero. [Step audit](static-matched/step-audit.json) · [Three-step curves](static-matched/curves/nasa-b-step-audit.csv)

[Calibrated metrics](static-matched/summary.json) · [Calibration grid](static-matched/static-calibration/grid.json) · [Beam model metrics](summary.json) · [Modal sensitivity](modal-sensitivity/summary.json) · [Dynamic force](static-matched/figures/nasa-revised-force.pdf) · [Quasi-static load shares](static-matched/figures/nasa-revised-statics.pdf) · [Profile comparison](static-matched/figures/nasa-revised-profile.pdf) · [Paper](../../paper/industrial.pdf)

[Isometric animation](static-matched/animations/nasa-b-isometric.mp4) · [Face-on animation](static-matched/animations/nasa-b-face.mp4)

## Reproduce

Set the OCCT runtime path using the [project build instructions](../../../../README.md#step-queries-on-windows). From the project root, with the NASA source already fetched as in the [original protocol](../README.md), run the following commands.

~~~powershell
cmake --build build/occt --config Release --target nc_gears
python benchmarks/gears/experimental/nasa_profile.py build/gears-experimental/source/nasa-tm-103232.pdf results/gears/experimental/revised/models
build/occt/Release/nc_gears.exe models rig28-flex results/gears/experimental/revised/models
python benchmarks/gears/experimental/nasa_revision.py build/occt/Release/nc_gears.exe results/gears/experimental
python benchmarks/gears/experimental/nasa_revision.py build/occt/Release/nc_gears.exe results/gears/experimental --fit-static --output results/gears/experimental/revised/static-matched
python benchmarks/gears/experimental/nasa_revision.py build/occt/Release/nc_gears.exe results/gears/experimental --timing
python benchmarks/gears/experimental/nasa_revision.py build/occt/Release/nc_gears.exe results/gears/experimental --modal-sensitivity
python benchmarks/gears/experimental/nasa_revision.py build/occt/Release/nc_gears.exe results/gears/experimental --step-audit
python benchmarks/gears/experimental/nasa_revision_artifacts.py results/gears/experimental/revised/static-matched
python benchmarks/gears/experimental/nasa_revision_render.py results/gears/experimental/revised/static-matched
python benchmarks/gears/paper.py results/gears
ctest --test-dir build/occt -C Release --output-on-failure
~~~
