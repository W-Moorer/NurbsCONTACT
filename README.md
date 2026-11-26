# NurbsCONTACT

NurbsCONTACT computes distance bounds and continuous clearance directly on trimmed CAD surfaces. The C++20 core queries the original parameter charts, including the trim wires that define holes and face boundaries. The STEP adapter uses Open CASCADE. The core also accepts model geometry constructed in memory.

## Build and verify

### Frozen CPU rigid-contact baseline

The CPU reference for porting CAD contact is [commit 1406914](https://github.com/W-Moorer/NurbsCONTACT/tree/1406914cf11e64fefd758d91075e0d547ace5fd8). The two-stage reducer uses 50,000 steps of 0.1 ms, shared penalty stiffness, predictor geometry and HHT spectral radius zero. The following are measured online times on the reference workstation with four workers. Each response matched all 50,001 frames of its own archived CPU reference.

| Geometry method | Online time for 5 s |
| --- | ---: |
| Gmsh triangles, `mesh-obb --mesh-region-scope contour-ray` | 222.02 s |
| Precompiled CAD (ours), `cad-compiled` | 193.63 s |

Build Release with `NC_WITH_CPU_MESH=ON` and oneTBB available for the comparison. Set `OMP_NUM_THREADS=4`. Run `nc_contact <scene> 0 0.0001 5 1 response --method cad-compiled --solver penalty --penalty-stiffness shared --geometry-update predictor --spectral-radius 0`. The standalone `nc_contact_runtime` also runs the CAD method directly from the four compiled parts. Part compilation and scene initialization are timed separately. The original STEP files and Gmsh meshes are in the [CPU input deck](results/aes/cpu-foundation/reducer-physics/cpu-deck.zip). Use the [compiled scene and parts](results/aes/cpu-bezier/standalone/reducer-inputs.zip), [validation results and known narrow-contact cases](results/aes/cpu-bezier/standalone/README.md) and [raw thread-scaling measurements](results/aes/cpu-bezier/standalone/simulation-timing/cpu-worker-scaling.json) to reproduce and interpret the comparison.

The frozen CAD assets have a separate [source-to-runtime equivalence audit](results/aes/cpu-bezier/standalone/README.md#source-to-runtime-geometry-check). It compares the compiled contact charts with the original STEP solids.

### Precompiled CAD on CUDA

Build with `NC_WITH_CUDA_COLUMNS=ON`. The experimental `cuda-resident` path uploads the compiled surface charts, spatial indices, quadrature data, joint coordinates and input loads during setup. A CUDA graph advances contact discovery, pressure integration, penalty forces and generalized-alpha time integration. After the graph finishes, the runtime downloads its status, dynamics trace and step timings. An unresolved geometric query, insufficient workspace or exhausted integration retries stops the simulation and reports the failing step.

```text
nc_contact_runtime <scene> 0 0.0001 5 1 gpu --method cad-compiled --device cuda-resident --solver penalty --penalty-stiffness shared --geometry-update predictor --spectral-radius 0
python benchmarks/aes/compare_compiled_devices.py cpu.csv gpu.csv comparison --plot
```

The five-second two-stage reducer took **444.02 s online on an RTX 3070**, with 5.90 s of setup. The frozen four-worker CPU run took 193.63 s. Both completed 50,000 main steps and 44 subdivisions. The two contact-pair flags agreed on all 50,001 frames and the largest absolute normal-load difference was 0.0000134 N. The median GPU step took 8.33 ms and every step exceeded its 0.10 ms deadline. Add `--require-realtime` to the comparison command to require both end-to-end throughput and zero missed step deadlines. [Run and phase timings](results/aes/cuda-compiled-resident/run.json) · [Responses and errors over five seconds](results/aes/cuda-compiled-resident/responses.png)

All four uploaded parts match the SHA-256 digests from the STEP equivalence audit. Device evaluation on 168,948 chart samples and 107,700 trim samples agreed with CPU material classification. The largest sampled position discrepancy was 3.56e-15 mm. The source audit also reports the continuous control-net bound for the polynomial surface spans. [GPU chart and boundary checks](results/aes/cuda-compiled-resident/part-audit.json) · [Original STEP checks](results/aes/cpu-bezier/standalone/part-equivalence.json)

The executable accepts the same scene and compiled parts as the CPU reference. This path currently supports up to 16 scalar coordinates, hinge and prismatic motion, prescribed coordinate loads and frictionless penalty regions. It checks the supported charts and solver settings during setup. The `.gpu.csv` report records online wall time, geometry phase times, step-duration percentiles and the number of missed step deadlines. Initialization and graph upload are included in `setup_s`.

The earlier hybrid prototype used `--device cuda-columns`. The host controlled the integrator, confirmed and grouped contact regions and handled unresolved rays. On the RTX 3070, this prototype and its four-worker CPU reference took 437.67 s and 219.52 s respectively. The 50,001 output frames matched exactly. [Archived hybrid comparison](results/aes/cuda-compiled-prototype/comparison.json)

### GPU mesh package

For CUDA simulation of rigid bodies on fixed parallel hinges, use the installable GPU backend. Its runtime includes mesh contact, the integrator and the native load-exchange library. [Download the GPU backend](https://github.com/W-Moorer/NurbsCONTACT/releases/tag/gpu-backend-v0.1.1) · [Installation and five-second reducer example](nurbscontact_gpu/README.md)

### Geometry core

The core has no third-party library dependency.

```powershell
cmake -S . -B build/core -DNC_WITH_OCCT=OFF
cmake --build build/core --config Release
ctest --test-dir build/core -C Release --output-on-failure
```

### STEP queries on Windows

The reference build uses Visual Studio 2022, CMake, and the Open CASCADE 7.9.3 Release SDK. Download and unpack the x64 OCCT and matching third-party archives from the release page. The example assumes both were unpacked under `build/deps`. [OCCT 7.9.3 release](https://github.com/Open-Cascade-SAS/OCCT/releases/tag/V7_9_3)

```powershell
$occtSdk = (Resolve-Path 'build/deps/opencascade-7.9.3-vc14-64').Path
$occtRuntime = (Resolve-Path 'build/deps/3rdparty-vc14-64').Path
$runtimeBins = Get-ChildItem -LiteralPath $occtRuntime -Recurse -File -Filter '*.dll' |
    Where-Object { $_.Directory.FullName -notmatch '[\\/]debug[\\/]' } |
    ForEach-Object { $_.DirectoryName } |
    Sort-Object -Unique
$env:PATH = "$occtSdk/win64/vc14/bin;" + ($runtimeBins -join ';') + ';' + $env:PATH

cmake -S . -B build/occt -A x64 "-DOpenCASCADE_DIR=$occtSdk/cmake"
cmake --build build/occt --config Release
ctest --test-dir build/occt -C Release --output-on-failure
```

The STEP test generates its fixtures in `build/occt/step_fixtures` and round-trips them through the translator. The fixtures can also be regenerated with `nc_step_tests <fixture-directory>`.

```powershell
build/occt/Release/nc_step.exe distance build/occt/step_fixtures/block-a.step build/occt/step_fixtures/block-b.step
build/occt/Release/nc_step.exe sweep build/occt/step_fixtures/block-a.step build/occt/step_fixtures/block-b.step --velocity 0 0 -1 --time 0 1 --clearance 0.1
build/occt/Release/nc_bench.exe build/occt/step_fixtures build/results.csv 9
python benchmarks/summarize.py build/results.csv
```

The CLI prints JSON. Time uses the units supplied with the velocity, rotations use radians, and the fixture lengths are in millimeters. Run `nc_step` without arguments to list the motion, tolerances, and search switches.

## Implicit HHT multibody contact

For the CPU spur-gear baseline, run `python benchmarks/aes/prepare_spur_cpu.py build/cpu-contact/spur-light`. This builds both Gmsh surfaces (18,742 triangles total), their CAD deviation reports and the dynamics scene. Use `--case reducer` for the four-gear reducer (58,974 triangles). The loaded solver used 30.1 and 58.2 MiB on the respective 5 ms checks. Select `--mesh-profile compact` to reproduce the 20,456- and 65,732-triangle controls. [Mesh preparation and measured memory](wiki/ContactDetection.md#default-spur-gear-mesh)

The same rigid scene selects `--method mesh-obb` with a Gmsh file beside each STEP asset or `--method cad-occt` directly on the solids. Use `--method chen-2026` for the safeguarded time-dependent envelopes. Configure `NC_CHEN_SOURCE_DIR` to enable `--method chen-2024` for TDIB patch candidates with trimmed-solid event checks. Select the response independently with `--solver penalty|sap|lagrange|augmented-lagrange|lcp|ncp|soccp`. The default penalty law uses HHT. The other options solve for simultaneous contact impulses. [Contact laws and reproduction](wiki/ContactDetection.md#surface-contact-solvers) · [CPU scene methods and impact comparison](benchmarks/chen/README.md#rigid-step-scenes)

The `nc_hht` benchmark couples prismatic and fixed-axis rigid joints through frictionless compliant contact. It compares native STEP boundaries against an AABB hierarchy of triangle faces using the same implicit HHT-α solver and interval clearance acceptance test. The examples include analytic validation, a six-cam assembly, and a driven twelve-ball train.

The `nc_contact` runner loads STEP components, generalized coordinates, material parameters and contact probes from a scene file. One signed-gap and compliance implementation serves all loaded scenes. A backend switch changes the surface query between native B-rep and an AABB hierarchy of triangles. The experimental comparisons cover rebound and sliding trajectories, electrical contact durations for steel-sphere impact on a beam, and clearance-joint slider motion. [Model interface and reproduction](results/contact_validation/README.md)

- [Experiment protocol and reproduction](results/hht/README.md)
- [Validated numerical summary](results/hht/metrics/summary.json)
- [Rigid-body comparison with native MuJoCo and Drake](benchmarks/engines/README.md)
- [AES manuscript source](docs/AES/Residual-Aware-CAD-Contact.tex) and [compiled paper](docs/manuscript.pdf)

## Industrial transmission and bearing benchmarks

The industrial suite adds four gear drives with involute or relieved spline flanks and a 14-roller bearing. It compares native STEP probes with AABB/BVH triangle intersections under the shared implicit HHT-α solver. Runs include mesh refinement, time refinement, helical axial quadrature refinement, and a 150,000-step reducer comparison.

- [Industrial models, results, animation links and reproduction](results/gears/README.md)
- [Industrial benchmark supplement](results/gears/paper/industrial.pdf)

## Query behavior

- `PreparedModel` builds an immutable BVH. Queries keep their own adaptive subdivisions.
- `distance` returns `[lower, upper]`, a feasible point pair when found, and search statistics. If the work budget is reached, the interval reports the uncertainty that remains.
- `sweep` returns `clear`, `hit`, or `unresolved`. Times strictly earlier than `safe_prefix` have been excluded. When `witness_time` exists, the first threshold entry lies between it and `safe_prefix`. `clear` certifies the full closed input interval. A `hit` can carry a wider bracket than requested if the budget or geometric resolution prevents further refinement.
- Witnesses carry chart coordinates, face/edge/vertex IDs, and enclosures of positions in body A's frame. The B feature ID and UV coordinates refer to B's native chart.

The trajectory is the position of B relative to stationary A. It consists of a fixed-axis rotation about B's local origin and an affine translation in A's frame. Compose longer piecewise paths by querying each segment with its own time interval and coefficients. The query measures unsigned boundary separation. It supports initially separated parts and positive clearance thresholds, with a conservative unresolved outcome for ambiguous grazing or zero-gap events.

Run `nc_step sweep A.step B.step --generic` to use the shape-independent patch and trim search evaluated in the AES contact comparison. This mode disables optional axial-envelope pruning and planar witness proposals. The trajectory still uses the supported fixed-axis rotation and affine translation. The default sweep retains radial and axial enclosures for compatible surfaces under fixed relative orientation. `--no-axial` disables that filter separately.

## Sources and evidence

| Path | Responsibility |
|---|---|
| `include/nurbscontact/interval.hpp` | Binary64 interval arithmetic and bounded trigonometric evaluation |
| `src/geometry.cpp` | Bézier support, analytic patches, p-curve membership and witnesses |
| `src/axial.cpp` | Radial envelopes, certified trim exclusion disks and axial time intervals |
| `src/query.cpp` | BVH search, static bounds, swept boxes, direction clipping and safe prefixes |
| `src/occt.cpp` | STEP adaptation, native curve extraction, knot insertion and feature IDs |
| `tests/` | Analytic and STEP round-trip verification |
| `benchmarks/run.cpp` | Matched comparison of directional, swept-box, chord-box, and ablation variants |
| `benchmarks/results.csv` | Raw results for six STEP scenarios, five configurations, and nine repetitions |
| `benchmarks/axial-results.csv` | Six controlled STEP scenarios with the optimization and original search configurations |
| `benchmarks/assemblies/README.md` | Public assembly provenance, component pairs and reproduction commands |
| `manuscript/ccd.md` | Derivation, guarantees, related work, and experimental protocol |

STEP support covers planes, cylinders, cones, spheres, tori, Bézier faces, and clamped nonperiodic rational B-splines. Trim curves can be lines, conics supported by the adapter, Bézier curves, or clamped nonperiodic B-splines. Import reports unsupported types and invalid trim wires. A face is evaluated in its source chart. The adapter bounds any UV connectors used to close wire endpoints and stores the maximum displacement in `Model::max_trim_join`. Certificates apply to this nominal boundary representation. CAD manufacturing tolerances and initial volumetric containment are separate queries.

## Last verified against

Windows x64 Release with MSVC 19.44 and OCCT 7.9.3 on 2026-11-21. CTest exercises the core and the generated STEP round trip. Tests cover rational knot-span extraction, distance bounds, a tilted radial envelope, circular bore passage, and eccentric pin entry at positive and negative times. The optimization reduced the eccentric pin query from 693.1 ms to 1.24 ms in nine matched repetitions, with all original analytic checks passing. The production assembly comparison covers the OpenAMRobot MMP wheel and lidar support STEP files. The full lidar insertion diagnostic retains a bracket wider than the time target at the configured task cap.
