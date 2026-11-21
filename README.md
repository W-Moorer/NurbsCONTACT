# NurbsCONTACT

NurbsCONTACT computes distance bounds and continuous clearance directly on trimmed CAD surfaces. The C++20 core queries the original parameter charts, including the trim wires that define holes and face boundaries. The STEP adapter uses Open CASCADE. The core also accepts model geometry constructed in memory.

## Build and verify

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

The `nc_hht` benchmark couples prismatic and fixed-axis rigid joints through frictionless compliant contact. It compares native STEP boundaries against an AABB hierarchy of triangle faces using the same implicit HHT-α solver and interval clearance acceptance test. The examples include analytic validation, a six-cam assembly, and a driven twelve-ball train.

The `nc_contact` runner loads STEP components, generalized coordinates, material parameters and contact probes from a scene file. One signed-gap and compliance implementation serves all loaded scenes. A backend switch changes the surface query between native B-rep and an AABB hierarchy of triangles. The experimental comparisons cover rebound and sliding trajectories, electrical contact durations for steel-sphere impact on a beam, and clearance-joint slider motion. [Model interface and reproduction](results/contact_validation/README.md)

- [Experiment protocol and reproduction](results/hht/README.md)
- [Validated numerical summary](results/hht/metrics/summary.json)
- [LaTeX source](manuscript/manuscript.tex) and [final paper](docs/manuscript.pdf)

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

For fixed relative orientation, the sweep also uses radial and axial enclosures of cylinders and planes with circular or elliptical trim arcs. The filter preserves a certified hole radius, intersects a cached face-pair time interval with the search interval, and evaluates witnesses on the original CAD geometry. Run with `--no-axial` to reproduce the directional search before this optimization. Angular-motion queries use the existing directional bounds.

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
