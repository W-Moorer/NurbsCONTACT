# Common untrimmed patch comparison with TDIB-CCD

The harness calls Chen et al.'s official OBB solver at commit b70aa3e95ab4f75d2166df7e0cbe52b64515a3cb. Both solvers receive the same untrimmed biquadratic control nets and linear control-point velocities. The fixed surface is the unit square at zero height. The moving graph has height \(0.375+0.25u-t\) in the planar-entry case and \(0.375+0.5[(u-0.5)^2+(v-0.5)^2]-t\) in the curved-entry case. The curved-clear case reduces the approach speed to 0.125. Thus both zero-distance entries occur at \(t=0.375\). [Chen et al. paper](https://doi.org/10.1145/3687960) · [Official implementation](https://github.com/xw-c/TDIB-CCD/tree/b70aa3e95ab4f75d2166df7e0cbe52b64515a3cb)

The OBB solver's deltaDist is \(5\times10^{-4}\) in combined parameter and candidate-time widths. The native solver requests \(5\times10^{-4}\) in distance and time, with a 25,000-task cap. The OBB solver returns a candidate time and the centers of the retained parameter cells. The native solver reports a certified prefix and requires a verified original-surface pair for a witnessed entry. Runtime observations include query work and exclude control-net construction. Nine measured repetitions follow one warm-up, alternating the order. They characterize these fixtures and termination rules.

Both solvers exclude the curved-clear case. The OBB candidates for the entry cases are near the analytic time, with center point-pair residuals of \(4.92\times10^{-4}\) and \(6.91\times10^{-4}\). The native safe prefixes reach 0.375000 and 0.374832, respectively. Exact zero-distance witnesses remain unresolved at the task cap. The native-only run at positive clearance 0.025 uses separate outputs.

- [Zero-distance observations](common-observations.csv)
- [Separate positive-margin observations](positive-observations.csv)

## Reproduce

~~~powershell
git clone https://github.com/xw-c/TDIB-CCD.git build/tdib-ccd
git -C build/tdib-ccd checkout b70aa3e95ab4f75d2166df7e0cbe52b64515a3cb
cmake -S benchmarks/chen -B build/chen-common -DTDIB_SOURCE_DIR="$PWD/build/tdib-ccd" -DEIGEN_INCLUDE_DIR=C:/vcpkg/installed/x64-windows/include/eigen3
cmake --build build/chen-common --config Release
build/chen-common/Release/nc_chen_common.exe
build/chen-common/Release/nc_chen_common.exe --positive-only
build/chen-common/Release/nc_chen_common.exe --published-single
~~~

The last command replays the two orthogonal bicubic patches from the official `singleTest` example. The exact first intersection time is `10/17`. The official solver sees its original two velocities and the native solver sees the equivalent relative translation. [Measured published case](../../results/aes/general-contact/chen-published-single.csv)

Substitute the location of the local Eigen headers when needed. The [rotation follow-up](../clearance-followup/README.md) adds a chord-error bound, outward verification of proposed exclusions and an original-trajectory witness check around the official OBB search.

## Rigid STEP scenes

Configure the main project with `-DNC_CHEN_SOURCE_DIR=<TDIB-CCD checkout>` and Eigen3 available to CMake. Build `nc_contact` and `nc_cad_solid_tests`. The same STEP scene runs with `--method cad-occt`, `--method chen-2024`, or, in the CPU mesh build, `--method mesh-obb`. The mesh reader looks for a matching `.msh` beside each STEP asset. Use `--mesh-dir DIR` for an alternate resolution. All three take `--solver penalty`. Chen enables event checks by default at one fifth of the main step. Supply the same positive `--event-tol-s` to all methods when comparing event-controlled runs.

Both CAD methods obtain each loaded region directly from the intersection of the two placed STEP solids. The centroid of the common volume supplies the application point. Integrating the oriented boundaries of its source faces gives an area-weighted normal. The extent of the region along this normal supplies the depth in metres. A fully enclosed body or another overlap with an indeterminate separating normal is reported as unresolved.

The event check interpolates the pivot position linearly and follows the shortest orientation arc. Swept body boxes reject separated motions. The Chen option represents the bounded supporting surfaces as rational Bézier patches of degree at most three in each parameter. It calls the authors' TDIB OBB search on the moving control nets, with an additional chord-error margin for rigid rotation. Each proposed time is checked against the actual trimmed solids. A distance within the numerical tolerance sets an upper time for the subsequent search. A larger distance can exclude a time interval using the relative speed bound. OCCT closest distance and conservative advancement process the remaining interval, including trims, higher-degree surfaces, and motions for which TDIB returned no candidate. The proposal search uses a normalized cell width of 0.01, a shared limit of 1,024 popped nodes and up to 131,072 patch-box comparisons per moving pair. The generated header reports boundary arithmetic failures to the caller. This is a numerical STEP event locator without an exact-arithmetic separation certificate.

The implementation checks entry from a separated endpoint or release by reversing the motion. Steps with two loaded endpoints currently receive no internal release/re-entry check. Runs record CAD Boolean calls, trimmed-solid distance evaluations, TDIB patch queries and confirmed proposals. The mesh method builds its own force regions from the Gmsh surfaces.

```powershell
python benchmarks/chen/rigid_impact.py build/cpu-swept/Release/nc_contact.exe build/chen-scene/impact build/chen-scene/impact/report.json
```

This replays the existing two-slider sphere impact with identical material, time step and event tolerance. The independent solution gives entry at 0.040 s, release at 0.07141593 s, peak load 22.5 N and impulse 0.45 N·s. The time brackets in the report are adjacent saved frames surrounding each change of load. CTest also checks step-interior collisions, separation at a circular trim, reverse-time release and rotation around an offset pivot. [Comparison against the analytical dynamics](../../results/aes/general-contact/chen-step-impact.json)

The default mesh is generated from the 41 mm radius STEP sphere with a 4.5 mm Gmsh target, giving 2,610 triangles per sphere. Pass one or more existing directories to `--mesh-dir` for a refinement study, or use `--cad-only` in a build without the mesh backend. At 250 µs steps and 50 µs event tolerance the following figures cover the same 641 output frames. The peak working set covers the full native process. Mesh generation and initialization are outside the integration timer.

| Contact method | Triangles per sphere | Peak working set (MiB) | Integration (s) | Max displacement error (µm) | Max force error (N) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mesh OBB, 4.5 mm | 2,610 | 18.2 | 3.09 | 65.04 | 0.53548 |
| Mesh OBB, 3 mm | 5,708 | 20.4 | 4.42 | 51.26 | 0.26360 |
| Mesh OBB, 1.5 mm | 22,482 | 32.2 | 10.35 | 12.61 | 0.06366 |
| CAD conservative advancement | — | 24.3 | 21.61 | 0.83 | 0.00458 |
| CAD with Chen 2024 proposals | — | 24.7 | 22.30 | 0.83 | 0.00458 |

Each timing is one run on a shared CPU. The two CAD histories match byte for byte. Chen made 56 patch queries, confirmed two proposed times, and performed 280 trimmed-solid distance evaluations. Plain conservative advancement performed 275. In a second [free rigid-body impact](../../results/aes/general-contact/chen-free-eccentric.json), the same Chen method ran with offset centers of mass, rotation and friction for 0.16 s. The CAD tests also check that splitting a loaded face preserves its contact point, normal and depth, and that two disconnected overlaps retain separate forces.

## 2026 labeled triangular-patch queries

The authors' [TDIB-E data release](https://github.com/xw-c/TDIB-E/tree/5012314ca15c01b1e368a102141af845fdf6add3) provides face–face and vertex–vertex cases with rational control points and exact binary labels. Both subsets encode pairs of quadratic triangular Bézier patches with linearly moving control points. The [2026 paper](https://doi.org/10.1145/3811310) introduces floating-point error analysis for its CCD solver. The executable here calls the official **2024** TDIB-OBB solver on the released **2026 data**.

```powershell
git clone https://github.com/xw-c/TDIB-E.git build/deps/tdib-e
git -C build/deps/tdib-e checkout 5012314ca15c01b1e368a102141af845fdf6add3
cmake --build build/chen-common --config Release --target nc_chen_triangles
python benchmarks/chen/tdibe_replay.py build/chen-common/Release/nc_chen_triangles.exe build/deps/tdib-e build/chen-common/tdibe.json --per-file 16 --width .0005 --query-timeout 8
```

The driver selects 16 indices at equal index intervals in each of eight published files. It preserves rational coordinates up to conversion to binary64, constructs velocity from the rounded endpoints and records the resulting maximum coordinate perturbation of each affine trajectory. This perturbation was zero for all 128 selected queries. At a parameter/time cell width of 0.0005 all 64 positive examples returned a candidate, 60 of 64 negative examples returned clear and all queries completed. The four extra candidates were in the face–face near-miss subset. Those four returned candidates again at width 0.00005. A candidate is an inclusion test result. The published labels concern geometric intersection in the entire time interval. The runs do not supply a contact force or a CAD trimming classifier. [Per-query results](../../results/aes/general-contact/chen-2024-on-tdibe-2026.json) · [Tighter-width rerun](../../results/aes/general-contact/tdibe-close-negatives-width-5e-5.json)
