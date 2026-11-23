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

Build `nc_contact` and `nc_cad_solid_tests` with OCCT. The same STEP scene accepts `--method cad-occt` or `--method chen-2026`. Configure `-DNC_CHEN_SOURCE_DIR=<TDIB-CCD checkout>` with Eigen3 to enable `--method chen-2024`. The CPU mesh build also accepts `--method mesh-obb`, `mesh-aabb` and `mesh-sap` and reads a matching Gmsh surface beside each STEP asset. Use `--mesh-dir DIR` for an alternate resolution. The geometry and response options are independent. Both Chen methods enable event checks by default at one fifth of the main step. Supply the same positive `--event-tol-s` to every method for an event-controlled comparison.

All three CAD methods obtain each loaded region directly from the intersection of the two placed STEP solids. The centroid of the common volume supplies the application point. Integrating the oriented boundaries of its source faces gives an area-weighted normal. The extent of the region along this normal supplies the depth in metres. A fully enclosed body or another overlap with an indeterminate separating normal is reported as unresolved.

The event check interpolates the pivot position linearly and follows the shortest orientation arc. Swept body boxes reject separated motions. The 2024 option represents the bounded supporting surfaces as rational Bézier patches of degree at most three in each parameter. It calls the authors' TDIB OBB search on the moving control nets, with an additional chord-error margin for rigid rotation. The 2026 variant uses the [safeguarded envelopes](#safeguarded-envelopes) on supported charts. Each proposal is checked on the trimmed solids. A distance within the numerical tolerance supplies an upper time for the subsequent search. A larger distance can exclude a time interval using the relative speed bound. OCCT distance and conservative advancement cover the remaining interval, including unsupported charts. The 2024 proposal search uses a normalized cell width of 0.01, a shared limit of 1,024 popped nodes and up to 131,072 patch-box comparisons per moving pair. The rigid STEP event locator is numerical.

The implementation checks entry from a separated endpoint or release by reversing the motion. Steps with two loaded endpoints currently receive no internal release/re-entry check. Runs record CAD Boolean calls, trimmed-solid distance evaluations, TDIB patch queries and confirmed proposals. The mesh method builds its own force regions from the Gmsh surfaces.

```powershell
python benchmarks/chen/rigid_impact.py build/cpu-swept/Release/nc_contact.exe build/chen-scene/impact build/chen-scene/impact/report.json
```

This replays the existing two-slider sphere impact with identical material, time step and event tolerance. The independent solution gives entry at 0.040 s, release at 0.07141593 s, peak load 22.5 N and impulse 0.45 N·s. The time brackets in the report are adjacent saved frames surrounding each change of load. CTest also checks step-interior collisions, separation at a circular trim, reverse-time release and rotation around an offset pivot. [Comparison against the analytical dynamics](../../results/aes/general-contact/chen-step-impact.json)

The default Gmsh target is 8 mm on the 41 mm radius sphere, giving 904 triangles per sphere. Pass existing directories to `--mesh-dir` for a refinement study, `--mesh-only` to run only the mesh controls or `--cad-only` for the STEP controls. At 250 µs steps and 50 µs event tolerance the following figures cover the same 641 output frames. The peak working set covers the full native process. Mesh generation and initialization are outside the integration timer. [462- and 904-triangle runs](../../results/aes/cpu-mesh/impact/low-memory.json)

| Contact method | Triangles per sphere | Peak working set (MiB) | Integration (s) | Max displacement error (µm) | Max force error (N) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mesh OBB, 12 mm | 462 | 17.2 | 3.79 | 474.42 | 2.56747 |
| Mesh OBB, 8 mm (default) | 904 | 17.5 | 5.03 | 29.95 | 0.56624 |
| Mesh OBB, 4.5 mm | 2,610 | 18.2 | 3.09 | 65.04 | 0.53548 |
| Mesh OBB, 3 mm | 5,708 | 20.4 | 4.42 | 51.26 | 0.26360 |
| Mesh OBB, 1.5 mm | 22,482 | 32.2 | 10.35 | 12.61 | 0.06366 |
| CAD conservative advancement | — | 24.3 | 21.61 | 0.83 | 0.00458 |
| CAD with Chen 2024 proposals | — | 24.7 | 22.30 | 0.83 | 0.00458 |

Each timing is one run on a shared CPU. The two CAD histories match byte for byte. Chen made 56 patch queries, confirmed two proposed times, and performed 280 trimmed-solid distance evaluations. Plain conservative advancement performed 275. In a second [free rigid-body impact](../../results/aes/general-contact/chen-free-eccentric.json), the same Chen method ran with offset centers of mass, rotation and friction for 0.16 s. The CAD tests also check that splitting a loaded face preserves its contact point, normal and depth, and that two disconnected overlaps retain separate forces.

The [three current STEP methods](../../results/aes/general-contact/surface-methods-impact.json) returned byte-identical traces on this impact, matching the earlier CAD trace. Their maximum displacement error was 0.831 µm and their impulse was 0.450015 N·s. The 904-triangle-per-sphere mesh gave 29.95 µm and 0.450029 N·s.

On the archived eccentric cam we also replayed the first contact and release over 0.12 s, with 1 ms HHT steps and 0.2 ms event tolerance. All methods recovered the two switches of the smooth-cylinder DOP853 reference in order. RMS follower displacement error was 15.93 µm for the [1,426-triangle control](../../results/aes/general-contact/cam-contact-cycle-mesh-obb.json) and 5.72 µm for [plain STEP](../../results/aes/general-contact/cam-contact-cycle-cad-occt.json), [Chen 2024](../../results/aes/general-contact/cam-contact-cycle-chen-2024.json) and [the 2026 variant](../../results/aes/general-contact/cam-contact-cycle-chen-2026.json). Integration took 2.68, 8.29, 28.75 and 8.51 s respectively on the shared CPU. The 2026 method confirmed one patch proposal on the trimmed solids. These timings cover one loaded episode.

~~~powershell
python benchmarks/aes/cpu_mesh_cam.py run build/cpu-swept/Release/nc_contact.exe build/cpu-contact/cam/model build/cpu-contact/cam/compact-planar build/chen-scene/cam build/chen-scene/cam.json --method chen-2026 --dt .001 --duration .12 --event-tol-s .0002
~~~

### Gear distance and event costs

The [20,456-triangle gear pair](../../results/aes/general-contact/spur-step-mesh-obb.json) took 40.49 s of integration for the first seven steps of 50 µs with 25 µs event tolerance. With the original whole-solid distance kernel, [plain CAD](../../results/aes/general-contact/spur-step-cad-occt.json) and [Chen-assisted CAD](../../results/aes/general-contact/spur-step-chen-2024.json) both reached the 30-minute limit. At two fixed poses from the mesh trace, each whole-solid OCCT distance call reached its 120 s limit. The solid intersections at those poses took 3.10 and 5.26 s and reported separation and overlap, respectively. [Static queries and original poses](../../results/aes/general-contact/spur-static-cad-cost.json)

The updated STEP event path builds a bounding-box tree over the original vertices, edges and faces for pairs with more than 16 faces in total. At each trial pose it searches features within the distance the bodies can close during the step, then checks shell containment in both directions. An actual vertex outside an opposing bounding box excludes containment of that connected shell once boundary crossings have been ruled out. Identical trial poses can reuse distance and force-region results. A confirmed event and a separated prefix bound the returned event time to one quarter of the time tolerance requested by the integrator.

At the initial gear pose the indexed query and independent OCCT shell query measured 0.04879230 mm. The queries took 13.32 s and 10.15 s, respectively. At 0.25 ms both measured zero, taking 0.88 s and 9.79 s. Building both indices took 0.089 s. The two poses have independently checked volume-overlap status. These single shared-host measurements cover static geometry. [Distance, timings and feature counts](../../results/aes/general-contact/spur-indexed-distances.json)

The [first, separated 50 µs step](../../results/aes/general-contact/spur-separated-first-step.json) took 395.49 s with indexed CAD conservative advancement and 15.68 s with the 20,456-triangle OBB control. Both produced identical traces. HHT evaluated the force twice. A single CAD step-acceptance check made 19 distance queries that visited 132,488 box pairs and called the feature extrema kernel 9,314 times. The run also performed three solid Booleans. These counts distinguish the cost of repeatedly verifying separation along a rotating trajectory from the cost of building the index.

With only the full-body boundary index, all three STEP methods reached the 1,800 s cutoff on the seven-step gear run. The [plain kernel](../../results/aes/general-contact/spur-indexed-cad-occt.json), [2024 proposals](../../results/aes/general-contact/spur-indexed-chen-2024.json) and [2026 proposals](../../results/aes/general-contact/spur-indexed-chen-2026.json) have no completed baseline trace on this interval.

Localizing the two STEP solids and bounding the relative normal speed over the reachable region brought all seven steps within that cutoff. At each pose the force query retains a padded piece of each original solid and reuses it while its window covers the possible overlap. The event query includes every point that can approach within its distance cap. The first separated step still produced the same two output rows. Original-surface distance calls fell from 19 to 7 and feature extrema from 9,314 to 2,487. Online time fell from 394.54 to 65.22 s in the paired first-step run. [First-step timing and counts](../../results/aes/general-contact/spur-local-first-step.json) · [Full shell distance controls](../../results/aes/general-contact/spur-local-distances.json)

The four runs below used the same 0.35 ms scene, 50 µs main step, 25 µs event tolerance and linear penalty law on the shared Ryzen 9 5900X. Runs overlapped in wall time. The peak is the largest of eight saved loads, including the initial frame. Impulse is the trapezoidal integral on this grid.

| Geometry and event method | Online (s) | Process (s) | Peak working set (MiB) | Peak (N) | Impulse (N·s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Gmsh mesh, OBB](../../results/aes/general-contact/spur-local-control-mesh-obb.json) | 68.25 | 70.53 | 29.9 | 5,847.69 | 0.524155 |
| [Local STEP + conservative advancement (ours)](../../results/aes/general-contact/spur-local-cad-occt.json) | 714.16 | 716.75 | 139.3 | 5,377.07 | 0.521094 |
| [Local STEP + Chen 2024 proposals](../../results/aes/general-contact/spur-local-chen-2024.json) | 718.97 | 722.93 | 146.8 | 5,377.07 | 0.521094 |
| [Local STEP + Chen 2026 variant](../../results/aes/general-contact/spur-local-chen-2026.json) | 726.36 | 730.15 | 156.4 | 5,377.07 | 0.521094 |

Running the mesh and plain local STEP jobs one after the other on the same workstation took 36.14 and 388.21 s online, respectively (37.54 and 390.87 s end to end). Both reproduced their corresponding concurrent-run response files byte for byte. Local STEP used 10.74 times the online time of the compact mesh over this first loading episode. [Sequential mesh run](../../results/aes/general-contact/spur-serial-mesh-obb.json) · [Sequential STEP run](../../results/aes/general-contact/spur-serial-cad-occt.json)

The three STEP traces were byte-identical. Both patch methods made 16 patch queries with zero confirmed proposals. Each STEP run made 143 CAD distances, 45,382 feature extrema and 58 solid Booleans. All four methods first carried load at the 0.25 ms saved frame. Full-solid replay at the eight saved STEP poses and three extra poses linearly interpolated through first loading gave identical region counts, centroids, normals and depths to local-solid replay. At 0.2125 ms the interpolated solids were separated and at 0.2250 ms two regions carried load. These intermediate poses sample the interpolated motion between saved frames. This same-pose comparison checks the geometry change. An independent dynamic reference is still needed to assess the gear's physical peak load. [Full-solid replay](../../results/aes/general-contact/spur-local-regions.json) · [STEP response](../../results/aes/general-contact/spur-local-cad-response.csv) · [Mesh response](../../results/aes/general-contact/spur-local-mesh-response.csv) · [Load and shaft speeds](../../results/aes/general-contact/spur-local-response.png)

~~~powershell
cmake --build build/cpu-swept --config Release --target nc_cad_distance_benchmark
python benchmarks/chen/gear_distance.py build/cpu-swept/Release/nc_cad_distance_benchmark.exe build/chen-scene/gear-distances.json
python benchmarks/chen/gear_distance.py build/cpu-swept/Release/nc_cad_distance_benchmark.exe build/chen-scene/local-distances.json --local
python benchmarks/aes/cpu_mesh_gear_dynamics.py build/cpu-swept/Release/nc_contact.exe build/cpu-contact/spur-edge-compact/spur.scene build/chen-scene/spur build/chen-scene/spur.json --method cad-occt --dt .00005 --duration .00035 --event-tol-s .000025
python benchmarks/chen/gear_region_audit.py build/cpu-swept/Release/nc_cad_distance_benchmark.exe build/cpu-contact/spur-edge-compact/spur.scene results/aes/general-contact/spur-local-cad-response.csv build/chen-scene/local-regions.json
python benchmarks/chen/plot_spur_loading.py
~~~

## 2026 labeled triangular-patch queries

For the separately implemented 2026 variant and its results see [Safeguarded envelopes](#safeguarded-envelopes).

The authors' [TDIB-E data release](https://github.com/xw-c/TDIB-E/tree/5012314ca15c01b1e368a102141af845fdf6add3) provides face–face and vertex–vertex cases with rational control points and exact binary labels. Both subsets encode pairs of quadratic triangular Bézier patches with linearly moving control points. The [2026 paper](https://doi.org/10.1145/3811310) introduces floating-point error analysis for its CCD solver. The executable here calls the official **2024** TDIB-OBB solver on the released **2026 data**.

```powershell
git clone https://github.com/xw-c/TDIB-E.git build/deps/tdib-e
git -C build/deps/tdib-e checkout 5012314ca15c01b1e368a102141af845fdf6add3
cmake --build build/chen-common --config Release --target nc_chen_triangles
python benchmarks/chen/tdibe_replay.py build/chen-common/Release/nc_chen_triangles.exe build/deps/tdib-e build/chen-common/tdibe.json --per-file 16 --width .0005 --query-timeout 8
```

The driver selects 16 indices at equal index intervals in each of eight published files. It preserves rational coordinates up to conversion to binary64, constructs velocity from the rounded endpoints and records the resulting maximum coordinate perturbation of each affine trajectory. This perturbation was zero for all 128 selected queries. At a parameter/time cell width of 0.0005 all 64 positive examples returned a candidate, 60 of 64 negative examples returned clear and all queries completed. The four extra candidates were in the face–face near-miss subset. Those four returned candidates again at width 0.00005. A candidate is an inclusion test result. The published labels concern geometric intersection in the entire time interval. The runs do not supply a contact force or a CAD trimming classifier. [Per-query results](../../results/aes/general-contact/chen-2024-on-tdibe-2026.json) · [Tighter-width rerun](../../results/aes/general-contact/tdibe-close-negatives-width-5e-5.json)

## Safeguarded envelopes

`--method chen-2026` selects an independent implementation of the error separation in Section 4 of Chen et al. The OCCT build includes this method. All seven surface response laws use the same scene. The authors' public release supplies the labeled data. Our implementation encloses the projected line coefficients and rounds the time intersection of each pair of lines outward. This changes the paper's sorted-envelope traversal and predicate-specific error estimates. [Paper](https://doi.org/10.1145/3811310) · [Authors' release](https://github.com/xw-c/TDIB-E)

With constant positive weights, a moving Bézier patch projects between the lower and upper envelopes of its moving poles. For the upper envelope of A to reach the lower envelope of B, at least one pair of lines must satisfy that inequality. The solver bounds each pair's possible times, takes their hull and tests both directions on coordinate and locally chosen axes. It uses outward rounding for projected endpoint ordinates, differences and roots. It retains endpoint equality and subdivides the patch in homogeneous space. Exhausting the budget gives an unresolved query. A terminal cell indicates possible contact. The rigid STEP adapter bounds rotation relative to the pole chords, then checks proposals on the actual trimmed solids. CAD conversion, pose construction and OCCT verification remain numerical.

| Normalized cell width | Solver | Candidates on 64 positives | Clear on 64 negatives | Extra candidates | Timed out |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.0005 | Upstream 2024 | 64 | 60 | 4 | 0 |
| 0.0005 | Safeguarded envelopes | 64 | 59 | 5 | 0 |
| 0.00005 | Safeguarded envelopes | 64 | 59 | 5 | 0 |
| 0.0000001 | Upstream 2024 | 63 | 60 | 4 | 1 positive |
| 0.0000001 | Safeguarded envelopes | 64 | 60 | 4 | 0 |

All runs used the same indices, binary64-exact input and eight-second per-query budget. The upstream timeout has no classification. [Default width](../../results/aes/general-contact/tdib-envelopes-on-tdibe-2026.json) · [Intermediate width](../../results/aes/general-contact/tdib-envelopes-width-5e-5.json) · [Fine safeguarded run](../../results/aes/general-contact/tdib-envelopes-width-1e-7.json) · [Fine upstream run](../../results/aes/general-contact/tdib-2024-width-1e-7.json)

~~~powershell
python benchmarks/chen/tdibe_replay.py build/chen-common/Release/nc_chen_triangles.exe build/deps/tdib-e build/chen26/labeled.json --method chen-2026 --per-file 16 --width .0005 --query-timeout 8
python benchmarks/aes/solver_impact.py build/cpu-swept/Release/nc_contact.exe build/chen26/spheres build/chen26/impact.json --method chen-2026
~~~

The [head-on impact](../../results/aes/cpu-solvers/impact-chen-2026.json) exercises all seven laws on one scene. The [eccentric impact](../../results/aes/general-contact/chen26-free-eccentric.json) adds free rotation, friction and offset centers of mass. Patch proposals preceded verified contact in both impacts, so the trimmed-solid search continued. The planar-crossing regression also exercises an accepted CAD proposal. Statistics distinguish visited cells, terminal candidates, unresolved patch searches and confirmed proposals.
