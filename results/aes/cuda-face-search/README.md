# GPU face confirmation and contact-region grouping

The comparison uses the four compiled solids of the two-stage reducer on an RTX 3070. Each executable starts from the same initial state and advances 50,000 steps of 0.1 ms. Online time includes the complete five-second closed loop. Model preparation, allocation and separate diagnostic runs are excluded. All variants use FP64 geometry, ordered accumulation, the shared penalty law, predictor geometry updates and the same implicit solver settings.

## Changes under test

1. At model setup, evaluate the position and the two surface derivatives at each material tile centre using the GPU evaluator. The first Newton iteration of every face projection reads this immutable value. Subsequent iterations evaluate the original CAD chart at their updated parameters.
2. Give independent endpoint crossing tests of a candidate connection to separate lanes of the same warp. If those tests leave the line unobstructed, check its visibility in both solids. The first blocking predicate and any preceding fault have the same priority as in the serial implementation. A blocked direct link can still be joined through the existing recursive path search.
3. Prepare the reciprocal direction of a ray once per coordinate frame while traversing its bounding boxes. This changes the cost of repeated slab tests in both preceding stages.

## Five-second measurements

| Cumulative implementation | Revision | Face confirmation (s) | Region grouping (s) | Region integration (s) | Total online (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| Previous ordered GPU backend | `8af8ca2` | 170.85 | 99.94 | 85.77 | 436.32 |
| Cache the material tile centre and derivatives | `7902a52` | 144.18 | 99.80 | 85.75 | 409.84 |
| Also evaluate direct-link predicates in parallel | `17fff9a` | 143.43 | 95.31 | 85.80 | 404.70 |
| Also reuse ray reciprocals | `1b890b4` | 143.10 | 95.50 | 83.61 | 402.41 |

The tile cache saved 26.67 s in face confirmation. Parallel direct-link checks saved another 4.49 s in grouping relative to the tile-only version. Reusing reciprocals saved 2.29 s overall in the third comparison, including 2.19 s in region integration. The four independently integrated response files matched byte for byte. Each run recorded 101,649 geometry queries, 185,760 force evaluations and 44 HHT subdivisions.

Two runs of the final binary (`4f710ff`) with diagnostics disabled took 402.776 s and 402.768 s. The earlier binary (`8af8ca2`) took 436.325 s and 435.534 s in the same session. The mean saving was 33.16 s (7.61%). Both final runs missed all 50,000 deadlines of 0.1 ms. The separate scalar link-and-wrench audit took 548.05 s.

## Contact and load verification

The final implementation (`4f710ff`) checked 652,575 proposed links against the scalar proof at their current poses. All connectivity decisions and accepted path points agreed. The per-step, per-stage region audit had the same SHA-256 as the archived region audit, including counts of confirmed face pairs entering and leaving support and loaded-region counts. It covered 239,624 region comparisons, including 114,709 at changes of confirmed faces or their immediately adjacent steps. All component differences between serial and parallel accumulation were zero for world force and moments about each body's own origin. The full independent dynamics files include both stage loads and the output contact torque. The initial torque peak occurs at 0.5 ms and measures 432.63084 N m.

[Performance and response curves](response.png) · [Validation, hashes and timings](comparison.json) · [Reference per-step region data](../cuda-region-reduction/region-audit.npz)

## CPU and GPU backends

The CPU rows use the frozen four-worker Ryzen 9 5900X measurements. Both corresponding CPU response hashes match the archive. The GPU rows use the RTX 3070 on the same workstation. All four advance the mechanism from its initial state through 5 s; the geometry and time integration details differ between the two mesh implementations.

| Device and geometry | Force construction | Online seconds |
| --- | --- | ---: |
| CPU, mesh OBB | Intersection contours and material-exit rays | 222.02 |
| CPU, compiled CAD | CAD regions with distributed penalty integration | 193.63 |
| GPU, sampled mesh | Bidirectional surface probes, per-pair stiffness, 16 contact subdivisions | 128.40 |
| GPU, compiled CAD (ours) | CAD regions with distributed penalty integration | 402.77 |

The sampled GPU mesh reproduced its archived 133.33 s run's entire response array. It performed 1,603,242 dynamics trials; the CPU mesh took 151,547 force evaluations. On CAD, CPU and GPU took respectively 101,648 and 101,649 geometry queries and both subdivided 44 main steps. On the GPU, face confirmation, grouping and region integration took approximately 143.33, 95.48 and 83.62 s. Together these account for 80.1% of online time. The implementation advances both body pairs through a single shared geometry workspace. The resident cooperative grid launches at most one 64-thread block per SM. On this RTX 3070 that is two scheduled warps per SM for the grid; timing data do not apportion the cost between synchronization, memory traffic and arithmetic.

The historical fixed-hinge GPU implementation precomputes loads and moments on a two-angle grid and recorded 2.19 s for a complete closed loop. The generic GPU triangle BVH benchmark takes prescribed poses and reports original intersecting triangle IDs. It measures contact discovery. A GPU triangle-BVH closed loop with the CPU contour-ray force construction and the same implicit settings is still unmeasured. [Frozen CPU runs](../cpu-bezier/standalone/simulation-timing/cpu-worker-scaling.json) · [Fixed-hinge comparison](../gpu-hinge-mesh/README.md) · [Generic GPU mesh geometry](../gpu-mesh-geometry/README.md)

The three GPU BVH replays took 7.039, 6.835 and 6.843 s. The median corresponds to 136.87 µs per pose for both body pairs. A separate instrumented run exceeded 100 µs on 49,802 of the 50,000 steps. Its triangle hit-count history matched the original FP64-validated replay.

Across the four parts, 37,244 identical 0.4 mm material-exit rays took 2.26–3.56 times longer per batch on compiled CAD than on the triangle BVH and 6.52–11.76 times longer than on the bounded-ray triangle grid. Each batch contained at most 128 rays. The two mesh indices agreed on exit classification; CAD and mesh differed on 76 queries. The sampled-mesh closed loop's peak output contact torque was 580.02 N m versus 432.63 N m on compiled CAD, with an RMS difference of 32.36 N m on the common time grid. These timings compare the implemented methods; matched dynamical accuracy has not been established. [Backend timings and matched local queries](backend-costs.png) · [Measurements and reference hashes](backend-costs.json)

## Reproduction

Build each indicated revision in Release mode with `NC_WITH_CUDA_COLUMNS=ON`. Pass the same frozen scene and four compiled parts to each executable. Run one process on the GPU at a time. The asset hashes are recorded in the [original resident run](../cuda-compiled-resident/run.json).

```powershell
$options = @('--method','cad-compiled','--device','cuda-resident','--solver','penalty','--penalty-stiffness','shared','--geometry-update','predictor','--spectral-radius','0')
& $executable $scene 0 0.0001 5 1 "$directory/response" @options --region-reduction ordered
& $selected $scene 0 0.0001 5 1 "$auditDirectory/response" @options --region-reduction audit
```

For the link check, audit mode recomputes each proposed connection in the original scalar order on the same two witnesses. It checks the decision, fault identity and all coordinates in an accepted path. The `checked_links` column in `.gpu.csv` counts these comparisons. The diagnostic reserves a larger CUDA thread stack for its additional caller above recursive surface subdivision. The moving-box GPU regression and the first 50 reducer steps under Compute Sanitizer exercise the setup and the first nonempty connection batch.

```powershell
& $tests --cuda
compute-sanitizer --tool memcheck --error-exitcode 47 $selected $scene 0 0.0001 0.005 1 "$auditDirectory/short" @options --region-reduction audit
```

The report reads the four production ablations from `search-before-full`, `search-jet-full`, `search-bridge-full` and `search-slabs-full` under the runs directory. Use `search-before-repeat` for the second baseline and `search-links-final`, `search-winner-repeat` and `search-links-audit` for the two final production runs and the final independent audit. Place the corresponding executables in `workspace-bin`, `jet-bin`, `bridge-bin`, `slabs-bin` and `search-checked-bin`. Pass the original ordered-reduction audit directory separately.

```powershell
python benchmarks/aes/compare_gpu_face_search.py $scene $runs $beforeAudit results/aes/cuda-face-search
```

To reproduce the cross-backend timings, run the GPU jobs individually and supply the frozen CPU CSV files for poses and reference responses. The CPU solver is not invoked.

```powershell
python benchmarks/aes/resident_assembly.py $meshDeck "$cross/mesh-dynamics.json" --trace "$cross/mesh-dynamics.npy" --method mesh --duration 5 --step-us 100 --broad-phase blocks --guard-mm 0.8 --stiffness-scope pair --ticks 4096 --contact-samples 16 --explicit-crossing --profile-phases
python benchmarks/aes/profile_gpu_mesh_queries.py $scene $cpuMesh "$cross/mesh-bvh.json" --leaf-size 4 --repeats 3 --profile-all
python benchmarks/aes/profile_gpu_exit_queries.py $scene $cpuMesh $rayExecutable "$cross/rays" --reach 0.4
python benchmarks/aes/compare_reducer_backend_costs.py $scene $cpuMesh $cpuCAD $gpuCAD $cross results/aes/cuda-face-search
```
