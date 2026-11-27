# GPU CAD segment-query optimization

Two complete reducer runs took **191.53 s and 191.68 s online**, versus the archived four-worker CPU CAD time of **193.63 s** and the preceding GPU time of **264.53 s**. The GPU measurements are only 1.0-1.1% below the archived CPU measurement: this is near parity, not evidence of a substantial GPU advantage. The CPU benchmark was not rerun. Both GPU runs have the same complete response hash.

The experiment uses the frozen two-stage reducer, a 0.1 ms main step, five seconds of physical time, shared penalty stiffness and generalized-alpha integration with spectral radius zero. Preprocessing, graph construction and upload are excluded from online timing; final synchronization and result download are included. Hardware: RTX 3070, Ryzen 9 5900X, CUDA 12.6, driver 610.88. No compilation or other simulation ran concurrently with the full measurements.

| Full 5 s / 50,000 main steps | Online time | Evidence |
|---|---:|---|
| CPU CAD, four workers, archived | 193.63 s | [CPU scaling](../cpu-bezier/standalone/simulation-timing/cpu-worker-scaling.json) |
| Previous GPU CAD, same scene | 264.53 s | [Previous experiment](../cuda-batched-execution/README.md) |
| GPU CAD (ours), run 1 | 191.53 s | [Run](run.json) |
| GPU CAD (ours), run 2 | 191.68 s | [Repeat](repeat.json) |

The new GPU version reduces online time by about 27.6% relative to the preceding GPU version. It remains about 38 times slower than real-time throughput, and all 50,000 main steps miss the 0.1 ms deadline. Whole-graph batched synccheck also remains unresolved. These results establish full-history numerical consistency and lower measured time than the archived CPU reference; they do not establish production readiness or real-time performance. [Acceptance scope](acceptance.json) · [Build and source hashes](build.json)

All 50,001 saved frames were compared. The largest absolute differences from CPU are 5.10e-8 rad/s in shaft speed, 1.31e-5 N in normal load and 7.87e-7 N m in contact torque. Both contact-pair flags and all their transition frames match. The solver accepts 50,044 substeps with 44 subdivisions, matching the trusted reference; the 3.901 s discrepancy of the rejected reciprocal trial is absent. These are consistency errors against an archived calculation, not independent physical-accuracy bounds. [Complete numerical comparison](comparison.json) · [Raw response arrays](response.npz)

[Full response curves](responses.png) · [Response differences](differences.png) · [Timing and phases](timing.png). Curves cover 0-5 s and are not filtered or smoothed. Contact torque is computed by subtracting the scene's prescribed external load from generalized force. Diagnostic short runs below supplement, and never replace, the complete performance/response runs.

## Implementation changes

- `src/compiled_cuda_pack.cpp` embeds the existing span trees in the face-tree preorder. `crossing_warp` can assign separate polynomial spans to separate lanes. The existing boxes, coordinate frames, preorder and first-hit/error rules remain in use.
- The general subdivision path chooses between parameter directions using their projection perpendicular to the ray. Splitting a long direction parallel to the ray made little progress in the transverse root equations. Both children still cover the parent rectangle; enclosure tests and failure limits remain in place.
- Restricting the bicubic uses two one-dimensional cubic Horner shifts instead of recomputing the double binomial expansion for every coefficient. It is the same finite Taylor polynomial, with the original roundoff padding.
- Static-cell tangent thresholds and the three absolute polynomial coefficient sums per span are formed during GPU table preparation, in the original GPU operation order. Isolation reuses the fixed threshold; bicubic restriction reuses the coefficient sums in the unchanged roundoff-padding formula. Device-created subdivision cells calculate their own thresholds after rejection tests. Parameter normalization and derivative scaling retain ordinary FP64 division.
- `NC_CUDA_FMA=ON` enables CUDA multiply-add contraction in this experiment. It does not enable `--use_fast_math`, change FP64 to FP32, or relax geometric/Newton tolerances. Forming each quadrature contribution in one shared device function and using explicit rounded additions preserves the scalar/parallel summation audit. The build option defaults to `OFF` so existing arithmetic controls remain available.

These are geometry/query scheduling changes. There is no tooth-index test, prescribed trajectory, fitted load table replacing contact, online OCCT call, or CPU force evaluation in the resident loop. The existing supported-chart and joint/solver limits still apply; this is not a claim of support for arbitrary rational CAD surfaces.

## Intermediate evidence and rejected attempts

| Full 5 s / 0.1 ms stage | Online time | Evidence |
|---|---:|---|
| Previous batched implementation | 264.53 s | [Frozen experiment](../cuda-batched-execution/README.md) |
| FMA-only trial | 238.19 s | [Record](fma-only.json); initial scalar/parallel audit failed, so this executable was not accepted |
| Span scheduling, corrected sum audit and transverse subdivision | 209.51 s | [Record](transverse.json) |
| Above plus Horner restriction | 194.48 s | [Record](horner.json) |
| Above plus static threshold preparation | 192.36 s | [Record](static-threshold.json) |

The transverse, Horner and static-threshold full runs have the same response SHA-256 as the FMA-only trajectory. The FMA trial's audit failure was traced to multiply-add contraction across contribution formation and scalar accumulation, then corrected without loosening the audit. The [original failure](fma-first-audit-failure.log) remains archived.

An exclusion-only port of the CPU transverse Bernstein clip was also tested. It passed the 500-step diagnostic's connection and reduction checks but increased direct-segment summed work from 0.6735 to 0.9965 s. It was reverted before final timing. Its [record](profile-clip-rejected.json) and [rejected patch](rejected-clip.patch) are retained; no full-duration speed claim is made for it.

A 32-thread face-search trial also preserved the diagnostic response but increased search time from 0.5176 to 0.6777 s. The [trial](profile-search-warp-rejected.json) was rejected and the original 128-thread search retained. Average proposal count alone did not predict its performance.

`--gpu-profile on` records time and call counts for prior-path checks, endpoint predicates, direct segment queries and recursive connection attempts. These are summed work times across concurrently executing connections, not disjoint wall-clock phases. Profiling is disabled in performance runs. The comparable audited diagnostics are [flat](profile-flat.json), [transverse](profile-transverse.json) and [Horner](profile-horner.json); the [earlier un-audited diagnostic](profile-before.json) must not be compared by total wall time to those audited runs.

The bare-reciprocal candidate at 173.22 / 173.60 s was rejected: a changed Newton subdivision near 3.901 s introduced a local load difference of 17.48 N. Its [complete evidence](uncorrected-reciprocal/README.md) remains archived. A rounded reciprocal variant passed 1,048,832 exact division checks and restored the trusted full response, but took 208.95 s. Its [run](rounded-division-rejected.json), [native checks](native-rounded-checks.json) and [rejected code](rejected-rounded-division.patch) are retained. Neither of these arbitrary-width reciprocal variants is in the final source. An exact power-of-two scaling trial also preserved the entire response, but took 197.25 s versus 192.36 s with ordinary division. It was removed; its [run](binary-scaling-rejected.json), [million-case arithmetic checks](native-binary-checks.json) and [patch](rejected-binary-scaling.patch) are retained.

## Geometry and response checks

The [input hashes](input-hashes.json) were rechecked against the same scene and four `.ncpart` files. Their source STEP equivalence record is [unchanged](../cpu-bezier/standalone/part-equivalence.json). No surface or trim data were retessellated for this optimization.

The [device chart audit](part-audit.json) covers 168,948 surface samples and 107,700 trim samples over 840 faces, with no material classification mismatch or undefined sample. Under FMA the maximum sampled position difference from CPU evaluation is 1.468e-14 mm. This sampled audit complements the original control-net bounds; sampling alone is not a proof over every point.

`tests/compiled_cad.cpp` includes independent analytic box displacement/force/release checks, independent contact pairs, disconnected pads, curved/revolved roots, and the supported coordinate limits. The new long-extrusion regression requires two parallel-ray misses to be resolved and a near-parallel crossing to agree with its analytic depth. The native checks retain the exact comparison between cooperative, cached and batched execution and between scalar and ordered region reductions.

The final [same-pose diagnostic](same-pose-audit.json) checked 6,789 connection decisions and 2,406 region evaluations over 500 steps, including 147 entering and 142 exiting face pairs. Scalar and ordered parallel reductions agreed exactly in all three force components and moments about both body origins. [Native checks](native-checks.json) retain the analytic and multiple-pair regressions.

Final [Compute Sanitizer checks](sanitizers.json) report zero errors for batched memcheck, batched initcheck and cached-schedule synccheck. Each uses a 50-step diagnostic that includes the first connection checks. The separate whole-graph batched synccheck issue remains unresolved; these passing checks do not replace it.

The full-history report measures absolute differences in angular speed, normal load and contact torque, compares every contact flag and transition frame, and compares raw adjacent increments to detect added spikes. Curves are not filtered or smoothed. Agreement with an archived solver response is numerical consistency evidence, not an independent claim of higher physical accuracy.

## Reproduction

Build `nc_contact_runtime` and `nc_compiled_tests` with `NC_WITH_CUDA_COLUMNS=ON`, `NC_CUDA_FMA=ON`, CUDA 12.6 and Release configuration. The measured GPU is an RTX 3070 on the same Ryzen 9 5900X host as the archived CPU run. Each experiment has a 1,800 s process timeout.

```text
nc_compiled_tests
nc_compiled_tests --cuda
python benchmarks/aes/cad_gpu_execution.py EXE SCENE RUN --step .0001 --execution batched
python benchmarks/aes/cad_gpu_execution.py EXE SCENE AUDIT --step .0001 --duration .05 --execution batched --reduction audit --profile-links
python benchmarks/aes/report_cad_segments.py BEFORE RUN CPU SCENE REPORT
```

`run.json` records the exact invocation, executable and scene hashes, frame count, full-response hash, online phases and deadline statistics. The source remains on the `gpu/compiled-cad-resident` branch. The original `cooperative` schedule remains the default. The conditional batched schedule is opt-in: its previously documented whole-graph synccheck issue is still [unresolved](../cuda-batched-execution/README.md), including a separate reproduction without CAD code.
