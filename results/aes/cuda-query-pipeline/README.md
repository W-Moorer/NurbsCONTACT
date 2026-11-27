# GPU CAD query pipeline

This experiment changes the device query schedule and the geometry-only preprocessing of the existing CAD backend. The reducer, force law, integration tolerances, quadrature rules and 0.1 ms main step remain fixed. Complete performance runs cover five seconds and all 50,001 output frames. CPU timings are archived measurements; the CPU benchmark is not rerun.

Two complete runs take **163.39 s and 163.49 s online**, versus the archived **193.63 s CPU CAD** and **191.53 s preceding GPU CAD** measurements. Setup takes 31.36 s and 38.63 s respectively and is excluded. The entire response has the same SHA-256 as the preceding GPU calculation. [Run](run.json) · [Repeat](repeat.json) · [Build and source hashes](build.json) · [Input hashes](input-hashes.json)

| Complete 5 s / 50,000 main steps | Online time | Source |
|---|---:|---|
| CPU CAD, four workers, archived | 193.63 s | [Archived CPU measurement](../cpu-bezier/standalone/simulation-timing/cpu-worker-scaling.json) |
| Previous GPU CAD, archived | 191.53 s | [Previous report](../cuda-segment-queries/README.md) |
| GPU CAD, original schedule in the new executable | 192.30 s | [Paired control](control.json) |
| GPU CAD (ours), first run | 163.39 s | [Run](run.json) |
| GPU CAD (ours), repeat | 163.49 s | [Repeat](repeat.json) |

The paired control uses the same binary and inputs. The measured reduction is 15.0% against that control and 15.6% against the archived CPU CAD time in both runs, rounded to one decimal place. The two new GPU runs differ by 0.10 s. This is a reproducible comparison, not a claim of a speedup across arbitrary mechanisms. The present throughput remains about 33 times slower than real time; all 50,000 main steps miss the 0.1 ms deadline.

## Measured stages

| Stage | Original schedule, same binary | New schedule | Time reduction |
|---|---:|---:|---:|
| Broad phase | 13.90 s | 13.89 s | 0.0% |
| Candidate sorting | 6.90 s | 6.91 s | -0.2% |
| Candidate confirmation | 49.43 s | 42.67 s | 13.7% |
| Region connectivity | 42.26 s | 28.83 s | 31.8% |
| Region integration | 54.06 s | 45.31 s | 16.2% |

Candidate counts, selected face counts, 105,669,028 integration rays, 185,780 force evaluations and 44 subdivisions match the old and paired controls. The precomputed query handles 105,667,750 rays; 1,278 defer to the complete GPU solver. The gain comes from executing the same work more efficiently. [Work counters](work-counts.json)

The five-level enclosure version completed the same full history in 169.73 s. Increasing the enclosure depth to seven reduced connectivity work further, with the complete response unchanged. [Five-level full run](depth5-full.json) · [Short diagnostics and removed trials](diagnostics.json). Short diagnostics are not used as full-run acceptance evidence.

## Implementation

Use `--gpu-execution wavefront` with a Release build and `NC_CUDA_FMA=ON`. `batched` remains available as a control; the default cooperative schedule is unchanged.

1. Prepare a second, padded control-net enclosure hierarchy for polynomial bicubic spans, up to seven binary levels (255 nodes) with early stopping for nearly linear spans. Its split direction follows second differences rather than physical length. This improves exclusion for rays grazing long, nearly linear surfaces. It only rejects impossible intersections: it does not replace the source surface, root equations, trims or normals with a mesh. Unsupported spans bypass this filter.
2. Evaluate previous seeds and independent candidate projections in device queues across face groups and body pairs. Consume the results in the original proposal order, including the original error and tie rules. Keep local adaptive refinement within a warp; splitting every refinement round into separate kernels was slower.
3. Dispatch quadrature rays through a common query that uses the existing precomputed cells. A ray needing adaptive subdivision, nonrectangular trim handling or analytic-surface handling is deferred to the complete GPU query. That query starts from the original ray and reach, preserving nearest-root and ambiguity handling. No unresolved query is treated as separation. Scatter results to their original column IDs and retain the original FP64 force/moment accumulation order.

Small query batches use fewer active lanes per block to spread independent work over the device. Scheduling depends on queue length and device SM count, not tooth count, a trajectory, a prescribed load map or a mechanism identifier. All online queries, region construction, force evaluation and integration remain on GPU. There is no online OCCT or CPU geometry fallback.

The four `.ncpart` files are unchanged. The existing [STEP-to-runtime equivalence audit](../cpu-bezier/standalone/part-equivalence.json) therefore continues to describe the source charts. The new hierarchy is an auxiliary exclusion structure built from those control nets, not a new fitted geometry. Device sample checks complement that source audit; sampled agreement alone is not a continuous equivalence proof.

## Verification and scope

The full-history comparison records every contact flag and transition frame, angular speed, normal load, contact torque and adjacent response increment. Curves cover 0-5 s without filtering or smoothing. Agreement with the archived CPU/GPU response is numerical consistency evidence, not independent proof of greater physical accuracy.[Full response curves](responses.png) · [Differences](differences.png) · [Timings](timing.png) · [Numerical comparison](comparison.json) · [Raw arrays](response.npz) · [Acceptance scope](acceptance.json). The largest differences from the archived CPU response are 5.10e-8 rad/s in speed, 1.31e-5 N in load and 7.87e-7 N m in torque. All response values match the preceding GPU result exactly.[Native CPU/CUDA checks](native-checks.json) pass. The [chart audit](part-audit.jsonl) covers 168,948 surface samples and 107,700 trim samples without classification failures; the maximum sampled position discrepancy is 1.468e-14 mm. [Memcheck and initcheck](sanitizers.json) both report zero errors for their 50-step diagnostics. These are diagnostic memory checks, not full-duration sanitizer runs.

Native regressions cover analytic box contact and release, independent body pairs, disconnected pads, a physical plane partitioned into multiple CAD faces, curved roots, near-tangent rays and the supported coordinate limits. Diagnostic audit mode compares queued candidate selection against the original search and fast resolved rays against the complete query at the same poses. Both searches in that audit share the new enclosure filter; the old executable's full response and separate unfiltered analytic ray checks provide the complementary controls.The final [same-pose audit](same-pose-audit.json) checks 963,668 resolved rays, 16,394 candidate groups and 6,789 connection decisions. Its [region records](same-pose-regions.csv) retain exact agreement between scalar and ordered parallel force/moment evaluation in every component. [Summary](same-pose-summary.json). The diagnostic response also has the same hash as its original unmodified control.

The backend retains its existing supported-chart and solver limits: up to 16 scalar hinge/prismatic coordinates and frictionless penalty regions. Arbitrary rational CAD and all complementarity solvers are not implemented by this change. Whole-graph conditional-schedule synccheck remains [unresolved](../cuda-batched-execution/README.md). Native, memory and numerical checks are separate evidence. Real-time performance remains unachieved.

## Rejected scheduling experiment

An extra graph branch intended to bypass geometry when all contact regions could be reused was removed. It triggered zero times in the reducer diagnostic. Inspection of `correct_step` showed that covered Newton trial poses already execute inside the integrator; only trials requiring geometry return to the phase graph. The diagnostic contradicted the initial empty-kernel hypothesis. No performance gain is attributed to that branch.

## ReproductionUse the [frozen compiled inputs](../cpu-bezier/standalone/reducer-inputs.zip) and the existing source-equivalence records linked above.

```text
nc_compiled_tests
nc_compiled_tests --cuda
python benchmarks/aes/cad_gpu_execution.py EXE SCENE RUN --step .0001 --execution wavefront
python benchmarks/aes/cad_gpu_execution.py EXE SCENE AUDIT --step .0001 --duration .05 --execution wavefront --reduction audit
python benchmarks/aes/report_cad_segments.py BEFORE RUN CPU SCENE REPORT --require-identical
```

Preprocessing, graph construction and upload are outside online timing. Final synchronization and result download are included. Each process is limited to 1,800 seconds. Full timing runs execute individually without another simulation or compilation. Hardware: RTX 3070, Ryzen 9 5900X, CUDA 12.6, driver 610.88.Last verified: 2026-11-28. Release native CPU/CUDA checks, chart audit, two complete new-schedule runs, a paired original-schedule run, same-pose audit, memcheck and initcheck passed. Python generated all figures; previews were inspected. The repository-local wiki-audit script is absent, so new report links and UTF-8 encoding were checked directly. Whole-graph synccheck remains open. `AGENTS.md` is unchanged.
