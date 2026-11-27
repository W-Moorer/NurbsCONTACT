# Batched compiled-CAD execution on CUDA

This experiment changes execution and BVH traversal, retaining the source CAD,
contact law, material quadrature and ordered FP64 reduction. It is an opt-in
`--gpu-execution batched` schedule. CPU results are archived references, not new runs.

## Implementation

Each contact pair owns its mutable workspace. Candidate windows and face searches
are scheduled across pairs. Valid integration rays enter a compact device queue;
their answers scatter to their original column IDs before the existing ordered
reduction. Conditional CUDA graphs keep step acceptance, subdivision, Newton
iterations and geometry refresh on the GPU. The CPU uploads once and reads the
completed trajectory; there is no online OCCT call or per-step force exchange.

The second experiment restores subtree skipping to the warp's preorder BVH
traversal. Prefix maxima of rejected-node end indices exclude descendants within
a 32-node window and jump over descendants beyond that window. First-hit and
first-error ordering are retained. Integration rays use 32-thread blocks to
spread small queues across more multiprocessors. Neither operation uses gear
dimensions, tooth identities or recorded motion.

Sources: [phase scheduling](../../../src/cuda_batched_contact.cuh),
[CAD traversal](../../../src/cuda_cad_roots.cuh),
[workspace ownership](../../../src/cuda_columns.cu).

## First complete comparison

The same executable (`e1137d0f...`) ran each mode separately on the local RTX 3070.
All runs advance the same reducer for **5 s at a 0.1 ms main step**. Setup,
compilation and preprocessing are excluded. No other GPU experiment or CPU build
ran during these measurements. The runner enforces a 1,800 s process timeout.

| Schedule | Online time | Candidate search | Region connections | Region integration |
| --- | ---: | ---: | ---: | ---: |
| Cached cooperative control | 358.00 s | 105.31 s | 94.28 s | 78.79 s |
| First batched schedule | 287.65 s | 59.35 s | 91.50 s | 87.23 s |

The first schedule saves 19.65% overall but does not outperform the 193.63 s CPU
CAD reference. It completes 50,000 main steps with 44 subdivisions. All 50,001
saved frames have the same complete CSV SHA-256 as the frozen GPU result:
`0b3c1efeb81f86b7c60b380b74b63f3357a1d097b75a5dbe8a570d867ed420bb`.
The separate full 5 s / 1 ms run takes 63.04 s and also reproduces its original
response exactly. This larger step is a regression case, not the main acceptance.

Raw records: [control](control-fine.json), [first batch](batch-fine.json),
[1 ms regression](batch-1ms.json).

## Pruning and smaller ray blocks

The updated batched schedule completes 5 s / 0.1 ms in **264.53 s**. A fresh
cached cooperative control from the **same executable** takes **358.61 s**,
so the reduction is **26.23%**. The initial 358.00 s control is retained above
as a separate historical measurement. The complete response SHA-256 is unchanged.
The 5 s / 1 ms regression takes 57.49 s and also matches its frozen response.
Both production runs retain the exact loads, contact flags and all coordinate
channels. They do not establish new physical accuracy beyond the existing model.

| Final batched stage | Time |
| --- | ---: |
| Candidate windows and generation | 14.92 s |
| Candidate ordering | 6.90 s |
| CAD face confirmation | 59.36 s |
| Connected-region construction | 91.32 s |
| Region integration | 64.24 s |
| Remaining time, including force/integrator/control and final downloads | 27.79 s |

The added BVH pruning has negligible measured effect on connection time
(91.50 to 91.32 s). Smaller integration workgroups reduce the integration stage
from 87.23 to 64.24 s. Thus extra traversal pruning alone does not explain or
resolve the remaining bottleneck. The kernel resource measurements and small
per-trial workload motivate finer-grained query work; achieved occupancy and
the costs of recursive versus direct connection checks are not yet measured.

**The CPU-speed and real-time targets are not met.** The frozen CPU CAD time is
193.63 s; the new GPU run remains 36.62% slower. Its median main step is 4.77 ms,
P99 11.46 ms, and every one of the 50,000 steps exceeds the 0.1 ms deadline.
The archived CPU mesh and GPU mesh times are 222.02 s and 91.99 s, respectively.
Their contact-force construction differs, so these are backend cost references,
not an equal-accuracy comparison. No CPU complete simulation was rerun.

[Final 0.1 ms record](pruned-fine.json)  |  [Final control](pruned-cached-fine.json)  |  [Final 1 ms record](pruned-1ms.json)

[Full unfiltered responses](responses.png)  |  [Timing comparison](timing.png)  |
[All channel differences and provenance](comparison.json)  |  [Numeric response](response.npz)

## Diagnostics

The short Nsight Systems trace is for hotspot diagnosis only. Its kernel totals
include setup and profiler effects; use the complete-run online timers above
for performance comparisons. [Kernel durations](batch-stats_cuda_gpu_kern_sum.csv)
and [host CUDA calls](batch-stats_cuda_api_sum.csv) record one graph launch.
[Compiled resource usage](kernel-resources.json) shows 255 registers per thread
for the search, connection and ray-query kernels. This is a resource limit,
not a measurement of achieved occupancy. Nsight Compute hardware counters are
unavailable under the current driver permissions.

Whole-record copies initially read `Hit` alignment padding. Initializing witness
storage during setup removes these initcheck reports without changing results.
The first batched version passes memcheck, initcheck and native CPU/CUDA tests.
Audit mode checks connection decisions and paths against scalar traversal, and
ordered force/moment sums against serial accumulation.

**Whole-graph synccheck remains unresolved.** It reports divergence at the first
unconditional block barrier and can then terminate with an internal sanitizer
error. A separate program without any CAD code reproduces the report with both
Compute Sanitizer 13.0.85 and 13.5.25. This is evidence of a graph/tool interaction,
not a passing synchronization check or a proof that every production kernel is
race-free. The native reproducer returns the expected result. The public default
schedule is unchanged.

See [check commands and executable hashes](sanitizers-uniform.json),
[memory log](first-memcheck.log), [initialization log](first-initcheck.log),
[initial synchronization log](first-synccheck.log), [final batch check](batched-synccheck.json) and [final log](batched-synccheck.log), and [minimal reproducer](sync-reproducer/main.cu).
The second experiment passes native CPU/CUDA regressions and batched memcheck/
initcheck. The updated warp traversal also passes whole-run synccheck in the
cached cooperative schedule. This last result is not a synccheck pass for the
conditional batched schedule. [Check commands](sanitizers-pruned.json)

The same-pose audit compares 680 connection checks and 327 reductions, including
face-entry and face-exit events. All force components and both bodies' three-axis
moment errors are zero. [Audit summary](same-pose-audit.json)  |  [Per-frame audit](same-pose-regions.csv)

## Scope and reproduction

The model remains the existing penalty/contact-region approximation. Preserving
its response does not establish convergence, full NURBS support or real-time
performance. The supported analytic/polynomial charts, bounded workspace and
axis-coordinate solver restrictions still apply.

The scene and four `.ncpart` inputs were rehashed in this session and are
[unchanged](input-hashes.json). Source equivalence is documented in the
[STEP audit](../cpu-bezier/standalone/part-equivalence.json) and
[GPU chart/trim audit](../cuda-compiled-resident/part-audit.json).
Use the [input deck](../cpu-bezier/standalone/reducer-inputs.zip).

```powershell
cmake --build $build --config Release --target nc_contact_runtime nc_compiled_tests --parallel 12
& "$build/Release/nc_compiled_tests.exe" --cuda
python benchmarks/aes/cad_gpu_execution.py $exe $scene $out --step .0001 --execution batched
python benchmarks/aes/report_cad_batched.py $before $after $archived_cpu $scene $report
```

The build was tested with CUDA 12.6 on Windows and requires conditional graph
support for this optional schedule. Figures use Python and unfiltered responses.

Last checked: 2026-11-28. No manuscript or AGENTS.md changes are part of this work.
