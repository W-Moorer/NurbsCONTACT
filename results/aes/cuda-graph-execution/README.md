# Compiled CAD GPU optimization experiments

**Performance acceptance is not met.** The verified CPU reference completes the
same reducer, 5 s / 0.1 ms, in 193.63 s. The cached-schedule result archived here is
357.28 s on the local RTX 3070. Later batching experiments are tracked
[separately](../cuda-batched-execution/README.md). CPU measurements are reused; they were not rerun.
This directory preserves failed hypotheses as well as verified changes.

## Current runtime

`--gpu-execution cooperative` retains the original captured cooperative solver.
`--gpu-execution cached` assigns a 128-thread block to each candidate face pair,
spreads independent region links across blocks, and distributes quadrature warps
across blocks without changing their column indices. Each face search retains
the first accepted proposal or fault in the original ordering. FP64 integration
sums retain their original order. The integrator clears and copies only active
coordinates. One graph launch advances all steps and Newton iterations on device;
there is no per-step CPU pose or force exchange and no online OCCT fallback.

The cached path also revalidates blocking-face roots, following the existing CPU
strategy. This cache had **zero hits** in the complete reducer runs and does not
explain the measured speed improvement. A separate disconnected-pad regression
exercises successful cache reuse and compares every state with the uncached
solver. Fixed 48/96 material quadrature is prepared before online timing; the
variable local rule remains online. This extra preparation had negligible
measured benefit on this reducer. It uses 56 bytes per cached point, or 645,120
bytes per face for both orders.

Long segment queries used in connectivity checks distribute consecutive BVH
leaves across a warp and preserve the first hit/fault in BVH preorder. Short
segments retain the original spatial-grid query. Scalar connection checks remain
available in audit mode, including all accepted path points.

A conditional-graph schedule was tried first. Native complete responses agreed,
but Compute Sanitizer memory/synchronization checks failed on that schedule.
It was removed from the runtime API and retained only as a
[rejected source artifact](rejected-conditional-graph.cuh). Its 365.90 s result
is not an accepted executable. [Diagnostic records](safety-investigation.json)

## Complete simulations

All production runs use the same four compiled parts, loads, shared penalty
stiffness, predictor geometry and generalized-alpha integration at spectral radius zero. Setup, compilation,
quadrature preparation and graph upload are excluded from online time. The CAD
online timer includes final status/response/timing downloads. The archived Python
GPU mesh timer excludes its final response download. No run exceeds 1,800 s.

| Full 5 s experiment | 1 ms main step | 0.1 ms main step |
| --- | ---: | ---: |
| Frozen original GPU CAD executable | 78.14 s | 402.78 s |
| Conditional graph, rejected on safety checks | 74.72 s | 365.90 s |
| Cooperative block search and blocking-face cache | 76.65 s | 371.36 s |
| Additional fixed material rules | 76.39 s | not run |
| Distribute links and integration warps | 74.02 s | 358.85 s |
| Parallel long-segment face queries | 73.31 s | 357.28 s |
| Archived GPU mesh | 21.84 s | 91.99 s |
| Archived CPU compiled CAD, four workers | not rerun | 193.63 s |
| Archived CPU mesh, four workers | not rerun | 222.02 s |

The mesh contour-ray force construction differs from CAD region integration;
this is a backend cost comparison, not an assertion of equal dynamics accuracy.
CPU phase counters overlap parallel work and must not be summed or treated as
exclusive equivalents of the GPU stage counters. The first cached 0.1 ms trial
overlapped a CPU compilation during part of its run; the selected long-segment
trial ran without another GPU workload or CPU compilation during its online phase.

The selected GPU run remains slower than CPU and is not real time. Its complete
0.1 ms response has the same SHA-256 as the original GPU reference:
`0b3c1efeb81f86b7c60b380b74b63f3357a1d097b75a5dbe8a570d867ed420bb`.
The 1 ms response SHA-256 is
`21e94e4436e6999fda82d11388195535ab9764daad427cbbcb6880a0530b4ce0`.
All states, forces, gaps and contact flags agree, including contact switches and
load peaks. This does not establish time-step convergence: the 1 ms run needs
773 subdivisions, versus 44 at 0.1 ms. The accepted-level histogram was available
only in the removed conditional schedule; current histogram columns are blank.

[Full 0.1 ms responses](responses-0.1ms.png) · [Archived CPU comparison](responses-cpu.png)
· [Full 1 ms responses](responses.png) · [Ablation timings](execution.png)
· [Measurements, hashes and differences](comparison.json) · [Performance audit](performance-audit.json)

## Verification and reproduction

The native regression checks analytic box contact and release, disconnected
contact pads, and independent constant-acceleration coordinates at 3 and 16
coordinates. Complete reducer traces check end-to-end response preservation.
Compute Sanitizer memcheck and synccheck pass on the 50-step diagnostic for the
cached cooperative schedule. Audit mode compares connection decisions and path
points with scalar traversal, and force/moment reduction with the serial sum.
Diagnostic timings are not used as production timings.

Use the scene/parts from the [CPU input deck](../cpu-bezier/standalone/reducer-inputs.zip).
The [STEP equivalence audit](../cpu-bezier/standalone/part-equivalence.json) and
[GPU chart/trim audit](../cuda-compiled-resident/part-audit.json) refer to the same
unchanged assets; [hashes were rechecked](input-hashes.json). This work does not
extend the supported chart types or resolve the documented narrow-region cases.

```powershell
cmake --build $build --config Release --target nc_contact_runtime nc_compiled_tests --parallel 12
& "$build/Release/nc_compiled_tests.exe" --cuda
python benchmarks/aes/cad_gpu_execution.py $exe $scene "$runs/final-1ms" --step .001 --execution cached
python benchmarks/aes/cad_gpu_execution.py $exe $scene "$runs/final-fine" --step .0001 --execution cached
python benchmarks/aes/report_cad_gpu_execution.py $runs final-1ms results/aes/cuda-graph-execution --fine "$runs/final-fine" --fine-before "$original/search-links-final" --cpu $archived_cpu
```

Direction-based rejection was subsequently checked against the original search
and completed the same 0.1 ms response in 357.01 s. The later pair-owned batching
comparison includes it in both modes; see the [subsequent experiment](../cuda-batched-execution/README.md).
No real-time acceptance has been granted.
