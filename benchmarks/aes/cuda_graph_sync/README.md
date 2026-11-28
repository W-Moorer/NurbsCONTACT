# Conditional-graph synchronization regression

This small program isolates the synchronization schedule from CAD and dynamics.
It runs the same shared-memory exchange in a flat graph, an IF graph or a WHILE
graph, using block barriers or independent warp barriers. The warp tests use
both 32-thread and 128-thread blocks; data never cross warp boundaries.

```text
cmake -S benchmarks/aes/cuda_graph_sync -B build/graph-sync -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build/graph-sync --config Release
python benchmarks/aes/cuda_graph_sync/check.py PROBE COMPUTE_SANITIZER OUTPUT
```

The checker requires clean flat/IF block cases and both WHILE warp cases. It
also requires an intentionally invalid warp mask to be reported as an error.
The mask is read from device memory so the compiler cannot remove the negative
control's barrier by constant folding. Do not run `invalid` mode outside the
sanitizer. The WHILE block case is recorded as a diagnostic, since the purpose
is to retain the trigger, not require future CUDA versions to fail.

These tests complement the **unfiltered production whole-graph** checks. They
cannot establish correctness of the contact solver. Production diagnostics
must additionally run native analytic regressions, memory checks, same-pose
query/force audits and complete response comparisons. A short diagnostic is
not a full-duration sanitizer run. See the [verification record](../../../results/aes/cuda-graph-sync/README.md).

Run the production diagnostics with:

```text
python benchmarks/aes/cuda_graph_sync/production.py RUNTIME NATIVE_TEST SCENE SANITIZER OUTPUT --racecheck RACE_SANITIZER
```

On Windows, `--dll-path` can prepend the dependency DLL directory. The script
records executable hashes, tool versions, commands and exit codes, and fails
on a diagnostic error or a 30-minute timeout. Each production graph check
executes 50 steps with query and force audits enabled. Use the separate full
five-second commands in the verification report for response and timing checks.
`--racecheck` makes any tool-version distinction explicit; no errors are suppressed.
