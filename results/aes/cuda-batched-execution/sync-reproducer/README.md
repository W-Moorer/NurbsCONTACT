# Isolated CUDA graph synchronization diagnostic

This program contains no contact or CAD implementation. Build with CUDA 12.6,
C++17, device linking and SM86 (see CMakeLists.txt). With MSVC 14.44 the test
configuration used `--allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH`.

`batch_sync_probe.exe 4` completes natively with result 16843011 (the initial
0x01010101 integer plus two increments). With Compute Sanitizer 13.5.25:

```
compute-sanitizer --tool synccheck --error-exitcode 47 --print-limit 2 batch_sync_probe.exe 4
```

it reports divergence at the first unconditional barrier in `work` and then
an unknown device error. Blocking-launch mode reproduces the report too. This
reproduces the local environment issue; it does not prove all application
synchronization is correct. The logs retain both the native and instrumented
outcomes. The GPU was an RTX 3070 with driver 610.88 on Windows.
