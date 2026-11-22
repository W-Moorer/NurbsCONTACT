# Live gear contact and real-time test

The test asks whether one simulation of two seconds of physical motion can finish in two seconds of wall time. Both 24/48-tooth spur gears start at the poses and speeds from the existing 1,800 rpm scene. Each trial state in the nonlinear solve supplies the next rigid transform. No measured trajectory, tooth index, line of action, face ID or quadrature position is supplied to the contact search.

The earlier CPU comparison in `nc_gears` derives the contact normal from the pressure angle, selects tooth indices from base pitch and rotation, and casts rays at selected flank stations. Both its CAD and triangle variants share those choices. Its HHT integrator and the underlying surface and triangle intersection routines can be used by other models. [Gear force evaluation](../../../benchmarks/gears/main.cpp) · [HHT](../../../src/hht.cpp)

## What runs where

1. A persistent CPU process imports the two closed STEP solids. With `--gpu-screen` it exports its conservative face boxes once. The existing GPU broad-phase kernel transforms B's boxes at every trial pose and sends the surviving indices to the OCCT narrow phase. The CPU-tree control enumerates the initial candidates on the host. Both then use the same transformed tight bounds, reusable separation tests, face-pair distance queries and trimmed straight-line construction. The resulting Gauss points move with the computed state.
2. The CAD variant projects each Gauss point on B onto A's fitted surface. The mesh variant searches all 70,960 triangles of A. Both use the GPU kernels from the resident replay, the same quadrature on B, 55 µm activation and the same line stiffness of 10⁹ N/m². This mesh control shares the CPU contact-line finder with CAD. A fully meshed engine that generates its own manifold would be a different comparison.
3. The GPU sums the wrenches, forms both shaft torques and advances the HHT equations. The two-by-two solve, residual and state updates run in CuPy. The host controls the Newton loop and calls the STEP worker on every trial. The fitted surface and the triangle BVH use the previous query as an initial guess. The prescribed loads, inertias, bearings and initial state come from the existing `nc-contact-v2` spur deck. The line stiffness is from the GPU line-contact experiment. The older `nc_gears` benchmark has a different force parameterization and uses a tooth-specific search.

## Timing and checks

Setup and the initial force at time zero have separate timings. The online clock starts after device synchronization and covers the feedback loop, including host/device transfers, live region discovery, force evaluation, convergence checks and accepted state updates. `region_s` measures only the CPU face-pair query. `assembly_s` measures the subsequent line construction. The server enforces a 30 minute timeout per run.

The short probe stops after the first accepted step that has already consumed the full two-second wall budget. The longer probe continues to 1.4 ms of physical motion, slightly more than one nominal tooth pitch at this input speed. Both report the physical time actually reached. The step size stays fixed, with HHT parameter α = −0.1 and absolute and relative residual tolerances of 10⁻⁷. The diagnostic reports unresolved regions and zero-distance witnesses as unsupported. The line model evaluated here covers straight contact lines with opposing face interiors.

Each geometry result is compared against direct CPU force accumulation at the original STEP witnesses from the same trial. The CPU reference independently solves the first HHT step using those witnesses and a SciPy root solve. It checks the angular response of this numerical model. The benchmark provides no experimental gear-load measurements.

## Measured result

On the Core i7-14700K / RTX 3090 server the very first accepted 100 µs CAD step using the CPU face tree took 6.27 s, after initialization. During the longer observation the CAD route discovered face pairs (4, 4) and (142, 10) from the full bodies, first engaging the second pair and then releasing the first. The input shaft advanced 1.009 tooth pitches. The two-second real-time target was already unattainable before the end of these observations. Each entry below is one measured online run. Times exclude setup and the initial force and cover only the physical interval actually completed.

| CPU face tree, 1.4 ms of motion | Steps | Force evaluations after initialization | Wall time (s) | In CPU face-pair query |
|---|---:|---:|---:|---:|
| Fitted CAD (ours), 100 µs | 14 | 28 | 92.16 | 99.64% |
| Whole-A mesh, 100 µs | 14 | 44 | 143.30 | 99.89% |
| Fitted CAD (ours), 50 µs | 28 | 56 | 183.01 | 99.90% |
| Whole-A mesh, 50 µs | 28 | 82 | 266.02 | 99.90% |

With GPU box screening at 100 µs, fitted CAD took 112.90 s for the same 14 steps. Mesh took 143.64 s. Across the live queries, the GPU supplied between 94 and 204 face pairs from 42,777 possible combinations. Both broad phases produced identical accepted states and active face pairs at this step size. They triggered the same number of OCCT face distance evaluations after initialization, 2,471 for fitted CAD and 3,851 for mesh. The CPU query took 99.90% of online wall time on both GPU-screened runs. The timings come from separate runs, so the matched query counts are the direct measure of work eliminated by broad-phase substitution.

At 100 µs the largest absolute first-step angle difference across the two shafts against the independent CPU root solve was 6.12 × 10⁻¹³ rad for fitted CAD and 3.27 × 10⁻⁸ rad for the mesh. The CPU solve reached a residual of 1.79 × 10⁻¹² Nm. These are checks at the first accepted state.

## Reproduce

From the repository root build the persistent OCCT worker and run the following commands with CUDA, CuPy, SciPy and the OCCT runtime available. Shown for the 100 µs step on Linux.

```sh
cmake --build build/occt --config Release --target nc_live_lines
worker=build/occt/nc_live_lines
models=results/gears/models
fits=results/aes/gear-field/faces.npz
mesh=results/aes/mesh-comparison/meshes/mesh-0.04.npz
out=results/aes/online-gear
timeout 1800 python benchmarks/aes/gpu_online_spur.py "$worker" "$models" "$fits" "$out/cycle-cad-100.json" --method cad --step-us 100 --observe-ms 1.4
timeout 1800 python benchmarks/aes/gpu_online_spur.py "$worker" "$models" "$mesh" "$out/cycle-mesh-100.json" --method mesh --step-us 100 --observe-ms 1.4
timeout 1800 python benchmarks/aes/gpu_online_spur.py "$worker" "$models" "$fits" "$out/screen-cad-100.json" --method cad --step-us 100 --observe-ms 1.4 --gpu-screen
timeout 1800 python benchmarks/aes/gpu_online_spur.py "$worker" "$models" "$fits" "$out/cpu-100.json" --method cpu-reference --step-us 100
```

For a single-step budget probe omit `--observe-ms`. Use `--step-us 50` for the smaller step and change the output filenames accordingly. On Windows the executable is usually `build/occt/Release/nc_live_lines.exe`. The server used OCCT 7.5.1 and an RTX 3090, the Windows cross-check used OCCT 7.9.3 and an RTX 3070.
