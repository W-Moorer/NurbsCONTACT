# Spur gear time-step and triangle mesh comparisons

The CPU runs advance a 24/48-tooth pair at 1,800 rpm for 2 s. The existing gear solver selects tooth pairs along the line of action. Both backends use its fixed force strategy, the same HHT settings and the same STEP flank files. Each mesh backend raycasts a double-precision triangle BVH for each of the two flanks. The straight spur pair uses one axial station under the solver's existing symmetry rule.

- `spur-cpu-steps.json` contains five runs per configuration on an AMD Ryzen 9 5900X. `median_integrator_s` covers integration and frame capture. `median_process_s` also covers geometry setup, the post-run CAD audit and writing every step. The mesh sizes count both representative flanks. The traces have repeatable hashes.
- `spur-refinement.json` reports a separate sweep for the finer mesh and the CAD backend. RMS differences compare the two-second histories at common times spaced 100 µs apart. Peaks use every saved step. The 5 µs trace provides a numerical step-refinement reference for each backend. Physical load accuracy would require an independent load measurement or solution.
- `shaft-balance.json` checks the mean normal force on the driven shaft. For each 100 ms window the balance gives the normal impulse as $(I\Delta\dot q+c\Delta q+\int M_{\mathrm{load}}\,dt)/r_b$. The audit integrates the prescribed torque analytically and takes the endpoint angle and speed from the trace. It compares this impulse with the trapezoidal integral of the reported force. A window spans three nominal driver revolutions. This balance checks the average load independently of the projection and contact force routine.
- The `50us` and `100us` directories contain independent CPU OCCT searches at the first tooth entry and exit windows after the 0.1 s midpoint of the load increase. Each window contains three consecutive poses from the CAD trajectory at its own step size. The archived `20us` case uses two windows of five poses. The search statistics record unresolved pairs. The line loads and Gauss points come from the two original STEP bodies.
- The GPU replays use those points on body B with identical quadrature and line force laws for all geometry methods. The fitted CAD method projects onto the supplied face of A. The face-mesh methods search a BVH of triangles tessellated from that supplied face. The full-mesh methods search all triangles of A using an independent whole-gear tessellation. Each GPU mesh witness is checked against a CPU exhaustive search of the same mesh. Reported force differences are Euclidean vector differences per contact line relative to CPU OCCT at the same pose.
- GPU times are device-event medians per known pose with data resident and a graph containing successive poses. The replay includes projection, force and moment accumulation. An online rigid simulation also has to discover the contacts, update body B's integration points and complete the integrator iterations. The RTX 3090 was shared. The device samples record total device activity, including other work.

## Results

The CPU process medians below include setup and trace export for 2 s of physical time.

| HHT step | CAD gear-flank solver (s) | Mesh BVH, 13,770 triangles (s) | Mesh BVH, 60,084 triangles (s) |
|---|---:|---:|---:|
| 20 µs | 10.04 | 25.75 | 26.85 |
| 50 µs | 5.09 | 10.92 | 14.86 |
| 100 µs | 2.50 | 6.57 | 9.96 |

The fine mesh gives a 941.39 N mean load at both 50 and 100 µs. The maximum difference from driven-shaft balance over the twenty 100 ms windows was 0.002 N and 0.0083 N, respectively. The force waveform differs from the fine mesh at 5 µs by 351 N RMS at 50 µs and 435 N RMS at 100 µs. At 10 µs this difference was 44 N. The figure compares instantaneous forces at the midpoint of the prescribed load increase. Each curve is one deterministic history sampled on its own integration grid. [Force history (PDF)](spur-force-window.pdf) · [Source samples](spur-force-window.csv)

On the shared RTX 3090, each step-specific replay contains six poses and twenty quadrature points. Deflections below refer to requested OCCT mesh settings. The same-face rows receive the same candidate face information as the fitted CAD backend.

| Geometry method | 50 µs trajectory, µs/pose | Maximum line force difference, N | 100 µs trajectory, µs/pose | Maximum line force difference, N |
|---|---:|---:|---:|---:|
| Fitted CAD, previous UV (ours) | 39.94 | 0.000651 | 40.28 | 0.000269 |
| Same-face mesh BVH, 0.10 µm | 37.38 | 1.235 | 38.91 | 1.230 |
| Same-face mesh BVH, 0.04 µm | 44.20 | 0.372 | 44.37 | 0.354 |
| Same-face mesh BVH, 0.01 µm | 55.13 | 0.192 | 56.06 | 0.158 |
| Full mesh BVH, 42,296 triangles | 92.25 | 1.235 | 91.99 | 1.230 |
| Full mesh BVH, 70,960 triangles | 109.91 | 0.372 | 110.42 | 0.354 |

## Reproduce

Build `nc_gears` and `nc_contact` with OCCT. Run from the repository root with their runtime libraries on the process path. The following commands use the Windows Release build. A single CPU run has a 1,800 s limit.

```
python benchmarks/aes/spur_mesh_steps.py build/occt/Release/nc_gears.exe results/gears/models build/aes-spur-mesh-steps results/aes/mesh-comparison/spur-cpu-steps.json
python benchmarks/aes/spur_mesh_steps.py build/occt/Release/nc_gears.exe results/gears/models build/aes-spur-refinement results/aes/mesh-comparison/spur-refinement.json --repeats 1 --refinement
python benchmarks/aes/spur_mesh_steps.py build/occt/Release/nc_gears.exe results/gears/models build/aes-spur-refinement results/aes/mesh-comparison/shaft-balance.json --balance
python benchmarks/aes/spur_mesh_force_plot.py build/aes-spur-refinement results/aes/mesh-comparison
python benchmarks/aes/spur_mesh_steps.py build/occt/Release/nc_gears.exe results/gears/models build/aes-spur-mesh-steps results/aes/mesh-comparison/spur-cpu-steps.json --replay-poses results/aes/mesh-comparison
```

For each larger step, set `D` to the corresponding input directory (shown here for 50 µs). The `meshes` directory contains both full triangle inputs. They can also be generated with `benchmarks/aes/gear_precision.py mesh OUTPUT --deflection-um D` at 0.1 and 0.04 µm. The face-mesh input is built directly from STEP at each requested deflection.

```sh
D=results/aes/mesh-comparison/50us
M=results/aes/mesh-comparison/meshes
build/occt/Release/nc_contact.exe regions-replay results/gears/models/spur-1a.step results/gears/models/spur-1b.step 0.000055 1e9 "$D/poses.txt" "$D/cold" cold --samples
python benchmarks/aes/gpu_line_wrench.py results/gears/models/spur-1a.step results/aes/gear-field/faces.npz "$D/cold.samples.csv" "$D/poses.txt" "$D/cold.loads.csv" gpu-50us.json --max-seed-age-s 0.0000625 --cutoff-mm 0.055 --stiffness 1e9 --mesh-um 0.1 0.04 0.01 --full-mesh "$M/mesh-0.1.npz" "$M/mesh-0.04.npz"
```

For 100 µs, use the corresponding directory and a maximum seed age of 0.000125 s. For the earlier 20 µs windows use `results/aes/gear-field/transition-points.csv`, `20us/spur-poses.txt`, `20us/cold.loads.csv` and 0.000025 s. All cases allow hints only from the immediately preceding pose and matching face pair and quadrature ordinal. The raw input hashes in the GPU reports include the CRLF line endings of the pose and OCCT files exported on Windows.
