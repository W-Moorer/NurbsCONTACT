# GPU contact discovery and HHT for rigid CAD parts

The [cam–follower transfer test](../../resident-cam/README.md) exercises the same kernel with a hinge and a slider. It includes independent displacement and contact-switch checks, a pure mesh control and measurements on both GPUs.

A single gear pair generates its own trajectory over two seconds. Each trial pose comes from HHT. The geometry receives rigid transforms, closed outward-oriented surfaces and face labels. The model specifies the joint axes, inertias and imposed loads. The search receives no tooth indices, contact stations, measured trajectories or gear profile equations.

## Geometry and force

The target in the gear test has 70,960 triangles and the source has 37,700. In the CAD-source variant, OCCT evaluates one point on each original trimmed source face for every UV triangle, and one point on the original STEP curve for each of 10,690 shared-edge segments. Trim checks protect face evaluations. Probes on a shared edge retain both incident source facet normals. The more strongly opposing face takes ownership at each pose. The archive stores the resulting local points and binds them to the source geometry and face labels with SHA-256. The pure mesh control uses flat triangle centroids and straight edge midpoints. Source normals in both variants come from the triangulation.

The target surface used online is the tessellation. The grid stores nearby triangle IDs and conservative separation tests. The GPU culls candidates using rigid-motion bounds, ranks nearby triangles in local coordinates, refines distance and normal in double precision and traverses the whole BVH if the local index cannot supply a sufficiently close triangle. The matched BVH control shares source screening, the starting triangle, force aggregation, residual tolerance and HHT. It obtains the nearest target triangle by double-precision BVH traversal. The spatial index and per-step candidate reuse can be used with other rigid shapes. [Contact implementation](../../../../benchmarks/aes/resident_contact.cu) · [Index construction](../../../../benchmarks/aes/resident_index.py)

For one source sample inside the target, the algorithm uses target feature pseudonormals to determine sign and the closest point to obtain penetration depth. For each active CAD face pair, the deepest sample supplies a unilateral normal force of stiffness times penetration. The resulting wrench drives the equations of motion. Face segmentation, source sampling density and the choice of source body can change this discrete force law. Contact onset is sampled in time. A grazing contact may occur entirely between samples.

## Integration and timing

The GPU kernel computes trial transforms, finds contacts, accumulates forces and tangents, solves the two-coordinate HHT residual and handles backtracking and subdivision. HHT uses α = −0.1 and accepts a residual below 10⁻⁷ times one plus the current/previous force scale. The host enqueues chunks of 64 steps. The public geometry query supports arbitrary rigid poses. The closed-loop driver supports one pair of bodies, each with one hinge or slider, and a frictionless normal law.

With `--impact-samples 32`, a positive contact following a separated state can trigger a retry from the last accepted state. The local interval targets 32 samples per half-cycle, using π divided by the square root of the trace of the mass-scaled contact stiffness. The window follows the observed impact and subdivisions are dyadic, down to 1/64 of a main step. This criterion controls resolution of an observed contact. The report gives both the maximum over accepted substeps and the maximum on the regular output grid. A peak can fall between output rows.

Wall time includes the online GPU feedback loop, synchronization at the end and all rejected trials. CAD sampling, index construction, solver initialization, the time-zero force and final trace export occur outside that interval. `setup_s`, per-run `initialization_s` and the CAD sampling time describe these costs separately. Device per-step percentiles cover the sequence of trials for each accepted main step. Host scheduling gaps between chunks appear in total wall time. Each reported full run represents one instance.

## Measurements

The 24/48-tooth gear pair starts at 1,800 rpm on the driver. All rows below use a 125 µs main step, the same loads and penalty stiffness, identical HHT settings, and local impact refinement. Each cell is the median online wall time of three complete two-second runs. All ended with 16,000 accepted main steps and zero terminal Newton failures.

| Source geometry and target search | RTX 3070 | RTX 3090 |
|---|---:|---:|
| Mesh probes, double-precision triangle BVH | 10.820 s | 8.258 s |
| Mesh probes, indexed search | 1.774 s | 1.625 s |
| Original STEP probes, double-precision triangle BVH | 10.491 s | 8.077 s |
| Original STEP probes, indexed search with double refinement (ours) | 1.746 s | 1.593 s |

Holding the STEP probes and integration policy fixed, indexing speeds up the complete simulation by 6.01× and 5.07×. On the RTX 3070, every one of the 16,001 output states, including time zero, matched the corresponding BVH run in shaft angles, speeds, active pair count and normal force. The pure mesh control also matched its BVH counterpart. The grid accounts for most of the speed gain. Moving probes onto original CAD affects the sampled response and coverage. [Response checks](response-checks.json) · [RTX 3070 timings](spur-cad-impact-grid-125-3070.json) · [RTX 3090 timings](spur-cad-impact-grid-125-3090.json)

The CAD run accepted 17,761 internal increments. On the RTX 3090 the median device P99 was 149.5 µs, with 693 of 16,000 main steps exceeding 125 µs in the median count. Overall real-time throughput coexists with missed per-step deadlines. The GPU had 0% utilization immediately before this comparison. With a competing workload, a separate 100 µs pure mesh run took 3.916 s on the same server, compared with 1.798 s in the idle run. The CAD cache took 38.1 s to build locally. On the server its primary run took 4.82 s for model setup plus 0.04 s for first-run solver initialization. [Server device sample](3090-before-cad.csv) · [Shared-device run](spur-grid-100-3090-shared.json) · [Idle-device run](spur-grid-100-3090.json)

### Accuracy at contact entry

The independent timestep check runs the same CAD-source model at a fixed 5 µs for the first 20 ms. Its first loaded output is at 220 µs and its peak is 6,078.79 N. The local substeps first register positive load at 218.75 µs and reach 6,108.22 N at 1.1992 ms. Sampling the latter history only at 125 µs gives a peak of 4,999.77 N. At the 161 common output times, enabling local impact refinement reduces RMS transmission-error difference from 30.824 µm to 3.426 µm and RMS normal-force difference from 1,632 N to 690 N. A close peak therefore accompanies appreciable pointwise force differences. The timestep check measures numerical resolution under the chosen force model. [Fine run](spur-cad-grid-5-3070.json) · [Fixed-step control](spur-cad-grid-125-3070.json) · [Trajectory differences](response-checks.json)

Across 2,159 archived trial poses, the CAD-source grid and the double BVH return identical face sets, depths and normals. At 150 winning witness points, the largest difference against independent CPU signed mesh distance is 2.03 × 10⁻⁷ mm. A separate all-triangle intersection audit reports 3,534 opposing face-pair occurrences using dot product < −0.95. The CAD-source sampler misses 16 occurrences and adds two relative to that mesh reference. It preserves a contact in every frame with opposing mesh contact and reports contact in one extra frame. With flat mesh probes, 28 occurrences and six entire contact frames were missed. The CAD and mesh surfaces differ, and intersection enumeration and sampled penetration have different outputs. These are fixed-pose coverage measurements. [CAD geometry audit](geometry-cad.json) · [Mesh geometry audit](geometry-checks.json)

The same backend also ran the repository's two-sphere case with zero-gap activation. At a 100 µs step it gave 22.4997 N against the analytic 22.5 N peak, first loaded at 66.8 ms against 66.667 ms, and first released at 98.2 ms against 98.083 ms. On the RTX 3070 the 0.2 s run took 0.206 s at 100 µs and 0.168 s at 125 µs. A separate 33-pose rotated-box regression compares sampled depths and identities to an analytic CPU box calculation. [Sphere run](spheres-grid-125-3070.json) · [Analytic checks](response-checks.json)

## Reproduction

A different mechanism can supply `--deck model.npz`. Arrays `a` and `b` hold local triangle coordinates in mm with shape (N, 3, 3). Integer arrays `fa` and `fb` label their original faces. The two rows of `rows` each contain joint type (0 hinge, 1 slider), a unit axis (3), joint origin in mm (3), nominal offset, nominal speed, inertia or mass, spring stiffness, damping and initial coordinate correction. The physical joint coordinate is offset + speed × time + correction. Angles use radians and slider coordinates use metres. `loads` has two columns of external torque or force sampled every `h` seconds, including time zero. `stiffness` is the contact penalty in N/m. The optional CAD probe cache must match the source triangles, face labels and sample density. Finite rigid-transform queries can also call `ResidentMeshContact.evaluate` without the two-coordinate driver. [Python geometry interface](../../../../benchmarks/aes/resident_contact.py) · [GPU dynamics driver](../../../../benchmarks/aes/resident_dynamics.py)

Run from the repository root with NumPy, SciPy, Numba, CuPy/CUDA and OCP. Independent CPU geometry checks also use trimesh and its spatial index. The archive includes the identical CAD probes used on both GPUs. To regenerate the cache from the original model, run

```text
python benchmarks/aes/resident_cad_probes.py results/gears/models/spur-1b.step results/aes/mesh-comparison/meshes/wheel-0.4.npz results/aes/online-gear/tracking/faces-b.npz build/wheel-step-uv.npz --deflection-um .4
```

The following commands reproduce the matched 125 µs comparison and the independent local step-refinement run. Output directories must exist.

```text
python benchmarks/aes/resident_dynamics.py build/cad-grid.json --backend grid --cad-probes results/aes/online-gear/resident/wheel-step-uv.npz --step-us 125 --impact-samples 32 --chunk 64 --trace build/spur-cad-impact-grid-125-3070.npy
python benchmarks/aes/resident_dynamics.py build/cad-bvh.json --backend bvh --cad-probes results/aes/online-gear/resident/wheel-step-uv.npz --step-us 125 --impact-samples 32 --chunk 64 --trace build/spur-cad-impact-bvh-125-3070.npy
python benchmarks/aes/resident_dynamics.py build/cad-fine.json --backend grid --cad-probes results/aes/online-gear/resident/wheel-step-uv.npz --duration .02 --step-us 5 --chunk 64 --repeats 1 --trace build/spur-cad-grid-5-3070.npy
python benchmarks/aes/check_resident_contact.py build/geometry-cad.json --cad-probes results/aes/online-gear/resident/wheel-step-uv.npz
```

Removing `--cad-probes` selects the pure mesh model. Removing `--impact-samples` selects the fixed main step with subdivision on Newton failure. The two-sphere case uses `--case spheres --duration .2 --pitch .8`. Saved trace columns are the two joint corrections, their two velocities, output-time normal load, active face-pair count, Newton iteration and residual. The filename identifies the device, source, backend, timestep and policy.
