# GPU reducer contact geometry

The original four Gmsh meshes contain 65,732 triangles. The mesh detector receives current rigid transforms, walks a per-body hierarchy and emits all intersecting original triangle IDs for the body pairs declared in the scene. FP32 tests reject pairs only when projected intervals separate beyond their roundoff allowance. The remaining pairs use the original FP64 separating-axis predicate. Timings use the local RTX 3070.

## Five-second pose replay

The input is the frozen CPU mesh response at 50,000 consecutive 0.1 ms steps. At each step both body pairs run in parallel. Steps run in temporal order. Input transfers, preprocessing and validation are outside the online wall time. Force construction and implicit time integration are separate work. [Original FP64 sequential run](original.json) · [Device timings and full triangle-ID audit](geometry.json) · [Candidate-reuse experiment](reuse.json)

The FP64 audit compares sets of triangle IDs at every step, including the count of occurrences of each expected pair. A second audit compares the batched output with the native CPU OBB implementation across the trajectory, including early contact transitions and peak hit counts. The CUDA unit tests cover spatial rotation, body order, independent motion of multiple body pairs, workspace overflow and contact starting inside a reused search window.

| Measurement | Result |
| --- | ---: |
| Original FP64 geometry, two pairs in sequence | 16.81 s |
| Batched geometry, three consecutive runs | 6.747, 6.748, 7.362 s |
| Mean time per step for the median run | 134.97 µs |
| Broad phase and narrow phase per step, separate instrumented replay | 49.35 and 91.53 µs on average |
| Instrumented detection above 100 µs | 49,810 of 50,000 steps |
| Instrumented detection, 99th percentile | 461.82 µs |
| Discrepancies against FP64 on all 100,000 body-pair poses | 0 |
| CPU OBB audit, 33 selected poses and 8,795 expected triangle pairs | 0 missing, extra or repeated pairs |

The optional cached-candidate run used a 1.5 mm displacement allowance on this model. Its median retained leaf-pair counts grew from 568 and 3,151 to 5,811 and 10,114 for the first and second stages. That run took 16.32 s. The default rebuilds the smaller candidate lists at each query. The solver’s two body pairs are declared by the input scene. The measured broad phase traverses the surfaces of these pairs.

## Identical local queries on both representations

At 41 poses sampled over the five seconds, the driver takes the midpoints of finite, transverse mesh-intersection segments. It displaces each midpoint by ±0.1 mm along each involved facet's outward normal and queries a 0.4 mm material-exit ray. Both representations receive identical FP64 origins and directions in each body's local frame. This input distribution samples the mesh contact boundary. It provides a controlled comparison of local query cost and agreement. Independent geometric truth is needed to assign discrepancies to a representation.

Mesh runs with a triangle BVH and with a bounded-ray spatial grid. Compiled CAD runs its production first-exit kernel with its spatial grid and device BVH. Both representations use batches of up to 128 rays, 64 threads per block and a device-resident CUDA graph. Each of three trials warms the graph for 250 ms and times 30 entire graph executions with CUDA events. The report lists the mean time per batch for each trial and their median. Shape preparation, transfer and output are excluded. The reported times include work to find candidate primitives within the index. [Matched ray results](exit-queries.json) · [CAD kernel](../../../src/cuda_cad_roots.cuh)

| Part | Rays | Triangle BVH, µs/batch | Triangle spatial grid, µs/batch | Precompiled CAD (ours), µs/batch | CAD–mesh exit-status differences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage 1, A | 2,452 | 174.5 | 94.9 | 625.8 | 42 |
| Stage 1, B | 2,452 | 166.0 | 50.3 | 600.9 | 15 |
| Stage 2, A | 16,170 | 233.6 | 54.0 | 613.1 | 6 |
| Stage 2, B | 16,170 | 198.6 | 41.9 | 455.5 | 13 |

Both mesh indices agreed on exit status for all 37,244 queries. Compiled CAD resolved every query. It differed from the mesh on 76 exit classifications.

## Reproduction

Use a CUDA-enabled Release build and the CPU `nc_mesh` executable. Supply the same scene, four sibling Gmsh and compiled CAD parts, and frozen CPU response. The JSON files contain the mesh and trace hashes. The reference CUDA source is at commit `7a87521`.

```text
cmake --build <build> --config Release --target nc_compiled_ray_batch
python -c "import pathlib,subprocess; pathlib.Path('fp64.cu').write_bytes(subprocess.check_output(['git','show','7a87521:nurbscontact_gpu/kernels/gpu_triangle_intersections.cu']))"
python benchmarks/aes/profile_gpu_mesh_queries.py <scene> <cpu-response.csv> geometry.json --leaf-size 4 --profile-all --cpu-check <nc_mesh> --reference-source fp64.cu
python benchmarks/aes/profile_gpu_exit_queries.py <scene> <cpu-response.csv> <nc_compiled_ray_batch> ray-output --reach .4
```

The mesh replay reads prescribed poses solely for the geometry measurement. The existing device-resident CAD run starts from the initial conditions and advances contact and dynamics together. The released fixed-hinge mesh package uses a configuration map and has its own validation. [CAD closed loop](../cuda-compiled-resident/run.json) · [Historical fixed-hinge method](../gpu-hinge-mesh/README.md)
