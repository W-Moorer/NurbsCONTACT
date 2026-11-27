# GPU contact backends

## Current-pose mesh dynamics (experimental)

`mesh_dynamics.MeshDynamics` reads the CPU `.scene` deck and corresponding Gmsh `.msh` files. It preprocesses closed, consistently oriented mesh adjacency, planar panels, straight ridges and local SAH AABB trees. Online queries use the current body transforms and finite triangle intersections. Connected triangle hits provide finite vertex/face or edge/edge witnesses; spatial groups can provide contour-ray contact information. The solver transports contact anchors during a substep and refreshes geometry when a witness expires. It rebuilds predictor contacts for each new substep.

```python
from nurbscontact_gpu.mesh_dynamics import MeshDynamics

model = MeshDynamics("model/common-200.scene", steps=50_000,
                     blocks=46, cache_dir="model/topology-cache")
response, timing = model.run(0.0001, execution="graph")
```

`graph` requires a CUDA driver supporting conditional graphs (tested with CUDA 12.6 headers, CuPy 13.6 and RTX 3070). The host prepares and launches one graph, waits for completion, then downloads the results. Device-side WHILE/IF nodes control main steps, subdivision retries, Newton iterations and actual geometry refreshes. `cooperative` selects the earlier resident-grid implementation; `blocks` is its launch grid and also bounds the graph's contact support workspaces. Topology caching depends only on mesh content and preprocessing code, not on future poses, gear teeth or loading.

Supported dynamics: 1–16 scalar coordinates with positive diagonal inertia, independently oriented hinge axes and at most one prismatic coordinate per body, prescribed coordinate load tables, zero gravity, frictionless shared penalty stiffness and HHT spectral radius zero. Body mass must already be represented by coordinate inertia. This is not a free-body or general constrained multibody solver. Unsupported scene fields, open meshes, queue overflow, enclosed components without boundary contacts, unresolved finite witnesses and exhausted substeps are errors, not silently omitted contacts. Closed mesh self-intersection and cavity orientation require valid input; preprocessing is not a complete solid-validity proof. Contact is discrete predictor contact, not CCD.

Seven analytic / validation tests cover finite face and crossed-edge contact, spatial frame changes, exchanged bodies, separation, disconnected supports, enclosure errors, queue overflow, invalid scenes and a unilateral oscillator in both execution modes. The full reducer completes 5 s at 0.1 ms, but its peak-load equivalence to CPU and real-time performance are **not accepted**. Near-coplanar end-cap triangles can change the contour-derived contact center. See the [full curves, timings and diagnostic](../results/aes/gpu-mesh-dynamics/README.md). Use the explicit API above; the historical `nc-gpu` CLI below retains its configuration-map behavior.

The expanded nine-test suite also covers trial-contact failure recovery and fatal initial enclosure. In both execution modes, noninitial geometry statuses 26/27 reject the trial and retry at half the step, up to the existing level-eight limit. The next attempt rebuilds every pair's contact cache from the accepted state. Initial geometry errors, persistent errors at the limit and capacity/model errors still stop execution. Graph timing includes `integration` with accepted substeps by level and geometry/Newton rejection counts. A 1 ms main-step reducer run now completes 5 s in 21.84 s; its response differs from the 0.1 ms result, and convergence is not a temporal-accuracy or energy guarantee. [Complete step comparison](../results/aes/gpu-mesh-dynamics/step-1ms-repaired/README.md).

## Historical implementation

The independent `bvh.IntersectionBatch` API accepts a current three-dimensional rigid transform for each declared body pair. Its default `fine_lanes=8` cooperatively evaluates the same finite FP64 triangle predicate; `fine_lanes=1` is the scalar control. This API returns intersections, without deriving forces from a configuration map. The five-second geometry replay and the separate sampled-mesh dynamics timings are documented in [the current optimization report](../results/aes/gpu-mesh-speed/README.md). The historical closed-loop package described below retains its original scope.

The GPU backend integrates rigid bodies on fixed parallel revolute axes using finite triangle intersection, a precomputed contact wrench or material point map and a generalized-alpha integrator. The wheel includes the CUDA kernels and the native library for synchronous load and wrench exchange. The validated model uses a shared linear normal penalty. See the [five-second reducer validation](https://github.com/W-Moorer/NurbsCONTACT/tree/main/results/aes/gpu-hinge-mesh).

## Install

Use Python 3.10 or later on a CUDA 12 system with an NVIDIA driver and NVRTC. Install the wheel for your operating system with the `cuda12` extra. For an environment that already provides a compatible CuPy, install the wheel directly. The source distribution builds the native exchange library with CMake and a C++20 compiler.

```text
python -m pip install "./nurbscontact_gpu-0.1.1-py3-none-win_amd64.whl[cuda12]"
nc-gpu --help
```

## Run the reducer

The development CLI requires an explicit `--method` for this configuration-map experiment. Current-pose triangle detection is available through [IntersectionBatch](bvh.py) and has a [separate validation](../results/aes/gpu-mesh-geometry/README.md).

Get the [CPU scene and Gmsh meshes](https://github.com/W-Moorer/NurbsCONTACT/raw/refs/heads/main/results/aes/cpu-foundation/reducer-physics/cpu-deck.zip) and the matching [geometry table](https://github.com/W-Moorer/NurbsCONTACT/raw/refs/heads/main/results/aes/gpu-hinge-mesh/geometry.npz). From the directory containing both downloads, run:

```text
python -m zipfile -e cpu-deck.zip model
nc-gpu deck windows --scene model/reducer-physical/common-200.scene --method mesh-wrench --prepare
nc-gpu deck windows --geometry geometry.npz --method mesh-wrench --exchange --output run
```

The default simulation lasts 5 s at a 0.1 ms step. Preparation builds candidate windows once. Omit `--exchange` for device-resident integration. `run/response.npz` contains the shaft and contact histories, and `run/timing.json` records setup, elapsed time and missed step deadlines. For a different model, generate its geometry table with `nc-gpu-map --help` and a CPU `nc_mesh` executable built with contour-ray support.

The equivalent Python calls are:

```python
from nurbscontact_gpu.backend import export_scene, prepare, run

export_scene("model/reducer-physical/common-200.scene", "deck", duration=5.0, step=0.0001)
prepare("deck", "windows")
timing = run("deck", "windows", "run", method="mesh-wrench", geometry="geometry.npz", exchange=True)
```
