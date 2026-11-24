# NurbsCONTACT GPU

The GPU backend integrates rigid bodies on fixed parallel revolute axes using finite triangle intersection, a precomputed contact wrench or material point map and a generalized-alpha integrator. The wheel includes the CUDA kernels and the native library for synchronous load and wrench exchange. The validated model uses a shared linear normal penalty. See the [five-second reducer validation](https://github.com/W-Moorer/NurbsCONTACT/tree/main/results/aes/gpu-hinge-mesh).

## Install

Use Python 3.10 or later on a CUDA 12 system with an NVIDIA driver and NVRTC. Install the wheel for your operating system with the `cuda12` extra. For an environment that already provides a compatible CuPy, install the wheel directly. The source distribution builds the native exchange library with CMake and a C++20 compiler.

```text
python -m pip install "./nurbscontact_gpu-0.1.0-py3-none-win_amd64.whl[cuda12]"
nc-gpu --help
```

## Run the reducer

Get the [CPU scene and Gmsh meshes](https://github.com/W-Moorer/NurbsCONTACT/raw/refs/heads/main/results/aes/cpu-foundation/reducer-physics/cpu-deck.zip) and the matching [geometry table](https://github.com/W-Moorer/NurbsCONTACT/raw/refs/heads/main/results/aes/gpu-hinge-mesh/geometry.npz). From the directory containing both downloads, run:

```text
python -m zipfile -e cpu-deck.zip model
nc-gpu deck windows --scene model/reducer-physical/common-200.scene --prepare
nc-gpu deck windows --geometry geometry.npz --method mesh-wrench --exchange --output run
```

The default simulation lasts 5 s at a 0.1 ms step. Preparation builds candidate windows once. Omit `--exchange` for device-resident integration. `run/response.npz` contains the shaft and contact histories, and `run/timing.json` records setup, elapsed time and missed step deadlines. For a different model, generate its geometry table with `nc-gpu-map --help` and a CPU `nc_mesh` executable built with contour-ray support.

The equivalent Python calls are:

```python
from nurbscontact_gpu.backend import export_scene, prepare, run

export_scene("model/reducer-physical/common-200.scene", "deck", duration=5.0, step=0.0001)
prepare("deck", "windows")
timing = run("deck", "windows", "run", geometry="geometry.npz", exchange=True)
```
