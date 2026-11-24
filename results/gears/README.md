# Industrial STEP transmission benchmarks

The suite compares native STEP contact against AABB/BVH and two-sided triangle intersections under the same implicit HHT-α solver, motion coordinates, contact law, and load schedule. Gear runs share the projected line-of-action stations on the finite flanks. The bearing runs share the radial probe axes and the interval-wide radial allowance bound.

## Fixtures and comparison results

| Fixture | Operating dimensions and loads | Native / fine mesh integration | Measured ratio |
|---|---|---:|---:|
| [Spur 24/48](assemblies/spur.step) | Normal module 2.5 mm, face 24 mm, 1,800 rpm, 56 N·m | 0.299 / 2.087 s | 6.99× |
| [Helical 25/45](assemblies/helical.step) | Normal module 2.5 mm, face 28 mm, 25° helix, 1,500 rpm, 72 N·m | 2.380 / 7.689 s | 3.23× |
| [NASA-rig geometry 28/28](assemblies/rig28.step) | Module 3.175 mm, face 6.35 mm, 23/25 µm relief, 2,000 rpm, 71.77 N·m | 0.373 / 1.412 s | 3.79× |
| [Two-stage reducer](assemblies/reducer.step) | 24/48 spur plus 22/44 helical, 4:1 overall, 1,800 rpm, 148 N·m | 2.577 / 9.700 s | 3.76× |
| [Fourteen-roller bearing](assemblies/bearing.step) | 45 × 85 × 19 mm envelope, 1,800 rpm shaft, 2.4–3.1 kN principal radial load with imbalance | 0.139 / 3.780 s | 27.18× |

Times are medians of three Release runs. The spur pair and NASA geometry use 0.60 s. The helical and reducer timings in the table use 0.36 s and 31 axial stations for their helical stages. The spur stages use one station. All listed runs use 20 µs nominal steps. Requested fine mesh deflections are 1.5625 µm for gears and 0.50 µm for the bearing. Integration timing includes contact evaluation and HHT iterations. STEP import, mesh construction, and disk output are excluded, with setup time recorded in each statistics file.

The 31-station helical pair and reducer helical stage show transmission RMS differences of 0.0082 and 0.0108 µm against 63 native stations. At matched 31-station sampling, fine meshes differ from native by 0.825 and 0.899 µm on those meshes. The corresponding normal-force RMS differences are 0.92% and 3.21% of mean native force. The reducer's spur stage and the single-stage spur pair exhibit larger force-waveform differences. The gear data include 10 µs references and three tessellation levels. The fine-mesh bearing shaft orbit differs from the native orbit by 0.754 µm RMS and 3.05 µm maximum.

Both reducer backends also complete 3 s and 150,000 nominal steps. Native integration takes 5.87 s and the fine-mesh run takes 27.75 s. The smallest native sampled working-flank clearance in the native reducer run is 32.27 µm. The long-run contact sampling uses seven axial stations on the helical stage. The bearing's smallest verified radial interval allowance across the native and fine-mesh runs is 44.04 µm.

The NASA-rig case is a numerical reconstruction of the published 28-tooth gear geometry, with a benchmark-defined load schedule, support model, and stiffness. The bearing shares catalog envelope dimensions with the NU 209 family. Its interior geometry and radial compliance are defined here. The reported comparisons characterize reduced models with rigid gears, normal-only projected compliance, or prescribed bearing cage motion. Physical gear-root stress, material fatigue, lubrication, and full-solid gear CCD require other observables and models. [NASA gear-noise-rig report](https://ntrs.nasa.gov/citations/19910013253) · [SKF envelope dimensions](https://www.emarketplace.in.skf.com/cylindrical-roller-bearing/nu-209-ecp)

## Figures and animations

The model images use triangulations of the output STEP solids and translations/rotations from the recorded solver states. The rendered dots mark projected contact stations. The videos slow short measured windows to make tooth motion and roller load transfer visible.

| Result | Figures or videos |
|---|---|
| Four gear CAD models | [Model plate](figures/gear-models.pdf) |
| Gear dynamics and performance | [Force and transmission error](figures/gear-dynamics.pdf) · [Cost and faceting](figures/gear-cost-accuracy.pdf) · [Axial refinement](figures/gear-axial-refinement.pdf) |
| Long reducer test | [Contact loads and gap envelopes](figures/reducer-duty.pdf) |
| Bearing | [Roller forces and orbit](bearing/figures/bearing-loads.pdf) |
| Spur | [Isometric](animations/spur-isometric.mp4) |
| Helical | [Isometric](animations/helical-isometric.mp4) |
| NASA-rig geometry | [Isometric](animations/rig28-isometric.mp4) |
| Two-stage reducer | [Isometric](animations/reducer-isometric.mp4) |
| Cylindrical roller bearing | [Isometric](bearing/animations/bearing-isometric.mp4) |

[Gear animation index](animations/index.json) · [Bearing animation index](bearing/animations/index.json)

## External gear-rig validation

The separate comparison replays the NASA single-tooth force tests and reconstructs UNSW transmission error from paired encoder records. Each rig uses one condition for calibration and frozen parameters on the remaining conditions. [Protocol, figures, metrics, and source hashes](experimental/README.md)

## Reproduce

Build the OCCT Release configuration and set the runtime path as described in the [project build instructions](../../README.md#step-queries-on-windows). Run the timed numerical commands serially. Use a separate output folder to preserve the delivered trials.

~~~powershell
python -m pip install numpy matplotlib pillow pyvista cadquery av pymupdf
cmake --build build/occt --config Release --target nc_gears nc_roller_bearing
python benchmarks/gears/run.py build/occt/Release/nc_gears.exe results/gears --resume
python benchmarks/gears/analyze.py results/gears
python benchmarks/gears/bearing_run.py build/occt/Release/nc_roller_bearing.exe results/gears/bearing
python benchmarks/gears/assemblies.py results/gears
python benchmarks/gears/render.py results/gears
python benchmarks/gears/plots.py results/gears
python benchmarks/gears/bearing_visual.py results/gears/bearing
python benchmarks/gears/paper.py results/gears
python benchmarks/gears/verify.py results/gears build/gears-qa
~~~

The gear solver is in [main.cpp](../../benchmarks/gears/main.cpp) and [flank.cpp](../../benchmarks/gears/flank.cpp). The bearing solver is in [bearing.cpp](../../benchmarks/gears/bearing.cpp). Both use [HHT](../../src/hht.cpp) and the [triangle BVH](../../src/triangle_bvh.cpp). The main APIs are covered by the geometric and HHT tests. The experiment scripts validate trace lengths, timing repetitions, solver clearance checks and native section equivalence.

[Industrial PDF](paper/industrial.pdf) · [LaTeX source](../../manuscript/industrial.tex) · [Gear protocol](protocol.json) · [Gear summary](summary.json) · [Gear trajectories](runs) · [Bearing protocol](bearing/protocol.json) · [Bearing summary](bearing/summary.json) · [Bearing trajectories](bearing/runs)
