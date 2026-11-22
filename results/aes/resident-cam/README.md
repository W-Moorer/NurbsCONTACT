# Cam–follower validation on the resident GPU contact backend

The eccentric cam runs the same contact queries, face-pair force assembly and HHT kernel as the [gear experiment](../online-gear/resident/README.md). Its inputs are a revolute joint, a slider, rigid-body geometry and loads. An independent CPU calculation uses the support function of the eccentric circle. A second check uses the published kinematic solution for a noncircular cam.

## Closed-loop eccentric cam

The single-cam scene supplies the cam radius (105 mm), eccentricity (24 mm), initial follower height (115 mm), the two masses/inertias, loads, springs, dampers and initial velocities. The original finite follower face is extruded 50 mm behind the working surface to provide a closed target. Its width and contact side are preserved. The zero-gap experiment loads the physical working face on contact. A separate run reproduces the repository's 12 mm activation by shifting the active plane 12 mm toward the cam, with unchanged follower displacement and moment arm about the cam. [Scene definition](../../../benchmarks/hht/scene.cpp) · [Deck generation and independent reference](../../../benchmarks/aes/resident_cam.py)

The GPU tests 6,104 source probes from 4,068 cam triangles against a 12-triangle follower. Both CAD runs use identical trimmed-face and shared-edge samples. The mesh control uses flat centroids and straight edge midpoints. All three use the same 125 µs step, 1 µm requested source tessellation deflection, 4 mm index pitch, contact stiffness and HHT settings as each other. The optional impact policy is enabled at 32 intervals per half-cycle and triggered zero restarts at this step and stiffness.

| 20 s physical time, 160,000 steps, median of three complete runs | RTX 3070 | RTX 3090 |
|---|---:|---:|
| Mesh probes + double triangle BVH | 8.900 s | 8.944 s |
| CAD probes + double triangle BVH | 8.850 s | 9.080 s |
| CAD probes + indexed search and double refinement (ours) | 8.836 s | 8.971 s |

The clock includes contact discovery, force, HHT iterations and final synchronization. Preprocessing and initialization are separate. The three methods reach average real time on both cards. The small target gives the index little opportunity to save BVH traversal work. The primary run on the RTX 3070 had a median P99 step time of 63.49 µs and a median count of 200 steps exceeding the 125 µs deadline. The matched CAD grid and BVH histories agree at all 160,001 states on both devices. [RTX 3070 measurements](0mm-125us-cad-grid-3070.json) · [RTX 3090 measurements](0mm-125us-cad-grid-3090.json) · [Device occupancy before the server batch](3090-load.csv)

## Independent displacement and event checks

In SI units, the CPU computes compression from `0.105 + 0.024 sin(θ) − (0.115 + q − skin)`, with `q` the absolute follower travel from its initial position and `θ` the actual driven angle. The positive part times 5,000 N/m loads the follower. The opposite cam torque is `−0.024 cos(θ) F`. DOP853 integrates the two coupled equations and locates zeros of compression in continuous time. Halving the reference maximum step from 2 ms to 1 ms changed reference displacements by at most 0.000205 µm and event times by at most 0.0032 µs in the 20 s zero-gap trajectory.

| Zero-gap model on RTX 3070 | Follower RMS error | Maximum displacement error | Entries / releases | Maximum output-midpoint event error |
|---|---:|---:|---:|---:|
| CAD, 125 µs | 0.164 µm | 0.927 µm | 36 / 35 | 62.81 µs |
| Pure mesh BVH, 125 µs | 0.784 µm | 3.873 µm | 36 / 35 | 85.23 µs |
| CAD, 62.5 µs | 0.130 µm | 0.597 µm | 36 / 35 | 37.92 µs |

All 71 events appear in the independently predicted order. After the last engagement at 3.00664 s, the follower remains in contact through 20 s. At 125 µs, 69 CPU event times lie in the GPU output brackets. The other two lie at most 0.309 µs outside them. A separate postprocess bisects the same GPU contact predicate along cubic Hermite joint paths between consecutive saved states. For the 125 µs trajectory it made 923 geometry queries in 1.15 s, reducing maximum event-time error to 10.11 µs and RMS error to 3.73 µs. These queries are excluded from the online timings. At 62.5 µs, the refined maximum was 15.73 µs. The refined event error did not decrease monotonically with timestep. The audit covers transitions visible between consecutive output states. [125 µs response](accuracy-0mm-125us.json) · [62.5 µs response](accuracy-0mm-62.5us.json) · [Refined events](events-0mm-125us.json)

The 12 mm repository model gives 44 transitions and a 0.311 µm RMS displacement difference from the independent ODE. The GPU trajectory differs by 0.095 µm RMS from the archived cylinder–plane single-cam solver at the same step. Its general GPU online time was 9.37 s in one RTX 3070 run. The repository's direct analytic cylinder–plane solver recorded a 1.19 s median on CPU for this model. [Original-model check](accuracy-12mm-125us.json) · [Specialized CPU timing](../cam-baselines.json)

## Published noncircular cam

Dhande and Rajaram's Example 1 specifies a constant-breadth profile with a flat translating follower, input speed 1 rad/s and a 1° evaluation grid. We retain the dimensions in Table 1 and choose 1 mm per paper length unit. For 3,601 poses over one revolution the GPU finds the support against the flat working surface from its previous measured position and a rotation bound. The closed-form displacement in Table 2 is used only for scoring. Arc owners at switches are identified by their independent STEP radii, including the periodic seam. [Original paper, Tables 1–2 and Example 1](https://citeseerx.ist.psu.edu/document?doi=e8e3c38ff1d1e409f44a58886669295db63b14a9&repid=rep1&type=pdf)

| Requested cam deflection | Source probes | Displacement RMS / max | Maximum arc-switch midpoint error |
|---|---:|---:|---:|
| 0.25 µm | 2,378 | 0.0197 / 0.0547 µm | 0.45° |
| 0.0625 µm | 4,724 | 0.0050 / 0.0139 µm | 0.25° |

Both resolutions recovered all six arc transitions around the full cycle in the published order. These transitions change the active profile segment while the follower remains touching. The original paper supplies a kinematic reference. The load and release results above come from the eccentric-cam dynamical model. [Primary kinematic audit](constant-breadth.json) · [Refined kinematic audit](constant-breadth-fine.json)

## Reproduction and figure

Run from the repository root. `prepare` exports portable decks, the backed follower and CAD probe cache into the specified working directory. The same deck can run on a GPU server with no online OCCT dependency.

```text
python benchmarks/aes/resident_cam.py prepare build/resident-cam
python benchmarks/aes/resident_dynamics.py build/resident-cam/grid.json --deck build/resident-cam/deck-0mm-125us.npz --cad-probes build/resident-cam/cam-probes.npz --duration 20 --step-us 125 --pitch 4 --chunk 64 --impact-samples 32 --backend grid --trace build/resident-cam/grid.npy
python benchmarks/aes/resident_dynamics.py build/resident-cam/bvh.json --deck build/resident-cam/deck-0mm-125us.npz --cad-probes build/resident-cam/cam-probes.npz --duration 20 --step-us 125 --pitch 4 --chunk 64 --impact-samples 32 --backend bvh --trace build/resident-cam/bvh.npy
python benchmarks/aes/resident_cam.py audit build/resident-cam/grid.json build/resident-cam/grid.npy build/resident-cam/bvh.json build/resident-cam/bvh.npy build/resident-cam/check.json --skin-mm 0
python benchmarks/aes/resident_cam.py refine-events build/resident-cam/deck-0mm-125us.npz build/resident-cam/cam-probes.npz build/resident-cam/grid.npy build/resident-cam/check.json build/resident-cam/events.json
python benchmarks/aes/resident_cam.py kinematics build/resident-cam build/resident-cam/kinematics.json --deflection-um .25
python benchmarks/aes/resident_cam_figure.py results/aes/resident-cam
```

The figure shows the published kinematic displacement, the 20 s closed-loop displacement, the signed GPU displacement error and switch-time errors for the zero-gap case. Panel d compares output-bracket midpoints with offline geometric refinement. Its shaded area spans half an output step on either side of zero. The displacement plot uses samples every 2 ms. The reported norms use all 160,001 outputs. [Vector figure](cam-validation.pdf) · [Kinematic curve data](kinematics-curves.npz) · [Dynamic curve data](zero-gap-curves.npz)
