# CAD face tracking on rigid meshes

The inputs are the two complete meshes from the zero-gap gear comparison (70,960 and 37,700 triangles). Each triangle carries the face ID of the original STEP solid. The exporter checks the vertices and triangle order against the archived meshes when reconstructing these labels. The detector takes only two face-tagged meshes and relative rigid transforms. The tooth count, shaft axes, pressure angle, candidate teeth and applied loads are absent from its query interface.

## Search and tracking

The reference traverses the pinion triangle BVH for each wheel triangle and returns the identities of all intersecting pairs. The spatial control groups nearby triangles into patches. The CAD variant groups triangles within each original face before forming patches of up to 16 triangles. Both use the same leaf size limit, oriented patch bounds, hierarchy, parallel triangle intersection tests and face-event output. Spatial grouping produced 7,272 pinion patches and 3,908 wheel patches. Grouping within CAD faces produced 6,394 and 3,822, respectively.

The fresh CAD search rebuilds the candidate patch list every pose. The fixed-margin control pads its search by 1 mm and reuses the list while the displacement of every corner of the moving body's bounding box stays within that margin. The directional variant extrapolates the most recent relative rotation increment and center translation for four poses. For each moving patch it bounds the current box and all four predicted boxes in the current patch axes, adding 0.15 mm of slack. Before each reuse it checks whether **every** current patch is contained within its saved envelope. A failed containment test triggers a whole-model candidate rebuild at that pose. The narrow phase always uses the current triangles. The margin controls only candidate discovery.

At one pose, if the moved patch is contained in its saved envelope, every triangle of that patch is contained as well. A fixed patch excluded by a separating axis from the envelope can therefore be skipped at that pose. This argument applies to the tessellated surfaces at the sampled poses. Continuous collision detection and CAD tessellation errors require separate checks.

## Evidence and timing

The startup window covers 211 accepted 20 µs poses, including the first impact and release. The 100 µs window covers 335 accepted poses over 33.4 ms at a nominal 1,800 rpm. The HHT input retains all 806 actual force-evaluation poses from this second window, in calling order, including successive Newton trials at the same physical time. The final trials at all 335 step times match the accepted trajectory. These are geometry replays. The integration that produced the inputs used the established gear solver.

The benchmark compares the **entire sorted set** of intersecting triangle IDs at every pose to the reference traversal, then checks all active, newly entered and departed CAD face IDs. Each timed run starts with an empty history. Wall timings use device-resident inputs and include synchronization, cache refreshes and capacity recovery. `mean_ms` ends when triangle IDs are available. `mean_with_face_updates_ms` includes the compact face-ID lists. Sorting and copying the full reference data for verification take place in an untimed sweep. The optional event summary counts faces having at least one intersecting pair with outward normals whose dot product is below −0.95. This reporting rule identifies opposing flanks on this example and never filters the collision search.

At seven checkpoints around initial engagement, release, single-to-double and double-to-single contact, a separate OCCT query checked all candidate original faces within 1 nm. It resolved every face-distance query. The complete sets of STEP face pairs and mesh-derived face pairs agreed at those checkpoints. This independent check is sampled. [Original CAD comparison](native-faces.json)

At 100.1 ms the opposing flanks carried the original STEP face IDs (4, 148). The search found (142, 154) at 100.2 ms while the first pair was still active. At 101.3 ms the first pair had left and the second remained. Over the revolution the same reporting rule recorded 24 entries after initialization and 24 exits.

### Whole-model search

Mean wall milliseconds per pose include producing the three face lists. Five interleaved timed passes followed one verification pass. Immediately before each RTX 3090 run the shared card reported 100% utilization. Compare methods within each run. The workstation RTX 3070 gives a second measurement. [Server device samples](rtx3090-device.csv)

| GPU and inputs | Triangle BVH | Spatial patches | CAD face patches, fresh (ours) | CAD patches, 1 mm reuse | CAD patches, directional reuse |
| --- | ---: | ---: | ---: | ---: | ---: |
| [3090, startup](startup-3090.json) | 10.145 | 6.336 | 5.480 | 5.990 | 5.024 |
| [3090, revolution](revolution-3090.json) | 11.070 | 6.812 | 5.535 | 6.790 | 5.274 |
| [3090, HHT trials](hht-3090.json) | 11.043 | 6.820 | 5.539 | 5.756 | 5.040 |
| [3070, startup](startup-3070.json) | 4.469 | 3.412 | 2.466 | 2.843 | 2.302 |
| [3070, revolution](revolution-3070.json) | 5.510 | 3.821 | 2.832 | 4.086 | 3.074 |
| [3070, HHT trials](hht-3070.json) | 5.918 | 4.071 | 3.647 | 3.526 | 3.711 |

Spatial grouping reduced total query time on both cards. Within the same patch implementation, forming patches along STEP face boundaries saved a further 19% on the 3090 HHT input and 10% on the 3070 HHT input. Directional reuse reduced candidate-list rebuilds from 806 to 113 on both devices. Its wall-time benefit varied by device and input. Even the fastest mean per force-evaluation pose exceeded the 0.100 ms budget for an entire real-time step.

### Reusing candidates during Newton corrections

In the HHT window the largest displacement of a corner of the moving body's local bounding box was 0.03745 mm across successive trials at the same physical time. Across successive physical times it was at least 3.879 mm. These observations motivated a second paired experiment with a fixed 0.05 mm search margin. It also uses a previously unused window of 807 force evaluations from 240 to 273.4 ms during brake-torque reduction. Both windows contain 335 distinct physical times. The narrow phase uses zero contact activation in every run.

Four interleaved timed passes on each window gave the following mean milliseconds per pose, including the face updates. Each row uses its own fresh-search control. All four paired comparisons reproduced every triangle and face identity from the full reference. Both cached runs rebuilt the candidate list once per physical time.

| GPU and HHT inputs | Fresh CAD patches (ours) | With 0.05 mm reuse (ours) | Time saved | Candidate rebuilds, fresh to cached |
| --- | ---: | ---: | ---: | ---: |
| [3090, revolution](hht-small-3090.json) | 5.543 | 5.129 | 7.5% | 806 to 335 |
| [3090, torque reduction](release-small-3090.json) | 5.539 | 5.125 | 7.5% | 807 to 335 |
| [3070, revolution](hht-small-3070.json) | 2.070 | 1.845 | 10.9% | 806 to 335 |
| [3070, torque reduction](release-small-3070.json) | 3.066 | 2.759 | 10.0% | 807 to 335 |

## Reproduce

Use the CUDA and OCCT environments described in the [parent benchmark](../README.md). With `M=results/aes/mesh-comparison/meshes` and `D=results/aes/online-gear/tracking`, replay one window as follows. The other two input files are `startup.txt` and `hht.txt`.

```sh
python benchmarks/aes/check_patch_tracker.py
python benchmarks/aes/contact_tracking_replay.py "$M/mesh-0.04.npz" "$M/wheel-0.4.npz" "$D/faces-a.npz" "$D/faces-b.npz" "$D/revolution.txt" revolution.json --repeats 5 --guards 0 1 --lookahead 4 --slack-mm 0.15
python benchmarks/aes/contact_tracking_replay.py "$M/mesh-0.04.npz" "$M/wheel-0.4.npz" "$D/faces-a.npz" "$D/faces-b.npz" "$D/release.txt" release-small.json --patches-only --repeats 4 --guards 0 0.05 --lookahead
python benchmarks/aes/contact_tracking_cad_check.py build/occt/Release/nc_contact.exe results/gears/models/spur-1a.step results/gears/models/spur-1b.step "$M/mesh-0.04.npz" "$M/wheel-0.4.npz" "$D/faces-a.npz" "$D/faces-b.npz" native.json --window "$D/startup.txt" 10,11,19 --window "$D/revolution.txt" 1,2,12,13
```

The `owners` command of `benchmarks/aes/contact_tracking_inputs.py` regenerates the face tags using the original STEP files and deflections (0.04 µm for the pinion and 0.4 µm for the wheel). Its `poses` command takes the gear parts CSV, shaft trace and output path. Use 0–0.0042 s for the startup input and 0.1–0.1334 s for the revolution and HHT inputs. The additional HHT trial-state export can be reproduced with the following command.

```sh
build/occt/Release/nc_gears.exe run spur results/gears/models 0 9.999999999999999e-05 .134 build/aes-zero-gap/tracked-states 1 31 1 fixed zero-gap build/aes-zero-gap/hht-trials.csv
build/occt/Release/nc_gears.exe run spur results/gears/models 0 9.999999999999999e-05 .274 build/aes-zero-gap/release-states 1 31 1 fixed zero-gap build/aes-zero-gap/release-trials.csv
python benchmarks/aes/contact_tracking_inputs.py poses results/gears/models/spur.csv build/aes-zero-gap/release-trials.csv "$D/release.txt" --start .24 --end .2734
```
