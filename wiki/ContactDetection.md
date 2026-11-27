# CPU contact mesh preparation

The [rigid contact backend roadmap](RigidContactBackend.md) pins the CPU reducer baselines and records the current experimental GPU implementations and remaining CAD work.

`apps/prepare_contact_mesh.py` imports a closed STEP solid through Gmsh's OpenCASCADE kernel, meshes its boundary with first-order triangles and writes an ASCII Gmsh 2.2 file in metres. The elementary entity tags identify the originating Gmsh faces and edges. The input model can declare its STEP length unit. Both the command line and the output use SI lengths.

Install the Python dependencies with `python -m pip install gmsh numpy libigl meshio`. The last package is used by the round-trip test.

```powershell
python apps/prepare_contact_mesh.py results/gears/models/spur-1a.step build/cpu-contact/pinion.msh --size-m 0.0015
python tests/test_contact_mesh.py
```

The sibling `.quality.json` contains hashes of the STEP and mesh files, the Gmsh version, preparation time, element counts, winding volume, minimum element angle, minSICN statistics, sampled CAD deviations and normal angles. `--max-error-m`, `--max-normal-deg` and `--min-sicn` enforce application-specific quality limits. A failed quality check preserves a previously generated mesh. A consumer reusing a cached mesh should match its SHA-256 against its quality report.

Each triangle contributes a centroid and three edge midpoints to the comparison with its supporting CAD surface. Each segment on a CAD edge also contributes its midpoint to a curve-distance check. These measurements are samples. The normal error is the angle between tangent planes, measured per CAD face. Topological checks cover duplicate or degenerate triangles, edge incidence, consistent winding, vertex manifoldness and self-intersection. Mesh refinement and an independent dynamics reference are needed to relate these measurements to a required response accuracy.

When a requested minSICN fails on a long narrow face, the generator measures its area and perimeter, increases edge subdivision along its length and remeshes. If neighboring triangles still miss the threshold, it applies a graded Gmsh size field around the affected curves. The final quality checks can reject this result. Each output records the number of edges refined.

For surface contact, a planar polygon can be covered by a few triangles while preserving the same plane and boundary. With `--compact-planar-solids`, Gmsh can use the solid's bounding-box diagonal as its size limit when every CAD face is planar and every trim edge is straight. Explicit CAD edges, watertightness, element quality and distance checks still apply. Solids containing curved faces or edges use the supplied size target.

The eccentric cam fixture uses this setting by default. Run `python benchmarks/aes/cpu_mesh_cam.py mesh MODEL_DIR MESH_DIR` after preparing the model. The cam retained its original 1,414 triangles. The follower went from 1,470 to 12 triangles. Across 3.2 s, all 3,201 contact states agreed, with a maximum follower displacement difference of 1.02 µm and a maximum load difference of 0.011 N. Both meshes reproduced the 71 switches of the smooth-cylinder reference in order, with follower displacement RMS error of about 0.574 mm. [Compact result](../results/aes/cpu-mesh/cam/compact-planar.json) · [Uniform-size control](../results/aes/cpu-mesh/cam/compact-control.json)

Using a 10.5 mm size target on the cam gave 2,360 curved-body triangles and 12 follower triangles. The cam's maximum sampled CAD deviation fell from 348 to 211 µm, but the simulation produced 69 switches and a follower displacement RMS error of 0.627 mm. The 14 mm target remains the lightweight baseline. [Curved-surface refinement trial](../results/aes/cpu-mesh/cam/compact-10p5mm.json)

## Default spur gear mesh

Generate the default lightweight CPU baseline and replay its archived poses from the repository root.

```powershell
python benchmarks/aes/prepare_spur_cpu.py build/cpu-contact/spur-light
python benchmarks/aes/cpu_mesh_replay.py build/cpu-swept/Release/nc_mesh.exe build/cpu-contact/spur-light/spur-1a.msh build/cpu-contact/spur-light/spur-1b.msh results/aes/cpu-mesh/spur-poses.txt build/cpu-contact/spur-light/replay.json --compare-overlap
```

The default `light` profile keeps the boundary curve discretization and mapped faces of the compact controls. Gmsh fills the remaining faces at 20 angular samples per circle. Comparing triangle vertices face by face found that only the cylindrical bore on each gear changed. Every tooth triangle and CAD-edge segment matches the compact mesh. The two gear files occupy 1.28 MiB and the four reducer files occupy 3.92 MiB. Rebuilding the reducer reproduced all four mesh hashes. Both 5 ms HHT response files (101 saved frames each) match the compact controls byte for byte. All 32 prescribed gear-pair poses also match the dense mesh's overlap decisions. The largest sampled surface deviations anywhere on the pair and reducer are 376 and 414 µm. A simulation that loads the bore needs its own resolution check there. [Mesh settings, changed faces, hashes, RAM, dynamics and pose comparisons](../results/aes/cpu-mesh/memory/light-mesh-sweep.json)

| Geometry | Light triangles | Peak RAM in preparation | Peak RAM in simulation | Preview triangles | Preview simulation peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spur pair | 18,742 | 371.0 MiB | 30.1 MiB | 6,474 | 21.3 MiB |
| Two-stage reducer | 58,974 | 438.6 MiB | 58.2 MiB | 22,112 | 32.5 MiB |

The `preview` profile coarsens both the mapped faces and their edges. On the spur pair it missed 22 contacts in the 32 prescribed poses relative to the dense control. Across the loaded reducer run the first and second stage contact states differed from the compact control at 44 and 36 of the 101 outputs. An intermediate trial with 14,562 triangles on the spur pair agreed on all 32 prescribed poses but changed contact activity at 22 of the 101 dynamics outputs.

To reproduce the previous compact controls, pass `--mesh-profile compact`. This profile uses 64 angular samples per circle on straight-gear edges and 96 on helical-gear edges. Gmsh fills the face interiors at 40 or 60 angular samples, respectively. The same two meshing passes are available for other STEP solids through `--edge-curvature-samples` and `--curvature-samples`. The finished compact assets satisfy a 160 µm sampled CAD deviation limit and an 8° normal-angle limit.

| Mechanism | Reference → compact triangles | Compact `.msh` files | Peak RAM during preparation | Peak RAM in loaded CPU simulation |
| --- | ---: | ---: | ---: | ---: |
| [Spur pair](../results/aes/cpu-mesh/memory/compact.json) | 23,470 → 20,456 | 1.39 MiB | 372.6 MiB | 30.7 MiB |
| [Two-stage reducer](../results/aes/cpu-mesh/memory/reducer-compact.json) | 76,692 → 65,732 | 4.36 MiB | 440.7 MiB | 61.3 MiB |

RAM denotes Windows process peak working set. Preparation includes STEP loading, meshing and the topology and sampled CAD audits. The archived meshes took 62.6 s and 154.7 s to prepare on a shared machine. Comparing individual segments and triangles by their vertex coordinates showed that each gear changed only on the cylindrical bore. The largest measured surface deviation there was 84.9 µm. All CAD-edge segments, end faces and tooth triangles matched the reference assets. These four gears mesh with one another at their teeth. A model loaded at its bore needs a separate check at that surface. [Per-asset geometry and memory measurements](../results/aes/cpu-mesh/memory/compact-preparation.json)

On the 5 ms, 100-step loaded runs, both compact response CSVs matched their respective reference files byte for byte. At all 101 prescribed poses per stage, the region counts, depths, normals and centroids also matched. The compact spur pair matched the dense mesh's overlap decisions on the 32-pose replay. The compact reducer spent 106.03 s of its 115.14 s integration constructing intersection solids and 3.81 s detecting overlap. [Spur pose replay](../results/aes/cpu-mesh/memory/spur-stage-1-compact-poses.json) · [Reducer stage 1](../results/aes/cpu-mesh/memory/reducer-stage-1-compact-poses.json) · [Reducer stage 2](../results/aes/cpu-mesh/memory/reducer-stage-2-compact-poses.json)

A separate 32/48-sample trial also relaxed the edge discretization on unmapped faces. The 61,182-triangle reducer changed contact activity at 38 and 30 of 101 output times on its two stages, respectively. The 19,008-triangle spur pair changed at three output times. [Reducer response](../results/aes/cpu-mesh/memory/coarse-edges-reducer.json) · [Spur response](../results/aes/cpu-mesh/memory/coarse-edges-spur.json)

### Previous reference mesh

Add `--mesh-profile reference` to reproduce the following measurements. In this archive the 23,470-triangle pair was called the routine mesh.

Both gears are closed solids. Gmsh uses compatible edge subdivisions on four-sided CAD faces, set by the lengths and sampled curvature of opposite edges. The routine mesh spends triangles along curved profiles and allows long triangles in straight directions. Other faces use the Gmsh surface mesher with 64 curvature samples per full circle. The recipe uses a 2 µm sampled chord target on mapped edges. Across the finished solids the largest sampled surface deviations were 19 and 37 µm at the bores. The CAD trim on the wheel permits some very slender contact triangles. The quality report retains their measured shape and the checks for watertightness, orientation and self-intersection.

The two ASCII mesh files occupy 1.58 MiB. One sequential preparation of both meshes, including the CAD and topology audits, took 44.5 s and reached a peak working set of 369.4 MiB in the Python process. The runtime loads the resulting meshes. [One-time preparation measurement](../results/aes/cpu-mesh/memory/standard-preparation.json)

| Mesh for the complete pair | Triangles | OBB detection peak | Loaded HHT peak | Poses matching the dense mesh |
| --- | ---: | ---: | ---: | ---: |
| [Routine](../results/aes/cpu-mesh/memory/standard.json) | 23,470 | 14.4 MiB | 36.1 MiB | 32/32 |
| [Refined](../results/aes/cpu-mesh/memory/refined.json) | 69,144 | 33.7 MiB | 66.9 MiB | 32/32 |
| [Uniform 6 mm target](../results/aes/cpu-mesh/memory/6mm.json) | 25,544 | 16.0 MiB | — | 10/32 |
| [Historical dense pair](../results/aes/cpu-mesh/memory/dense.json) | 1,139,266 | 485.0 MiB | — | Reference |

These are Windows working-set measurements of the entire native process, including mesh loading and acceleration structures. Detection replays 32 archived poses, with 31 intersections on the dense pair. HHT integrates 5 ms in 100 steps at 50 µs with the same scene and penalty law. On the routine and refined meshes the peak normal loads were 5.263 and 5.109 kN and the impulses were 5.041 and 5.000 N·s. Contact activity differed at 16 of 101 output times and the force RMS difference was 873 N. Use the routine pair for detection and development. The results leave the pointwise tooth load unresolved. [Comparison of the two time histories](../results/aes/cpu-mesh/memory/refinement-check.json)

The three CPU broad phases also produced identical triangle witnesses and enclosure decisions on the routine mesh at all 32 poses. The archive includes the [per-frame replay](../results/aes/cpu-mesh/memory/standard-replay.json), the [routine force history](../results/aes/cpu-mesh/memory/standard-response.csv) and the [refined force history](../results/aes/cpu-mesh/memory/refined-response.csv).

## Historical dense geometry audit

`--audit-workers N` distributes face samples over independent Gmsh sessions. With the optional `cadquery-ocp` package installed, the default `--projection auto` uses the B-spline control net to bound the Hessian of squared point-to-surface distance. Where the objective is strictly convex, the kernel solves in surface parameters and evaluates the point and normal on the Gmsh CAD face. For partitioned faces it also checks distance lower bounds on all competing subpatches. Unresolved samples, unsupported surfaces and installations without OCP use Gmsh closest-point queries. Select `--projection gmsh` to use those queries throughout. Both modes check the centroid and three edge midpoints of every triangle, evaluate both incident facet normals at shared samples, and check CAD feature edges separately. These are sampled errors against the originating surfaces and curves.

In the [spur pair preparation trial](../results/aes/cpu-mesh/spur-preparation.json), the 50,060-triangle pinion passed and the coarse wheel had a minSICN of 0.00429. The old audit timed out on the refined wheel with both one and six workers. With the original grading width, the accelerated audit checked 8,083,976 face samples on 2,020,994 triangles. Face checking took 52.34 s, topology checks took 37.24 s and the whole preparation took 367.70 s. The narrower grading field first lets the target edge size recover over 1.25 nominal element lengths. If a neighboring element fails the shape threshold, the mesher retries with the previous four-length width.

The [paired projection check](../benchmarks/aes/cpu_mesh_audit_comparison.py) compares coordinates and CAD normals at the same positions. On [all 276,936 coarse-wheel samples](../results/aes/cpu-mesh/wheel-projection-coarse.json), the maximum difference in projected position was 5.84 pm. The [refined-wheel comparison](../results/aes/cpu-mesh/wheel-projection-refined.json) used 32 triangles on each of the 291 faces. The full preparation checks all triangle samples, including those outside this independent comparison.

The dense wheel has 1,089,206 triangles. Its peak sampled deviation is 10.207 µm, its maximum facet-normal error is 2.029° and its minimum SICN is 0.21682. Two preparations produced the same mesh hash in 304.37 s and 637.90 s on the shared Ryzen 9 5900X machine. The latter run spent 29.89 s through the initial mesh, 441.72 s on refinement and element quality, and 131.46 s on CAD and topology checks. The following commands reproduce the historical dense reference pair.

```powershell
python apps/prepare_contact_mesh.py results/gears/models/spur-1a.step build/cpu-contact/spur/spur-1a.msh --size-m .0015 --min-sicn .3 --max-error-m .00005 --max-normal-deg 8
python apps/prepare_contact_mesh.py results/gears/models/spur-1b.step build/cpu-contact/spur/spur-1b.msh --size-m .001 --min-sicn .2 --max-error-m .00005 --max-normal-deg 8 --audit-workers 3
```

## Discrete triangle contact

Build the optional CPU mesh target with FCL 0.7 and CGAL 6. In a vcpkg environment on Windows:

```powershell
vcpkg install fcl:x64-windows cgal:x64-windows
cmake -S . -B build/cpu-mesh -DNC_WITH_CPU_MESH=ON -DNC_WITH_OCCT=OFF "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build build/cpu-mesh --config Release
ctest --test-dir build/cpu-mesh -C Release --output-on-failure
```

`nc_mesh --method mesh-aabb A.msh B.msh` takes one frame per input line with `x y z qw qx qy qz` per body (metres, unit quaternion). Additional mesh arguments add bodies. Repeated mesh paths share the local tree. Each output line contains the body and triangle indices of each intersecting pair and its intersection points in world coordinates. One point denotes a touch, two points a segment and three or more a coplanar polygon. A separate list identifies a contained component by body, enclosing body and representative vertex. The program also reports the body candidates, node tests, triangle tests and elapsed query time.

| Method | Body pairs | Triangle hierarchy |
| --- | --- | --- |
| `mesh-aabb` | FCL dynamic AABB tree | World AABBs enclosing placed local AABBs |
| `mesh-obb` | FCL dynamic AABB tree | Rigidly placed, locally fitted OBBs |
| `mesh-sap` | Sweep and prune on the axis of greatest body-center spread, followed by overlap on all three axes | Same local AABB tree as `mesh-aabb` |

The leaf predicate and intersection construction use the same CGAL exact kernel on the binary64 world vertices in every branch. The public witnesses are rounded to binary64. The mesh input is a closed triangular boundary. Initial containment uses an inside test for each connected boundary component without a crossing. Output triangle pairs describe geometry; individual triangle hits carry no individual spring.

Before invoking the point-in-solid test, the detector checks six actual extreme vertices of the candidate component against the other solid's local bounds. A single exterior vertex excludes enclosure of the connected boundary when the surfaces have no crossing. On the [32 recorded gear poses](../results/aes/cpu-mesh/spur-poses.txt), this removed an expensive first query on a separated pair. With the 2,020,994-triangle wheel, the OBB first-pose time fell from 6.97 s to 22.46 ms. Every per-frame intersection witness and enclosure agreed with the previous build for all three methods. With the compact wheel the median OBB query was 123.66 ms and 31 of 32 poses contained surface intersections. The [timing report](../results/aes/cpu-mesh/spur-detection.json) records single per-frame measurements on the shared host. These times exclude mesh loading and cover discrete detection through witness construction.

```powershell
python benchmarks/aes/cpu_mesh_replay.py build/cpu-swept/Release/nc_mesh.exe build/cpu-contact/spur/spur-1a.msh build/cpu-contact/spur/spur-1b.msh results/aes/cpu-mesh/spur-poses.txt build/cpu-contact/spur/replay.json
```

Add `--contours` to a discrete query to group connected triangle intersections within each body pair. Groups meet at exact world-space witnesses on triangle boundaries. A group can contain a point, a curve, or coplanar polygons. A contact region with holes can have several boundary curves. Force integration and penetration depth require further geometry.

Add `--any` for a Boolean overlap query. The detector retests the previous intersecting triangle pair at the new poses, then searches the hierarchy until an intersection is found. When the boundaries are disjoint it still tests for enclosure. Penalty dynamics uses this query before constructing the overlap volume. The solver reuses computed force-output data only at an identical time, position and velocity for compatible force laws. On the 3.2 s [coarse eccentric cam](../results/aes/cpu-mesh/cam/current-coarse-mesh-obb-0.001-3.2.json), the two executables produced byte-identical 3,201-frame histories over 71 contact switches. The [original run](../results/aes/cpu-mesh/cam/reference-coarse-mesh-obb-0.001-3.2.json) took 145.0 s of integration and the updated run took 96.8 s.

For a closed transverse loop, `vector_area_a_m2` is half the directed sum of cross products of consecutive boundary vertices. Segment orientation follows the cross product of the two outward face normals. This gives the oriented area vector of the bounded surface on body A, and reverses on exchanging the bodies. Tangencies, coplanar regions, and branching boundaries can lack a unique oriented loop. The field is null for these cases. The magnitude is a projected area for a planar patch.

## Continuous surface detection

In the CPU mesh build, `NC_WITH_MESH_CCD` enables continuous queries and fetches the pinned Tight Inclusion and spdlog sources. Use `-DNC_WITH_MESH_CCD=OFF` for a discrete-only build.

```text
nc_mesh --method mesh-obb --detection continuous --tolerance-m 1e-7 A.msh B.msh
```

Each pair of consecutive input frames defines one motion interval. Body origins interpolate linearly and orientations follow the shortest quaternion arc. Supply intermediate poses for larger rotations or curved origin paths. The output identifies the earliest retained surface candidate by its time fraction and triangle pair. Run the discrete query at the first pose to identify components already entirely inside a solid.

For a pair with constant orientations, exact rational separating-axis projections give the triangle intersection time for the specified translations. Coplanar pairs include the in-plane edge normals. Rotating pairs use swept boxes expanded by a circular-arc deviation bound. Subintervals can be discarded using midpoint triangle distance and maximum rigid displacement. Small arcs use Tight Inclusion vertex–face and edge–edge tests with both bodies' chord deviations. A candidate that exhausts the primitive solver's accuracy budget is subdivided further. Spatial tolerance can admit a close pass. It describes the mesh query and does not bound the deviation from the original STEP surface.

## Last verified against

On 2026-11-25, the Release CAD solid tests compared the Bézier finite-ray kernel
with OCCT on a positive-weight rational face and checked a removed trim,
a UV corner, tangency and a two-root span. The reducer replay covers the
two gear pairs with identical input poses. The dynamics comparison advances
each method independently for 50,000 steps.
[Regression cases](../tests/cad_solids.cpp) · [Benchmark](../results/aes/cpu-bezier/README.md)

`tests/test_contact_mesh.py` checks sphere size and volume in the written file, convergence of the sampled surface error, narrow-face refinement, topology failures and preservation of previous outputs after a quality rejection. The native tests check the discrete search against exhaustive triangle pairs and cover coplanar intersections, full containment, translation through a solid, rotating contact with clear endpoints and a separated rotating path.

The [two-sphere replay](../benchmarks/aes/cpu_mesh_spheres.py) uses all 641 poses of the legacy 250 µs HHT trace and its 41 mm radius contact surfaces. It checks every discrete intersection witness across all three search methods, then sweeps the entry interval forward and the exit interval backward. On the same interpolated motion, smooth spheres enter at 40.000 ms and separate at 71.418 ms.

| Gmsh target size | Triangles per sphere | Mesh entry | Mesh separation |
| --- | ---: | ---: | ---: |
| [3 mm](../results/aes/cpu-mesh/spheres-3mm.json) | 5,708 | 40.129 ms | 71.289 ms |
| [1.5 mm](../results/aes/cpu-mesh/spheres-1p5mm.json) | 22,482 | 40.040 ms | 71.378 ms |

At the frames with the most intersecting triangle pairs, the coarse mesh has 128 pairs and the fine mesh has 250. Grouping the exact boundary witnesses gives one connected intersection in both cases. These comparisons measure event timing and connectivity on prescribed poses. A mesh-driven force response requires a separate simulation.

At the same maximum-compression pose, the sphere intersection projects to a disk of area `π(R² − d²/4)`, where `d` is center spacing. The geometric reference area is 563.708 mm². The 3 mm mesh gives 556.942 mm² and the 1.5 mm mesh gives 561.967 mm². The angles between the loop area vector and the line of centers are 0.00640° and 0.000255°, respectively. Both bodies use the same triangulation at the same orientation for this translational trajectory.

## Mesh-driven penalty dynamics

```text
nc_contact MODEL.scene 0 DT_S DURATION_S STRIDE OUTPUT --method mesh-obb --mesh-dir MESH_DIRECTORY --solver penalty
```

Use the same model, material and initial conditions across `mesh-aabb`, `mesh-obb` and `mesh-sap`. Each body resolves its scene asset to a matching relative Gmsh filename. The v2 pair record `surface A B stiffness exponent damping friction slip_speed maximum_depth` selects contact when the surfaces meet. Existing surface probe records also work with the mesh methods. Add `--event-tol-s 0.00005 --ccd-tol-m 1e-7` to check for step-interior entry and release. Both-endpoints-active checks currently track the pair as a whole. They do not detect the birth of another disconnected patch on an already active pair.

At each HHT force evaluation the selected hierarchy tests boundary intersection and enclosure. For positive candidates the CPU backend constructs their Boolean intersection. Each connected overlap volume contributes one contact at its volume centroid. The difference of the area-weighted outward normals on the two source boundaries gives the loading direction. Depth is the overlap width along that direction. Both bodies receive equal and opposite forces at the same point. A fully enclosed or symmetric overlap with no resolved direction reports a failure.

### Local construction of the overlap volume

`--mesh-region-scope local` is the default. When one side of a body lies beyond the other body's world bounding box, the backend can remove whole triangles there and close the retained surface. Contact-side triangles keep their original vertices. A candidate cap must give a closed, consistently oriented, manifold solid without self-intersections. At every new pose the backend checks that all vertices of removed triangles and the cap tip still lie strictly beyond the other body's bounding box on the selected axis. The tip must also remain inside its own body's original world box. These tests keep the two bodies' changes disjoint from the possible overlap even when both solids are reduced. Ineligible cuts use the full solid. The discrete detector and swept triangle tests always use the original meshes.

`--mesh-region-scope full` constructs the overlap from both complete solids with the same force law. Both paths convert exact intersection vertices to floating point once and order facets by coordinates for moment accumulation. Use `nc_mesh --method mesh-obb --regions local A.msh B.msh` for a two-body pose replay. Each seven-value region contains its centroid (m), unit normal and depth (m). Substitute `full` to compare the same poses with complete solids.

On the [32 prescribed spur gear poses](../results/aes/cpu-mesh/local-regions/spur-32-poses.json) all region counts, centroids, normals and depths matched. On the [5 ms loaded gear simulation](../results/aes/cpu-mesh/local-regions/spur-comparison-0.005-5e-05.json) both scopes produced identical response files. The [full run](../results/aes/cpu-mesh/local-regions/spur-full-0.005-5e-05.json) processed 23.52 million input faces over 1,002 Boolean calls. The [local run](../results/aes/cpu-mesh/local-regions/spur-local-0.005-5e-05.json) processed 5.93 million. These are cumulative counts across force trials on the same 23,470-triangle mesh.

The [3.2 s eccentric cam run](../results/aes/cpu-mesh/cam/localized-coarse-mesh-obb-0.001-3.2.json) produced the same 3,201 output rows as its [full-solid control](../results/aes/cpu-mesh/cam/canonical-full-coarse-mesh-obb-0.001-3.2.json). Both reproduced all 71 contact switches of the smooth-cylinder reference in order. A [gear step-size check](../results/aes/cpu-mesh/local-regions/spur-step-comparison.json) also compares the 250 µs and 50 µs runs on 21 common output times. They disagree on contact activity at 15 times. Larger steps materially change this loaded response.

The [paired spur timings](../results/aes/cpu-mesh/local-regions/spur-timing-pairs.json) reverse the run order on the shared CPU. The second pair took 50.65 s for full solids and 34.77 s for local solids. The local integration spent 0.31 s on discrete detection and 32.63 s constructing overlap regions.

For the archived two-stage reducer, run `python benchmarks/aes/prepare_spur_cpu.py build/cpu-contact/reducer-standard --case reducer --mesh-profile reference`. Four Gmsh meshes have 76,692 triangles in total. The front pair is straight and the rear pair helical. Two separate preparations reproduced all mesh hashes. An [initial single-cut trial](../results/aes/cpu-mesh/local-regions/reducer-local-5ms.json) used local solids on the [first stage](../results/aes/cpu-mesh/local-regions/reducer-stage-1-poses.json) and full solids on the [second stage](../results/aes/cpu-mesh/local-regions/reducer-stage-2-poses.json).

The backend can now move a cut farther from the opposing mesh when a narrow cut leaves disconnected strips. It tries up to four positions and applies the same closure and separation checks each time. All 40 loaded poses of the [helical stage](../results/aes/cpu-mesh/local-regions/reducer-stage-2-adaptive-poses.json) used at least one local solid. All region counts, centroids, normals and depths matched the full-solid replay on both this stage and the [straight stage](../results/aes/cpu-mesh/local-regions/reducer-stage-1-adaptive-poses.json).

The [adaptive-cut run](../results/aes/cpu-mesh/local-regions/reducer-adaptive-local-5ms.json) and [full-solid control](../results/aes/cpu-mesh/local-regions/reducer-adaptive-full-5ms.json) each integrated 5 ms in 100 main steps. The resulting [three shaft speeds and both stage loads](../results/aes/cpu-mesh/local-regions/reducer-response.csv) matched byte for byte, including 30 output frames with both stages under load. Integration took 111.28 s and 167.41 s, respectively, on the shared CPU. This short segment measures the contact pipeline through loaded gear transitions.

The same scene also ran with [AABB traversal](../results/aes/cpu-mesh/local-regions/reducer-mesh-aabb-5ms.json) and [sweep and prune](../results/aes/cpu-mesh/local-regions/reducer-mesh-sap-5ms.json). All three methods produced the same response file. AABB, OBB and sweep and prune took 162.89 s, 111.28 s and 125.83 s of integration in these single runs. The two broad phases that use local AABBs ran on the same four meshes as OBB.

```powershell
python benchmarks/aes/cpu_mesh_gear_dynamics.py build/cpu-swept/Release/nc_contact.exe build/cpu-contact/reducer-standard/reducer.scene build/cpu-contact/reducer-standard/run build/cpu-contact/reducer-standard/run.json --method mesh-obb --scope local --dt 0.00005 --duration 0.005
```

The [impact benchmark](../benchmarks/aes/cpu_mesh_impact.py) integrates both slider masses from their initial state. The independent linear-spring solution gives an entry time of 40 ms, an impulse of 0.45 N·s and a peak of 22.5 N. On a fixed mesh the three broad phases produced matching response histories in these runs.

| Target size | HHT step | Entry error | Force RMSE | Impulse | CPU integration (OBB) |
| --- | --- | --- | --- | --- | --- |
| [3 mm](../results/aes/cpu-mesh/impact/3mm-mesh-obb-250us.json) | 250 µs | 128.9 µs | 0.07862 N | 0.449976 N·s | 15.72 s |
| [1.5 mm](../results/aes/cpu-mesh/impact/1p5mm-mesh-obb-250us.json) | 250 µs | 40.2 µs | 0.01917 N | 0.450005 N·s | 23.82 s |
| [1.5 mm](../results/aes/cpu-mesh/impact/1p5mm-mesh-obb-125us.json) | 125 µs | 40.2 µs | 0.01871 N | 0.450001 N·s | 46.04 s |

Each run spans 0.16 s. The timings are single measurements on the local CPU. The [entry/release guarded run](../results/aes/cpu-mesh/impact/3mm-mesh-obb-250us-guard.json) used 648 acceptance checks and subdivided four trials. Its entry error remained 128.9 µs at the coarse mesh resolution.

## Contact on STEP solids

For surface contact on the original STEP solids, `nc_contact --method cad-occt` selects CAD intersection regions. The `chen-2024` and `chen-2026` methods also test moving surface patches for possible contact. Both use a default event time tolerance of one fifth of the main step. The 2024 method requires `NC_CHEN_SOURCE_DIR` at build time. On the same scene `--method mesh-obb` reads Gmsh files alongside the STEP parts or under `--mesh-dir`. Select the response independently with `--solver`. The named options follow the scene, deflection, step, duration, stride and output arguments. [Algorithm, supported motions and reproduction](../benchmarks/chen/README.md#rigid-step-scenes)

The three STEP methods verify event candidates on the original trimmed CAD geometry. Above 16 faces per body pair, a cached BVH prunes OCCT vertex, edge and face distance queries using the motion bound for the current step. It also checks shell containment in either direction after excluding boundary contact. The solver records box tests, feature extrema and exact-pose cache hits.

With at least 64 faces in the pair, directional bounds enclose the material that can reach the other body. OCCT cuts each solid outside this region and its safety margin. The solver checks each new pose against the retained window before reusing a cut. A failed cut uses the full solid. Event advancement also bounds relative normal motion within the reachable region. Contact points, normals and penetration depths still come from intersections of STEP solids. The same code handles every body pair without tooth numbers or gear parameters. The bounded search and full OCCT shell search returned 48.7923 µm and zero at two archived spur poses. [Distances and timings](../results/aes/general-contact/spur-local-distances.json) · [Seven-step dynamics comparison](../benchmarks/chen/README.md#gear-distance-and-event-costs)

The two-sphere comparison defaults to 904 triangles per sphere (8 mm target, 17.5 MiB solver peak). In the 0.16 s collision, maximum displacement error against the closed-form motion was 29.95 µm. With 462 triangles the peak fell to 17.2 MiB, displacement error rose to 474.42 µm and first load fell between 41.00 and 41.25 ms against the analytical 40.00 ms. The 904-triangle mesh first loaded between 40.25 and 40.50 ms. The two resolutions gave peak loads of 22.52 and 22.53 N against 22.50 N. [Low-memory impact measurements](../results/aes/cpu-mesh/impact/low-memory.json) · [Refinement controls](../results/aes/general-contact/chen-step-impact.json)

For successive poses of the same boundary trees, the distance query first re-evaluates the previous closest feature pair. A current distance inside the search cap tightens the upper bound used by the BVH. Other feature pairs can replace it, and a rebuilt local tree invalidates both feature indices. The solver checks containment after a separated boundary search. The counters `cad_witness_tests` and `cad_witness_improvements` count seed evaluations and reductions of the initial cap.

The search also uses the vector between a pair of separated witness points to choose a second coordinate frame for each body. It builds boxes from the CAD features in those frames once per boundary index. At later poses it updates the relative transform between frames and tests both sets of bounds. A pair of nodes or features can be discarded when either lower bound exceeds the current upper bound, with a numerical allowance on the secondary boxes. A frame remains a valid enclosure when the nearest feature changes. [Implementation](../src/cad_boundary_distance.cpp) · [Gear comparisons and switch replay](../benchmarks/chen/README.md#reusing-the-closest-features-across-poses)

## Surface contact solvers

All six named surface methods return connected contact regions. Choose a geometry method and response on the same scene, for example `--method mesh-obb --solver ncp` or `--method chen-2024 --solver sap`. Sweep and prune (`--method mesh-sap`) selects broad-phase geometry. The discrete compliant solver (`--solver sap`) selects the contact law. The 2024 option uses the upstream source checkout. The OCCT build includes the independent 2026 variant. [Implementation and labeled comparison](../benchmarks/chen/README.md#safeguarded-envelopes)

| Solver | Normal and friction model |
| --- | --- |
| `penalty` (default) | Scene penalty force and regularized sliding friction, coupled to HHT. |
| `sap` | Convex compliant friction cone with normal stiffness and relaxation time, solved by projected block iterations. The scene's power-law stiffness is linearized at the predicted depth. |
| `lagrange` | Unilateral normal multiplier with a local mass-weighted circular friction-disk solve. |
| `augmented-lagrange` | Proximal multiplier updates with a circular friction disk and a final check of the unregularized constraints. |
| `lcp` | Inelastic normal complementarity and four tangential friction generators. The friction pyramid depends on the tangent basis. |
| `ncp` | Inelastic normal complementarity and maximum dissipation on a circular friction disk. |
| `soccp` | The same circular Coulomb law expressed using a friction cone and the de Saxcé correction to normal velocity. |

The impulse path predicts free motion under the external and gyroscopic loads, obtains the surface geometry at the predicted end pose and couples all active regions using the generalized inverse mass. The resulting impulses correct velocity and position. The contact correction is first order. Saved loads give impulse divided by main-step duration, summed over its accepted substeps. Use output stride one to integrate each impulse. The saved gap on this path is the linearly corrected predicted gap. The hard laws assume perfectly inelastic normal impact. A prescribed penetration with zero normal mobility is incompatible with a hard constraint. SAP with incidence-dependent damping currently supports one loaded region per pair. A step-interior event search checks both the free and corrected trajectories when enabled.

The [two-slider driver](../benchmarks/aes/solver_impact.py) checks the compliant runs against the elastic oscillator (0.45 N·s) and hard runs against the inelastic impact (0.225 N·s). The [mesh](../results/aes/cpu-solvers/impact-mesh-obb.json) and [CAD](../results/aes/cpu-solvers/impact-cad-occt.json) results include final velocities and displacement errors. The [eccentric collision](../results/aes/cpu-solvers/free-eccentric-ncp.json) also checks angular momentum. The [reducer SAP](../results/aes/cpu-solvers/reducer-sap.json) and [NCP](../results/aes/cpu-solvers/reducer-ncp.json) runs exercise coupled constraints on a shared intermediate shaft over 100 steps.

An independent fixed-step control used discrete Drake 1.58.0 SAP with analytic spheres, 1 kg per body, pair stiffness 5,000 N/m, zero friction and zero relaxation time. Across 0.16 s the STEP and Drake displacement histories agreed within 1.3 × 10⁻¹⁵ m at 500 and 250 µs and within 3.06 µm at 125 µs. At 250 µs their common maximum displacement error relative to the continuous oscillator was 0.768 mm. [Drake comparison driver](../benchmarks/engines/drake_sap_impact.py) · [All three step sizes](../results/aes/cpu-solvers/impact-drake-sap.json) · [Drake contact parameters](https://drake.mit.edu/doxygen_cxx/group__contact__defaults.html)

## Local STEP exit depths

`nc_contact --method cad-local` locates connected overlap components with the native
solid intersection. From each component seed and area-derived normal it intersects
short segments with both original trimmed solids. The two first outward exits give
the local depth. Their midpoint gives the common force arm. Each query traverses
the body-local face boxes even when a previous face supplies a hint. Indeterminate
exits propagate to the solver. [Implementation](../src/cad_solid_contact.cpp)

The CPU replay executable `nc_cad_probes` explores faster discovery. It searches
both bodies' parameter charts at every pose, rechecks previous sites and refines
near misses in surface parameters. Each reported witness has two verified exits.
With `--groups`, a polyline verified inside both solids can join two witnesses.
The finite search can miss a component and the path search can split one. The
reducer replay checks both effects by classifying witnesses in independently
computed STEP overlap volumes. [Verification and reproduction](../results/aes/cpu-local-cad/README.md)

### Distributed pressure on CPU CAD patches

`nc_contact --method cad-patch --solver penalty` uses the same scene and
implicit integrator as `mesh-obb`. Both bodies provide trimmed STEP charts.
The indexed search traverses the site and tile trees together. Positive
witnesses have two oriented exits. Disjoint spline control hulls prune
segment queries during grouping. Workers have private intersectors, and the
solver assembles results in face-pair order. Polynomial bicubic knot spans
use precomputed coefficients. Other surfaces use the OCCT adaptor.
[Search and quadrature](../src/cad_patch_search.cpp)

For each discovered component, quadrature on the participating faces computes
the directed area, its first moment and a thickness-weighted centroid. A
trimmed STEP chord through that centroid sets the scalar penetration. The
shared spring stiffness sets the magnitude of the normal force. Uniform
pressure over the region distributes that force and contributes a three-axis
couple at the centroid. HHT transports both the point and couple during a
predictor update. The area rule starts at 12 by 12 per face and retries
unresolved components at 24, 48 and 96. If these rules cannot return a patch,
the backend tries a whole-solid intersection at that pose. Empty witness
sets require separated body boxes or a native solid-distance check,
including enclosure testing. Possible overlap takes the whole-solid path.
The distance kernel tests separating directions against original spline
control hulls. For a bounded planar face it can also clip the opposing hull
to the plane's distance band and classify its footprint against the face
trims. A winding test is used only when the entire footprint clears the
enclosures of the original trim curves. Remaining candidates use OCCT
feature extrema. This optimization applies during separation checks. Once
the search finds a loaded region, the default uses the existing area and
moment integration and each worker's original ray intersector.
[Implementation and CPU comparisons](../results/aes/cpu-cad-pressure/pruning/README.md)

The CPU `cad-patch` backend caches the native UV search grid, local BVHs and
surface quadrature by part geometry. Set `NC_CAD_CACHE_DIR` to select a
persistent directory. Run `nc_cad_probes --prepare-parts part1.step part2.step`
to build the 12, 24, 48 and 96 point-per-axis rules ahead of simulation.
The online search keeps a list of pairs of BVH nodes and rebuilds it when
a bound on the relative displacement exceeds the list margin. Each integration
block is tested against the opposing faces before individual quadrature.
Search and integration still use the existing finite witness and pressure rules.
[Timing, cache sizes and fixed-pose checks](../results/aes/cpu-cad-offline/README.md)

`--method cad-bezier` uses that same CPU patch search, shared penalty law and pressure integration.
It also persists a table of rational Bézier spans for each part. The table contains homogeneous
control points, parameter ranges, position bounds and derivative ranges. Nonperiodic B-spline
surfaces and Bézier faces up to degree three in each direction can use the compiled ray kernel.
All workers share the immutable table in body coordinates. The optional offline command is
`nc_cad_probes --prepare-parts part1.step part2.step --bezier`.
[Bézier ray kernel](../src/cad_bezier_ray.cpp)

The kernel tests candidate spans against the finite ray, bounds the two projected equations
and recovers a single interior root with its oriented normal. A poorly conditioned projection,
an ambiguous span, a hit near a span boundary, an irregular trim or an unsupported surface
can require the native OCCT intersector. This path handles exits and segment queries during
contact discovery, grouping and centroid-depth evaluation. The simulation retains the existing
finite seed grid and its limitations for finding small disconnected regions. Step-interior
collision detection remains a separate option.
[CPU replay and dynamics measurements](../results/aes/cpu-bezier/README.md)

The separate `cad-compiled --device cuda-resident` runtime loads precompiled
charts and trims without an online OCCT fallback. Its optional
`--gpu-execution cached` schedule gives each candidate face pair a 128-thread
block. Four warp reductions select the first accepted proposal or fault in the
original order; an unsuccessful batch retains the greatest projection depth
with the original tie order. Independent connection tests occupy separate blocks.
Quadrature warps are spread across blocks while retaining the original column
indices and ordered FP64 sums. The blocking-face cache is revalidated at each
new pose; unsuccessful hints use the full compiled geometry query. Fixed 48/96
quadrature tables are prepared on the GPU before online timing. The variable
local rule remains online. The conditional-graph experiment is archived and is
not a runtime option because synchronization safety checks failed.
[Implementation](../src/cuda_resident_geometry.cuh) ·
[Validation, timings and complete responses](../results/aes/cuda-graph-execution/README.md)

#### Known limitations

At a degenerate chart pole, two rings around the centroid ray must agree
before the solver accepts the limiting depth.
A resolved coarse rule has no a priori error bound. A nonempty witness set
can still miss an additional disconnected component.
[Force assembly](../src/contact_dynamics.cpp)

`nc_cad_distance_benchmark --pressure-patches` (following the two STEP paths,
with a reach in metres) integrates the boundary of every trimmed surface of
the complete overlap under the same pressure convention. The replay tool
`nc_cad_probes --patch-wrench` evaluates the discovered faces independently
of this control. CPU force, torque and timing comparisons use common shaft
poses and independently advanced trajectories. [Reducer results](../results/aes/cpu-cad-pressure/README.md)

## Free rigid bodies

The optional v2 scene block after `drives` adds independent spatial bodies alongside existing hinges and sliders.

```text
free-bodies 1
free BODY Ixx Iyy Izz Ixy Ixz Iyz vx vy vz wx wy wz
```

Specify a positive mass and local center of mass in the existing body record. The six inertia entries are the symmetric tensor components about that center in body axes (kg·m²). Initial center velocity is in world axes and angular velocity is in body axes. The input pose still positions the original CAD or mesh origin. Contact forces act at a shared world point with its moment arm measured from each body's own center of mass. An HHT step advances center position and a homogeneous quaternion with Euler's gyroscopic term. The trace includes each free body's scalar-first quaternion and world angular velocity. Mesh event checks interpolate the moving center and rotate about it. Offset hinges use the same construction about their pivot. An event check interpolates the endpoints of each trial step.

The [free-body driver](../benchmarks/aes/cpu_free_body.py) uses the archived offset-sphere contact mesh. For a freely rotating symmetric top with a nondiagonal local inertia and displaced center of mass, maximum orientation error over 2.4 s was 174.37, 43.50 and 11.32 µrad at 4, 2 and 1 ms. The [1 ms result](../results/aes/cpu-mesh/free/top-mesh-obb-0.001-0.json) also checks the world angular velocity, center trajectory and local accelerometer. In the [free central impact](../results/aes/cpu-mesh/free/head-on-mesh-obb-0.00025-0.json), the peak normal force was 22.4960 N and impulse was 0.449974 N·s against 22.5 N and 0.45 N·s for smooth spheres.

The [eccentric frictional impact](../results/aes/cpu-mesh/free/eccentric-mesh-obb-0.00025-0.json) exercises off-center force application, rotation of both bodies and contact release. For the same scene, the three broad phases produced byte-identical 641-frame traces. The maximum total angular momentum drift was 1.50×10⁻⁸ kg·m²/s at 250 µs and [3.74×10⁻⁹ kg·m²/s at 125 µs](../results/aes/cpu-mesh/free/eccentric-mesh-obb-0.000125-0.json). Using the [50 µs surface-event check](../results/aes/cpu-mesh/free/eccentric-mesh-obb-0.00025-0.00005.json) on the 250 µs grid produced six subdivisions. The eccentric test checks system momentum and dissipation. Its force has no independent closed-form reference.


### Pair-owned GPU work queues (2026-11-28)

`cuda_batched_contact.cuh` owns one mutable workspace per contact pair. Its device-controlled phase graphs batch face queries and compact valid quadrature rays across pairs. Results retain the original column IDs and reduction order. The warp connection traversal now carries rejected subtree end indices across preorder windows, preserving the earliest hit/fault while skipping descendants. Ray queries use 32-thread blocks; the CAD equations, force law and quadrature points are unchanged. The optional schedule remains experimental: whole-graph synccheck reports a barrier error also reproduced by an independent graph-only program. Native and memory checks are separate evidence, not a substitute for that unresolved check.

Last verified against: `src/cuda_batched_contact.cuh`, `src/cuda_cad_roots.cuh`, `tests/compiled_cad.cpp`; native CPU/CUDA regressions, memcheck and initcheck passed on 2026-11-28. The pruned warp traversal also passed synccheck under the cached cooperative schedule. Full-run records and exact commands are in the [batching report](../results/aes/cuda-batched-execution/README.md).
