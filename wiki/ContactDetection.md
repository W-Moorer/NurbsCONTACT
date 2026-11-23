# CPU contact mesh preparation

`apps/prepare_contact_mesh.py` imports a closed STEP solid through Gmsh's OpenCASCADE kernel, meshes its boundary with first-order triangles and writes an ASCII Gmsh 2.2 file in metres. The elementary entity tags identify the originating Gmsh faces and edges. The input model can declare its STEP length unit. Both the command line and the output use SI lengths.

Install the Python dependencies with `python -m pip install gmsh numpy libigl meshio`. The last package is used by the round-trip test.

```powershell
python apps/prepare_contact_mesh.py results/gears/models/spur-1a.step build/cpu-contact/pinion.msh --size-m 0.0015
python tests/test_contact_mesh.py
```

The sibling `.quality.json` contains hashes of the STEP and mesh files, the Gmsh version, preparation time, element counts, winding volume, minimum element angle, minSICN statistics, sampled CAD deviations and normal angles. `--max-error-m`, `--max-normal-deg` and `--min-sicn` enforce application-specific quality limits. A failed quality check preserves a previously generated mesh. A consumer reusing a cached mesh should match its SHA-256 against its quality report.

Each triangle contributes a centroid and three edge midpoints to the comparison with its supporting CAD surface. Each segment on a CAD edge also contributes its midpoint to a curve-distance check. These measurements are samples. The normal error is the angle between tangent planes, measured per CAD face. Topological checks cover duplicate or degenerate triangles, edge incidence, consistent winding, vertex manifoldness and self-intersection. Mesh refinement and an independent dynamics reference are needed to relate these measurements to a required response accuracy.

When a requested minSICN fails on a long narrow face, the generator measures its area and perimeter, increases edge subdivision along its length and remeshes. If neighboring triangles still miss the threshold, it applies a graded Gmsh size field around the affected curves. The final quality checks can reject this result. Each output records the number of edges refined.

## Default spur gear mesh

Generate the routine CPU baseline and replay its archived poses from the repository root.

```powershell
python benchmarks/aes/prepare_spur_cpu.py build/cpu-contact/spur-standard
python benchmarks/aes/cpu_mesh_replay.py build/cpu-swept/Release/nc_mesh.exe build/cpu-contact/spur-standard/spur-1a.msh build/cpu-contact/spur-standard/spur-1b.msh results/aes/cpu-mesh/spur-poses.txt build/cpu-contact/spur-standard/replay.json --compare-overlap
```

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

At each HHT force evaluation the selected hierarchy finds intersecting triangles and enclosed components. For positive candidates the CPU reference implementation constructs the Boolean intersection of the two closed meshes. Each connected overlap volume contributes one contact at its volume centroid. The difference of the area-weighted outward normals on the two source boundaries gives the loading direction. Depth is the overlap width along that direction. Both bodies receive equal and opposite forces at the same point. A fully enclosed or symmetric overlap with no resolved direction reports a failure. Constructing the overlap of the full meshes is part of the measured cost.

The [impact benchmark](../benchmarks/aes/cpu_mesh_impact.py) integrates both slider masses from their initial state. The independent linear-spring solution gives an entry time of 40 ms, an impulse of 0.45 N·s and a peak of 22.5 N. On a fixed mesh the three broad phases produced matching response histories in these runs.

| Target size | HHT step | Entry error | Force RMSE | Impulse | CPU integration (OBB) |
| --- | --- | --- | --- | --- | --- |
| [3 mm](../results/aes/cpu-mesh/impact/3mm-mesh-obb-250us.json) | 250 µs | 128.9 µs | 0.07862 N | 0.449976 N·s | 15.72 s |
| [1.5 mm](../results/aes/cpu-mesh/impact/1p5mm-mesh-obb-250us.json) | 250 µs | 40.2 µs | 0.01917 N | 0.450005 N·s | 23.82 s |
| [1.5 mm](../results/aes/cpu-mesh/impact/1p5mm-mesh-obb-125us.json) | 125 µs | 40.2 µs | 0.01871 N | 0.450001 N·s | 46.04 s |

Each run spans 0.16 s. The timings are single measurements on the local CPU. The [entry/release guarded run](../results/aes/cpu-mesh/impact/3mm-mesh-obb-250us-guard.json) used 648 acceptance checks and subdivided four trials. Its entry error remained 128.9 µs at the coarse mesh resolution.

## Free rigid bodies

The optional v2 scene block after `drives` adds independent spatial bodies alongside existing hinges and sliders.

```text
free-bodies 1
free BODY Ixx Iyy Izz Ixy Ixz Iyz vx vy vz wx wy wz
```

Specify a positive mass and local center of mass in the existing body record. The six inertia entries are the symmetric tensor components about that center in body axes (kg·m²). Initial center velocity is in world axes and angular velocity is in body axes. The input pose still positions the original CAD or mesh origin. Contact forces act at a shared world point with its moment arm measured from each body's own center of mass. An HHT step advances center position and a homogeneous quaternion with Euler's gyroscopic term. The trace includes each free body's scalar-first quaternion and world angular velocity. Mesh event checks interpolate the moving center and rotate about it. Offset hinges use the same construction about their pivot. An event check interpolates the endpoints of each trial step.

The [free-body driver](../benchmarks/aes/cpu_free_body.py) uses the archived offset-sphere contact mesh. For a freely rotating symmetric top with a nondiagonal local inertia and displaced center of mass, maximum orientation error over 2.4 s was 174.37, 43.50 and 11.32 µrad at 4, 2 and 1 ms. The [1 ms result](../results/aes/cpu-mesh/free/top-mesh-obb-0.001-0.json) also checks the world angular velocity, center trajectory and local accelerometer. In the [free central impact](../results/aes/cpu-mesh/free/head-on-mesh-obb-0.00025-0.json), the peak normal force was 22.4960 N and impulse was 0.449974 N·s against 22.5 N and 0.45 N·s for smooth spheres.

The [eccentric frictional impact](../results/aes/cpu-mesh/free/eccentric-mesh-obb-0.00025-0.json) exercises off-center force application, rotation of both bodies and contact release. For the same scene, the three broad phases produced byte-identical 641-frame traces. The maximum total angular momentum drift was 1.50×10⁻⁸ kg·m²/s at 250 µs and [3.74×10⁻⁹ kg·m²/s at 125 µs](../results/aes/cpu-mesh/free/eccentric-mesh-obb-0.000125-0.json). Using the [50 µs surface-event check](../results/aes/cpu-mesh/free/eccentric-mesh-obb-0.00025-0.00005.json) on the 250 µs grid produced six subdivisions. The eccentric test checks system momentum and dissipation. Its force has no independent closed-form reference.
