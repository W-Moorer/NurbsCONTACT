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

`--audit-workers N` distributes the face projection samples over independent Gmsh sessions. Each session reimports the STEP file and checks its face tags, surface types, areas and centroids against the meshed model. All workers use the same projection and normal evaluations. The report includes the worker count and audit time.

In the [spur pair preparation trial](../results/aes/cpu-mesh/spur-preparation.json), the 50,060-triangle pinion met its requested checks. The coarse wheel had a minSICN of 0.00429. Refinement of its 7.44 µm-wide strips reached the element quality gate of 0.2. Both the one-worker and six-worker full-wheel runs exhausted the 1,800 s budget during the remaining CAD audit. This gear pair still lacks a qualified wheel for dynamics comparisons.

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

Add `--contours` to a discrete query to group connected triangle intersections within each body pair. Groups meet at exact world-space witnesses on triangle boundaries. A group can contain a point, a curve, or coplanar polygons. A contact region with holes can have several boundary curves. Force integration and penetration depth require further geometry.

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
