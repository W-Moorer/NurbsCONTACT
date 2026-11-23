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

These comparisons measure event timing on prescribed poses. They do not measure the force response of a mesh-driven simulation.
