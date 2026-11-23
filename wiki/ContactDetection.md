# CPU contact mesh preparation

`apps/prepare_contact_mesh.py` imports a closed STEP solid through Gmsh's OpenCASCADE kernel, meshes its boundary with first-order triangles and writes an ASCII Gmsh 2.2 file in metres. The elementary entity tags identify the originating Gmsh faces and edges. The input model can declare its STEP length unit. Both the command line and the output use SI lengths.

Install the Python dependencies with `python -m pip install gmsh numpy libigl meshio`. The last package is used by the round-trip test.

```powershell
python apps/prepare_contact_mesh.py results/gears/models/spur-1a.step build/cpu-contact/pinion.msh --size-m 0.0015
python tests/test_contact_mesh.py
```

The sibling `.quality.json` contains hashes of the STEP and mesh files, the Gmsh version, preparation time, element counts, winding volume, minimum element angle, minSICN statistics, sampled CAD deviations and normal angles. `--max-error-m`, `--max-normal-deg` and `--min-sicn` enforce application-specific quality limits. A failed quality check preserves a previously generated mesh. A consumer reusing a cached mesh should match its SHA-256 against its quality report.

Each triangle contributes a centroid and three edge midpoints to the comparison with its supporting CAD surface. Each segment on a CAD edge also contributes its midpoint to a curve-distance check. These measurements are samples. The normal error is the angle between tangent planes, measured per CAD face. Topological checks cover duplicate or degenerate triangles, edge incidence, consistent winding, vertex manifoldness and self-intersection. Mesh refinement and an independent dynamics reference are needed to relate these measurements to a required response accuracy.

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

## Last verified against

`tests/test_contact_mesh.py` checks sphere size and volume in the written file, convergence of the sampled surface error, topology failures and preservation of previous outputs after a quality rejection.
