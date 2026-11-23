# CPU contact mesh preparation

`apps/prepare_contact_mesh.py` imports a closed STEP solid through Gmsh's OpenCASCADE kernel, meshes its boundary with first-order triangles and writes an ASCII Gmsh 2.2 file in metres. The elementary entity tags identify the originating Gmsh faces and edges. The input model can declare its STEP length unit. Both the command line and the output use SI lengths.

Install the Python dependencies with `python -m pip install gmsh numpy libigl meshio`. The last package is used by the round-trip test.

```powershell
python apps/prepare_contact_mesh.py results/gears/models/spur-1a.step build/cpu-contact/pinion.msh --size-m 0.0015
python tests/test_contact_mesh.py
```

The sibling `.quality.json` contains hashes of the STEP and mesh files, the Gmsh version, preparation time, element counts, winding volume, minimum element angle, minSICN statistics, sampled CAD deviations and normal angles. `--max-error-m`, `--max-normal-deg` and `--min-sicn` enforce application-specific quality limits. A failed quality check preserves a previously generated mesh. A consumer reusing a cached mesh should match its SHA-256 against its quality report.

Each triangle contributes a centroid and three edge midpoints to the comparison with its supporting CAD surface. Each segment on a CAD edge also contributes its midpoint to a curve-distance check. These measurements are samples. The normal error is the angle between tangent planes, measured per CAD face. Topological checks cover duplicate or degenerate triangles, edge incidence, consistent winding, vertex manifoldness and self-intersection. Mesh refinement and an independent dynamics reference are needed to relate these measurements to a required response accuracy.

## Last verified against

`tests/test_contact_mesh.py` checks sphere size and volume in the written file, convergence of the sampled surface error, topology failures and preservation of previous outputs after a quality rejection.
