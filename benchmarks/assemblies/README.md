# Production STEP assembly comparison

The source models are published production assemblies of the OpenAMRobot MMP platform. The benchmark reads the assembly occurrences from STEP via XCAF, retains the source instance transforms and queries two selected components per scenario. The input files remain in the build directory. The fetch script pins the upstream commit and checks each file's SHA-256. [Source production files](https://github.com/openAMRobot/openamr-platform-hw/tree/8ed2b06a46115abe2dffadbf0d902ae731d278d1/mechanical/cad/production_files)

| Scenario | Source assembly | Stationary component (ID, faces) | Moving component (ID, faces) | Motion relative to the original assembly placement |
|---|---|---|---|---|
| Wheel washer stack | MMP.03.00.00.000 Wheel assembly | DIN 125 flat washer (26, 6) | DIN 7980 lock washer (25, 6) | \(+5\) mm to \(0\) on \(z\), threshold 0.15 mm |
| Wheel eccentric bolt 1 | MMP.03.00.00.000 Wheel assembly | DIN 125 flat washer (26, 6) | DIN 912 M10×45 screw (24, 27) | \(+48\) mm to \(0\) on \(z\) with a fixed \(+0.2\) mm \(x\) offset, threshold 0.12 mm |
| Wheel eccentric bolt 2 | MMP.03.00.00.000 Wheel assembly | DIN 125 flat washer (31, 6) | DIN 912 M10×45 screw (29, 27) | \(-48\) mm to \(0\) on \(z\) with a fixed \(+0.2\) mm \(x\) offset, threshold 0.12 mm |
| Lidar pre-seat | MMP.07.00.00.000 Lidar support assembly | Lidar support (1, 116) | M4 rivet nut (2, 27) | \(-12\) mm to \(-7.5\) mm on \(y\), threshold 0.15 mm |
| Lidar full insertion diagnostic | MMP.07.00.00.000 Lidar support assembly | Lidar support (1, 116) | M4 rivet nut (2, 27) | \(-12\) mm to \(0\) on \(y\), threshold 0.15 mm |

The component IDs are one-based depth-first XCAF enumeration indices from OCCT 7.9.3. The published wheel file contains 97 component occurrences and the lidar support file contains 5. The motions are prescribed translations over normalized time \([0,1]\).

Both configurations use the same native geometry, threshold, spatial tolerance \(2\times10^{-4}\), time tolerance \(5\times10^{-4}\), and 200,000 task cap. Median latency comes from nine query-only repetitions after warm-up, with alternating order. A separate OCCT boundary distance oracle provides a numerical cross-check using adaptive time subdivision, the translation speed, and a \(10^{-6}\) mm numerical margin. The reported safe prefixes and witness intervals come from the query kernel's directed bounds.

## Reproduce

After setting the OCCT runtime path as described in the [build instructions](../../README.md), run from the project root.

```powershell
python benchmarks/assemblies/fetch.py build/assemblies
$bench = 'build/occt/Release/nc_assembly_bench.exe'
& $bench catalog build/assemblies/mmp-wheel.step build/assemblies/wheel-catalog.csv
& $bench catalog build/assemblies/mmp-lidar.step build/assemblies/lidar-catalog.csv
& $bench measure build/assemblies/mmp-wheel.step benchmarks/assemblies/wheel.txt build/assemblies/wheel-results.csv 9
& $bench measure build/assemblies/mmp-lidar.step benchmarks/assemblies/lidar.txt build/assemblies/lidar-results.csv 9
python benchmarks/summarize.py build/assemblies/wheel-results.csv
python benchmarks/summarize.py build/assemblies/lidar-results.csv
```

The full lidar insertion diagnostic exercises the finite work budget. It keeps its brackets in the CSV and returns exit code 1 when they exceed the requested resolution.

```powershell
& $bench measure build/assemblies/mmp-lidar.step benchmarks/assemblies/lidar-diagnostic.txt build/assemblies/lidar-diagnostic.csv 1
```

The checked-in measurements are [wheel-results.csv](wheel-results.csv), [lidar-results.csv](lidar-results.csv), and [lidar-diagnostic.csv](lidar-diagnostic.csv).

## Voron Trident

The follow-up uses the original Front Idler B occurrence subtree and tests two screw sizes with a spacer, washer and bearing. The three-way `measure-all` mode records witnessed brackets and budget-limited prefixes with the same numerical reference. [Source revision, motions and measurements](../clearance-followup/README.md)

## Sampled distances against a triangle BVH

The clearance command queries 17 fixed poses per configured motion. It times the native B-rep interval search with a 0.0002 mm width target and 30,000 task cap, then repeats the same poses using BVHs of OCCT triangulations at linear deflections of 0.20, 0.05 and 0.0125 mm. Each comparison uses the OCCT boundary-extrema distance as its independent numerical reference. The records preserve the interval when the native search reaches its budget.

~~~powershell
& $bench clearance build/assemblies/mmp-wheel.step benchmarks/assemblies/wheel.txt build/assemblies/wheel-clearance.csv
& $bench clearance build/assemblies/mmp-lidar.step benchmarks/assemblies/lidar.txt build/assemblies/lidar-clearance.csv
~~~

[Measured wheel clearance](../../results/hht/metrics/production-wheel.csv) · [Measured lidar clearance](../../results/hht/metrics/production-lidar.csv)

## Continuous clearance baselines

The `compare` and `compare-pair` commands use the same STEP boundaries, threshold and prescribed translation on `[0,1]` across methods. Cases cover the eight published assembly paths and nine generated STEP paths with analytic answers. The time-width target is `5e-4`. Each method warms up once per case. Five timed repetitions rotate method order. The CSV separates query time from one-time model preparation and records the achieved bracket, work and triangle count. STEP import and validation are outside the timed work.

| CSV method | Query |
| --- | --- |
| `native-sweep` | Direct interval search over the original trimmed CAD boundaries with the production settings (`2e-4` mm spatial tolerance, 200,000 tasks). |
| `cad-ca-bounds` | Static native distance intervals combined with a conservative advance using the translation speed. The search observes both ends and refines remaining intervals in time order. Up to 256 distance requests and 200,000 total static tasks are allowed, with 8,192 tasks per request. |
| `occt-ca` | The same time search using the Open CASCADE boundary extrema distance with a `1e-6` mm numerical margin. The stationary boundary is loaded once. |
| `mesh-bvh-ccd` | Continuous triangle-BVH search on OCCT triangulations with 50, 12.5 and 3.125 µm settings. The edge fixtures add 390.625, 48.828 and 6.104 nm settings. Under a translation, a moving triangle sweeps a triangular prism. The BVH prunes by swept boxes, the narrow phase tests Euclidean triangle-to-prism proximity, and bisection bounds the first event on the meshes. |
| `mesh-ccd-cad-bounds` | The 3.125 µm mesh proposes an event time. The static native distance search verifies that time and certifies the safe prefix under the same limits as `cad-ca-bounds`. The recorded time includes both queries. |

The two methods using static interval bounds accept a time interval only when its separation is proved by the distance lower bound and translation speed. A static upper bound at an evaluated pose establishes that an event has occurred by that time. A query that cannot provide either a complete safe interval or a witnessed bracket within the width target remains visible in the results. The `occt-ca` row is a numerical variant of that time search. The pure mesh row measures Euclidean clearance on the triangulations. A mesher setting supplies no verified Hausdorff bound against the STEP boundary, so the pure mesh result has numerical rather than CAD-level assurance. The hybrid obtains its proof from the original geometry. The benchmark is scoped to translation.

The auditor checks repeated outcomes, interval widths, consistency between CAD bounds and analytic truth for the generated cases. It checks compatibility with a separate adaptive OCCT cross-check for assembly paths. The OCCT reference is numerical and shares the OCCT extrema kernel with `occt-ca`. Its unresolved results are reported as unknown, and an overlapping reference bracket establishes compatibility at the reference resolution.

### Results on the production paths

MSVC Release, OCCT 7.9.3, Ryzen 9 5900X. The raw records and method-wise medians are in [continuous-comparison.csv](continuous-comparison.csv) and [continuous-comparison-summary.csv](continuous-comparison-summary.csv).

| Method | Production paths resolved within target and compatible with the reference | Count with median query time at most 100 ms |
| --- | ---: | ---: |
| Native sweep | 6/8 | 2 |
| CAD interval advancement | 3/8 | 1 |
| Mesh CCD with CAD interval advancement | 3/8 | 3 |
| OCCT numerical advancement | 7/8 | 0 |
| Numerical mesh CCD at 3.125 µm | 7/8 | 6 |

The mesh-assisted CAD method cuts the two eccentric wheel bolt query medians from 68.50 to 33.50 ms and from 151.13 to 27.52 ms. Including preparation, the native query is faster on the first bolt and the hybrid is faster on the second. On the lidar pre-seat path, the native sweep certifies the whole path in 904.22 ms while both static CAD interval methods remain unresolved. The native result for full lidar insertion has width 0.03503 (target 0.0005). The 3.125 µm mesh result for that path is compatible with the numerical reference, and both coarser mesh results are early. The Voron spacer has an unresolved reference and is excluded from the success numerators.

In the analytic edge-entry case with minimum separation 100 nm below the threshold, the native sweep completes in 1.68 ms. The 3.125 µm mesh brackets an event too early. Refining the mesh to 48.828 nm gives the quickest compatible mesh bracket of the tested meshes at 4.29 ms. In the exact grazing case the native sweep excludes earlier times and leaves the upper end unresolved.

### Reproduce the continuous comparison

The commands below assume an OCCT Release build and the runtime environment from the [root instructions](../../README.md).

```powershell
$output = 'build/clearance-baselines'
New-Item -ItemType Directory -Force $output | Out-Null
python benchmarks/assemblies/fetch.py build/assemblies
python benchmarks/assemblies/voron_fixture.py build/assemblies
cmake --build build/occt --config Release --target nc_assembly_bench nc_clearance_followup
$bench = 'build/occt/Release/nc_assembly_bench.exe'
$fixtures = 'build/occt/Release/nc_clearance_followup.exe'
& $fixtures counterbore $output "$output/counterbore-fixture.csv" 1
& $fixtures audit $output "$output/edge-fixture.csv" 1
& $bench compare-pair "$output/counterbore.step" "$output/pin.step" benchmarks/assemblies/counterbore-proximity.txt "$output/counterbore.csv" 5
& $bench compare-pair "$output/edge-plate.step" "$output/edge-disk.step" benchmarks/assemblies/edge-proximity.txt "$output/edge.csv" 5 --refine-mesh
& $bench compare build/assemblies/mmp-wheel.step benchmarks/assemblies/wheel.txt "$output/wheel.csv" 5
& $bench compare build/assemblies/mmp-lidar.step benchmarks/assemblies/lidar.txt "$output/lidar.csv" 5
& $bench compare build/assemblies/mmp-lidar.step benchmarks/assemblies/lidar-diagnostic.txt "$output/lidar-diagnostic.csv" 5
& $bench compare build/assemblies/voron-front-idler-b.step benchmarks/assemblies/voron-front-idler.txt "$output/voron.csv" 5
python benchmarks/assemblies/audit_comparison.py "$output/summary.csv" "$output/counterbore.csv" "$output/edge.csv" "$output/wheel.csv" "$output/lidar.csv" "$output/lidar-diagnostic.csv" "$output/voron.csv" --repeats 5
```
