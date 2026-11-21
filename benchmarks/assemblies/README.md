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

## Sampled distances against a triangle BVH

The clearance command queries 17 fixed poses per configured motion. It times the native B-rep interval search with a 0.0002 mm width target and 30,000 task cap, then repeats the same poses using BVHs of OCCT triangulations at linear deflections of 0.20, 0.05 and 0.0125 mm. Each comparison uses the OCCT boundary-extrema distance as its independent numerical reference. The records preserve the interval when the native search reaches its budget.

~~~powershell
& $bench clearance build/assemblies/mmp-wheel.step benchmarks/assemblies/wheel.txt build/assemblies/wheel-clearance.csv
& $bench clearance build/assemblies/mmp-lidar.step benchmarks/assemblies/lidar.txt build/assemblies/lidar-clearance.csv
~~~

[Measured wheel clearance](../../results/hht/metrics/production-wheel.csv) · [Measured lidar clearance](../../results/hht/metrics/production-lidar.csv)
