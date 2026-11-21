# Continuous-clearance follow-up

The four CSVs contain nine query-only repetitions per case and method after warm-up. The variants share a clearance of 0.03125, a spatial tolerance of `2e-4`, a normalized time target of `5e-4` on `[0,1]`, and a 200,000-task cap. STEP cases use millimeters. Runs were measured in MSVC Release with OCCT 7.9.3 on a Ryzen 9 5900X. The summary gives medians, nearest-rank P95, worst bracket widths, and outcomes. With nine repetitions P95 is the observed maximum.

| File | Fixtures and reference |
| --- | --- |
| [rotation.csv](rotation.csv) | Shared untrimmed quadratic patches. Closed-form first entry on the original arc, evaluated at 80 digits. The adapted Chen OBB run uses chords with acceleration-based padding and outward rechecking of rejected segments. |
| [counterbore.csv](counterbore.csv) | Generated, reimported STEP counterbore and pin. The analytic reference takes the first qualifying rim using its radial and axial distance. |
| [assembly.csv](assembly.csv) | Original bolt, spacer, washer and bearing occurrences from Voron Trident Front Idler B. All variants use the same adaptive OCCT reference, with a `1e-6` mm margin, a `2e-5` time resolution and a 1,024-query cap. |
| [audit.csv](audit.csv) | Original-STEP trim-edge witnesses and their coordinate enclosures. The independent checker reconstructs the residual and membership against the exact plate and disk geometry. |

The real assembly is extracted at revision `7fcb3257a3063225934d9608f82037aa6587d844`. The script retains the original STEP geometry, the 31 occurrence links of product definition `#2547975`, and their transforms within the idler frame. The unreferenced model outside that subtree is omitted from the derived input. The scoped pairs use source components in their original relative placements before the prescribed translations. The extractor checks the archive, complete STEP and subtree SHA-256 values. [Original Voron STEP](https://github.com/VoronDesign/Voron-Trident/blob/7fcb3257a3063225934d9608f82037aa6587d844/CAD/trident_assembly.step.zip) · [Occurrence IDs and motions](../assemblies/voron-front-idler.txt)

For rotation, the generated header adds a projected Euclidean margin and a search counter to Chen's pinned time-dependent OBB solver. Each chord bounds the curved point displacement by `M*h*h/8` plus the outward enclosure of the chord input rounding. A point on the original moving patch and its projection inside the fixed rectangle certify an upper time. The time includes all wrapper work, including outward checks of the solver's proposed exclusions. [Official baseline and linear protocol](../chen/README.md)

An unresolved result has a certified prefix and no witnessed upper time. A numerical OCCT reference can likewise reach its work cap. The summary keeps that outcome and the numerical reference prefix. A result with a witness but excessive width remains a hit and is marked separately.

## Reproduce

Use the Release builds, the [OCCT runtime setup](../assemblies/README.md), and Python with `mpmath`. Configure the Chen checkout and Eigen headers as described [in the baseline README](../chen/README.md).

~~~powershell
New-Item -ItemType Directory -Force build/clearance-followup | Out-Null
cmake --build build/chen-common --config Release --target nc_chen_rotation
build/chen-common/Release/nc_chen_rotation.exe 9 > build/clearance-followup/rotation.csv
cmake --build build/occt --config Release --target nc_clearance_followup nc_assembly_bench
build/occt/Release/nc_clearance_followup.exe counterbore build/clearance-followup build/clearance-followup/counterbore.csv 9
python benchmarks/assemblies/voron_fixture.py build/assemblies
build/occt/Release/nc_assembly_bench.exe catalog build/assemblies/voron-front-idler-b.step build/assemblies/voron-front-idler-catalog.csv
build/occt/Release/nc_assembly_bench.exe measure-all build/assemblies/voron-front-idler-b.step benchmarks/assemblies/voron-front-idler.txt build/clearance-followup/assembly.csv 9
build/occt/Release/nc_clearance_followup.exe audit build/clearance-followup build/clearance-followup/audit.csv 9
python benchmarks/verify_clearance_followup.py build/clearance-followup
~~~
