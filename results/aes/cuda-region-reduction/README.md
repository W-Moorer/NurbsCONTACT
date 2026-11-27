# GPU accumulation of distributed contact loads

This experiment changes the accumulation of accepted quadrature columns on the RTX 3070. All modes read the same masks, FP64 material points, normals, quadrature weights and penetration depths. The serial path visits the occupied columns in increasing index order. The ordered parallel path compacts active IDs in that order using a warp scan, forms up to 32 column contributions together, and adds ten independent components on ten lanes. Each component visits the same sequence of addends as the original. The chord query, pressure-moment correction, penalty law and integrator then use the resulting region.

The `warp` ablation partitions the columns among the lanes and combines partial sums with a shuffle tree. It changes the parenthesization of the double-precision additions. The `audit` mode checks the ordered implementation, including equality of every accumulated component before the chord query. The `audit-warp` mode compares the tree with the serial reference at identical poses.

The five-second reducer uses 50,000 steps of 0.1 ms. The original executable and both arithmetic modes run independently from the initial state. The fourth run computes both reductions at each freshly integrated region on the same trial pose and drives dynamics with the parallel result. It checks region construction and loading decisions and evaluates the force on body B and the torque on each body about its own origin in world axes. A discrepancy in the construction or loading decision terminates the audit. Per-axis absolute maxima are retained per body pair and step, including evaluations during implicit retries. Wrench comparison includes the penalty sharing factor across loaded regions.

The face-switch diagnostic follows confirmed pairs of original CAD faces between refreshed geometry queries. It covers changes of supporting faces within either gear stage. Body-pair on/off flags are checked separately on the accepted, uniform time grid. The code uses scene body IDs and CAD faces throughout. The timings exclude asset preparation. The reduction timer covers accumulation up to the final area, moment, volume and centroid sums. It is nested within the integration timer. No audit time is included in the production comparison.

The ordered implementation needs 6,656 bytes for indices and intermediate contributions. One warp uses these buffers at a time. The first version (commit `a24673e`) reserved the storage per block and took 445.63 s. The retained version allocates it once per trajectory on the device and took 436.96 s on the first run. Both results match the original response byte for byte. The first layout's same-binary serial control took 450.40 s. [Shared-memory timings and validation](shared/comparison.json)

## Timings on the RTX 3070

Both modes used the same executable and model. Each row is a separate five-second simulation. The second pair reversed the execution order. Timings are online seconds and exclude scene compilation, setup and diagnostic instrumentation.

| Pair | Run order | Reduction | Total online (s) | Region accumulation (s) |
| --- | --- | --- | ---: | ---: |
| 1 | first | ordered parallel | 436.961 | 1.172 |
| 1 | second | serial | 441.903 | 6.056 |
| 2 | first | serial | 441.896 | 6.054 |
| 2 | second | ordered parallel | 436.748 | 1.178 |

The paired savings were 4.94 s and 5.15 s online. Mean accumulation time fell by 80.6%, and mean online time fell by 1.14%. In the first ordered run, face-pair search and contact region grouping accounted for 171.15 s and 100.07 s. Mean online time was 87.37 times the simulated duration. Both ordered runs missed all 50,000 deadlines of 0.1 ms. The archived executable took 444.34 s when rerun on the same machine.

## Accuracy during contact switching

At identical poses, both stages together supplied 239,624 region comparisons. Of these, 114,709 fell on a step where a supporting CAD face pair entered or exited, or on one of its two adjacent steps. The largest absolute error was zero for each of the three force components and for each of the six torque components about the two body origins. Across independent closed-loop runs the two stage normal loads, intermediate shaft speed and output contact torque agreed at all 50,001 saved times. Both modes used 44 HHT subdivisions and 185,760 force evaluations. The ordered run's response CSV has SHA-256 `0b3c1efeb81f86b7c60b380b74b63f3357a1d097b75a5dbe8a570d867ed420bb`, identical to the original executable's output. The moving-box GPU regression also recovered the analytical displacement, penalty load and release step (158) with the same backend.

[Response curves and switch detail](response.png) · [Paired measurements](comparison.json) · [Per-step, per-stage region errors](region-audit.npz)

## Effect of changing the addition order

With the tree, the force and moment comparisons at identical poses agreed to within 2.5e-9 N and 2e-10 N m per component. The independent closed loop nonetheless required 46 subdivisions versus 44 in serial. The full-step velocity update identifies steps 35,680 and 37,281 as the two additional subdivisions. The count of identified main steps accounts for every recorded subdivision in both runs. At 3.5681 s the output contact torques were 198.5434 N m in serial and 193.0508 N m with the tree. This step was within one main step of a change of confirmed faces. Peak output torque and body-pair on/off flags agree between the runs. [Tree timings and comparison](tree/comparison.json) · [Response and local torque discrepancy](tree/response.png) · [Same-pose wrench audit](tree/region-audit.npz)

## Reproduction

Build Release with `NC_WITH_CUDA_COLUMNS=ON`. Use the frozen four-part reducer scene and its audited `.ncpart` files. Run the archived executable from commit `6b74f3f` or its previously verified equivalent with the first command. Run the other commands with the modified executable, one process at a time on the same GPU.

```powershell
$common = @('--method','cad-compiled','--device','cuda-resident','--solver','penalty','--penalty-stiffness','shared','--geometry-update','predictor','--spectral-radius','0')
& $original $scene 0 0.0001 5 1 original/response @common
& $current $scene 0 0.0001 5 1 parallel/response @common --region-reduction ordered
& $current $scene 0 0.0001 5 1 serial/response @common --region-reduction serial
& $current $scene 0 0.0001 5 1 audit/response @common --region-reduction audit
& $current $scene 0 0.0001 5 1 serial-repeat/response @common --region-reduction serial
& $current $scene 0 0.0001 5 1 parallel-repeat/response @common --region-reduction ordered
python benchmarks/aes/compare_region_reduction.py $scene $cpuResponse original serial parallel audit report --serial-repeat serial-repeat --parallel-repeat parallel-repeat --original-binary $original --binary $current --require-identical
```
