# GPU accumulation of distributed contact loads

This experiment changes the accumulation of accepted quadrature columns on the RTX 3070. All modes read the same masks, FP64 material points, normals, quadrature weights and penetration depths. The serial path visits the occupied columns in increasing index order. The ordered parallel path compacts active IDs in that order using a warp scan, forms up to 32 column contributions together, and adds ten independent components on ten lanes. Each component visits the same sequence of addends as the original. The chord query, pressure-moment correction, penalty law and integrator then use the resulting region.

The `warp` ablation partitions the columns among the lanes and combines partial sums with a shuffle tree. It changes the parenthesization of the double-precision additions. The `audit` mode checks the ordered implementation, including equality of every accumulated component before the chord query. The `audit-warp` mode compares the tree with the serial reference at identical poses.

The five-second reducer uses 50,000 steps of 0.1 ms. The original executable and both arithmetic modes run independently from the initial state. The fourth run computes both reductions at each freshly integrated region on the same trial pose and drives dynamics with the parallel result. It checks region construction and loading decisions and evaluates the force on body B and the torque on each body about its own origin in world axes. A discrepancy in the construction or loading decision terminates the audit. Per-axis absolute maxima are retained per body pair and step, including evaluations during implicit retries. Wrench comparison includes the penalty sharing factor across loaded regions.

The face-switch diagnostic follows confirmed pairs of original CAD faces between refreshed geometry queries. It covers changes of supporting faces within either gear stage. Body-pair on/off flags are checked separately on the accepted, uniform time grid. The code uses scene body IDs and CAD faces throughout. The timings exclude asset preparation. The reduction timer covers accumulation up to the final area, moment, volume and centroid sums. It is nested within the integration timer. No audit time is included in the production comparison.

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
python benchmarks/aes/compare_region_reduction.py $scene $cpuResponse original serial parallel audit report --serial-repeat serial-repeat --parallel-repeat parallel-repeat --original-binary $original --binary $current
```
