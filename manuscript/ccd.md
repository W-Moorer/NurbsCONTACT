# Certified Continuous Clearance of Trimmed CAD Boundaries under Fixed-Axis Rigid Motion

## Abstract

Clearance queries in CAD assemblies must resolve narrow gaps throughout a motion. We use directional projections of enclosing control hulls to exclude time intervals before refining trimmed surface domains. With fixed relative orientation, we also use radial and axial bounds to retain open space inside circular rims. The search propagates surviving intervals through the hierarchy and brackets a first threshold entry with a certified safe prefix and a feasible surface point pair. On six controlled STEP scenarios, the optimized configuration retained the \(5\times10^{-4}\) normalized time target and reduced the eccentric bore-rim query from 693.1 ms to 1.24 ms in the matched run. Across the timed component-pair queries from two published subassemblies of a mobile robot, the median speedups ranged from 1.09 to 5.97 times. A longer lidar insertion exceeded the interval-width target at the fixed task cap.

## 1. Introduction

Assembly verification requires both a geometric and a temporal decision. A projected tool point may lie on a supporting surface while falling inside a hole in the actual face. Conversely, two parts can remain separated at the endpoints of a step and approach in its interior. The first issue changes the feasible parameter domain. The second changes the trajectories that must be bounded. Resolving both at once by repeatedly subdividing every spatial and temporal variable can spend substantial work on intervals in which the parts were already separated.

We study the following question. Can a time-dependent support bound eliminate enough work to make direct clearance queries practical on trimmed CAD geometry under rotation? The key observation is that rigid motion preserves convex combinations, so every point on a positive-weight rational patch stays within the moving convex hull of its control points. In a frame fixed to one body, the projection of each controlling point on the other body has a simple scalar trajectory. A single projection can therefore exclude an entire range of times for a surface-cell pair. The same exclusion remains valid for descendants of the pair's spatial domains.

We make three contributions. We define a search over trimmed CAD cells that shares static distance bounds and feasible witnesses with a continuous query. We derive directional exclusion for fixed-axis relative motion and radial and axial exclusion for selected faces under fixed relative orientation. Both bounds remove time intervals that can stay excluded through spatial refinement. We evaluate the original and optimized search on controlled STEP geometry and on component pairs from published production assemblies.

## 2. Related work

Direct NURBS proximity queries and local contact generation address different outputs. Krishnamurthy, McMains, and Haller used bounding hierarchies and local triangle approximations for clearance queries on trimmed NURBS. Crozet, Léon, and Merlhiot computed multiple local minima on smooth CAD boundaries and accelerated them with geometric features and normal cones. Our output is a globally bounded clearance at a pose or a safe prefix for a prescribed trajectory. We use trim validity in the feasible domain and maintain lower and upper bounds during search. [Krishnamurthy et al., 2011](https://doi.org/10.1109/TVCG.2010.114) · [Crozet et al., 2018](https://doi.org/10.1080/16864360.2017.1419646)

Hierarchical interval methods for moving parametric surfaces date to Von Herzen, Barr, and Zatz and to Snyder et al. Chen et al. developed time-dependent inclusions using control points with linear trajectories. Rotation of a rigid body yields curved control-point trajectories. Ferguson et al. addressed curved trajectories in rigid dynamics by reducing the query to linear CCD with a separation margin. We retain a nonlinear projection enclosure inside the trimmed spatial search for the class of relative rigid motions defined below. [Von Herzen et al., 1990](https://authors.library.caltech.edu/records/68ah1-f0859) · [Snyder et al., 1993](https://www.gg.caltech.edu/papers/collideabstract.html) · [Chen et al., 2024](https://doi.org/10.1145/3687960) · [Ferguson et al., 2021](https://doi.org/10.1145/3450626)

## 3. Geometry and query model

A face is the image of a closed trimmed parameter domain \(D\) under a map \(S(u,v)\). The domain is given by closed parameter-space wires. Its interior is defined by odd winding parity, and points on the wires belong to the closed domain. The geometry of a part is the union of these face images. A 3D trim witness is evaluated as \(S(c(s))\) for the p-curve \(c\) of the same face. A STEP adapter retains the original p-curve chart and records the maximum UV gap enclosed by the short connectors it adds between wire endpoints. The certificates apply to this explicitly defined nominal boundary.

We represent clamped nonperiodic NURBS spans by homogeneous Bézier control nets, using knot insertion with outward-rounded arithmetic. Patch subdivision restricts a control net in its native chart. For an elementary analytic face, we construct a convex enclosure in the native rectangular parameter cell. For a circular arc of width below \(\pi\), the arc lies in the triangle given by its endpoints and the intersection of the endpoint tangents. Larger arc ranges use a bounding square. A cylindrical cell takes the product of this arc hull and an axial interval. Cone, sphere, and torus cells combine the azimuth enclosure and a meridian enclosure by multilinear interpolation.

The static problem computes \(d(t)=\min_{x,y}\|x-y\|\) over the two boundary sets at time \(t\). The continuous problem minimizes \(t\) on a user-supplied interval subject to \(d(t)\le\delta\), where \(\delta\ge0\). A continuous result is a safe prefix, optionally paired with a feasible time \(t_+\) and two native points. A result without a feasible time remains unresolved if potential event intervals are still present. A point pair lying exactly at zero clearance can remain unresolved under finite precision.

## 4. Directional time bounds

### 4.1 Relative motion

We express \(B\) in the frame fixed to \(A\). A point \(p\) on \(B\) has trajectory

\[
x_p(t)=q_0+v t+R_a(\theta_0+\omega t)p,
\]

where \(a\) is a fixed unit axis. Let \(p_\parallel=a(a^\top p)\). Rodrigues' formula gives \(R_a(\theta)p=p_\parallel+(p-p_\parallel)\cos\theta+(a\times p)\sin\theta\). Thus every fixed direction \(n\) yields coefficients \(\alpha_i,\beta_i,\gamma_i,\eta_i\) for the support point \(p_i\) with projection

\[
f_i(t)=\alpha_i+\beta_i t+\gamma_i\cos(\theta_0+\omega t)
       +\eta_i\sin(\theta_0+\omega t).
\]

The direction can be proposed by any heuristic. The implementation starts with the vector between the enclosure centers at the middle of the current interval. For a mixed trim pair separated along at least two coordinates, it uses the displacement between the closest points of the boxes. The predicate retains the norm of either direction.

### 4.2 Directional exclusion

Let \(X\) and \(Y\) be the fixed and moving convex envelopes of a cell pair, with support vertices \(\{a_j\}\) and \(\{p_i\}\). Set \(H=\max_j n^\top a_j\) and \(K=H+\delta\|n\|\). Define the outer candidate time set for the direction \(n\) by

\[
\mathcal{I}_n=\bigcup_i\{t:f_i(t)\le K\}.
\]

**Proposition 1.** Any time at which the two trimmed cell images are separated by at most \(\delta\) belongs to \(\mathcal{I}_n\).

**Proof.** Take a feasible pair \(x,y\) with \(\|y-x\|\le\delta\). The hull of \(A\) gives \(n^\top x\le H\). Rigid motion preserves the convex combination expressing \(y\) in the moving hull, hence \(\min_i f_i(t)\le n^\top y\). Cauchy–Schwarz gives \(n^\top y\le H+\delta\|n\|\). The argument holds in every tested direction, so an intersection of conservative directional candidate sets is conservative as well. \(\square\)

For fixed orientation, each \(f_i\) is affine in time. We clip its lower affine envelope directly, using the lower slope for nonnegative time and the upper slope for negative time. For rotation, we enclose the sine and cosine ranges over the input interval. If all lower projections exceed the upper threshold, the interval is excluded. If some upper projection lies below the threshold, the interval remains a candidate for this direction. We bisect residual intervals and merge adjacent candidates. When a clipping cap is reached, remaining intervals stay in the outer candidate set.

Consider the center trajectory \((\cos t,\sin t)\) against a stationary center at \((0.9,0.9)\), with \(t\in[0,\pi/2]\). The swept axis-aligned box of the moving center includes the stationary center. Yet their separation projected onto \((-1,-1)/\sqrt{2}\) remains at least \(0.9\sqrt{2}-1\). For spheres of radius \(0.002\) at both centers, the minimum surface clearance is approximately \(0.26879\). This explains how a directional bound can certify a rotating near miss even when an interval-wide swept box overlaps.

### 4.3 Trim decisions and the coverage invariant

We mark a spatial rectangle outside only after proving it avoids every trim curve and certifying that an interior point has even winding parity. Parametric trim bounds test whether a curve can meet the rectangle. When a ray-origin point lies clearly to one side of a curve enclosure, the crossing test can use the enclosure and the sides of the end points. Ambiguity leaves the cell mixed. Candidate upper bounds use points certified inside a trim domain or points obtained by evaluating a trim curve.

**Proposition 2.** Suppose every hull contains its cell image, every trim exclusion removes only an empty cell, and directional exclusion uses outward enclosures. If the initial collection covers the input domain, then the earliest event is either enclosed by a remaining candidate or occurs at a confirmed witness time.

**Proof.** A task covers a product of two parameter cells and a time set. Spatial subdivision partitions the parent rectangle with shared boundaries and retains the parent time set on each child. Temporal subdivision covers its parent interval. Proposition 1 and the trim condition rule out discarded regions, and a swept-box separation removes only pairs at distance greater than \(\delta\). A finite witness time can end the search over later times because an event is already feasible there. A task halted by a depth or work budget contributes its earliest possible time to the retained frontier. Induction on task operations preserves coverage. If the minimum retained time is \(t_-\) and a witness exists at \(t_+\), then \(t_-\le\tau_\delta\le t_+\). An empty frontier and absence of a witness certifies the entire interval. \(\square\)

The same argument gives static distance bounds. A bounding-box separation is a lower bound for a cell pair, while a feasible point pair supplies a global upper bound. Cells pruned at an earlier upper bound cannot improve a later, smaller upper bound. The minimum lower bound among retained cells and the current upper bound enclose the global distance. During continuous search, the choice between spatial and temporal subdivision compares the size of a cell pair's static bound spread with the distance the moving cell may travel in the current interval. This choice affects work and resolution. The coverage invariant applies to either split.

### 4.4 Radial and axial bounds for circular rims

Cartesian convex enclosures may fill the hole of an annular face. An eccentric pin can then overlap a bounding box while its nearest feasible point remains far from the hole rim. We retain a radial interval and an axial interval for selected faces. The bounds are computed once per source face and reused by all its spatial descendants.

Let \(e\) be a unit axis through \(o\). For a point \(x\), write \(r(x)=\|e\times(x-o)\|\) and \(z(x)=e^\top(x-o)\). If \(R_A,R_B\) and \(Z_A,Z_B\) enclose the values of these coordinates for a face pair in a common frame, define \(g_r=\operatorname{dist}(R_A,R_B)\) and \(g_z=\operatorname{dist}(Z_A,Z_B)\).

**Proposition 3.** The face-pair distance is at least \(\sqrt{g_r^2+g_z^2}\). Under fixed relative orientation and affine translation, a uniform lower radial gap \(g_r\) over a motion interval yields an outer candidate time interval by requiring axial overlap to within \(\sqrt{\delta^2-g_r^2}\). If \(g_r>\delta\), the entire interval is excluded.

Orthogonal decomposition gives a squared axial difference plus a squared transverse difference. The reverse triangle inequality bounds the transverse difference from below by the difference of the two radii, establishing the first claim. With fixed orientation the axial projections are intervals translated by a scalar affine function of time. The two inequalities for allowable axial overlap can therefore be clipped on the time axis. Outward bounds on coefficients, the radial gap, and the axial allowance preserve coverage at the interval endpoints.

For a plane, an arc center proposes a pivot in the original UV chart. We bound the minimum and maximum radius of every trim curve relative to the pivot. A UV disk of radius below the minimum meets no boundary, so one parity test anywhere inside that disk certifies its membership. An outside disk produces a positive lower radius. The upper radius comes from the boundary, since the maximum distance to a point on a bounded planar domain occurs on its boundary. The singular-value enclosure of the plane's two basis vectors transfers the UV radii to physical units. For a cylinder, we obtain the bounds from its native axis, radial basis, and axial parameter interval. Eigenvalue bounds on the two-by-two Gram matrices suffice for the required singular-value enclosures. Trim evaluation and feasible witnesses still use the original face geometry.

To combine the two envelopes, let \(s=|e_A\times e_B|\), \(c=e_A^\top e_B\), and \(D=o_B(t)-o_A\). If a moving point has radius \(r_B\) and axial coordinate \(z_B\) in its own frame, its radius around the fixed axis is between \(|c|r_B-E\) and \(r_B+E\), with \(E=\|e_A\times D\|+s|z_B|\). Its axial projection is within \(sr_B\) of \(e_A^\top D+cz_B\). These bounds follow by projecting the radial plane and applying the triangle inequality. We cache the resulting interval per face pair and intersect it with the current search interval before directional clipping. The intersection is another exclusion in the coverage invariant. An early sample inside a shortened interval can confirm a witness, while an unsuccessful sample leaves the exclusion unchanged.

## 5. Evaluation

### 5.1 Protocol

The experiment uses generated STEP solids and one positive-weight rational B-spline face. The box pair provides a planar entry time. A pin through a bore tests a true clearance of \(0.75\) length units, and an eccentric pin gives a known approach to the circular rim. Two spheres give an exact center-distance model for a rotary near miss and a rotary threshold entry. A bounded spline face and a moving box exercise support for rational geometry during rotation.

All variants use the same STEP geometry, nominal relative path, clearance threshold, spatial stopping resolution \(2\times10^{-4}\), temporal target \(5\times10^{-4}\), and work cap of 200,000 tasks. The swept-box baseline bounds the original curvilinear trajectories componentwise. The chord-box variant encloses each endpoint-interpolated trajectory with the error bound \(M h^2/8\), with \(M\) an upper bound on the magnitude of the second derivative over a time interval of width \(h\). These baselines share the same trim and spatial search. The no-inheritance variant begins a spatial child with its parent's interval before directional clipping. The late-trim variant moves trim classification after the swept-box test. Runs exclude STEP loading and model preparation, use one warm-up per configuration, and rotate the timing order across repetitions.

Nine measured repetitions per configuration ran on an AMD Ryzen 9 5900X with Windows x64, MSVC 19.44 Release, and OCCT 7.9.3. The six scenarios and five configurations yielded 270 measurements. The runner checks hit intervals and clear outcomes against independently derived geometric references. The raw measurements and aggregation procedure accompany the implementation. [Raw measurements](../benchmarks/results.csv) · [Aggregation](../benchmarks/summarize.py)

### 5.2 Accuracy and search cost

All 135 hit intervals enclosed their analytic event time, with a maximum width of \(4.992\times10^{-4}\) after rounding. All 135 no-event queries certified the full input interval. Table 1 gives median runtime and visited tasks under the matched tolerances.

| STEP scenario | Directional ms / tasks | Swept box ms / tasks | Chord box ms / tasks | Directional hit interval width |
|---|---:|---:|---:|---:|
| Box entry | 0.876 / 15 | 10.783 / 297 | 13.288 / 297 | 0.000146484 |
| Bore passage | 25.417 / 391 | 47.238 / 977 | 53.856 / 977 | — |
| Bore rim entry | 1029.308 / 33,728 | 1446.567 / 97,861 | 2423.825 / 97,861 | 0.000499177 |
| Rotation clear | 1.076 / 3 | 4.428 / 39 | 7.345 / 39 | — |
| Rotation entry | 161.882 / 1,293 | 741.676 / 10,955 | 1134.470 / 10,951 | 0.000271227 |
| Spline clear | 0.0315 / 1 | 0.0312 / 1 | 0.0478 / 1 | — |

The rotating near miss illustrates the effect of retaining directional information across time. The directional variant visited 3 tasks, compared with 39 for each box variant. In the rotary entry, it reduced the search from 10,955 tasks to 1,293 relative to swept boxes. The eccentric bore remained the dominant cost. It visited 33,728 tasks and took about one second even with directional clipping, consistent with the need to resolve both the circular rim and the narrow incoming time interval.

### 5.3 Ablation

Omitting inheritance increased the visited tasks from 15 to 122 at box entry and from 1,293 to 2,678 at rotary entry. On those scenes, the median runtimes increased from 0.876 to 5.390 ms and from 161.882 to 448.194 ms, respectively. The effect was smaller for bore passage, where inheritance reduced visits from 479 to 391. Both trim schedules made the same geometric decisions and visited the same number of query tasks on this suite. Early trim changed the order and cost of classification without changing the output. These measurements identify temporal inheritance as a direct contributor to the observed reduction in search work.

### 5.4 Targeted optimization of bore-rim entry

In the original eccentric pin scene, 17,164 of 17,295 subdivisions were spatial. The bounding boxes of an annular face contained empty material around the bore. Retaining both a radial lower gap and an axial gap allowed the search to exclude an interval approaching the analytical entry time before refining spatial cells.

We reran the controlled suite with the radial and axial filter enabled and disabled in the same binary, with nine query-only repetitions per configuration. All thresholds, tolerances, and work caps remained those in Section 5.1. The old and new median latencies for bore-rim entry in this paired run were 693.056 ms and 1.240 ms, a 559-fold ratio. Task count fell from 33,728 to 25, and the first-entry interval width tightened from \(4.992\times10^{-4}\) to \(3.125\times10^{-5}\). Both intervals contained the analytic entry at \(0.2861928813\). Centered bore passage fell from 16.957 ms and 391 tasks to 0.403 ms and 25 tasks. The other scenes continued to satisfy the original reference checks. [Controlled-suite raw measurements](../benchmarks/axial-results.csv)

### 5.5 Production STEP assembly protocol

We imported the published wheel and lidar support assemblies of the OpenAMRobot MMP platform with OCCT XCAF. The solver queried pairs of original component occurrences with their native geometry and placement. The wheel cases used two DIN 125 flat washers, a DIN 7980 lock washer, and two DIN 912 M10 screws. The lidar case used the support and one of its M4 rivet nuts. The motions were prescribed relative translations from offset positions in the source assembly frame. The two eccentric bolt trajectories added a fixed transverse offset of 0.2 mm. [Original STEP assemblies](https://github.com/openAMRobot/openamr-platform-hw/tree/8ed2b06a46115abe2dffadbf0d902ae731d278d1/mechanical/cad/production_files) · [Recorded component IDs, motions and reproduction](../benchmarks/assemblies/README.md)

The two kernel configurations used identical positive clearance thresholds, a spatial stopping resolution of \(2\times10^{-4}\) mm, a time target of \(5\times10^{-4}\), and a 200,000-task cap. Each result was measured nine times after warm-up. We alternated configuration order and timed the query after STEP loading and model preparation. For numerical cross-validation, we used OCCT boundary distances with a \(10^{-6}\) mm margin and adaptive time subdivision using the translation speed. The certified exclusion decisions came from our interval bounds and native point witnesses.

### 5.6 Production STEP assembly results

The 54 wheel queries all returned first-entry brackets that overlapped the OCCT cross-check and met the same time-width target. Both configurations certified the full interval of the 18 lidar pre-seat queries. Table 2 compares median query latency and visited tasks. The three wheel entries gained 2.80 to 5.97 times in median latency. The lidar pre-seat case showed a smaller gain, with 54,843 tasks still visited after enclosure filtering. [Wheel measurements](../benchmarks/assemblies/wheel-results.csv) · [Lidar measurements](../benchmarks/assemblies/lidar-results.csv)

| Production assembly scenario | Directional search ms / tasks | With radial and axial bounds ms / tasks | Median latency ratio | Optimized interval width |
|---|---:|---:|---:|---:|
| Wheel washer stack | 23.887 / 205 | 7.605 / 109 | 3.14 | 0.000468750 |
| Wheel eccentric bolt 1 | 905.886 / 26,121 | 151.779 / 3,173 | 5.97 | 0.000437224 |
| Wheel eccentric bolt 2 | 925.171 / 29,436 | 330.183 / 8,985 | 2.80 | 0.000494202 |
| Lidar nut pre-seat | 1647.266 / 59,031 | 1508.700 / 54,843 | 1.09 | Full interval clear |

The full lidar insertion tests the same parts along the path to the nominal seat. Both configurations exhausted the 200,000-task cap and returned a first-entry interval approximately \([0.386675,0.421701]\). It overlapped the OCCT numerical cross-check near \(0.4001\) and exceeded the \(5\times10^{-4}\) width target. This case exposes the cost of refining several close faces on the same path. [Full insertion diagnostic](../benchmarks/assemblies/lidar-diagnostic.csv)

### 5.7 Segmented rotation, a second STEP source, and edge audit

The common rotation fixtures compare the original-trajectory query with Chen's time-dependent OBB search on piecewise-linear chords. The wrapper pads projections by the threshold and a bound on geometric and rounding deviation, verifies proposed exclusions outward, and checks the upper witness on the original trajectory. The counterbore and Voron Trident Front Idler B interfaces add a swept-box variant to the paired directional and radial/axial ablations. The trim-edge audit reconstructs witness residuals from coordinate intervals and uses a closed-form closest point on the circular inner trim. [Protocol, raw CSVs and independent verifier](../benchmarks/clearance-followup/README.md)

## 6. Scope

The algorithm treats separate boundary sets and returns native point pairs. Volumetric containment, recovery from initial interpenetration, and sustained-contact response require additional semantics. Relative poses outside the fixed-axis motion model require a different trajectory enclosure. The STEP adapter supports elementary analytic faces and clamped nonperiodic splines. Invalid or unsupported face data produces an import error with the offending face ID. A small nominal UV connector is part of the geometry evaluated by the prototype.

The measured speedups apply to the reported controlled scenarios and selected production assembly interfaces. The radial and axial filter applies to the supported face types at fixed relative orientation. Its value depends on whether the radial enclosure separates a significant part of the original candidate domain. The screw and connecting-rod STEP files supplied with OCCT 7.9.3 also imported successfully and yielded static distance brackets narrower than 0.1 mm against a displaced reference block.

## 7. Conclusion

Directional support bounds transfer a fixed-axis trajectory into scalar exclusion tests. Radial and axial enclosures add a bound on the joint margin at a circular rim under translation, where Cartesian hulls discard the hole. Both exclusions preserve coverage during spatial refinement. The paired experiments show that the cylindrical enclosure can reduce the cost of the controlled eccentric-bore query and of bolt and washer queries from an original STEP assembly. The full lidar insertion remains limited by spatial refinement under the evaluated task cap.

## Appendix. Trigonometric enclosures

The primitive interval operations enlarge round-to-nearest binary64 results by one representable value in the required direction. To bound sine and cosine, the implementation reduces an input interval using an explicit binary64 enclosure of \(2\pi\). It evaluates the endpoints of the reduced interval and includes the exact extrema at enclosed critical points. If efficient range reduction cannot be certified within the supported evaluation neighborhood, the trigonometric range is conservatively set to \([-1,1]\).

For a reduced endpoint \(x\) with \(|x|\le4.25\), write \(y=x/4\). Evaluate the alternating sine and cosine series through terms of degree 21 and 20, respectively, and apply two angle doublings. On \(|y|\le17/16\) the first omitted term has absolute magnitude below \(10^{-20}\). The recurrence for each noninitial term multiplies the previous term by at most \((17/16)^2/2<0.565\) in magnitude. For round-to-nearest binary64 this bounds the per-term absolute arithmetic error by \(2^{-48}\). Summing the errors and roundoff of eleven additions bounds the initial series errors by \(2^{-43}\). Each doubling amplifies that error by less than a factor of five in the evaluated neighborhood, including its arithmetic roundoff. The final endpoint enclosures use radius \(2^{-38}\).
