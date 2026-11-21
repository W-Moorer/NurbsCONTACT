# Contact on deforming trimmed CAD surfaces

The second study targets contact between elastic bodies whose exposed surfaces are described by a trimmed CAD parameterization. The geometric tests below cover a deforming rational patch. The cantilever tests cover an independently driven first-mode model.

## Material chart and interval enclosure

Let \(B_i(u,v)\) be the tensor-product Bernstein basis, \(w_i>0\) fixed rational weights, \(P_i\) the rest poles, \(D_{ik}\) the Cartesian pole displacement in mode \(k\), and \(\eta_k\) the modal amplitudes. Use the surface

\[
X(u,v,\eta)=
\frac{\sum_i w_i B_i(u,v)\bigl(P_i+\sum_k\eta_k D_{ik}\bigr)}
     {\sum_i w_i B_i(u,v)}.
\]

The trim curves are defined in the material chart. The image of each trim moves with \(X\). This representation assumes fixed material trim topology. Multi-patch imports also require consistent modal fields on shared edges.

For a box of modal amplitudes, the homogeneous numerator coordinates of pole \(i\) are enclosed by \(w_iP_i+w_i\sum_k[\eta_k]D_{ik}\) with outward rounding. The denominator pole stays \(w_i\). Any de Casteljau restriction to a parameter rectangle encloses the corresponding restricted homogeneous control net for every amplitude in the box. Its restricted weights are positive. The rational basis therefore takes convex combinations of its Cartesian poles on the restricted rectangle. The enclosing pole boxes give a valid support bound for the whole amplitude box, including the trim domain's intersection with the rectangle.

The core implements this update in [`BezierSurface::deformed`](../src/geometry.cpp). The [geometry test](../tests/queries.cpp) uses a weighted surface with \(z=2\eta u^2/(1+u^2)\). It checks point evaluation and restricted support over \(\eta\in[0.01,0.03]\). At the fixed amplitude \(\eta=0.02\), the surface and a plane at \(z=0.06\) have analytical minimum gap \(0.0456\) on the trim edge \(u=0.75\), in model length units. The existing distance query brackets this gap with width at most \(0.001\) and returns a feasible witness. For a time window with interval amplitudes, use the patch family for exclusion and evaluate candidate witnesses at specified times and amplitudes.

## Coupling to implicit dynamics

A surface mode has pointwise Jacobian \(\Phi_k=\partial X/\partial\eta_k\), using the same rational basis as \(X\). For two surfaces at a regular isolated closest pair, let \(n\) point from \(A\) to \(B\). If the closest pair is interior or varies along a fixed active trim feature, the derivative of the positive distance is

\[
\partial_{\eta_k}d = n^\mathsf{T}(\Phi_{B,k}-\Phi_{A,k}).
\]

Tangent terms cancel by stationarity along that feature. Multiplying this gap Jacobian by the normal force gives generalized contact forces by virtual work. A time interval bound requires a conservative amplitude enclosure over that interval. Feature changes and singular points require separate handling of one-sided derivatives. The deforming-patch test verifies geometry for prescribed amplitudes. Implicit coupling of the moving patches through their normals, generalized forces, and consistent tangents remains open.

## Driven cantilever measurements

The public dataset provides simultaneous base and tip acceleration at 1024 Hz for a 170 mm steel cantilever with a 31 g tip mass and two stops. The generalized inertial input is tabulated from measured base acceleration. A single bending coordinate translates the tip collision patch, and the sensor at 150 mm is evaluated using the assumed mode shape. The frequency, loss, participation, and stop parameters are calibrated on the first two random records and held fixed thereafter. [Anastasio–Marchesiello dataset](https://doi.org/10.5281/zenodo.19205934)

On the held-out 13 Hz stepped-sine experiment, the raw acceleration RMSE is \(0.871\) m/s² and correlation is \(0.997\). The same modal model with stops disabled gives \(4.727\) m/s² RMSE. Halving the HHT step from 122 to 61 μs changes the waveform by \(0.009\) m/s² RMS. In the last excitation block the simulated fundamental amplitude reaches \(80.3\%\) of the measured amplitude and its phase error is \(-2.33^\circ\). The stronger held-out random test gives \(5.426\) m/s² RMSE and \(0.629\) correlation. All comparisons use the instrument timestamps and the raw simulated output. The plane contacts give the same sampled results in STEP and triangle BVH. [Reproduction and archived traces](../results/contact_validation/external-validation.md) · [Measured and simulated waveforms](../results/contact_validation/cantilever-gap/cantilever-gap-response.pdf)
