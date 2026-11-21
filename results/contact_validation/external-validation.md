# Clearance slider-crank and copper-ring insertion

## Clearance slider-crank

The horizontal rig has a 50 mm crank, 300 mm rod and 22.25 mm sleeve diameter. Nominal pin diameters follow the reported clearances. The simulation places the rod center of mass at midspan and uses an ideal slider guide. The motor drives the crank at constant angular speed. The dynamic degrees of freedom are rod angle and slider translation. The first four revolutions are warm-up. The last two are compared at the published encoder angle, with no phase adjustment. The 200 rpm, 0.25 mm position comparison uses the single revolution shown in Fig. 5(b).

The Python extractor uses the vector paths of the measured traces in Figs. 5(b), 6 and 8. The source PDF checksum and panel identifiers identify the digitization. The source figures represent acceleration sampled for publication. The original accelerometer stream, sample rate and filter settings were unavailable. Retained simulated extrema are also reported to show the effect of resampling. Each integration uses the same radial force differential, HHT settings, STEP solids and contact parameters. The triangle runs differ in the chordal deflection and calculate signed gaps on tessellated surfaces. [Experimental rig and test conditions](https://doi.org/10.1007/s11071-010-9899-8)

At 200 rpm and 0.25 mm, the B-rep travel RMSE is 0.66 mm. Acceleration RMSE is 10.19 m/s² for native faces, 50.89 m/s² for the 5 µm mesh, and 140.46 m/s² for the 25 µm mesh. The maximum radial gap error on the nominal circular STEP shapes is below 0.001 µm for B-rep and 2.48 µm for the 5 µm mesh. The negative acceleration extremum within the first half revolution is at 46.6° in the physical test and 3.6° in the B-rep simulation. Halving the step from 10 to 5 µs moves the predicted extremum to 5.9°. The 30-revolution simulation accepts all 900,000 time steps.

The median B-rep integration time across the seven conditions is 2.40 times the 5 µm mesh time and 3.65 times the 25 µm mesh time. Each backend and condition was timed once.

### Why the acceleration differs

Flores et al. excluded their shaft torque signal because of a persistent residual component. They used a separate slider accelerometer, LVDT and crank encoder in the published comparison. The authors inferred a separation near 43° and recontact near 63° from Fig. 5(d). Their rig could only infer joint impact from slider acceleration. Raw samples and accelerometer processing settings were unavailable for the present comparison. [Flores et al., Sections 3–5](https://doi.org/10.1007/s11071-010-9899-8)

The first-half-revolution negative acceleration extremum is at 46.6–46.7° in the published experimental traces in Figs. 5(d), 6(b) and 11(c), and at 6.2° in the authors' numerical trace in Fig. 11(d). These are extrema in the digitized plots. The corresponding figures may show the same experimental record. Our B-rep trace has a positive normal contact force throughout the reported 43–63° window in both scored revolutions at 10 and 5 µs and in the final two revolutions of the 30-revolution run.

At 200 rpm, a 43° phase offset corresponds to about 36 ms. Applying a common ±43° offset to the periodic simulated position raises the position RMSE from 0.66 mm to 28.95–29.16 mm. The position agreement argues against a uniform shift of the crank reference. The publication lacks the raw synchronized signals needed to assess a delay specific to the accelerometer channel.

The published rig mounts the sleeve on the rod and the journal on the slider. The baseline replay decks interchange those attachments. This preserves the nominal circular gap calculation and changes the tangential slip and moment arm. To isolate this detail at 200 rpm and 0.25 mm, assign the rod body fields to the sleeve, assign the slider body fields to the pin, and move the orbit drive from body 1 to body 0. Retain the other inputs and six revolutions at 10 µs. In this test the negative extremum occurs at 0.3° and the waveform RMSE is 13.07 m/s². The positive normal force persists from 43° to 63° in both scored revolutions. The original authors discuss joint flexibility, preload friction, misalignment, and roundness. Their dissipation term uses the approach speed at each impact in Eq. (3), while the shared solver uses a fixed rate coefficient for the case. [Flores et al., Eqs. (3)–(5) and Figs. 5–6 and 11](https://doi.org/10.1007/s11071-010-9899-8)

- [Per-condition and per-backend metrics](clearance/summary.json) · [Travel and acceleration comparison](figures/clearance-slider-validation.pdf)
- [200 rpm and 0.25 mm acceleration samples](clearance/paired-200rpm-0.25mm-brep.csv) · [Position samples](clearance/position-200rpm-0.25mm-brep.csv)
- [STEP fixtures and scene decks](clearance/models)
- [Plan-view animation](animations/clearance-slider-plan.mp4) · [Isometric animation](animations/clearance-slider-isometric.mp4)

### Dissipation and compliance revision

The follow-up publication describes a friction plate between the rod and slider and viscous resistance from the guide lubricant. The revised STEP scene uses the physical pin and sleeve mounting, a linear guide resistance of 2 N·s/m, and 1.5 N of planar sliding resistance under prescribed preload. The contact rate is five times the reference rate computed with the scaled stiffness. The normal exponent is 1.35 and the effective elastic force at 0.3 µm is 0.533% of the bare Hertz value. This fitted compliance represents the rest of the load path. The geometry queries and depth rejection rule are the same as in the corrected-mounting control. [Koshy et al., Sections 2–3](https://doi.org/10.1007/s11071-013-0787-x)

The model was selected on the 200 rpm, 0.25 mm curve, with attention to the full-rate peaks and the 90–180° response. The parameters are unchanged at the three other speed and clearance settings marked by their separate runs. Each run scores the last two of six revolutions. The RMSE at the integration times uses a piecewise linear reconstruction of the published accelerometer trace. The comparison figure shows the full-rate acceleration envelope for each plotted half-degree bin. The revision improves the broad response while the primary negative peak at 46.6° and the inferred separation still differ at 200 rpm.

On the calibration curve the full-rate RMSE falls from 13.05 to 7.86 m/s² relative to the corrected-mounting control. The 90–180° RMSE falls from 11.57 to 5.99 m/s² and the raw peak in that interval falls from 52.09 to 36.95 m/s². The travel RMSE changes from 0.65 to 0.68 mm, while maximum equivalent compression rises from 0.77 to 21.78 µm. Halving the step changes the waveform by 0.0015 m/s² RMS on the phase grid. The 5 µm BVH run has 7.90 m/s² full-rate RMSE. The three transfer cases have full-rate RMSEs of 2.67, 13.53 and 20.85 m/s² at 150 rpm and 0.25 mm, 200 rpm and 0.50 mm, and 250 rpm and 0.25 mm, respectively. The 150 rpm simulation still has a 36.89 m/s² positive peak, compared with 11.07 m/s² on the published trace.

The published negative peak lies inside the interval the authors inferred to be open. If the joint were open at that sample, the revised model at its sliding speed of −0.85 m/s would have at least +0.20 m/s² of axial acceleration from the guide and support plate alone. The experimental acceleration near −50.7 m/s² requires additional load in this reduced model. The published trace cannot establish whether the contact was open at the peak or quantify the response of another mechanical load path.

- [Revision metrics and fixed parameters](clearance/revised/summary.json) · [Acceleration comparison and full-rate envelopes](clearance/revised/dissipation-comparison.pdf)

### Explicit structural flexibility

The pin, slider carriage, sleeve tip and rod core participate in one HHT solve. Translational Kelvin–Voigt elements connect the two internal nodes to their mounts. The rod tip element uses the rotating local axial and transverse directions. The selected case also allows lateral motion of the guide carriage. Contact uses the nominal Hertz law. Local penetration and structural deformation have separate meanings in this model.

For 200 rpm and 0.25 mm, nine sensitivity cases leave the measured negative acceleration peak unresolved. The selected case uses axial and lateral pin-support stiffnesses of 1.6 MN/m, rod-tip axial and transverse stiffnesses of 80 and 0.35 MN/m, and lateral guide stiffness of 0.45 MN/m. Its maximum acceleration in the quiet window is 35.69 m/s² versus 36.95 m/s² for the equivalent-compliance model. Quiet-window RMSE is 6.01 versus 5.99 m/s². The predicted acceleration at 46.6° remains approximately −14.9 m/s². The measured trace is −50.7 m/s² at that phase.

The results archive includes the raw carriage, gap, and contact-force values at integration times and the selected STEP deck. Peak and RMSE metrics use the final two revolutions of six. The reference waveform comes from the published encoder-phase plot. The code applies neither a registered phase offset nor a filter to the simulation.

- [Screened cases](clearance/flexibility/screening.json) · [Selected case and step refinement](clearance/flexibility/summary.json) · [Acceleration and raw envelopes](clearance/flexibility/structural-flexibility.pdf)

## Archived robot copper-ring insertion

The archived records provide commanded and measured wrist pose and wrist force/torque for 300 attempts. The dataset covers four tooling slots and 13 commanded lateral offsets. The paired CSV includes transformed measurements from Mihael Simonič and Matevž Majcen Horvat's *Copper Rings Insertion Attempts Dataset*, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Each trial is registered to the first minimum of commanded Z. The model uses measured axial position relative to that command target. The measured tool-frame axial force is centered on the median of the first 35 samples. The evaluation window covers sample offsets −28 through +23. The files contain no timestamps, so the curves use sample index and the model uses rate-independent compliance.

The STEP model comprises an annulus and a planar stop. Slot groups 1–2 fit a common stiffness and one effective stop level per offset. Slot groups 3–4 provide the held-out force records. This probes the portability of the identified axial contact law. Exact lug and socket surfaces, workcell frame registration, ring deformation and controller parameters are absent from the model. The annulus dimensions are illustrative. The full industrial assembly, lateral force and wrench timing are outside the measured comparison. [Versioned dataset](https://doi.org/10.5281/zenodo.6364950) · [Workcell and ring description](https://doi.org/10.3390/s22207962)

The 150 held-out trials give 2.11 N waveform RMSE, against 3.01 N for zero force and 0.76 N for within-condition repeat variation. The absolute error in the mean force of the final eight samples averages 1.18 N. Measured and predicted median peaks are 8.10 N and 2.34 N. The replay misses most of the transient peak across both geometry backends.

- [Held-out metrics and fitted parameters](insertion/summary.json) · [Paired data for all 300 trials](insertion/paired.csv)
- [Axial STEP fixtures and representative replay deck](insertion/models)
- [Measured and computed axial force](figures/copper-ring-validation.pdf) · [Front animation](animations/copper-ring-front.mp4) · [Isometric animation](animations/copper-ring-isometric.mp4)

## Reproduce

Run these commands with the [OCCT release runtime on PATH](../../README.md#step-queries-on-windows). Install the Python dependencies NumPy, SciPy, PyMuPDF, Matplotlib and imageio-ffmpeg. Put the [publisher version of the Flores paper](https://link.springer.com/content/pdf/10.1007/s11071-010-9899-8.pdf) in the source path below. The copper-ring runner downloads its archive if the specified file is absent and checks the publisher's MD5. The runners write the STEP solids, scene decks and unthinned simulation traces in the build directory. The plotting script writes PDF and PNG figures and optionally four MP4 animations.

~~~powershell
cmake --build build/occt --config Release --target nc_contact nc_step_tests
ctest --test-dir build/occt -C Release --output-on-failure
python benchmarks/contact_validation/clearance.py build/occt/Release/nc_contact.exe build/external-validation/source/springer.pdf build/external-validation/clearance results/contact_validation/clearance --extended
python benchmarks/contact_validation/clearance_revision.py build/occt/Release/nc_contact.exe build/external-validation/slider-revision results/contact_validation/clearance results/contact_validation/clearance/revised
python benchmarks/contact_validation/clearance_flexibility.py build/occt/Release/nc_contact.exe build/external-validation/flex-screen results/contact_validation/clearance results/contact_validation/clearance/flexibility
python benchmarks/contact_validation/clearance_flexibility.py build/occt/Release/nc_contact.exe build/external-validation/flex-selected results/contact_validation/clearance results/contact_validation/clearance/flexibility --selected compliant-guide
python benchmarks/contact_validation/insertion.py build/occt/Release/nc_contact.exe build/external-validation/source/insertion-dataset.zip build/external-validation/insertion results/contact_validation/insertion
python benchmarks/contact_validation/external_plots.py results/contact_validation --animate
python benchmarks/hht/paper.py results/hht
~~~

The recorded integration time excludes STEP import, mesh generation, diagnostic sampling and CSV output. A replay timing measures contact evaluations separately from loading or sampling. Planar copper-ring forces coincide with both backends within the exported precision.
