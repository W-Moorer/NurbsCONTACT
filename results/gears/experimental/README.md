# External gear-rig comparison

The runs compare the same HHT contact law on native STEP flanks and their OCCT triangle BVHs. The fixed fine-mesh linear deflection is 1.5625 µm. One case per rig sets parameters, with the remaining cases held out. Both contact backends receive the same calibrated parameters.

## NASA single-tooth load

The four tests correspond to Fig. 7 of NASA TM-103232. The recorded forces come from digitizing the solid experimental curves on PDF page 6, with roughly 5.8 N per source pixel. Each CSV contains the original pixel clicks, converted roll angle, and force. The reconstruction uses the 28/28-tooth STEP flanks, four torsional inertias as printed in SI, and the reported input and load shaft stiffnesses. The contact stiffness is the fitted equivalent normal stiffness. The listed gearbox stiffness denotes a distinct quantity in the reference.

The printed customary entries for the four inertias give an order-of-magnitude difference under standard force-inch conversion. The main runs use the printed SI values. The times-ten inertia run at Fig. 7(c) documents sensitivity with the fit unchanged.

The model outputs a physical tooth index, driven-flank roll angle at its witness, and each tooth-pair normal force. RMSE uses a 0.125° grid from 12° to 29.5°, with the lower endpoint set to 12.625° for (d). Averages use complete simulated engagements in the second half of each record. Figure 7(a) is the fit condition. The (b), (c), and (d) states are held out. The roll-angle origin is common. [Original NASA report](https://ntrs.nasa.gov/api/citations/19910013253/downloads/19910013253.pdf)

## UNSW encoder transmission error

Three CC BY 4.0 healthy records supply both encoder waveforms, the input tachometer, and actual mean input torque. The acquisition rate is 200 kHz. The processing repeats the DTE reconstruction using rising and falling edges separately. It uses input tachometer phase, zero-crossing interpolation, the two base radii, and complete 44-revolution hunting-tooth blocks. The reported waveforms are zero-mean mesh-synchronous averages, with 128 samples per mesh cycle. The block range is empirical spread within the record.

The 27/44-tooth ideal involute STEP reconstruction uses nominal geometry from the thesis and two estimated effective shaft inertias. The published raw encoder data supply speed and torque. Shaft transfer dynamics and manufacturing deviations are outside the reconstruction. The nominal 5 Hz and 10 N·m file supplies the calibration objective. The 5 Hz/5 N·m and 10 Hz/15 N·m files are held out. One angular shift is fixed from the calibration, then applied unchanged to both models at both test conditions. [CC BY 4.0 dataset](https://data.mendeley.com/datasets/p72x3m92cv/1) · [Test geometry](https://doi.org/10.26190/UNSWORKS/24007)

## Artifacts

- [NASA aligned single-tooth curves](figures/nasa-single-tooth.pdf)
- [UNSW aligned DTE and harmonic amplitudes](figures/unsw-encoder-dte.pdf)
- [Scored curves](curves) · [Summary with per-condition errors](summary.json)
- [Derived measurements](measurements) · [Source URLs and SHA-256](measurements/sources.json)
- [Calibration grid and trials](calibration) · [HHT traces and work counts](runs)
- [Updated industrial supplement](../paper/industrial.pdf)

NASA 4000 rpm and 16% torque exhibits a measured loss of contact while both modeled curves retain positive normal load in the center of that interval. The high-speed UNSW test has a dominant second mesh harmonic whose amplitude is substantially underpredicted. Time-step refinement has small effects on the two refined cases relative to their measured discrepancies.

## NASA profile and compliance revision

The [revision](revised/README.md) uses converted customary inertias, a new pair of STEP models derived from the measured profile traces, a position-dependent tooth compliance, and quasi-static torque balance. It saves the original and revised waveforms on the same grid. The revised high-torque curve has lower RMSE. The light-load contact-loss event remains unresolved.

## Reproduce

Set the OCCT runtime PATH as in the project README and run from the project root. The fetch step downloads only the three selected healthy records and checks them against Mendeley's SHA-256 metadata.

~~~powershell
cmake --build build/occt --config Release --target nc_gears
python benchmarks/gears/experimental/fetch.py build/gears-experimental/source
python benchmarks/gears/experimental/measurements.py build/gears-experimental/source results/gears/experimental/measurements build/gears-experimental/qa
build/occt/Release/nc_gears.exe models unsw results/gears/experimental/models
python benchmarks/gears/experimental/compare.py build/occt/Release/nc_gears.exe results/gears/experimental
python benchmarks/gears/experimental/plots.py results/gears/experimental
python benchmarks/gears/experimental/paper_section.py results/gears/experimental
python benchmarks/gears/paper.py results/gears
~~~
