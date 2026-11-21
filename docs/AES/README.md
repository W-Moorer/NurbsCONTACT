# Residual-Aware CAD Contact for Rigid Mechanisms with Certified Clearance Events

[Manuscript PDF](Residual-Aware-CAD-Contact.pdf) · [LaTeX source](Residual-Aware-CAD-Contact.tex) · [Benchmark protocol](../../benchmarks/aes/protocol.md) · [Measured results](../../results/aes/)

The source follows the Elsevier CAS single-column layout of the ICRL `CoFlexSim-AES` manuscript. Add the confirmed author order, affiliations, and corresponding author before preparing the submission metadata.

## Build

From the repository root, run `python benchmarks/hht/paper.py results/hht` with XeLaTeX, BibTeX, and `latexmk` on `PATH`. The build uses the figures here and publishes the same PDF in this directory and at `docs/manuscript.pdf`. The template uses Times New Roman, Arial, and Consolas and the TeX Live STIX Two Math font.

## Reproduce the figures

Run the [numerical protocol](../../benchmarks/aes/protocol.md) first. The figure script uses the recorded fine-step sphere traces, the diagonal trim audit, and the selected industrial runs named in its `industrial` function. The run folders used for the published figures were under `build/aes-benchmark`.

Prepare the model renders by tessellating each gear solid named in `results/gears/models/spur.csv` and `reducer.csv` with `python benchmarks/gears/tessellate.py STEP_FILE STEM.npz`. Put the resulting files in `build/aes-benchmark/paper/step-meshes`. Populate `build/aes-benchmark/paper/frames` with three completed traces named `cam.csv`, `spur.csv`, and `reducer.csv`. The published inputs were `cam-native-24-125us.csv`, `spur-native-0nm-10us-31.csv`, and `reducer-native-0nm-10us-31.csv`.

```powershell
$work = 'build/aes-benchmark'
blender -b --factory-startup --python-exit-code 1 --python benchmarks/aes/blender_models.py -- results/gears/models "$work/paper/step-meshes" "$work/paper/frames" "$work/paper/renders"
python benchmarks/aes/paper_diagrams.py docs/AES/figures
python benchmarks/aes/paper_responses.py $work docs/AES/figures
```

The render script uses Blender 5.1 through Python. The model panels show the nominal geometry at representative poses. Display backings on planar surfaces give the plates a visible edge. The response plots use recorded solver forces and motion histories.
