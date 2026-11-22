# Residual-Controlled CAD Contact Simulation of Rigid Mechanisms

[Manuscript PDF](Residual-Aware-CAD-Contact.pdf) · [LaTeX source](Residual-Aware-CAD-Contact.tex) · [Benchmark protocol](../../benchmarks/aes/protocol.md) · [Measured results](../../results/aes/)

The source follows the Elsevier CAS single-column layout of the ICRL `CoFlexSim-AES` manuscript. Add the confirmed author order, affiliations, and corresponding author before preparing the submission metadata.

## Build

From the repository root, run `python benchmarks/hht/paper.py results/hht` with XeLaTeX, BibTeX, and `latexmk` on `PATH`. The build uses the figures here and publishes the same PDF in this directory and at `docs/manuscript.pdf`. The template uses Times New Roman, Arial, and Consolas and the TeX Live STIX Two Math font.

The sampled-contact section uses the [cam accuracy and representation experiments](../../results/aes/resident-cam/accuracy-study.md) and the [shared-shaft reducer comparison](../../results/aes/resident-reducer/README.md). Both reports include the Python commands used to reproduce the response figures.

## Reproduce the figures

Run the [numerical protocol](../../benchmarks/aes/protocol.md) first. The published-case figure script uses the traces in `build/aes-literature` and a Blender 5.1 executable on `PATH`.

```powershell
python benchmarks/aes/literature_figures.py build/aes-literature docs/AES/figures (Get-Command blender).Source
```

The industrial and appendix figure script uses the cam query comparison in `build/aes-cam-baselines` and the recorded gear and fine-step sphere traces in `build/aes-benchmark`.

Prepare the model renders by tessellating each gear solid named in `results/gears/models/spur.csv` and `reducer.csv` with `python benchmarks/gears/tessellate.py STEP_FILE STEM.npz`. Put the resulting files in `build/aes-benchmark/paper/step-meshes`. Populate `build/aes-benchmark/paper/frames` with three completed traces named `cam.csv`, `spur.csv`, and `reducer.csv`. The published inputs were `cam-native-24-125us.csv`, `spur-native-0nm-10us-31.csv`, and `reducer-native-0nm-10us-31.csv`.

```powershell
$work = 'build/aes-benchmark'
blender -b --factory-startup --python-exit-code 1 --python benchmarks/aes/blender_models.py -- results/gears/models "$work/paper/step-meshes" "$work/paper/frames" "$work/paper/renders"
python benchmarks/aes/paper_diagrams.py docs/AES/figures
python benchmarks/aes/paper_responses.py $work docs/AES/figures
```

The render script uses Blender 5.1 through Python. The model panels show the nominal geometry at representative poses. Display backings on planar surfaces give the plates a visible edge. The response plots use recorded solver forces and motion histories.
