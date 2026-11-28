# Residual-Controlled CAD Contact Simulation of Rigid Mechanisms

[Manuscript PDF](Residual-Aware-CAD-Contact.pdf) · [LaTeX source](Residual-Aware-CAD-Contact.tex) · [Benchmark protocol](../../benchmarks/aes/protocol.md) · [Measured results](../../results/aes/)

The source follows the Elsevier CAS single-column layout of the ICRL `CoFlexSim-AES` manuscript. Add the confirmed author order, affiliations, and corresponding author before preparing the submission metadata.

## Review the manuscript argument

Open [Manuscript-Argument.papergraph](Manuscript-Argument.papergraph) with File → Open in [Papergraph 0.16.0-preview.8](https://github.com/wangyuer1-lang/papergraph/releases/tag/v0.16.0-preview.8). On Windows, extract `Papergraph-0.16.0-preview.8-windows-x64.zip` and run `papergraph.exe` from the extracted folder. Use **F** on the canvas to fit the board and **Shift+Space** to show proposition titles.

The middle column follows the introduction's contribution order. The left and right columns connect each claim to its evidence and scope. Select a proposition and open Notes **2** for the LaTeX source path and label or section name. Open Notes **3** on a middle-column proposition for its revision notes. Use **Full text** to review the argument in order with the linked evidence and qualifications shown in parentheses.

Revise submission prose, citations, and cross-references in the [LaTeX source](Residual-Aware-CAD-Contact.tex) and its included sections. When a revision changes a claim, its evidence, or its scope, update the corresponding proposition and source note in the graph. The graph can be edited in Papergraph or read through its [local agent interface](https://github.com/wangyuer1-lang/papergraph/blob/4ff5962/Agent%20guide.md) while the app is open.

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
