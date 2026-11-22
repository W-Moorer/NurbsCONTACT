"""Build and publish the AES manuscript using the ICRL CAS layout."""

from pathlib import Path
import filecmp
import shutil
import subprocess


NAME = "Residual-Aware-CAD-Contact"


def main():
    source = Path(__file__).resolve().parent
    results = source.parents[1] / "results" / "aes"
    for report, figure in (("resident-cam", "accuracy-cost"), ("resident-reducer", "shaft-response")):
        pdf = results / report / f"{figure}.pdf"
        target = source / "figures" / f"{report}-{figure}.pdf"
        if pdf.is_file() and (not target.is_file() or not filecmp.cmp(pdf, target, shallow=False)):
            shutil.copyfile(pdf, target)
    build = source.parents[1] / "build" / "aes-paper"
    build.mkdir(parents=True, exist_ok=True)
    subprocess.run(["latexmk", "-xelatex", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
                    f"-outdir={build.as_posix()}", f"{NAME}.tex"], cwd=source, check=True)
    compiled = build / f"{NAME}.pdf"
    for target in (source / compiled.name, source.parent / "manuscript.pdf"):
        shutil.copyfile(compiled, target)
        print(target)


if __name__ == "__main__":
    main()
