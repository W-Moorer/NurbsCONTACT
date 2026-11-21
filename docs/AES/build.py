"""Build and publish the AES manuscript using the ICRL CAS layout."""

from pathlib import Path
import shutil
import subprocess


NAME = "Residual-Aware-CAD-Contact"


def main():
    source = Path(__file__).resolve().parent
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
