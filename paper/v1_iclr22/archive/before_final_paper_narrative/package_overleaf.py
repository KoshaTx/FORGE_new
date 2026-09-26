"""Build the portable 22-family draft source package and record exact input pins."""

import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPORT = HERE / "FORGE_ICLR2027_22Families_Overleaf"


def pin(path):
    return dict(
        path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )


def main():
    EXPORT.mkdir(exist_ok=True)
    mapping = {HERE / "FORGE_ICLR2027_paper.tex": EXPORT / "main.tex"}
    for name in (
        "forge.bib",
        "references.bib",
        "iclr2027_conference.sty",
        "iclr2027_conference.bst",
        "iclr2027_conference.bib",
        "fancyhdr.sty",
        "natbib.sty",
        "math_commands.tex",
        "mathematical_review.json",
        "RESULTS_LEDGER.json",
        "QUALITY_METRICS.json",
        "RESULTS_CHECKLIST.md",
    ):
        if (HERE / name).exists():
            mapping[HERE / name] = EXPORT / name
    for directory in ("sections", "generated", "figures"):
        for source in sorted((HERE / directory).rglob("*")):
            if source.is_file() and source.suffix.lower() in {
                ".tex",
                ".pdf",
                ".png",
                ".jpg",
                ".svg",
                ".json",
                ".csv",
            }:
                mapping[source] = EXPORT / source.relative_to(HERE)
    for source, destination in mapping.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (EXPORT / "README.md").write_text(
        "# FORGE, 22-family working draft\n\n"
        "Compile main.tex with pdfLaTeX (or latexmk -pdf main.tex).\n"
        "New results are explicitly pending. Original three-family computational results remain labeled historical.\n"
        "Use RESULTS_CHECKLIST.md and RESULTS_LEDGER.json to populate verified results.\n"
        "No shell escape, external figure generation or experimental images are required.\n"
    )
    receipt = dict(
        schema_version="forge.iclr22.overleaf_package.v1",
        status="Working draft with placeholders; no final new-model measurements",
        implementation=pin(Path(__file__)),
        inputs=[pin(source) for source in sorted(mapping)],
        files={
            str(destination.relative_to(EXPORT)): hashlib.sha256(
                destination.read_bytes()
            ).hexdigest()
            for destination in sorted(mapping.values())
        },
    )
    (EXPORT / "package_provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    archive = HERE / "FORGE_ICLR2027_22Families_Overleaf.zip"
    files = [*mapping.values(), EXPORT / "README.md", EXPORT / "package_provenance.json"]
    with ZipFile(archive, "w", ZIP_DEFLATED) as zipped:
        for path in sorted(files):
            zipped.write(path, path.relative_to(EXPORT))
    (HERE / "overleaf_package.json").write_text(
        json.dumps(
            dict(
                package=pin(archive),
                files=len(files),
                provenance=pin(EXPORT / "package_provenance.json"),
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(dict(package=pin(archive), files=len(files))))


if __name__ == "__main__":
    main()
