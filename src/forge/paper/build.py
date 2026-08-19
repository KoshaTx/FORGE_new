"""Isolated, deterministic paper build and Overleaf bundle utilities."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_file
from forge.core.io import atomic_write
from forge.paper.contract import PaperContract

_DOC = "FORGE_ICLR2027_paper"
_BUILD_SUFFIXES = {".aux", ".blg", ".fdb_latexmk", ".fls", ".log", ".out", ".pdf"}


def _copy_sources(repo: Path, destination: Path, contract: PaperContract) -> None:
    paper = repo / "paper"
    destination.mkdir(parents=True, exist_ok=True)
    for source in sorted(paper.iterdir()):
        if source.is_dir():
            continue
        if source.suffix in _BUILD_SUFFIXES:
            continue
        shutil.copy2(source, destination / source.name)
    for pin in contract.figure_outputs:
        relative = Path(pin.path).relative_to("paper")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / pin.path, target)


def _latex_environment() -> dict[str, str]:
    environment = dict(os.environ)
    # The submission bundle carries styles at its root. Clearing overrides verifies that the build
    # does not accidentally depend on a caller's private TEXINPUTS configuration.
    environment.pop("TEXINPUTS", None)
    environment.pop("BIBINPUTS", None)
    environment.pop("BSTINPUTS", None)
    # pdfTeX otherwise embeds wall-clock creation/modification dates, so identical sources produce
    # different bytes. SOURCE_DATE_EPOCH is the standard TeX Live reproducible-build interface.
    environment["SOURCE_DATE_EPOCH"] = "0"
    environment["FORCE_SOURCE_DATE"] = "1"
    environment["TZ"] = "UTC"
    return environment


def _run_latex(work: Path) -> None:
    environment = _latex_environment()
    if shutil.which("latexmk"):
        command = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"{_DOC}.tex"]
        completed = subprocess.run(
            command, cwd=work, env=environment, capture_output=True, text=True, check=False
        )
        if completed.returncode:
            raise RuntimeError(f"paper build failed:\n{completed.stdout[-2000:]}\n{completed.stderr[-1000:]}")
        return
    if not shutil.which("pdflatex"):
        raise RuntimeError("paper build requires latexmk or pdflatex")
    command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{_DOC}.tex"]
    for _ in range(2):
        completed = subprocess.run(
            command, cwd=work, env=environment, capture_output=True, text=True, check=False
        )
        if completed.returncode:
            raise RuntimeError(
                f"paper build failed:\n{completed.stdout[-2000:]}\n{completed.stderr[-1000:]}"
            )


def build_pdf(repo: Path, contract_path: Path, output: Path) -> dict[str, Any]:
    """Compile from a clean staging directory and atomically publish only the PDF."""

    contract = PaperContract.load(contract_path)
    with tempfile.TemporaryDirectory(prefix="forge-paper-") as temporary:
        work = Path(temporary)
        _copy_sources(repo, work, contract)
        _run_latex(work)
        pdf = work / f"{_DOC}.pdf"
        if not pdf.is_file():
            raise RuntimeError("LaTeX completed without producing a PDF")
        atomic_write(output, pdf.read_bytes())
    return {
        "bytes": output.stat().st_size,
        "output": str(output),
        "schema_version": "forge.paper_build.v1",
        "sha256": str(sha256_file(output)),
    }


def _referenced_figures(source: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                match.group(1)
                for match in re.finditer(
                    r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source
                )
            }
        )
    )


def build_overleaf_bundle(
    repo: Path,
    contract_path: Path,
    output: Path,
    *,
    verify_compile: bool = True,
) -> dict[str, Any]:
    """Create a byte-reproducible zip containing only paper source and referenced figures."""

    contract = PaperContract.load(contract_path)
    with tempfile.TemporaryDirectory(prefix="forge-overleaf-") as temporary:
        stage = Path(temporary)
        _copy_sources(repo, stage, contract)
        wanted = _referenced_figures((stage / f"{_DOC}.tex").read_text())
        staged_figures = {
            path.relative_to(stage / "figures").as_posix()
            for path in (stage / "figures").rglob("*")
            if path.is_file()
        }
        if staged_figures != set(wanted):
            raise RuntimeError(
                "bundle figure set differs from LaTeX references: "
                f"missing={sorted(set(wanted) - staged_figures)}, "
                f"extra={sorted(staged_figures - set(wanted))}"
            )
        if verify_compile:
            _run_latex(stage)

        members = [path for path in stage.rglob("*") if path.is_file()]
        members = [path for path in members if path.suffix not in _BUILD_SUFFIXES]
        with tempfile.NamedTemporaryFile(prefix="forge-overleaf-", suffix=".zip", delete=False) as handle:
            temporary_zip = Path(handle.name)
        try:
            with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(members, key=lambda item: item.relative_to(stage).as_posix()):
                    relative = path.relative_to(stage).as_posix()
                    info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o100644 << 16
                    archive.writestr(info, path.read_bytes())
            atomic_write(output, temporary_zip.read_bytes())
        finally:
            temporary_zip.unlink(missing_ok=True)
    return {
        "files": len(members),
        "output": str(output),
        "schema_version": "forge.paper_bundle.v1",
        "sha256": str(sha256_file(output)),
        "verified_compile": verify_compile,
    }


__all__ = ["build_overleaf_bundle", "build_pdf"]
