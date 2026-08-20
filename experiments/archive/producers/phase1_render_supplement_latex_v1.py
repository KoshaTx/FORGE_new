#!/usr/bin/env python3
"""Render the generated supplement (Tables S1--S16) into LaTeX the ICLR paper can \\input.

The supplement body is produced by phase1_build_manuscript_supplement_v1.py as markdown, because
the Nature Biotechnology draft is a markdown document. The ICLR paper is authored in LaTeX
directly, so the body has to be converted. This script owns that conversion so it is reproducible
rather than a sequence of shell steps nobody can repeat:

    build() -> markdown -> pandoc -> nest headings -> make SMILES breakable -> generated_supplement.tex

Three transformations are applied after pandoc, each for a stated reason.

1.  Headings are demoted one level. The generator emits each table as a top-level heading, which
    pandoc maps to \\section; inside the appendix that would give every table its own appendix
    letter. Demoting nests all sixteen under one section.

2.  Long SMILES inside \\texttt{} get explicit break opportunities. \\ttfamily disables
    hyphenation, so a 30-character SMILES in a narrow table column overflows the margin with no
    place to break; without this the supplement produces over two hundred overfull boxes, which
    are margin violations at submission. seqsplit would do this but is absent from this TeX
    install, so the breaks are inserted at conversion time instead. \\allowbreak is used rather
    than \\- so no hyphen is introduced: a hyphen inside a SMILES would read as a bond.

3.  Nothing else. The generated content is not edited. If a number is wrong, fix the generator or
    the artifact it reads, never this output.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
GENERATOR = "scripts/phase1_build_manuscript_supplement_v1.py"
# Break after this many characters of an unbroken run; short enough for a narrow table column.
RUN = 1


def load_generator(repo: Path):
    spec = importlib.util.spec_from_file_location("supplement", repo / GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def latex_chars(body: str) -> list[str]:
    """Split a LaTeX token run into printable units.

    Three unit kinds, because pandoc's escaping produces all three inside a SMILES:
    an escape such as ``\\#``, a balanced brace group such as ``{[}``, and a bare character.
    Treating ``{[}`` as one unit is what lets bracketed atoms be broken around rather than
    skipped, which is where most of the remaining margin violations lived.
    """
    out, i = [], 0
    while i < len(body):
        if body[i] == "\\" and i + 1 < len(body):
            out.append(body[i:i + 2])
            i += 2
        elif body[i] == "{":
            depth, j = 1, i + 1
            while j < len(body) and depth:
                depth += (body[j] == "{") - (body[j] == "}")
                j += 1
            if depth:
                return []                     # unbalanced; caller leaves the run untouched
            out.append(body[i:j])
            i = j
        else:
            out.append(body[i])
            i += 1
    return out


def find_texttt(tex: str) -> list[tuple[int, int, str]]:
    """Locate every \\texttt{...} with balanced-brace scanning, which a regex cannot do."""
    spans, needle = [], r"\texttt{"
    start = tex.find(needle)
    while start != -1:
        depth, j = 1, start + len(needle)
        while j < len(tex) and depth:
            depth += (tex[j] == "{") - (tex[j] == "}")
            j += 1
        if not depth:
            spans.append((start, j, tex[start + len(needle):j - 1]))
        start = tex.find(needle, j if not depth else start + len(needle))
    return spans


def breakable_texttt(tex: str) -> tuple[str, int]:
    """Insert \\allowbreak inside every long \\texttt{...} so SMILES wrap in a table cell."""
    count, out, cursor = 0, [], 0
    for start, end, inner in find_texttt(tex):
        units = latex_chars(inner)
        if len(units) <= 12:                  # short enough to fit; no need to fragment it
            continue
        pieces = []
        for index, unit in enumerate(units):
            pieces.append(unit)
            if (index + 1) % RUN == 0 and index + 1 < len(units):
                pieces.append(r"\allowbreak{}")
        out.append(tex[cursor:start])
        out.append(r"\texttt{" + "".join(pieces) + "}")
        cursor = end
        count += 1
    out.append(tex[cursor:])
    return "".join(out), count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "paper/generated_supplement.tex")
    args = parser.parse_args()

    module = load_generator(REPO)
    body = module.build(REPO)

    # The generator's own document title would become a stray heading inside the appendix.
    body = re.sub(r"^# Supplementary Information.*$", "", body, flags=re.M)
    # Promote "## Table Sn" so pandoc maps it to \section, then demote below; and drop the
    # role-group headings one level with it.
    body = re.sub(r"^## (Table S)", r"# \1", body, flags=re.M)
    body = re.sub(r"^### ", "## ", body, flags=re.M)

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "supplement.md"
        source.write_text(body)
        try:
            tex = subprocess.run(
                ["pandoc", str(source), "-f", "markdown", "-t", "latex", "--wrap=preserve"],
                check=True, capture_output=True, text=True).stdout
        except FileNotFoundError:
            sys.exit("pandoc is required to render the supplement to LaTeX")
        except subprocess.CalledProcessError as error:
            sys.exit(f"pandoc failed: {error.stderr[:400]}")

    tex = tex.replace(r"\subsection{", r"\subsubsection{").replace(r"\section{", r"\subsection{")
    tex, broken = breakable_texttt(tex)

    header = "% autogenerated supplement tables - regenerate these, dont edit by hand\n"
    args.output.write_text(header + tex)

    tables = len(re.findall(r"\\subsection\{Table S", tex))
    print(f"wrote {args.output.relative_to(REPO)}")
    print(f"{tables} tables, {tex.count(chr(92) + 'begin{longtable}')} longtables, "
          f"{broken} texttt runs made breakable")


if __name__ == "__main__":
    main()
