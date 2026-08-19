#!/usr/bin/env python3
"""Render matched full molecules and functional support skeletons."""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import html
import io
import os
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

from forge.product.lipid_support_skeleton import (
    FUNCTIONAL_SUPPORT,
    encode_lipid_support_skeleton,
    skeleton_statistics,
)

REPO = Path(__file__).resolve().parents[1]


def _read_rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _png_image(
    molecule: Chem.Mol, *, highlights: dict[int, tuple[float, ...]] | None = None
) -> Image.Image:
    prepared = Chem.Mol(molecule)
    rdDepictor.Compute2DCoords(prepared)
    return Draw.MolToImage(
        prepared,
        size=(620, 320),
        highlightAtoms=sorted(highlights or {}),
        highlightAtomColors=highlights or {},
    )


def _png_data(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _induced_support(molecule: Chem.Mol, removed_atoms: tuple[int, ...]) -> Chem.Mol:
    editable = Chem.RWMol(molecule)
    for index in sorted(removed_atoms, reverse=True):
        editable.RemoveAtom(index)
    support = editable.GetMol()
    support.UpdatePropertyCache(strict=False)
    return support


def _render_card(
    structure_id: str, smiles: str, corpus: str
) -> tuple[str, tuple[Image.Image, Image.Image, Image.Image]]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise RuntimeError(f"invalid molecule selected for rendering: {structure_id}")
    encoding = encode_lipid_support_skeleton(
        molecule,
        structure_id=structure_id,
        variant=FUNCTIONAL_SUPPORT,
    )
    canonical = Chem.MolFromSmiles(encoding.canonical_smiles)
    statistics = skeleton_statistics(encoding)
    colors = {
        index: ((0.14, 0.62, 0.58) if index in encoding.retained_atoms else (0.91, 0.30, 0.24))
        for index in range(encoding.node_count)
    }
    support = _induced_support(canonical, encoding.removed_atoms)
    images = (
        _png_image(canonical),
        _png_image(canonical, highlights=colors),
        _png_image(support),
    )
    full_png, overlay_png, support_png = (_png_data(image) for image in images)
    metrics = (
        f"atoms {statistics['full_atoms']} -> {statistics['retained_atoms']}; "
        f"leaves {statistics['full_leaves']} -> {statistics['skeleton_leaves']}; "
        f"junctions {statistics['full_junctions']} -> {statistics['skeleton_junctions']}; "
        f"resolved apparent junctions {statistics['resolved_false_junctions']}"
    )
    card = f"""
    <section class="card">
      <h2>{html.escape(corpus)} · {html.escape(structure_id)}</h2>
      <p>{html.escape(metrics)}</p>
      <div class="row">
        <figure><img src="data:image/png;base64,{full_png}"><figcaption>Complete molecule</figcaption></figure>
        <figure><img src="data:image/png;base64,{overlay_png}"><figcaption>Support atoms teal; generated decorations red</figcaption></figure>
        <figure><img src="data:image/png;base64,{support_png}"><figcaption>Functional support skeleton</figcaption></figure>
      </div>
    </section>
    """
    return card, images


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--r0",
        type=Path,
        default=REPO / "results/m0_03/r0_constitutional.csv.gz",
    )
    parser.add_argument(
        "--ugi-l1",
        type=Path,
        default=REPO / "data/splits/phase1/ugi_l1_assignments.csv.gz",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=REPO / "results/phase1/lipid_support_skeleton_ledger.csv.gz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/phase1/lipid_support_skeleton_examples.html",
    )
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        default=REPO / "results/phase1/lipid_support_skeleton_examples.png",
    )
    parser.add_argument("--per-corpus", type=int, default=6)
    args = parser.parse_args()

    ledger = pd.read_csv(args.ledger)
    support = ledger[ledger["variant"] == FUNCTIONAL_SUPPORT].copy()
    cards = []
    image_rows = []

    r0_rows = {row["r0_structure_id"]: row for row in _read_rows(args.r0)}
    r0_candidates = support[
        (support["corpus"] == "R0")
        & (support["full_atoms"].between(35, 90))
        & (support["resolved_false_junctions"].between(2, 8))
    ].sort_values(["resolved_false_junctions", "full_atoms", "structure_id"])
    r0_indices = (
        pd.Series(range(len(r0_candidates)))
        .quantile([index / max(1, args.per_corpus - 1) for index in range(args.per_corpus)])
        .round()
        .astype(int)
        .drop_duplicates()
        .tolist()
    )
    for index in r0_indices:
        row = r0_candidates.iloc[index]
        source = r0_rows[str(row["structure_id"])]
        card, images = _render_card(
            str(row["structure_id"]),
            source["canonical_constitutional_smiles"],
            "Observed R0",
        )
        cards.append(card)
        image_rows.append(images)

    ugi_rows = {row["product_id"]: row for row in _read_rows(args.ugi_l1)}
    ugi_candidates = support[
        (support["corpus"] == "Ugi_L1") & (support["fold"] == "measured")
    ].sort_values(["full_atoms", "structure_id"])
    ugi_indices = (
        pd.Series(range(len(ugi_candidates)))
        .quantile([index / max(1, args.per_corpus - 1) for index in range(args.per_corpus)])
        .round()
        .astype(int)
        .drop_duplicates()
        .tolist()
    )
    for index in ugi_indices:
        row = ugi_candidates.iloc[index]
        source = ugi_rows[str(row["structure_id"])]
        card, images = _render_card(
            str(row["structure_id"]),
            source["canonical_product_smiles"],
            "Measured Ugi",
        )
        cards.append(card)
        image_rows.append(images)

    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>FORGE lipid support skeleton audit</title>
<style>
body {{ font: 15px system-ui; margin: 24px; color: #17212b; background: #f6f8fa; }}
h1 {{ margin-bottom: 4px; }} .note {{ max-width: 1000px; }}
.card {{ background: white; border: 1px solid #d7dde3; border-radius: 12px; margin: 20px 0; padding: 16px; }}
.row {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
figure {{ margin: 0; text-align: center; }} img {{ width: 100%; border: 1px solid #edf0f2; }}
figcaption {{ margin-top: 6px; font-weight: 600; }}
</style></head><body>
<h1>Matched real-lipid support skeletons</h1>
<p class="note">These are real observed or measured molecules, not generated samples. The support
skeleton retains carbon, nitrogen, phosphorus, rings, charged atoms and connector heteroatoms. Red
terminal decorations remain atom-level targets of the chemistry flow.</p>
{''.join(cards)}
</body></html>"""
    _atomic_write(args.output, document)
    sheet = Image.new("RGB", (620 * 3, 320 * len(image_rows)), "white")
    for row_index, images in enumerate(image_rows):
        for column_index, image in enumerate(images):
            sheet.paste(image, (620 * column_index, 320 * row_index))
    _atomic_write_image(args.contact_sheet, sheet)
    print(args.output)
    print(args.contact_sheet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
