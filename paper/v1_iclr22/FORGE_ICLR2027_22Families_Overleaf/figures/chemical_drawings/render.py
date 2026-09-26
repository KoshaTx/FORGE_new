"""Redraw the frozen manuscript examples as monochrome skeletal vector graphics.

Run with the repository environment: .venv/bin/python paper/v1_iclr/figures/chemical_drawings/render.py
The saved selection receipts determine every identity and display order. No reselection,
reaction inference, conformer optimization, or molecular standardization is performed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

ROOT = Path(__file__).resolve().parent
SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)
BOND_COLOR = "#727272"
ATOM_COLOR = "#303030"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canvas(width: int, height: int) -> ET.Element:
    root = ET.Element(
        f"{{{SVG}}}svg", width=str(width), height=str(height), viewBox=f"0 0 {width} {height}"
    )
    ET.SubElement(root, f"{{{SVG}}}rect", width="100%", height="100%", fill="white")
    return root


def label(root: ET.Element, x: int, y: int, text: str, size: int = 31) -> None:
    node = ET.SubElement(
        root,
        f"{{{SVG}}}text",
        x=str(x),
        y=str(y),
        fill=ATOM_COLOR,
        attrib={"font-family": "Times New Roman, serif", "font-size": str(size)},
    )
    node.text = text


def molecule(
    smiles: str,
    width: int,
    height: int,
    font_size: int,
    *,
    fit_labels: bool = False,
    bond_width: float = 2.2,
) -> ET.Element:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid saved molecular structure: {smiles}")
    original = Chem.MolToSmiles(mol, isomericSmiles=True)
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.padding = 0.04
    options.bondLineWidth = bond_width
    if fit_labels:
        options.minFontSize = 12
        options.maxFontSize = font_size
    else:
        options.fixedFontSize = font_size
    options.multipleBondOffset = 0.14
    options.useBWAtomPalette()
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    if Chem.MolToSmiles(mol, isomericSmiles=True) != original:
        raise ValueError("Depiction changed molecular identity")
    root = ET.fromstring(drawer.GetDrawingText())
    for node in root.iter():
        cls = node.get("class", "")
        color = ATOM_COLOR if cls.startswith("atom-") else BOND_COLOR
        if node.tag.endswith("}path"):
            for key in ("style", "fill", "stroke"):
                if key in node.attrib:
                    node.set(key, node.get(key, "").replace("#000000", color))
    return root


def place(root: ET.Element, drawing: ET.Element, x: int, y: int) -> None:
    drawing.set("x", str(x))
    drawing.set("y", str(y))
    root.append(drawing)


def save(root: ET.Element, name: str) -> dict[str, str]:
    svg = ROOT / f"{name}.svg"
    svg.write_bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True))
    pdf = svg.with_suffix(".pdf")
    subprocess.run(["rsvg-convert", "--format=pdf", "--output", str(pdf), str(svg)], check=True)
    return {svg.name: sha(svg), pdf.name: sha(pdf)}


def precursors(components: dict[str, str]) -> ET.Element:
    root = canvas(950, 560)
    for role, title, x, y, width, height in (
        ("amine_head", "Amine", 10, 0, 450, 270),
        ("isocyanide_tail", "Isocyanide", 485, 0, 450, 270),
        ("oxoester_aldehyde_body_tail", "Aldehyde", 10, 280, 925, 270),
    ):
        label(root, x + 8, y + 34, title)
        place(root, molecule(components[role], width, height - 55, 27, fit_labels=True), x, y + 50)
    return root


def main() -> None:
    sample_path = ROOT / "selected_samples.json"
    atlas_path = ROOT / "atlas_samples.json"
    samples = json.loads(sample_path.read_text())["selected"]
    atlas = json.loads(atlas_path.read_text())["selected"]
    if len(samples) != 3 or len(atlas) != 12:
        raise ValueError("The frozen display selection changed")
    outputs = {}
    identities = []
    for index, sample in enumerate(samples[:2], 1):
        root = canvas(900, 560)
        label(root, 18, 44, f"FORGE-Ugi-{sample['attempt_index']:04d}", 34)
        place(root, molecule(sample["canonical_smiles"], 880, 470, 32), 10, 80)
        outputs.update(save(root, f"sample_{index:02d}_product"))

        components = sample["components_by_role"]
        for role in components:
            identities.append({"sample": index, "role": role, "smiles": components[role]})
        outputs.update(save(precursors(components), f"sample_{index:02d}_precursors"))
    for sample in atlas:
        index = sample["display_diversity_rank"]
        root = molecule(sample["canonical_smiles"], 1800, 680, 48, bond_width=4.4)
        outputs.update(save(root, f"atlas_{index:02d}_2d"))
        identities.append({"atlas_rank": index, "smiles": sample["canonical_smiles"]})
        components = sample["components_by_role"]
        outputs.update(save(precursors(components), f"atlas_{index:02d}_precursors"))
        for role in components:
            identities.append({"atlas_rank": index, "role": role, "smiles": components[role]})
    receipt = {
        "scope": "Presentation only; saved molecular identities and display order retained.",
        "inputs": {p.name: sha(p) for p in (sample_path, atlas_path)},
        "renderer_sha256": sha(Path(__file__)),
        "rdkit_version": rdBase.rdkitVersion,
        "svg_converter": subprocess.check_output(["rsvg-convert", "--version"], text=True).strip(),
        "style": {
            "bonds": BOND_COLOR,
            "atom_labels": ATOM_COLOR,
            "layout": "deterministic RDKit 2D coordinates",
            "vector_outputs": True,
        },
        "identities": identities,
        "outputs": outputs,
    }
    (ROOT / "rendering_record.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Rendered {len(outputs) // 2} vector panels; all parsed identities preserved.")


if __name__ == "__main__":
    main()
