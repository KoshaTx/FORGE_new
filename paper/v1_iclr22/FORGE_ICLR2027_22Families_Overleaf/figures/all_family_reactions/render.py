"""Render the qualified registry transformations as compact vector reaction panels."""

from __future__ import annotations

import hashlib
import html
import json
import math
import subprocess
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from schematic import schematic_reaction

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = ROOT / "results/phase1/compose_lipid_iclr22_reaction_figures_v1/reaction_records.json"
GROUPS = [
    (
        "multicomponent_a",
        "Multicomponent assemblies I",
        ["a3_amine_aldehyde_alkyne", "aldehyde_ugi4"],
    ),
    (
        "ketone_ugi4",
        "Ketone Ugi four-component assembly",
        ["ketone_ugi4"],
    ),
    (
        "multicomponent_b",
        "Multicomponent assemblies II",
        ["ketone_isocyanide_amide", "alpha_isocyanoester_dihydroimidazole", "passerini_3cr"],
    ),
    (
        "conjugate_additions",
        "Conjugate additions",
        ["aza_michael_acrylamide", "disulfide_michael", "maleate_addition"],
    ),
    (
        "functionalization",
        "Nitrogen and oxygen functionalization",
        [
            "aryl_reductive_amination",
            "amine_alkylation",
            "amine_epoxide_opening",
        ],
    ),
    (
        "oxygen_phosphate",
        "Esterification and phosphate opening",
        [
            "o_esterification",
            "iphos_ring_opening",
        ],
    ),
    (
        "ordered_assemblies_a",
        "Ordered assemblies I",
        ["acid_epoxide_diester_multistep", "aema_aza_thiol_addition"],
    ),
    ("ordered_assemblies_b", "Ordered assemblies II", ["epoxide_opening_o_acylation"]),
    (
        "ordered_assemblies_c",
        "Ordered assemblies III",
        ["preassembled_thiol_yne_tail_amidation"],
    ),
    (
        "ordered_assemblies_d",
        "Ordered assemblies IV",
        ["thiolactone_aminolysis_michael"],
    ),
]

NAMES = {
    "a3_amine_aldehyde_alkyne": "A3 amine–aldehyde–alkyne",
    "aldehyde_ugi4": "Aldehyde Ugi four-component assembly",
    "ketone_ugi4": "Ketone Ugi four-component assembly",
    "ketone_isocyanide_amide": "Ketone–isocyanide aminoamide",
    "alpha_isocyanoester_dihydroimidazole": "α-Isocyanoester dihydroimidazole",
    "passerini_3cr": "Passerini three-component assembly",
    "aza_michael_acrylamide": "Aza-Michael addition to acrylamide",
    "disulfide_michael": "Disulfide Michael addition",
    "maleate_addition": "Maleate addition",
    "aryl_reductive_amination": "Aryl reductive amination",
    "amine_alkylation": "Amine alkylation",
    "amine_epoxide_opening": "Amine–epoxide opening",
    "o_esterification": "O-Esterification",
    "iphos_ring_opening": "iPhos ring opening",
    "acid_epoxide_diester_multistep": "Acid–epoxide opening, then esterification",
    "aema_aza_thiol_addition": "AEMA: aza addition, then thiol addition",
    "epoxide_opening_o_acylation": "Amine–epoxide opening, then O-acylation",
    "preassembled_thiol_yne_tail_amidation": "Thiol–yne tail assembly, then amidation",
    "thiolactone_aminolysis_michael": "Thiolactone aminolysis, then thiol-Michael addition",
}


def pin(path):
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def svg_body(svg):
    return svg[svg.index(">", svg.index("<svg")) + 1 : svg.rindex("</svg>")]


def molecule_drawing(molecule):
    molecule = Chem.Mol(molecule)
    rdDepictor.Compute2DCoords(molecule, forceRDKit=False)
    positions = molecule.GetConformer().GetPositions()
    if molecule.GetNumAtoms() > 12 and (positions[:, 1].max() - positions[:, 1].min()) > 1.3 * (
        positions[:, 0].max() - positions[:, 0].min()
    ):
        conformer = molecule.GetConformer()
        for index, (x, y, z) in enumerate(positions):
            conformer.SetAtomPosition(index, (-float(y), float(x), float(z)))
        positions = conformer.GetPositions()
    width = max(90, math.ceil(float(positions[:, 0].max() - positions[:, 0].min()) * 30 + 85))
    height = max(100, math.ceil(float(positions[:, 1].max() - positions[:, 1].min()) * 30 + 85))
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.useBWAtomPalette()
    options.fixedFontSize = 20
    options.fixedBondLength = 30
    options.bondLineWidth = 1.6
    options.padding = 0.05
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    return svg_body(drawer.GetDrawingText()), width, height


def reaction_svg(step, width=1000):
    reaction, audit = schematic_reaction(step)
    drawings = [molecule_drawing(m) for m in reaction.GetReactants()]
    product = molecule_drawing(reaction.GetProductTemplate(0))
    gap = 38
    reactant_width = sum(item[1] for item in drawings) + gap * (len(drawings) - 1)
    reactant_height = max(item[2] for item in drawings)
    total_width = reactant_width + 90 + product[1]
    horizontal = total_width <= width
    canvas_width = max(width, reactant_width, product[1])
    total_height = (
        max(reactant_height, product[2]) if horizontal else reactant_height + 60 + product[2]
    )
    x = (canvas_width - (total_width if horizontal else reactant_width)) / 2
    pieces = []
    for index, (svg, w, h) in enumerate(drawings):
        y = ((total_height if horizontal else reactant_height) - h) / 2
        pieces.append(f'<g transform="translate({x},{y})">{svg}</g>')
        x += w
        if index + 1 < len(drawings):
            cy = (total_height if horizontal else reactant_height) / 2
            pieces.append(
                f'<text x="{x+gap/2}" y="{cy+7}" text-anchor="middle" font-family="Arial" font-size="27">+</text>'
            )
            x += gap
    if horizontal:
        cy = total_height / 2
        pieces.append(
            f'<path d="M {x+10},{cy} L {x+80},{cy} M {x+70},{cy-6} L {x+80},{cy} L {x+70},{cy+6}" fill="none" stroke="black" stroke-width="1.8"/>'
        )
        px, py = x + 90, (total_height - product[2]) / 2
    else:
        cx, cy = canvas_width / 2, reactant_height + 8
        pieces.append(
            f'<path d="M {cx},{cy} L {cx},{cy+40} M {cx-6},{cy+30} L {cx},{cy+40} L {cx+6},{cy+30}" fill="none" stroke="black" stroke-width="1.8"/>'
        )
        px, py = (canvas_width - product[1]) / 2, reactant_height + 60
    pieces.append(f'<g transform="translate({px},{py})">{product[0]}</g>')
    scale = width / canvas_width
    audit["layout"] = "horizontal" if horizontal else "reactants_above_product"
    audit["drawing_font_size_at_432pt_width"] = 20 * scale * 0.432
    return (
        f'<g transform="scale({scale})">' + "".join(pieces) + "</g>",
        audit,
        math.ceil(total_height * scale),
    )


def main():
    rdDepictor.SetPreferCoordGen(True)
    document = json.loads(DATA.read_text())
    records = {record["family_id"]: record for record in document["records"]}
    output = []
    audits = []
    review = json.loads((DATA.parent / "chemistry_review.json").read_text())
    for key, title, families in GROUPS:
        y = 10
        parts = []
        represented = []
        for number, family in enumerate(families):
            record = records[family]
            steps = []
            for step in record["steps"]:
                key_id = (step["stage_index"], step["reaction_id"])
                if steps and steps[-1]["key"] == key_id:
                    steps[-1]["count"] += 1
                else:
                    steps.append({"key": key_id, "count": 1, "step": step})
            parts.append(
                f'<text x="8" y="{y+24}" font-family="Arial,Helvetica,sans-serif" '
                f'font-size="25" font-weight="bold">{chr(97+number)}  {html.escape(NAMES[family])}</text>'
            )
            y += 38
            for step_number, grouped in enumerate(steps, 1):
                step = grouped["step"]
                labels = review["families"][family]["suggested_arrow_labels"]
                label = labels[min(step_number - 1, len(labels) - 1)]
                # Panel text names the transformation; precise source conditions stay in the records.
                label = label.split(" (")[0]
                if family == "aryl_reductive_amination":
                    label = "Condensation, then reduction"
                if family == "maleate_addition":
                    label = "Aza-Michael addition (N-addition branch)"
                if family == "iphos_ring_opening":
                    label += "; deprotonated product"
                if grouped["count"] > 1:
                    label += f"; {grouped['count']} incorporated events, first shown"
                parts.append(
                    f'<text x="8" y="{y+17}" font-family="Arial,Helvetica,sans-serif" '
                    f'font-size="19">{html.escape(label)}</text>'
                )
                y += 24
                drawing, audit, drawing_height = reaction_svg(step)
                audit.update(
                    family=family,
                    displayed_step=step["step_index"],
                    represented_events=grouped["count"],
                )
                audits.append(audit)
                parts.append(f'<g transform="translate(0,{y})">{drawing}</g>')
                y += drawing_height
            y += 15
            represented.append(
                {
                    "family": family,
                    "registered_events": len(record["steps"]),
                    "displayed_transformations": len(steps),
                    "source_target_id": record["source_target_id"],
                }
            )
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:rdkit="http://www.rdkit.org/xml" '
            f'width="432pt" height="{y*0.432:.3f}pt" viewBox="0 0 1000 {y}">'
            f'<rect width="1000" height="{y}" fill="white"/>' + "\n".join(parts) + "</svg>\n"
        )
        path = HERE / f"{key}.svg"
        path.write_text(svg)
        subprocess.run(
            [
                "rsvg-convert",
                "--format",
                "pdf",
                "--output",
                str(path.with_suffix(".pdf")),
                str(path),
            ],
            check=True,
        )
        subprocess.run(
            [
                "rsvg-convert",
                "--width",
                "1500",
                "--output",
                str(path.with_suffix(".png")),
                str(path),
            ],
            check=True,
        )
        output.append(
            {
                "title": title,
                "families": represented,
                "svg": pin(path),
                "pdf": pin(path.with_suffix(".pdf")),
            }
        )
    receipt = {
        "schema_version": "forge.iclr22.reaction_figure_render.v1",
        "inputs": [
            pin(DATA),
            pin(Path(__file__)),
            pin(HERE / "schematic.py"),
            pin(DATA.parent / "chemistry_review.json"),
        ],
        "rdkit_version": rdBase.rdkitVersion,
        "figures": output,
        "schematic_audits": audits,
        "existing_figure_families": [
            "aldehyde_ugi3",
            "aza_michael_acrylate",
            "reductive_amination",
        ],
        "semantics": "Exact replayed source structures with step-local unchanged-substituent abbreviations. Repeated events are identified; no experimental outcomes are inferred.",
        "visual_review_receipt": "paper/v1_iclr22/visual_review.json",
    }
    (HERE / "render.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"groups": len(output), "families": sum(len(x["families"]) for x in output)}))


if __name__ == "__main__":
    main()
