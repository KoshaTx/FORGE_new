"""Render a local diagnostic gallery from the complete, unchanged attempt ledger."""

import hashlib
import html
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

REPO = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
RESULT = REPO / "results/phase1/combinatorial_generation_v1/result.json"
result = json.loads(RESULT.read_text())
ledger = REPO / result["artifacts"]["attempts.jsonl"]["path"]
if (
    hashlib.sha256(ledger.read_bytes()).hexdigest()
    != result["artifacts"]["attempts.jsonl"]["sha256"]
):
    raise ValueError("attempt ledger hash mismatch")
rows = [json.loads(line) for line in ledger.read_text().splitlines()]
sections = []
displayed = {}
for family, metrics in result["per_arm"]["trained"].items():
    scope = result["family_target_scope"][family]
    cards, seen, identifiers = [], set(), []
    for row in rows:
        if (
            row["arm"] != "trained"
            or row["program_id"] != family
            or not row["assembly"]["programs"]
        ):
            continue
        smiles = row["canonical_smiles"]
        if smiles in seen:
            continue
        seen.add(smiles)
        identifiers.append(row["sample_index"])
        drawer = rdMolDraw2D.MolDraw2DSVG(500, 260)
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, Chem.MolFromSmiles(smiles))
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        svg = svg[svg.index("<svg") :]
        novelty = (
            "Absent from this cache's train set"
            if row["novel_vs_train"]
            else "Present in train set"
        )
        cards.append(
            f'<article>{svg}<p>Attempt {row["sample_index"]} · {row["requested_depth"]} reaction step(s)</p>'
            f"<p>{novelty}</p><details><summary>Structure and computed witnesses</summary>"
            f'<code>{html.escape(smiles)}</code><pre>{html.escape(json.dumps(row["assembly"], indent=2))}</pre>'
            "</details></article>"
        )
        if len(cards) == 3:
            break
    displayed[family] = identifiers
    scope_label = (
        "Precursor / neutral structure" if scope.startswith("precursor") else "Assembly support"
    )
    sections.append(
        f'<section id="{family}"><h2>{html.escape(family.replace("_", " "))}</h2>'
        f'<p class="scope">{scope_label}</p>'
        f'<p>{metrics["valid_connected"]}/{metrics["attempts"]} valid connected · '
        f'{metrics["exact_program"]}/{metrics["attempts"]} exact computed program · '
        f'{metrics["unique_novel_exact_products"]} distinct train-novel exact products</p>'
        f'<div class="cards">{"".join(cards)}</div></section>'
    )
trained = result["per_arm"]["trained"].values()
valid = sum(m["valid_connected"] for m in trained)
exact = sum(m["exact_program"] for m in trained)
attempts = sum(m["attempts"] for m in trained)
document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FORGE · Twelve-library generation diagnostic</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;color:#192a35;background:#f3f6f7;margin:0 auto;padding:36px;max-width:1500px}}
h1{{font-size:32px}}h2{{font-size:23px}}.intro{{max-width:1000px}}section{{margin-top:40px;border-top:2px solid #c6d4db}}
.scope{{font-weight:600;color:#365b73}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}
article{{background:white;border:1px solid #d4dee3;padding:14px;border-radius:8px;min-width:0}}svg{{width:100%;height:auto}}
pre,code{{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere}}details{{margin-top:12px}}footer{{margin-top:40px;font-size:12px}}
</style><h1>FORGE: generation across twelve libraries</h1><div class="intro">
<p><b>{valid}/{attempts}</b> trained attempts are valid and connected; <b>{exact}/{attempts}</b> have exact computed reaction programs.</p>
<p>Every family produced a distinct train-novel exact product. This small single-seed diagnostic uses training semantic layouts.
Exact computed assembly does not establish lipid realism, ionizability, experimental synthesis or complete precursor routes.</p>
<p>Each section shows the first three distinct trained outputs with an exact computed program, in saved attempt order.
These are diagnostic examples, not a candidate panel. Rates include every attempt; failures remain in the full ledger.</p>
<p>Four families have precursor/neutral target scope. Assembly-support labels also do not establish ionizability.</p></div>
{"".join(sections)}<footer>Primary result SHA-256: {hashlib.sha256(RESULT.read_bytes()).hexdigest()}</footer></html>"""
(OUTPUT / "review.html").write_text(document)
receipt = {
    "schema_version": "forge.combinatorial_generation_gallery.v1",
    "inputs": {
        str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (RESULT, ledger, Path(__file__).resolve())
    },
    "selection_rule": "first_three_distinct_exact_trained_outputs_per_family_in_saved_attempt_order",
    "displayed_attempt_indices": displayed,
    "review_html_sha256": hashlib.sha256((OUTPUT / "review.html").read_bytes()).hexdigest(),
    "scientific_reviewer_judgments": "not_collected",
}
(OUTPUT / "gallery_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
