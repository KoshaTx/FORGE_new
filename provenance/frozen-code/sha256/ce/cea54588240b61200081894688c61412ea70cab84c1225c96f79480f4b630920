"""Verify placeholder coverage, archived numerical inputs and local PDF build receipts."""

import argparse
import ast
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}


def pdf_text(path):
    return subprocess.check_output(["pdftotext", "-layout", str(path), "-"], text=True)


def verify_pin_tree(value, verified):
    """Verify nested source receipts without loading large corpora into memory."""
    if isinstance(value, dict):
        if {"path", "sha256"} <= value.keys():
            path = ROOT / value["path"]
            assert path.is_file(), str(path)
            assert re.fullmatch(r"[0-9a-f]{64}", value["sha256"]), value
            if value["path"] not in verified:
                verified[value["path"]] = digest(path)
            assert verified[value["path"]] == value["sha256"], value["path"]
        else:
            for child in value.values():
                verify_pin_tree(child, verified)
    elif isinstance(value, list):
        for child in value:
            verify_pin_tree(child, verified)


def verify_reaction_figures(ledger, sections, manuscript, export_root):
    """Link all-family drawings, source-event checks and portable figure bytes."""
    render_path = HERE / "figures/all_family_reactions/render.json"
    render = json.loads(render_path.read_text())
    if "reaction_figure_contract" in ledger:
        assert ledger["reaction_figure_contract"] == pin(render_path)
    verified = {}
    verify_pin_tree(render["inputs"], verified)
    records_pin = next(
        p for p in render["inputs"] if Path(p["path"]).name == "reaction_records.json"
    )
    review_pin = next(
        p for p in render["inputs"] if Path(p["path"]).name == "chemistry_review.json"
    )
    records_path, review_path = ROOT / records_pin["path"], ROOT / review_pin["path"]
    records = json.loads(records_path.read_text())
    review = json.loads(review_path.read_text())
    verification_path = records_path.with_name("verification.json")
    verification = json.loads(verification_path.read_text())
    for document in (records, review, verification):
        verify_pin_tree(document, verified)
    assert verification["output"] == records_pin
    assert verification["families"] == 22 and verification["events"] == 34
    for key in (
        "all_22_qualified_families_present",
        "all_exact_source_roundtrips",
        "all_events_exact_unsaturated",
        "all_final_steps_equal_source_products",
    ):
        assert verification[key] is True, key
    assert verification["heldout_structures_used"] is False
    assert verification["model_calls"] == 0
    final_validation = verification["final_validation"]
    for key in ("black_check", "ruff_check", "independent_repeat_identical_record_bytes"):
        assert final_validation[key] is True, key
    for key in ("record_sha256_before_repeat", "record_sha256_after_repeat"):
        assert final_validation[key] == records_pin["sha256"]
    by_family = {r["family_id"]: r for r in records["records"]}
    assert len(by_family) == len(records["records"]) == 22
    assert set(by_family) == set(ledger["family_ids"])
    assert sum(len(r["steps"]) for r in by_family.values()) == 34
    for record in by_family.values():
        assert record["complete_source_roundtrip"] is True
        assert record["steps"][-1]["product_smiles"] == record["product_smiles"]
        for step in record["steps"]:
            assert step["forward_exact"] is True and step["forward_saturated"] is False

    assert review["status"].startswith("passed_")
    checked = review["record_review"]
    assert checked["inputs"]["reaction_records"] == records_pin
    assert checked["families_checked"] == 22 and checked["events_checked"] == 34
    reviewed = {r["family_id"]: r for r in checked["families"]}
    assert len(reviewed) == len(checked["families"]) == 22
    assert set(reviewed) == set(by_family) == set(review["families"])
    for family, source in by_family.items():
        assessed = reviewed[family]
        assert assessed["source_target_id"] == source["source_target_id"]
        assert assessed["ordered_trace_continuity_pass"] is True
        assert len(assessed["steps"]) == len(source["steps"])
        for step, original in zip(assessed["steps"], source["steps"], strict=True):
            for field in ("reaction_id", "stage_index", "step_index"):
                assert step[field] == original[field]
            assert step["exact_product_reproduced"] is True
            assert step["registered_net_balance_matches"] is True
            assert step["search_cap_reached"] is False

    existing = render["existing_figure_families"]
    assert len(existing) == len(set(existing)) == 3
    assert set(existing) == {"aldehyde_ugi3", "aza_michael_acrylate", "reductive_amination"}
    families = [family for figure in render["figures"] for family in figure["families"]]
    companion_ids = [f["family"] for f in families]
    assert len(companion_ids) == len(set(companion_ids)) == 19
    assert not (set(companion_ids) & set(existing))
    assert set(companion_ids) | set(existing) == set(by_family)
    assert r"\input{sections/reaction_figures_22.tex}" in manuscript
    assert r"\input{figures/forge_reaction_schemes/schemes_body.tex}" in manuscript
    assert r"\label{fig:reaction-schemes}" in manuscript
    scheme = HERE / "figures/forge_reaction_schemes/schemes_body.tex"
    assert len(re.findall(r"^\\rxnrow\{", scheme.read_text(), re.M)) == 3
    reaction_text = sections["reaction_figures_22.tex"]
    includes = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", reaction_text)
    expected_includes = [
        str((ROOT / figure["pdf"]["path"]).relative_to(HERE)) for figure in render["figures"]
    ]
    assert Counter(includes) == Counter(expected_includes)
    assert len(expected_includes) == len(set(expected_includes))
    figure_files = []
    for figure in render["figures"]:
        assert figure["families"]
        for extension in ("svg", "pdf"):
            source_pin = figure[extension]
            verify_pin_tree(source_pin, verified)
            source = ROOT / source_pin["path"]
            assert source.suffix == "." + extension
            exported = export_root / source.relative_to(HERE)
            assert digest(exported) == source_pin["sha256"], str(exported)
            figure_files.append(source_pin)
    assert digest(export_root / scheme.relative_to(HERE)) == digest(scheme)

    audits = defaultdict(list)
    required_checks = {
        "exact_registry_replay",
        "source_inputs_unchanged",
        "ordinary_atoms_and_bonds_only",
        "retained_atoms_hydrogens_charges_exact",
        "retained_bonds_exact",
        "all_hidden_fragments_identical",
        "all_hidden_attachments_identical",
        "all_heteroatoms_and_rings_retained",
        "all_double_and_triple_bond_endpoints_retained",
        "all_molecule_formal_charges_exact",
    }
    for audit in render["schematic_audits"]:
        family = audit["family"]
        assert family in companion_ids
        assert required_checks <= audit["checks"].keys()
        assert all(value is True for value in audit["checks"].values())
        verify_pin_tree(audit["registry_pin"], verified)
        assert audit["actual_formal_charges"] == audit["shown_formal_charges"]
        assert all(
            group["identical_atom_hydrogen_charge_bond_state"] is True
            for group in audit["r_groups"]
        )
        assert any(
            step["reaction_id"] == audit["reaction_id"]
            and step["registry_pin"] == audit["registry_pin"]
            and step["product_smiles"] == audit["actual_product_smiles"]
            and [r["smiles"] for r in step["reactants"]] == audit["actual_reactant_smiles"]
            for step in by_family[family]["steps"]
        ), (family, audit["displayed_step"])
        audits[family].append(audit)
    assert set(audits) == set(companion_ids)
    for item in families:
        family, group = item["family"], audits[item["family"]]
        source = by_family[family]
        assert item["source_target_id"] == source["source_target_id"]
        assert item["registered_events"] == len(source["steps"])
        assert item["displayed_transformations"] == len(group)
        assert sorted(a["displayed_step"] for a in group) == list(range(1, len(group) + 1))
        assert all(
            type(a["represented_events"]) is int and a["represented_events"] > 0 for a in group
        )
        assert sum(a["represented_events"] for a in group) == len(source["steps"])
    return {
        "render": pin(render_path),
        "source_records": records_pin,
        "source_verification": pin(verification_path),
        "chemistry_review": review_pin,
        "existing_source_scheme": review["existing_scheme"],
        "active_existing_scheme": pin(scheme),
        "family_ids": sorted(by_family),
        "source_records_count": len(by_family),
        "source_events": 34,
        "existing_families": existing,
        "companion_families": companion_ids,
        "companion_figure_groups": len(render["figures"]),
        "displayed_schematic_audits": len(render["schematic_audits"]),
        "represented_companion_events": sum(
            a["represented_events"] for a in render["schematic_audits"]
        ),
        "all_schematic_checks_pass": True,
        "source_and_exported_svg_pdf_pins_verified": figure_files,
        "verified_dependency_pins": [
            {"path": path, "sha256": sha} for path, sha in sorted(verified.items())
        ],
        "renderer_visual_status": render.get("visual_review"),
        "visual_review_scope": "Separate rendered-page review receipt; chemical-record review and hash checks do not establish visual legibility.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-pdf", type=Path, required=True)
    args = parser.parse_args()
    ledger = json.loads((HERE / "RESULTS_LEDGER.json").read_text())
    for value in ledger["inputs"].values():
        assert digest(ROOT / value["path"]) == value["sha256"], value["path"]
    methods_verified = {}
    assert ledger["methods_implementation_inputs"]
    verify_pin_tree(ledger["methods_implementation_inputs"], methods_verified)
    cohort_path = ROOT / "results/phase1/compose_lipid_training_cohort_v1/cohort.json"
    cohort = json.loads(cohort_path.read_text())
    assert ledger["family_ids"] == cohort["families"]
    sections = {p.name: p.read_text() for p in (HERE / "sections").glob("*.tex")}
    new_results = sections["results_22.tex"] + sections["appendix_22.tex"]
    metric_pin = ledger["quality_metric_contract"]
    assert digest(ROOT / metric_pin["path"]) == metric_pin["sha256"]
    metrics = json.loads((ROOT / metric_pin["path"]).read_text())
    for value in metrics["inputs"]:
        assert digest(ROOT / value["path"]) == value["sha256"], value["path"]
    assert metrics["table_labels"] == ledger["explicit_quality_tables"]
    for label in metrics["table_labels"]:
        assert "\\label{" + label + "}" in new_results
    for group, expected_count in (("quality_metrics", 11), ("realism_metrics", 9)):
        assert len(metrics[group]) == expected_count
        assert len({metric["key"] for metric in metrics[group]}) == expected_count
        for metric in metrics[group]:
            assert metric["final_value"] is None
            assert metric["label"] in new_results
    descriptor_source = ROOT / "forge/model/common_lipid_realism.py"
    assignments = ast.parse(descriptor_source.read_text()).body
    descriptor_names = next(
        ast.literal_eval(node.value)
        for node in assignments
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "DESCRIPTOR_NAMES"
            for target in node.targets
        )
    )
    assert [row["key"] for row in metrics["descriptors"]] == list(descriptor_names)
    for row in metrics["descriptors"]:
        assert row["label"] in new_results
        assert all(
            row[key] is None for key in ("reference", "raw", "final", "normalized_wasserstein")
        )
    for block in ledger["blocks"]:
        assert block["status"] == "PENDING" and block["final_result_artifact"] is None
        assert "\\label{" + block["latex_label"] + "}" in new_results
    tables = re.findall(r"\\begin\{table\}.*?\\end\{table\}", new_results, flags=re.S)
    matrix_rows = {}
    for label in ledger["family_matrices"]:
        table = next(t for t in tables if "\\label{" + label + "}" in t)
        rows = [line.split(" & ")[0] for line in table.splitlines() if " & \\PendingCell" in line]
        rows = [
            r
            for r in rows
            if r not in {"Equal-family macro", "Pooled all requests", "Worst-family result"}
        ]
        assert len(rows) == len(set(rows)) == len(ledger["family_ids"]), label
        matrix_rows[label] = rows
    assert all(rows == next(iter(matrix_rows.values())) for rows in matrix_rows.values())
    for label in ledger["figure_slots"]:
        assert "\\label{" + label + "}" in new_results
    for text in sections.values():
        assert not any(ord(c) < 32 and c not in "\n\t" for c in text)
    numerical = []
    for p in sorted((HERE / "generated").glob("*.tex")):
        historical = HERE / "archive/inherited_exports/FORGE_ICLR2027_Overleaf/generated" / p.name
        assert historical.exists() and digest(p) == digest(historical), p.name
        numerical.append(pin(p))
    log = (HERE / "FORGE_ICLR2027_paper.log").read_text()
    assert not re.search(r"Overfull|undefined|multiply defined|^!", log, re.M)
    export_log = args.export_pdf.with_suffix(".log").read_text()
    assert not re.search(r"Overfull|undefined|multiply defined|^!", export_log, re.M)
    pdf = HERE / "FORGE_ICLR2027_paper.pdf"
    text = pdf_text(pdf)
    assert text == pdf_text(args.export_pdf), "Portable export differs from canonical PDF text"
    pages = text.split("\f")
    count = sum(bool(p.strip()) for p in pages)
    manuscript = (HERE / "FORGE_ICLR2027_paper.tex").read_text()
    original = (HERE / "archive/three_family_source.tex").read_text()
    assert r"\section{Results}" in sections["results_22.tex"]
    active_text = (
        manuscript
        + new_results
        + sections["implementation_22.tex"]
        + sections["reaction_figures_22.tex"]
    )
    reaction_figures = verify_reaction_figures(
        ledger, sections, manuscript, args.export_pdf.resolve().parent
    )
    for forbidden in (
        "Working draft",
        "working draft",
        "This extension",
        "result slots",
        "Results to be completed",
        r"\ResultSlot",
        r"\TwentyTwoScope",
        r"\input{sections/historical_three_family_results.tex}",
    ):
        assert forbidden not in active_text, forbidden
    assert r"\title{FORGE: Reaction-Guided Generative Design of Ionizable Lipids}" in manuscript
    original_intro = original.split(r"\section{Introduction}", 1)[1].split(
        "Our contributions are:", 1
    )[0]
    final_intro = manuscript.split(r"\section{Introduction}", 1)[1].split(
        "Our contributions are:", 1
    )[0]
    assert original_intro == final_intro, "Original Introduction prose changed"
    theory_start = r"\section{Theoretical analysis and proofs}"
    original_theory = original.split(theory_start, 1)[1].split(
        r"\section{Reaction schemes and generated examples}", 1
    )[0]
    final_theory = manuscript.split(theory_start, 1)[1].split(
        r"\section{Assembly families and reaction schemes}", 1
    )[0]
    assert original_theory == final_theory, "Original theory changed"
    assert not any(ord(c) < 32 and c not in "\n\t" for c in active_text)
    header_pages = [i + 1 for i, p in enumerate(pages) if "AI USE STATEMENT" in p]
    report = dict(
        schema_version="forge.iclr22.placeholder_validation.v1",
        status="passed",
        claim="Editorial placeholder coverage and local builds only; no final scientific result admission",
        source_inputs=ledger["inputs"],
        cohort=pin(cohort_path),
        family_count=len(ledger["family_ids"]),
        evidence_blocks=len(ledger["blocks"]),
        all_family_matrices=len(matrix_rows),
        matrix_rows=matrix_rows,
        figure_slots=ledger["figure_slots"],
        explicit_quality_tables=metrics["table_labels"],
        quality_metric_count=len(metrics["quality_metrics"]),
        realism_metric_count=len(metrics["realism_metrics"]),
        descriptor_count=len(descriptor_names),
        quality_metric_contract=metric_pin,
        quality_metric_implementation_pins_verified=metrics["inputs"],
        methods_implementation_pins_verified=ledger["methods_implementation_inputs"],
        reaction_figures=reaction_figures,
        pending_cell_count=new_results.count("\\PendingCell"),
        inline_value_placeholders=active_text.count("\\PendingValue"),
        original_introduction_prose_preserved=True,
        original_theory_preserved=True,
        final_paper_narrative_without_draft_meta_commentary=True,
        all_final_result_artifacts_missing_and_explicit=True,
        historical_numerical_inputs_unchanged=numerical,
        compiled_pages=count,
        disclosure_starts_page=header_pages[0] if header_pages else None,
        canonical_and_fresh_zip_build_text_identical=True,
        unresolved_references=False,
        overfull_boxes=False,
        manuscript_inputs=[
            pin(p)
            for p in [
                HERE / "FORGE_ICLR2027_paper.tex",
                *sorted((HERE / "sections").glob("*.tex")),
                HERE / "RESULTS_LEDGER.json",
                HERE / "RESULTS_CHECKLIST.md",
                HERE / "QUALITY_METRICS.json",
            ]
        ],
        implementation=pin(Path(__file__)),
        outputs=[pin(pdf), pin(HERE / "FORGE_ICLR2027_22Families_Overleaf.zip")],
        no_new_training_sampling_or_test_access_for_document_task=True,
        sibling_v1_iclr_not_edited_by_this_task=True,
    )
    (HERE / "iclr22_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "family_count",
                    "evidence_blocks",
                    "all_family_matrices",
                    "pending_cell_count",
                    "compiled_pages",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
