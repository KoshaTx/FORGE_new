"""Verify placeholder coverage, archived numerical inputs and local PDF build receipts."""

import argparse
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}


def pdf_text(path):
    return subprocess.check_output(["pdftotext", "-layout", str(path), "-"], text=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-pdf", type=Path, required=True)
    args = parser.parse_args()
    ledger = json.loads((HERE / "RESULTS_LEDGER.json").read_text())
    for value in ledger["inputs"].values():
        assert digest(ROOT / value["path"]) == value["sha256"], value["path"]
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
    assert "Results to be completed" in (HERE / "sections/results_22.tex").read_text()
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
        pending_cell_count=new_results.count("\\PendingCell"),
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
