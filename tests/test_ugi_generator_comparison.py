from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

from forge.product.ugi_generator_comparison import compare_ugi_generator_arms


def _write_result(path: Path, smiles: str) -> None:
    path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "valid": True,
                        "smiles": smiles,
                        "program": {
                            "node_counts": [1, 1, 1],
                            "junction_budgets": [0, 0, 0],
                            "cycle_ranks": [0, 0, 0],
                            "attachment_counts": [1, 1, 1],
                        },
                        "offspring_by_role": {
                            "amine_head": [0],
                            "oxoester_aldehyde_body_tail": [0],
                            "isocyanide_tail": [0],
                        },
                    }
                ],
                "sampling": {"terminal_tree_repairs": 0},
            }
        )
    )


def test_matched_comparison_detects_unsupported_oxygen_bond(tmp_path: Path) -> None:
    joint = tmp_path / "joint.json"
    staged = tmp_path / "staged.json"
    probe = tmp_path / "probe.json"
    reference = tmp_path / "reference.csv.gz"
    _write_result(joint, "COOC")
    _write_result(staged, "CCN")
    probe.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "source_stratum": "current",
                        "branch_class": "linear",
                        "program": {
                            "node_counts": [1, 1, 1],
                            "junction_budgets": [0, 0, 0],
                            "cycle_ranks": [0, 0, 0],
                            "attachment_counts": [1, 1, 1],
                        },
                    }
                ]
            }
        )
    )
    with gzip.open(reference, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("product_id", "product_smiles"))
        writer.writeheader()
        writer.writerow({"product_id": "r1", "product_smiles": "CCN"})

    result = compare_ugi_generator_arms(
        joint_result_path=joint,
        staged_result_path=staged,
        probe_path=probe,
        reference_products_path=reference,
    )

    assert result["status"] == "fail"
    assert result["gates"]["joint_no_unsupported_reference_motif"] is False
    assert result["gates"]["staged_no_unsupported_reference_motif"] is True
    assert "oxygen_oxygen_bond" in result["unsupported_reference_motifs"]
    assert result["arms"]["joint"]["overall"]["motif_fractions"]["oxygen_oxygen_bond"] == 1.0
    assert result["arms"]["staged"]["overall"]["motif_fractions"]["oxygen_oxygen_bond"] == 0.0
