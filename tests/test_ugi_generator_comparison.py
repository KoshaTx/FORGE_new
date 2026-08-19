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


def _write_result_samples(path: Path, smiles: list[str]) -> None:
    """Like `_write_result` but for several samples, to exercise cross-sample deduplication."""
    program = {
        "node_counts": [1, 1, 1],
        "junction_budgets": [0, 0, 0],
        "cycle_ranks": [0, 0, 0],
        "attachment_counts": [1, 1, 1],
    }
    path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "valid": True,
                        "smiles": entry,
                        "program": program,
                        "offspring_by_role": {
                            "amine_head": [0],
                            "oxoester_aldehyde_body_tail": [0],
                            "isocyanide_tail": [0],
                        },
                    }
                    for entry in smiles
                ],
                "sampling": {"terminal_tree_repairs": 0},
            }
        )
    )


def _write_probe(path: Path, count: int) -> None:
    path.write_text(
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
                * count
            }
        )
    )


def test_unique_valid_fraction_is_stereo_free(tmp_path: Path) -> None:
    """Two enantiomers are one constitution, so they must not count as two unique products.

    Phase 1 model identity is constitutional and stereo-free (AGENTS.md), and
    `unique_valid_fraction` is compared across modules against a shared 0.98 gate. This site once
    relied on RDKit's `isomericSmiles` default, which is True, making it the only module in the
    package computing the metric stereo-bearing. That fails here: the enantiomer pair would
    deduplicate to two rather than one and the fraction would read 1.0 instead of 0.5.
    """
    joint = tmp_path / "joint.json"
    staged = tmp_path / "staged.json"
    probe = tmp_path / "probe.json"
    reference = tmp_path / "reference.csv.gz"

    _write_result_samples(joint, ["C[C@H](N)CC", "C[C@@H](N)CC"])
    _write_result_samples(staged, ["CCN", "CCO"])
    _write_probe(probe, 2)
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

    assert result["arms"]["joint"]["overall"]["unique_valid_fraction"] == 0.5
    # Control: two genuinely distinct constitutions still count as two.
    assert result["arms"]["staged"]["overall"]["unique_valid_fraction"] == 1.0


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
