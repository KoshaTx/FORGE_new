from __future__ import annotations

import json
from pathlib import Path

from experiments.phase1.product_l1.evaluation.ugi_v0_transformer_assessment import (
    assess_role_morphology,
    native_samples_to_attempts,
)
from forge.core.io import write_jsonl
from forge.model.local_chemistry_support import LocalChemistrySupport


def test_native_conversion_retains_failed_attempts() -> None:
    attempts = native_samples_to_attempts(
        (
            {"valid": True, "smiles": "CCN"},
            {"valid": False, "smiles": None},
        ),
        method_id="forge_v0_production",
        seed_label=0,
    )

    assert len(attempts) == 2
    assert attempts[0].status == "generated"
    assert attempts[0].product_smiles == "CCN"
    assert attempts[1].status == "invalid"
    assert attempts[1].product_smiles is None
    assert sum(attempt.generator_calls for attempt in attempts) == 2


def test_role_morphology_uses_exact_l1_components(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    support = LocalChemistrySupport.from_mapping(
        json.loads((repo / "results/phase1/local_morphology_support_v2/policy.json").read_text())
    )
    assessed = tmp_path / "assessed.jsonl.gz"
    row = {
        "schema_version": "forge.common_ugi_assessed_attempt.v1",
        "method_id": "forge_ugi_transformer",
        "seed": 0,
        "attempt_index": 0,
        "valid": True,
        "canonical_smiles": "CCCCCCCCNC(=O)C(CCCCCCCC)NCCN(C)C",
        "exact_l1_program": True,
        "exact_l1_traces": [
            {
                "components_by_role": {
                    "amine_head": "NCCN(C)C",
                    "oxoester_aldehyde_body_tail": "CCCCCCCC(=O)CCCC=O",
                    "isocyanide_tail": "[C-]#[N+]CCCCCCCC",
                }
            }
        ],
    }
    write_jsonl(
        assessed,
        [
            {"schema_version": "forge.common_ugi_assessed_attempts.v1", "rows": 1},
            row,
        ],
    )

    rows, result = assess_role_morphology(
        assessed,
        method_id="forge_ugi_transformer",
        seed_label=0,
        support=support,
    )

    assert len(rows) == 1
    assert result["counts"]["exact_l1_products"] == 1
    assert result["counts"]["exact_l1_products_with_any_supported_tail_trace"] == 1
    assert result["precision_among_exact_l1"]["any_supported_tail_trace"] == 1.0
    assert result["training_fold_only_policy"] is True
