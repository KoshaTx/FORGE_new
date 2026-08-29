from __future__ import annotations

from pathlib import Path

import pytest

from experiments.phase1.multireaction.attempt_label_supersession import (
    AttemptLabelSupersessionError,
    supersede_common_ugi_method_label,
)
from forge.core.hashing import artifact_record
from forge.core.io import write_json
from forge.model.common_ugi_benchmark import (
    CommonUgiAttempt,
    load_attempt_ledger,
    write_attempt_ledger,
)


def _attempt(index: int) -> CommonUgiAttempt:
    return CommonUgiAttempt(
        method_id="shared_bias_program_role_source",
        seed=20260825,
        attempt_index=index,
        status="generated",
        product_smiles="CC",
        method_visible_component_ids=(),
        generator_calls=1,
        reaction_calls=0,
        route_calls=0,
        oracle_calls=0,
        wall_seconds=0.0,
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    attempts = tmp_path / "source.jsonl.gz"
    write_attempt_ledger(attempts, [_attempt(0), _attempt(1)])
    receipt = artifact_record(attempts)
    receipt.update(
        {
            "method_id": "shared_bias_program_role_source",
            "seed": 20260825,
            "attempts": 2,
        }
    )
    evaluation = tmp_path / "evaluation.json"
    write_json(
        evaluation,
        {
            "schema_version": "forge.synthesis_program_production_evaluation_result.v1",
            "status": "pass",
            "seed": 20260825,
            "checkpoint_metrics": {"shared_bias_program_role_source": {}},
            "common_ugi_attempt_ledgers": {"shared_bias_program_role_source": receipt},
            "selection": {"candidate_selection": False},
            "calls": {"oracle": 0, "route": 0},
        },
    )
    return attempts, evaluation


def test_supersession_changes_only_method_label(tmp_path: Path) -> None:
    attempts, evaluation = _fixture(tmp_path)
    result = supersede_common_ugi_method_label(
        attempts,
        evaluation,
        tmp_path / "corrected",
        tmp_path,
        source_method_id="shared_bias_program_role_source",
        target_method_id="forge_transformer",
        seed=20260825,
        expected_attempts=2,
    )

    corrected = load_attempt_ledger(
        tmp_path / "corrected/attempts.jsonl.gz",
        expected_method="forge_transformer",
        expected_seed=20260825,
        expected_attempts=2,
    )
    assert result["status"] == "pass"
    assert result["changed_fields"] == ["method_id"]
    assert [row.product_smiles for row in corrected] == ["CC", "CC"]


def test_supersession_rejects_unapproved_alias(tmp_path: Path) -> None:
    attempts, evaluation = _fixture(tmp_path)

    with pytest.raises(AttemptLabelSupersessionError, match="unsupported"):
        supersede_common_ugi_method_label(
            attempts,
            evaluation,
            tmp_path / "corrected",
            tmp_path,
            source_method_id="shared_bias_program_role_source",
            target_method_id="different_model",
            seed=20260825,
            expected_attempts=2,
        )
