"""Supersede a mislabeled internal common-Ugi attempt ledger without changing attempts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from forge.core.hashing import artifact_record, pin_record, sha256_file
from forge.core.io import read_json_object, write_json
from forge.model.common_ugi_benchmark import load_attempt_ledger, write_attempt_ledger

from .production_evaluation import RESULT_SCHEMA as EVALUATION_RESULT_SCHEMA

RESULT_SCHEMA = "forge.common_ugi_attempt_label_supersession.v1"
ALLOWED_ALIASES = {
    "shared_bias_program_role_source": "forge_transformer",
}


class AttemptLabelSupersessionError(ValueError):
    """A requested label correction is not a provable metadata-only change."""


def supersede_common_ugi_method_label(
    attempts_path: Path,
    evaluation_result_path: Path,
    output_dir: Path,
    repo: Path,
    *,
    source_method_id: str,
    target_method_id: str,
    seed: int,
    expected_attempts: int,
) -> dict[str, Any]:
    """Rewrite only ``method_id`` after authenticating the producing evaluation receipt."""

    if ALLOWED_ALIASES.get(source_method_id) != target_method_id:
        raise AttemptLabelSupersessionError(
            f"unsupported method-label supersession: {source_method_id!r} -> {target_method_id!r}"
        )
    evaluation = read_json_object(
        evaluation_result_path,
        error=AttemptLabelSupersessionError,
        label="source evaluation result",
    )
    ledgers = evaluation.get("common_ugi_attempt_ledgers")
    ledger_receipt = ledgers.get(source_method_id) if isinstance(ledgers, dict) else None
    if (
        evaluation.get("schema_version") != EVALUATION_RESULT_SCHEMA
        or evaluation.get("status") != "pass"
        or int(evaluation.get("seed", -1)) != seed
        or not isinstance(ledger_receipt, dict)
        or ledger_receipt.get("method_id") != source_method_id
        or int(ledger_receipt.get("seed", -1)) != seed
        or int(ledger_receipt.get("attempts", -1)) != expected_attempts
        or ledger_receipt.get("sha256") != str(sha256_file(attempts_path))
        or source_method_id not in evaluation.get("checkpoint_metrics", {})
    ):
        raise AttemptLabelSupersessionError(
            "source evaluation does not authenticate the requested attempt ledger and identity"
        )

    source_attempts = load_attempt_ledger(
        attempts_path,
        expected_method=source_method_id,
        expected_seed=seed,
        expected_attempts=expected_attempts,
    )
    corrected_attempts = tuple(
        replace(attempt, method_id=target_method_id) for attempt in source_attempts
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    corrected_path = output_dir / "attempts.jsonl.gz"
    write_attempt_ledger(corrected_path, corrected_attempts)
    reloaded = load_attempt_ledger(
        corrected_path,
        expected_method=target_method_id,
        expected_seed=seed,
        expected_attempts=expected_attempts,
    )
    source_without_label = [
        {key: value for key, value in attempt.to_mapping().items() if key != "method_id"}
        for attempt in source_attempts
    ]
    corrected_without_label = [
        {key: value for key, value in attempt.to_mapping().items() if key != "method_id"}
        for attempt in reloaded
    ]
    gates = {
        "source_evaluation_authenticated": True,
        "attempt_denominator_preserved": len(reloaded) == expected_attempts,
        "seed_preserved": {attempt.seed for attempt in reloaded} == {seed},
        "only_method_id_changed": source_without_label == corrected_without_label,
        "candidate_selection_absent": evaluation.get("selection", {}).get(
            "candidate_selection"
        )
        is False,
        "route_or_oracle_calls_zero": evaluation.get("calls") == {"oracle": 0, "route": 0},
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "source_method_id": source_method_id,
        "target_method_id": target_method_id,
        "seed": seed,
        "attempts": expected_attempts,
        "source_attempt_ledger": pin_record(attempts_path, repo),
        "source_evaluation_result": pin_record(evaluation_result_path, repo),
        "corrected_attempt_ledger": artifact_record(corrected_path),
        "changed_fields": ["method_id"],
        "candidate_selection": False,
        "gates": gates,
    }
    write_json(output_dir / "result.json", result)
    if result["status"] != "pass":
        raise AttemptLabelSupersessionError(f"label-supersession gates failed: {gates}")
    return result


__all__ = [
    "ALLOWED_ALIASES",
    "RESULT_SCHEMA",
    "AttemptLabelSupersessionError",
    "supersede_common_ugi_method_label",
]
