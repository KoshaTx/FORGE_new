"""Apply the frozen HeLa policy after exact terminal chemical admission.

The v2 biological policy remains scientifically unchanged.  This adapter adds
the production-v2 generator contract: a terminal must first pass exact L1 and
all registry-defined precursor-handle checks before any biological
applicability classification or oracle call is permitted.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.phase1.hela_potency.evaluation import (
    FrozenHeLaOracleWorker,
    HeLaBatchPredictor,
    HeLaPotencyDiagnosticPolicy,
)
from experiments.phase1.hela_potency.ranking import ugi_production_full_support_rescoring as v2
from experiments.phase1.hela_potency.ranking.ugi_production_terminal_ranking import EXPECTED_ARMS
from forge.core.hashing import sha256_file

CONFIG_SCHEMA_VERSION = "phase1_ugi_production_full_support_rescoring_config.v3"
RESULT_SCHEMA_VERSION = "phase1_ugi_production_full_support_rescoring.v3"
LEDGER_SCHEMA_VERSION = "phase1_ugi_production_full_support_rescoring_ledger.v3"
GENERATION_STATUS = "complete_constrained_stochastic_production_candidate_generation"


class UgiProductionFullSupportRescoringV3Error(RuntimeError):
    """Raised when the chemically admitted rescoring contract changes."""


def _prepare_rows(
    rows: Sequence[Mapping[str, Any]], *, expected_draws_per_arm: int
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    """Validate admission records and mask chemically inadmissible terminals."""

    counts = Counter(str(row.get("arm_id", "")) for row in rows)
    expected = {arm: expected_draws_per_arm for arm in EXPECTED_ARMS}
    if counts != expected:
        raise UgiProductionFullSupportRescoringV3Error(
            f"production row counts changed: {dict(counts)} != {expected}"
        )

    prepared: list[dict[str, Any]] = []
    metadata: dict[tuple[str, int], dict[str, Any]] = {}
    for source in rows:
        arm = str(source["arm_id"])
        draw_index = int(source["draw_index"])
        key = (arm, draw_index)
        if key in metadata:
            raise UgiProductionFullSupportRescoringV3Error(f"duplicate production draw: {key}")
        admission = source.get("terminal_chemical_admission")
        if not isinstance(admission, Mapping):
            raise UgiProductionFullSupportRescoringV3Error(
                f"terminal admission is absent for {key}"
            )
        admitted = admission.get("admitted")
        raw_exact_l1 = admission.get("exact_l1")
        raw_valid = admission.get("raw_molecule_valid")
        reason = admission.get("reason")
        if (
            not isinstance(admitted, bool)
            or not isinstance(raw_exact_l1, bool)
            or not isinstance(raw_valid, bool)
            or not isinstance(reason, str)
            or not reason
        ):
            raise UgiProductionFullSupportRescoringV3Error(
                f"terminal admission is malformed for {key}"
            )
        if admitted and (not raw_exact_l1 or not raw_valid or reason != "admitted"):
            raise UgiProductionFullSupportRescoringV3Error(
                f"admitted terminal has inconsistent evidence for {key}"
            )
        copied = dict(source)
        if not admitted:
            copied["native_terminal"] = None
        prepared.append(copied)
        metadata[key] = {
            "terminal_chemical_admitted": admitted,
            "terminal_admission_reason": reason,
            "raw_exact_l1": raw_exact_l1,
            "raw_molecule_valid": raw_valid,
        }
    return prepared, metadata


def _attach_admission_metadata(
    classified: list[dict[str, Any]], metadata: Mapping[tuple[str, int], Mapping[str, Any]]
) -> None:
    for row in classified:
        key = (str(row["arm_id"]), int(row["draw_index"]))
        values = metadata[key]
        row.update(values)
        if not values["terminal_chemical_admitted"]:
            row["reason"] = f"terminal_chemical_admission:{values['terminal_admission_reason']}"


def _arm_summary(rows: Sequence[dict[str, Any]], arm: str) -> dict[str, Any]:
    summary = v2._arm_summary(rows, arm)
    selected = [row for row in rows if row["arm_id"] == arm]
    summary.update(
        {
            "raw_exact_l1": sum(bool(row["raw_exact_l1"]) for row in selected),
            "terminal_chemical_admitted": sum(
                bool(row["terminal_chemical_admitted"]) for row in selected
            ),
            "terminal_admission_reasons": dict(
                sorted(Counter(str(row["terminal_admission_reason"]) for row in selected).items())
            ),
        }
    )
    return summary


def build_full_support_rescoring_v3(
    repo: Path,
    config_path: Path,
    *,
    predictor: HeLaBatchPredictor | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Build the chemically admitted, frozen-policy terminal rescore."""

    repo = repo.resolve()
    config = v2._load(config_path.resolve(), label="full-support rescoring config v3")
    if (
        config.get("schema_version") != CONFIG_SCHEMA_VERSION
        or config.get("status") != "frozen_before_chemically_admitted_full_support_oracle_rescoring"
    ):
        raise UgiProductionFullSupportRescoringV3Error("unsupported rescoring config")
    paths = {
        label: v2._pin(repo, record, label=label) for label, record in config["inputs"].items()
    }
    generation = v2._load(paths["generation_result"], label="production generation v2")
    if generation.get("status") != GENERATION_STATUS:
        raise UgiProductionFullSupportRescoringV3Error("production generation status changed")
    ledger_record = generation.get("artifacts", {}).get("terminal_ledger.jsonl.gz", {})
    if ledger_record.get("sha256") != sha256_file(paths["terminal_ledger"]):
        raise UgiProductionFullSupportRescoringV3Error(
            "generation result does not pin the terminal ledger"
        )

    rows = v2._read_jsonl(paths["terminal_ledger"])
    prepared, metadata = _prepare_rows(
        rows, expected_draws_per_arm=int(config["production_contract"]["draws_per_arm"])
    )
    policy = HeLaPotencyDiagnosticPolicy(repo, paths["hela_diagnostic_policy"])
    pattern_policies = config["pattern_policies"]
    expected_patterns = {
        v2.pattern_id_for_roles(roles)
        for roles in (
            (),
            ("amine",),
            ("aldehyde",),
            ("isocyanide",),
            ("amine", "aldehyde"),
            ("amine", "isocyanide"),
            ("aldehyde", "isocyanide"),
            ("amine", "aldehyde", "isocyanide"),
        )
    }
    if set(pattern_policies) != expected_patterns:
        raise UgiProductionFullSupportRescoringV3Error("novelty pattern set changed")

    extended = v2._extended_scales(policy, config["extended_calibration_scales"])
    scales: dict[str, Any] = {**policy.scales, **extended}
    classified = v2._classify(prepared, policy, pattern_policies)
    _attach_admission_metadata(classified, metadata)
    unique = v2._unique_scoring_rows(classified)
    batch_size = int(config["scoring"]["batch_size"])
    if predictor is None:
        with FrozenHeLaOracleWorker(policy) as worker:
            receipts = v2._apply_scores(classified, unique, worker, scales, batch_size)
            ready_receipt = worker.ready_receipt_sha256
    else:
        receipts = v2._apply_scores(classified, unique, predictor, scales, batch_size)
        ready_receipt = "test_predictor"

    fields = tuple(key for key in classified[0] if key != "candidate")
    ledger = v2._gzip_csv(classified, fields)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_chemically_admitted_full_support_terminal_rescoring",
        "scope": dict(config["scope"]),
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "policy": {
            "terminal_chemical_admission_precedes_biological_applicability": True,
            "all_admitted_nonmeasured_interpolative_products_scored": True,
            "exact_novelty_is_provenance_not_applicability": True,
            "claim_authority_is_role_specific": True,
            "potency_changes_generation": False,
            "pattern_policies": pattern_policies,
        },
        "summary": {
            "arms": {arm: _arm_summary(classified, arm) for arm in EXPECTED_ARMS},
            "global_unique_oracle_calls": len(unique),
            "oracle_response_receipts": sorted(receipts),
        },
        "worker_ready_receipt_sha256": ready_receipt,
        "artifacts": {
            "terminal_rescoring.csv.gz": {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "rows": len(classified),
                "sha256": hashlib.sha256(ledger).hexdigest(),
            }
        },
        "next_gate": "diversity_balanced_route_shortlists_stratified_by_evidence_tier",
        "nonclaims": [
            "A raw oracle score is not a potency-generalization guarantee.",
            "Exploratory tiers do not authorize potency tilting.",
            "This rescore does not change generation, applicability boundaries, or panel lock.",
        ],
    }
    result["result_sha256"] = hashlib.sha256(v2._stable_json(result).encode()).hexdigest()
    return result, ledger


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "LEDGER_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "UgiProductionFullSupportRescoringV3Error",
    "build_full_support_rescoring_v3",
]
