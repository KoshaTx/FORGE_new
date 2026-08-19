"""Frozen proposal-only exact-route recovery audit for AiZynthFinder.

Proposal generation sees only the public 120-target manifest.  The separately
frozen scoring truth is loaded after the proposal ledger has been written.  A
recovered reactant set measures proposal recall; it is not route evidence and
cannot enter synthesis value without independent FORGE adjudication.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from rdkit import Chem

CONFIG_SCHEMA_VERSION = "phase1_aizynthfinder_single_step_recovery_benchmark_config.v1"
TARGET_SCHEMA_VERSION = "forge.single_step_benchmark_lane_targets.v1"
TRUTH_SCHEMA_VERSION = "forge.single_step_benchmark_scoring_truth.v1"
LEDGER_SCHEMA_VERSION = "phase1_aizynthfinder_single_step_recovery_ledger.v1"
PROPOSAL_RESULT_SCHEMA_VERSION = "phase1_aizynthfinder_single_step_recovery_proposal_result.v1"
SCORE_SCHEMA_VERSION = "phase1_aizynthfinder_single_step_recovery_score.v1"


class AiZynthFinderRecoveryError(RuntimeError):
    """Raised when frozen benchmark identities or proposal rows are invalid."""


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def content_sha256(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_gzip_json(path: Path) -> dict[str, Any]:
    try:
        with gzip.open(path, "rt") as stream:
            value = json.load(stream)
    except (OSError, gzip.BadGzipFile, json.JSONDecodeError) as exc:
        raise AiZynthFinderRecoveryError(f"invalid frozen gzip JSON: {path}") from exc
    if not isinstance(value, dict):
        raise AiZynthFinderRecoveryError(f"frozen artifact must be an object: {path}")
    return value


def canonical_connected_smiles(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise AiZynthFinderRecoveryError(f"{label} must be nonempty SMILES")
    molecule = Chem.MolFromSmiles(value)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise AiZynthFinderRecoveryError(f"{label} must be one connected molecule")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def validate_public_targets(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if payload.get("schema_version") != TARGET_SCHEMA_VERSION:
        raise AiZynthFinderRecoveryError("target-manifest schema changed")
    if payload.get("truth_fields_present") is not False:
        raise AiZynthFinderRecoveryError("proposal targets contain truth fields")
    if payload.get("known_routes_exposed_to_lanes") is not False:
        raise AiZynthFinderRecoveryError("known routes are exposed to proposal execution")
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list) or len(raw_targets) != 120:
        raise AiZynthFinderRecoveryError("frozen target panel must contain 120 targets")
    targets: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_smiles: set[str] = set()
    for raw in raw_targets:
        if not isinstance(raw, dict):
            raise AiZynthFinderRecoveryError("target record is malformed")
        target_id = raw.get("target_id")
        role = raw.get("declared_role")
        stratum = raw.get("primary_stratum")
        if not all(isinstance(value, str) and value for value in (target_id, role, stratum)):
            raise AiZynthFinderRecoveryError("target identity is malformed")
        smiles = canonical_connected_smiles(raw.get("canonical_smiles"), label="target")
        if target_id in seen_ids or smiles in seen_smiles:
            raise AiZynthFinderRecoveryError("target panel is not constitutionally unique")
        seen_ids.add(target_id)
        seen_smiles.add(smiles)
        targets.append(
            {
                "target_id": target_id,
                "canonical_smiles": smiles,
                "declared_role": role,
                "primary_stratum": stratum,
                "secondary_chemotype_tags": list(raw.get("secondary_chemotype_tags", [])),
            }
        )
    return tuple(targets)


def proposal_result(
    *,
    config_path: str,
    config_sha256: str,
    target_path: str,
    target_sha256: str,
    runtime_path: str,
    runtime_sha256: str,
    ledger_path: str,
    ledger_sha256: str,
    rows: Iterable[Mapping[str, Any]],
    maximum_proposals: int,
) -> dict[str, Any]:
    records = list(rows)
    if len(records) != 120:
        raise AiZynthFinderRecoveryError("proposal ledger must contain all 120 targets")
    ids = [str(record.get("target_id")) for record in records]
    if len(set(ids)) != len(ids):
        raise AiZynthFinderRecoveryError("proposal ledger contains duplicate targets")
    proposal_counts = [len(record.get("proposals", [])) for record in records]
    result: dict[str, Any] = {
        "schema_version": PROPOSAL_RESULT_SCHEMA_VERSION,
        "status": "proposal_ledger_frozen_before_truth_scoring",
        "config": {"path": config_path, "sha256": config_sha256},
        "proposal_execution_inputs": {
            "target_manifest": {"path": target_path, "sha256": target_sha256},
            "runtime_manifest": {"path": runtime_path, "sha256": runtime_sha256},
        },
        "hidden_truth_loaded_during_proposal_execution": False,
        "summary": {
            "targets": len(records),
            "targets_with_proposals": sum(count > 0 for count in proposal_counts),
            "proposals": sum(proposal_counts),
            "maximum_proposals_per_target": maximum_proposals,
        },
        "artifacts": {
            "proposal_ledger": {
                "path": ledger_path,
                "sha256": ledger_sha256,
                "rows": len(records),
                "schema_version": LEDGER_SCHEMA_VERSION,
            }
        },
        "scientific_authority": {
            "proposal_only": True,
            "route_evidence_created": False,
            "route_closure_authorized": False,
            "may_enter_synthesis_value": False,
            "synthesis_tilting_activated": False,
        },
    }
    result["result_sha256"] = content_sha256(result)
    return result


def _fraction(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "fraction": None if denominator == 0 else numerator / denominator,
    }


def score_frozen_proposals(
    *,
    proposal_result_payload: Mapping[str, Any],
    proposal_rows: Iterable[Mapping[str, Any]],
    truth_payload: Mapping[str, Any],
    truth_path: str,
    truth_sha256: str,
) -> dict[str, Any]:
    """Score exact reactant recovery after proposal-ledger freeze."""

    if proposal_result_payload.get("schema_version") != PROPOSAL_RESULT_SCHEMA_VERSION:
        raise AiZynthFinderRecoveryError("proposal result schema changed")
    if proposal_result_payload.get("hidden_truth_loaded_during_proposal_execution") is not False:
        raise AiZynthFinderRecoveryError("proposal execution accessed hidden truth")
    if truth_payload.get("schema_version") != TRUTH_SCHEMA_VERSION:
        raise AiZynthFinderRecoveryError("truth-manifest schema changed")
    if truth_payload.get("lane_input") is not False:
        raise AiZynthFinderRecoveryError("scoring truth was marked as a lane input")

    rows = list(proposal_rows)
    by_target: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        target_id = str(row.get("target_id"))
        if target_id in by_target:
            raise AiZynthFinderRecoveryError("proposal rows contain duplicate targets")
        by_target[target_id] = row

    truth_records = truth_payload.get("records")
    if not isinstance(truth_records, list):
        raise AiZynthFinderRecoveryError("truth records are malformed")
    exact: list[Mapping[str, Any]] = []
    adversarial_ids: set[str] = set()
    for record in truth_records:
        if not isinstance(record, dict):
            raise AiZynthFinderRecoveryError("truth record is malformed")
        kind = record.get("truth_kind")
        if kind == "documented_exact_forward_unique_reactant_multiset":
            exact.append(record)
        elif kind == "valid_connected_wrong_handle_role_swap_control":
            adversarial_ids.add(str(record.get("target_id")))
        else:
            raise AiZynthFinderRecoveryError("unsupported truth kind")
    if len(exact) != 36 or len(adversarial_ids) != 12:
        raise AiZynthFinderRecoveryError("frozen truth census changed")

    top_k_counts = Counter({1: 0, 5: 0, 10: 0, 20: 0})
    by_stratum: dict[str, Counter[int]] = defaultdict(Counter)
    per_target: list[dict[str, Any]] = []
    for truth in exact:
        target_id = str(truth.get("target_id"))
        row = by_target.get(target_id)
        if row is None:
            raise AiZynthFinderRecoveryError(f"missing proposal row for {target_id}")
        expected = tuple(
            sorted(
                canonical_connected_smiles(value, label="truth reactant")
                for value in truth.get("canonical_reactant_multiset", [])
            )
        )
        proposals = row.get("proposals")
        if not isinstance(proposals, list):
            raise AiZynthFinderRecoveryError("proposal list is malformed")
        first_rank: int | None = None
        for proposal in proposals:
            if not isinstance(proposal, dict):
                raise AiZynthFinderRecoveryError("proposal record is malformed")
            reactants = tuple(
                sorted(
                    canonical_connected_smiles(value, label="proposed reactant")
                    for value in proposal.get("canonical_reactants", [])
                )
            )
            if reactants == expected:
                first_rank = int(proposal.get("rank"))
                break
        stratum = str(row.get("primary_stratum"))
        for k in top_k_counts:
            recovered = first_rank is not None and first_rank <= k
            top_k_counts[k] += int(recovered)
            by_stratum[stratum][k] += int(recovered)
        per_target.append(
            {
                "target_id": target_id,
                "primary_stratum": stratum,
                "transformation": truth.get("transformation"),
                "first_exact_recovery_rank": first_rank,
            }
        )

    stratum_denominators = Counter(
        str(by_target[str(record.get("target_id"))].get("primary_stratum")) for record in exact
    )
    score: dict[str, Any] = {
        "schema_version": SCORE_SCHEMA_VERSION,
        "status": "proposal_recovery_scored_after_ledger_freeze",
        "proposal_result_sha256": proposal_result_payload.get("result_sha256"),
        "proposal_ledger_sha256": proposal_result_payload.get("artifacts", {})
        .get("proposal_ledger", {})
        .get("sha256"),
        "scoring_truth": {"path": truth_path, "sha256": truth_sha256},
        "summary": {
            "exact_route_targets": len(exact),
            "adversarial_controls_reserved_for_full_operational_benchmark": len(adversarial_ids),
            "known_route_top_k": {
                str(k): _fraction(top_k_counts[k], len(exact)) for k in sorted(top_k_counts)
            },
            "known_route_top_k_by_primary_stratum": {
                stratum: {
                    str(k): _fraction(counts[k], stratum_denominators[stratum])
                    for k in sorted(top_k_counts)
                }
                for stratum, counts in sorted(by_stratum.items())
            },
        },
        "per_exact_target": sorted(per_target, key=lambda record: record["target_id"]),
        "scientific_authority": {
            "exact_recovery_is_route_evidence": False,
            "route_evidence_created": False,
            "route_closure_authorized": False,
            "may_enter_synthesis_value": False,
            "production_backend_activated": False,
            "synthesis_tilting_activated": False,
        },
        "next_gate": (
            "Run source-neutral forward, scope, evidence, operational and terminal-material "
            "adjudication; then compare the proposal-augmented evaluator against strict "
            "routing before any matched synthesis-tilt rerun."
        ),
    }
    score["score_sha256"] = content_sha256(score)
    return score


__all__ = [
    "AiZynthFinderRecoveryError",
    "CONFIG_SCHEMA_VERSION",
    "LEDGER_SCHEMA_VERSION",
    "PROPOSAL_RESULT_SCHEMA_VERSION",
    "SCORE_SCHEMA_VERSION",
    "canonical_connected_smiles",
    "content_sha256",
    "load_gzip_json",
    "proposal_result",
    "score_frozen_proposals",
    "sha256_file",
    "stable_json",
    "validate_public_targets",
]
