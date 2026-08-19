"""Build the complementary AiZynthFinder checkpoint-residual target panel."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from rdkit import Chem

TARGET_SCHEMA_VERSION = "forge.aizynthfinder_checkpoint_residual_target.v1"


class CheckpointResidualError(RuntimeError):
    """Raised when the checkpoint-residual target contract is malformed."""


def _canonical_smiles(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise CheckpointResidualError("target SMILES must be nonempty")
    molecule = Chem.MolFromSmiles(value)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise CheckpointResidualError(f"invalid connected target SMILES: {value}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _target_id(cohort: str, role: str, smiles: str) -> str:
    payload = {
        "schema_version": TARGET_SCHEMA_VERSION,
        "cohort": cohort,
        "role": role,
        "canonical_smiles": smiles,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_residual_targets(
    component_rows: Sequence[Mapping[str, Any]],
    leaf_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return unique Graph2Edits misses and upstream leaves for complementary search."""

    targets: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in component_rows:
        if bool(row.get("terminal_closed_route_hypothesis")) or bool(row.get("proposal_coherent")):
            continue
        role = str(row["role"])
        smiles = _canonical_smiles(row["target_smiles"])
        key = ("graph2edits_residual_component", role, smiles)
        targets[key] = {
            "schema_version": TARGET_SCHEMA_VERSION,
            "target_id": _target_id(*key),
            "cohort": key[0],
            "role": role,
            "canonical_smiles": smiles,
            "source_priority_rank": None,
            "linked_component_targets": [smiles],
            "graph2edits_proposal_coherent": False,
            "target_is_current_terminal_claim": False,
        }

    for row in leaf_rows:
        smiles = _canonical_smiles(row["leaf_smiles"])
        role = "upstream_terminal_leaf"
        key = ("unresolved_upstream_leaf", role, smiles)
        targets[key] = {
            "schema_version": TARGET_SCHEMA_VERSION,
            "target_id": _target_id(*key),
            "cohort": key[0],
            "role": role,
            "canonical_smiles": smiles,
            "source_priority_rank": int(row["priority_rank"]),
            "linked_component_targets": sorted(
                _canonical_smiles(value) for value in row.get("affected_targets", [])
            ),
            "graph2edits_proposal_coherent": None,
            "target_is_current_terminal_claim": False,
        }
    return [targets[key] for key in sorted(targets)]


def summarize_residual_records(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize complementary proposal and bounded-stock-search outcomes."""

    if not rows:
        raise CheckpointResidualError("residual worker ledger is empty")
    counts: Counter[str] = Counter()
    by_cohort: dict[str, Counter[str]] = {}
    seen: set[str] = set()
    for row in rows:
        target = row.get("target")
        if not isinstance(target, Mapping):
            raise CheckpointResidualError("worker target is malformed")
        target_id = str(target.get("target_id"))
        if target_id in seen:
            raise CheckpointResidualError(f"duplicate target_id: {target_id}")
        seen.add(target_id)
        cohort = str(target["cohort"])
        bucket = by_cohort.setdefault(cohort, Counter())
        proposals = row.get("single_step_proposals")
        if not isinstance(proposals, list):
            raise CheckpointResidualError("single-step proposals are malformed")
        search = row.get("full_search")
        if not isinstance(search, Mapping):
            raise CheckpointResidualError("full-search record is malformed")
        values = {
            "targets": 1,
            "targets_with_single_step_proposals": int(bool(proposals)),
            "single_step_proposals": len(proposals),
            "full_search_solved_to_public_stock": int(
                bool(search.get("is_solved_to_public_stock"))
            ),
            "full_search_execution_failures": int(search.get("execution_status") == "failed"),
        }
        counts.update(values)
        bucket.update(values)
    return {
        "total": dict(sorted(counts.items())),
        "by_cohort": {
            cohort: dict(sorted(bucket.items())) for cohort, bucket in sorted(by_cohort.items())
        },
    }


__all__ = [
    "CheckpointResidualError",
    "TARGET_SCHEMA_VERSION",
    "build_residual_targets",
    "summarize_residual_records",
]
