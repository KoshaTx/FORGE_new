"""Proposal-only Graph2Edits audit for frozen synthesis-guidance checkpoints.

The earlier checkpoint contrast audit joined checkpoint components to an
exact component ledger.  It did not rerun the proposal engine for unresolved
checkpoint chemistry.  This module closes that diagnostic gap while retaining
the proposal engine's deliberately limited authority: proposals and model
scores cannot create evidence, close routes, or enter synthesis value until an
independent forward and terminal-material adjudication has passed.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.route.graph2edits_one_gap_diagnostic import ProposalBackend, build_request
from forge.route.proposal_discovery_status import SourceNeutralProposalDiscoveryResolver
from forge.route.semantic_family_equivalence import audit_semantic_family_equivalence_v2
from forge.value.ugi3_source_bounded_route_value_contrast import EXACT
from forge.value.ugi3_stepwise_route_value_contrast import FAMILY_ALL

RESULT_SCHEMA_VERSION = "forge.graph2edits_checkpoint_contrast.v1"
LEDGER_SCHEMA_VERSION = "forge.graph2edits_checkpoint_contrast_ledger.v1"


class CheckpointContrastProposalError(RuntimeError):
    """Raised when the frozen checkpoint proposal contract is malformed."""


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CheckpointComponentTarget:
    """One unresolved role component deduplicated across checkpoint completions."""

    role: str
    canonical_smiles: str
    occurrence_count: int
    phase_counts: tuple[tuple[str, int], ...]
    representative_row: Mapping[str, Any]

    @property
    def target_id(self) -> str:
        return f"checkpoint-component:{_sha256_payload([self.role, self.canonical_smiles])[:20]}"

    @property
    def arm_counts(self) -> tuple[tuple[str, int], ...]:
        """Compatibility view required by the shared proposal-request builder."""

        return self.phase_counts


def _component_key(value: Mapping[str, Any]) -> tuple[str, str]:
    role = value.get("role")
    smiles = value.get("canonical_smiles")
    if not isinstance(role, str) or not role or not isinstance(smiles, str) or not smiles:
        raise CheckpointContrastProposalError("checkpoint component is malformed")
    return role, smiles


def _proposal_representative(row: Mapping[str, Any]) -> dict[str, Any]:
    strict = row.get("strict_assessment_receipt")
    audit = row.get("support_audit")
    if not isinstance(strict, Mapping) or not isinstance(audit, Mapping):
        raise CheckpointContrastProposalError(
            "checkpoint row lacks strict assessment or support audit"
        )
    return {
        **dict(row),
        "assessment_receipt": dict(strict),
        "support_audit": dict(audit),
    }


def extract_checkpoint_component_targets(
    support_payload: Mapping[str, Any],
    *,
    arm: str,
    checkpoint_phase: str,
    terminal_phase: str,
    checkpoints: Sequence[int],
) -> tuple[CheckpointComponentTarget, ...]:
    """Return every unique unresolved component in frozen checkpoint/final records."""

    records = support_payload.get("records")
    if not isinstance(records, list) or not records:
        raise CheckpointContrastProposalError("support payload contains no records")
    allowed_checkpoints = {int(value) for value in checkpoints}
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in records:
        if not isinstance(row, Mapping) or row.get("arm") != arm:
            continue
        phase = str(row.get("assessment_phase"))
        if phase == checkpoint_phase:
            if int(row.get("checkpoint", -1)) not in allowed_checkpoints:
                continue
        elif phase != terminal_phase:
            continue
        receipt = row.get("graded_route_readiness")
        if not isinstance(receipt, Mapping) or not isinstance(receipt.get("components"), list):
            raise CheckpointContrastProposalError("graded checkpoint receipt is malformed")
        for component in receipt["components"]:
            if not isinstance(component, Mapping):
                raise CheckpointContrastProposalError("graded component receipt is malformed")
            if str(component.get("graded_evidence_class")) in {EXACT, FAMILY_ALL}:
                continue
            grouped.setdefault(_component_key(component), []).append(row)

    targets: list[CheckpointComponentTarget] = []
    for (role, smiles), rows in grouped.items():
        phases = Counter(str(row["assessment_phase"]) for row in rows)
        representative = min(
            rows,
            key=lambda row: (
                0 if row.get("assessment_phase") == checkpoint_phase else 1,
                int(row.get("checkpoint", 10**9)),
                int(row.get("program_index", 10**9)),
                int(row.get("particle_index", 10**9)),
                str(row.get("terminal_sha256")),
            ),
        )
        targets.append(
            CheckpointComponentTarget(
                role=role,
                canonical_smiles=smiles,
                occurrence_count=len(rows),
                phase_counts=tuple(sorted(phases.items())),
                representative_row=_proposal_representative(representative),
            )
        )
    return tuple(sorted(targets, key=lambda value: (value.role, value.canonical_smiles)))


def run_checkpoint_component_proposals(
    *,
    support_payload: Mapping[str, Any],
    support_path: Path,
    backend: ProposalBackend,
    resolver: SourceNeutralProposalDiscoveryResolver,
    arm: str,
    checkpoint_phase: str,
    terminal_phase: str,
    checkpoints: Sequence[int],
    maximum_proposals: int = 10,
    repeat_count: int = 2,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run deterministic proposals for every unresolved checkpoint component."""

    if maximum_proposals < 1 or repeat_count < 1:
        raise CheckpointContrastProposalError("proposal and repeat counts must be positive")
    targets = extract_checkpoint_component_targets(
        support_payload,
        arm=arm,
        checkpoint_phase=checkpoint_phase,
        terminal_phase=terminal_phase,
        checkpoints=checkpoints,
    )
    if not targets:
        raise CheckpointContrastProposalError("checkpoint proposal population is empty")

    rows: list[dict[str, Any]] = []
    target_summaries: list[dict[str, Any]] = []
    raw_counts: Counter[str] = Counter()
    semantic_counts: Counter[str] = Counter()
    for target in targets:
        request = build_request(target)  # type: ignore[arg-type]
        repeats = [
            backend.propose_with_trace(request, maximum_proposals=maximum_proposals)
            for _ in range(repeat_count)
        ]
        identities = [
            tuple(
                (proposal.reactant_smiles, proposal.model_score, proposal.rank)
                for proposal in batch.proposals
            )
            for batch in repeats
        ]
        if not all(identity == identities[0] for identity in identities[1:]):
            raise CheckpointContrastProposalError(
                f"Graph2Edits output was not deterministic for {target.target_id}"
            )
        target_raw: Counter[str] = Counter()
        target_semantic: Counter[str] = Counter()
        for proposal in repeats[0].proposals:
            resolution = resolver.resolve(proposal)
            semantic = audit_semantic_family_equivalence_v2(resolution, resolver.transforms)
            raw_counts[resolution.status.value] += 1
            semantic_counts[str(semantic["semantic_resolution_status"])] += 1
            target_raw[resolution.status.value] += 1
            target_semantic[str(semantic["semantic_resolution_status"])] += 1
            rows.append(
                {
                    "schema_version": LEDGER_SCHEMA_VERSION,
                    "target_id": target.target_id,
                    "role": target.role,
                    "target_smiles": target.canonical_smiles,
                    "target_occurrence_count": target.occurrence_count,
                    "rank": proposal.rank,
                    "model_score": proposal.model_score,
                    "reactants": list(proposal.reactant_smiles),
                    "proposal_sha256": proposal.proposal_sha256,
                    "raw_discovery_resolution": resolution.to_dict(),
                    "semantic_equivalence": semantic,
                    "graph_consistent_discovery_hypothesis": semantic[
                        "graph_consistent_discovery_hypothesis"
                    ],
                    "route_closure_authorized": False,
                    "evidence_created": False,
                    "may_enter_synthesis_value": False,
                }
            )
        target_summaries.append(
            {
                "target_id": target.target_id,
                "role": target.role,
                "target_smiles": target.canonical_smiles,
                "occurrence_count": target.occurrence_count,
                "phase_counts": dict(target.phase_counts),
                "proposal_count": len(repeats[0].proposals),
                "deterministic_across_repeats": True,
                "raw_status_counts": dict(sorted(target_raw.items())),
                "semantic_status_counts": dict(sorted(target_semantic.items())),
            }
        )
    graph_consistent = {
        row["target_id"] for row in rows if row["graph_consistent_discovery_hypothesis"]
    }
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "checkpoint_component_proposal_diagnostic_complete",
        "input": {
            "path": str(support_path),
            "sha256": _sha256_file(support_path),
        },
        "population": {
            "arm": arm,
            "checkpoint_phase": checkpoint_phase,
            "terminal_phase": terminal_phase,
            "checkpoints": sorted({int(value) for value in checkpoints}),
        },
        "settings": {
            "maximum_proposals_per_target": maximum_proposals,
            "repeat_count": repeat_count,
            "operational_screen_run": False,
        },
        "summary": {
            "unique_unresolved_targets": len(targets),
            "proposal_count": len(rows),
            "targets_with_graph_consistent_discovery_hypothesis": len(graph_consistent),
            "raw_status_counts": dict(sorted(raw_counts.items())),
            "semantic_status_counts": dict(sorted(semantic_counts.items())),
        },
        "targets": target_summaries,
        "scientific_authority": {
            "proposal_only": True,
            "proposal_model_score_used": False,
            "reaction_family_filter_applied_before_proposal": False,
            "may_create_route_evidence": False,
            "may_close_route": False,
            "may_enter_synthesis_value": False,
            "may_influence_generation_without_independent_adjudication": False,
        },
        "next_gate": (
            "Independently execute proposal-derived steps, establish step evidence, "
            "and close every terminal material before recomputing route-value contrast."
        ),
    }
    return content, rows


def finalize_checkpoint_component_proposals(
    content: Mapping[str, Any], *, ledger_sha256: str, row_count: int
) -> dict[str, Any]:
    """Attach write-once artifact metadata before hashing the result."""

    result = {
        **dict(content),
        "artifacts": {
            "proposal_ledger.jsonl.gz": {
                "path": "proposal_ledger.jsonl.gz",
                "sha256": ledger_sha256,
                "rows": row_count,
                "schema_version": LEDGER_SCHEMA_VERSION,
            }
        },
    }
    result["result_sha256"] = _sha256_payload(result)
    return result


__all__ = [
    "CheckpointComponentTarget",
    "CheckpointContrastProposalError",
    "LEDGER_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "extract_checkpoint_component_targets",
    "finalize_checkpoint_component_proposals",
    "run_checkpoint_component_proposals",
]
