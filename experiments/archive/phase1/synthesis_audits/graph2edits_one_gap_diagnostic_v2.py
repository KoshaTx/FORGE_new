"""Graph2Edits one-gap diagnostic with duplicate-family adjudication.

Version 2 preserves the proposal-only authority of version 1, records complete
source-neutral discovery receipts, distinguishes duplicate registry provenance
from genuinely different reaction-family assignments, and computes its final
result hash only after artifact metadata is attached.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_json as _sha256_payload
from forge.synthesis.assessment.proposal_discovery_status import SourceNeutralProposalDiscoveryResolver
from experiments.archive.phase1.synthesis_audits.graph2edits_one_gap_diagnostic import (
    ProposalBackend,
    _load_jsonl_gz,
    _sha256_file,
    build_request,
    extract_one_gap_targets,
)
from forge.synthesis.evidence.semantic_family_equivalence import audit_semantic_family_equivalence

RESULT_SCHEMA_VERSION = "forge.graph2edits_one_gap_diagnostic.v2"
LEDGER_SCHEMA_VERSION = "forge.graph2edits_one_gap_proposal_ledger.v2"


class OneGapDiagnosticV2Error(RuntimeError):
    """Raised when the version-2 diagnostic fails closed."""


def run_one_gap_diagnostic_v2(
    *,
    route_ledger_path: Path,
    backend: ProposalBackend,
    resolver: SourceNeutralProposalDiscoveryResolver,
    maximum_proposals: int = 10,
    repeat_count: int = 2,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run proposal discovery while preserving raw and semantic statuses."""

    if maximum_proposals < 1 or repeat_count < 1:
        raise OneGapDiagnosticV2Error("proposal and repeat counts must be positive")
    route_rows = _load_jsonl_gz(route_ledger_path)
    targets = extract_one_gap_targets(route_rows)
    proposal_rows: list[dict[str, Any]] = []
    target_summaries: list[dict[str, Any]] = []
    raw_counts: Counter[str] = Counter()
    semantic_counts: Counter[str] = Counter()
    for target in targets:
        request = build_request(target)
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
            raise OneGapDiagnosticV2Error(
                f"Graph2Edits output was not deterministic for {target.target_id}"
            )
        per_target_raw: Counter[str] = Counter()
        per_target_semantic: Counter[str] = Counter()
        for proposal in repeats[0].proposals:
            resolution = resolver.resolve(proposal)
            semantic = audit_semantic_family_equivalence(resolution, resolver.transforms)
            raw_counts[resolution.status.value] += 1
            semantic_counts[semantic["semantic_resolution_status"]] += 1
            per_target_raw[resolution.status.value] += 1
            per_target_semantic[semantic["semantic_resolution_status"]] += 1
            proposal_rows.append(
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
                "arm_counts": dict(target.arm_counts),
                "proposal_count": len(repeats[0].proposals),
                "deterministic_across_repeats": True,
                "raw_status_counts": dict(sorted(per_target_raw.items())),
                "semantic_status_counts": dict(sorted(per_target_semantic.items())),
            }
        )
    graph_consistent_targets = {
        row["target_id"] for row in proposal_rows if row["graph_consistent_discovery_hypothesis"]
    }
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "proposal_only_semantic_family_diagnostic_complete",
        "route_ledger": {
            "path": str(route_ledger_path),
            "sha256": _sha256_file(route_ledger_path),
            "row_count": len(route_rows),
        },
        "settings": {
            "maximum_proposals_per_target": maximum_proposals,
            "repeat_count": repeat_count,
            "operational_screen_run": False,
        },
        "summary": {
            "unique_one_gap_targets": len(targets),
            "proposal_count": len(proposal_rows),
            "targets_with_graph_consistent_discovery_hypothesis": len(graph_consistent_targets),
            "raw_status_counts": dict(sorted(raw_counts.items())),
            "semantic_status_counts": dict(sorted(semantic_counts.items())),
        },
        "targets": target_summaries,
        "scientific_authority": {
            "proposal_only": True,
            "semantic_duplicate_collapse_is_evidence": False,
            "may_create_route_evidence": False,
            "may_close_route": False,
            "may_enter_synthesis_value": False,
            "may_influence_generation_or_tilting_without_independent_adjudication": False,
        },
        "next_gate": (
            "Join proposal discovery to independently frozen family evidence and terminal-"
            "material closure; only the independent adjudication may define route readiness."
        ),
    }
    return content, proposal_rows


def finalize_one_gap_diagnostic_v2(
    content: dict[str, Any], *, ledger_sha256: str, row_count: int
) -> dict[str, Any]:
    """Attach artifact metadata before computing the immutable result hash."""

    result = {
        **content,
        "artifacts": {
            "proposal_ledger.jsonl.gz": {
                "path": "proposal_ledger.jsonl.gz",
                "sha256": ledger_sha256,
                "row_count": row_count,
                "schema_version": LEDGER_SCHEMA_VERSION,
            }
        },
    }
    result["result_sha256"] = _sha256_payload(result)
    return result


__all__ = [
    "LEDGER_SCHEMA_VERSION",
    "OneGapDiagnosticV2Error",
    "RESULT_SCHEMA_VERSION",
    "finalize_one_gap_diagnostic_v2",
    "run_one_gap_diagnostic_v2",
]
