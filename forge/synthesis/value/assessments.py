"""Small constructors for explicit, non-imputed synthesis assessment states."""

from __future__ import annotations

from pathlib import Path

from forge.core.hashing import sha256_bytes, sha256_file
from forge.synthesis.engine.planner import (
    AssessmentOutcome,
    AssessmentTraceEvent,
    AvailabilityState,
    EvidenceRecord,
    EvidenceTier,
    ForwardVerificationState,
    RouteTarget,
    SynthesisAssessment,
    SynthesisRouteNode,
    TraceAction,
)


def missing_generated_assessment(
    *,
    target: RouteTarget,
    provenance_ledger_path: Path,
) -> SynthesisAssessment:
    """Represent an admitted generated component with no exact route evidence.

    This state is deliberately ``missing_knowledge``. It must never be converted into chemical
    incompatibility or an imputed synthesis-success score.
    """

    evidence = EvidenceRecord(
        evidence_id=(
            "generated-component-missing-route:"
            f"{target.role}:{sha256_bytes(target.canonical_smiles.encode())[:16]}"
        ),
        tier=EvidenceTier.PROVENANCE_ONLY,
        source_sha256=sha256_file(provenance_ledger_path),
        source_locator=(
            f"{provenance_ledger_path}#role={target.role}"
            f"&canonical_component_smiles={target.canonical_smiles}"
        ),
        exact_substrate=True,
        forward_verification=ForwardVerificationState.NOT_RUN,
        availability=AvailabilityState.UNASSESSED,
    )
    detail = "generated component is structurally admitted but lacks an exact route assessment"
    node = SynthesisRouteNode(
        target=target,
        outcome=AssessmentOutcome.MISSING_KNOWLEDGE,
        evidence=(evidence,),
        detail=detail,
    )
    trace = (
        AssessmentTraceEvent(
            sequence=0,
            depth=0,
            action=TraceAction.DECISION,
            target=target,
            outcome=AssessmentOutcome.MISSING_KNOWLEDGE,
            evidence_ids=(evidence.evidence_id,),
            detail=detail,
        ),
    )
    return SynthesisAssessment(
        target=target,
        outcome=AssessmentOutcome.MISSING_KNOWLEDGE,
        route_tree=node,
        trace=trace,
    )


__all__ = ["missing_generated_assessment"]
