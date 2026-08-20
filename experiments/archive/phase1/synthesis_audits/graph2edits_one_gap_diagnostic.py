"""Proposal-only Graph2Edits diagnostic for frozen one-gap Ugi components.

This module deliberately stops before route evidence.  It extracts components
for which the frozen route census closed exactly two of the three Ugi roots,
asks a learned backend for bounded single-step hypotheses, and sends every
hypothesis through the independent source-neutral forward resolver.  A
forward-consistent family projection is still a hypothesis, never route
closure or synthesis value.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from forge.core.hashing import sha256_file as _sha256_file
from forge.core.hashing import sha256_json as _sha256_payload
from forge.synthesis.assessment.proposal_discovery_status import SourceNeutralProposalDiscoveryResolver
from forge.synthesis.assessment.ugi3_support_boundary import MolecularSupportState, TargetQualification
from forge.synthesis.engine.planner import RouteTarget
from forge.synthesis.engine.proposal_engine import (
    ProposalRequest,
    ProposalTargetKind,
    RootQualificationReceipt,
)

RESULT_SCHEMA_VERSION = "forge.graph2edits_one_gap_diagnostic.v1"
LEDGER_SCHEMA_VERSION = "forge.graph2edits_one_gap_proposal_ledger.v1"
OPERATIONAL_POLICY_ID = "forge.proposal_only_one_gap_diagnostic.v1"


class OneGapDiagnosticError(ValueError):
    """Raised when the frozen input or proposal-only result is malformed."""


class ProposalBackend(Protocol):
    def propose_with_trace(
        self,
        request: ProposalRequest,
        *,
        maximum_proposals: int,
    ) -> Any: ...


def _load_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    try:
        with gzip.open(path, "rt") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OneGapDiagnosticError("route ledger is not valid gzip JSONL") from error
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise OneGapDiagnosticError("route ledger must contain JSON objects")
    return rows


def _target_key(target: Mapping[str, Any]) -> tuple[str, str]:
    role = target.get("role")
    smiles = target.get("canonical_smiles")
    if not isinstance(role, str) or not role or not isinstance(smiles, str) or not smiles:
        raise OneGapDiagnosticError("component target is malformed")
    return role, smiles


@dataclass(frozen=True)
class OneGapTarget:
    role: str
    canonical_smiles: str
    occurrence_count: int
    arm_counts: tuple[tuple[str, int], ...]
    representative_row: Mapping[str, Any]

    @property
    def target_id(self) -> str:
        return f"one-gap:{_sha256_payload([self.role, self.canonical_smiles])[:20]}"


def extract_one_gap_targets(rows: Iterable[Mapping[str, Any]]) -> tuple[OneGapTarget, ...]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        product_value = row.get("product_value")
        if not isinstance(product_value, Mapping):
            raise OneGapDiagnosticError("route row has no product synthesis value")
        components = product_value.get("components")
        if not isinstance(components, list) or len(components) != 3:
            raise OneGapDiagnosticError("route row must contain three Ugi components")
        missing = [
            component
            for component in components
            if isinstance(component, Mapping)
            and isinstance(component.get("value"), Mapping)
            and component["value"].get("assessment_outcome") != "complete"
        ]
        if len(missing) != 1:
            continue
        value = missing[0]["value"]
        target = value.get("target")
        if not isinstance(target, Mapping):
            raise OneGapDiagnosticError("missing component has no typed route target")
        key = _target_key(target)
        grouped.setdefault(key, []).append(row)

    targets: list[OneGapTarget] = []
    for (role, smiles), target_rows in grouped.items():
        arms = Counter(str(row.get("arm_id")) for row in target_rows)
        representative = min(
            target_rows,
            key=lambda row: (
                str(row.get("arm_id")),
                int(row.get("draw_index", 0)),
                str(row.get("terminal_sha256")),
            ),
        )
        targets.append(
            OneGapTarget(
                role=role,
                canonical_smiles=smiles,
                occurrence_count=len(target_rows),
                arm_counts=tuple(sorted(arms.items())),
                representative_row=representative,
            )
        )
    return tuple(sorted(targets, key=lambda item: (item.role, item.canonical_smiles)))


def _qualification_from_dict(value: Mapping[str, Any]) -> TargetQualification:
    try:
        state = MolecularSupportState(value["molecular_support_state"])
        return TargetQualification(
            exact_l1_eligible=value["exact_l1_eligible"],
            supported_ugi_role=value["supported_ugi_role"],
            role_handle_qualified=value["role_handle_qualified"],
            molecular_support_state=state,
            declared_exclusion_code=value.get("declared_exclusion_code"),
            declared_exclusion_policy_locator=value.get("declared_exclusion_policy_locator"),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise OneGapDiagnosticError("root qualification is malformed") from error


def build_request(target: OneGapTarget) -> ProposalRequest:
    row = target.representative_row
    audit = row.get("support_audit")
    if not isinstance(audit, Mapping):
        raise OneGapDiagnosticError("one-gap row has no support audit")
    support = audit.get("support")
    if not isinstance(support, Mapping):
        raise OneGapDiagnosticError("one-gap support receipt is missing")
    roots = support.get("root_targets")
    qualifications = support.get("root_qualifications")
    if not isinstance(roots, list) or not isinstance(qualifications, list):
        raise OneGapDiagnosticError("one-gap root qualifications are missing")
    matches = [
        (root, qualification)
        for root, qualification in zip(roots, qualifications, strict=True)
        if isinstance(root, Mapping)
        and isinstance(qualification, Mapping)
        and _target_key(root) == (target.role, target.canonical_smiles)
    ]
    if len(matches) != 1:
        raise OneGapDiagnosticError("one-gap target does not match exactly one qualified root")
    root_dict, qualification_dict = matches[0]
    route_root = RouteTarget(
        role=target.role,
        canonical_smiles=target.canonical_smiles,
        product_context_smiles=tuple(root_dict.get("product_context_smiles", ())),
    )
    receipt = RootQualificationReceipt(
        route_root=route_root,
        qualification=_qualification_from_dict(qualification_dict),
        terminal_sha256=support["terminal_sha256"],
        generator_checkpoint_sha256=support["generator_checkpoint_sha256"],
        l1_reaction_sha256=row["assessment_receipt"]["payload"]["l1_reaction_sha256"],
        qualification_artifact_sha256=audit["support_sha256"],
    )
    policy_sha256 = _sha256_payload(
        {
            "policy_id": OPERATIONAL_POLICY_ID,
            "proposal_only": True,
            "operational_screen_run": False,
            "route_closure_authorized": False,
        }
    )
    return ProposalRequest(
        route_root=route_root,
        target=route_root,
        depth=0,
        target_kind=ProposalTargetKind.ROOT,
        root_qualification=receipt,
        operational_policy_id=OPERATIONAL_POLICY_ID,
        operational_policy_sha256=policy_sha256,
    )


def run_one_gap_diagnostic(
    *,
    route_ledger_path: Path,
    backend: ProposalBackend,
    resolver: SourceNeutralProposalDiscoveryResolver,
    maximum_proposals: int = 10,
    repeat_count: int = 2,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if maximum_proposals < 1 or repeat_count < 1:
        raise OneGapDiagnosticError("proposal and repeat counts must be positive")
    route_rows = _load_jsonl_gz(route_ledger_path)
    targets = extract_one_gap_targets(route_rows)
    proposal_rows: list[dict[str, Any]] = []
    target_summaries: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
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
        deterministic = all(identity == identities[0] for identity in identities[1:])
        if not deterministic:
            raise OneGapDiagnosticError(
                f"Graph2Edits output was not deterministic for {target.target_id}"
            )
        per_target = Counter()
        for proposal in repeats[0].proposals:
            resolution = resolver.resolve(proposal)
            per_target[resolution.status.value] += 1
            status_counts[resolution.status.value] += 1
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
                    "discovery_status": resolution.status.value,
                    "exact_resolution_status": resolution.exact_resolution_status.value,
                    "forward_calls": resolution.forward_calls,
                    "graph_consistent": resolution.graph_consistent,
                    "retained_for_discovery": resolution.retained_for_discovery,
                    "rejection_reason": resolution.rejection_reason,
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
                "deterministic_across_repeats": deterministic,
                "status_counts": dict(sorted(per_target.items())),
            }
        )
    targets_with_graph_consistent = sum(
        any(
            row["target_id"] == target["target_id"] and row["graph_consistent"]
            for row in proposal_rows
        )
        for target in target_summaries
    )
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "proposal_only_diagnostic_complete",
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
            "targets_with_graph_consistent_proposal": targets_with_graph_consistent,
            "status_counts": dict(sorted(status_counts.items())),
        },
        "targets": target_summaries,
        "scientific_authority": {
            "proposal_only": True,
            "may_create_route_evidence": False,
            "may_close_route": False,
            "may_enter_synthesis_value": False,
            "may_influence_generation_or_tilting": False,
        },
        "next_gate": (
            "Independently verify graph-consistent proposals for substrate scope, precursor "
            "availability, operational compatibility and complete L2/L3 closure."
        ),
    }
    result = {**content, "result_sha256": _sha256_payload(content)}
    return result, proposal_rows


__all__ = [
    "LEDGER_SCHEMA_VERSION",
    "OneGapDiagnosticError",
    "OneGapTarget",
    "RESULT_SCHEMA_VERSION",
    "build_request",
    "extract_one_gap_targets",
    "run_one_gap_diagnostic",
]
