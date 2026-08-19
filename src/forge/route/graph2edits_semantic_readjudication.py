"""Re-adjudicate frozen Graph2Edits receipts without rerunning the model."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from forge.route.l2_forward_resolver import L2ForwardResolutionStatus
from forge.route.proposal_discovery_status import (
    FamilyProjectionTrace,
    ProposalDiscoveryResolution,
    ProposalDiscoveryStatus,
)
from forge.route.semantic_family_equivalence import (
    FamilyTransform,
    audit_semantic_family_equivalence_v2,
)

RESULT_SCHEMA_VERSION = "forge.graph2edits_semantic_readjudication.v1"
LEDGER_SCHEMA_VERSION = "forge.graph2edits_semantic_readjudication_ledger.v1"


class Graph2EditsSemanticReadjudicationError(RuntimeError):
    """Raised when a frozen proposal receipt cannot be re-adjudicated."""


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


def _read_jsonl_gzip(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise Graph2EditsSemanticReadjudicationError("proposal ledger is malformed")
    return rows


def _resolution(value: Any) -> ProposalDiscoveryResolution:
    if not isinstance(value, dict):
        raise Graph2EditsSemanticReadjudicationError("raw discovery receipt is missing")
    try:
        traces = tuple(
            FamilyProjectionTrace(
                reaction_id=str(item["reaction_id"]),
                transform_sha256=str(item["transform_sha256"]),
                role_ordered_reactants=tuple(item["role_ordered_reactants"]),
                products=tuple(item["products"]),
                target_reconstructed=bool(item["target_reconstructed"]),
            )
            for item in value["traces"]
        )
        return ProposalDiscoveryResolution(
            proposal_sha256=str(value["proposal_sha256"]),
            target_smiles=str(value["target_smiles"]),
            status=ProposalDiscoveryStatus(value["status"]),
            exact_resolution_status=L2ForwardResolutionStatus(value["exact_resolution_status"]),
            exact_resolver_config_sha256=str(value["exact_resolver_config_sha256"]),
            exact_forward_calls=int(value["exact_forward_calls"]),
            family_projection_calls=int(value["family_projection_calls"]),
            maximum_family_projection_calls=int(value["maximum_family_projection_calls"]),
            traces=traces,
            accepted_trace_index=value.get("accepted_trace_index"),
            rejection_reason=value.get("rejection_reason"),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise Graph2EditsSemanticReadjudicationError(
            "raw discovery receipt cannot be reconstructed"
        ) from error


def build_semantic_readjudication(
    *, proposal_ledger_path: Path, transforms: Sequence[FamilyTransform]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Recompute semantic status from frozen raw discovery traces."""

    source_rows = _read_jsonl_gzip(proposal_ledger_path)
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    targets: set[str] = set()
    for source in source_rows:
        resolution = _resolution(source.get("raw_discovery_resolution"))
        audit = audit_semantic_family_equivalence_v2(resolution, transforms)
        status = str(audit["semantic_resolution_status"])
        counts[status] += 1
        if audit["graph_consistent_discovery_hypothesis"]:
            targets.add(str(source["target_id"]))
        rows.append(
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "target_id": source["target_id"],
                "role": source["role"],
                "target_smiles": source["target_smiles"],
                "target_occurrence_count": source["target_occurrence_count"],
                "rank": source["rank"],
                "model_score": source["model_score"],
                "reactants": source["reactants"],
                "proposal_sha256": source["proposal_sha256"],
                "semantic_equivalence": audit,
                "graph_consistent_discovery_hypothesis": audit[
                    "graph_consistent_discovery_hypothesis"
                ],
                "route_closure_authorized": False,
                "evidence_created": False,
                "may_enter_synthesis_value": False,
            }
        )
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "frozen_proposal_semantic_readjudication_complete",
        "input": {
            "path": str(proposal_ledger_path),
            "sha256": _sha256_file(proposal_ledger_path),
            "rows": len(source_rows),
        },
        "summary": {
            "proposal_count": len(rows),
            "semantic_status_counts": dict(sorted(counts.items())),
            "targets_with_graph_consistent_discovery_hypothesis": len(targets),
        },
        "scientific_authority": {
            "proposal_model_rerun": False,
            "semantic_duplicate_collapse_is_evidence": False,
            "may_create_route_evidence": False,
            "may_close_route": False,
            "may_enter_synthesis_value": False,
        },
    }
    return content, rows


def finalize_semantic_readjudication(
    content: dict[str, Any], *, ledger_sha256: str, row_count: int
) -> dict[str, Any]:
    result = {
        **content,
        "artifacts": {
            "semantic_ledger.jsonl.gz": {
                "path": "semantic_ledger.jsonl.gz",
                "sha256": ledger_sha256,
                "rows": row_count,
                "schema_version": LEDGER_SCHEMA_VERSION,
            }
        },
    }
    result["result_sha256"] = _sha256_payload(result)
    return result


__all__ = [
    "Graph2EditsSemanticReadjudicationError",
    "LEDGER_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "build_semantic_readjudication",
    "finalize_semantic_readjudication",
]
