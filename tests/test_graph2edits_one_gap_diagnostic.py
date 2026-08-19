from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from forge.route.graph2edits_one_gap_diagnostic import (
    build_request,
    extract_one_gap_targets,
    run_one_gap_diagnostic,
)
from forge.route.l2_forward_resolver import L2ForwardResolutionStatus
from forge.route.proposal_discovery_status import (
    ProposalDiscoveryStatus,
    SourceNeutralProposalDiscoveryResolver,
)
from forge.route.proposal_engine import ProposalBackendManifest, SingleStepRetrosynthesisProposal


def _hash(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode()).hexdigest()


def _row(*, arm: str = "broad_prior") -> dict:
    components = {
        "amine_head": ("CN", "complete"),
        "oxoester_aldehyde_body_tail": ("CCCC=O", "missing_knowledge"),
        "isocyanide_tail": ("[C-]#[N+]CC", "complete"),
    }
    roots = [
        {"role": role, "canonical_smiles": smiles, "product_context_smiles": []}
        for role, (smiles, _) in components.items()
    ]
    qualifications = [
        {
            "exact_l1_eligible": True,
            "supported_ugi_role": True,
            "role_handle_qualified": True,
            "molecular_support_state": "within_declared_support",
            "declared_exclusion_code": None,
            "declared_exclusion_policy_locator": None,
        }
        for _ in roots
    ]
    return {
        "arm_id": arm,
        "draw_index": 1,
        "terminal_sha256": _hash(f"terminal-{arm}"),
        "product_value": {
            "components": [
                {
                    "role": role,
                    "value": {
                        "assessment_outcome": outcome,
                        "target": roots[index],
                    },
                }
                for index, (role, (_, outcome)) in enumerate(components.items())
            ]
        },
        "assessment_receipt": {"payload": {"l1_reaction_sha256": _hash("l1")}},
        "support_audit": {
            "support_sha256": _hash(f"support-{arm}"),
            "support": {
                "terminal_sha256": _hash(f"terminal-{arm}"),
                "generator_checkpoint_sha256": _hash("generator"),
                "root_targets": roots,
                "root_qualifications": qualifications,
            },
        },
    }


class _Backend:
    manifest = ProposalBackendManifest(
        backend_id="fixture",
        implementation_version="1",
        checkpoint_sha256=_hash("checkpoint"),
        checkpoint_license="fixture",
        training_corpus_id="fixture",
        training_corpus_snapshot_sha256=_hash("corpus"),
        source_locator="fixture",
    )

    def propose_with_trace(self, request, *, maximum_proposals):
        proposal = SingleStepRetrosynthesisProposal(
            proposal_id="fixture-proposal",
            backend=self.manifest,
            request=request,
            reactant_smiles=("CCCCO",),
            rank=1,
            model_score=0.9,
        )
        return SimpleNamespace(proposals=(proposal,))


class _ExactResolver:
    transforms = ()
    resolver_config_sha256 = _hash("resolver")

    def resolve(self, proposal, *, maximum_forward_calls=None):
        return SimpleNamespace(
            status=L2ForwardResolutionStatus.EXACT_UNIQUE,
            forward_calls=1,
        )


def test_extract_and_build_one_gap_request() -> None:
    targets = extract_one_gap_targets([_row(), _row(arm="support_enriched")])
    assert len(targets) == 1
    assert targets[0].occurrence_count == 2
    assert dict(targets[0].arm_counts) == {"broad_prior": 1, "support_enriched": 1}
    request = build_request(targets[0])
    assert request.target.canonical_smiles == "CCCC=O"
    assert request.root_qualification.qualification.role_handle_qualified


def test_run_one_gap_diagnostic_is_proposal_only(tmp_path: Path) -> None:
    ledger = tmp_path / "route.jsonl.gz"
    with gzip.open(ledger, "wt") as handle:
        handle.write(json.dumps(_row()) + "\n")
    result, rows = run_one_gap_diagnostic(
        route_ledger_path=ledger,
        backend=_Backend(),
        resolver=SourceNeutralProposalDiscoveryResolver(exact_resolver=_ExactResolver()),
        maximum_proposals=3,
        repeat_count=2,
    )
    assert result["summary"]["targets_with_graph_consistent_proposal"] == 1
    assert result["summary"]["status_counts"] == {
        ProposalDiscoveryStatus.EXACT_KNOWN_ROUTE.value: 1
    }
    assert len(rows) == 1
    assert rows[0]["route_closure_authorized"] is False
    assert rows[0]["evidence_created"] is False
