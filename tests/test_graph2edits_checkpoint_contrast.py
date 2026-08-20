from __future__ import annotations

import hashlib
from types import SimpleNamespace

from experiments.archive.phase1.synthesis_audits.graph2edits_checkpoint_contrast import (
    extract_checkpoint_component_targets,
    run_checkpoint_component_proposals,
)
from forge.synthesis.assessment.l2_forward_resolver import L2ForwardResolutionStatus
from forge.synthesis.assessment.proposal_discovery_status import (
    SourceNeutralProposalDiscoveryResolver,
)
from forge.synthesis.engine.proposal_engine import (
    ProposalBackendManifest,
    SingleStepRetrosynthesisProposal,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _record(*, phase: str, checkpoint: int = 2, particle: int = 0) -> dict:
    components = [
        {
            "role": "amine_head",
            "canonical_smiles": "CN",
            "graded_evidence_class": "exact_complete_current",
        },
        {
            "role": "oxoester_aldehyde_body_tail",
            "canonical_smiles": "CCCC=O",
            "graded_evidence_class": "missing_knowledge",
        },
        {
            "role": "isocyanide_tail",
            "canonical_smiles": "[C-]#[N+]CC",
            "graded_evidence_class": "exact_complete_current",
        },
    ]
    roots = [
        {
            "role": component["role"],
            "canonical_smiles": component["canonical_smiles"],
            "product_context_smiles": [],
        }
        for component in components
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
        "arm": "post_hoc",
        "assessment_phase": phase,
        "checkpoint": checkpoint,
        "program_index": 0,
        "particle_index": particle,
        "terminal_sha256": _hash(f"terminal-{phase}-{particle}"),
        "graded_route_readiness": {"components": components},
        "strict_assessment_receipt": {"payload": {"l1_reaction_sha256": _hash("l1")}},
        "support_audit": {
            "support_sha256": _hash("support"),
            "support": {
                "terminal_sha256": _hash(f"terminal-{phase}-{particle}"),
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
        del maximum_proposals
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
        del proposal, maximum_forward_calls
        return SimpleNamespace(
            status=L2ForwardResolutionStatus.EXACT_UNIQUE,
            forward_calls=1,
        )


def test_checkpoint_targets_are_unresolved_deduplicated_and_include_finals() -> None:
    payload = {
        "records": [
            _record(phase="checkpoint_shadow", particle=0),
            _record(phase="checkpoint_shadow", particle=1),
            _record(phase="productive_final", particle=2),
        ]
    }
    targets = extract_checkpoint_component_targets(
        payload,
        arm="post_hoc",
        checkpoint_phase="checkpoint_shadow",
        terminal_phase="productive_final",
        checkpoints=(2, 4, 6),
    )
    assert len(targets) == 1
    assert targets[0].role == "oxoester_aldehyde_body_tail"
    assert targets[0].occurrence_count == 3
    assert dict(targets[0].phase_counts) == {"checkpoint_shadow": 2, "productive_final": 1}
    assert targets[0].representative_row["assessment_receipt"]["payload"][
        "l1_reaction_sha256"
    ] == _hash("l1")


def test_checkpoint_proposals_remain_non_authoritative(tmp_path) -> None:
    support_path = tmp_path / "support.json"
    support_path.write_text("{}")
    payload = {"records": [_record(phase="checkpoint_shadow")]}
    result, rows = run_checkpoint_component_proposals(
        support_payload=payload,
        support_path=support_path,
        backend=_Backend(),
        resolver=SourceNeutralProposalDiscoveryResolver(exact_resolver=_ExactResolver()),
        arm="post_hoc",
        checkpoint_phase="checkpoint_shadow",
        terminal_phase="productive_final",
        checkpoints=(2, 4, 6),
        maximum_proposals=3,
        repeat_count=2,
    )
    assert result["summary"]["unique_unresolved_targets"] == 1
    assert result["summary"]["targets_with_graph_consistent_discovery_hypothesis"] == 1
    assert len(rows) == 1
    assert rows[0]["route_closure_authorized"] is False
    assert result["scientific_authority"]["proposal_model_score_used"] is False
