from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from importlib import metadata
from pathlib import Path

import pytest

from forge.synthesis.assessment.ugi3_support_boundary import (
    MolecularSupportState,
    TargetQualification,
)
from forge.synthesis.engine.graph2edits_backend import (
    GRAPH2EDITS_BACKEND_ID,
    GRAPH2EDITS_CHECKPOINT_LICENSE,
    GRAPH2EDITS_IMPLEMENTATION_VERSION,
    GRAPH2EDITS_SOURCE_COMMIT,
    GRAPH2EDITS_SOURCE_LOCATOR,
    GRAPH2EDITS_TRAINING_CORPUS_ID,
    Graph2EditsBackendError,
    Graph2EditsDependencyError,
    Graph2EditsInferencePolicy,
    Graph2EditsLocalArtifacts,
    Graph2EditsProposalBackend,
    Graph2EditsRawPrediction,
    build_syntheseus_worker,
)
from forge.synthesis.engine.planner import RouteTarget
from forge.synthesis.engine.proposal_engine import (
    ProposalBackendManifest,
    ProposalRequest,
    ProposalTargetKind,
    RootQualificationReceipt,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash(label: str) -> str:
    return _sha256_bytes(label.encode())


def _manifest(*, checkpoint_sha256: str, corpus_sha256: str) -> ProposalBackendManifest:
    return ProposalBackendManifest(
        backend_id=GRAPH2EDITS_BACKEND_ID,
        implementation_version=GRAPH2EDITS_IMPLEMENTATION_VERSION,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_license=GRAPH2EDITS_CHECKPOINT_LICENSE,
        training_corpus_id=GRAPH2EDITS_TRAINING_CORPUS_ID,
        training_corpus_snapshot_sha256=corpus_sha256,
        source_locator=GRAPH2EDITS_SOURCE_LOCATOR,
    )


def _artifacts(tmp_path: Path) -> tuple[Graph2EditsLocalArtifacts, ProposalBackendManifest]:
    checkpoint = tmp_path / "epoch_123.pt"
    corpus_manifest = tmp_path / "corpus_manifest.json"
    checkpoint_bytes = b"frozen graph2edits checkpoint test fixture"
    checkpoint.write_bytes(checkpoint_bytes)
    split_receipts = []
    for representation in ("raw", "canonicalized"):
        for split in ("train", "valid", "test"):
            relative_path = Path("corpus") / f"{representation}_{split}.csv"
            path = tmp_path / relative_path
            path.parent.mkdir(exist_ok=True)
            content = f"reaction\n{representation}-{split}\n".encode()
            path.write_bytes(content)
            split_receipts.append(
                {
                    "representation": representation,
                    "split": split,
                    "path": relative_path.as_posix(),
                    "size_bytes": len(content),
                    "row_count": 1,
                    "sha256": _sha256_bytes(content),
                }
            )
    selection_log = tmp_path / "corpus" / "logs.csv"
    selection_bytes = b"epoch,loss\n123,0.1\n"
    selection_log.write_bytes(selection_bytes)
    corpus_payload = {
        "schema_version": "forge.graph2edits_training_corpus_manifest.v1",
        "corpus_id": GRAPH2EDITS_TRAINING_CORPUS_ID,
        "source": {"commit": GRAPH2EDITS_SOURCE_COMMIT},
        "splits": split_receipts,
        "checkpoint_selection_receipt": {
            "path": "corpus/logs.csv",
            "size_bytes": len(selection_bytes),
            "sha256": _sha256_bytes(selection_bytes),
        },
    }
    corpus_bytes = json.dumps(
        corpus_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    corpus_manifest.write_bytes(corpus_bytes)
    return (
        Graph2EditsLocalArtifacts(
            checkpoint_path=checkpoint,
            training_corpus_manifest_path=corpus_manifest,
            corpus_path_root=tmp_path,
        ),
        _manifest(
            checkpoint_sha256=_sha256_bytes(checkpoint_bytes),
            corpus_sha256=_sha256_bytes(corpus_bytes),
        ),
    )


def _request() -> ProposalRequest:
    root = RouteTarget(role="oxoester_aldehyde_body_tail", canonical_smiles="CCCC=O")
    receipt = RootQualificationReceipt(
        route_root=root,
        qualification=TargetQualification(
            exact_l1_eligible=True,
            supported_ugi_role=True,
            role_handle_qualified=True,
            molecular_support_state=MolecularSupportState.WITHIN_DECLARED_SUPPORT,
        ),
        terminal_sha256=_hash("terminal"),
        generator_checkpoint_sha256=_hash("generator"),
        l1_reaction_sha256=_hash("l1"),
        qualification_artifact_sha256=_hash("qualification"),
    )
    return ProposalRequest(
        route_root=root,
        target=root,
        depth=0,
        target_kind=ProposalTargetKind.ROOT,
        root_qualification=receipt,
        operational_policy_id="forge-lipid-component-operational-v1",
        operational_policy_sha256=_hash("operational-policy"),
    )


class RecordingWorker:
    def __init__(self, predictions: list[Graph2EditsRawPrediction]) -> None:
        self.predictions = predictions
        self.calls: list[tuple[str, int]] = []

    def __call__(
        self,
        target_smiles: str,
        *,
        num_results: int,
    ) -> list[Graph2EditsRawPrediction]:
        self.calls.append((target_smiles, num_results))
        return self.predictions[:num_results]


def test_manifest_and_local_artifacts_are_verified_before_worker_use(tmp_path: Path) -> None:
    artifacts, manifest = _artifacts(tmp_path)
    worker = RecordingWorker([])

    backend = Graph2EditsProposalBackend(
        manifest=manifest,
        artifacts=artifacts,
        worker=worker,
    )
    assert backend.manifest is manifest

    with pytest.raises(Graph2EditsBackendError, match="backend_id"):
        Graph2EditsProposalBackend(
            manifest=replace(manifest, backend_id="different-model"),
            artifacts=artifacts,
            worker=worker,
        )
    with pytest.raises(Graph2EditsBackendError, match="checkpoint SHA-256"):
        Graph2EditsProposalBackend(
            manifest=replace(manifest, checkpoint_sha256=_hash("wrong-checkpoint")),
            artifacts=artifacts,
            worker=worker,
        )
    with pytest.raises(Graph2EditsBackendError, match="corpus-manifest SHA-256"):
        Graph2EditsProposalBackend(
            manifest=replace(
                manifest,
                training_corpus_snapshot_sha256=_hash("wrong-corpus"),
            ),
            artifacts=artifacts,
            worker=worker,
        )

    assert worker.calls == []

    (tmp_path / "corpus" / "raw_train.csv").write_text("mutated\n")
    with pytest.raises(Graph2EditsBackendError, match="byte count|SHA-256"):
        Graph2EditsProposalBackend(
            manifest=manifest,
            artifacts=artifacts,
            worker=worker,
        )


def test_normalization_dedup_reranking_multiplicity_and_bound_are_deterministic(
    tmp_path: Path,
) -> None:
    artifacts, manifest = _artifacts(tmp_path)
    worker = RecordingWorker(
        [
            Graph2EditsRawPrediction("[CH3:1][CH2:2][OH:3].O", 0.4),
            Graph2EditsRawPrediction("O.CCO", 0.9),
            Graph2EditsRawPrediction("CC.CC", 0.8),
            Graph2EditsRawPrediction("not-smiles", 100.0),
            Graph2EditsRawPrediction("CCBr", None),
        ]
    )
    policy = Graph2EditsInferencePolicy(oversampling_factor=3)
    backend = Graph2EditsProposalBackend(
        manifest=manifest,
        artifacts=artifacts,
        worker=worker,
        policy=policy,
    )
    request = _request()

    first_batch = backend.propose_with_trace(request, maximum_proposals=2)
    first = first_batch.proposals
    second = backend.propose(request, maximum_proposals=2)

    assert worker.calls == [("CCCC=O", 6), ("CCCC=O", 6)]
    assert first == second
    assert tuple(proposal.rank for proposal in first) == (1, 2)
    assert first[0].reactant_smiles == ("CCO", "O")
    assert first[0].model_score == pytest.approx(0.9)
    assert first[1].reactant_smiles == ("CC", "CC")
    assert first[1].model_score == pytest.approx(0.8)
    assert first[0].proposal_id == second[0].proposal_id
    assert len(first) == 2
    assert tuple(record.worker_rank for record in first_batch.trace) == (2, 3)
    assert tuple(record.normalized_rank for record in first_batch.trace) == (1, 2)
    assert all(
        record.to_dict()["reaction_family_filter_applied"] is False for record in first_batch.trace
    )
    assert all(
        record.to_dict()["lipid_precedent_state"] == "unassessed" for record in first_batch.trace
    )
    assert all(record.to_dict()["evidence_tier"] is None for record in first_batch.trace)


def test_scores_are_opaque_and_proposals_have_zero_evidence_authority(tmp_path: Path) -> None:
    artifacts, manifest = _artifacts(tmp_path)
    backend = Graph2EditsProposalBackend(
        manifest=manifest,
        artifacts=artifacts,
        worker=RecordingWorker([Graph2EditsRawPrediction("CCO", 0.999)]),
    )

    [proposal] = backend.propose(_request(), maximum_proposals=1)
    payload = proposal.to_dict()

    assert proposal.model_score == pytest.approx(0.999)
    assert proposal.predicted_reaction_class is None
    assert payload["success_probability"] is None
    assert payload["evidence_tier"] is None
    assert payload["route_closure_authorized"] is False
    assert not hasattr(proposal, "evidence")


def test_worker_schema_is_fail_closed_but_invalid_chemistry_is_filtered(tmp_path: Path) -> None:
    artifacts, manifest = _artifacts(tmp_path)

    def malformed_worker(target_smiles: str, *, num_results: int):
        del target_smiles, num_results
        return [object()]

    malformed = Graph2EditsProposalBackend(
        manifest=manifest,
        artifacts=artifacts,
        worker=malformed_worker,
    )
    with pytest.raises(Graph2EditsBackendError, match="malformed prediction"):
        malformed.propose(_request(), maximum_proposals=1)

    invalid = Graph2EditsProposalBackend(
        manifest=manifest,
        artifacts=artifacts,
        worker=RecordingWorker(
            [
                Graph2EditsRawPrediction("not-smiles", 1.0),
                Graph2EditsRawPrediction("CC..O", 0.9),
            ]
        ),
    )
    assert invalid.propose(_request(), maximum_proposals=2) == ()

    with pytest.raises(Graph2EditsBackendError, match="finite"):
        Graph2EditsRawPrediction("CCO", float("nan"))


def test_worker_cannot_exceed_requested_bound(tmp_path: Path) -> None:
    artifacts, manifest = _artifacts(tmp_path)

    def unbounded_worker(target_smiles: str, *, num_results: int):
        del target_smiles
        return tuple(Graph2EditsRawPrediction("CCO", 1.0) for _ in range(num_results + 1))

    backend = Graph2EditsProposalBackend(
        manifest=manifest,
        artifacts=artifacts,
        worker=unbounded_worker,
    )
    with pytest.raises(Graph2EditsBackendError, match="exceeded"):
        backend.propose(_request(), maximum_proposals=1)


def test_optional_runtime_fails_clearly_when_dependency_is_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_version(distribution_name: str) -> str:
        raise metadata.PackageNotFoundError(distribution_name)

    monkeypatch.setattr(
        "forge.synthesis.engine.graph2edits_backend.metadata.version", missing_version
    )

    with pytest.raises(Graph2EditsDependencyError, match="optional dependency syntheseus==0.8.0"):
        build_syntheseus_worker(model_dir=tmp_path)
