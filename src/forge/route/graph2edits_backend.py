"""Optional, proposal-only Graph2Edits adapter.

The adapter deliberately knows nothing about route evidence, terminal closure or
synthesis-success probability.  It converts one frozen local Graph2Edits beam
into :class:`SingleStepRetrosynthesisProposal` records and delegates all
scientific admission checks to the existing proposal-screening boundary.

The architectural direction is one-way and fail-closed::

    exact Ugi L1 decomposition
        -> Graph2Edits L2 hypothesis proposal
        -> independent forward/operational/evidence screening
        -> bounded planner assessment
        -> structured synthesis value

This module implements only the second box.  It does not import the product
sampler or value layer and cannot promote a hypothesis into evidence or route
closure.

No Graph2Edits or Syntheseus dependency is imported at module import time.  A
caller can inject a local worker directly, while ``build_syntheseus_worker`` is
an optional convenience boundary for a separately locked environment.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

from rdkit import Chem

from forge.route.proposal_engine import (
    PROPOSAL_REACTANT_IDENTITY_POLICY,
    ProposalBackendManifest,
    ProposalEngineError,
    ProposalRequest,
    SingleStepRetrosynthesisProposal,
    validate_proposal_batch,
)

GRAPH2EDITS_SOURCE_COMMIT = "b12e97391c378e46e79e617d80017e95b21bba65"
GRAPH2EDITS_BACKEND_ID = "graph2edits-uspto50k-reaction-class-unknown"
GRAPH2EDITS_IMPLEMENTATION_VERSION = (
    "syntheseus==0.8.0;syntheseus-graph2edits==0.2.0;" f"graph2edits@{GRAPH2EDITS_SOURCE_COMMIT}"
)
GRAPH2EDITS_CHECKPOINT_LICENSE = "MIT"
GRAPH2EDITS_TRAINING_CORPUS_ID = "graph2edits-official-uspto50k-canonicalized"
GRAPH2EDITS_SOURCE_LOCATOR = (
    "https://github.com/Jamson-Zhong/Graph2Edits/tree/" + GRAPH2EDITS_SOURCE_COMMIT
)
GRAPH2EDITS_SYNTHESEUS_VERSION = "0.8.0"
GRAPH2EDITS_PACKAGE_VERSION = "0.2.0"
GRAPH2EDITS_ADAPTER_SCHEMA_VERSION = "forge.graph2edits_proposal_backend.v1"


class Graph2EditsBackendError(ProposalEngineError):
    """Raised when local artifacts, worker output or policy are malformed."""


class Graph2EditsDependencyError(Graph2EditsBackendError):
    """Raised when the optional pinned runtime is absent or incompatible."""


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Graph2EditsLocalArtifacts:
    """Local files whose bytes instantiate the backend manifest.

    ``training_corpus_manifest_path`` is the canonical manifest over the exact
    train/validation/test files, not a directory name or DOI.  Its own SHA-256
    is the value stored in ``ProposalBackendManifest``.
    """

    checkpoint_path: Path
    training_corpus_manifest_path: Path
    corpus_path_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        object.__setattr__(
            self,
            "training_corpus_manifest_path",
            Path(self.training_corpus_manifest_path),
        )
        object.__setattr__(self, "corpus_path_root", Path(self.corpus_path_root))

    def verify(self, manifest: ProposalBackendManifest) -> None:
        for label, path in (
            ("checkpoint", self.checkpoint_path),
            ("training corpus manifest", self.training_corpus_manifest_path),
        ):
            if not path.is_file():
                raise Graph2EditsBackendError(f"Graph2Edits {label} is not a local file: {path}")

        checkpoint_sha256 = _file_sha256(self.checkpoint_path)
        if checkpoint_sha256 != manifest.checkpoint_sha256:
            raise Graph2EditsBackendError(
                "Graph2Edits checkpoint SHA-256 does not match the frozen backend manifest"
            )
        corpus_sha256 = _file_sha256(self.training_corpus_manifest_path)
        if corpus_sha256 != manifest.training_corpus_snapshot_sha256:
            raise Graph2EditsBackendError(
                "Graph2Edits corpus-manifest SHA-256 does not match the frozen backend manifest"
            )
        verify_graph2edits_training_corpus_receipt(
            manifest_path=self.training_corpus_manifest_path,
            corpus_path_root=self.corpus_path_root,
            expected_corpus_id=manifest.training_corpus_id,
        )


def _resolve_receipt_file(*, path_value: Any, corpus_path_root: Path, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise Graph2EditsBackendError(f"Graph2Edits {label} path must be nonempty")
    relative_path = Path(path_value)
    if relative_path.is_absolute():
        raise Graph2EditsBackendError(f"Graph2Edits {label} path must be relative")
    root = corpus_path_root.resolve()
    resolved = (root / relative_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise Graph2EditsBackendError(
            f"Graph2Edits {label} path escapes the declared corpus root"
        ) from error
    if not resolved.is_file():
        raise Graph2EditsBackendError(f"Graph2Edits {label} is not a local file: {resolved}")
    return resolved


def _verify_receipt_file(
    *,
    record: Any,
    corpus_path_root: Path,
    label: str,
) -> None:
    if not isinstance(record, dict):
        raise Graph2EditsBackendError(f"Graph2Edits {label} receipt is malformed")
    path = _resolve_receipt_file(
        path_value=record.get("path"),
        corpus_path_root=corpus_path_root,
        label=label,
    )
    expected_size = record.get("size_bytes")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 1:
        raise Graph2EditsBackendError(f"Graph2Edits {label} size_bytes is malformed")
    if path.stat().st_size != expected_size:
        raise Graph2EditsBackendError(f"Graph2Edits {label} byte count does not match its receipt")
    expected_sha256 = record.get("sha256")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise Graph2EditsBackendError(f"Graph2Edits {label} SHA-256 is malformed")
    if _file_sha256(path) != expected_sha256:
        raise Graph2EditsBackendError(f"Graph2Edits {label} SHA-256 does not match its receipt")


def verify_graph2edits_training_corpus_receipt(
    *,
    manifest_path: Path,
    corpus_path_root: Path,
    expected_corpus_id: str = GRAPH2EDITS_TRAINING_CORPUS_ID,
) -> None:
    """Verify every frozen USPTO-50K split named by the corpus receipt.

    Paths are resolved only beneath an explicit caller-owned root.  This makes
    the receipt portable while preventing a crafted manifest from reading
    arbitrary host files.
    """

    try:
        payload = json.loads(Path(manifest_path).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Graph2EditsBackendError("Graph2Edits corpus manifest is not valid JSON") from error
    if not isinstance(payload, dict):
        raise Graph2EditsBackendError("Graph2Edits corpus manifest must be a JSON object")
    if payload.get("schema_version") != "forge.graph2edits_training_corpus_manifest.v1":
        raise Graph2EditsBackendError("Graph2Edits corpus-manifest schema is unsupported")
    if payload.get("corpus_id") != expected_corpus_id:
        raise Graph2EditsBackendError("Graph2Edits corpus ID does not match the backend manifest")
    source = payload.get("source")
    if not isinstance(source, dict) or source.get("commit") != GRAPH2EDITS_SOURCE_COMMIT:
        raise Graph2EditsBackendError("Graph2Edits corpus source commit is not frozen")

    splits = payload.get("splits")
    if not isinstance(splits, list):
        raise Graph2EditsBackendError("Graph2Edits corpus split receipts are malformed")
    expected_keys = {
        (representation, split)
        for representation in ("raw", "canonicalized")
        for split in ("train", "valid", "test")
    }
    observed_keys: set[tuple[str, str]] = set()
    for index, record in enumerate(splits):
        if not isinstance(record, dict):
            raise Graph2EditsBackendError("Graph2Edits corpus split receipt is malformed")
        key = (record.get("representation"), record.get("split"))
        if key not in expected_keys or key in observed_keys:
            raise Graph2EditsBackendError("Graph2Edits corpus split identities are not exact")
        observed_keys.add(key)
        _verify_receipt_file(
            record=record,
            corpus_path_root=Path(corpus_path_root),
            label=f"corpus split {index}",
        )
    if observed_keys != expected_keys:
        raise Graph2EditsBackendError("Graph2Edits corpus receipt must contain all six splits")

    selection_receipt = payload.get("checkpoint_selection_receipt")
    if selection_receipt is not None:
        _verify_receipt_file(
            record=selection_receipt,
            corpus_path_root=Path(corpus_path_root),
            label="checkpoint-selection log",
        )


@dataclass(frozen=True)
class Graph2EditsInferencePolicy:
    """Frozen proposal-normalization policy; scores remain opaque."""

    max_edit_steps: int = 9
    oversampling_factor: int = 4
    maximum_worker_results: int = 256
    score_order: str = "higher_is_better"
    reaction_class_mode: str = "unknown"
    reactant_identity_policy: str = PROPOSAL_REACTANT_IDENTITY_POLICY

    def __post_init__(self) -> None:
        for name in ("max_edit_steps", "oversampling_factor", "maximum_worker_results"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise Graph2EditsBackendError(f"{name} must be a positive integer")
        if self.score_order != "higher_is_better":
            raise Graph2EditsBackendError("Graph2Edits score_order must be higher_is_better")
        if self.reaction_class_mode != "unknown":
            raise Graph2EditsBackendError("only reaction-class-unknown Graph2Edits is admitted")
        if self.reactant_identity_policy != PROPOSAL_REACTANT_IDENTITY_POLICY:
            raise Graph2EditsBackendError("Graph2Edits reactant identity policy is unsupported")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GRAPH2EDITS_ADAPTER_SCHEMA_VERSION,
            "max_edit_steps": self.max_edit_steps,
            "oversampling_factor": self.oversampling_factor,
            "maximum_worker_results": self.maximum_worker_results,
            "score_order": self.score_order,
            "reaction_class_mode": self.reaction_class_mode,
            "reactant_identity_policy": self.reactant_identity_policy,
            "remove_atom_maps": True,
            "isomeric_smiles": False,
        }

    @property
    def policy_sha256(self) -> str:
        return _content_sha256(self.to_dict())


@dataclass(frozen=True)
class Graph2EditsRawPrediction:
    """One ranked worker output before FORGE identity normalization."""

    reactants_smiles: str
    model_score: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.reactants_smiles, str) or not self.reactants_smiles.strip():
            raise Graph2EditsBackendError("raw Graph2Edits reactants must be nonempty SMILES")
        if self.model_score is not None:
            if isinstance(self.model_score, bool) or not isinstance(self.model_score, (int, float)):
                raise Graph2EditsBackendError("raw Graph2Edits model score must be numeric or null")
            if not math.isfinite(float(self.model_score)):
                raise Graph2EditsBackendError("raw Graph2Edits model score must be finite or null")


@dataclass(frozen=True)
class Graph2EditsProposalTrace:
    """Deterministic join metadata for downstream two-lane evaluation.

    Lipid precedent is intentionally *not* assigned here.  An independent,
    hash-bound chemistry screen may join on ``proposal_id`` to compare a
    lipid-precedented-first soft ordering with the unrestricted proposal list.
    Neither lane is evidence, and no reaction family is filtered by this
    adapter.
    """

    proposal_id: str
    canonical_reactants: tuple[str, ...]
    worker_rank: int
    normalized_rank: int
    model_score: float | None
    inference_policy_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GRAPH2EDITS_ADAPTER_SCHEMA_VERSION,
            "proposal_id": self.proposal_id,
            "canonical_reactants": list(self.canonical_reactants),
            "worker_rank": self.worker_rank,
            "normalized_rank": self.normalized_rank,
            "model_score": self.model_score,
            "inference_policy_sha256": self.inference_policy_sha256,
            "reaction_family_filter_applied": False,
            "lipid_precedent_state": "unassessed",
            "evidence_tier": None,
            "route_closure_authorized": False,
        }


@dataclass(frozen=True)
class Graph2EditsProposalBatch:
    """Typed proposals plus non-authoritative deterministic audit metadata."""

    proposals: tuple[SingleStepRetrosynthesisProposal, ...]
    trace: tuple[Graph2EditsProposalTrace, ...]


class Graph2EditsWorker(Protocol):
    """Injected local inference boundary; it must not perform downloads."""

    def __call__(
        self,
        target_smiles: str,
        *,
        num_results: int,
    ) -> Sequence[Graph2EditsRawPrediction]: ...


def _verify_manifest_identity(manifest: ProposalBackendManifest) -> None:
    if not isinstance(manifest, ProposalBackendManifest):
        raise Graph2EditsBackendError("Graph2Edits backend requires a typed manifest")
    expected = {
        "backend_id": GRAPH2EDITS_BACKEND_ID,
        "implementation_version": GRAPH2EDITS_IMPLEMENTATION_VERSION,
        "checkpoint_license": GRAPH2EDITS_CHECKPOINT_LICENSE,
        "training_corpus_id": GRAPH2EDITS_TRAINING_CORPUS_ID,
        "source_locator": GRAPH2EDITS_SOURCE_LOCATOR,
    }
    for field, expected_value in expected.items():
        if getattr(manifest, field) != expected_value:
            raise Graph2EditsBackendError(
                f"Graph2Edits manifest {field} does not match the frozen release"
            )


def _canonicalize_reactants(raw_smiles: str) -> tuple[str, ...] | None:
    tokens = raw_smiles.split(".")
    if not tokens or any(not token.strip() for token in tokens):
        return None
    canonical: list[str] = []
    for token in tokens:
        try:
            molecule = Chem.MolFromSmiles(token)
        except Exception:  # pragma: no cover - defensive RDKit boundary
            return None
        if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
            return None
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        Chem.RemoveStereochemistry(molecule)
        normalized = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
        if not normalized:
            return None
        canonical.append(normalized)
    return tuple(sorted(canonical))


@dataclass(frozen=True)
class _NormalizedPrediction:
    reactants: tuple[str, ...]
    score: float | None
    worker_rank: int

    @property
    def sort_key(self) -> tuple[int, float, int, tuple[str, ...]]:
        return (
            1 if self.score is None else 0,
            0.0 if self.score is None else -float(self.score),
            self.worker_rank,
            self.reactants,
        )


class Graph2EditsProposalBackend:
    """Proposal-only adapter over one verified local Graph2Edits worker."""

    def __init__(
        self,
        *,
        manifest: ProposalBackendManifest,
        artifacts: Graph2EditsLocalArtifacts,
        worker: Graph2EditsWorker,
        policy: Graph2EditsInferencePolicy | None = None,
    ) -> None:
        _verify_manifest_identity(manifest)
        if not isinstance(artifacts, Graph2EditsLocalArtifacts):
            raise Graph2EditsBackendError("Graph2Edits local-artifact receipt is malformed")
        artifacts.verify(manifest)
        if not callable(worker):
            raise Graph2EditsBackendError("Graph2Edits worker must be callable")
        resolved_policy = Graph2EditsInferencePolicy() if policy is None else policy
        if not isinstance(resolved_policy, Graph2EditsInferencePolicy):
            raise Graph2EditsBackendError("Graph2Edits inference policy is malformed")
        self._manifest = manifest
        self._worker = worker
        self._policy = resolved_policy

    @property
    def manifest(self) -> ProposalBackendManifest:
        return self._manifest

    @property
    def policy(self) -> Graph2EditsInferencePolicy:
        return self._policy

    def propose(
        self,
        request: ProposalRequest,
        *,
        maximum_proposals: int,
    ) -> tuple[SingleStepRetrosynthesisProposal, ...]:
        return self.propose_with_trace(
            request,
            maximum_proposals=maximum_proposals,
        ).proposals

    def propose_with_trace(
        self,
        request: ProposalRequest,
        *,
        maximum_proposals: int,
    ) -> Graph2EditsProposalBatch:
        if not isinstance(request, ProposalRequest):
            raise Graph2EditsBackendError("Graph2Edits proposal request is malformed")
        if (
            isinstance(maximum_proposals, bool)
            or not isinstance(maximum_proposals, int)
            or maximum_proposals < 1
        ):
            raise Graph2EditsBackendError("maximum_proposals must be a positive integer")

        requested_worker_results = min(
            self._policy.maximum_worker_results,
            max(maximum_proposals, maximum_proposals * self._policy.oversampling_factor),
        )
        raw_predictions = self._worker(
            request.target.canonical_smiles,
            num_results=requested_worker_results,
        )
        if isinstance(raw_predictions, (str, bytes)) or not isinstance(raw_predictions, Sequence):
            raise Graph2EditsBackendError("Graph2Edits worker must return a sequence")
        if len(raw_predictions) > requested_worker_results:
            raise Graph2EditsBackendError("Graph2Edits worker exceeded its requested result bound")

        normalized: list[_NormalizedPrediction] = []
        for worker_rank, raw_prediction in enumerate(raw_predictions, start=1):
            if not isinstance(raw_prediction, Graph2EditsRawPrediction):
                raise Graph2EditsBackendError(
                    "Graph2Edits worker returned a malformed prediction record"
                )
            reactants = _canonicalize_reactants(raw_prediction.reactants_smiles)
            if reactants is None:
                continue
            score = (
                None if raw_prediction.model_score is None else float(raw_prediction.model_score)
            )
            normalized.append(
                _NormalizedPrediction(
                    reactants=reactants,
                    score=score,
                    worker_rank=worker_rank,
                )
            )

        unique: dict[tuple[str, ...], _NormalizedPrediction] = {}
        for prediction in sorted(normalized, key=lambda item: item.sort_key):
            unique.setdefault(prediction.reactants, prediction)

        selected = list(unique.values())[:maximum_proposals]
        proposals = tuple(
            self._build_proposal(request, prediction, rank=rank)
            for rank, prediction in enumerate(
                selected,
                start=1,
            )
        )
        proposals = validate_proposal_batch(
            self._manifest,
            request,
            proposals,
            maximum_proposals=maximum_proposals,
        )
        trace = tuple(
            Graph2EditsProposalTrace(
                proposal_id=proposal.proposal_id,
                canonical_reactants=proposal.reactant_smiles,
                worker_rank=prediction.worker_rank,
                normalized_rank=proposal.rank,
                model_score=proposal.model_score,
                inference_policy_sha256=self._policy.policy_sha256,
            )
            for proposal, prediction in zip(proposals, selected, strict=True)
        )
        return Graph2EditsProposalBatch(proposals=proposals, trace=trace)

    def _build_proposal(
        self,
        request: ProposalRequest,
        prediction: _NormalizedPrediction,
        *,
        rank: int,
    ) -> SingleStepRetrosynthesisProposal:
        identity = {
            "schema_version": GRAPH2EDITS_ADAPTER_SCHEMA_VERSION,
            "backend": self._manifest.to_dict(),
            "inference_policy_sha256": self._policy.policy_sha256,
            "request": request.to_dict(),
            "reactants": list(prediction.reactants),
        }
        proposal_id = f"graph2edits:{_content_sha256(identity)}"
        return SingleStepRetrosynthesisProposal(
            proposal_id=proposal_id,
            backend=self._manifest,
            request=request,
            reactant_smiles=prediction.reactants,
            rank=rank,
            model_score=prediction.score,
            predicted_reaction_class=None,
        )


def build_syntheseus_worker(
    *,
    model_dir: Path,
    device: str = "cpu",
    max_edit_steps: int = 9,
) -> Graph2EditsWorker:
    """Build a local Syntheseus worker without permitting checkpoint download.

    This helper is intentionally separate from ``Graph2EditsProposalBackend``
    so importing and unit-testing FORGE never requires Syntheseus, PyTorch or
    the external Graph2Edits package.
    """

    resolved_model_dir = Path(model_dir)
    if not resolved_model_dir.is_dir():
        raise Graph2EditsDependencyError(
            f"Graph2Edits model_dir must be an existing local directory: {resolved_model_dir}"
        )
    if (
        isinstance(max_edit_steps, bool)
        or not isinstance(max_edit_steps, int)
        or max_edit_steps < 1
    ):
        raise Graph2EditsDependencyError("max_edit_steps must be a positive integer")

    required_versions = {
        "syntheseus": GRAPH2EDITS_SYNTHESEUS_VERSION,
        "syntheseus-graph2edits": GRAPH2EDITS_PACKAGE_VERSION,
    }
    for package_name, required_version in required_versions.items():
        try:
            installed_version = metadata.version(package_name)
        except metadata.PackageNotFoundError as error:
            raise Graph2EditsDependencyError(
                f"optional dependency {package_name}=={required_version} is not installed; "
                "create the separately locked Graph2Edits environment and pass a local model_dir"
            ) from error
        if installed_version != required_version:
            raise Graph2EditsDependencyError(
                f"optional dependency {package_name}=={required_version} is required; "
                f"found {installed_version}"
            )

    try:
        from syntheseus import Molecule
        from syntheseus.reaction_prediction.inference import Graph2EditsModel
    except (ImportError, ModuleNotFoundError) as error:  # pragma: no cover - optional boundary
        raise Graph2EditsDependencyError(
            "pinned Syntheseus/Graph2Edits imports failed in the optional environment"
        ) from error

    try:
        model = Graph2EditsModel(
            model_dir=resolved_model_dir,
            device=device,
            max_edit_steps=max_edit_steps,
            use_cache=False,
        )
    except (ImportError, ModuleNotFoundError) as error:  # pragma: no cover - optional boundary
        raise Graph2EditsDependencyError(
            "the optional graph2edits runtime is incomplete in the locked environment"
        ) from error

    def worker(
        target_smiles: str,
        *,
        num_results: int,
    ) -> tuple[Graph2EditsRawPrediction, ...]:
        try:
            [reactions] = model([Molecule(target_smiles)], num_results=num_results)
        except (TypeError, ValueError, RuntimeError) as error:
            raise Graph2EditsBackendError("local Graph2Edits inference failed") from error
        outputs: list[Graph2EditsRawPrediction] = []
        for reaction in reactions:
            reactants_smiles = ".".join(molecule.smiles for molecule in reaction.reactants)
            raw_score = reaction.metadata.get("probability")
            outputs.append(
                Graph2EditsRawPrediction(
                    reactants_smiles=reactants_smiles,
                    model_score=None if raw_score is None else float(raw_score),
                )
            )
        return tuple(outputs)

    return worker


__all__ = [
    "GRAPH2EDITS_BACKEND_ID",
    "GRAPH2EDITS_CHECKPOINT_LICENSE",
    "GRAPH2EDITS_IMPLEMENTATION_VERSION",
    "GRAPH2EDITS_SOURCE_COMMIT",
    "GRAPH2EDITS_SOURCE_LOCATOR",
    "GRAPH2EDITS_TRAINING_CORPUS_ID",
    "Graph2EditsBackendError",
    "Graph2EditsDependencyError",
    "Graph2EditsInferencePolicy",
    "Graph2EditsLocalArtifacts",
    "Graph2EditsProposalBatch",
    "Graph2EditsProposalBackend",
    "Graph2EditsProposalTrace",
    "Graph2EditsRawPrediction",
    "Graph2EditsWorker",
    "build_syntheseus_worker",
    "verify_graph2edits_training_corpus_receipt",
]
