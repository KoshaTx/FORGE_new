"""Hash-bound production callback for the selected step-1000 Ugi generator.

This module is deliberately a thin composition layer.  It authenticates the
frozen product-plus-L1 generator, its sparse-closure checkpoint, its exact
training cache and atom vocabulary, the qualified Ugi reaction, the declared
graph-support context, the generated-component recovery implementation, and
the restartable-sampler equivalence receipt.  A callback then executes one
and only one native eight-step completion for the schedule's canonical
morphology program and productive seed before sealing the native completion
row through :mod:`ugi_restartable_terminal_support_adapter`.

It performs no route search, synthesis-value calculation, biological scoring,
candidate selection, repair, retry, or holdout inspection.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from experiments.phase1.product_l1.sampling.ugi_end_to_end_sampling import (
    _closure_model,
    _load_checkpoint,
)
from experiments.phase1.product_l1.sampling.ugi_joint_end_to_end_sampling import (
    complete_ugi_joint_terminals,
)
from experiments.phase1.product_l1.sampling.ugi_joint_sparse_sampling import (
    sample_restartable_terminals,
)
from experiments.phase1.product_l1.sampling.ugi_selected_generator_implementation import (
    SelectedGeneratorImplementationQualification,
    build_selected_generator_implementation_qualification,
    require_selected_generator_implementation_unchanged,
)
from experiments.phase1.product_l1.training.ugi_training_cache import load_ugi_training_cache
from experiments.phase1.synthesis_guidance.adapters.terminal_support import (
    AdaptedRestartableGeneratedTerminal,
    UgiRestartableTerminalSupportAdapterError,
    adapt_restartable_completion_row_for_route_support,
    decode_canonical_morphology_program_bytes,
    lock_unqualified_restartable_completion_row,
    native_completion_record,
    native_completion_record_bytes,
    native_completion_record_from_locked_terminal,
)
from experiments.phase1.synthesis_guidance.guidance.ugi_zero_guidance_rehearsal import (
    RestartableGeneratorClosureAdapter,
    RestartableGeneratorClosureIdentity,
)
from forge.corpus.ugi_generated_terminal_support import (
    DeclaredGraphSupportContext,
    UgiGeneratedTerminalSupportError,
    declared_graph_support_context_sha256,
    qualify_locked_generated_ugi_terminal_support,
)
from forge.corpus.ugi_held_component_gate import load_ugi_reaction_contract
from forge.model.defog_feasibility import sha256_file
from forge.model.ugi_joint_sparse_flow import UgiJointSparseFlow
from forge.synthesis.matched import (
    LockedMatchedTerminal,
    MatchedGenerationRequest,
)
from forge.synthesis.terminals.terminal_assessment import (
    QualifiedUgiL1Reverifier,
    UgiTerminalRouteAssessmentError,
    ValidatedUgiTerminalPayload,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - product sampling requires torch
    torch = None


SELECTED_RESTARTABLE_GENERATOR_SCHEMA_VERSION = "forge.selected_step1000_restartable_generator.v1"
SAMPLE_STEPS = 8
TERMINAL_DECODER_ID = "ugi_joint_terminal_completion:argmax:v1"

GENERATOR_CHECKPOINT_SHA256 = "90f5f0bd3e41e2884f6588d9875db0b7bf34e5c5ea7fdc1f9d8565fba8c0c532"
CLOSURE_CHECKPOINT_SHA256 = "a97507ac6a9eeba41d0cc351666cffdd21069d13bc78c0db671ab3a30c710b5d"
PRODUCTION_GENERATOR_MANIFEST_SHA256 = (
    "c27352c11a2e210fd76a6e4ae510bbbb89e51c92c28514fd9b4693f82a50e68a"
)
TRAINING_CACHE_SHA256 = "b862b7a54c0cb325f0a962fea42a78138df1a81c88b9c0e86d56b6448e519c46"
ATOM_VOCABULARY_SHA256 = "90ab7430354a9ffd91cfd6e19f2a125020a6547c67cd462a11f77074fd2a0cc5"
QUALIFIED_REACTION_REGISTRY_SHA256 = (
    "296bf06238ef22acc1f55117f5ce0adaee21b1bafaf5a83f89182b0f31cc4fcf"
)
L1_REACTION_SHA256 = "5b97e062b115fcc137b4a05d8f72b74e9cc67a584c05c054ef5f983969bf1427"
MODEL_CONFIG_SHA256 = "f3279a9b64aac4dfae7372c784a863fb93f25cfb4a0f1b5cb1ceb8c4c4a77f52"
DECLARED_GRAPH_SUPPORT_SHA256 = "331f40d5bf1ffe61688def4d9c971d79ae7607e6139bcc3ceb355871e097f505"
COMPONENT_RECOVERY_CONTRACT_SHA256 = (
    "14a84998891a814db287ed808d650a7c8e79175f85ee1508645fa5de0e10b944"
)


class UgiSelectedRestartableGeneratorError(RuntimeError):
    """Raised when the selected productive generator lane cannot fail closed."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    except (TypeError, ValueError) as error:
        raise UgiSelectedRestartableGeneratorError(
            "selected generator binding is not canonically serializable"
        ) from error


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_file_hash(path: Path, expected_sha256: str, *, label: str) -> None:
    if not path.is_file():
        raise UgiSelectedRestartableGeneratorError(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise UgiSelectedRestartableGeneratorError(
            f"{label} hash changed: expected {expected_sha256}, observed {observed}"
        )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiSelectedRestartableGeneratorError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise UgiSelectedRestartableGeneratorError(f"{label} must contain one JSON object")
    return value


def _atom_vocabulary_states(path: Path) -> frozenset[tuple[str, int, bool, int]]:
    value = _read_json(path, label="atom vocabulary")
    rows = value.get("atom_vocabulary")
    if not isinstance(rows, list) or not rows:
        raise UgiSelectedRestartableGeneratorError("atom vocabulary is empty or malformed")
    output: set[tuple[str, int, bool, int]] = set()
    try:
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError
            state = (
                str(row["symbol"]),
                int(row["formal_charge"]),
                bool(row["aromatic"]),
                int(row["explicit_hydrogens"]),
            )
            if not state[0] or state[3] < 0:
                raise ValueError
            output.add(state)
    except (KeyError, TypeError, ValueError) as error:
        raise UgiSelectedRestartableGeneratorError("atom vocabulary is malformed") from error
    if len(output) != len(rows):
        raise UgiSelectedRestartableGeneratorError(
            "atom vocabulary contains duplicate constitutional atom states"
        )
    return frozenset(output)


@dataclass(frozen=True)
class SelectedStep1000Artifacts:
    """Exact filesystem bindings for the frozen step-1000 productive lane."""

    repository: Path
    generator_checkpoint: Path
    closure_checkpoint: Path
    production_generator_manifest: Path
    restartable_equivalence_receipt: Path
    training_cache: Path
    atom_vocabulary: Path
    qualified_reaction_registry: Path
    l1_reaction_variant: Path
    component_recovery_contract: Path

    @classmethod
    def from_repository(cls, repository: Path) -> SelectedStep1000Artifacts:
        root = Path(repository).resolve()
        return cls(
            repository=root,
            generator_checkpoint=(
                root / "results/phase1/ugi_joint_sparse_balanced_v2_full/checkpoint_step_1000.pt"
            ),
            closure_checkpoint=(
                root / "results/phase1/ugi_closure_expanded_full/checkpoint_best.pt"
            ),
            production_generator_manifest=(
                root / "results/phase1/ugi_product_l1_production_generator_v1.json"
            ),
            restartable_equivalence_receipt=(
                root / "results/phase1/ugi_restartable_sampler_equivalence_v2/result.json"
            ),
            training_cache=(
                root / "results/phase1/ugi_balanced_training_cache_v2/ugi_training_cache.pt"
            ),
            atom_vocabulary=root / "results/phase1/product_v3_atom_vocabulary.json",
            qualified_reaction_registry=root / "data/vendor/qualified_reactions_v1.json",
            l1_reaction_variant=root / "configs/assembly/ugi_variant.yaml",
            component_recovery_contract=(root / "forge/corpus/ugi_generated_components.py"),
        )


@dataclass(frozen=True)
class SelectedRestartableGeneratorBindings:
    """Content identities rechecked before model construction."""

    generator_checkpoint_sha256: str
    closure_checkpoint_sha256: str
    production_generator_manifest_sha256: str
    restartable_equivalence_receipt_sha256: str
    generator_implementation_sha256: str
    training_cache_sha256: str
    atom_vocabulary_sha256: str
    qualified_reaction_registry_sha256: str
    l1_reaction_sha256: str
    model_config_sha256: str
    declared_graph_support_sha256: str
    component_recovery_contract_sha256: str
    sample_steps: int
    terminal_decoder_id: str

    @property
    def canonical_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": SELECTED_RESTARTABLE_GENERATOR_SCHEMA_VERSION,
                **self.__dict__,
            }
        )


def _require_manifest_contract(value: dict[str, Any]) -> None:
    try:
        valid = (
            value["schema_version"] == "phase1_ugi_product_l1_production_generator.v1"
            and value["status"] == "frozen"
            and value["identity"]["architecture"] == "full_morphology_program_conditioning"
            and value["identity"]["checkpoint_step"] == 1000
            and value["selection"]["fresh_selected_sample"]["sample_steps"] == SAMPLE_STEPS
            and value["model"]["checkpoint"]["sha256"] == GENERATOR_CHECKPOINT_SHA256
            and value["common_training_inputs"]["prepared_cache"]["sha256"] == TRAINING_CACHE_SHA256
            and value["common_training_inputs"]["atom_vocabulary"]["sha256"]
            == ATOM_VOCABULARY_SHA256
            and value["sampling_dependencies"]["closure_checkpoint"]["sha256"]
            == CLOSURE_CHECKPOINT_SHA256
            and value["sampling_dependencies"]["qualified_reaction_registry"]["sha256"]
            == QUALIFIED_REACTION_REGISTRY_SHA256
        )
    except (KeyError, TypeError) as error:
        raise UgiSelectedRestartableGeneratorError(
            "production generator manifest is malformed"
        ) from error
    if not valid:
        raise UgiSelectedRestartableGeneratorError(
            "production generator manifest does not bind the selected step-1000 lane"
        )


def _require_restartable_receipt(
    value: dict[str, Any],
    *,
    generator_implementation_sha256: str,
) -> None:
    try:
        comparisons = value["comparisons"]
        valid = (
            value["schema_version"] == "phase1_ugi_restartable_sampler_equivalence.v1"
            and value["status"] == "complete"
            and value["decision"] == "restartable_zero_guidance_schedule_is_bitwise_equivalent"
            and value["sample_steps"] == SAMPLE_STEPS
            and value["production_synthesis_guidance"] is False
            and value["biological_guidance"] is False
            and value["inputs"]["joint_checkpoint"]["sha256"] == GENERATOR_CHECKPOINT_SHA256
            and value["inputs"]["generator_implementation"]["implementation_sha256"]
            == generator_implementation_sha256
            and isinstance(comparisons, list)
            and comparisons
            and all(
                row["bitwise_equal"] is True
                and row["metadata"]["sample_steps"] == SAMPLE_STEPS
                and row["metadata"]["terminal_tree_repairs"] == 0
                for row in comparisons
            )
        )
    except (KeyError, TypeError) as error:
        raise UgiSelectedRestartableGeneratorError(
            "restartable equivalence receipt is malformed"
        ) from error
    if not valid:
        raise UgiSelectedRestartableGeneratorError(
            "restartable equivalence receipt does not authorize the selected eight-step lane"
        )


def _corpus_atom_states(corpus: Any) -> frozenset[tuple[str, int, bool, int]]:
    try:
        return frozenset(
            (
                str(state.symbol),
                int(state.formal_charge),
                bool(state.aromatic),
                int(state.explicit_hydrogens),
            )
            for state in corpus.atom_vocabulary
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise UgiSelectedRestartableGeneratorError(
            "prepared cache contains a malformed atom vocabulary"
        ) from error


@dataclass
class SelectedRestartableGeneratorCallback:
    """One callable native completion lane with retained support-bound records."""

    model: Any
    closure_model: Any
    source_marginals: dict[str, np.ndarray]
    corpus: Any
    reaction: Any
    graph_support: DeclaredGraphSupportContext
    l1_reverifier: QualifiedUgiL1Reverifier
    bindings: SelectedRestartableGeneratorBindings
    allowed_ring_sizes: tuple[int, ...]
    maximum_heavy_degree: int
    repository: Path
    implementation_qualification: SelectedGeneratorImplementationQualification

    def __post_init__(self) -> None:
        if not isinstance(self.repository, Path) or not self.repository.is_dir():
            raise UgiSelectedRestartableGeneratorError(
                "selected generator repository must be a directory"
            )
        if not isinstance(
            self.implementation_qualification,
            SelectedGeneratorImplementationQualification,
        ):
            raise UgiSelectedRestartableGeneratorError(
                "selected generator implementation qualification is malformed"
            )
        if self.bindings.generator_implementation_sha256 != (
            self.implementation_qualification.implementation_sha256
        ):
            raise UgiSelectedRestartableGeneratorError(
                "generator bindings and implementation qualification differ"
            )

    def __call__(self, request: MatchedGenerationRequest) -> LockedMatchedTerminal:
        if torch is None:
            raise UgiSelectedRestartableGeneratorError(
                "selected productive generation requires torch"
            )
        if not isinstance(request, MatchedGenerationRequest):
            raise UgiSelectedRestartableGeneratorError(
                "productive request must be a MatchedGenerationRequest"
            )
        require_selected_generator_implementation_unchanged(
            self.repository,
            self.implementation_qualification,
        )
        entry = request.entry
        if entry.generator_checkpoint_sha256 != self.bindings.generator_checkpoint_sha256:
            raise UgiSelectedRestartableGeneratorError(
                "schedule generator checkpoint differs from the selected checkpoint"
            )
        if entry.closure_checkpoint_sha256 != self.bindings.closure_checkpoint_sha256:
            raise UgiSelectedRestartableGeneratorError(
                "schedule closure checkpoint differs from the selected closure checkpoint"
            )
        if entry.productive_generation_calls != 1:
            raise UgiSelectedRestartableGeneratorError(
                "one scheduled unit must reserve exactly one productive generation call"
            )
        if (
            isinstance(request.productive_seed, bool)
            or not isinstance(request.productive_seed, int)
            or request.productive_seed < 0
        ):
            raise UgiSelectedRestartableGeneratorError(
                "productive seed must be a nonnegative integer"
            )
        try:
            program = decode_canonical_morphology_program_bytes(entry.morphology_program)
        except UgiRestartableTerminalSupportAdapterError as error:
            raise UgiSelectedRestartableGeneratorError(
                "schedule morphology program is not canonical"
            ) from error

        terminals, metadata = sample_restartable_terminals(
            self.model,
            (program,),
            self.source_marginals,
            sample_steps=SAMPLE_STEPS,
            batch_size=1,
            seed=request.productive_seed,
            device="cpu",
            allowed_ring_sizes=self.allowed_ring_sizes,
            maximum_heavy_degree=self.maximum_heavy_degree,
            maximum_adjacent_branch_runs=(None, None, None),
        )
        if (
            len(terminals) != 1
            or metadata.get("sample_steps") != SAMPLE_STEPS
            or metadata.get("terminal_tree_repairs") != 0
            or metadata.get("samples") != 1
        ):
            raise UgiSelectedRestartableGeneratorError(
                "native restartable sampler violated the one-terminal eight-step contract"
            )
        completion = complete_ugi_joint_terminals(
            self.model,
            self.closure_model,
            terminals,
            self.corpus,
            program_metadata=({},),
            closure_generator_state=(
                torch.Generator().manual_seed(request.productive_seed + 1).get_state()
            ),
            allowed_ring_sizes=self.allowed_ring_sizes,
            maximum_heavy_degree=self.maximum_heavy_degree,
            l1_reaction=self.reaction,
            terminal_decoder_mode="argmax",
            terminal_generator_state=None,
            terminal_temperature=1.0,
        )
        if len(completion.rows) != 1:
            raise UgiSelectedRestartableGeneratorError(
                "native closure/chemistry completion did not return exactly one row"
            )
        row = completion.rows[0]
        try:
            native = native_completion_record(row)
        except UgiRestartableTerminalSupportAdapterError as error:
            raise UgiSelectedRestartableGeneratorError(
                f"native terminal for {entry.unit_id} is not an assessed completion row"
            ) from error
        if native["valid"] is not True:
            return lock_unqualified_restartable_completion_row(
                native,
                generation_request=request,
                nonqualification_reason="native_molecule_invalid",
            )
        if native["component_reconstruction_valid"] is not True:
            return lock_unqualified_restartable_completion_row(
                native,
                generation_request=request,
                nonqualification_reason="component_reconstruction_failed",
            )
        try:
            ValidatedUgiTerminalPayload.from_recovered_components(
                product_smiles=native["smiles"],
                components_by_role=native["component_smiles_by_role"],
                l1_reaction=self.reaction,
                l1_reaction_sha256=self.bindings.l1_reaction_sha256,
                component_recovery_contract_sha256=(
                    self.bindings.component_recovery_contract_sha256
                ),
            )
        except UgiTerminalRouteAssessmentError:
            return lock_unqualified_restartable_completion_row(
                native,
                generation_request=request,
                nonqualification_reason="independent_l1_forward_failed",
            )
        try:
            return adapt_restartable_completion_row_for_route_support(
                native,
                generation_request=request,
                l1_reaction=self.reaction,
                l1_reaction_sha256=self.bindings.l1_reaction_sha256,
                component_recovery_contract_sha256=(
                    self.bindings.component_recovery_contract_sha256
                ),
                graph_support=self.graph_support,
                l1_reverifier=self.l1_reverifier,
            ).locked_terminal
        except UgiRestartableTerminalSupportAdapterError as error:
            raise UgiSelectedRestartableGeneratorError(
                f"native exact-L1 terminal for {entry.unit_id} violated support; no retry was attempted"
            ) from error

    def adapted_for_terminal(
        self,
        terminal: LockedMatchedTerminal,
    ) -> AdaptedRestartableGeneratedTerminal:
        """Recreate support from the immutable trace, without a side registry."""

        if not isinstance(terminal, LockedMatchedTerminal):
            raise UgiSelectedRestartableGeneratorError(
                "support lookup requires a LockedMatchedTerminal"
            )
        if (
            terminal.generator_checkpoint_sha256 != self.bindings.generator_checkpoint_sha256
            or terminal.closure_checkpoint_sha256 != self.bindings.closure_checkpoint_sha256
        ):
            raise UgiSelectedRestartableGeneratorError(
                "terminal checkpoint identity differs from this callback lane"
            )
        if not terminal.terminal_valid or not terminal.exact_l1:
            raise UgiSelectedRestartableGeneratorError(
                "invalid or nonexact terminals have no route-root support"
            )
        try:
            candidate = native_completion_record_from_locked_terminal(terminal)
            support = qualify_locked_generated_ugi_terminal_support(
                terminal,
                candidate_record=candidate,
                graph_support=self.graph_support,
                l1_reverifier=self.l1_reverifier,
            )
        except (
            UgiGeneratedTerminalSupportError,
            UgiRestartableTerminalSupportAdapterError,
        ) as error:
            raise UgiSelectedRestartableGeneratorError(
                "locked terminal failed immutable native-record support resolution"
            ) from error
        return AdaptedRestartableGeneratedTerminal(
            locked_terminal=terminal,
            candidate_record_bytes=native_completion_record_bytes(candidate),
            support=support,
        )


@dataclass(frozen=True)
class SelectedRestartableGeneratorLane:
    """Factory product exposed to zero-guidance orchestration and route seams."""

    adapter: RestartableGeneratorClosureAdapter
    callback: SelectedRestartableGeneratorCallback
    bindings: SelectedRestartableGeneratorBindings
    graph_support: DeclaredGraphSupportContext
    implementation_qualification: SelectedGeneratorImplementationQualification


def build_selected_step1000_restartable_generator_lane(
    repository: Path,
    *,
    artifacts: SelectedStep1000Artifacts | None = None,
) -> SelectedRestartableGeneratorLane:
    """Authenticate and construct the frozen step-1000 restartable callback."""

    if torch is None:
        raise UgiSelectedRestartableGeneratorError("selected productive generation requires torch")
    resolved = artifacts or SelectedStep1000Artifacts.from_repository(repository)
    expected_root = Path(repository).resolve()
    if resolved.repository.resolve() != expected_root:
        raise UgiSelectedRestartableGeneratorError(
            "artifact repository differs from the requested repository"
        )
    implementation_qualification = build_selected_generator_implementation_qualification(
        resolved.repository
    )
    restartable_equivalence_receipt_sha256 = sha256_file(resolved.restartable_equivalence_receipt)
    checks = (
        (resolved.generator_checkpoint, GENERATOR_CHECKPOINT_SHA256, "generator checkpoint"),
        (resolved.closure_checkpoint, CLOSURE_CHECKPOINT_SHA256, "closure checkpoint"),
        (
            resolved.production_generator_manifest,
            PRODUCTION_GENERATOR_MANIFEST_SHA256,
            "production generator manifest",
        ),
        (resolved.training_cache, TRAINING_CACHE_SHA256, "prepared training cache"),
        (resolved.atom_vocabulary, ATOM_VOCABULARY_SHA256, "atom vocabulary"),
        (
            resolved.qualified_reaction_registry,
            QUALIFIED_REACTION_REGISTRY_SHA256,
            "qualified Ugi reaction registry",
        ),
        (
            resolved.l1_reaction_variant,
            L1_REACTION_SHA256,
            "qualified Ugi L1 variant",
        ),
        (
            resolved.component_recovery_contract,
            COMPONENT_RECOVERY_CONTRACT_SHA256,
            "generated-component recovery contract",
        ),
    )
    for path, expected, label in checks:
        _require_file_hash(path, expected, label=label)
    _require_manifest_contract(
        _read_json(resolved.production_generator_manifest, label="production generator manifest")
    )
    _require_restartable_receipt(
        _read_json(
            resolved.restartable_equivalence_receipt,
            label="restartable equivalence receipt",
        ),
        generator_implementation_sha256=(implementation_qualification.implementation_sha256),
    )

    joint_checkpoint = _load_checkpoint(
        resolved.generator_checkpoint,
        "phase1_ugi_joint_sparse_checkpoint.v1",
    )
    closure_checkpoint = _load_checkpoint(
        resolved.closure_checkpoint,
        (
            "phase1_ugi_sparse_closure_checkpoint.v1",
            "phase1_ugi_sparse_closure_checkpoint.v2",
        ),
    )
    model_config = dict(joint_checkpoint.get("model_config", {}))
    if _canonical_sha256(model_config) != MODEL_CONFIG_SHA256:
        raise UgiSelectedRestartableGeneratorError("selected model configuration hash changed")
    checkpoint_inputs = joint_checkpoint.get("inputs")
    if not isinstance(checkpoint_inputs, Mapping):
        raise UgiSelectedRestartableGeneratorError("selected checkpoint inputs are malformed")
    try:
        input_valid = (
            checkpoint_inputs["prepared_cache"]["sha256"] == TRAINING_CACHE_SHA256
            and checkpoint_inputs["atom_vocabulary"]["sha256"] == ATOM_VOCABULARY_SHA256
        )
    except (KeyError, TypeError) as error:
        raise UgiSelectedRestartableGeneratorError(
            "selected checkpoint lacks cache or vocabulary bindings"
        ) from error
    if not input_valid:
        raise UgiSelectedRestartableGeneratorError(
            "selected checkpoint cache or vocabulary binding changed"
        )

    corpus, _ = load_ugi_training_cache(resolved.training_cache)
    vocabulary_states = _atom_vocabulary_states(resolved.atom_vocabulary)
    if _corpus_atom_states(corpus) != vocabulary_states:
        raise UgiSelectedRestartableGeneratorError(
            "prepared cache atom vocabulary differs from the frozen vocabulary artifact"
        )
    graph_support = DeclaredGraphSupportContext(
        generator_checkpoint_sha256=GENERATOR_CHECKPOINT_SHA256,
        model_config=MappingProxyType(model_config.copy()),
        atom_vocabulary=vocabulary_states,
    )
    graph_support_sha256 = declared_graph_support_context_sha256(graph_support)
    if graph_support_sha256 != DECLARED_GRAPH_SUPPORT_SHA256:
        raise UgiSelectedRestartableGeneratorError("declared graph-support context hash changed")

    architecture = model_config.copy()
    source_probability_floor = architecture.pop("source_probability_floor", None)
    if source_probability_floor != 1e-5:
        raise UgiSelectedRestartableGeneratorError(
            "selected checkpoint source probability floor changed"
        )
    model = UgiJointSparseFlow(
        atom_classes=len(corpus.atom_vocabulary),
        **architecture,
    )
    model.load_state_dict(joint_checkpoint["model_state"])
    model.eval()
    closure_model = _closure_model(closure_checkpoint)
    closure_model.eval()
    reaction = load_ugi_reaction_contract(resolved.qualified_reaction_registry)
    reverifier = QualifiedUgiL1Reverifier(
        reaction_contract=reaction,
        l1_reaction_sha256=L1_REACTION_SHA256,
    )
    try:
        allowed_ring_sizes = tuple(int(value) for value in closure_checkpoint["allowed_ring_sizes"])
        maximum_heavy_degree = int(closure_checkpoint["maximum_heavy_degree"])
    except (KeyError, TypeError, ValueError) as error:
        raise UgiSelectedRestartableGeneratorError(
            "closure checkpoint support bounds are malformed"
        ) from error
    if allowed_ring_sizes != (5, 6, 7) or maximum_heavy_degree != 4:
        raise UgiSelectedRestartableGeneratorError("closure checkpoint support bounds changed")
    source_marginals = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in joint_checkpoint["source_marginals"].items()
    }
    bindings = SelectedRestartableGeneratorBindings(
        generator_checkpoint_sha256=GENERATOR_CHECKPOINT_SHA256,
        closure_checkpoint_sha256=CLOSURE_CHECKPOINT_SHA256,
        production_generator_manifest_sha256=PRODUCTION_GENERATOR_MANIFEST_SHA256,
        restartable_equivalence_receipt_sha256=(restartable_equivalence_receipt_sha256),
        generator_implementation_sha256=(implementation_qualification.implementation_sha256),
        training_cache_sha256=TRAINING_CACHE_SHA256,
        atom_vocabulary_sha256=ATOM_VOCABULARY_SHA256,
        qualified_reaction_registry_sha256=QUALIFIED_REACTION_REGISTRY_SHA256,
        l1_reaction_sha256=L1_REACTION_SHA256,
        model_config_sha256=MODEL_CONFIG_SHA256,
        declared_graph_support_sha256=graph_support_sha256,
        component_recovery_contract_sha256=COMPONENT_RECOVERY_CONTRACT_SHA256,
        sample_steps=SAMPLE_STEPS,
        terminal_decoder_id=TERMINAL_DECODER_ID,
    )
    callback = SelectedRestartableGeneratorCallback(
        model=model,
        closure_model=closure_model,
        source_marginals=source_marginals,
        corpus=corpus,
        reaction=reaction,
        graph_support=graph_support,
        l1_reverifier=reverifier,
        bindings=bindings,
        allowed_ring_sizes=allowed_ring_sizes,
        maximum_heavy_degree=maximum_heavy_degree,
        repository=resolved.repository,
        implementation_qualification=implementation_qualification,
    )
    identity = RestartableGeneratorClosureIdentity(
        generator_checkpoint_sha256=GENERATOR_CHECKPOINT_SHA256,
        closure_checkpoint_sha256=CLOSURE_CHECKPOINT_SHA256,
        production_generator_manifest_sha256=PRODUCTION_GENERATOR_MANIFEST_SHA256,
        restartable_equivalence_receipt_sha256=(restartable_equivalence_receipt_sha256),
        generator_implementation_sha256=(implementation_qualification.implementation_sha256),
        terminal_decoder_id=TERMINAL_DECODER_ID,
    )
    return SelectedRestartableGeneratorLane(
        adapter=RestartableGeneratorClosureAdapter(
            identity=identity,
            generate_locked_terminal=callback,
        ),
        callback=callback,
        bindings=bindings,
        graph_support=graph_support,
        implementation_qualification=implementation_qualification,
    )


__all__ = [
    "ATOM_VOCABULARY_SHA256",
    "CLOSURE_CHECKPOINT_SHA256",
    "COMPONENT_RECOVERY_CONTRACT_SHA256",
    "DECLARED_GRAPH_SUPPORT_SHA256",
    "GENERATOR_CHECKPOINT_SHA256",
    "L1_REACTION_SHA256",
    "MODEL_CONFIG_SHA256",
    "PRODUCTION_GENERATOR_MANIFEST_SHA256",
    "QUALIFIED_REACTION_REGISTRY_SHA256",
    "SAMPLE_STEPS",
    "SELECTED_RESTARTABLE_GENERATOR_SCHEMA_VERSION",
    "SelectedRestartableGeneratorBindings",
    "SelectedRestartableGeneratorCallback",
    "SelectedRestartableGeneratorLane",
    "SelectedStep1000Artifacts",
    "TERMINAL_DECODER_ID",
    "TRAINING_CACHE_SHA256",
    "UgiSelectedRestartableGeneratorError",
    "build_selected_step1000_restartable_generator_lane",
    "declared_graph_support_context_sha256",
]
