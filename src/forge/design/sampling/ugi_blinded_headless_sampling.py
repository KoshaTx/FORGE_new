"""Renderer-free Ugi joint-flow preflight and molecular completion.

This module exists for blinded execution.  It deliberately has no dependency
on the ordinary sampling modules that import ``rdkit.Chem.Draw`` or write
molecular figures.  Runtime preflight deserializes every model input, verifies
the checkpoint-owned prepared-cache hash, constructs both models, loads their
states, validates the reaction contract and parses the sealed morphology
programs before any irreversible one-shot claim is created.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

from forge.data.r1_prime_audit import sha256_file
from forge.design.ugi_adapter_features import ORIGIN_TO_INDEX
from forge.design.ugi_chemistry_flow import (
    chemistry_sample_statistics,
    chemistry_sample_to_molecule,
    valence_constrained_terminal_sample,
)
from forge.design.ugi_chemistry_interface import assemble_ugi_chemistry_topology_condition
from forge.design.ugi_closure_placement import UgiSparseClosureScorer, sample_sparse_closures
from forge.design.ugi_generated_components import (
    UgiGeneratedComponentError,
    generated_ugi_component_smiles,
)
from forge.design.ugi_held_component_gate import (
    exact_forward_reconstructs_ugi_product,
    load_ugi_reaction_contract,
)
from forge.design.ugi_joint_sparse_flow import (
    UgiJointSparseFlow,
    UgiJointSparseTerminal,
    sample_ugi_joint_sparse_terminals,
)
from forge.design.ugi_morphology_program import UgiMorphologyProgram
from forge.design.ugi_training_cache import load_ugi_training_cache
from forge.potency.audit.ugi_semantic_annotations import ROLE_NAMES

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - checked by runtime preflight
    torch = None


JOINT_CHECKPOINT_SCHEMA = "phase1_ugi_joint_sparse_checkpoint.v1"
CLOSURE_CHECKPOINT_SCHEMAS = (
    "phase1_ugi_sparse_closure_checkpoint.v1",
    "phase1_ugi_sparse_closure_checkpoint.v2",
)


class UgiBlindedHeadlessSamplingError(RuntimeError):
    """Raised when renderer-free preflight or completion fails closed."""


@dataclass(frozen=True)
class HeadlessRuntime:
    """Fully loaded resources that are safe to carry across the one-shot claim."""

    model: Any
    closure_model: Any
    corpus: Any
    l1_reaction: Any
    programs: tuple[UgiMorphologyProgram, ...]
    program_metadata: tuple[dict[str, Any], ...]
    closure_checkpoint: dict[str, Any]
    source_marginals: dict[str, Any]
    preflight_summary: dict[str, Any]


@dataclass(frozen=True)
class HeadlessCompletion:
    """Terminal completion state without a rendering surface."""

    conditions: tuple[Any, ...]
    samples: tuple[Any, ...]
    rows: tuple[dict[str, Any], ...]
    closure_generator_state: Any
    terminal_generator_state: Any | None


def _load_checkpoint(path: Path, schemas: str | tuple[str, ...]) -> dict[str, Any]:
    if torch is None or not path.is_file():
        raise UgiBlindedHeadlessSamplingError(f"missing checkpoint: {path}")
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise UgiBlindedHeadlessSamplingError(f"checkpoint failed to deserialize: {path}") from exc
    supported = (schemas,) if isinstance(schemas, str) else schemas
    if not isinstance(value, dict) or value.get("schema_version") not in supported:
        raise UgiBlindedHeadlessSamplingError(f"unexpected checkpoint schema: {path}")
    return value


def _resolve_checkpoint_cache(
    repo: Path,
    checkpoint: Mapping[str, Any],
    *,
    expected_path: Path,
    expected_sha256: str,
) -> Path:
    inputs = checkpoint.get("inputs")
    specification = inputs.get("prepared_cache") if isinstance(inputs, dict) else None
    if not isinstance(specification, dict) or set(specification) != {"path", "sha256"}:
        raise UgiBlindedHeadlessSamplingError(
            "joint checkpoint lacks one exact prepared-cache specification"
        )
    raw_path = Path(str(specification["path"]))
    path = raw_path
    if not path.is_file() and str(raw_path).startswith("/root/forge_repo/"):
        path = repo / raw_path.relative_to("/root/forge_repo")
    if path.resolve() != expected_path.resolve():
        raise UgiBlindedHeadlessSamplingError("checkpoint prepared-cache path changed")
    embedded_hash = specification.get("sha256")
    if embedded_hash != expected_sha256:
        raise UgiBlindedHeadlessSamplingError("checkpoint prepared-cache hash pin changed")
    if sha256_file(path) != embedded_hash:
        raise UgiBlindedHeadlessSamplingError("checkpoint prepared-cache bytes changed")
    return path


def _load_matched_programs(
    path: Path,
) -> tuple[tuple[UgiMorphologyProgram, ...], tuple[dict[str, Any], ...]]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise UgiBlindedHeadlessSamplingError("sealed morphology programs are unreadable") from exc
    rows = value.get("samples") if isinstance(value, dict) else None
    if not isinstance(rows, list) or not rows:
        raise UgiBlindedHeadlessSamplingError("sealed morphology program draw is empty")
    programs: list[UgiMorphologyProgram] = []
    metadata: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("program"), dict):
            raise UgiBlindedHeadlessSamplingError(f"sealed program row {index} is malformed")
        program = row["program"]
        try:
            programs.append(
                UgiMorphologyProgram(
                    node_counts=tuple(program["node_counts"]),
                    junction_budgets=tuple(program["junction_budgets"]),
                    cycle_ranks=tuple(program["cycle_ranks"]),
                    attachment_counts=tuple(program.get("attachment_counts", (1, 1, 1))),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise UgiBlindedHeadlessSamplingError(
                f"sealed program row {index} violates the morphology schema"
            ) from exc
        metadata.append(
            {
                key: row[key]
                for key in (
                    "product_id",
                    "source_stratum",
                    "branch_class",
                    "component_novelty_class",
                    "held_role_class",
                )
                if key in row
            }
        )
    return tuple(programs), tuple(metadata)


def preflight_headless_runtime(
    repo: Path,
    *,
    joint_checkpoint_path: Path,
    closure_checkpoint_path: Path,
    prepared_cache_path: Path,
    prepared_cache_sha256: str,
    qualified_reactions_path: Path,
    program_draw_path: Path,
    expected_program_rows: int,
) -> HeadlessRuntime:
    """Load all runtime dependencies without generating a molecular sample."""

    if torch is None:
        raise UgiBlindedHeadlessSamplingError("renderer-free preflight requires torch")
    joint_checkpoint = _load_checkpoint(joint_checkpoint_path, JOINT_CHECKPOINT_SCHEMA)
    closure_checkpoint = _load_checkpoint(closure_checkpoint_path, CLOSURE_CHECKPOINT_SCHEMAS)
    cache_path = _resolve_checkpoint_cache(
        repo,
        joint_checkpoint,
        expected_path=prepared_cache_path,
        expected_sha256=prepared_cache_sha256,
    )
    try:
        corpus, records_by_fold = load_ugi_training_cache(cache_path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise UgiBlindedHeadlessSamplingError("prepared training cache failed to load") from exc

    architecture = dict(joint_checkpoint.get("model_config", {}))
    architecture.pop("source_probability_floor", None)
    try:
        model = UgiJointSparseFlow(atom_classes=len(corpus.atom_vocabulary), **architecture)
        model.load_state_dict(joint_checkpoint["model_state"], strict=True)
        closure_model = UgiSparseClosureScorer(**closure_checkpoint["model_config"])
        closure_model.load_state_dict(closure_checkpoint["model_state_dict"], strict=True)
        model.eval()
        closure_model.eval()
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise UgiBlindedHeadlessSamplingError(
            "checkpoint model construction or state loading failed"
        ) from exc
    try:
        l1_reaction = load_ugi_reaction_contract(qualified_reactions_path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise UgiBlindedHeadlessSamplingError("qualified Ugi reaction failed to load") from exc
    programs, program_metadata = _load_matched_programs(program_draw_path)
    if len(programs) != expected_program_rows:
        raise UgiBlindedHeadlessSamplingError("sealed program row count changed")
    source_marginals = joint_checkpoint.get("source_marginals")
    if not isinstance(source_marginals, dict) or not source_marginals:
        raise UgiBlindedHeadlessSamplingError("joint checkpoint lacks source marginals")

    summary = {
        "status": "runtime_dependencies_preflighted_without_sampling",
        "joint_checkpoint_schema": joint_checkpoint["schema_version"],
        "closure_checkpoint_schema": closure_checkpoint["schema_version"],
        "prepared_cache_path": str(cache_path),
        "prepared_cache_sha256": prepared_cache_sha256,
        "prepared_cache_schema": "phase1_ugi_training_cache.v1",
        "prepared_cache_folds": sorted(records_by_fold),
        "atom_vocabulary_size": len(corpus.atom_vocabulary),
        "program_rows": len(programs),
        "qualified_reaction_loaded": True,
        "models_constructed_and_state_loaded": True,
        "molecular_sampling_performed": False,
        "rdkit_molecular_renderer_imported": False,
    }
    return HeadlessRuntime(
        model=model,
        closure_model=closure_model,
        corpus=corpus,
        l1_reaction=l1_reaction,
        programs=programs,
        program_metadata=program_metadata,
        closure_checkpoint=closure_checkpoint,
        source_marginals=dict(source_marginals),
        preflight_summary=summary,
    )


def _annotate_l1_terminal_admission(row: dict[str, Any], reaction: Any) -> None:
    row["raw_molecule_valid"] = bool(row.get("valid"))
    if not row["raw_molecule_valid"]:
        row["terminal_valid"] = False
        row["terminal_failure_type"] = row.get("failure_type") or "RawMoleculeInvalid"
        return
    if not row.get("component_reconstruction_valid"):
        row["terminal_valid"] = False
        row["terminal_failure_type"] = "L1ComponentRecoveryFailure"
        return
    exact_forward, saturated, outcome_count = exact_forward_reconstructs_ugi_product(
        reaction,
        row["component_smiles_by_role"],
        row["smiles"],
    )
    row["l1_forward_verification"] = {
        "exact_product_reconstructed": exact_forward,
        "maximum_outcomes_saturated": saturated,
        "outcome_count": outcome_count,
    }
    row["terminal_valid"] = exact_forward
    row["terminal_failure_type"] = None if exact_forward else "L1ForwardConsistencyFailure"


def complete_headless_terminals(
    runtime: HeadlessRuntime,
    terminals: Sequence[UgiJointSparseTerminal],
    *,
    closure_generator_state: Any,
    terminal_decoder_mode: str,
    terminal_generator_state: Any,
    terminal_temperature: float,
) -> HeadlessCompletion:
    """Complete terminal logits into private molecular rows without rendering."""

    if torch is None or len(terminals) != len(runtime.program_metadata):
        raise UgiBlindedHeadlessSamplingError("terminal completion inputs are misaligned")
    closure_generator = torch.Generator()
    closure_generator.set_state(closure_generator_state)
    terminal_generator = torch.Generator()
    terminal_generator.set_state(terminal_generator_state)
    conditions: list[Any] = []
    samples: list[Any] = []
    rows: list[dict[str, Any]] = []
    model = runtime.model
    corpus = runtime.corpus
    closure_checkpoint = runtime.closure_checkpoint
    for sample_index, terminal in enumerate(terminals):
        offspring_by_role = {
            role: terminal.offspring[role_index] for role_index, role in enumerate(ROLE_NAMES)
        }
        left_by_role: dict[str, Any] = {}
        right_by_role: dict[str, Any] = {}
        for role_index, role in enumerate(ROLE_NAMES):
            left, right = sample_sparse_closures(
                runtime.closure_model,
                terminal.offspring[role_index],
                role_index=role_index,
                cycle_rank=terminal.program.cycle_ranks[role_index],
                attachment_count=terminal.program.attachment_counts[role_index],
                generator=closure_generator,
                allowed_ring_sizes=closure_checkpoint["allowed_ring_sizes"],
                maximum_heavy_degree=int(closure_checkpoint["maximum_heavy_degree"]),
                device="cpu",
            )
            left_by_role[role] = left
            right_by_role[role] = right
        condition = assemble_ugi_chemistry_topology_condition(
            structure_id=f"joint_generated_{sample_index:04d}",
            offspring_by_role=offspring_by_role,
            attachment_counts_by_role={
                role: terminal.program.attachment_counts[role_index]
                for role_index, role in enumerate(ROLE_NAMES)
            },
            closure_left_by_role=left_by_role,
            closure_right_by_role=right_by_role,
            schema=corpus.core_schema,
        )
        exterior_full_indices: list[int] = []
        for role in ROLE_NAMES:
            exterior_full_indices.extend(
                np.flatnonzero(
                    (condition.origin_states == ORIGIN_TO_INDEX[role]) & ~condition.fixed_atom_mask
                ).tolist()
            )
        if len(exterior_full_indices) != terminal.program.node_count:
            raise UgiBlindedHeadlessSamplingError("joint exterior/full topology mismatch")
        sequence_by_full = {full: sequence for sequence, full in enumerate(exterior_full_indices)}
        atom_logits = torch.zeros(
            (1, condition.node_count, len(corpus.atom_vocabulary)), dtype=torch.float32
        )
        parent_logits = torch.zeros((1, condition.node_count, 4), dtype=torch.float32)
        for sequence, full in enumerate(exterior_full_indices):
            atom_logits[0, full] = torch.from_numpy(terminal.atom_logits[sequence])
            parent_logits[0, full] = torch.from_numpy(terminal.parent_bond_logits[sequence])
        closure_sequence_pairs = [
            (sequence_by_full[int(left)], sequence_by_full[int(right)])
            for left, right in zip(condition.closure_left, condition.closure_right, strict=True)
        ]
        if closure_sequence_pairs:
            hidden = torch.from_numpy(terminal.hidden)
            closure_features = torch.stack(
                [torch.cat((hidden[left], hidden[right])) for left, right in closure_sequence_pairs]
            )
            closure_logits = model.closure_bond_output(closure_features)[None, :, :]
        else:
            closure_logits = torch.zeros((1, 0, 4), dtype=torch.float32)
        decoration_anchor_logits = torch.full(
            (1, model.maximum_decorations, condition.node_count + 1),
            -torch.inf,
            dtype=torch.float32,
        )
        decoration_anchor_logits[:, :, 0] = torch.from_numpy(
            terminal.decoration_anchor_logits[:, 0]
        )
        for sequence, full in enumerate(exterior_full_indices):
            decoration_anchor_logits[0, :, full + 1] = torch.from_numpy(
                terminal.decoration_anchor_logits[:, sequence + 1]
            )
        terminal_chemistry = {
            "nodes": atom_logits,
            "parent_bonds": parent_logits,
            "closure_bonds": closure_logits,
        }
        if model.maximum_decorations == 1:
            terminal_chemistry["decoration_anchor"] = decoration_anchor_logits[:, 0, :]
        else:
            terminal_chemistry.update(
                {
                    "decoration_anchors": decoration_anchor_logits,
                    "decoration_atoms": torch.from_numpy(terminal.decoration_atom_logits)[
                        None, :, :
                    ],
                    "decoration_bonds": torch.from_numpy(terminal.decoration_bond_logits)[
                        None, :, :
                    ],
                }
            )
        try:
            decoded = valence_constrained_terminal_sample(
                condition,
                terminal_chemistry,
                0,
                corpus.atom_vocabulary,
                model.maximum_decorations,
                mode=terminal_decoder_mode,
                generator=terminal_generator,
                temperature=terminal_temperature,
            )
        except RuntimeError:
            decoded = None
        conditions.append(condition)
        samples.append(decoded)
        rows.append(
            {
                "structure_id": condition.structure_id,
                **runtime.program_metadata[sample_index],
                "program": terminal.program.__dict__,
                "offspring_by_role": {
                    role: offspring_by_role[role].tolist() for role in ROLE_NAMES
                },
                "terminal_decoder_mode": terminal_decoder_mode,
                "terminal_temperature": terminal_temperature,
            }
        )

    for row, condition, sample in zip(rows, conditions, samples, strict=True):
        if sample is None:
            row.update(
                {
                    "smiles": None,
                    "valid": False,
                    "failure_type": "TerminalSupportFailure",
                    "component_smiles_by_role": None,
                    "component_reconstruction_valid": False,
                    "component_reconstruction_error": "terminal support failed",
                }
            )
            _annotate_l1_terminal_admission(row, runtime.l1_reaction)
            continue
        try:
            molecule = chemistry_sample_to_molecule(condition, sample, corpus.atom_vocabulary)
            row.update(
                {
                    "smiles": Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False),
                    "valid": True,
                    "failure_type": None,
                }
            )
            try:
                row["component_smiles_by_role"] = generated_ugi_component_smiles(
                    condition,
                    sample,
                    corpus.atom_vocabulary,
                )
                row["component_reconstruction_valid"] = True
                row["component_reconstruction_error"] = None
            except UgiGeneratedComponentError as exc:
                row["component_smiles_by_role"] = None
                row["component_reconstruction_valid"] = False
                row["component_reconstruction_error"] = str(exc)
            _annotate_l1_terminal_admission(row, runtime.l1_reaction)
        except (ValueError, RuntimeError):
            row.update(
                {
                    "smiles": None,
                    "valid": False,
                    "failure_type": "MoleculeSanitizationFailure",
                    "component_smiles_by_role": None,
                    "component_reconstruction_valid": False,
                    "component_reconstruction_error": "product molecule is invalid",
                }
            )
            _annotate_l1_terminal_admission(row, runtime.l1_reaction)
    return HeadlessCompletion(
        conditions=tuple(conditions),
        samples=tuple(samples),
        rows=tuple(rows),
        closure_generator_state=closure_generator.get_state().clone(),
        terminal_generator_state=terminal_generator.get_state().clone(),
    )


def sample_headless_runtime(
    runtime: HeadlessRuntime,
    *,
    settings: Mapping[str, Any],
    pinned_inputs: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Generate the sealed rows once using already-preflighted resources."""

    if torch is None:
        raise UgiBlindedHeadlessSamplingError("renderer-free sampling requires torch")
    started = time.perf_counter()
    terminals, joint_sampling = sample_ugi_joint_sparse_terminals(
        runtime.model,
        runtime.programs,
        {
            key: np.asarray(value, dtype=np.float64)
            for key, value in runtime.source_marginals.items()
        },
        sample_steps=settings["sample_steps"],
        batch_size=settings["batch_size"],
        seed=settings["flow_seed"],
        device=settings["device"],
        allowed_ring_sizes=runtime.closure_checkpoint["allowed_ring_sizes"],
        maximum_heavy_degree=int(runtime.closure_checkpoint["maximum_heavy_degree"]),
        maximum_adjacent_branch_runs=settings["maximum_adjacent_branch_runs"],
    )
    completion = complete_headless_terminals(
        runtime,
        terminals,
        closure_generator_state=torch.Generator().manual_seed(settings["closure_seed"]).get_state(),
        terminal_decoder_mode=settings["terminal_decoder_mode"],
        terminal_generator_state=torch.Generator()
        .manual_seed(settings["terminal_seed"])
        .get_state(),
        terminal_temperature=settings["terminal_temperature"],
    )
    rows = [dict(row) for row in completion.rows]
    return {
        "schema_version": "phase1_ugi3_route_saturation_private_sample.v1",
        "status": "sampled_once_headless",
        "visibility": "private_identity_bearing_hash_pinned",
        "render": None,
        "rendered_molecules": 0,
        "rendering_disabled": True,
        "samples": rows,
        "statistics": chemistry_sample_statistics(
            completion.conditions,
            completion.samples,
            runtime.corpus.atom_vocabulary,
        ),
        "sampling": {
            **joint_sampling,
            "total_seconds": time.perf_counter() - started,
            "matched_global_programs": len(runtime.programs),
            "evaluate_exact_l1_terminal_admission": True,
            "terminal_decoder": {
                "mode": settings["terminal_decoder_mode"],
                "seed": settings["terminal_seed"],
                "temperature": settings["terminal_temperature"],
            },
            "closure_seed": settings["closure_seed"],
        },
        "pinned_inputs": dict(pinned_inputs),
    }
