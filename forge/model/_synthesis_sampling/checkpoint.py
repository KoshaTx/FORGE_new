"""Authenticated checkpoint loading and source marginal construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.io import read_json_object
from forge.model._synthesis_sampling.contracts import (
    CHECKPOINT_SCHEMA,
    SynthesisProgramSamplingError,
)
from forge.model.defog_feasibility import AtomState, _model_state_sha256
from forge.model.reaction_program_conditioning import ReactionProgramVocabulary
from forge.model.synthesis_program_graph import SynthesisProgramGraphRecord
from forge.model.synthesis_program_training import build_synthesis_program_flow
from forge.model.tensor_checkpoint import TensorCheckpointError, decode_tensor_state

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]


def synthesis_program_source_marginals(
    records: Sequence[SynthesisProgramGraphRecord],
    weights: np.ndarray,
    *,
    node_classes: int,
    bond_classes: int,
    probability_floor: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit full-support sources under explicit record weights, never raw family counts."""

    if (
        not records
        or weights.shape != (len(records),)
        or np.any(weights <= 0)
        or not np.isfinite(weights).all()
        or not np.isclose(weights.sum(), 1.0)
        or probability_floor <= 0
    ):
        raise SynthesisProgramSamplingError("source marginals require a finite normalized measure")
    nodes = np.full(node_classes, probability_floor, dtype=np.float64)
    bonds = np.full(bond_classes, probability_floor, dtype=np.float64)
    for record, weight in zip(records, weights, strict=True):
        variable_atoms = record.graph.node_states[~record.fixed_atom_mask]
        variable_parent_bonds = record.graph.parent_bonds[1:][~record.fixed_parent_bond_mask[1:]]
        variable_closure_bonds = record.graph.closure_bonds[~record.fixed_closure_bond_mask]
        nodes += float(weight) * np.bincount(variable_atoms, minlength=node_classes)
        bonds += float(weight) * np.bincount(variable_parent_bonds, minlength=bond_classes)
        bonds += float(weight) * np.bincount(variable_closure_bonds, minlength=bond_classes)
    return nodes / nodes.sum(), bonds / bonds.sum()


def load_synthesis_program_checkpoint(
    checkpoint_path: Path,
    *,
    device: str,
) -> tuple[
    Any,
    ReactionProgramVocabulary,
    tuple[AtomState, ...],
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Load an authenticated non-executable tensor checkpoint."""

    if torch is None:
        raise SynthesisProgramSamplingError("checkpoint loading requires torch")
    package = read_json_object(
        checkpoint_path,
        error=SynthesisProgramSamplingError,
        label="shared synthesis-program checkpoint",
    )
    if (
        package.get("schema_version") != CHECKPOINT_SCHEMA
        or package.get("trusted_local_checkpoint") is not True
    ):
        raise SynthesisProgramSamplingError("checkpoint is not a trusted shared-flow checkpoint")
    raw_vocabulary = package.get("program_vocabulary")
    raw_atoms = package.get("atom_vocabulary")
    model_config = package.get("model_config")
    if (
        not isinstance(raw_vocabulary, Mapping)
        or not isinstance(raw_atoms, list)
        or not isinstance(model_config, Mapping)
    ):
        raise SynthesisProgramSamplingError("checkpoint is missing its model contract")
    vocabulary = ReactionProgramVocabulary(
        program_states=tuple(str(value) for value in raw_vocabulary["program_states"]),
        role_states=tuple(str(value) for value in raw_vocabulary["role_states"]),
        core_position_states=tuple(str(value) for value in raw_vocabulary["core_position_states"]),
        maximum_steps=int(raw_vocabulary["maximum_steps"]),
    )
    atom_vocabulary = tuple(
        AtomState(
            symbol=str(row["symbol"]),
            formal_charge=int(row["formal_charge"]),
            aromatic=bool(row["aromatic"]),
            explicit_hydrogens=int(row["explicit_hydrogens"]),
        )
        for row in raw_atoms
    )
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise SynthesisProgramSamplingError("CUDA sampling requested but unavailable")
    if resolved_device.type == "mps" and not torch.backends.mps.is_available():
        raise SynthesisProgramSamplingError("MPS sampling requested but unavailable")
    model = build_synthesis_program_flow(
        vocabulary=vocabulary,
        node_classes=len(atom_vocabulary),
        model_config=model_config,
        device=resolved_device,
    )
    raw_state = package.get("model_state")
    if not isinstance(raw_state, Mapping):
        raise SynthesisProgramSamplingError("checkpoint has no deterministic tensor state")
    try:
        state = decode_tensor_state(raw_state)
    except TensorCheckpointError as error:
        raise SynthesisProgramSamplingError(str(error)) from error
    model.load_state_dict(state, strict=True)
    if _model_state_sha256(model) != package.get("model_state_sha256"):
        raise SynthesisProgramSamplingError("checkpoint model-state hash mismatch")
    node_marginal = np.asarray(package.get("node_marginal"), dtype=np.float64)
    bond_marginal = np.asarray(package.get("bond_marginal"), dtype=np.float64)
    expected_prefix = (len(vocabulary.program_states), len(vocabulary.role_states))
    global_sources = node_marginal.shape == (len(atom_vocabulary),) and bond_marginal.shape == (
        int(model_config["bond_classes"]),
    )
    program_role_sources = node_marginal.shape == (
        *expected_prefix,
        len(atom_vocabulary),
    ) and bond_marginal.shape == (*expected_prefix, int(model_config["bond_classes"]))
    if (
        not (global_sources or program_role_sources)
        or np.any(node_marginal <= 0)
        or np.any(bond_marginal <= 0)
        or not np.allclose(node_marginal.sum(axis=-1), 1.0)
        or not np.allclose(bond_marginal.sum(axis=-1), 1.0)
    ):
        raise SynthesisProgramSamplingError("checkpoint source marginals are invalid")
    model.eval()
    return model, vocabulary, atom_vocabulary, node_marginal, bond_marginal, package
