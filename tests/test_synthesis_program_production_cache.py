from __future__ import annotations

from pathlib import Path

import numpy as np
from rdkit import Chem

from forge.core.hashing import sha256_file
from forge.corpus.synthesis_program_production_cache import (
    CACHE_SCHEMA,
    SynthesisProgramProductionCache,
    SynthesisProgramSourceRecord,
    _PackedArrays,
    _write_deterministic_npz,
)
from forge.model.defog_feasibility import AtomState
from forge.model.reaction_program_conditioning import ReactionProgramVocabulary
from forge.model.synthesis_program_graph import tensorize_synthesis_program_product


def _atom_vocabulary(smiles: str) -> tuple[AtomState, ...]:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return tuple(
        sorted(
            {
                AtomState(
                    atom.GetSymbol(),
                    atom.GetFormalCharge(),
                    atom.GetIsAromatic(),
                    atom.GetNumExplicitHs(),
                )
                for atom in molecule.GetAtoms()
            },
            key=AtomState.key,
        )
    )


def _mini_cache(path: Path) -> None:
    smiles = "CN"
    atom_vocabulary = _atom_vocabulary(smiles)
    program_ids = ("bl", "lx", "ugi")
    vocabulary = ReactionProgramVocabulary.from_semantics(
        program_ids=program_ids,
        roles=("head", "tail"),
        core_positions=tuple(f"{program}:core" for program in program_ids),
        maximum_steps=2,
    )
    packed = _PackedArrays()
    for index, program_id in enumerate(program_ids):
        record = tensorize_synthesis_program_product(
            record_id=f"record-{program_id}",
            program_id=program_id,
            canonical_product_smiles=smiles,
            atom_roles=("head", "tail"),
            atom_core_positions=(f"{program_id}:core", "exterior"),
            program_depth=1,
            vocabulary=vocabulary,
            atom_vocabulary=atom_vocabulary,
        )
        packed.append(
            SynthesisProgramSourceRecord(
                record=record,
                fold="train",
                source_weight=float(index + 1),
            )
        )
    metadata = {
        "schema_version": CACHE_SCHEMA,
        "program_vocabulary": {
            "program_states": list(vocabulary.program_states),
            "role_states": list(vocabulary.role_states),
            "core_position_states": list(vocabulary.core_position_states),
            "maximum_steps": vocabulary.maximum_steps,
        },
        "atom_vocabulary": [
            {
                "symbol": state.symbol,
                "formal_charge": state.formal_charge,
                "aromatic": state.aromatic,
                "explicit_hydrogens": state.explicit_hydrogens,
            }
            for state in atom_vocabulary
        ],
        "fold_counts": {
            program: {"train": 1, "calibration": 0, "heldout": 0} for program in program_ids
        },
    }
    _write_deterministic_npz(path, packed.arrays(metadata))


def test_packed_cache_roundtrips_records_and_balances_program_mass(tmp_path: Path) -> None:
    path = tmp_path / "cache.npz"
    _mini_cache(path)

    with SynthesisProgramProductionCache(path) as cache:
        assert len(cache) == 3
        assert cache.fold_counts() == {
            program: {"train": 1, "calibration": 0, "heldout": 0} for program in ("bl", "lx", "ugi")
        }
        measure = cache.training_measure({"bl": 1 / 3, "lx": 1 / 3, "ugi": 1 / 3})
        assert np.allclose(measure, np.full(3, 1 / 3))
        assert cache.record(2).graph.edges.shape == (0, 0)
        assert cache.record(2).graph.canonical_smiles == "CN"
        nodes, bonds = cache.source_marginals(
            measure,
            node_classes=len(cache.atom_vocabulary),
            bond_classes=4,
            probability_floor=1e-5,
        )
        assert np.isclose(nodes.sum(), 1.0)
        assert np.isclose(bonds.sum(), 1.0)
        assert np.all(nodes > 0)
        assert np.all(bonds > 0)


def test_packed_cache_bytes_are_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _mini_cache(first)
    _mini_cache(second)
    assert sha256_file(first) == sha256_file(second)
