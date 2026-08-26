from __future__ import annotations

import gzip
import json

import numpy as np

from experiments.phase1.multireaction.local_morphology_resampling_adjudication import (
    SAMPLES_SCHEMA,
    _audit_samples,
)
from forge.model.defog_feasibility import AtomState
from forge.model.local_chemistry_support import build_local_chemistry_support
from forge.model.sparse_topology_feasibility import SparseGraphRecord
from forge.model.synthesis_program_graph import (
    SynthesisProgramComponentBlock,
    SynthesisProgramGraphRecord,
)

PROGRAM = "test_program"
VOCABULARY = (AtomState("C", 0, False, 0), AtomState("O", 0, False, 0))


def _ring_record() -> SynthesisProgramGraphRecord:
    count = 5
    edges = np.zeros((count, count), dtype=np.int8)
    for left, right in ((0, 1), (1, 2), (2, 3), (3, 4), (0, 4)):
        edges[left, right] = edges[right, left] = 1
    graph = SparseGraphRecord(
        structure_id="supported-five-member-ring",
        canonical_smiles="C1CCCO1",
        node_states=np.asarray((0, 0, 0, 0, 1), dtype=np.int64),
        parents=np.asarray((0, 0, 1, 2, 3), dtype=np.int64),
        parent_bonds=np.zeros(count, dtype=np.int64),
        closure_left=np.asarray((0,), dtype=np.int64),
        closure_right=np.asarray((4,), dtype=np.int64),
        closure_bonds=np.zeros(1, dtype=np.int64),
        edges=edges,
    )
    return SynthesisProgramGraphRecord(
        graph=graph,
        canonical_atom_order=np.arange(count, dtype=np.int64),
        program_id=PROGRAM,
        program_state=1,
        program_depth=1,
        role_states=np.ones(count, dtype=np.int64),
        core_position_states=np.zeros(count, dtype=np.int64),
        component_blocks=(SynthesisProgramComponentBlock("tail", 1, 0, count),),
        fixed_atom_mask=np.zeros(count, dtype=np.bool_),
        fixed_parent_bond_mask=np.zeros(count, dtype=np.bool_),
        fixed_closure_bond_mask=np.zeros(1, dtype=np.bool_),
    )


def _write_sample(path, smiles: str, *, components: object | None = None) -> None:
    row = {
        "arm_id": "arm",
        "canonical_smiles": smiles,
        "checkpoint_step": 7,
        "evaluation_split": "heldout",
        "exact_l1_program": True,
        "exact_l1_traces": [
            {"components_by_role": {"tail": smiles if components is None else components}}
        ],
        "local_chemistry_policy_applied": True,
        "program_id": PROGRAM,
        "valid": True,
    }
    with gzip.open(path, "wt") as stream:
        stream.write(json.dumps({"schema_version": SAMPLES_SCHEMA, "rows": 1}) + "\n")
        stream.write(json.dumps(row) + "\n")


def _evaluation() -> dict[str, object]:
    return {
        "arm_id": "arm",
        "checkpoint_step": 7,
        "split": "heldout",
        "attempts_per_program": 1,
        "programs": [PROGRAM],
    }


def test_sample_audit_accepts_supported_five_membered_oxygen_ring(tmp_path) -> None:
    path = tmp_path / "samples.jsonl.gz"
    _write_sample(path, "C1CCCO1")
    support = build_local_chemistry_support((_ring_record(),), VOCABULARY)

    audit = _audit_samples(path, support=support, evaluation=_evaluation())[PROGRAM]

    assert audit["oxygen_containing_three_or_four_membered_rings"] == 0
    assert audit["unsupported_role_ring_signatures_in_exact_l1_components"] == 0
    assert audit["exact_l1_component_rings_assessed"] == 1


def test_sample_audit_detects_unsupported_four_membered_oxygen_ring(tmp_path) -> None:
    path = tmp_path / "samples.jsonl.gz"
    _write_sample(path, "C1CCO1")
    support = build_local_chemistry_support((_ring_record(),), VOCABULARY)

    audit = _audit_samples(path, support=support, evaluation=_evaluation())[PROGRAM]

    assert audit["oxygen_containing_three_or_four_membered_rings"] == 1
    assert audit["unsupported_role_ring_signatures_in_exact_l1_components"] == 1


def test_sample_audit_accepts_repeated_component_lists(tmp_path) -> None:
    path = tmp_path / "samples.jsonl.gz"
    _write_sample(path, "C1CCCO1", components=["C1CCCO1", "C1CCCO1"])
    support = build_local_chemistry_support((_ring_record(),), VOCABULARY)

    audit = _audit_samples(path, support=support, evaluation=_evaluation())[PROGRAM]

    assert audit["unsupported_role_ring_signatures_in_exact_l1_components"] == 0
    assert audit["exact_l1_component_rings_assessed"] == 2
