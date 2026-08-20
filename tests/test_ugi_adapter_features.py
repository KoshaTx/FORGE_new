from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np

from experiments.archive.phase1.design_audits.canonical_representation_audit import (
    load_atom_vocabulary,
)
from forge.model.ugi_adapter_features import (
    ASSEMBLY_INTRODUCED,
    CORE_POSITION_TO_INDEX,
    ORIGIN_TO_INDEX,
    PORT_TO_INDEX,
    tensorize_ugi_l1_sparse_record,
    tensorize_ugi_l1_support_record,
    ugi_adapter_features,
)
from forge.model.v5_sparse_representation import v5_constitutional_roundtrip_exact
from forge.potency.annotations import ROLE_NAMES, annotate_qualified_ugi_product
from forge.synthesis.engine.qualified_forward import load_qualified_forward_reaction

REPO = Path(__file__).resolve().parents[1]


def _semantic_example():
    with gzip.open(
        REPO / "data/splits/phase1/ugi_l1_assignments.csv.gz",
        "rt",
        newline="",
    ) as handle:
        row = next(csv.DictReader(handle))
    compiled = load_qualified_forward_reaction(
        REPO / "data/vendor/qualified_reactions_v1.json",
        REPO / "configs/assembly/ugi_variant.yaml",
        reaction_id="ugi_3cr_agile",
    )
    product, atoms, _, _ = annotate_qualified_ugi_product(
        compiled,
        product_id=row["product_id"],
        target_smiles=row["canonical_product_smiles"],
        component_smiles_by_role={role: row[f"{role}_smiles"] for role in ROLE_NAMES},
        source_evidence_record_id="unit_test",
        max_outcomes=100,
    )
    return product, atoms


def test_ugi_adapter_features_keep_origin_and_core_orthogonal() -> None:
    product, atoms = _semantic_example()
    features = ugi_adapter_features(product, atoms)

    assert features.node_count == len(atoms)
    assert int(features.core_membership.sum()) == 5
    assert np.count_nonzero(features.port_states) == 3
    assert features.distances_to_all_ports.shape == (len(atoms), 3)
    introduced = features.origin_states == ORIGIN_TO_INDEX[ASSEMBLY_INTRODUCED]
    assert int(introduced.sum()) == 1
    assert bool(features.core_membership[int(np.flatnonzero(introduced)[0])])
    for role in ROLE_NAMES:
        anchor = int(np.flatnonzero(features.port_states == PORT_TO_INDEX[role])[0])
        assert features.origin_states[anchor] == ORIGIN_TO_INDEX[role]
        assert features.core_membership[anchor]
        assert features.distance_to_own_port[anchor] == 0
    assert set(features.core_position_states[features.core_membership]) == {
        CORE_POSITION_TO_INDEX["map_1"],
        CORE_POSITION_TO_INDEX["map_2"],
        CORE_POSITION_TO_INDEX["map_3"],
        CORE_POSITION_TO_INDEX["map_4"],
        CORE_POSITION_TO_INDEX["template_introduced_0"],
    }


def test_ugi_sparse_serialization_roots_in_core_without_using_component_ids() -> None:
    product, atoms = _semantic_example()
    vocabulary = load_atom_vocabulary(REPO / "results/phase1/product_v3_atom_vocabulary.json")
    record = tensorize_ugi_l1_sparse_record(
        product,
        atoms,
        {state: index for index, state in enumerate(vocabulary)},
        preserve_aromaticity=True,
    )

    assert record.adapter.core_membership[0]
    assert record.graph.node_count == record.adapter.node_count
    assert sorted(record.canonical_atom_order.tolist()) == list(range(record.graph.node_count))
    assert v5_constitutional_roundtrip_exact(record.graph, vocabulary)


def test_ugi_support_record_protects_core_and_retains_full_expansion_target() -> None:
    product, atoms = _semantic_example()
    vocabulary = load_atom_vocabulary(REPO / "results/phase1/product_v3_atom_vocabulary.json")
    record = tensorize_ugi_l1_support_record(
        product,
        atoms,
        {state: index for index, state in enumerate(vocabulary)},
        preserve_aromaticity=True,
    )

    assert record.support_graph.node_count == record.support_adapter.node_count
    assert record.support_graph.node_count == record.skeleton.retained_count
    assert record.full_adapter.node_count == record.skeleton.node_count
    assert record.support_adapter.core_membership[0]
    assert int(record.support_adapter.core_membership.sum()) == 5
    assert len(set(record.support_full_atom_order.tolist())) == record.support_graph.node_count
    assert v5_constitutional_roundtrip_exact(record.support_graph, vocabulary)
