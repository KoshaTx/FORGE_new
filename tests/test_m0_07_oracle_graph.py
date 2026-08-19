from __future__ import annotations

import csv
import dataclasses
import gzip
import inspect
import json
from pathlib import Path

import pytest
import torch
from rdkit import Chem

from forge.potency.oracle_graph import (
    DMPNNEncoder,
    EdgeGINEncoder,
    GraphFeatureVocabulary,
    GraphModelInput,
    OracleGraphError,
    UgiRoleAwareRegressor,
    WholeGraphRegressor,
    batch_graphs,
    collate_oracle_records,
    load_oracle_graph_records,
    permute_graph_tensor,
    tensorize_smiles,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def vocabulary() -> GraphFeatureVocabulary:
    return GraphFeatureVocabulary.from_corpus_result(
        REPO / "results/m0_07/oracle_graph_corpus_result.json"
    )


@pytest.fixture(scope="module")
def records(vocabulary: GraphFeatureVocabulary):
    return load_oracle_graph_records(
        REPO / "results/m0_07/agile_oracle_curated.csv.gz",
        vocabulary,
        allowed_labels={"A1B1C1", "A20B12C5"},
    )


def _dmpnn(vocabulary: GraphFeatureVocabulary) -> DMPNNEncoder:
    torch.manual_seed(1729)
    return DMPNNEncoder(
        vocabulary.atom_feature_dim,
        vocabulary.bond_feature_dim,
        hidden_dim=16,
        depth=2,
        dropout=0.0,
    )


def _gin(vocabulary: GraphFeatureVocabulary) -> EdgeGINEncoder:
    torch.manual_seed(1729)
    return EdgeGINEncoder(
        vocabulary.atom_feature_dim,
        vocabulary.bond_feature_dim,
        hidden_dim=16,
        depth=2,
        dropout=0.0,
    )


def test_graph_vocabulary_is_derived_from_frozen_corpus(
    vocabulary: GraphFeatureVocabulary,
) -> None:
    assert vocabulary.elements == ("C", "F", "N", "O", "P", "S", "Si")
    assert vocabulary.formal_charges == ("-1", "0", "1")
    assert vocabulary.total_degrees == ("1", "2", "3", "4")
    assert vocabulary.total_hydrogens == ("0", "1", "2", "3")
    assert vocabulary.hybridizations == ("SP", "SP2", "SP3")
    assert vocabulary.bond_types == ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
    assert vocabulary.atom_feature_dim == 23
    assert vocabulary.bond_feature_dim == 6


def test_tensorization_preserves_graph_and_reverse_edges(
    vocabulary: GraphFeatureVocabulary,
    records,
) -> None:
    graph = records[0].product
    molecule = Chem.MolFromSmiles("CCCCCCCCCCCCNC(=O)C(CCCCCOC(=O)CCCCCCCC)NCCN(C)C")
    assert graph.num_nodes == molecule.GetNumAtoms()
    assert graph.num_directed_edges == 2 * molecule.GetNumBonds()
    reverse = graph.reverse_edge_index
    assert torch.equal(reverse[reverse], torch.arange(graph.num_directed_edges))
    source, destination = graph.edge_index
    assert torch.equal(source, destination[reverse])
    assert torch.equal(destination, source[reverse])


def test_tensorization_retains_isocyanide_formal_charges(
    vocabulary: GraphFeatureVocabulary,
    records,
) -> None:
    graph = records[0].isocyanide
    offset = len(vocabulary.elements)
    charge_features = graph.node_features[:, offset : offset + len(vocabulary.formal_charges)]
    observed = {
        vocabulary.formal_charges[index] for index in torch.argmax(charge_features, dim=1).tolist()
    }
    assert {"-1", "1"} <= observed


def test_boundary_support_has_no_truncation(
    vocabulary: GraphFeatureVocabulary,
) -> None:
    with gzip.open(
        REPO / "results/m0_07/oracle_graph_r0_pretraining.csv.gz",
        "rt",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    maximum = max(rows, key=lambda row: int(row["atom_count"]))
    fluorinated = next(row for row in rows if "F" in row["constitutional_smiles"])
    silicon = next(row for row in rows if "[Si" in row["constitutional_smiles"])
    for row in (maximum, fluorinated, silicon):
        graph = tensorize_smiles(
            row["constitutional_smiles"],
            vocabulary,
            label=row["graph_id"],
        )
        assert graph.num_nodes == int(row["atom_count"])
    assert int(maximum["atom_count"]) == 282


def test_unknown_feature_and_disconnected_graph_fail_closed(
    vocabulary: GraphFeatureVocabulary,
) -> None:
    with pytest.raises(OracleGraphError, match="unsupported element"):
        tensorize_smiles("[Na+]", vocabulary, label="unsupported sodium")
    with pytest.raises(OracleGraphError, match="disconnected"):
        tensorize_smiles("CC.CC", vocabulary, label="two fragments")


def test_metadata_cannot_enter_model_forward(
    vocabulary: GraphFeatureVocabulary,
    records,
) -> None:
    supervised, metadata = collate_oracle_records(records)
    assert dataclasses.fields(type(metadata))[0].name == "labels"
    assert tuple(field.name for field in dataclasses.fields(GraphModelInput)) == (
        "product",
        "amine",
        "aldehyde",
        "isocyanide",
    )
    assert tuple(inspect.signature(WholeGraphRegressor.forward).parameters) == (
        "self",
        "inputs",
    )
    model = WholeGraphRegressor(_dmpnn(vocabulary), hidden_dim=16, outputs=2)
    with pytest.raises(AttributeError):
        model(supervised)


def test_changing_audit_label_does_not_change_tensors(records) -> None:
    original, _ = collate_oracle_records([records[0]])
    renamed_record = dataclasses.replace(records[0], label="arbitrary-audit-key")
    renamed, renamed_metadata = collate_oracle_records([renamed_record])
    assert renamed_metadata.labels == ("arbitrary-audit-key",)
    assert torch.equal(
        original.inputs.product.node_features,
        renamed.inputs.product.node_features,
    )
    assert torch.equal(
        original.inputs.product.edge_index,
        renamed.inputs.product.edge_index,
    )
    assert torch.equal(original.targets, renamed.targets)


@pytest.mark.parametrize("architecture", ["dmpnn", "gin"])
def test_whole_graph_models_are_permutation_and_batch_invariant(
    vocabulary: GraphFeatureVocabulary,
    records,
    architecture: str,
) -> None:
    encoder = _dmpnn(vocabulary) if architecture == "dmpnn" else _gin(vocabulary)
    model = WholeGraphRegressor(encoder, hidden_dim=16, outputs=2).eval()
    original, _ = collate_oracle_records([records[0]])
    graph = records[0].product
    generator = torch.Generator().manual_seed(1729)
    permuted = permute_graph_tensor(
        graph,
        torch.randperm(graph.num_nodes, generator=generator),
        torch.randperm(graph.num_directed_edges, generator=generator),
    )
    permuted_inputs = dataclasses.replace(
        original.inputs,
        product=batch_graphs([permuted]),
    )
    together, _ = collate_oracle_records(records)
    with torch.no_grad():
        reference = model(original.inputs)[0]
        permuted_prediction = model(permuted_inputs)[0]
        cobatched = model(together.inputs)[0]
    torch.testing.assert_close(reference, permuted_prediction, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(reference, cobatched, rtol=1e-5, atol=1e-6)


def test_role_aware_model_uses_structures_and_roles_not_component_ids(
    vocabulary: GraphFeatureVocabulary,
    records,
) -> None:
    model = UgiRoleAwareRegressor(
        _dmpnn(vocabulary),
        hidden_dim=16,
        outputs=2,
    ).eval()
    supervised, _ = collate_oracle_records([records[0]])
    generator = torch.Generator().manual_seed(42)
    permuted_graphs = []
    for graph in (
        records[0].product,
        records[0].amine,
        records[0].aldehyde,
        records[0].isocyanide,
    ):
        permuted_graphs.append(
            permute_graph_tensor(
                graph,
                torch.randperm(graph.num_nodes, generator=generator),
                torch.randperm(graph.num_directed_edges, generator=generator),
            )
        )
    permuted_inputs = GraphModelInput(
        product=batch_graphs([permuted_graphs[0]]),
        amine=batch_graphs([permuted_graphs[1]]),
        aldehyde=batch_graphs([permuted_graphs[2]]),
        isocyanide=batch_graphs([permuted_graphs[3]]),
    )
    with torch.no_grad():
        reference = model(supervised.inputs)
        permuted = model(permuted_inputs)
    torch.testing.assert_close(reference, permuted, rtol=1e-5, atol=1e-6)


def test_graph_profile_config_is_train_only() -> None:
    config = json.loads((REPO / "configs/bio/m0_07_oracle_graph_profile.json").read_text())
    assert config["partition"] == {
        "scheme": "held_aldehyde_5fold",
        "fold": 0,
        "stage": "train",
        "calibration_or_test_targets_used": False,
    }
    assert config["expected"]["profile_train_records"] == 700
    assert config["training"]["target_scaling"] == "train_rows_only_float64"
    assert config["training"]["early_stopping"] is False
    assert config["training"]["calibration_used"] is False
