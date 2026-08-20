from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from forge.potency.oracle.oracle_graph import GraphFeatureVocabulary, tensorize_smiles
from forge.potency.oracle.oracle_graph_pretraining import (
    MaskedFeatureDMPNN,
    OracleGraphPretrainingError,
    PretrainingRecord,
    class_weights,
    connectivity_split,
    deterministic_mask_indices,
    load_pretraining_config,
    make_masked_batch,
    reconstruction_loss,
    split_pretraining_records,
    standard_inchi_connectivity_key,
)

REPO = Path(__file__).resolve().parents[1]


def _vocabulary() -> GraphFeatureVocabulary:
    return GraphFeatureVocabulary(
        elements=("C", "N", "O"),
        formal_charges=("-1", "0", "1"),
        total_degrees=("1", "2", "3", "4"),
        total_hydrogens=("0", "1", "2", "3"),
        hybridizations=("SP", "SP2", "SP3"),
        bond_types=("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"),
    )


def _record(graph_id: str, smiles: str) -> PretrainingRecord:
    vocabulary = _vocabulary()
    graph = tensorize_smiles(smiles, vocabulary, label=graph_id)
    element_stop = len(vocabulary.elements)
    charge_stop = element_stop + len(vocabulary.formal_charges)
    forward = torch.arange(0, graph.num_directed_edges, 2)
    return PretrainingRecord(
        graph_id=graph_id,
        connectivity_key=standard_inchi_connectivity_key(smiles),
        graph=graph,
        element_targets=graph.node_features[:, :element_stop].argmax(dim=1),
        charge_targets=graph.node_features[:, element_stop:charge_stop].argmax(dim=1),
        bond_targets=graph.edge_features[forward, : len(vocabulary.bond_types)].argmax(dim=1),
    )


def test_config_rejects_any_biological_input(tmp_path: Path) -> None:
    config = json.loads((REPO / "configs/bio/m0_07_oracle_graph_pretraining.json").read_text())
    config["inputs"]["agile_labels"] = {
        "path": "forbidden.csv",
        "sha256": "0" * 64,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(OracleGraphPretrainingError, match="only"):
        load_pretraining_config(path)


def test_connectivity_group_split_is_deterministic_and_grouped() -> None:
    acid = _record("acid", "CC(=O)O")
    anion = _record("anion", "CC(=O)[O-]")
    assert acid.connectivity_key == anion.connectivity_key
    expected = connectivity_split(
        acid.connectivity_key,
        seed=1729,
        validation_fraction=0.5,
    )
    assert expected == connectivity_split(
        anion.connectivity_key,
        seed=1729,
        validation_fraction=0.5,
    )
    other = _record("other", "NCCO")
    records = [acid, anion, other]
    for fraction in (0.25, 0.5, 0.75):
        try:
            train, validation = split_pretraining_records(
                records,
                seed=1729,
                validation_fraction=fraction,
            )
        except OracleGraphPretrainingError:
            continue
        assignments = {record.graph_id: "train" for record in train} | {
            record.graph_id: "validation" for record in validation
        }
        assert assignments["acid"] == assignments["anion"]
        break
    else:
        raise AssertionError("test fixture did not produce two nonempty partitions")


def test_deterministic_atom_and_paired_bond_masks() -> None:
    vocabulary = _vocabulary()
    records = [_record("first", "CCO"), _record("second", "NCCO")]
    first = make_masked_batch(
        records,
        vocabulary,
        seed=1729,
        epoch=3,
        atom_mask_fraction=0.15,
        bond_mask_fraction=0.15,
    )
    second = make_masked_batch(
        list(reversed(list(reversed(records)))),
        vocabulary,
        seed=1729,
        epoch=3,
        atom_mask_fraction=0.15,
        bond_mask_fraction=0.15,
    )
    assert torch.equal(first.node_mask, second.node_mask)
    assert torch.equal(first.directed_edge_mask, second.directed_edge_mask)
    for forward in first.masked_bond_forward_edges.tolist():
        reverse = int(first.graph.reverse_edge_index[forward])
        assert first.directed_edge_mask[forward]
        assert first.directed_edge_mask[reverse]
    assert deterministic_mask_indices(
        10,
        0.15,
        seed=1729,
        epoch=3,
        graph_id="graph",
        entity="atom",
    ) == deterministic_mask_indices(
        10,
        0.15,
        seed=1729,
        epoch=3,
        graph_id="graph",
        entity="atom",
    )


def test_masked_dmpnn_loss_is_finite_and_trains_all_encoder_blocks() -> None:
    vocabulary = _vocabulary()
    records = [
        _record("neutral", "CCO"),
        _record("positive", "C[NH2+]C"),
        _record("negative", "CC(=O)[O-]"),
        _record("triple", "N#CC"),
        _record("double", "CC=O"),
        _record("aromatic", "c1ccncc1"),
    ]
    masked = make_masked_batch(
        records,
        vocabulary,
        seed=1729,
        epoch=0,
        atom_mask_fraction=0.5,
        bond_mask_fraction=0.5,
    )
    model = MaskedFeatureDMPNN(
        vocabulary,
        hidden_dim=16,
        depth=2,
        dropout=0.0,
    )
    weights = {
        target: class_weights(
            records,
            classes=classes,
            target=target,
            maximum_weight=10.0,
        )
        for target, classes in (
            ("element", len(vocabulary.elements)),
            ("charge", len(vocabulary.formal_charges)),
            ("bond", len(vocabulary.bond_types)),
        )
    }
    logits = model(masked)
    assert logits.element.shape == (
        masked.element_targets.numel(),
        len(vocabulary.elements),
    )
    assert logits.charge.shape == (
        masked.charge_targets.numel(),
        len(vocabulary.formal_charges),
    )
    assert logits.bond.shape == (
        masked.bond_targets.numel(),
        len(vocabulary.bond_types),
    )
    loss, components = reconstruction_loss(logits, masked, weights)
    assert torch.isfinite(loss)
    assert set(components) == {"element", "charge", "bond"}
    loss.backward()
    assert model.encoder.edge_input.weight.grad is not None
    assert model.encoder.atom_output.weight.grad is not None
    assert model.encoder.readout[0].weight.grad is not None
