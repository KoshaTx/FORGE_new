from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

torch = pytest.importorskip("torch")

from forge.model.defog_feasibility import (  # noqa: E402
    AtomState,
    DenseGraphFlowProbe,
    GraphRecord,
    _connected,
    _is_sanitizable,
    _jensen_shannon,
    _rstar_step,
    _wasserstein_integer_support,
    audit_input_support,
    collate_records,
    deterministic_stratified_subset,
    pair_mask,
    sample_linear_interpolation,
    sha256_file,
    tensorize_row,
    topology_hash,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_rstar_step_keeps_scatter_source_on_cuda() -> None:
    device = torch.device("cuda")
    current = torch.tensor([[0, 1, 0]], device=device)
    probabilities = torch.tensor(
        [[[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]]],
        device=device,
    )
    marginal = torch.tensor([0.5, 0.5], device=device)
    valid = torch.ones_like(current, dtype=torch.bool)

    output = _rstar_step(
        current,
        probabilities,
        marginal,
        0.5,
        0.1,
        valid,
        torch.Generator(device=device).manual_seed(7),
    )

    assert output.device.type == "cuda"
    assert output.shape == current.shape


def test_rstar_step_never_enters_zero_interpolant_support() -> None:
    current = torch.tensor([[0]])
    clean_probabilities = torch.tensor([[[0.0, 1.0, 0.0]]])
    marginal = torch.tensor([0.5, 0.5, 0.0])
    valid = torch.ones_like(current, dtype=torch.bool)

    for seed in range(64):
        output = _rstar_step(
            current,
            clean_probabilities,
            marginal,
            0.0,
            0.9,
            valid,
            torch.Generator().manual_seed(seed),
        )
        assert int(output[0, 0]) in {0, 1}


def _row(structure_id: str, smiles: str) -> dict[str, str]:
    molecule = Chem.MolFromSmiles(smiles)
    return {
        "r0_structure_id": structure_id,
        "canonical_isomeric_smiles": smiles,
        "heavy_atoms": str(molecule.GetNumHeavyAtoms()),
        "elements": "|".join(sorted({atom.GetSymbol() for atom in molecule.GetAtoms()})),
    }


def test_tensorization_is_symmetric_and_preserves_explicit_no_bond() -> None:
    row = _row("test", "C[NH+](C)CC(=O)[O-]")
    vocabulary = (
        AtomState("C", 0, False),
        AtomState("N", 1, False, 1),
        AtomState("O", -1, False),
        AtomState("O", 0, False),
    )
    record = tensorize_row(row, {state: index for index, state in enumerate(vocabulary)})

    assert record.node_count == 7
    assert np.array_equal(record.edges, record.edges.T)
    assert np.all(np.diag(record.edges) == 0)
    assert np.count_nonzero(np.triu(record.edges, 1)) == 6
    assert _connected(record.edges)
    assert _is_sanitizable(record.node_states, record.edges, vocabulary)


def test_linear_interpolation_endpoints_and_edge_symmetry() -> None:
    generator = torch.Generator().manual_seed(7)
    clean = torch.tensor([[[0, 1, 0], [1, 0, 2], [0, 2, 0]]])
    valid = torch.tensor([[[False, True, True], [False, False, True], [False, False, False]]])
    marginal = torch.tensor([0.8, 0.15, 0.05])

    at_clean = sample_linear_interpolation(clean, marginal, torch.ones(1), valid, generator)
    at_noise = sample_linear_interpolation(clean, marginal, torch.zeros(1), valid, generator)

    assert torch.equal(at_clean, clean)
    assert torch.equal(at_noise, at_noise.transpose(1, 2))


def test_dense_model_is_permutation_equivariant_without_dropout() -> None:
    torch.manual_seed(11)
    model = DenseGraphFlowProbe(4, 5, hidden_dim=16, layers=2, dropout=0.0)
    nodes = torch.tensor([[0, 1, 2, 3]])
    edges = torch.tensor([[[0, 1, 0, 2], [1, 0, 1, 0], [0, 1, 0, 1], [2, 0, 1, 0]]])
    mask = torch.ones((1, 4), dtype=torch.bool)
    permutation = torch.tensor([2, 0, 3, 1])

    node_logits, edge_logits = model(nodes, edges, torch.tensor([0.4]), mask)
    permuted_nodes = nodes[:, permutation]
    permuted_edges = edges[:, permutation][:, :, permutation]
    p_node_logits, p_edge_logits = model(permuted_nodes, permuted_edges, torch.tensor([0.4]), mask)

    assert torch.allclose(p_node_logits, node_logits[:, permutation], atol=1e-6)
    expected_edges = edge_logits[:, permutation][:, :, permutation]
    assert torch.allclose(p_edge_logits, expected_edges, atol=1e-6)


def test_stratified_subset_and_collation_are_deterministic() -> None:
    records = []
    for index, count in enumerate((8, 10, 35, 40, 52, 60, 70, 88)):
        nodes = np.zeros(count, dtype=np.int64)
        edges = np.zeros((count, count), dtype=np.int64)
        for position in range(count - 1):
            edges[position, position + 1] = edges[position + 1, position] = 1
        records.append(GraphRecord(str(index), "C", nodes, edges))
    first = deterministic_stratified_subset(records, 6, 3)
    second = deterministic_stratified_subset(records, 6, 3)
    assert [record.structure_id for record in first] == [record.structure_id for record in second]
    assert len({_connected(record.edges) for record in first}) == 1

    nodes, edges, mask = collate_records(first[:2], 96, np.random.default_rng(5), permute=True)
    assert nodes.shape == (2, 96)
    assert edges.shape == (2, 96, 96)
    assert pair_mask(mask, upper_only=True).sum() == sum(
        record.node_count * (record.node_count - 1) // 2 for record in first[:2]
    )


def test_distribution_metrics_and_topology_hash() -> None:
    same = np.asarray([0.0, 0.5, 0.5])
    shifted = np.asarray([0.0, 1.0, 0.0])
    assert _jensen_shannon(same, same) == pytest.approx(0.0)
    assert _wasserstein_integer_support(same, same) == pytest.approx(0.0)
    assert _jensen_shannon(same, shifted) > 0
    assert _wasserstein_integer_support(same, shifted) > 0

    path = np.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    permuted = path[np.ix_([2, 0, 1], [2, 0, 1])]
    triangle = np.asarray([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    assert topology_hash(path) == topology_hash(permuted)
    assert topology_hash(path) != topology_hash(triangle)


@pytest.mark.needs_vendor
def test_hash_pinned_r0_support_profile_is_reported_from_current_input() -> None:
    config = json.loads((REPO / "configs/model/m0_06_defog_feasibility.json").read_text())
    with (REPO / "data/vendor/r0_observed_real_structures.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    audit = audit_input_support(
        rows,
        set(config["declared_support"]["elements"]),
        config["declared_support"]["documented_profile_to_audit"],
        config["decision_thresholds"]["support_profile_fraction_tolerance"],
    )

    assert audit["actual_full_r0"]["rows"] == 15_433
    assert audit["actual_full_r0"]["n64_count"] == 10_877
    assert audit["actual_full_r0"]["n96_count"] == 14_079
    assert audit["actual_full_r0"]["maximum_heavy_atoms"] == 282
    assert audit["rows_outside_declared_element_vocabulary"] == 344
    assert audit["unsupported_element_row_incidence"] == {"F": 92, "Si": 252}
    assert audit["profile_matches"] is False


@pytest.mark.needs_vendor
def test_frozen_result_contains_both_sizes_objectives_and_required_metrics() -> None:
    result_path = REPO / "results/m0_06/result.json"
    if not result_path.exists():
        pytest.skip("M0-06 result has not been generated")
    result = json.loads(result_path.read_text())

    assert result["schema_version"] == "m0_06_defog_feasibility_result.v1"
    assert result["status"] == "completed_bounded_probe"
    assert result["scope"].startswith("M0 feasibility probe only")
    assert result["decision"]["bounded_probe_quality_established"] is False
    assert result["decision"]["recommendation"] == (
        "adopt_sparse_or_hierarchical_edge_parameterization_before_full_product_prior"
    )
    assert result["decision"]["support_profile_drift"] is True
    assert {(run["objective"]["name"], run["n_max"]) for run in result["runs"]} == {
        ("standard_edge_ce", 64),
        ("standard_edge_ce", 96),
        ("bond_recall_auxiliary", 64),
        ("bond_recall_auxiliary", 96),
    }

    for run in result["runs"]:
        assert run["training"]["graphs_per_second"] > 0
        assert run["memory"]["peak_rss_after_bytes"] > 0
        assert set(run["heldout_reconstruction"]["metrics"]) == {
            "node_accuracy",
            "edge_accuracy",
            "bond_recall",
            "null_false_positive_rate",
            "exact_graph_reconstruction",
        }
        assert set(run["endpoints"]) >= {
            "endpoint_validity",
            "connectedness",
            "atom_count_fidelity",
            "bond_sparsity",
            "topology_novelty_among_valid",
        }

    for record in result["inputs"].values():
        path = REPO / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert sha256_file(path) == record["sha256"]
    config_path = REPO / result["config"]["path"]
    assert sha256_file(config_path) == result["config"]["sha256"]
