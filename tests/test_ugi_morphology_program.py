from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

import forge.model.ugi_morphology_program as morphology_program
from experiments.archive.phase1.design_audits.canonical_representation_audit import (
    load_atom_vocabulary,
)
from forge.model.ugi_adapter_features import tensorize_ugi_l1_support_record
from forge.model.ugi_morphology_program import (
    UgiProductMorphology,
    attached_tree_matches_program,
    component_weighted_offspring_marginals,
    component_weighted_program_pool,
    maximum_adjacent_branch_graph_run,
    preorder_attached_forest_to_parents,
    sample_attached_offspring_with_exact_budget,
    sample_attached_offspring_with_exact_budget_and_cycle_rank,
    sample_attached_offspring_without_budget,
    sample_component_weighted_programs,
    split_ugi_support_morphology,
)
from forge.potency.annotations import ROLE_NAMES

torch = pytest.importorskip("torch")
REPO = Path(__file__).resolve().parents[1]


def _example() -> UgiProductMorphology:
    with gzip.open(
        REPO / "results/phase1/ugi_l1_semantics/ugi_l1_semantic_products.csv.gz",
        "rt",
        newline="",
    ) as handle:
        product = next(csv.DictReader(handle))
    atoms = []
    with gzip.open(
        REPO / "results/phase1/ugi_l1_semantics/ugi_l1_semantic_atoms.csv.gz",
        "rt",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            if row["product_id"] == product["product_id"]:
                atoms.append(row)
            elif atoms:
                break
    vocabulary = load_atom_vocabulary(REPO / "results/phase1/product_v3_atom_vocabulary.json")
    support = tensorize_ugi_l1_support_record(
        product,
        atoms,
        {state: index for index, state in enumerate(vocabulary)},
        preserve_aromaticity=True,
    )
    return split_ugi_support_morphology(
        support,
        json.loads(product["component_smiles_json"]),
        product_id=product["product_id"],
    )


def test_support_projection_is_three_exact_core_attached_component_trees() -> None:
    record = _example()

    assert tuple(component.role for component in record.components) == ROLE_NAMES
    assert record.program.node_count == sum(component.node_count for component in record.components)
    for component in record.components:
        assert attached_tree_matches_program(
            component.offspring,
            node_count=component.node_count,
            junction_budget=component.junction_budget,
        )
    assert record.components[ROLE_NAMES.index("isocyanide_tail")].junction_budget == 0


def test_component_weighted_sources_keep_full_support_and_ignore_row_duplicates() -> None:
    record = _example()
    once = component_weighted_offspring_marginals((record,), maximum_children=4)
    repeated = component_weighted_offspring_marginals((record,) * 20, maximum_children=4)

    assert once.shape == (3, 5)
    assert np.all(once > 0)
    assert np.allclose(once.sum(axis=1), 1.0)
    assert np.allclose(once, repeated)


def test_program_sampler_combines_roles_without_returning_component_ids() -> None:
    record = _example()
    pools = component_weighted_program_pool((record,))
    programs = sample_component_weighted_programs(
        pools,
        count=3,
        rng=np.random.default_rng(4),
    )

    assert len(programs) == 3
    assert all(program == record.program for program in programs)
    assert all(not hasattr(program, "component_key") for program in programs)


def test_exact_attached_decoder_obeys_tree_and_junction_budget() -> None:
    logits = torch.zeros((8, 4), dtype=torch.float32)
    generator = torch.Generator().manual_seed(17)

    for budget in (0, 1, 2):
        offspring = sample_attached_offspring_with_exact_budget(
            logits,
            junction_budget=budget,
            generator=generator,
        )
        assert attached_tree_matches_program(
            offspring,
            node_count=8,
            junction_budget=budget,
        )


def test_size_only_attached_decoder_generates_branching_without_a_supplied_budget() -> None:
    linear_logits = torch.full((4, 4), -30.0, dtype=torch.float32)
    linear_logits[torch.arange(4), torch.tensor([1, 1, 1, 0])] = 30.0
    branched_logits = torch.full((4, 4), -30.0, dtype=torch.float32)
    branched_logits[torch.arange(4), torch.tensor([2, 1, 0, 0])] = 30.0

    linear = sample_attached_offspring_without_budget(
        linear_logits,
        generator=torch.Generator().manual_seed(31),
    )
    branched = sample_attached_offspring_without_budget(
        branched_logits,
        generator=torch.Generator().manual_seed(31),
    )

    assert linear.tolist() == [1, 1, 1, 0]
    assert branched.tolist() == [2, 1, 0, 0]
    assert attached_tree_matches_program(linear, node_count=4, junction_budget=0)
    assert attached_tree_matches_program(branched, node_count=4, junction_budget=1)


def test_graph_branch_run_is_not_serialization_adjacency() -> None:
    separated_in_preorder = np.asarray([2, 1, 0, 2, 0, 0], dtype=np.int64)

    assert attached_tree_matches_program(
        separated_in_preorder,
        node_count=6,
        junction_budget=2,
    )
    assert not any(
        separated_in_preorder[index] >= 2 and separated_in_preorder[index + 1] >= 2
        for index in range(len(separated_in_preorder) - 1)
    )
    assert maximum_adjacent_branch_graph_run(separated_in_preorder) == 2


def test_exact_attached_decoder_conditions_on_decoded_graph_branch_run() -> None:
    logits = torch.full((8, 4), -30.0, dtype=torch.float32)
    logits[torch.arange(8), torch.tensor([2, 2, 1, 1, 0, 0, 0, 0])] = 30.0

    offspring = sample_attached_offspring_with_exact_budget(
        logits,
        junction_budget=2,
        generator=torch.Generator().manual_seed(23),
        maximum_adjacent_branch_run=1,
    )

    assert attached_tree_matches_program(
        offspring,
        node_count=8,
        junction_budget=2,
    )
    assert maximum_adjacent_branch_graph_run(offspring) <= 1


def test_secondary_amine_exterior_is_an_exact_two_root_virtual_port_forest() -> None:
    target = np.asarray([1, 0, 1, 0], dtype=np.int64)
    parents = preorder_attached_forest_to_parents(target, attachment_count=2)

    assert parents.tolist() == [-1, 0, -1, 2]
    assert attached_tree_matches_program(
        target,
        node_count=4,
        junction_budget=0,
        attachment_count=2,
    )
    sampled = sample_attached_offspring_with_exact_budget(
        torch.zeros((4, 3), dtype=torch.float32),
        junction_budget=0,
        attachment_count=2,
        generator=torch.Generator().manual_seed(29),
    )
    assert attached_tree_matches_program(
        sampled,
        node_count=4,
        junction_budget=0,
        attachment_count=2,
    )


def test_cycle_aware_decoder_excludes_nonclosable_branch_word() -> None:
    logits = torch.full((5, 3), -20.0, dtype=torch.float32)
    # This otherwise valid word cannot make a five- or six-member ring.
    logits[torch.arange(5), torch.tensor([1, 1, 2, 0, 0])] = 20.0
    offspring = sample_attached_offspring_with_exact_budget_and_cycle_rank(
        logits,
        junction_budget=1,
        cycle_rank=1,
        generator=torch.Generator().manual_seed(11),
    )

    assert attached_tree_matches_program(offspring, node_count=5, junction_budget=1)
    assert offspring.tolist() != [1, 1, 2, 0, 0]


def test_cycle_decoder_uses_neutral_constrained_fallback_without_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nonclosable = np.asarray([1, 1, 2, 0, 0], dtype=np.int64)
    closable = np.asarray([2, 1, 1, 0, 0], dtype=np.int64)

    def fake_sample(logits: object, **_: object) -> np.ndarray:
        return closable.copy() if int(torch.count_nonzero(logits)) == 0 else nonclosable.copy()

    monkeypatch.setattr(
        morphology_program,
        "sample_attached_offspring_with_exact_budget",
        fake_sample,
    )
    offspring = sample_attached_offspring_with_exact_budget_and_cycle_rank(
        torch.ones((5, 3), dtype=torch.float32),
        junction_budget=1,
        cycle_rank=1,
        generator=torch.Generator().manual_seed(13),
        maximum_rejection_attempts=1,
        maximum_enumerated_nodes=0,
    )

    assert offspring.tolist() == closable.tolist()
