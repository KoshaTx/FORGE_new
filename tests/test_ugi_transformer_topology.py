from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from forge.core.io import read_json_object
from forge.corpus.synthesis_program_training import load_synthesis_program_training_cache
from forge.model.local_chemistry_support import (
    LocalChemistrySupport,
    tree_path_indices,
)
from forge.model.reaction_core_saturation import ReactionCoreSaturationPolicy
from forge.model.reaction_program_flow import (
    collate_synthesis_program_records,
    derive_role_morphology_states,
)
from forge.model.reaction_program_transformer import synthesis_program_offspring_targets
from forge.model.synthesis_program_sampling import (
    COUPLED_UGI_TOPOLOGY_TERMINAL_DECODE_POLICY,
    LEARNED_TOPOLOGY_THEN_CHEMISTRY_FACTORIZATION,
    UGI_ESTER_TOPOLOGY_ROLE_LOCAL_STOCHASTIC_TERMINAL_DECODE_POLICY,
    UGI_ESTER_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY,
    UGI_TOPOLOGY_ROLE_LOCAL_STOCHASTIC_TERMINAL_DECODE_POLICY,
    UGI_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY,
    _terminal_role_morphology,
    sample_synthesis_program_products,
)
from forge.model.ugi_amine_semantic_program import UgiAmineSemanticTarget
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy
from forge.model.ugi_morphology_program import (
    attached_tree_matches_program,
    preorder_attached_forest_to_parents,
)
from forge.model.ugi_role_chemistry_prior import UgiRoleChemistryPrior
from forge.model.ugi_transformer_topology import (
    UgiLocalSemanticTopology,
    UgiTransformerTopologyPolicy,
    _amine_topology_support_context,
    _full_directional_ester_side_carbon_counts_for_tree,
    _full_ester_side_carbon_counts_for_tree,
    _has_ester_capable_tree,
    _sample_constructive_ester_offspring,
    _select_role_closures,
    amine_semantic_topology_supports_target,
    decode_ugi_exact_topology,
    enumerate_amine_semantic_topologies,
)

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "results/phase1/shared_synthesis_program_training_integration_v1/cache.json"


def _ugi_record():
    cache = load_synthesis_program_training_cache(CACHE)
    record = next(item for item in cache.records if item.program_id == "ugi_3cr_agile")
    return replace(record, role_morphology_states=derive_role_morphology_states(record))


def _policy() -> UgiTransformerTopologyPolicy:
    return UgiTransformerTopologyPolicy.from_mapping(
        {
            "allowed_ring_sizes": [5, 6, 7],
            "maximum_heavy_degree": 4,
            "maximum_adjacent_branch_run_by_role": {
                "amine_head": 2,
                "oxoester_aldehyde_body_tail": 1,
                "isocyanide_tail": 1,
            },
        }
    )


def _identity_only_chemistry_prior() -> UgiRoleChemistryPrior:
    return UgiRoleChemistryPrior(
        reaction_id="ugi_3cr_agile",
        roles=("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail"),
        maximum_depth_bucket=6,
        smoothing=1.0,
        minimum_context_count=1,
        assignments_path=Path("assignments.csv.gz"),
        assignments_sha256="0" * 64,
        semantic_atoms_path=Path("semantic_atoms.csv.gz"),
        semantic_atoms_sha256="1" * 64,
        semantic_bonds_path=Path("semantic_bonds.csv.gz"),
        semantic_bonds_sha256="2" * 64,
        representative_components_by_role=(("amine_head", 1),),
        representative_pairs_sha256="3" * 64,
        atom_counts=(("amine_head", 1, 2, 0, "C", 0, 0, 0, 1),),
        bond_counts=(("amine_head", 1, 0, "C", "C", 0, 1),),
    )


def test_amine_semantic_topology_enumeration_conditions_on_complete_head_diameter() -> None:
    topologies = enumerate_amine_semantic_topologies(
        node_count=4,
        junction_budget=2,
        cycle_rank=0,
        attachment_count=1,
        target=UgiAmineSemanticTarget(3, 3, 1, 0),
        maximum_children=3,
        policy=_policy(),
    )

    assert [topology.offspring.tolist() for topology in topologies] == [[3, 0, 0, 0]]
    assert topologies[0].closures == ()


def test_topology_support_context_is_rooted_at_the_reaction_core() -> None:
    topology = UgiLocalSemanticTopology(
        offspring=np.asarray((1, 1, 0), dtype=np.int64),
        closures=(),
    )

    nodes, depths, neighbors, ring_nodes = _amine_topology_support_context(
        topology,
        attachment_count=1,
    )

    assert nodes == (0, 1, 2)
    assert depths == {0: 1, 1: 2, 2: 3}
    assert neighbors[0] == {1, 3}
    assert ring_nodes == frozenset()


def test_constructive_ester_topology_conditions_on_exact_precursor_side_lengths() -> None:
    offspring = _sample_constructive_ester_offspring(
        torch.zeros((17, 4)),
        generator=torch.Generator().manual_seed(101),
        minimum_side_carbons=6,
        minimum_long_side_carbons=10,
        exact_full_side_carbons=(6, 10),
    )
    parents = preorder_attached_forest_to_parents(offspring, attachment_count=1)

    assert (6, 10) in _full_ester_side_carbon_counts_for_tree(parents)
    assert _has_ester_capable_tree(
        parents,
        minimum_side_carbons=6,
        minimum_long_side_carbons=10,
    )


def test_constructive_ester_topology_preserves_alkoxy_handle_direction() -> None:
    offspring = _sample_constructive_ester_offspring(
        torch.zeros((17, 4)),
        generator=torch.Generator().manual_seed(101),
        minimum_side_carbons=6,
        minimum_long_side_carbons=10,
        exact_full_side_carbons=(6, 10),
        exact_alkoxy_handle_and_acyl_side_carbons=(6, 10),
    )
    parents = preorder_attached_forest_to_parents(offspring, attachment_count=1)

    assert (6, 10) in _full_directional_ester_side_carbon_counts_for_tree(parents)


def test_amine_semantic_topology_excludes_unqualified_three_site_amine_assignment() -> None:
    topologies = enumerate_amine_semantic_topologies(
        node_count=4,
        junction_budget=2,
        cycle_rank=0,
        attachment_count=1,
        target=UgiAmineSemanticTarget(3, 2, 3, 0),
        maximum_children=3,
        policy=_policy(),
    )

    assert topologies == ()


def test_amine_semantic_topology_excludes_nitrogen_on_degree_four_branch() -> None:
    topology = UgiLocalSemanticTopology(
        offspring=np.asarray([3, 0, 0, 0], dtype=np.int64),
        closures=(),
    )

    assert not amine_semantic_topology_supports_target(
        topology,
        target=UgiAmineSemanticTarget(3, 1, 2, 0),
        attachment_count=1,
    )


def test_amine_semantic_topology_enforces_donor_and_branch_semantics() -> None:
    topology = UgiLocalSemanticTopology(
        offspring=np.asarray([3, 0, 0, 0], dtype=np.int64),
        closures=(),
    )

    assert amine_semantic_topology_supports_target(
        topology,
        target=UgiAmineSemanticTarget(3, 3, 1, 0, 1, 1),
        attachment_count=1,
    )
    assert not amine_semantic_topology_supports_target(
        topology,
        target=UgiAmineSemanticTarget(3, 3, 1, 0, 0, 1),
        attachment_count=1,
    )
    assert not amine_semantic_topology_supports_target(
        topology,
        target=UgiAmineSemanticTarget(3, 3, 1, 0, 1, 2),
        attachment_count=1,
    )


def test_amine_semantic_topology_conditions_on_complete_cycle_chemistry_support() -> None:
    topology = UgiLocalSemanticTopology(
        offspring=np.asarray([2, 0, 1, 1, 1, 0], dtype=np.int64),
        closures=((1, 5),),
    )
    target = UgiAmineSemanticTarget(5, 5, 2, 0)
    support = LocalChemistrySupport.from_mapping(
        read_json_object(
            REPO / "results/phase1/local_morphology_support_v2/policy.json",
            error=ValueError,
            label="production local chemistry support",
        )
    )

    assert amine_semantic_topology_supports_target(
        topology,
        target=target,
        attachment_count=1,
        local_chemistry_support=support,
    )
    without_amine_cycles = replace(
        support,
        role_cycles={**support.role_cycles, "ugi_3cr_agile": frozenset()},
    )
    assert not amine_semantic_topology_supports_target(
        topology,
        target=target,
        attachment_count=1,
        local_chemistry_support=without_amine_cycles,
    )


def _target_predictions(record):
    clean = collate_synthesis_program_records((record,), maximum_closures=3)
    offspring, _ = synthesis_program_offspring_targets(clean, maximum_children=3)
    offspring_logits = torch.full((1, record.node_count, 4), -30.0)
    offspring_logits.scatter_(2, offspring[:, : record.node_count, None], 30.0)
    left = torch.full((1, 3, record.node_count), -30.0)
    right = torch.full_like(left, -30.0)
    for slot, (left_node, right_node) in enumerate(
        zip(record.graph.closure_left, record.graph.closure_right, strict=True)
    ):
        left[0, slot, int(left_node)] = 30.0
        right[0, slot, int(right_node)] = 30.0
    return {"offspring": offspring_logits, "closure_left": left, "closure_right": right}


def test_exact_transformer_topology_round_trips_a_supported_ugi_program() -> None:
    record = _ugi_record()
    predictions = _target_predictions(record)
    first = decode_ugi_exact_topology(
        predictions,
        index=0,
        record=record,
        policy=_policy(),
        generator=torch.Generator().manual_seed(101),
    )
    second = decode_ugi_exact_topology(
        predictions,
        index=0,
        record=record,
        policy=_policy(),
        generator=torch.Generator().manual_seed(101),
    )

    assert np.array_equal(first.parents, record.graph.parents)
    assert np.array_equal(first.closure_left, record.graph.closure_left)
    assert np.array_equal(first.closure_right, record.graph.closure_right)
    assert np.array_equal(first.parents, second.parents)
    assert np.array_equal(first.closure_left, second.closure_left)
    assert np.array_equal(first.closure_right, second.closure_right)

    observed = _terminal_role_morphology(
        record,
        parents=first.parents,
        closure_left=first.closure_left,
        closure_right=first.closure_right,
    )
    expected = {
        role_state: tuple(int(value) - 1 for value in values[0])
        for role_state in sorted(set(int(value) for value in record.role_states if value > 0))
        if (
            values := np.unique(
                record.role_morphology_states[record.role_states == role_state], axis=0
            )
        ).shape
        == (1, 4)
    }
    assert observed == expected


def test_exact_transformer_topology_rejects_an_unpinned_role_policy() -> None:
    with pytest.raises(ValueError, match="every precursor role"):
        UgiTransformerTopologyPolicy.from_mapping(
            {
                "allowed_ring_sizes": [5, 6],
                "maximum_heavy_degree": 4,
                "maximum_adjacent_branch_run_by_role": {"amine_head": 1},
            }
        )


class _TargetTopologyModel:
    maximum_closures = 3

    def __init__(self, record, node_classes: int) -> None:
        self.record = record
        self.node_classes = node_classes
        self.calls = 0

    def eval(self):
        return self

    def __call__(self, **inputs):
        self.calls += 1
        batch, nodes = inputs["nodes"].shape
        clean = collate_synthesis_program_records((self.record,), maximum_closures=3)

        def categorical(target, classes):
            logits = torch.full((batch, *target.shape[1:], classes), -30.0)
            expanded = target.expand(batch, *target.shape[1:])
            return logits.scatter(-1, expanded.unsqueeze(-1), 30.0)

        parent_logits = torch.full((batch, nodes, nodes), -30.0)
        parent_logits.scatter_(
            2,
            clean["parents"].expand(batch, -1).unsqueeze(-1),
            30.0,
        )
        closure_left = torch.full((batch, 3, nodes), -30.0)
        closure_right = torch.full_like(closure_left, -30.0)
        closure_left.scatter_(
            2,
            clean["closure_left"].expand(batch, -1).unsqueeze(-1),
            30.0,
        )
        closure_right.scatter_(
            2,
            clean["closure_right"].expand(batch, -1).unsqueeze(-1),
            30.0,
        )
        offspring, _ = synthesis_program_offspring_targets(clean, maximum_children=3)
        return {
            "nodes": categorical(clean["nodes"], self.node_classes),
            "parents": parent_logits,
            "parent_bonds": categorical(clean["parent_bonds"], 4),
            "closure_left": closure_left,
            "closure_right": closure_right,
            "closure_bonds": categorical(clean["closure_bonds"], 4),
            "offspring": categorical(offspring, 4),
        }


def test_coupled_sampler_conditions_chemistry_on_exact_topology_without_repair() -> None:
    cache = load_synthesis_program_training_cache(CACHE)
    record = _ugi_record()
    model = _TargetTopologyModel(record, len(cache.atom_vocabulary))
    node_marginal = np.full(len(cache.atom_vocabulary), 1 / len(cache.atom_vocabulary))
    bond_marginal = np.full(4, 0.25)

    rows, receipt = sample_synthesis_program_products(
        model,
        (record,),
        cache.atom_vocabulary,
        node_marginal,
        bond_marginal,
        samples_per_program=1,
        sample_steps=2,
        batch_size=1,
        seed=211,
        device="cpu",
        terminal_decode_policy=COUPLED_UGI_TOPOLOGY_TERMINAL_DECODE_POLICY,
        ugi_topology_policy=_policy(),
    )

    assert model.calls == 4  # two flow steps, terminal prediction, topology-conditioned chemistry
    assert rows[0]["valid"] is True
    assert rows[0]["exact_target_graph"] is True
    assert receipt["strict_constraint_abstentions"] == 0
    assert receipt["topology_coupling_second_pass_applied"] is True
    assert receipt["topology_selection"] == "exact_program_conditional_sample_then_argmax_chemistry"
    assert receipt["topology_seed"] == 212
    assert receipt["repairs"] == {}


def test_coupled_sampler_repeatedly_refreshes_chemistry_on_one_fixed_topology() -> None:
    cache = load_synthesis_program_training_cache(CACHE)
    record = _ugi_record()
    node_marginal = np.full(len(cache.atom_vocabulary), 1 / len(cache.atom_vocabulary))
    bond_marginal = np.full(4, 0.25)

    class RecordingModel(_TargetTopologyModel):
        def __init__(self) -> None:
            super().__init__(record, len(cache.atom_vocabulary))
            self.parents_seen: list[torch.Tensor] = []

        def __call__(self, **inputs):
            self.parents_seen.append(inputs["parents"].detach().cpu().clone())
            return super().__call__(**inputs)

    def draw():
        model = RecordingModel()
        rows, receipt = sample_synthesis_program_products(
            model,
            (record,),
            cache.atom_vocabulary,
            node_marginal,
            bond_marginal,
            samples_per_program=1,
            sample_steps=2,
            batch_size=1,
            seed=251,
            device="cpu",
            terminal_decode_policy=COUPLED_UGI_TOPOLOGY_TERMINAL_DECODE_POLICY,
            ugi_topology_policy=_policy(),
            topology_conditioned_chemistry_steps=3,
            topology_conditioned_chemistry_seed=253,
        )
        return model, rows, receipt

    first_model, first_rows, first_receipt = draw()
    _, second_rows, second_receipt = draw()
    assert first_rows == second_rows
    assert first_receipt == second_receipt
    assert first_model.calls == 7  # joint flow, terminal topology, three chemistry steps, terminal
    expected_parents = torch.as_tensor(record.graph.parents)
    assert all(
        torch.equal(observed[0, : record.node_count], expected_parents)
        for observed in first_model.parents_seen[-4:]
    )
    assert first_rows[0]["valid"] is True, first_receipt["strict_constraint_abstention_reasons"]
    assert first_rows[0]["topology_conditioned_chemistry_flow_applied"] is True
    assert first_receipt["topology_conditioned_chemistry_steps"] == 3
    assert first_receipt["topology_conditioned_chemistry_seed"] == 253
    assert first_receipt["topology_conditioned_chemistry_neural_evaluations_per_batch"] == 4
    assert first_receipt["topology_conditioned_chemistry_source_reset"] is True
    assert first_receipt["topology_selection"] == (
        "exact_program_conditional_sample_then_3_step_conditioned_chemistry_flow_"
        "then_argmax_chemistry"
    )
    assert first_receipt["repairs"] == {}


def test_topology_conditioned_chemistry_flow_requires_an_explicit_valid_seed() -> None:
    cache = load_synthesis_program_training_cache(CACHE)
    record = _ugi_record()
    model = _TargetTopologyModel(record, len(cache.atom_vocabulary))
    node_marginal = np.full(len(cache.atom_vocabulary), 1 / len(cache.atom_vocabulary))
    bond_marginal = np.full(4, 0.25)
    common = {
        "model": model,
        "records": (record,),
        "atom_vocabulary": cache.atom_vocabulary,
        "node_marginal": node_marginal,
        "bond_marginal": bond_marginal,
        "samples_per_program": 1,
        "sample_steps": 2,
        "batch_size": 1,
        "seed": 271,
        "device": "cpu",
        "terminal_decode_policy": COUPLED_UGI_TOPOLOGY_TERMINAL_DECODE_POLICY,
        "ugi_topology_policy": _policy(),
    }
    with pytest.raises(ValueError, match="requires one explicit seed"):
        sample_synthesis_program_products(
            **common,
            topology_conditioned_chemistry_steps=3,
        )
    with pytest.raises(ValueError, match="disabled or use at least two steps"):
        sample_synthesis_program_products(
            **common,
            topology_conditioned_chemistry_steps=1,
            topology_conditioned_chemistry_seed=273,
        )
    with pytest.raises(ValueError, match="learned topology-then-chemistry"):
        sample_synthesis_program_products(
            **common,
            sampling_factorization=LEARNED_TOPOLOGY_THEN_CHEMISTRY_FACTORIZATION,
        )


def test_topology_first_sampler_composes_role_local_chemistry_and_core_saturation() -> None:
    cache = load_synthesis_program_training_cache(CACHE)
    record = _ugi_record()
    model = _TargetTopologyModel(record, len(cache.atom_vocabulary))
    model.vocabulary = cache.vocabulary
    node_marginal = np.full(len(cache.atom_vocabulary), 1 / len(cache.atom_vocabulary))
    bond_marginal = np.full(4, 0.25)
    local_support = LocalChemistrySupport.from_mapping(
        read_json_object(
            REPO / "results/phase1/local_morphology_support_v2/policy.json",
            error=ValueError,
            label="production local chemistry support",
        )
    )
    core_policy = ReactionCoreSaturationPolicy.from_qualified_registry(
        REPO / "data/vendor/qualified_reactions_v1.json",
        reaction_id="ugi_3cr_agile",
    )

    rows, receipt = sample_synthesis_program_products(
        model,
        (record,),
        cache.atom_vocabulary,
        node_marginal,
        bond_marginal,
        samples_per_program=1,
        sample_steps=2,
        batch_size=1,
        seed=311,
        device="cpu",
        terminal_decode_policy=UGI_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY,
        local_chemistry_support=local_support,
        ugi_topology_policy=_policy(),
        reaction_core_saturation_policy=core_policy,
    )

    assert model.calls == 4
    assert rows[0]["valid"] is True
    assert rows[0]["exact_target_graph"] is True
    assert rows[0]["program_topology_policy_applied"] is True
    assert rows[0]["program_role_cycle_policy_applied"] is True
    assert rows[0]["component_confined_policy_applied"] is True
    assert rows[0]["topology_coupling_second_pass_applied"] is True
    assert rows[0]["reaction_core_saturation_policy_applied"] is True
    assert receipt["strict_constraint_abstentions"] == 0
    assert receipt["repairs"] == {}


def test_topology_first_stochastic_chemistry_is_seeded_and_never_repairs() -> None:
    cache = load_synthesis_program_training_cache(CACHE)
    record = _ugi_record()
    node_marginal = np.full(len(cache.atom_vocabulary), 1 / len(cache.atom_vocabulary))
    bond_marginal = np.full(4, 0.25)
    local_support = LocalChemistrySupport.from_mapping(
        read_json_object(
            REPO / "results/phase1/local_morphology_support_v2/policy.json",
            error=ValueError,
            label="production local chemistry support",
        )
    )
    core_policy = ReactionCoreSaturationPolicy.from_qualified_registry(
        REPO / "data/vendor/qualified_reactions_v1.json",
        reaction_id="ugi_3cr_agile",
    )

    def draw():
        model = _TargetTopologyModel(record, len(cache.atom_vocabulary))
        model.vocabulary = cache.vocabulary
        return sample_synthesis_program_products(
            model,
            (record,),
            cache.atom_vocabulary,
            node_marginal,
            bond_marginal,
            samples_per_program=1,
            sample_steps=2,
            batch_size=1,
            seed=411,
            device="cpu",
            terminal_decode_policy=(UGI_TOPOLOGY_ROLE_LOCAL_STOCHASTIC_TERMINAL_DECODE_POLICY),
            terminal_seed=413,
            terminal_temperature=1.0,
            local_chemistry_support=local_support,
            ugi_topology_policy=_policy(),
            reaction_core_saturation_policy=core_policy,
        )

    first_rows, first_receipt = draw()
    second_rows, second_receipt = draw()
    assert first_rows == second_rows
    assert first_rows[0]["valid"] is True
    assert first_receipt == second_receipt
    assert first_receipt["terminal_seed"] == 413
    assert first_receipt["terminal_chemistry_readout"] == (
        "support_constrained_categorical_atom_and_bond_draw"
    )
    assert first_receipt["repairs"] == {}


def test_ugi_ester_chemotype_is_a_registry_bound_decode_condition() -> None:
    cache = load_synthesis_program_training_cache(CACHE)
    record = _ugi_record()
    model = _TargetTopologyModel(record, len(cache.atom_vocabulary))
    model.vocabulary = cache.vocabulary
    node_marginal = np.full(len(cache.atom_vocabulary), 1 / len(cache.atom_vocabulary))
    bond_marginal = np.full(4, 0.25)
    local_support = LocalChemistrySupport.from_mapping(
        read_json_object(
            REPO / "results/phase1/local_morphology_support_v2/policy.json",
            error=ValueError,
            label="production local chemistry support",
        )
    )
    registry = REPO / "data/vendor/qualified_reactions_v1.json"
    core_policy = ReactionCoreSaturationPolicy.from_qualified_registry(registry)
    ester_policy = UgiEsterChemotypePolicy.from_qualified_registry(
        registry,
        training_assignments_path=(
            REPO / "results/phase1/ugi_balanced_chemistry_corpus_v2/assignments.csv.gz"
        ),
    )
    assert dict(ester_policy.minimum_role_exterior_atoms) == {
        "amine_head": 6,
        "oxoester_aldehyde_body_tail": 17,
        "isocyanide_tail": 12,
    }
    assert ester_policy.minimum_constructive_aldehyde_exterior_atoms == 17
    assert ester_policy.minimum_amine_exterior_nitrogens == 1
    assert ester_policy.maximum_amine_exterior_nitrogens == 1
    assert dict(ester_policy.amine_cycle_sizes_by_exterior_count) == {
        5: (5,),
        6: (6,),
        7: (5,),
        8: (5, 6),
        9: (6,),
    }
    assert len(ester_policy.measured_training_product_ids) == 480
    assert ester_policy.to_mapping()["morphology_program_sampling"] == (
        "occurrence_weighted_joint_count_only_programs_from_measured_training_products"
    )
    # This legacy one-record fixture deliberately contains a 27-atom aldehyde exterior, beyond
    # the production interquartile chemotype support.  Widen only the fixture's upper bound so the
    # behavior test can exercise terminal chemistry; production uses the derived bound above.
    ester_policy = replace(
        ester_policy,
        minimum_role_exterior_atoms=tuple(
            (role, 1) for role, _ in ester_policy.minimum_role_exterior_atoms
        ),
        maximum_role_exterior_atoms=tuple(
            (role, max(limit, 30)) for role, limit in ester_policy.maximum_role_exterior_atoms
        ),
        amine_cycle_sizes_by_exterior_count=tuple(
            (count, (5, 6) if count == 6 else sizes)
            for count, sizes in ester_policy.amine_cycle_sizes_by_exterior_count
        ),
    )

    rows, receipt = sample_synthesis_program_products(
        model,
        (record,),
        cache.atom_vocabulary,
        node_marginal,
        bond_marginal,
        samples_per_program=1,
        sample_steps=2,
        batch_size=1,
        seed=511,
        device="cpu",
        terminal_decode_policy=UGI_ESTER_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY,
        local_chemistry_support=local_support,
        ugi_topology_policy=_policy(),
        reaction_core_saturation_policy=core_policy,
        ugi_ester_chemotype_policy=ester_policy,
    )

    assert rows[0]["valid"] is True
    assert rows[0]["ugi_ester_chemotype_policy_applied"] is True
    assert receipt["ugi_ester_chemotype_policy_applied"] is True
    assert receipt["topology_selection"] == (
        "measured_joint_program_then_constructive_ester_and_role_local_cycle_topology_"
        "then_argmax_chemistry"
    )
    assert receipt["repairs"] == {}

    lambda_zero_rows, lambda_zero_receipt = sample_synthesis_program_products(
        model,
        (record,),
        cache.atom_vocabulary,
        node_marginal,
        bond_marginal,
        samples_per_program=1,
        sample_steps=2,
        batch_size=1,
        seed=511,
        device="cpu",
        terminal_decode_policy=UGI_ESTER_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY,
        local_chemistry_support=local_support,
        ugi_topology_policy=_policy(),
        reaction_core_saturation_policy=core_policy,
        ugi_ester_chemotype_policy=ester_policy,
        ugi_role_chemistry_prior=_identity_only_chemistry_prior(),
        ugi_role_chemistry_prior_strength=0.0,
    )
    assert lambda_zero_rows == rows
    assert (
        lambda_zero_receipt["terminal_chemistry_readout"] == receipt["terminal_chemistry_readout"]
    )
    assert lambda_zero_receipt["ugi_role_chemistry_prior_applied"] is False

    def stochastic_draw():
        return sample_synthesis_program_products(
            model,
            (record,),
            cache.atom_vocabulary,
            node_marginal,
            bond_marginal,
            samples_per_program=1,
            sample_steps=2,
            batch_size=1,
            seed=511,
            device="cpu",
            terminal_decode_policy=(
                UGI_ESTER_TOPOLOGY_ROLE_LOCAL_STOCHASTIC_TERMINAL_DECODE_POLICY
            ),
            terminal_seed=517,
            terminal_temperature=0.7,
            local_chemistry_support=local_support,
            ugi_topology_policy=_policy(),
            reaction_core_saturation_policy=core_policy,
            ugi_ester_chemotype_policy=ester_policy,
        )

    stochastic_rows, stochastic_receipt = stochastic_draw()
    repeated_rows, repeated_receipt = stochastic_draw()
    assert stochastic_rows == repeated_rows
    assert stochastic_receipt == repeated_receipt
    assert stochastic_rows[0]["valid"] is True
    assert stochastic_rows[0]["ugi_ester_chemotype_policy_applied"] is True
    assert stochastic_receipt["terminal_chemistry_readout"] == (
        "support_constrained_categorical_atom_and_bond_draw"
    )
    assert stochastic_receipt["topology_selection"] == (
        "measured_joint_program_then_constructive_ester_and_role_local_cycle_topology_"
        "then_sequential_masked_categorical_chemistry"
    )
    assert stochastic_receipt["terminal_neural_refresh_after_each_choice"] is False
    assert stochastic_receipt["repairs"] == {}


def test_learned_topology_then_chemistry_uses_parent_heads_and_freezes_topology() -> None:
    cache = load_synthesis_program_training_cache(CACHE)
    record = _ugi_record()
    node_marginal = np.full(len(cache.atom_vocabulary), 1 / len(cache.atom_vocabulary))
    bond_marginal = np.full(4, 0.25)
    local_support = LocalChemistrySupport.from_mapping(
        read_json_object(
            REPO / "results/phase1/local_morphology_support_v2/policy.json",
            error=ValueError,
            label="production local chemistry support",
        )
    )
    registry = REPO / "data/vendor/qualified_reactions_v1.json"
    core_policy = ReactionCoreSaturationPolicy.from_qualified_registry(
        registry,
        reaction_id="ugi_3cr_agile",
    )
    ester_policy = UgiEsterChemotypePolicy.from_qualified_registry(
        registry,
        training_assignments_path=(
            REPO / "results/phase1/ugi_balanced_chemistry_corpus_v2/assignments.csv.gz"
        ),
    )
    ester_policy = replace(
        ester_policy,
        minimum_role_exterior_atoms=tuple(
            (role, 1) for role, _ in ester_policy.minimum_role_exterior_atoms
        ),
        maximum_role_exterior_atoms=tuple(
            (role, max(limit, 30)) for role, limit in ester_policy.maximum_role_exterior_atoms
        ),
        amine_cycle_sizes_by_exterior_count=tuple(
            (count, (5, 6, 7) if count == 6 else sizes)
            for count, sizes in ester_policy.amine_cycle_sizes_by_exterior_count
        ),
    )

    class RecordingModel(_TargetTopologyModel):
        def __init__(self) -> None:
            super().__init__(record, len(cache.atom_vocabulary))
            self.vocabulary = cache.vocabulary
            self.parents_seen: list[torch.Tensor] = []

        def __call__(self, **inputs):
            self.parents_seen.append(inputs["parents"].detach().cpu().clone())
            return super().__call__(**inputs)

    def draw():
        model = RecordingModel()
        rows, receipt = sample_synthesis_program_products(
            model,
            (record,),
            cache.atom_vocabulary,
            node_marginal,
            bond_marginal,
            samples_per_program=1,
            sample_steps=2,
            batch_size=1,
            seed=701,
            device="cpu",
            sampling_factorization=LEARNED_TOPOLOGY_THEN_CHEMISTRY_FACTORIZATION,
            terminal_decode_policy=UGI_ESTER_TOPOLOGY_ROLE_LOCAL_TERMINAL_DECODE_POLICY,
            topology_conditioned_chemistry_steps=3,
            topology_conditioned_chemistry_seed=703,
            local_chemistry_support=local_support,
            ugi_topology_policy=_policy(),
            reaction_core_saturation_policy=core_policy,
            ugi_ester_chemotype_policy=ester_policy,
        )
        return model, rows, receipt

    first_model, first_rows, first_receipt = draw()
    _, second_rows, second_receipt = draw()
    assert first_rows == second_rows
    assert first_receipt == second_receipt
    expected_parents = torch.as_tensor(record.graph.parents)
    assert all(
        torch.equal(observed[0, : record.node_count], expected_parents)
        for observed in first_model.parents_seen[-4:]
    )
    assert first_rows[0]["valid"] is True, first_receipt["strict_constraint_abstention_reasons"]
    assert first_rows[0]["sampled_topology"]["parents"] == record.graph.parents.tolist()
    assert first_rows[0]["sampled_topology"]["source"] == ("transformer_parent_and_closure_heads")
    assert first_receipt["sampling_factorization"] == (
        LEARNED_TOPOLOGY_THEN_CHEMISTRY_FACTORIZATION
    )
    assert first_receipt["learned_topology_flow_applied"] is True
    assert first_receipt["topology_proposal_context"] == (
        "joint_flow_with_provisional_chemistry_then_discard_chemistry"
    )
    assert first_receipt["topology_conditioned_chemistry_source_reset"] is True
    assert first_receipt["topology_seed"] == 701
    assert first_receipt["repairs"] == {}


def test_constructive_ester_topology_has_no_rejection_path() -> None:
    logits = torch.randn((25, 4), generator=torch.Generator().manual_seed(619))
    observed = set()
    for seed in range(32):
        offspring = _sample_constructive_ester_offspring(
            logits,
            generator=torch.Generator().manual_seed(seed),
            minimum_side_carbons=6,
            minimum_long_side_carbons=10,
        )
        observed.add(tuple(int(value) for value in offspring))
        assert attached_tree_matches_program(
            offspring,
            node_count=25,
            junction_budget=1,
            attachment_count=1,
        )
        assert _has_ester_capable_tree(
            preorder_attached_forest_to_parents(offspring, attachment_count=1),
            minimum_side_carbons=6,
            minimum_long_side_carbons=10,
        )
    assert len(observed) > 1


def test_measured_head_ring_size_overrides_the_broad_closure_support() -> None:
    offspring = np.asarray([2, 0, 1, 1, 1, 1, 0], dtype=np.int64)
    parents = preorder_attached_forest_to_parents(offspring, attachment_count=1)
    left_logits = np.zeros((1, len(offspring)), dtype=np.float64)
    right_logits = np.zeros_like(left_logits)

    left, right = _select_role_closures(
        offspring=offspring,
        attachment_count=1,
        cycle_rank=1,
        exterior=np.arange(len(offspring), dtype=np.int64),
        permutation=np.arange(len(offspring), dtype=np.int64),
        closure_left_logits=left_logits,
        closure_right_logits=right_logits,
        first_slot=0,
        policy=_policy(),
        allowed_ring_sizes=(5,),
    )

    assert len(tree_path_indices(parents, left[0], right[0])) == 5
