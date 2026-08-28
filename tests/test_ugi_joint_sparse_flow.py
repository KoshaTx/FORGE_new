from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest

from experiments.phase1.product_l1.sampling.ugi_joint_sparse_sampling import (
    UgiJointSparseTrajectoryState,
    advance_ugi_joint_sparse_state,
    extract_ugi_joint_sparse_particles,
    finalize_ugi_joint_sparse_state,
    initialize_ugi_joint_sparse_state,
    resample_ugi_joint_sparse_ancestry,
)
from experiments.phase1.product_l1.sampling.ugi_joint_sparse_sampling import (
    sample_restartable_terminals as sample_ugi_joint_sparse_terminals,
)
from experiments.phase1.product_l1.training.ugi_joint_sparse_training import (
    UgiJointSparseTrainingError,
    _validated_checkpoint_steps,
)
from forge.corpus.ugi_chemistry_corpus import load_expanded_ugi_chemistry_corpus
from forge.model.ugi_joint_sparse_flow import (
    UgiJointSparseFlow,
    UgiJointSparseFlowError,
    _legacy_sample_ugi_joint_sparse_terminals,
    collate_ugi_joint_sparse_records,
    joint_sparse_source_marginals,
    noise_ugi_joint_sparse_batch,
    project_joint_sparse_record,
    ugi_joint_sparse_loss,
)
from forge.model.ugi_morphology_program import (
    UgiMorphologyProgram,
    attached_tree_matches_program,
)

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]


def _diagnostic_trajectory() -> UgiJointSparseTrajectoryState:
    repeated = UgiMorphologyProgram(
        node_counts=(2, 2, 2),
        junction_budgets=(0, 0, 0),
        cycle_ranks=(0, 0, 0),
        attachment_counts=(1, 1, 1),
    )
    distinct = UgiMorphologyProgram(
        node_counts=(3, 2, 2),
        junction_budgets=(0, 0, 0),
        cycle_ranks=(0, 0, 0),
        attachment_counts=(1, 1, 1),
    )
    batch = torch.arange(21).reshape(3, 7)
    return UgiJointSparseTrajectoryState(
        programs=(repeated, repeated, distinct),
        layout={"node_mask": batch >= 0, "programs": batch.clone()},
        sources={"atoms": torch.ones(3, 2)},
        channels={"nodes": batch.clone(), "offspring": batch.clone() + 100},
        sample_steps=8,
        step=3,
        generator_state=torch.Generator().manual_seed(17).get_state(),
        device="cpu",
    )


def test_particle_extraction_uses_independent_keyed_rng_without_mutation() -> None:
    trajectory = _diagnostic_trajectory()
    extracted = extract_ugi_joint_sparse_particles(
        trajectory,
        (1,),
        generator_seed=91,
    )

    assert extracted.programs == (trajectory.programs[1],)
    assert torch.equal(extracted.channels["nodes"], trajectory.channels["nodes"][1:2])
    assert torch.equal(trajectory.channels["nodes"], torch.arange(21).reshape(3, 7))
    assert torch.equal(
        extracted.generator_state,
        torch.Generator().manual_seed(91).get_state(),
    )
    assert not torch.equal(extracted.generator_state, trajectory.generator_state)


def test_ancestry_resampling_is_within_program_and_keeps_productive_rng() -> None:
    trajectory = _diagnostic_trajectory()
    resampled = resample_ugi_joint_sparse_ancestry(trajectory, (1, 0, 2))

    assert resampled.programs == trajectory.programs
    assert torch.equal(resampled.channels["nodes"][0], trajectory.channels["nodes"][1])
    assert torch.equal(resampled.channels["nodes"][1], trajectory.channels["nodes"][0])
    assert torch.equal(resampled.generator_state, trajectory.generator_state)
    with pytest.raises(
        UgiJointSparseFlowError, match="cannot cross frozen morphology-program groups"
    ):
        resample_ugi_joint_sparse_ancestry(trajectory, (2, 1, 0))


@cache
def _corpus():
    root = REPO / "results/phase1/ugi_expanded_chemistry_exemplars"
    required = (
        root / "assignments.csv.gz",
        root / "semantic_products.csv.gz",
        root / "semantic_atoms.csv.gz",
    )
    if any(not path.is_file() for path in required):
        pytest.skip("legacy 421-record expanded-chemistry fixture is not materialized")
    return load_expanded_ugi_chemistry_corpus(
        *required,
        REPO / "results/phase1/product_v3_atom_vocabulary.json",
    )


@cache
def _joint_records():
    corpus = _corpus()
    return tuple(
        project_joint_sparse_record(record)
        for fold in ("train", "calibration", "heldout")
        for record in corpus.records_by_fold[fold]
    )


def test_every_expanded_record_has_an_exact_joint_sparse_projection() -> None:
    corpus = _corpus()
    projected = _joint_records()
    assert len(projected) == 421
    source_by_id = {
        record.product_id: record
        for fold in ("train", "calibration", "heldout")
        for record in corpus.records_by_fold[fold]
    }
    for record in projected:
        source = source_by_id[record.product_id]
        assert np.array_equal(record.atom_states, source.target.atom_states[record.full_indices])
        assert np.array_equal(
            record.parent_bond_states,
            source.target.parent_bond_states[record.full_indices],
        )
        offset = 0
        for role_index, count in enumerate(record.program.node_counts):
            offspring = record.offspring[offset : offset + count]
            assert attached_tree_matches_program(
                offspring,
                node_count=count,
                junction_budget=record.program.junction_budgets[role_index],
                attachment_count=record.program.attachment_counts[role_index],
            )
            offset += count


def test_joint_sparse_model_shares_topology_and_chemistry_backbone() -> None:
    corpus = _corpus()
    records = _joint_records()[:8]
    maximum_nodes = max(record.node_count for record in records)
    maximum_closures = max(record.closure_left.size for record in records)
    batch = collate_ugi_joint_sparse_records(
        records,
        maximum_nodes=maximum_nodes,
        maximum_children=3,
        maximum_closures=maximum_closures,
        maximum_decorations=5,
    )
    model = UgiJointSparseFlow(
        maximum_children=3,
        atom_classes=len(corpus.atom_vocabulary),
        bond_classes=4,
        maximum_component_atoms=85,
        maximum_total_atoms=85,
        maximum_junction_budget=8,
        maximum_cycle_rank=2,
        maximum_attachment_count=2,
        maximum_decorations=5,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
    )
    predictions = model(
        offspring=batch["offspring"],
        nodes=batch["nodes"],
        parent_bonds=batch["parent_bonds"],
        role_states=batch["role_states"],
        within_role_positions=batch["within_role_positions"],
        programs=batch["programs"],
        node_mask=batch["node_mask"],
        t=torch.full((len(records),), 0.5),
        closure_left=batch["closure_left"],
        closure_right=batch["closure_right"],
    )
    loss, metrics = ugi_joint_sparse_loss(predictions, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["offspring_ce"] > 0
    assert metrics["atom_ce"] > 0
    assert model.sequence.weight_ih_l0.grad is not None

    sources = joint_sparse_source_marginals(
        records,
        atom_classes=len(corpus.atom_vocabulary),
        bond_classes=4,
        maximum_children=3,
        maximum_decorations=5,
        probability_floor=1e-5,
    )
    assert np.allclose(sources["offspring"].sum(axis=1), 1.0)
    assert np.allclose(sources["atoms"].sum(axis=1), 1.0)
    assert sources["decoration"][0] > sources["decoration"][1]

    weighted_sources = joint_sparse_source_marginals(
        records,
        atom_classes=len(corpus.atom_vocabulary),
        bond_classes=4,
        maximum_children=3,
        maximum_decorations=5,
        probability_floor=1e-5,
        record_weights=np.asarray([1.0, *([0.0] * (len(records) - 1))]),
    )
    assert np.allclose(weighted_sources["offspring"].sum(axis=1), 1.0)
    assert not np.allclose(weighted_sources["atoms"], sources["atoms"])


def test_anchor_local_decoration_state_is_bidirectionally_conditioned() -> None:
    corpus = _corpus()
    records = _joint_records()[:4]
    batch = collate_ugi_joint_sparse_records(
        records,
        maximum_nodes=max(record.node_count for record in records),
        maximum_children=3,
        maximum_closures=max(record.closure_left.size for record in records),
        maximum_decorations=5,
    )
    model = UgiJointSparseFlow(
        maximum_children=3,
        atom_classes=len(corpus.atom_vocabulary),
        bond_classes=4,
        maximum_component_atoms=85,
        maximum_total_atoms=85,
        maximum_junction_budget=8,
        maximum_cycle_rank=2,
        maximum_attachment_count=2,
        maximum_decorations=5,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        decoration_state_conditioning="bidirectional_anchor_local",
    )
    arguments = {
        "offspring": batch["offspring"],
        "nodes": batch["nodes"],
        "parent_bonds": batch["parent_bonds"],
        "role_states": batch["role_states"],
        "within_role_positions": batch["within_role_positions"],
        "programs": batch["programs"],
        "node_mask": batch["node_mask"],
        "t": torch.full((len(records),), 0.5),
        "closure_left": batch["closure_left"],
        "closure_right": batch["closure_right"],
        "decoration_anchors": batch["decoration_anchors"],
        "decoration_atoms": batch["decoration_atoms"],
        "decoration_bonds": batch["decoration_bonds"],
    }
    baseline = model(**arguments)
    changed_atoms = batch["decoration_atoms"].clone()
    changed_atoms[:, 0] = (changed_atoms[:, 0] + 1) % len(corpus.atom_vocabulary)
    changed = model(**{**arguments, "decoration_atoms": changed_atoms})

    assert not torch.equal(baseline["nodes"], changed["nodes"])
    assert not torch.equal(baseline["decoration_atoms"], changed["decoration_atoms"])
    loss, _ = ugi_joint_sparse_loss(baseline, batch)
    loss.backward()
    assert model.decoration_atom_embedding.weight.grad is not None

    with pytest.raises(UgiJointSparseFlowError, match="complete noisy slot state"):
        model(**{key: value for key, value in arguments.items() if key != "decoration_atoms"})


def test_decoration_state_noise_matches_sampling_on_absent_slots() -> None:
    records = _joint_records()[:4]
    batch = collate_ugi_joint_sparse_records(
        records,
        maximum_nodes=max(record.node_count for record in records),
        maximum_children=3,
        maximum_closures=max(record.closure_left.size for record in records),
        maximum_decorations=5,
    )
    sources = {
        "offspring": torch.full((3, 4), 0.25),
        "atoms": torch.full((3, len(_corpus().atom_vocabulary)), 1.0),
        "bonds": torch.full((3, 4), 0.25),
        "closure_bonds": torch.full((4,), 0.25),
        "decoration": torch.asarray((1.0, 0.0)),
        "decoration_atoms": torch.nn.functional.one_hot(
            torch.tensor(1), num_classes=len(_corpus().atom_vocabulary)
        ).to(torch.float32),
        "decoration_bonds": torch.asarray((0.0, 1.0, 0.0, 0.0)),
    }
    sources["atoms"] = sources["atoms"] / sources["atoms"].sum(dim=1, keepdim=True)
    noisy = noise_ugi_joint_sparse_batch(
        batch,
        sources,
        torch.zeros(len(records)),
        torch.Generator().manual_seed(13),
    )

    absent = ~batch["decoration_present_mask"]
    assert bool(absent.any())
    assert torch.all(noisy["decoration_atoms"][absent] == 1)
    assert torch.all(noisy["decoration_bonds"][absent] == 1)


def test_size_only_model_does_not_receive_clean_branch_cycle_or_attachment_programs() -> None:
    corpus = _corpus()
    records = _joint_records()[:8]
    batch = collate_ugi_joint_sparse_records(
        records,
        maximum_nodes=max(record.node_count for record in records),
        maximum_children=3,
        maximum_closures=max(record.closure_left.size for record in records),
        maximum_decorations=7,
    )
    model = UgiJointSparseFlow(
        maximum_children=3,
        atom_classes=len(corpus.atom_vocabulary),
        bond_classes=4,
        maximum_component_atoms=85,
        maximum_total_atoms=85,
        maximum_junction_budget=8,
        maximum_cycle_rank=2,
        maximum_attachment_count=2,
        maximum_decorations=7,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        conditioning_mode="size_only",
    )
    altered_programs = batch["programs"].clone()
    altered_programs[:, 3:6] = 0
    altered_programs[:, 6:9] = 2
    altered_programs[:, 9:12] = 2
    arguments = {
        "offspring": batch["offspring"],
        "nodes": batch["nodes"],
        "parent_bonds": batch["parent_bonds"],
        "role_states": batch["role_states"],
        "within_role_positions": batch["within_role_positions"],
        "node_mask": batch["node_mask"],
        "t": torch.full((len(records),), 0.5),
        "closure_left": batch["closure_left"],
        "closure_right": batch["closure_right"],
    }
    model.eval()
    baseline = model(programs=batch["programs"], **arguments)
    altered = model(programs=altered_programs, **arguments)

    for key in (
        "offspring",
        "nodes",
        "parent_bonds",
        "cycle_ranks",
        "attachment_counts",
    ):
        assert torch.equal(baseline[key], altered[key])
    loss, metrics = ugi_joint_sparse_loss(baseline, batch)
    assert torch.isfinite(loss)
    assert metrics["cycle_rank_ce"] > 0
    assert metrics["attachment_count_ce"] > 0


def test_size_only_sampler_generates_global_topology_fields_from_sizes() -> None:
    corpus = _corpus()
    records = _joint_records()[:8]
    model = UgiJointSparseFlow(
        maximum_children=3,
        atom_classes=len(corpus.atom_vocabulary),
        bond_classes=4,
        maximum_component_atoms=16,
        maximum_total_atoms=48,
        maximum_junction_budget=8,
        maximum_cycle_rank=2,
        maximum_attachment_count=2,
        maximum_decorations=7,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
        conditioning_mode="size_only",
    )
    sources = joint_sparse_source_marginals(
        records,
        atom_classes=len(corpus.atom_vocabulary),
        bond_classes=4,
        maximum_children=3,
        maximum_decorations=7,
        probability_floor=1e-5,
    )
    supplied = UgiMorphologyProgram(
        node_counts=(4, 5, 6),
        junction_budgets=(7, 7, 7),
        cycle_ranks=(2, 2, 2),
        attachment_counts=(2, 2, 2),
    )
    samples, metadata = sample_ugi_joint_sparse_terminals(
        model,
        (supplied,),
        sources,
        sample_steps=2,
        batch_size=1,
        seed=19,
        device="cpu",
        maximum_adjacent_branch_runs=(2, 1, 1),
    )

    assert metadata["program_fields_supplied"] == ["node_counts"]
    assert metadata["program_fields_generated"] == [
        "junction_budgets",
        "cycle_ranks",
        "attachment_counts",
    ]
    assert metadata["maximum_adjacent_branch_runs"] == [2, 1, 1]
    assert samples[0].program.node_counts == supplied.node_counts
    for role_index, offspring in enumerate(samples[0].offspring):
        assert attached_tree_matches_program(
            offspring,
            node_count=supplied.node_counts[role_index],
            junction_budget=samples[0].program.junction_budgets[role_index],
            attachment_count=samples[0].program.attachment_counts[role_index],
        )
        maximum_run = current_run = 0
        for child_count in offspring:
            current_run = current_run + 1 if child_count >= 2 else 0
            maximum_run = max(maximum_run, current_run)
        assert maximum_run <= (2, 1, 1)[role_index]


def test_restartable_sampler_is_bitwise_equal_to_frozen_monolithic_schedule() -> None:
    corpus = _corpus()
    records = _joint_records()[:8]
    torch.manual_seed(20260802)
    model = UgiJointSparseFlow(
        maximum_children=3,
        atom_classes=len(corpus.atom_vocabulary),
        bond_classes=4,
        maximum_component_atoms=85,
        maximum_total_atoms=85,
        maximum_junction_budget=8,
        maximum_cycle_rank=2,
        maximum_attachment_count=2,
        maximum_decorations=5,
        hidden_dim=32,
        layers=1,
        dropout=0.0,
    )
    sources = joint_sparse_source_marginals(
        records,
        atom_classes=len(corpus.atom_vocabulary),
        bond_classes=4,
        maximum_children=3,
        maximum_decorations=5,
        probability_floor=1e-5,
    )
    programs = (records[0].program, records[1].program)
    expected = None
    for batch_size in (1, 2):
        arguments = {
            "sample_steps": 3,
            "batch_size": batch_size,
            "seed": 41,
            "device": "cpu",
        }
        local_expected, expected_metadata = _legacy_sample_ugi_joint_sparse_terminals(
            model, programs, sources, **arguments
        )
        observed, observed_metadata = sample_ugi_joint_sparse_terminals(
            model, programs, sources, **arguments
        )

        assert observed_metadata == expected_metadata
        assert len(observed) == len(local_expected)
        for left, right in zip(observed, local_expected, strict=True):
            assert left.program == right.program
            for left_role, right_role in zip(left.offspring, right.offspring, strict=True):
                assert np.array_equal(left_role, right_role)
            for field in (
                "atom_logits",
                "parent_bond_logits",
                "decoration_anchor_logits",
                "decoration_atom_logits",
                "decoration_bond_logits",
                "hidden",
                "flow_endpoint_atom_states",
                "flow_endpoint_parent_bond_states",
                "flow_endpoint_decoration_anchors",
                "flow_endpoint_decoration_atom_states",
                "flow_endpoint_decoration_bond_states",
            ):
                assert np.array_equal(getattr(left, field), getattr(right, field))
        if batch_size == 2:
            expected = local_expected

    assert expected is not None

    trajectory = initialize_ugi_joint_sparse_state(
        model,
        programs,
        sources,
        sample_steps=3,
        device="cpu",
        seed=41,
    )
    original = trajectory.clone()
    trajectory = advance_ugi_joint_sparse_state(model, trajectory, target_step=1)
    trajectory = advance_ugi_joint_sparse_state(model, trajectory, target_step=3)
    tree_state = torch.Generator().manual_seed(42).get_state()
    finalized = finalize_ugi_joint_sparse_state(
        model,
        trajectory,
        tree_generator_state=tree_state,
    )
    assert original.step == 0
    assert trajectory.step == 3
    for left, right in zip(finalized.terminals, expected, strict=True):
        assert left.program == right.program
        for left_role, right_role in zip(left.offspring, right.offspring, strict=True):
            assert np.array_equal(left_role, right_role)
        assert np.array_equal(left.atom_logits, right.atom_logits)


def test_serial_checkpoint_steps_are_unique_sorted_evaluation_steps() -> None:
    assert _validated_checkpoint_steps(
        {
            "steps": 1_000,
            "eval_every": 250,
            "checkpoint_steps": [1_000, 250, 500, 250],
        }
    ) == (250, 500, 1_000)


@pytest.mark.parametrize(
    "runtime",
    [
        {"steps": 1_000, "eval_every": 0, "checkpoint_steps": []},
        {"steps": 1_000, "eval_every": 250, "checkpoint_steps": [0]},
        {"steps": 1_000, "eval_every": 250, "checkpoint_steps": [300]},
        {"steps": 1_000, "eval_every": 250, "checkpoint_steps": [1_250]},
    ],
)
def test_invalid_serial_checkpoint_steps_are_rejected(runtime: dict[str, object]) -> None:
    with pytest.raises(UgiJointSparseTrainingError):
        _validated_checkpoint_steps(runtime)
