"""Baseline selector equivalence and topology-only graph-label invariants."""

from __future__ import annotations

import csv
import gzip
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from forge.core.hashing import resolve_pin
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model import ugi_topology_candidates as candidate_module
from forge.model import ugi_transformer_topology as original_topology
from forge.model.defog_feasibility import AtomState
from forge.model.local_chemistry_support import LocalChemistrySupport
from forge.model.reaction_program_flow import derive_role_morphology_states
from forge.model.sparse_topology_feasibility import SparseGraphRecord
from forge.model.synthesis_program_graph import (
    SynthesisProgramComponentBlock,
    SynthesisProgramGraphRecord,
)
from forge.model.ugi_amine_semantic_program import UgiAmineSemanticTarget, amine_semantic_target
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy
from forge.model.ugi_morphology_program import (
    ROLE_NAMES,
    UgiMorphologyProgram,
    preorder_attached_forest_to_parents,
)
from forge.model.ugi_sampling_trace import SamplingTrace
from forge.model.ugi_topology_candidates import (
    TopologyCandidate,
    UgiTopologyCandidatesError,
    _colored_graph_key,
    clear_topology_candidate_caches,
    score_candidates,
    topology_candidates,
)
from forge.model.ugi_transformer_topology import (
    UGI_PROGRAM_ID,
    UgiTransformerTopologyPolicy,
    _role_targets,
    _sample_amine_semantic_topology,
    _sample_constructive_ester_offspring,
)

REPO = Path(__file__).resolve().parents[1]


def _fixture(*, cyclic=False, head_word=None, target=None):
    """Synthetic graph/policy fixtures are enumeration tests, not chemical evidence."""
    if head_word is None:
        head_word = (2, 0, 1, 1, 1, 0) if cyclic else (2, 0, 1, 0)
    words = (head_word, (1, 1, 2, 0, 1, 1, 1, 1, 0), (1, 0))
    parents, roles, core_states, fixed, fixed_parent, blocks = [], [], [], [], [], []
    core_nodes, exterior_by_role = [], {}
    for role_index, (role, word) in enumerate(zip(ROLE_NAMES, words, strict=True)):
        start = len(parents)
        core_nodes.append(start)
        parents.append(0 if start == 0 else core_nodes[role_index - 1])
        roles.append(role_index + 1)
        core_states.append(role_index + 2)
        fixed.append(True)
        fixed_parent.append(start != 0)
        local_parents = preorder_attached_forest_to_parents(np.asarray(word), attachment_count=1)
        exterior_by_role[role] = tuple(range(start + 1, start + 1 + len(word)))
        for parent in local_parents:
            parents.append(start if parent < 0 else start + 1 + int(parent))
            roles.append(role_index + 1)
            core_states.append(1)
            fixed.append(False)
            fixed_parent.append(parent < 0)
        blocks.append(SynthesisProgramComponentBlock(role, role_index + 1, start, len(parents)))
    closures = ((2, 6),) if cyclic else ()
    count = len(parents)
    graph = SparseGraphRecord(
        structure_id="synthetic-topology-fixture",
        canonical_smiles="synthetic-heavy-graph-not-a-molecule",
        node_states=np.zeros(count, dtype=np.int64),
        parents=np.asarray(parents),
        parent_bonds=np.zeros(count, dtype=np.int64),
        closure_left=np.asarray([pair[0] for pair in closures], dtype=np.int64),
        closure_right=np.asarray([pair[1] for pair in closures], dtype=np.int64),
        closure_bonds=np.zeros(len(closures), dtype=np.int64),
        edges=np.empty((0, 0), dtype=np.int64),
    )
    record = SynthesisProgramGraphRecord(
        graph=graph,
        canonical_atom_order=np.arange(count),
        program_id=UGI_PROGRAM_ID,
        program_state=1,
        program_depth=1,
        role_states=np.asarray(roles),
        core_position_states=np.asarray(core_states),
        component_blocks=tuple(blocks),
        fixed_atom_mask=np.asarray(fixed),
        fixed_parent_bond_mask=np.asarray(fixed_parent),
        fixed_closure_bond_mask=np.zeros(len(closures), dtype=bool),
    )
    policy = UgiEsterChemotypePolicy(
        reaction_id=UGI_PROGRAM_ID,
        amine_role=ROLE_NAMES[0],
        aldehyde_role=ROLE_NAMES[1],
        isocyanide_role=ROLE_NAMES[2],
        registry_path=Path("fixture-registry.json"),
        registry_sha256="0" * 64,
        training_assignments_path=Path("fixture-train.csv.gz"),
        training_assignments_sha256="1" * 64,
        minimum_role_exterior_atoms=tuple((role, 1) for role in ROLE_NAMES),
        maximum_role_exterior_atoms=tuple((role, 30) for role in ROLE_NAMES),
        minimum_ester_side_carbons=2,
        minimum_ester_long_side_carbons=3,
        minimum_amine_exterior_nitrogens=0,
        maximum_amine_exterior_nitrogens=3,
        morphology_quantile=0.25,
    )
    return {
        "record": record,
        "role": ROLE_NAMES[0],
        "atom_vocabulary": (AtomState("C", 0, False, 0), AtomState("N", 0, False, 0)),
        "topology_policy": UgiTransformerTopologyPolicy((5, 6, 7), 4, (2, 1, 1)),
        "ester_policy": policy,
        "local_chemistry_support": None,
        "amine_target": target
        or UgiAmineSemanticTarget(*((5, 5, 2, 0) if cyclic else (4, 3, 2, 0))),
        "maximum_children": 3,
        "core_position_classes": 8,
    }


def _predictions(record, *, dtype=torch.float32):
    generator = torch.Generator().manual_seed(9701)
    nodes = record.node_count + 2  # Existing batch padding is allowed.
    return {
        "offspring": torch.randn(1, nodes, 4, generator=generator, dtype=dtype),
        "closure_left": torch.randn(1, 3, nodes, generator=generator, dtype=dtype),
        "closure_right": torch.randn(1, 3, nodes, generator=generator, dtype=dtype),
    }


def _original_choice(args, predictions, generator):
    record = args["record"]
    record = replace(record, role_morphology_states=derive_role_morphology_states(record))
    targets = _role_targets(record)
    role = args["role"]
    block = next(block for block in record.component_blocks if block.role == role)
    exterior = np.asarray(
        [node for node in range(block.start, block.stop) if record.core_position_states[node] == 1]
    )
    roots = [
        local
        for local, node in enumerate(exterior)
        if record.fixed_parent_bond_mask[node]
        and record.core_position_states[int(record.graph.parents[node])] > 1
    ]
    fields = {
        name: predictions.get(f"structured_{name}", predictions[name])
        for name in ("offspring", "closure_left", "closure_right")
    }
    policy = args["ester_policy"]
    if role == policy.aldehyde_role:
        return _sample_constructive_ester_offspring(
            fields["offspring"][0, exterior],
            generator=generator,
            minimum_side_carbons=policy.minimum_ester_side_carbons,
            minimum_long_side_carbons=policy.minimum_ester_long_side_carbons,
        )
    count, junctions, cycles, attachments = targets[role]
    return _sample_amine_semantic_topology(
        logits=fields["offspring"][0, exterior],
        target=args["amine_target"],
        junction_budget=junctions,
        cycle_rank=cycles,
        attachment_count=attachments,
        fixed_roots=roots,
        exterior=exterior,
        closure_left_logits=fields["closure_left"][0].numpy(),
        closure_right_logits=fields["closure_right"][0].numpy(),
        first_slot=sum(targets[name][2] for name in ROLE_NAMES[: ROLE_NAMES.index(role)]),
        policy=args["topology_policy"],
        allowed_ring_sizes=policy.allowed_amine_cycle_sizes(count) if cycles else None,
        local_chemistry_support=args["local_chemistry_support"],
        semantic_guidance_policy=None,
        program=UgiMorphologyProgram(
            *(tuple(targets[name][field] for name in ROLE_NAMES) for field in range(4))
        ),
        all_role_target=None,
        generator=generator,
    )


def _assert_trace_equivalence(args, predictions):
    rng_before = torch.random.get_rng_state().clone()
    result = topology_candidates(**args)
    scores = score_candidates(result, predictions)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert scores.dtype == torch.float64 and not scores.requires_grad
    with SamplingTrace(selected_attempts=(0,)) as trace:
        _original_choice(args, predictions, torch.Generator().manual_seed(824))
    event = next(row for row in trace.events if row["kind"] == "categorical_choice")
    expected = [
        (
            {
                "offspring": list(candidate.offspring),
                "closures": [list(pair) for pair in candidate.closures],
            }
            if args["role"] == args["ester_policy"].amine_role
            else list(candidate.offspring)
        )
        for candidate in result.candidates
    ]
    assert [row["identity"] for row in event["candidates"]] == expected
    assert scores.tolist() == [row["neural_score"] for row in event["candidates"]]
    assert scores.softmax(0).tolist() == event["final_probabilities"]
    selected = torch.multinomial(scores.softmax(0), 1, generator=torch.Generator().manual_seed(824))
    assert int(selected) == event["selected_candidate_index"]
    return result


@pytest.mark.parametrize("cyclic", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_head_order_scores_probabilities_and_rng_match_unchanged_selector(cyclic, dtype):
    args = _fixture(cyclic=cyclic)
    result = _assert_trace_equivalence(args, _predictions(args["record"], dtype=dtype))
    assert result.candidate_count > 1 and result.target_present


def test_ester_order_scores_and_no_extra_directional_target_match_unchanged_selector():
    args = _fixture(cyclic=True)
    args["role"] = ROLE_NAMES[1]
    result = _assert_trace_equivalence(args, _predictions(args["record"]))
    assert result.candidate_count > result.graph_class_count > 1
    assert result.target_present
    assert result.first_closure_slot == 1


def test_symmetry_labels_every_correct_serialization_without_collapsing_the_universe():
    args = _fixture()
    result = topology_candidates(**args)
    assert result.candidate_count == 3 and result.graph_class_count == 2
    assert result.positive_mask.tolist() == [False, True, True]
    assert result.graph_class_ids == (0, 1, 1)
    other = topology_candidates(**_fixture(head_word=(2, 1, 0, 0)))
    assert other.target_graph_key == result.target_graph_key
    assert np.array_equal(other.positive_mask, result.positive_mask)
    assert other.ordered_identity_sha256 == result.ordered_identity_sha256


def test_two_fixed_roots_use_decoder_alignment_in_scores_and_target_labels():
    args = _fixture(target=UgiAmineSemanticTarget(5, 2, 1, 0))
    record = args["record"]
    parents = record.graph.parents.copy()
    parents[1:5] = (0, 0, 1, 2)  # Two attachment roots precede their children.
    fixed = record.fixed_parent_bond_mask.copy()
    fixed[1:5] = (True, True, False, False)
    args["record"] = replace(
        record, graph=replace(record.graph, parents=parents), fixed_parent_bond_mask=fixed
    )
    result = _assert_trace_equivalence(args, _predictions(args["record"]))
    assert result.target_present
    assert result.candidates[0].offspring == (1, 0, 1, 0)
    # Decoder alignment maps local forest roots 0,2 onto fixed global exterior slots 1,2.
    assert result.adjacency[0, 0, 2] and result.adjacency[0, 1, 3]


def test_sparse_target_label_ignores_absent_or_stale_dense_edges():
    args = _fixture(cyclic=True)
    first = topology_candidates(**args)
    record = args["record"]
    args["record"] = replace(
        record,
        graph=replace(record.graph, edges=np.zeros((record.node_count, record.node_count))),
    )
    second = topology_candidates(**args)
    assert first.target_graph_key == second.target_graph_key
    assert first.candidate_graph_keys == second.candidate_graph_keys
    assert np.array_equal(first.positive_mask, second.positive_mask)


def test_colors_fix_anchor_identity_but_ignore_exterior_index_and_chemistry():
    colors = (("role", 2, 1), ("role", 3, 0), ("role", 1, 1), ("role", 1, 0))
    assert _colored_graph_key(colors, ((0, 1), (0, 2), (2, 3))) != _colored_graph_key(
        colors, ((0, 1), (1, 2), (2, 3))
    )
    args = _fixture()
    first = topology_candidates(**args)
    record = args["record"]
    states = np.ones_like(record.graph.node_states)
    args["record"] = replace(record, graph=replace(record.graph, node_states=states))
    second = topology_candidates(**args)
    assert second.target_graph_key == first.target_graph_key
    assert np.array_equal(second.positive_mask, first.positive_mask)
    assert second.target_record_sha256 != first.target_record_sha256
    args["role"] = ROLE_NAMES[1]
    ester = topology_candidates(**args)
    assert ester.node_colors.shape[1] == first.node_colors.shape[1] == 9
    assert ester.color_feature_names == first.color_feature_names
    for array in (first.node_colors, first.adjacency, first.global_closures, first.positive_mask):
        assert not array.flags.writeable


def test_absent_target_singleton_and_empty_outcomes_remain_explicit():
    singleton = topology_candidates(**_fixture(target=UgiAmineSemanticTarget(4, 3, 1, 0)))
    assert singleton.candidate_count == 1 and not singleton.target_present
    assert singleton.reason == "target_heavy_skeleton_absent"
    one_class = topology_candidates(**_fixture(target=UgiAmineSemanticTarget(4, 2, 2, 0)))
    assert one_class.candidate_count == 2 and one_class.graph_class_count == 1
    assert one_class.target_present and one_class.reason == "single_heavy_skeleton_class"
    empty = topology_candidates(**_fixture(target=UgiAmineSemanticTarget(2, 0, 5, 0)))
    assert empty.candidate_count == 0 and empty.adjacency.shape[0] == 0
    assert empty.reason == "no_legal_baseline_candidates"
    assert score_candidates(empty, _predictions(_fixture()["record"])).shape == (0,)


def test_support_cache_keys_change_with_policy_and_never_recompute_on_same_support(monkeypatch):
    clear_topology_candidate_caches()
    args = _fixture()
    first = topology_candidates(**args)

    def unexpected(**kwargs):
        raise AssertionError("enumerator called on cached support")

    monkeypatch.setattr(candidate_module, "enumerate_amine_semantic_topologies", unexpected)
    assert topology_candidates(**args).candidates is first.candidates
    changed = dict(
        args, topology_policy=replace(args["topology_policy"], allowed_ring_sizes=(5, 6))
    )
    with pytest.raises(AssertionError, match="enumerator called"):
        topology_candidates(**changed)


def test_float32_closure_addition_and_structured_precedence_are_preserved():
    args = _fixture(cyclic=True)
    predictions = _predictions(args["record"])
    # This addition rounds in float32 before float64 candidate accumulation.
    predictions["closure_left"].fill_(2**24)
    predictions["closure_right"].fill_(1)
    predictions["structured_offspring"] = predictions["offspring"] * -2
    result = _assert_trace_equivalence(args, predictions)
    baseline = dict(predictions)
    baseline["closure_left"] = torch.zeros_like(predictions["closure_left"])
    baseline["closure_right"] = torch.zeros_like(predictions["closure_right"])
    assert torch.equal(
        score_candidates(result, predictions), score_candidates(result, baseline) + 2**24
    )


@pytest.mark.parametrize("kind", ["nan", "batch", "children", "closure_nodes", "dtype"])
def test_malformed_predictions_fail_before_scoring(kind):
    args = _fixture()
    result = topology_candidates(**args)
    predictions = _predictions(args["record"])
    if kind == "nan":
        predictions["offspring"][0, 0, 0] = float("nan")
    elif kind == "batch":
        predictions["offspring"] = predictions["offspring"].repeat(2, 1, 1)
    elif kind == "children":
        predictions["offspring"] = predictions["offspring"][:, :, :3]
    elif kind == "closure_nodes":
        predictions["closure_left"] = predictions["closure_left"][:, :, :-1]
    else:
        predictions["offspring"] = predictions["offspring"].long()
    with pytest.raises(UgiTopologyCandidatesError):
        score_candidates(result, predictions)


def test_malformed_targets_and_unknown_role_fail_instead_of_truncating_support():
    args = _fixture()
    with pytest.raises(UgiTopologyCandidatesError, match="baseline Ugi"):
        topology_candidates(**dict(args, role="unknown"))
    with pytest.raises(UgiTopologyCandidatesError, match="four amine"):
        topology_candidates(**dict(args, amine_target=UgiAmineSemanticTarget(4, 3, 2, 0, 1, 1)))
    with pytest.raises(UgiTopologyCandidatesError, match="vocabulary"):
        topology_candidates(**dict(args, core_position_classes=3))
    too_large = _fixture(
        head_word=tuple([1] * 10 + [0]), target=UgiAmineSemanticTarget(12, 11, 1, 0)
    )
    with pytest.raises(UgiTopologyCandidatesError, match="unsupported.*enumeration"):
        topology_candidates(**too_large)


@pytest.fixture(scope="module")
def real_train_inputs():
    """Authenticate fixture sources; retain component structures only for TRAIN rows."""
    config = json.loads(
        (
            REPO
            / "configs/multireaction/ugi_donor_slack_tiered_head_group_entropy_terminal_offset_bondweight2_vs_amine_joint_support_cpu_seed0_v1.json"
        ).read_text()
    )
    required = (
        "production_cache",
        "ugi_assignments",
        "qualified_reactions",
        "role_morphology_policy",
        "topology_closure_config",
        "topology_morphology_config",
    )
    missing = [
        config["inputs"][key]["path"]
        for key in required
        if not (REPO / config["inputs"][key]["path"]).is_file()
    ]
    if missing:
        pytest.skip(f"documented pinned production fixture assets absent: {missing}")
    paths = {key: resolve_pin(config["inputs"][key], REPO, label=key) for key in required}
    assignments = {}
    with gzip.open(paths["ugi_assignments"], "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["primary_product_fold"] == "train":
                assignments[row["product_id"]] = row["amine_head_smiles"]
    ester = UgiEsterChemotypePolicy.from_qualified_registry(
        paths["qualified_reactions"], training_assignments_path=paths["ugi_assignments"]
    )
    topology = UgiTransformerTopologyPolicy.from_support_documents(
        json.loads(paths["topology_closure_config"].read_text()),
        json.loads(paths["topology_morphology_config"].read_text()),
    )
    local = LocalChemistrySupport.from_mapping(
        json.loads(paths["role_morphology_policy"].read_text())
    )
    return paths, assignments, ester, topology, local


@pytest.fixture(scope="module")
def real_train_case(real_train_inputs):
    """Materialize only explicit TRAIN indices; assert membership before every record read."""
    paths, assignments, ester, topology, local = real_train_inputs
    with SynthesisProgramProductionCache(paths["production_cache"]) as cache:
        train = cache.indices(program_id=UGI_PROGRAM_ID, fold="train")
        for raw_index in train:
            index = int(raw_index)
            if cache.arrays["closure_offsets"][index + 1] == cache.arrays["closure_offsets"][index]:
                continue
            assert cache.fold(index) == "train" and index in train
            record = cache.record(index)
            assert record.graph.structure_id in assignments
            target = amine_semantic_target(assignments[record.graph.structure_id])
            args = {
                "record": record,
                "role": ester.amine_role,
                "atom_vocabulary": cache.atom_vocabulary,
                "topology_policy": topology,
                "ester_policy": ester,
                "local_chemistry_support": local,
                "amine_target": target,
                "maximum_children": 3,
                "core_position_classes": len(cache.vocabulary.core_position_states),
            }
            head = topology_candidates(**args)
            tail = topology_candidates(**dict(args, role=ester.aldehyde_role))
            if head.candidate_count and head.global_closures.shape[1] and tail.candidate_count:
                return args
    pytest.fail("pinned TRAIN fixture has no supported cyclic-head/ester case")


def test_real_selected_train_alignment_failure_preserves_whole_unavailable_law(
    real_train_inputs, monkeypatch
):
    """Regression for prefit v1 failure: no pruning, root changes, or hidden RNG draw."""
    paths, assignments, ester, topology, local = real_train_inputs
    with SynthesisProgramProductionCache(paths["production_cache"]) as cache:
        train = cache.indices(program_id=UGI_PROGRAM_ID, fold="train")
        index = 8075
        assert index in train and cache.fold(index) == "train"
        record = cache.record(index)
        assert record.graph.structure_id == "EUGI-07222adc9f6a6b3563d1"
        assert record.graph.structure_id in assignments
        args = {
            "record": record,
            "role": ester.amine_role,
            "atom_vocabulary": cache.atom_vocabulary,
            "topology_policy": topology,
            "ester_policy": ester,
            "local_chemistry_support": local,
            "amine_target": amine_semantic_target(assignments[record.graph.structure_id]),
            "maximum_children": 3,
            "core_position_classes": len(cache.vocabulary.core_position_states),
        }
        result = topology_candidates(**args)
    assert result.status == "unavailable"
    assert (
        result.reason
        == result.unavailable_reason
        == "root_alignment_would_violate_sparse_parent_ordering"
    )
    assert result.candidates == () and result.candidate_count == result.graph_class_count == 0
    assert result.adjacency.shape[0] == result.positive_mask.size == 0
    assert result.enumerated_candidate_count > 1
    assert result.failed_candidate_index is not None
    predictions = _predictions(record)
    assert score_candidates(result, predictions).numel() == 0
    observed = {}
    original_alignment = original_topology._root_aligned_permutation

    def observe_alignment(parents, roots):
        frame = sys._getframe(1)
        if frame.f_code is _sample_amine_semantic_topology.__code__:
            observed["candidates"] = tuple(
                TopologyCandidate(tuple(map(int, item.offspring)), tuple(item.closures))
                for item in frame.f_locals["candidates"]
            )
            observed["candidate_index"] = len(frame.f_locals["candidate_scores"])
        return original_alignment(parents, roots)

    monkeypatch.setattr(original_topology, "_root_aligned_permutation", observe_alignment)
    generator = torch.Generator().manual_seed(824)
    state_before = generator.get_state().clone()
    with pytest.raises(
        original_topology.UgiTransformerTopologyError, match="sparse parent ordering"
    ):
        with SamplingTrace(selected_attempts=(0,)) as trace:
            _original_choice(args, predictions, generator)
    assert observed["candidates"] == result.enumerated_candidates
    assert observed["candidate_index"] == result.failed_candidate_index
    assert torch.equal(state_before, generator.get_state())
    assert not any(row["kind"] == "categorical_choice" for row in trace.events)


@pytest.mark.parametrize("role", [ROLE_NAMES[0], ROLE_NAMES[1]])
def test_real_train_record_exactly_matches_original_helpers_including_cyclic_head(
    real_train_case, role
):
    args = dict(real_train_case, role=role)
    result = _assert_trace_equivalence(args, _predictions(args["record"]))
    assert result.candidate_count > 1
    if role == ROLE_NAMES[0]:
        assert result.global_closures.shape[1] > 0
