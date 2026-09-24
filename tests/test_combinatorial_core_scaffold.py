from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.combinatorial_core_scaffold import (
    CoreScaffoldError,
    CoreScaffoldPrior,
    apply_core_scaffold,
    canonical_core_slots,
    core_coordinates,
    extract_core,
)
from forge.model.combinatorial_layout_prior import CombinatorialLayoutPrior
from forge.model.generated_program_layout import layout_signatures
from forge.model.paired_core_decoding import paired_terminal_core_decode
from forge.model.synthesis_program_sampling import load_synthesis_program_checkpoint

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def priors():
    with SynthesisProgramProductionCache(
        ROOT / "results/phase1/combinatorial_program_cache_v2/cache.npz"
    ) as cache:
        yield CombinatorialLayoutPrior(cache), CoreScaffoldPrior(cache), cache


def test_core_bank_retains_all_train_weight_without_product_or_exterior_identity(priors):
    _, prior, cache = priors
    assert prior.training_records == len(cache.indices(fold="train")) == 64316
    assert len(prior.report()["by_family"]) == 12
    for family in prior.report()["by_family"]:
        record = cache.record(int(cache.indices(program_id=family, fold="train")[0]))
        key, core = extract_core(record)
        assert core in prior.support[key][0]
        assert len(core.node_states) == np.count_nonzero(record.core_position_states > 1)
        assert all(0 <= a < b < len(core.node_states) for a, b, _ in core.edges)
    assert all(np.isclose(p.sum(), 1) for _, p in prior.support.values())
    assert not hasattr(prior, "cache")


def test_core_clamping_preserves_counts_roles_and_all_variable_exterior_slots(priors):
    counts, cores, _ = priors
    for family in counts.vocabulary.program_states[1:]:
        for record in counts.sample(family, sample_count=4, seed=731):
            base = canonical_core_slots(record)
            assert core_coordinates(base)[0] == core_coordinates(record)[0]
            core = cores.sample(base, seed=732)
            revised, reason = apply_core_scaffold(base, core)
            if revised is None:
                assert reason == "core_serialization_exceeds_sampled_closure_slots"
                continue
            assert revised.node_count == base.node_count
            assert revised.graph.closure_count == base.graph.closure_count
            assert revised.component_blocks == base.component_blocks
            assert np.array_equal(revised.role_states, base.role_states)
            assert np.array_equal(revised.core_position_states, base.core_position_states)
            exterior = base.core_position_states == 1
            assert np.array_equal(
                revised.graph.node_states[exterior], base.graph.node_states[exterior]
            )
            assert np.array_equal(revised.fixed_atom_mask[exterior], base.fixed_atom_mask[exterior])
            assert not revised.graph.canonical_smiles
            assert np.all(revised.fixed_atom_mask[~exterior])


def test_already_fixed_ugi_core_is_identity_control(priors):
    counts, cores, _ = priors
    r = counts.sample("ugi_3cr_agile", sample_count=1, seed=761)[0]
    after, reason = apply_core_scaffold(r, cores.sample(r, seed=762))
    assert reason is None
    assert layout_signatures(r, maximum_closures=1) == layout_signatures(after, maximum_closures=1)


def test_scaffold_rejects_semantic_substitution_and_preserves_abstention(priors):
    counts, cores, _ = priors
    r = canonical_core_slots(counts.sample("acetal_aldehyde_diol", sample_count=1, seed=771)[0])
    c = cores.sample(r, seed=772)
    with pytest.raises(CoreScaffoldError, match="support differs"):
        apply_core_scaffold(r, replace(c, semantic_key=("wrong",)))
    no_closures = replace(
        r,
        graph=replace(
            r.graph,
            closure_left=np.zeros(0, dtype=np.int64),
            closure_right=np.zeros(0, dtype=np.int64),
            closure_bonds=np.zeros(0, dtype=np.int64),
        ),
        fixed_closure_bond_mask=np.zeros(0, dtype=np.bool_),
    )
    result, reason = apply_core_scaffold(no_closures, c)
    assert result is None and reason == "core_serialization_exceeds_sampled_closure_slots"


def test_terminal_identity_control_and_capture_cleanup(priors):
    counts, _, _ = priors
    model, _, atoms, nodes, bonds, _ = load_synthesis_program_checkpoint(
        ROOT / "results/phase1/combinatorial_checkpoint_v1/checkpoint.json", device="cpu"
    )
    records = counts.sample("ugi_3cr_agile", sample_count=1, seed=781)
    before = len(model._forward_hooks)
    a, b, evidence = paired_terminal_core_decode(
        model, records, records, atoms, nodes, bonds, seed=782, flow_steps=2
    )
    assert a == b
    assert evidence["model_forward_calls"] == 3
    assert evidence["identity_control_exact"]
    assert len(model._forward_hooks) == before
    bad = replace(records[0], program_depth=2)
    with pytest.raises(ValueError, match="identical layout"):
        paired_terminal_core_decode(
            model, records, [bad], atoms, nodes, bonds, seed=782, flow_steps=2
        )
