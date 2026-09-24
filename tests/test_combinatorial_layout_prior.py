from pathlib import Path

import numpy as np
import pytest

from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.combinatorial_layout_prior import CombinatorialLayoutPrior
from forge.model.generated_program_layout import layout_signatures
from forge.model.synthesis_program_layout import (
    SynthesisProgramLayoutError,
    SynthesisProgramLayoutPrior,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def prior():
    with SynthesisProgramProductionCache(
        ROOT / "results/phase1/combinatorial_program_cache_v2/cache.npz"
    ) as cache:
        value = CombinatorialLayoutPrior(cache)
        old = SynthesisProgramLayoutPrior.__new__(SynthesisProgramLayoutPrior)
        with pytest.raises(SynthesisProgramLayoutError, match="topology varies.*urea"):
            old.__init__(cache)
    return value, old


def test_all_twelve_count_laws_work_without_source_graphs_or_open_cache(prior):
    model, _ = prior
    assert model.support_report["programs"] == 12
    for family in model.vocabulary.program_states[1:]:
        records = model.sample(family, sample_count=8, seed=2026091604)
        assert len(records) == 8
        for r in records:
            assert r.graph.canonical_smiles == ""
            assert r.node_count <= model.maximum_heavy_atoms
            assert r.graph.closure_count <= model.maximum_closures
            assert not np.any(r.graph.node_states[~r.fixed_atom_mask])
            assert not np.any(r.graph.parents[~r.fixed_parent_bond_mask])
            assert not np.any(r.graph.parent_bonds[~r.fixed_parent_bond_mask])
            assert r.graph.structure_id.startswith("factorized-layout-")


def test_supported_family_draws_match_the_frozen_existing_prior(prior):
    model, old = prior
    # The existing initializer constructs the other eleven laws before rejecting urea.
    for family in model.vocabulary.program_states[1:]:
        if family == "urea_amine_isocyanate":
            continue
        a = model.sample(family, sample_count=4, seed=7331)
        b = old.sample(family, sample_count=4, seed=7331)
        assert [layout_signatures(r, maximum_closures=1) for r in a] == [
            layout_signatures(r, maximum_closures=1) for r in b
        ]


def test_urea_variation_is_retained_and_not_promoted_to_unique_exact_topology(prior):
    model, _ = prior
    assert max(model.topology_support_counts["urea_amine_isocyanate"].values()) > 1
    with pytest.raises(TypeError):
        model.sample("urea_amine_isocyanate", sample_count=1, seed=1, exact_program_topology=True)
    a = model.sample("urea_amine_isocyanate", sample_count=10, seed=91)
    b = model.sample("urea_amine_isocyanate", sample_count=10, seed=91)
    assert [layout_signatures(r, maximum_closures=1) for r in a] == [
        layout_signatures(r, maximum_closures=1) for r in b
    ]
