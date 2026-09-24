from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.combinatorial_core_order import CoreOrderPrior, order_core_slots
from forge.model.combinatorial_core_scaffold import (
    CoreScaffoldError,
    CoreScaffoldPrior,
    apply_core_scaffold,
)
from forge.model.combinatorial_layout_prior import CombinatorialLayoutPrior

ROOT = Path(__file__).resolve().parents[1]


def graph_signature(core):
    """Canonical labelled graph; artificial isotope labels encode semantics, not chemistry."""
    labels = [
        (role, position) for role, positions in core.semantic_key[2] for position in positions
    ]
    mol = Chem.RWMol()
    for (role, position), state in zip(labels, core.node_states, strict=True):
        atom = Chem.Atom(6)
        atom.SetIsotope((role * 64 + position) * 32 + state)
        mol.AddAtom(atom)
    for a, b, bond in core.edges:
        mol.AddBond(
            a,
            b,
            (
                Chem.BondType.SINGLE,
                Chem.BondType.DOUBLE,
                Chem.BondType.TRIPLE,
                Chem.BondType.AROMATIC,
            )[bond],
        )
    return Chem.MolToSmiles(mol, canonical=True)


def test_supported_order_preserves_typed_core_graph_and_all_count_support():
    with SynthesisProgramProductionCache(
        ROOT / "results/phase1/combinatorial_program_cache_v2/cache.npz"
    ) as cache:
        counts, cores, orders = (
            CombinatorialLayoutPrior(cache),
            CoreScaffoldPrior(cache),
            CoreOrderPrior(cache),
        )
    assert not hasattr(orders, "cache")
    qualified = set()
    for family in counts.vocabulary.program_states[1:]:
        for record in counts.sample(family, sample_count=4, seed=981):
            core = cores.sample(record, seed=982)
            ordering = orders.sample(core, seed=983)
            after, moved = order_core_slots(record, core, ordering)
            assert graph_signature(core) == graph_signature(moved)
            assert after.node_count == record.node_count
            assert after.graph.closure_count == record.graph.closure_count
            assert Counter((b.role_state, b.atom_count) for b in after.component_blocks) == Counter(
                (b.role_state, b.atom_count) for b in record.component_blocks
            )
            assert not np.any(after.graph.node_states[~after.fixed_atom_mask])
            scaffold, reason = apply_core_scaffold(after, moved)
            if scaffold is not None:
                qualified.add(family)
            else:
                assert reason == "core_serialization_exceeds_sampled_closure_slots"
            with pytest.raises(CoreScaffoldError, match="traversal differs"):
                order_core_slots(record, core, ())
    assert qualified == set(counts.vocabulary.program_states[1:])
