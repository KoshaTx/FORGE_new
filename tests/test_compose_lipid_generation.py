"""Coarse independent layouts, scoped topology constraints and explicit generated reuse."""

import copy
from collections import defaultdict

import numpy as np
import torch
from rdkit import Chem

from forge.corpus.qualified_program_cache import QualifiedProgramExample
from forge.model.compose_lipid_generation import constrained_readout, propose_generated_reuse
from forge.model.compose_lipid_layout import (
    ComposeLipidLayoutPrior,
    build_layout,
    collate_generated_layouts,
    encoded,
    summarize_layout,
)
from forge.model.precursor_reuse_projection import graph_smiles, state_graph
from forge.model.synthesis_program_sampling import _terminal_smiles
from tests.test_compose_lipid_restoration import source
from tests.test_source_instance_coordinates import record


def prior_document(examples, maximum_atoms=254, maximum_closures=12):
    bank = {}
    for e in examples:
        bundle, choices, rings = summarize_layout(e)
        item = bank.setdefault(
            encoded(bundle),
            dict(bundle=bundle, mass=0.0, roles=defaultdict(list), rings=defaultdict(set)),
        )
        item["mass"] += 1
        for role, choice in choices.items():
            item["roles"][str(role)].append(dict(shape=choice, mass=1.0))
        for role, values in rings.items():
            item["rings"][str(role)].update(values)
    for item in bank.values():
        item["rings"] = {k: sorted(v) for k, v in item["rings"].items()}
    return dict(
        schema_version="forge.compose_lipid_layout_prior.v1",
        bundles=list(bank.values()),
        support=dict(maximum_atoms=maximum_atoms, maximum_closures=maximum_closures),
    )


def gold_predictions(record, atoms, maximum_closures=12):
    g = record.graph
    values = dict(
        nodes=g.node_states,
        parents=g.parents,
        parent_bonds=g.parent_bonds,
        closure_left=g.closure_left,
        closure_right=g.closure_right,
        closure_bonds=g.closure_bonds,
    )
    out = {}
    for field, values in values.items():
        classes = (
            len(atoms)
            if field == "nodes"
            else (3 if field in ("parent_bonds", "closure_bonds") else record.node_count)
        )
        if field.startswith("closure"):
            values = np.pad(values, (0, maximum_closures - len(values)))
        out[field] = torch.nn.functional.one_hot(torch.tensor(values)[None], classes).float() * 100
    return out


def test_independent_role_size_draws_and_exact_support_conditioning():
    examples = []
    for head, tail in ((3, 2), (5, 4)):
        n = head + tail
        r, _, _ = record(
            "N" + "C" * (n - 1),
            ["head"] * head + ["tail"] * tail,
            ["exterior"] * (head - 1) + ["left", "right"] + ["exterior"] * (tail - 1),
        )
        examples.append(
            QualifiedProgramExample(
                r, "f", (("head", "private-A", 1), ("tail", "private-B", 1)), ()
            )
        )
    doc = prior_document(examples, maximum_atoms=7)
    assert len(doc["bundles"]) == 1
    assert not any(
        v in encoded(doc) for v in ("private-A", "private-B", "source-control", "SMILES")
    )
    prior = ComposeLipidLayoutPrior(doc)
    rng = np.random.default_rng(29)
    counts = defaultdict(int)
    for i in range(1500):
        layout = prior.sample("f", rng=rng, identity=f"draw-{i}")
        sizes = tuple(sorted(b.atom_count for b in layout.record.component_blocks))
        counts[sizes] += 1
        assert layout.record.node_count <= 7
        assert not layout.record.graph.canonical_smiles
        assert not layout.record.graph.node_states[~layout.record.fixed_atom_mask].any()
        collate_generated_layouts([layout], maximum_closures=12)
    # Independent role draws yield three equally weighted feasible combinations;
    # the fourth (5+4) exceeds support and is conditioned out, not clipped/retried.
    assert set(counts) == {(2, 3), (3, 4), (2, 5)}
    assert all(abs(n / 1500 - 1 / 3) < 0.04 for n in counts.values())
    assert not any(b.atom_count < 2 for b in layout.record.component_blocks)


def test_ring_touching_core_remains_exact_under_role_constraints():
    e, _, atoms = source()
    bundle, choices, rings = summarize_layout(e)
    layout = build_layout(bundle, choices, rings, identity="independent")
    batch = collate_generated_layouts([layout], maximum_closures=12)
    prediction = gold_predictions(e.record, atoms)
    state, reasons = constrained_readout(
        prediction, batch, [layout], atoms, rings=True, morphology=True
    )
    assert reasons == (None,)
    assert _terminal_smiles(
        state, 0, e.record.node_count, e.record.graph.closure_count, atoms
    ) == Chem.MolToSmiles(Chem.MolFromSmiles(e.record.graph.canonical_smiles))
    unsupported = copy.deepcopy(layout)
    unsupported.ring_sizes_by_role.clear()
    _, reasons = constrained_readout(prediction, batch, [unsupported], atoms, rings=True)
    assert reasons[0] is not None


def test_explicit_repeat_completion_preserves_core_and_copies_generated_interiors():
    r, _, atoms = record(
        "N(CC)CO",
        ["head", "tail", "tail", "tail", "tail"],
        ["amine", "carbon", "exterior", "carbon", "exterior"],
    )
    e = QualifiedProgramExample(r, "f", (("head", "head-id", 1), ("tail", "same-id", 2)), ())
    bundle, choices, rings = summarize_layout(e)
    layout = build_layout(bundle, choices, rings, identity="generated")
    state = dict(
        nodes=r.graph.node_states.tolist(),
        parents=r.graph.parents.tolist(),
        parent_bonds=r.graph.parent_bonds.tolist(),
        closure_left=[],
        closure_right=[],
        closure_bonds=[],
    )
    proposals = propose_generated_reuse(layout, state, atoms)
    assert proposals["status"] == "proposals_complete"
    assert {p["smiles"] for p in proposals["proposals"]} == {"CCNCC", "OCNCO"}
    assert all(p["status"] == "requires_exact_program_check" for p in proposals["proposals"])
    # The original is retained; copying generated donors is a separately labeled intervention.
    n, a = state_graph(state)
    assert graph_smiles(n, a, atoms) == Chem.MolToSmiles(
        Chem.MolFromSmiles(r.graph.canonical_smiles)
    )
