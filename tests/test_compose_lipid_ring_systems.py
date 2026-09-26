"""Ring topology corrections preserve generated slots and immutable chemistry cores."""

from dataclasses import replace
from functools import partial

import numpy as np
import pytest
from rdkit import Chem

from forge.model.compose_lipid_ring_systems import (
    RingSystemLimits,
    extract_ring_system_motifs,
    propose_ring_systems,
)
from forge.model.precursor_reuse_projection import fixed_graph_preserved, state_graph
from forge.model.qualified_vocabulary import QualifiedAtomVocabulary
from tests.test_compose_lipid_ring_quality import prepared


def run(smiles, source, *, allowed=None):
    atoms, layout, state, pred = prepared(smiles)
    nodes, edges = state_graph(state)
    if allowed is not None:
        layout = replace(
            layout,
            ring_sizes_by_role={
                b.role_state: tuple(allowed) for b in layout.record.component_blocks
            },
        )
    result = propose_ring_systems(
        layout, nodes, edges, atoms, pred, {"one": extract_ring_system_motifs(source)}
    )
    return atoms, layout, nodes, edges, result


def test_macrocycle_is_shortened_without_dropping_its_oxygenated_atoms():
    atoms, layout, nodes, edges, result = run(
        "CCC1CCOCCOCCCCCOCCCCC=CCCCCCCO1", ["CCc1ccccc1"], allowed=(6,)
    )
    assert result["proposals"]
    for candidate in result["proposals"]:
        mol = Chem.MolFromSmiles(candidate["smiles"])
        assert [len(r) for r in mol.GetRingInfo().AtomRings()] == [6]
        assert candidate["nodes"] == nodes.tolist()
        assert fixed_graph_preserved(nodes, np.asarray(candidate["edges"]), layout.record)
        assert np.array_equal(state_graph(candidate["tree_state"])[1], candidate["edges"])
        assert mol.GetNumAtoms() == len(nodes)
        assert sum(a.GetAtomicNum() == 8 for a in mol.GetAtoms()) == 4
        assert candidate["design_audit"]["complete_typed_ring_systems_observed"]
    assert len(result["proposals"]) <= 2
    assert result["accounting"]["states_explored"] <= 256


def test_cage_motif_rewires_existing_carbon_slots():
    atoms, layout, state, pred = prepared("CC12CC3CC(CC(C3)C1)C2")
    nodes, edges = state_graph(state)
    # A non-core bond swap creates a different connected ten-carbon three-cycle cage.
    pool = np.flatnonzero(layout.record.core_position_states == 1)
    motifs = {"one": extract_ring_system_motifs(["CC12CC3CC(CC(C3)C1)C2"])}
    wrong = None
    for a, b in zip(*np.nonzero(np.triu(edges)), strict=True):
        if a not in pool or b not in pool:
            continue
        for c in pool:
            if c in (a, b) or edges[a, c]:
                continue
            trial = edges.copy()
            trial[a, b] = trial[b, a] = 0
            trial[a, c] = trial[c, a] = 1
            from forge.model.precursor_reuse_projection import graph_smiles

            s = graph_smiles(nodes, trial, atoms)
            if s and sorted(len(r) for r in Chem.MolFromSmiles(s).GetRingInfo().AtomRings()) != [
                6,
                6,
                6,
                6,
            ]:
                wrong = trial
                break
        if wrong is not None:
            break
    assert wrong is not None
    result = propose_ring_systems(layout, nodes, wrong, atoms, pred, motifs)
    assert result["proposals"]
    for candidate in result["proposals"]:
        molecule = Chem.MolFromSmiles(candidate["smiles"])
        assert sorted(len(r) for r in molecule.GetRingInfo().AtomRings()) == [6, 6, 6, 6]
        assert candidate["nodes"] == nodes.tolist()
        assert np.array_equal(state_graph(candidate["tree_state"])[1], candidate["edges"])


@pytest.mark.parametrize("smiles", ["CCc1ncccc1", "CCC1CCCCC1", "CCC12CC3CC(CC(C3)C1)C2"])
def test_observed_source_ring_systems_are_unchanged(smiles):
    *_, result = run(smiles, [smiles])
    assert result["proposals"] == []
    assert result["original"]["design_audit"]["complete_typed_ring_systems_observed"]


def test_incompatible_atom_types_abstain_and_keep_original():
    *_, result = run("CC1NNNNN1", ["CCc1ccccc1"])
    assert not result["proposals"]
    assert result["original"]["smiles"]


@pytest.mark.parametrize("smiles", ["CCn1cccc1", "CCc1ccc2[nH]ccc2c1"])
def test_aromatic_fragment_missing_exterior_context_is_unassessed(smiles):
    # The full molecule is valid, but isolating its ring loses the pyrrolic
    # nitrogen's exterior/explicit-H context. It must not abort a whole panel.
    *_, result = run(smiles, ["CCc1ccccc1"])
    assert result["proposals"] == []
    assert result["original"]["smiles"] == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    assert "ring_fragment_requires_external_context" in result["abstentions"]
    audit = result["original"]["design_audit"]
    assert not audit["complete_typed_ring_systems_observed"]
    assert audit["ring_systems"][0]["identity"] is None
    assert audit["ring_systems"][0]["reason"] == "ring_fragment_requires_external_context"


def test_fixed_ring_core_is_never_rewired():
    atoms, layout, state, pred = prepared("CCC1CCCCCCC1")
    nodes, edges = state_graph(state)
    mask = layout.record.fixed_atom_mask.copy()
    mask[:] = True
    layout = replace(layout, record=replace(layout.record, fixed_atom_mask=mask))
    result = propose_ring_systems(
        layout, nodes, edges, atoms, pred, {"one": extract_ring_system_motifs(["CCc1ccccc1"])}
    )
    assert not result["proposals"]


def test_full_support_254_atoms_12_closures_br(monkeypatch):
    monkeypatch.setattr(
        "tests.test_source_instance_coordinates.QualifiedAtomVocabulary",
        partial(QualifiedAtomVocabulary, neutral_monovalent_extensions=("Br",)),
    )
    atoms, layout, nodes, edges, result = run("C" * 181 + "C1CCCCC1" * 12 + "Br", ["CC1CCCCC1"])
    assert len(nodes) == 254 and layout.record.graph.closure_count == 12
    assert result["original"]["smiles"] and not result["proposals"]


def test_invalid_search_caps_rejected():
    with pytest.raises(ValueError, match="positive"):
        RingSystemLimits(maximum_states_per_system=0)


def test_saved_a54_macrocycle_repair_keeps_three_arm_identity():
    import gzip
    import json
    from pathlib import Path

    import torch

    root = Path("results/phase1/compose_lipid_quality_confirmation_v1")
    if not (root / "constructed/base-4-0040.json.gz").exists():
        pytest.skip("Pinned diagnostic artifact unavailable")
    payload = torch.load(root / "input.pt", weights_only=False, map_location="cpu")
    pred = torch.load(root / "logits/draw-4-0040.pt", weights_only=False, map_location="cpu")
    rows = json.loads(gzip.decompress((root / "constructed/base-4-0040.json.gz").read_bytes()))
    before = next(r for r in rows if r["index"] == 41)["branches"]["d1"]["proposals"][-1]
    reference = json.loads(
        Path("results/phase1/compose_lipid_quality_v1/reference.json").read_text()
    )
    patterns = {
        role: extract_ring_system_motifs(
            [reference["component_smiles_by_identity"][i] for i in ids]
        )
        for role, ids in reference["component_ids_by_family_role"][
            "a3_amine_aldehyde_alkyne"
        ].items()
    }
    result = propose_ring_systems(
        payload["layouts"][41],
        before["nodes"],
        before["edges"],
        payload["atoms"],
        {k: v[1].numpy() for k, v in pred.items()},
        patterns,
    )
    assert result["proposals"]
    for candidate in result["proposals"]:
        molecule = Chem.MolFromSmiles(candidate["smiles"])
        assert molecule.GetNumAtoms() == 154
        assert sorted(len(r) for r in molecule.GetRingInfo().AtomRings()) == [6, 6, 6]
        assert candidate["motif_ids"][0]["tied_occurrences"] == 3
        assert candidate["nodes"] == before["nodes"]


def test_observed_motif_does_not_override_narrower_requested_ring_size():
    *_, result = run("CCC1CCCCCCC1", ["CCC1CCCCCCC1", "CCc1ccccc1"], allowed=(6,))
    assert result["proposals"]
    for candidate in result["proposals"]:
        molecule = Chem.MolFromSmiles(candidate["smiles"])
        assert [len(r) for r in molecule.GetRingInfo().AtomRings()] == [6]


def test_malformed_logits_fail_before_search():
    atoms, layout, state, pred = prepared("CCC1CCCCC1")
    nodes, edges = state_graph(state)
    pred["parents"] = np.zeros(3)
    with pytest.raises(ValueError, match="insufficient atom support"):
        propose_ring_systems(layout, nodes, edges, atoms, pred, {})
