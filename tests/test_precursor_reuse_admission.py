from __future__ import annotations

import copy
from collections import Counter

import pytest

from forge.model.precursor_reuse import PrecursorReuseError
from forge.model.precursor_reuse_admission import admit_completions


def original(smiles, *, exact=False, family="fixture"):
    return {
        "canonical_smiles": smiles,
        "program_id": family,
        "valid_connected": smiles is not None,
        "assembly": {"status": "exact_computed_program" if exact else "no_exact_program"},
    }


def completion(smiles, *, donor=0, edits=1, status="exact_computed_program"):
    return {
        "proposals": [
            {
                "smiles": smiles,
                "status": status,
                "donor": donor,
                "atom_edits": edits,
                "bond_edits": 0,
            }
        ]
    }


def test_rejects_loss_of_novelty_but_accepts_a_novel_alternative():
    choices = completion("familiar", edits=1)
    choices["proposals"] += completion("novel", donor=1, edits=2)["proposals"]
    selected = admit_completions([original("original")], [choices], {"familiar"})[0]
    assert selected["selected_smiles"] == "novel" and selected["selected_donor"] == 1
    assert selected["proposal_checks"][0]["status"] == "would_reduce_train_novelty"


def test_existing_exact_and_invalid_attempts_are_immutable():
    rows = [original(None), original("existing", exact=True)]
    result = admit_completions(rows, [completion("novel")] * 2, set())
    assert [r["selected_smiles"] for r in result] == [None, "existing"]
    assert [r["disposition"] for r in result] == ["original_invalid", "original_exact"]


def test_failed_exact_gate_is_never_admitted():
    result = admit_completions(
        [original("old")], [completion("new", status="no_exact_program")], set()
    )
    assert result[0]["selected_smiles"] == "old"


def test_prevents_batch_collisions_without_removing_attempts():
    rows = [original("a"), original("b"), original("c", exact=True)]
    proposals = [completion("new"), completion("new"), completion("new")]
    before = copy.deepcopy((rows, proposals))
    result = admit_completions(rows, proposals, set())
    assert [r["selected_smiles"] for r in result] == ["new", "b", "c"]
    assert result[1]["proposal_checks"][0]["status"] == "would_concentrate_products"
    assert (rows, proposals) == before


def test_inventory_is_separate_for_each_family():
    rows = [original("a", family="one"), original("b", family="two")]
    result = admit_completions(rows, [completion("new")] * 2, set())
    assert [r["selected_smiles"] for r in result] == ["new", "new"]


def test_multiplicity_rule_preserves_diversity_over_small_populations():
    # Exhaust actual before/after distributions, including duplicate originals and candidates.
    for a in range(1, 8):
        for b in range(8):
            rows = [original("old")] * a + [original("new", exact=True)] * b
            proposals = [completion("new")] + [{"proposals": []}] * (a + b - 1)
            chosen = admit_completions(rows, proposals, set())
            before = Counter(r["canonical_smiles"] for r in rows)
            after = Counter(r["selected_smiles"] for r in chosen)
            assert len(after) >= len(before)
            assert sum(n * n for n in after.values()) <= sum(n * n for n in before.values())
            assert len(chosen) == len(rows)
            assert sum(r["selected_smiles"] == "new" for r in chosen) >= b


def test_deterministic_ties_and_inconvenient_proposals_are_recorded():
    proposal = completion("z", donor=0)
    proposal["proposals"] += completion("a", donor=1)["proposals"]
    first = admit_completions([original("old")], [proposal], set())
    assert first[0]["selected_smiles"] == "a" and len(first[0]["proposal_checks"]) == 2
    assert first == admit_completions([original("old")], [proposal], set())


def test_malformed_population_and_exact_proposal_fail():
    with pytest.raises(PrecursorReuseError, match="populations"):
        admit_completions([original("old")], [], set())
    with pytest.raises(PrecursorReuseError, match="lacks a graph"):
        admit_completions([original("old")], [completion(None)], set())
