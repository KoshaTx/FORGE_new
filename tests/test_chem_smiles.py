"""Tests for forge.chem.smiles.

The load-bearing tests are the agreement ones. `forge.chem` may only replace the existing
canonicalizers if it returns exactly what they return -- canonical identity decides which
structures count as the same molecule, so a difference here would not raise, it would quietly
change what the corpus contains.
"""

from __future__ import annotations

import pytest

from forge.chem.smiles import (
    ChemError,
    cache_stats,
    canonical_connected_constitution,
    canonical_constitution,
    canonical_isomeric,
    clear_caches,
    is_valid,
    parse_smiles,
    same_constitution,
)

CIS = r"CCCCCCCC/C=C\CCCCCCCC"
TRANS = r"CCCCCCCC/C=C/CCCCCCCC"
SALT = "CCN.Cl"


# ------------------------------------------------------------------ agreement with what it replaces


def test_agrees_with_the_canonicalizer_most_modules_route_through() -> None:
    legacy = pytest.importorskip("forge.potency.ugi_distributional_applicability")
    for smiles in ("CCO", CIS, TRANS, SALT, "c1ccccc1", "C1=CC=CC=C1", "C[C@H](N)C(=O)O"):
        assert canonical_constitution(smiles) == legacy._canonical(smiles)


def test_agrees_with_the_connected_variant() -> None:
    legacy = pytest.importorskip("forge.route.ugi3_source_neutral_proposal_adjudication")
    for smiles in ("CCO", CIS, "c1ccccc1"):
        assert canonical_connected_constitution(smiles) == legacy._canonical(smiles)


# ------------------------------------------------------------------ the identity the project declares


def test_identity_is_stereo_free() -> None:
    """Model-facing identity is constitutional: cis and trans collapse to one graph."""
    assert canonical_constitution(CIS) == canonical_constitution(TRANS)
    assert same_constitution(CIS, TRANS)


def test_identity_ignores_chirality_and_isotopes() -> None:
    assert same_constitution("C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O")
    assert same_constitution("[13CH4]", "C")


def test_identity_is_insensitive_to_input_spelling() -> None:
    assert same_constitution("c1ccccc1", "C1=CC=CC=C1")
    assert same_constitution("OCC", "CCO")


def test_identity_still_separates_different_molecules() -> None:
    """Stereo-free must not mean everything collapses."""
    assert not same_constitution("CCO", "CCN")
    assert not same_constitution("CCO", "CCCO")


# ------------------------------------------------------------------ the two forms differ only in admission


def test_the_two_forms_agree_on_connected_input() -> None:
    for smiles in ("CCO", CIS, "c1ccccc1"):
        assert canonical_constitution(smiles) == canonical_connected_constitution(smiles)


def test_only_the_connected_form_rejects_a_salt() -> None:
    assert canonical_constitution(SALT) == "CCN.Cl"
    with pytest.raises(ChemError, match="single connected molecule"):
        canonical_connected_constitution(SALT)


# ------------------------------------------------------------------ failure behaviour


@pytest.mark.parametrize("bad", ["not a molecule", "C(((", "[CH5]"])
def test_invalid_smiles_raises_rather_than_returning_none(bad: str) -> None:
    with pytest.raises(ChemError):
        canonical_constitution(bad)
    assert not is_valid(bad)


def test_empty_smiles_is_accepted_as_an_empty_molecule() -> None:
    """A documented RDKit quirk, preserved deliberately.

    `Chem.MolFromSmiles("")` returns an empty molecule rather than None, so the empty string
    round-trips instead of raising. The canonicalizer this replaces behaves identically, and
    matching it is the requirement -- but it does mean an empty structure can travel through a
    pipeline looking well formed, so callers that require a real molecule must check separately.
    """
    assert canonical_constitution("") == ""
    assert is_valid("")


def test_parse_smiles_raises_on_invalid_input() -> None:
    with pytest.raises(ChemError, match="invalid molecular graph"):
        parse_smiles("C(((")


# ------------------------------------------------------------------ caching


def test_molecules_are_not_shared_between_callers() -> None:
    """Chem.Mol is mutable, so parse_smiles must never hand back a cached instance."""
    first, second = parse_smiles("CCO"), parse_smiles("CCO")
    assert first is not second


def test_canonicalization_is_memoized() -> None:
    clear_caches()
    canonical_constitution("CCO")
    canonical_constitution("CCO")
    stats = cache_stats()["canonical_constitution"]
    assert stats["misses"] == 1 and stats["hits"] == 1


def test_cache_does_not_change_results() -> None:
    clear_caches()
    cold = canonical_constitution(CIS)
    warm = canonical_constitution(CIS)
    clear_caches()
    assert cold == warm == canonical_constitution(CIS)


# ------------------------------------------------------------------ the caller's own error type


class DomainError(ValueError):
    """Stands in for the module-specific errors every local helper raises."""


def test_error_parameter_raises_the_callers_type() -> None:
    """Without this, not one existing helper is a drop-in replacement."""
    for fn in (canonical_constitution, canonical_connected_constitution, canonical_isomeric):
        with pytest.raises(DomainError):
            fn("C(((", error=DomainError)
    with pytest.raises(DomainError):
        parse_smiles("C(((", error=DomainError)


def test_translated_error_keeps_the_original_as_cause() -> None:
    """Losing the cause turns a bad row in a large ledger into a dead end."""
    try:
        canonical_constitution("C(((", error=DomainError)
    except DomainError as exc:
        assert isinstance(exc.__cause__, ChemError)
    else:
        pytest.fail("expected DomainError")


def test_default_error_is_still_chemerror() -> None:
    with pytest.raises(ChemError):
        canonical_constitution("C(((")


def test_error_parameter_does_not_alter_success() -> None:
    assert canonical_constitution("CCO", error=DomainError) == canonical_constitution("CCO")


def test_connected_check_also_translates() -> None:
    with pytest.raises(DomainError, match="single connected molecule"):
        canonical_connected_constitution(SALT, error=DomainError)


# ------------------------------------------------------------------ the isomeric form


def test_isomeric_form_keeps_what_the_constitutional_form_discards() -> None:
    """The whole reason it is a separate function."""
    assert canonical_isomeric(CIS) != canonical_isomeric(TRANS)
    assert canonical_constitution(CIS) == canonical_constitution(TRANS)


def test_isomeric_form_keeps_chirality_and_isotopes() -> None:
    assert canonical_isomeric("C[C@H](N)C(=O)O") != canonical_isomeric("C[C@@H](N)C(=O)O")
    assert canonical_isomeric("[13CH4]") != canonical_isomeric("C")


def test_isomeric_and_constitutional_agree_when_there_is_no_stereochemistry() -> None:
    for smiles in ("CCO", "CCN", "c1ccccc1"):
        assert canonical_isomeric(smiles) == canonical_constitution(smiles)


def test_isomeric_form_is_cached_separately() -> None:
    clear_caches()
    canonical_isomeric(CIS)
    canonical_isomeric(CIS)
    assert cache_stats()["canonical_isomeric"] == {"hits": 1, "misses": 1, "size": 1}
