"""Tests for forge.core.types.

NewType is erased at runtime, so there is nothing to assert about the aliases themselves -- their
value is entirely to the type checker. What is worth testing is the enums, because they encode
vocabularies AGENTS.md treats as settled and a typo in one currently fails silently at whatever
downstream string comparison happens to run.
"""

from __future__ import annotations

from forge.core.types import (
    ClaimClass,
    EvidenceStatus,
    RoleName,
    Smiles,
    SupportTier,
    SynthesisLayer,
)


def test_newtypes_are_free_at_runtime() -> None:
    """Adoption has to be incremental, which requires the alias to be the underlying value."""
    assert Smiles("CCO") == "CCO"
    assert isinstance(Smiles("CCO"), str)


def test_role_names_serialize_the_spelling_the_artifacts_use() -> None:
    """The long form dominates serialized data ~500:1, so it is what round-trips."""
    assert {role.value for role in RoleName} == {
        "amine_head",
        "oxoester_aldehyde_body_tail",
        "isocyanide_tail",
    }


def test_role_names_parse_either_spelling() -> None:
    """Two vocabularies exist across 17 declarations; both must resolve to one role."""
    for long_form, short_form in (
        ("amine_head", "amine"),
        ("oxoester_aldehyde_body_tail", "aldehyde"),
        ("isocyanide_tail", "isocyanide"),
    ):
        assert RoleName.parse(long_form) is RoleName.parse(short_form)


def test_role_short_spellings_round_trip() -> None:
    for role in RoleName:
        assert RoleName.parse(role.short) is role
        assert RoleName.parse(role.value) is role


def test_unknown_role_spelling_is_rejected() -> None:
    """A third spelling must fail loudly rather than pass through as a string."""
    import pytest

    with pytest.raises(ValueError, match="unknown Ugi role"):
        RoleName.parse("acid")


def test_support_tiers_use_the_current_vocabulary() -> None:
    """E0-E3, not the superseded I/F/B/N. E2 is the paper's primary open-endedness claim."""
    assert [tier.value for tier in SupportTier] == ["E0", "E1", "E2", "E3"]


def test_synthesis_layers_are_ordered_assembly_to_procurement() -> None:
    assert [layer.value for layer in SynthesisLayer] == ["L1", "L2", "L3"]


def test_enums_compare_equal_to_their_string_values() -> None:
    """Str-valued so they can be adopted without rewriting existing comparisons or JSON."""
    assert RoleName.ALDEHYDE == "oxoester_aldehyde_body_tail"
    assert SupportTier.E2 == "E2"
    assert EvidenceStatus.VERIFIED_FROZEN == "verified_frozen"
    assert ClaimClass.COMPUTED == "Computed"


def test_evidence_statuses_cover_the_matrix_vocabulary() -> None:
    """These are the six the evidence matrix declares; a seventh would be a contract change."""
    assert {status.value for status in EvidenceStatus} == {
        "verified_frozen",
        "verified_nonselecting",
        "diagnostic_only",
        "trained_pending_evaluation",
        "proposed_pending",
        "historical_stale",
    }


def test_nonselecting_is_distinct_from_frozen() -> None:
    """The distinction the manuscript contract cares about: a descriptive audit is not a gate."""
    assert EvidenceStatus.VERIFIED_NONSELECTING != EvidenceStatus.VERIFIED_FROZEN
