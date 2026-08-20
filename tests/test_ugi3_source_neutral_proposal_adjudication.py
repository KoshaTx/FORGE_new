from __future__ import annotations

from forge.synthesis.sources.ugi3_source_neutral_proposal_adjudication import (
    DIRECT_ALDEHYDE,
    ISOCYANIDE,
    SourcePair,
    assess_pair_scope,
    build_scope_profiles,
    structural_features,
)


def test_structural_features_distinguish_supported_handles() -> None:
    aldehyde = structural_features("CCCCCCCCCCCCCCCC=O")
    isocyanide = structural_features("[C-]#[N+]CCCCCCCCCCCC")
    assert aldehyde["aldehyde_count"] == 1
    assert aldehyde["isocyanide_count"] == 0
    assert isocyanide["isocyanide_count"] == 1
    assert isocyanide["aldehyde_count"] == 0


def test_source_pair_scope_accepts_interpolation_and_rejects_wrong_functionality() -> None:
    pairs = (
        SourcePair(DIRECT_ALDEHYDE, "a", "CCCCCCCCCCCCCCCCO", "CCCCCCCCCCCCCCCC=O"),
        SourcePair(DIRECT_ALDEHYDE, "b", "CCCCCCCCCCCCCCCCCCO", "CCCCCCCCCCCCCCCCCC=O"),
        SourcePair(ISOCYANIDE, "c", "CCCCCCCCCCCNC=O", "[C-]#[N+]CCCCCCCCCCC"),
        SourcePair(ISOCYANIDE, "d", "CCCCCCCCCCCCNC=O", "[C-]#[N+]CCCCCCCCCCCC"),
    )
    # The production builder expects all three programs; isolate the same logic
    # through a minimal direct profile assembled from two legitimate members.
    from forge.synthesis.sources.ugi3_source_neutral_proposal_adjudication import (
        PairScopeProfile,
        _feature_profile,
        _pair_similarity,
    )

    members = pairs[:2]
    target_numeric, target_categorical = _feature_profile([row.target for row in members])
    precursor_numeric, precursor_categorical = _feature_profile([row.precursor for row in members])
    profile = PairScopeProfile(
        program_family=DIRECT_ALDEHYDE,
        pairs=members,
        similarity_threshold=max(0.55, _pair_similarity(members[0], members[1])),
        target_numeric_bounds=target_numeric,
        precursor_numeric_bounds=precursor_numeric,
        target_categorical_values=target_categorical,
        precursor_categorical_values=precursor_categorical,
    )
    accepted = assess_pair_scope(
        program_family=DIRECT_ALDEHYDE,
        precursor="CCCCCCCCCCCCCCCCCO",
        target="CCCCCCCCCCCCCCCCC=O",
        profile=profile,
    )
    rejected = assess_pair_scope(
        program_family=DIRECT_ALDEHYDE,
        precursor="CCCCCCCCCCCCCCCCCNC=O",
        target="[C-]#[N+]CCCCCCCCCCCCCCCCC",
        profile=profile,
    )
    assert accepted["qualified"] is True
    assert rejected["qualified"] is False


def test_profile_builder_requires_every_declared_program() -> None:
    pairs = (
        SourcePair(DIRECT_ALDEHYDE, "a", "CCCCCCCCCCCCCCCCO", "CCCCCCCCCCCCCCCC=O"),
        SourcePair(DIRECT_ALDEHYDE, "b", "CCCCCCCCCCCCCCCCCCO", "CCCCCCCCCCCCCCCCCC=O"),
    )
    try:
        build_scope_profiles(pairs, minimum_similarity_floor=0.55)
    except Exception as error:  # narrow assertion on the fail-closed contract
        assert "at least two route pairs" in str(error)
    else:  # pragma: no cover
        raise AssertionError("missing source program unexpectedly passed")
