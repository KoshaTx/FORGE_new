"""Tests for the role-decomposed Bayesian surrogate.

The properties that matter here are not accuracy. They are that the posterior is the posterior
we think it is, that the closed-form design quantities agree with the direct computation, and
that the role blocks actually transfer information between designs sharing a precursor. The
acquisition layer reads all three, so an error in any of them would silently choose the wrong
experiments.
"""

from __future__ import annotations

import numpy as np
import pytest

from forge.potency.oracle.reaction_factorized_surrogate import (
    ReactionFactorizedSurrogate,
    ReactionFactorizedSurrogateError,
    RoleWeights,
    featurize,
)

RECORDS = [
    {
        "product": "CCCCNC(=O)C(CCCOC(=O)CCCC)NCCN(C)C",
        "amine": "NCCN(C)C",
        "aldehyde": "CCCC(=O)OCCCC=O",
        "isocyanide": "[C-]#[N+]CCCC",
    },
    {
        "product": "CCCCCNC(=O)C(CCCOC(=O)CCCCC)NCCN(C)C",
        "amine": "NCCN(C)C",
        "aldehyde": "CCCCC(=O)OCCCC=O",
        "isocyanide": "[C-]#[N+]CCCCC",
    },
    {
        "product": "CCCCCCNC(=O)C(CCCOC(=O)CCCCCC)NCCCN(C)C",
        "amine": "NCCCN(C)C",
        "aldehyde": "CCCCCC(=O)OCCCC=O",
        "isocyanide": "[C-]#[N+]CCCCCC",
    },
    {
        "product": "CCCCCCCNC(=O)C(CCCOC(=O)CCCCCCC)NCCN1CCCC1",
        "amine": "NCCN1CCCC1",
        "aldehyde": "CCCCCCC(=O)OCCCC=O",
        "isocyanide": "[C-]#[N+]CCCCCCC",
    },
]
TARGETS = [1.0, 2.0, 3.0, 4.0]


def fitted(**kwargs) -> ReactionFactorizedSurrogate:
    surrogate = ReactionFactorizedSurrogate(n_bits=64, **kwargs)
    return surrogate.fit(RECORDS, TARGETS)


def test_featurize_blocks_are_separate_and_weighted():
    weights = RoleWeights(product=4.0, amine=1.0, aldehyde=0.0, isocyanide=0.0)
    matrix = featurize(RECORDS, weights=weights, n_bits=32)
    assert matrix.shape == (4, 128)
    # a zero-weighted block contributes nothing, so the aldehyde and isocyanide blocks vanish
    assert np.allclose(matrix[:, 64:], 0.0)
    # the product block is scaled by sqrt(4) = 2 relative to an unweighted featurisation
    plain = featurize(RECORDS, weights=RoleWeights(), n_bits=32)
    assert np.allclose(matrix[:, :32], 2.0 * plain[:, :32])


def test_count_features_distinguish_homologues():
    """Binary fingerprints cannot see a CH2; that is why this model uses counts."""
    short = featurize([RECORDS[0]], weights=RoleWeights(), n_bits=256)
    longer = featurize([RECORDS[1]], weights=RoleWeights(), n_bits=256)
    assert not np.allclose(short, longer)


def test_fit_recovers_training_targets_with_low_noise():
    surrogate = ReactionFactorizedSurrogate(n_bits=256, noise_variance=1e-4, prior_variance=10.0)
    surrogate.fit(RECORDS, TARGETS)
    mu, _ = surrogate.predict(RECORDS)
    assert np.allclose(mu, TARGETS, atol=0.15)


def test_predict_is_shifted_by_the_target_mean():
    surrogate = fitted()
    shifted = ReactionFactorizedSurrogate(n_bits=64).fit(RECORDS, [t + 100.0 for t in TARGETS])
    base, _ = surrogate.predict(RECORDS)
    moved, _ = shifted.predict(RECORDS)
    assert np.allclose(moved - base, 100.0, atol=1e-6)


def test_closed_form_variance_reduction_matches_woodbury():
    surrogate = fitted()
    gram = surrogate.target_gram(RECORDS)
    _, covariance = surrogate._require_fit()
    closed = surrogate.variance_reduction_scores(RECORDS, gram, covariance)
    direct = np.array([surrogate.variance_reduction([r], RECORDS) for r in RECORDS])
    assert np.allclose(closed, direct, atol=1e-9)


def test_rank_one_update_matches_batch_update():
    surrogate = fitted()
    _, covariance = surrogate._require_fit()
    stepwise = surrogate.rank_one_update(covariance, RECORDS[0])
    stepwise = surrogate.rank_one_update(stepwise, RECORDS[1])
    joint = surrogate.covariance_after(RECORDS[:2])
    assert np.allclose(stepwise, joint, atol=1e-9)


def test_variance_over_gram_matches_variance_over():
    surrogate = fitted()
    gram = surrogate.target_gram(RECORDS)
    _, covariance = surrogate._require_fit()
    assert surrogate.variance_over_gram(gram, covariance) == pytest.approx(
        surrogate.variance_over(RECORDS), rel=1e-9
    )


def test_observing_a_design_reduces_variance_and_never_increases_it():
    surrogate = fitted()
    before = surrogate.variance_over(RECORDS)
    after_cov = surrogate.covariance_after([RECORDS[0]])
    phi = surrogate.design_matrix(RECORDS)
    after = float(np.einsum("ij,jk,ik->i", phi, after_cov, phi).sum())
    assert after < before
    assert surrogate.variance_reduction([RECORDS[0]], RECORDS) > 0


def test_shared_precursor_transfers_information():
    """The point of the role blocks: measuring one design must inform its relatives.

    RECORDS[0] and RECORDS[1] share an amine head. A design sharing nothing should gain less
    from that observation than one sharing a precursor.
    """
    surrogate = fitted()
    related = surrogate.variance_reduction([RECORDS[0]], [RECORDS[1]])
    unrelated = surrogate.variance_reduction([RECORDS[0]], [RECORDS[3]])
    assert related > unrelated


def test_covariance_is_label_independent():
    """Experimental design has to be solvable before the experiment runs."""
    a = ReactionFactorizedSurrogate(n_bits=64).fit(RECORDS, TARGETS)
    b = ReactionFactorizedSurrogate(n_bits=64).fit(RECORDS, [9.0, -3.0, 0.5, 7.0])
    _, cov_a = a._require_fit()
    _, cov_b = b._require_fit()
    assert np.allclose(cov_a, cov_b)


def test_lower_bound_is_below_the_mean():
    surrogate = fitted()
    mu, _ = surrogate.predict(RECORDS)
    assert np.all(surrogate.lower_bound(RECORDS) < mu)


def test_role_variance_decomposes_over_blocks():
    surrogate = fitted()
    parts = surrogate.role_variance_over(RECORDS)
    assert set(parts) == {"product", "amine", "aldehyde", "isocyanide"}
    assert all(value >= 0 for value in parts.values())


def test_unfitted_surrogate_refuses_to_predict():
    with pytest.raises(ReactionFactorizedSurrogateError, match="not been fitted"):
        ReactionFactorizedSurrogate(n_bits=32).predict(RECORDS)


def test_rejects_mismatched_lengths_and_empty_fits():
    with pytest.raises(ReactionFactorizedSurrogateError, match="against"):
        ReactionFactorizedSurrogate(n_bits=32).fit(RECORDS, TARGETS[:2])
    with pytest.raises(ReactionFactorizedSurrogateError, match="empty"):
        ReactionFactorizedSurrogate(n_bits=32).fit([], [])


def test_rejects_bad_hyperparameters_and_weights():
    with pytest.raises(ReactionFactorizedSurrogateError, match="positive"):
        ReactionFactorizedSurrogate(prior_variance=0.0)
    with pytest.raises(ReactionFactorizedSurrogateError, match="non-negative"):
        RoleWeights(amine=-1.0).validate()
    with pytest.raises(ReactionFactorizedSurrogateError, match="at least one"):
        RoleWeights(0.0, 0.0, 0.0, 0.0).validate()


def test_rejects_unparseable_and_incomplete_records():
    with pytest.raises(ReactionFactorizedSurrogateError, match="unparseable"):
        featurize([{**RECORDS[0], "amine": "not-a-molecule"}], weights=RoleWeights(), n_bits=32)
    with pytest.raises(ReactionFactorizedSurrogateError, match="missing"):
        featurize([{"product": "CCO"}], weights=RoleWeights(), n_bits=32)
