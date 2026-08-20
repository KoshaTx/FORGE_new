from __future__ import annotations

import pytest

from experiments.phase1.hela_potency.morphology.ugi_morphology_proposal_confirmation_analysis import (
    UgiMorphologyProposalConfirmationAnalysisError,
    _cluster_bootstrap_differences,
    _proposal_weighted_summary,
    _summarize,
)


def test_summarize_counts_unique_supported_products() -> None:
    summary = _summarize(
        [
            {"support": True, "valid_exact_l1": True, "smiles": "A"},
            {"support": True, "valid_exact_l1": True, "smiles": "A"},
            {"support": False, "valid_exact_l1": False, "smiles": ""},
        ]
    )
    assert summary["attempts"] == 3
    assert summary["supported"] == 2
    assert summary["unique_supported_smiles"] == 1
    assert summary["support_rate"] == pytest.approx(2 / 3)


def test_empty_summary_fails_closed() -> None:
    with pytest.raises(UgiMorphologyProposalConfirmationAnalysisError):
        _summarize([])


def _row(
    group: str,
    *,
    support: bool,
    top: bool,
    proposal: float,
    prior: float = 0.25,
) -> dict[str, object]:
    return {
        "program_sha256": group,
        "support": support,
        "valid_exact_l1": True,
        "score_quartile": 1 if top else 4,
        "proposal_probability": proposal,
        "prior_probability": prior,
        "importance_ratio_prior_over_proposal": prior / proposal,
    }


def test_proposal_weighted_summary_uses_frozen_population_probabilities() -> None:
    rows = [
        _row("a", support=True, top=True, proposal=0.4),
        _row("b", support=True, top=True, proposal=0.3),
        _row("c", support=False, top=False, proposal=0.2),
        _row("d", support=False, top=False, proposal=0.1),
    ]
    summary = _proposal_weighted_summary(rows)
    assert summary["prior_expected_support_rate"] == pytest.approx(0.5)
    assert summary["proposal_expected_support_rate"] == pytest.approx(0.7)
    assert summary["proposal_support_relative_improvement"] == pytest.approx(0.4)


def test_clustered_bootstrap_is_deterministic_and_keeps_duplicate_programs_together() -> None:
    rows = [
        _row("a", support=True, top=True, proposal=0.3),
        _row("a", support=True, top=True, proposal=0.3),
        _row("b", support=True, top=True, proposal=0.2),
        _row("c", support=False, top=False, proposal=0.1),
        _row("d", support=False, top=False, proposal=0.1),
    ]
    for row in rows:
        row["prior_probability"] = 0.2
        row["importance_ratio_prior_over_proposal"] = 0.2 / float(row["proposal_probability"])
    first = _cluster_bootstrap_differences(rows, replicates=100, seed=17)
    second = _cluster_bootstrap_differences(rows, replicates=100, seed=17)
    assert first == second
    assert first["unit"] == "program_sha256"
    assert first["clusters"] == 4
