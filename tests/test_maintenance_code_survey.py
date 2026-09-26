"""The single-contract survey must not label unreferenced code safe to delete."""

from __future__ import annotations

from forge_maintenance.code_survey import _classify_reachability


def test_unreached_code_requires_manual_study_review() -> None:
    assert (
        _classify_reachability(in_cli=False, in_paper=False, historical_pins_archived=True)
        == "review_unreached"
    )
    assert (
        _classify_reachability(in_cli=False, in_paper=False, historical_pins_archived=False)
        == "blocked_historical_pin"
    )
