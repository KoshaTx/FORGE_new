from __future__ import annotations

import pytest

from forge.design.training.ugi_closure_training import (
    UgiClosureTrainingError,
    _closure_selection_key,
)


def _evaluation(*, components: int, mean_nll: float | None, exact: float | None):
    value = {
        "components": components,
        "mean_nll": mean_nll,
        "exact_set_fraction": exact,
    }
    return {"calibration_novel": value, "calibration_all": value}


def test_closure_selection_prefers_lower_nll_before_exact_recovery() -> None:
    lower_nll = _closure_selection_key(_evaluation(components=22, mean_nll=0.10, exact=0.90))
    higher_recovery = _closure_selection_key(_evaluation(components=22, mean_nll=0.20, exact=1.00))

    assert lower_nll < higher_recovery


def test_closure_selection_uses_exact_recovery_as_tiebreaker() -> None:
    high_recovery = _closure_selection_key(_evaluation(components=22, mean_nll=0.10, exact=0.95))
    low_recovery = _closure_selection_key(_evaluation(components=22, mean_nll=0.10, exact=0.90))

    assert high_recovery < low_recovery


def test_closure_selection_rejects_undefined_calibration() -> None:
    with pytest.raises(UgiClosureTrainingError, match="undefined"):
        _closure_selection_key(_evaluation(components=0, mean_nll=None, exact=None))
