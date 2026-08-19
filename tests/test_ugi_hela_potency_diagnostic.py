from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from forge.potency.ugi_hela_potency_diagnostic import (
    HeLaPotencyDiagnosticPolicy,
    UgiHeLaPotencyDiagnosticError,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/bio/phase1_ugi_hela_potency_diagnostic_authorization_v1.json"


def test_policy_regenerates_only_two_calibration_scales() -> None:
    policy = HeLaPotencyDiagnosticPolicy(REPO, CONFIG)

    assert set(policy.scales) == {"amine_only", "aldehyde_isocyanide_pair"}
    assert policy.scales["amine_only"].values_sha256 == (
        "dc64914c344b0a4c30508616fca9b2f8892c973001db15306559cf48fa1cd955"
    )
    assert policy.scales["aldehyde_isocyanide_pair"].values_sha256 == (
        "fcea2201043fda6d4bb618b0726db4f7d8a691db8413f96cf4b11260e0e2514b"
    )


def test_calibration_median_and_lower_values_are_neutral() -> None:
    policy = HeLaPotencyDiagnosticPolicy(REPO, CONFIG)
    for scale in policy.scales.values():
        median = float(np.median(scale.sorted_lcb90))
        _, cdf, potential = scale.score(median + scale.max_q90)
        assert cdf == 0.5
        assert potential == 0.0
        assert scale.score(scale.sorted_lcb90[0] - 1.0 + scale.max_q90)[2] == 0.0
        assert scale.score(scale.sorted_lcb90[-1] + 1.0 + scale.max_q90)[2] == 1.0


def test_policy_fails_closed_if_nonzero_execution_is_prematurely_enabled(
    tmp_path: Path,
) -> None:
    changed = json.loads(CONFIG.read_text())
    changed["scope"]["nonzero_execution_authorized"] = True
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed))

    with pytest.raises(UgiHeLaPotencyDiagnosticError, match="scope changed"):
        HeLaPotencyDiagnosticPolicy(REPO, changed_path)
