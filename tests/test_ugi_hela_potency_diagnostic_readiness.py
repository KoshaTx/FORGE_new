from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.design.flow.ugi_hela_potency_diagnostic_readiness import (
    UgiHeLaPotencyReadinessError,
    audit_historical_checkpoint_coverage,
    build_hela_potency_diagnostic_readiness,
)
from forge.potency.audit.ugi_hela_potency_diagnostic import HeLaPotencyDiagnosticPolicy

REPO = Path(__file__).resolve().parents[1]
BIO_CONFIG = REPO / "configs/bio/phase1_ugi_hela_potency_diagnostic_authorization_v1.json"
READINESS_CONFIG = REPO / "configs/model/phase1_ugi_hela_potency_guidance_readiness_v1.json"
HISTORICAL_RUN = REPO / "results/phase1/ugi_production_zero_guidance_seam_v3/run.json"
HISTORICAL_SUPPORT = (
    REPO / "results/phase1/ugi_production_zero_guidance_seam_v3/support_audits.json"
)


def test_historical_checkpoint_coverage_is_reproduced_exactly() -> None:
    policy = HeLaPotencyDiagnosticPolicy(REPO, BIO_CONFIG)
    coverage = audit_historical_checkpoint_coverage(
        json.loads(HISTORICAL_RUN.read_text()),
        json.loads(HISTORICAL_SUPPORT.read_text()),
        policy,
    )

    assert coverage["scheduled_checkpoint_attempts"] == 192
    assert coverage["classified_exact_l1_attempts"] == 182
    assert coverage["invalid_terminal_attempts"] == 10
    assert coverage["overall_bins"] == {
        "boundary": 6,
        "extrapolative": 173,
        "interpolative": 3,
    }
    assert coverage["interpolative_unseen_role_patterns"] == {"amine+isocyanide": 3}
    assert coverage["active_eligible_attempts"] == 0


def test_readiness_fails_before_runtime_if_execution_is_enabled(tmp_path: Path) -> None:
    changed = json.loads(READINESS_CONFIG.read_text())
    changed["scope"]["nonzero_execution_authorized"] = True
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed))

    with pytest.raises(UgiHeLaPotencyReadinessError, match="contract changed"):
        build_hela_potency_diagnostic_readiness(REPO, path)
