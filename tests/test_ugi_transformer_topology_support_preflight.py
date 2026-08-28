from __future__ import annotations

import json
from pathlib import Path

from forge.core.hashing import sha256_file

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/model/phase1_ugi_transformer_topology_support_preflight_v2.json"
RESULT = REPO / "results/phase1/ugi_transformer_topology_support_preflight_v2/result.json"
READINESS = REPO / "results/phase1/ugi_transformer_topology_correction_v2/result.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_all_frozen_programs_have_exact_decoder_support_without_selection() -> None:
    config = _read(CONFIG)
    result = _read(RESULT)

    assert result["status"] == "pass"
    assert result["programs"] == config["expected_programs"] == 3072
    assert result["observed"]["decoded"] == 3072
    assert result["observed"]["exact_program_morphology"] == 3072
    assert result["observed"]["failures"] == 0
    assert result["observed"]["mismatches"] == 0
    assert result["observed"]["by_total_cycle_rank"] == {
        "0": {"exact": 1040},
        "1": {"exact": 1845},
        "2": {"exact": 186},
        "3": {"exact": 1},
    }
    assert all(result["gates"].values())
    assert result["config"]["sha256"] == sha256_file(CONFIG)
    for label, pin in config["inputs"].items():
        assert result["inputs"][label]["sha256"] == pin["sha256"]
        assert sha256_file(REPO / pin["path"]) == pin["sha256"]


def test_readiness_receipt_does_not_claim_the_production_gap_is_closed() -> None:
    readiness = _read(READINESS)

    assert readiness["status"] == "implementation_qualified_locally_h100_pending"
    assert readiness["topology_support_preflight"]["sha256"] == sha256_file(RESULT)
    assert readiness["gates"]["h100_preflight_passed"] is None
    assert readiness["gates"]["production_exact_l1_gap_closed"] is None
    assert readiness["gates"]["heldout_used_for_training_or_policy_selection"] is False
