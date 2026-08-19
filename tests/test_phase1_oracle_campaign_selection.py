from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from forge.bio.oracle_campaign_selection import (
    OracleCampaignSelectionError,
    run_campaign_selection,
    select_campaign_oracle,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/bio/phase1_oracle_campaign_selection.json"
FREEZE_PATH = REPO / "results/m0_07/oracle_freeze_result.json"
EXPECTED_CANDIDATE = "supervised_graph::ugi_component_role_aware_dmpnn::neural_3seed_ensemble"


def _inputs() -> tuple[dict, dict]:
    return json.loads(CONFIG_PATH.read_text()), json.loads(FREEZE_PATH.read_text())


def test_hela_alone_selects_the_campaign_oracle() -> None:
    config, freeze = _inputs()
    result = select_campaign_oracle(freeze, config)
    selected = result["selected_model"]

    assert selected["candidate_id"] == EXPECTED_CANDIDATE
    assert selected["endpoint"] == "expt_Hela"
    assert selected["campaign_selection_rank"] == 1
    assert selected["calibration_r2"] == pytest.approx(0.5042091281650285)
    assert result["selection_contract"]["excluded_endpoints"] == ["expt_Raw"]
    assert result["selection_contract"]["outer_test_metrics_used_for_selection"] is False
    assert result["guidance_policy"]["authorized"] is False
    assert result["comparison_with_immutable_m0_freeze"] == {
        "m0_shared_candidate_id": EXPECTED_CANDIDATE,
        "campaign_candidate_id": EXPECTED_CANDIDATE,
        "same_candidate": True,
        "m0_freeze_rewritten": False,
    }


def test_raw_metrics_cannot_change_the_hela_ranking() -> None:
    config, freeze = _inputs()
    tampered = copy.deepcopy(freeze)
    for index, candidate in enumerate(tampered["candidate_ranking"]):
        raw = next(row for row in candidate["endpoint_rows"] if row["endpoint"] == "expt_Raw")
        raw["equal_scheme_mean_calibration_r2"] = 10000.0 - index
        raw["equal_scheme_mean_calibration_rmse"] = float(index + 1)
        raw["equal_scheme_mean_calibration_spearman_rho"] = -10000.0 + index

    original = select_campaign_oracle(freeze, config)
    changed = select_campaign_oracle(tampered, config)
    assert [row["candidate_id"] for row in changed["candidate_ranking"]] == [
        row["candidate_id"] for row in original["candidate_ranking"]
    ]


def test_outer_test_metrics_cannot_change_the_hela_ranking() -> None:
    config, freeze = _inputs()
    tampered = copy.deepcopy(freeze)
    for index, candidate in enumerate(tampered["candidate_ranking"]):
        hela = next(row for row in candidate["endpoint_rows"] if row["endpoint"] == "expt_Hela")
        hela["equal_scheme_mean_test_r2"] = 10000.0 - index
        hela["equal_scheme_mean_test_rmse"] = float(index + 1)
        hela["equal_scheme_mean_test_spearman_rho"] = -10000.0 + index
        hela["equal_scheme_mean_absolute_90pct_coverage_gap"] = float(index)

    original = select_campaign_oracle(freeze, config)
    changed = select_campaign_oracle(tampered, config)
    assert [row["candidate_id"] for row in changed["candidate_ranking"]] == [
        row["candidate_id"] for row in original["candidate_ranking"]
    ]


def test_noncampaign_endpoint_must_be_explicitly_excluded() -> None:
    config, freeze = _inputs()
    tampered = copy.deepcopy(config)
    tampered["campaign"]["excluded_endpoints"] = []
    with pytest.raises(
        OracleCampaignSelectionError,
        match="every noncampaign endpoint",
    ):
        select_campaign_oracle(freeze, tampered)


def test_campaign_selection_cannot_authorize_guidance() -> None:
    config, freeze = _inputs()
    tampered = copy.deepcopy(config)
    tampered["guidance"]["authorized"] = True
    with pytest.raises(
        OracleCampaignSelectionError,
        match="cannot silently authorize",
    ):
        select_campaign_oracle(freeze, tampered)


def test_end_to_end_campaign_selection_is_hash_pinned(tmp_path: Path) -> None:
    output = tmp_path / "campaign_selection.json"
    first = run_campaign_selection(CONFIG_PATH, output, REPO)
    first_bytes = output.read_bytes()
    second = run_campaign_selection(CONFIG_PATH, output, REPO)

    assert second == first
    assert output.read_bytes() == first_bytes
    assert first["inputs"]["oracle_freeze_result"]["sha256"] == (
        "8476be6e013e12715bdb55a7ef6d73162b01010f7b560297c6e8406028cda338"
    )
