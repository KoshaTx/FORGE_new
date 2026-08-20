import json
from pathlib import Path

from forge.design.guidance.ugi_current_source_zero_guidance_requalification import (
    build_current_source_zero_guidance_requalification,
)


def test_current_source_and_locked_zero_guidance_values_requalify() -> None:
    repo = Path(__file__).resolve().parents[1]
    result = build_current_source_zero_guidance_requalification(
        repo,
        repo / "configs/model/phase1_ugi_current_source_zero_guidance_requalification_v1.json",
    )
    assert result["status"] == "current_source_l3_and_zero_guidance_route_values_requalified"
    assert result["current_cumulative_source"]["inputs_sha256"] == (
        "0fca91fae36da762eda53695405ec74cf5f9eacd19aee6ca3e4495a6a4d427c6"
    )
    replay = result["zero_guidance_v2_route_requalification"]
    assert replay["locked_unit_count"] == 12
    assert replay["exact_product_value_reproduction_count"] == 12
    assert replay["all_product_values_exactly_reproduced"] is True
    assert all(item["product_value_exactly_reproduced"] for item in replay["replays"])
    assert result["scope"] == {
        "nonzero_guidance_authorized": False,
        "candidate_selection": False,
        "biology_used": False,
        "sealed_holdout_accessed": False,
        "success_probability": None,
    }


def test_committed_current_source_receipt_matches_builder() -> None:
    repo = Path(__file__).resolve().parents[1]
    expected = json.loads(
        (
            repo / "results/phase1/ugi_current_source_zero_guidance_requalification_v1/result.json"
        ).read_text()
    )
    assert expected == build_current_source_zero_guidance_requalification(
        repo,
        repo / "configs/model/phase1_ugi_current_source_zero_guidance_requalification_v1.json",
    )
