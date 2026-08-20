from experiments.archive.phase1.potency_audits.ugi_fresh_pool_oracle_audit import (
    _quantiles,
    summarize_oracle_rows,
)


def _row(
    *,
    domain: str,
    mean: float,
    deviation: float,
    complete_count: int,
    route_complete: bool = False,
) -> dict:
    return {
        "oracle_domain": domain,
        "ensemble_mean": mean,
        "ensemble_standard_deviation": deviation,
        "route_complete_component_count": complete_count,
        "route_complete": route_complete,
        "product_structural_provenance_stratum": "catalog_absent",
        "combination_seen_in_measured_training": False,
        "exact_forward_verified": True,
        "guidance_action": "abstain",
        "guidance_score": None,
    }


def test_quantiles_use_linear_interpolation() -> None:
    result = _quantiles([1.0, 2.0, 3.0, 4.0])
    assert result["median"] == 2.5
    assert result["q25"] == 1.75
    assert result["q75"] == 3.25


def test_summary_keeps_route_and_domain_intersection_nonselecting() -> None:
    result = summarize_oracle_rows(
        [
            _row(
                domain="known_components_novel_combination",
                mean=4.0,
                deviation=0.2,
                complete_count=3,
                route_complete=True,
            ),
            _row(
                domain="unseen_aldehyde",
                mean=6.0,
                deviation=0.4,
                complete_count=2,
            ),
        ]
    )
    assert result["records"] == 2
    assert result["route_complete_products_by_oracle_domain"] == {
        "known_components_novel_combination": 1
    }
    assert result["one_gap_products_by_oracle_domain"] == {"unseen_aldehyde": 1}
    assert result["ensemble_mean_distribution_descriptive_only"]["median"] == 5.0
    assert result["all_guidance_actions_abstain"]


def test_summary_accepts_csv_empty_string_as_null_guidance_score() -> None:
    row = _row(
        domain="unseen_head",
        mean=3.0,
        deviation=0.3,
        complete_count=1,
    )
    row["guidance_score"] = ""
    assert summarize_oracle_rows([row])["all_guidance_actions_abstain"]
