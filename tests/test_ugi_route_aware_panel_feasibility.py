from __future__ import annotations

from experiments.phase1.synthesis_guidance.feasibility import summarize_route_records


def _record(
    arm: str,
    product: str,
    *,
    route_complete: bool,
    selected: bool,
    high: bool,
) -> dict:
    roles = []
    components = {}
    for role in (
        "amine_head",
        "oxoester_aldehyde_body_tail",
        "isocyanide_tail",
    ):
        component = f"{arm}-{role}-{product}"
        components[role] = component
        roles.append(
            {
                "role": role,
                "strict_complete": route_complete,
                "assessment_outcome": "complete" if route_complete else "missing_knowledge",
                "reason": (
                    "strict_exact_route_complete" if route_complete else "missing_route_knowledge"
                ),
            }
        )
    return {
        "arm_id": arm,
        "canonical_product": product,
        "canonical_components": components,
        "pattern_id": "aldehyde_isocyanide_pair",
        "route_complete": route_complete,
        "oracle_selected": selected,
        "conservative_high_potency": high,
        "potential": {"roles": roles},
    }


def test_summary_separates_production_and_matched_cohorts() -> None:
    rows = [
        _record("broad_prior", "b1", route_complete=True, selected=True, high=True),
        _record("broad_prior", "b2", route_complete=False, selected=True, high=False),
        _record("support_enriched", "s1", route_complete=True, selected=True, high=True),
        _record("support_enriched", "s2", route_complete=True, selected=False, high=False),
        _record("support_enriched", "s2", route_complete=True, selected=False, high=False),
    ]
    result = summarize_route_records(rows)
    assert result["arms"]["support_enriched"]["eligible_rows"] == 3
    assert result["arms"]["support_enriched"]["eligible_unique_products"] == 2
    assert result["arms"]["support_enriched"]["oracle_selected_rows"] == 1
    assert result["balanced_fillable_route_complete_unique_products"] == {
        "production_yield_cohort": 1,
        "causal_matched_oracle_cohort": 1,
        "conservative_high_cohort_rows": 1,
    }


def test_summary_rejects_inconsistent_duplicate_product_outcomes() -> None:
    rows = [
        _record("broad_prior", "same", route_complete=True, selected=True, high=True),
        _record("broad_prior", "same", route_complete=False, selected=True, high=True),
        _record("support_enriched", "other", route_complete=True, selected=True, high=True),
    ]
    try:
        summarize_route_records(rows)
    except RuntimeError as error:
        assert "inconsistent route outcomes" in str(error)
    else:  # pragma: no cover
        raise AssertionError("inconsistent duplicate products must fail closed")
