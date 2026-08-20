from __future__ import annotations

from forge.potency.morphology.ugi_morphology_potency_matched_adjudication import EXPECTED_ARMS
from forge.potency.ranking.ugi_continuous_novelty_matched_ranking import _select_equal_budgets


def test_equal_oracle_budget_is_applied_separately_by_novelty_pattern() -> None:
    rows = []
    counts = {
        "amine_only": (2, 3, 4),
        "aldehyde_isocyanide_pair": (5, 4, 6),
    }
    for pattern, per_arm in counts.items():
        for arm, count in zip(EXPECTED_ARMS, per_arm, strict=True):
            for draw in range(count):
                rows.append(
                    {
                        "arm_id": arm,
                        "draw_index": draw,
                        "pattern_id": pattern,
                        "eligible": True,
                        "canonical_product": f"{pattern}-{arm}-{draw}",
                        "oracle_selected": False,
                    }
                )
    budgets, selected = _select_equal_budgets(rows, tuple(counts))
    assert budgets == {"amine_only": 2, "aldehyde_isocyanide_pair": 4}
    assert {arm: len(values) for arm, values in selected.items()} == {
        arm: 6 for arm in EXPECTED_ARMS
    }
    assert all(row["oracle_selected"] for values in selected.values() for row in values)
