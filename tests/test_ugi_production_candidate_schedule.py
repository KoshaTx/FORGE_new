from __future__ import annotations

from forge.product.ugi_production_candidate_schedule import ARM_IDS


def test_production_schedule_contains_only_promoted_causal_arms() -> None:
    assert ARM_IDS == ("broad_prior", "support_enriched")


def test_production_schedule_excludes_rejected_tilts() -> None:
    assert "nested_potency" not in ARM_IDS
    assert "synthesis_guided" not in ARM_IDS
