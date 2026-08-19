"""Read-only audit of the failed Ugi morphology-potency promotion gate."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_file as _sha256_file
from forge.core.hashing import sha256_json as _sha256_payload

RESULT_SCHEMA_VERSION = "forge.ugi_potency_promotion_failure_audit.v1"


class PotencyPromotionFailureAuditError(ValueError):
    """Raised when a frozen input no longer satisfies the audit contract."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PotencyPromotionFailureAuditError(f"invalid input JSON: {path}") from error
    if not isinstance(value, dict):
        raise PotencyPromotionFailureAuditError(f"input is not a JSON object: {path}")
    return value


def _pattern(result: Mapping[str, Any], arm: str, pattern: str) -> Mapping[str, Any]:
    try:
        value = result["arms"][arm]["patterns"][pattern]
    except (KeyError, TypeError) as error:
        raise PotencyPromotionFailureAuditError(
            f"missing {arm}/{pattern} potency record"
        ) from error
    if not isinstance(value, Mapping):
        raise PotencyPromotionFailureAuditError("potency pattern record is malformed")
    return value


def audit_potency_promotion_failure(
    *,
    confirmatory_path: Path,
    continuous_path: Path,
    signal_path: Path,
) -> dict[str, Any]:
    confirmatory = _load(confirmatory_path)
    continuous = _load(continuous_path)
    signal = _load(signal_path)
    if confirmatory.get("status") != "matched_terminal_potency_adjudication_complete":
        raise PotencyPromotionFailureAuditError("confirmatory result status is not frozen")
    if continuous.get("status") != "continuous_novelty_matched_ranking_complete":
        raise PotencyPromotionFailureAuditError("continuous-support result status is not frozen")
    if signal.get("status") != "morphology_to_observed_hela_signal_gate_complete":
        raise PotencyPromotionFailureAuditError("morphology signal result status is not frozen")

    support_tail = _pattern(continuous, "support_enriched", "aldehyde_isocyanide_pair")
    potency_tail = _pattern(continuous, "nested_potency", "aldehyde_isocyanide_pair")
    support_head = _pattern(continuous, "support_enriched", "amine_only")
    potency_head = _pattern(continuous, "nested_potency", "amine_only")
    ci = continuous["support_vs_nested_potency"]["aldehyde_isocyanide_pair"]
    signal_metrics = signal["selection"]["primary_metrics"]
    signal_ci = signal["selection"]["clustered_bootstrap"]
    confirmatory_arms = confirmatory["arms"]
    support_generation = confirmatory_arms["support_enriched"]
    potency_generation = confirmatory_arms["nested_potency"]

    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "potency_promotion_failure_audited",
        "inputs": {
            "confirmatory": {
                "path": str(confirmatory_path),
                "sha256": _sha256_file(confirmatory_path),
            },
            "continuous_support": {
                "path": str(continuous_path),
                "sha256": _sha256_file(continuous_path),
            },
            "morphology_signal": {
                "path": str(signal_path),
                "sha256": _sha256_file(signal_path),
            },
        },
        "diagnosis": {
            "initial_exact_identity_policy_failure": {
                "broad_eligible_terminals": confirmatory_arms["broad_prior"][
                    "eligible_terminals_before_oracle_budget"
                ],
                "support_eligible_terminals": support_generation[
                    "eligible_terminals_before_oracle_budget"
                ],
                "potency_eligible_terminals": potency_generation[
                    "eligible_terminals_before_oracle_budget"
                ],
                "fixability": "fixed_in_read_only_continuous_support_analysis",
                "interpretation": (
                    "Exact novelty was an overrestrictive applicability proxy; continuous "
                    "product- and role-level support recovered eligible generated chemistry."
                ),
            },
            "measured_morphology_signal": {
                "midrank_spearman": signal_metrics["midrank_spearman"],
                "midrank_spearman_ci95": signal_ci["midrank_spearman_ci95"],
                "top_quartile_observed_gain": signal_metrics["top_quartile_observed_gain"],
                "top_quartile_observed_gain_ci95": signal_ci["top_quartile_observed_gain_ci95"],
                "interpretation": (
                    "Coarse morphology carries reproducible HeLa signal in measured AGILE "
                    "products, but that does not guarantee efficient terminal generation."
                ),
            },
            "authorized_familiar_head_new_tail_lane": {
                "support_eligible": support_tail["eligible_before_budget"],
                "potency_eligible": potency_tail["eligible_before_budget"],
                "support_unique_conservative_high": support_tail[
                    "unique_conservative_high_potency_products"
                ],
                "potency_unique_conservative_high": potency_tail[
                    "unique_conservative_high_potency_products"
                ],
                "unique_high_yield_difference_per_generator_call": ci[
                    "point_difference_unique_high_products_per_generator_call"
                ],
                "difference_ci95": [
                    ci["confidence_interval_low"],
                    ci["confidence_interval_high"],
                ],
                "promotion_gate_passed": ci["confidence_interval_low"] > 0,
                "interpretation": (
                    "The point estimate favors the nested proposal, but the frozen interval "
                    "includes no gain and the eligible-terminal count did not improve."
                ),
            },
            "weakly_authorized_new_head_lane": {
                "support_eligible": support_head["eligible_before_budget"],
                "potency_eligible": potency_head["eligible_before_budget"],
                "support_unique_conservative_high": support_head[
                    "unique_conservative_high_potency_products"
                ],
                "potency_unique_conservative_high": potency_head[
                    "unique_conservative_high_potency_products"
                ],
                "interpretation": (
                    "The largest eligibility increase occurred in exact-new heads, where held-head "
                    "absolute generalization is not strong enough for a primary potency claim."
                ),
            },
            "generator_quality": {
                "support_exact_l1": support_generation["exact_l1_terminals"],
                "potency_exact_l1": potency_generation["exact_l1_terminals"],
                "support_internal_diversity": support_generation["internal_product_diversity"],
                "potency_internal_diversity": potency_generation["internal_product_diversity"],
                "diversity_collapse_explains_failure": False,
            },
        },
        "fixability": {
            "already_fixed": [
                "replace exact-component identity veto with frozen continuous support",
            ],
            "not_fixable_by_threshold_tuning": [
                "weak translation from morphology score to exact terminal chemistry",
                "uncertainty from the small number of role-level biological chemistries",
                "limited held-head generalization",
            ],
            "legitimate_future_gates": [
                "one untouched, frozen larger-budget replication if computational efficiency warrants it",
                "prospective calibration across continuously interpolative novelty strata",
                "new independent labels before expanding novel-head oracle authority",
            ],
            "forbidden_repairs": [
                "loosen the applicability boundary to improve yield",
                "retune the frozen potency proposal on this matched outcome",
                "promote on a point estimate whose interval includes zero",
            ],
        },
        "decision": {
            "current_potency_tilting_promoted": False,
            "production_biological_method": (
                "promoted applicability proposal plus continuous support check plus conservative "
                "terminal HeLa ranking"
            ),
            "future_promotion_remains_possible": True,
            "promotion_requires": (
                "an untouched matched efficiency gain with preserved diversity and biological "
                "authority, preferably followed by prospective confirmation"
            ),
            "paper_blocked_without_potency_tilting": False,
        },
    }
    return {**content, "result_sha256": _sha256_payload(content)}


__all__ = [
    "PotencyPromotionFailureAuditError",
    "RESULT_SCHEMA_VERSION",
    "audit_potency_promotion_failure",
]
