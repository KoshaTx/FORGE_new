"""Freeze the binary exact-dossier route-completion utility."""

from __future__ import annotations

import gzip
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.data.r1_prime_audit import sha256_file
from forge.value.guidance.ugi_exact_closure_guidance import (
    GuidanceDisposition,
    exact_closure_guidance_policy_manifest,
    exact_closure_potential_from_product_value,
    smc_utility_bridge_from_exact_closure,
)
from forge.value.synthesis.synthesis import ProductSynthesisValue

CONFIG_SCHEMA_VERSION = "phase1_ugi_route_completion_utility_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_route_completion_utility_qualification.v1"


class UgiRouteCompletionUtilityQualificationError(ValueError):
    """Raised when the route-completion utility cannot be frozen safely."""


def _load(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise UgiRouteCompletionUtilityQualificationError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiRouteCompletionUtilityQualificationError(f"{label} must be an object")
    return value


def _load_gzip(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with gzip.open(path, "rt") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise UgiRouteCompletionUtilityQualificationError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiRouteCompletionUtilityQualificationError(f"{label} must be an object")
    return value


def _pin(repo: Path, record: Mapping[str, Any], *, label: str) -> Path:
    if set(record) != {"path", "sha256"}:
        raise UgiRouteCompletionUtilityQualificationError(f"{label} pin is malformed")
    path = (repo / str(record.get("path"))).resolve()
    try:
        relative = path.relative_to(repo.resolve())
    except ValueError as error:
        raise UgiRouteCompletionUtilityQualificationError(f"{label} escapes repository") from error
    lowered = "/".join(relative.parts).lower()
    if "holdout" in lowered or "sealed" in lowered:
        raise UgiRouteCompletionUtilityQualificationError(f"{label} is forbidden")
    if sha256_file(path) != record.get("sha256"):
        raise UgiRouteCompletionUtilityQualificationError(f"{label} hash changed")
    return path


def build_route_completion_utility_qualification(repo: Path, config_path: Path) -> dict[str, Any]:
    """Qualify all fresh-pool v6 product values under the frozen binary policy."""

    repo = repo.resolve()
    config = _load(config_path, label="route-completion utility config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiRouteCompletionUtilityQualificationError(
            "unsupported route-completion utility config"
        )
    policy = config.get("policy")
    required_policy = {
        "utility_name",
        "interpretation",
        "support_bonus",
        "neutral_utility",
        "censored_utility",
        "censored_incremental_log_weight",
        "censor_fraction_ceiling",
        "success_probability",
        "richer_weighting_authorized",
        "selection_authorized",
        "sealed_holdout_access_authorized",
    }
    if not isinstance(policy, dict) or set(policy) != required_policy:
        raise UgiRouteCompletionUtilityQualificationError("utility policy changed")
    if (
        policy.get("utility_name") != "binary_exact_dossier_route_completion_utility"
        or policy.get("interpretation") != "route_completion_utility_not_success_probability"
        or policy.get("support_bonus") != 1.0
        or policy.get("neutral_utility") != 0.0
        or policy.get("censored_utility") is not None
        or policy.get("censored_incremental_log_weight") != 0.0
        or isinstance(policy.get("censor_fraction_ceiling"), bool)
        or not isinstance(policy.get("censor_fraction_ceiling"), (int, float))
        or not 0 <= float(policy["censor_fraction_ceiling"]) <= 1
        or policy.get("success_probability") is not None
        or policy.get("richer_weighting_authorized") is not False
        or policy.get("selection_authorized") is not False
        or policy.get("sealed_holdout_access_authorized") is not False
    ):
        raise UgiRouteCompletionUtilityQualificationError(
            "utility policy violates the conservative binary contract"
        )
    inputs = config.get("inputs")
    required_inputs = {
        "fresh_pool_v6_result",
        "fresh_pool_v6_product_values",
        "guidance_preregistration",
        "utility_source",
        "qualification_source",
        "qualification_runner",
        "qualification_tests",
    }
    if not isinstance(inputs, dict) or set(inputs) != required_inputs:
        raise UgiRouteCompletionUtilityQualificationError("utility inputs changed")
    paths = {
        label: _pin(repo, record, label=label)
        for label, record in inputs.items()
        if isinstance(record, Mapping)
    }
    if set(paths) != required_inputs:
        raise UgiRouteCompletionUtilityQualificationError("utility pin is malformed")

    v6 = _load(paths["fresh_pool_v6_result"], label="fresh-pool v6 result")
    if (
        v6.get("status") != "immutable_v5_values_requalified_under_current_source"
        or v6.get("artifacts", {}).get("product_synthesis_values.json.gz", {}).get("sha256")
        != sha256_file(paths["fresh_pool_v6_product_values"])
        or v6.get("adjudication", {}).get("nonzero_guidance_authorized") is not False
        or v6.get("adjudication", {}).get("sealed_holdout_accessed") is not False
    ):
        raise UgiRouteCompletionUtilityQualificationError("fresh-pool v6 is not qualified")
    prereg = _load(paths["guidance_preregistration"], label="guidance preregistration")
    expected_safeguards = {
        "validity_absolute_drop": 0.02,
        "exact_l1_absolute_drop": 0.02,
        "uniqueness_ratio": 0.9,
        "internal_diversity_ratio": 0.9,
        "broad_coverage_ratio": 0.9,
        "beyond_catalog_ratio": 0.8,
        "role_effective_count_ratio": 0.8,
        "role_max_concentration_absolute_increase": 0.1,
        "oracle_assessed_fraction_floor": 0.99,
    }
    if prereg.get("safeguards") != expected_safeguards:
        raise UgiRouteCompletionUtilityQualificationError(
            "anti-collapse or oracle safeguards changed"
        )

    ledger = _load_gzip(paths["fresh_pool_v6_product_values"], label="v6 product values")
    records = ledger.get("records")
    if not isinstance(records, list) or not records:
        raise UgiRouteCompletionUtilityQualificationError("v6 product-value ledger is empty")
    counts: Counter[str] = Counter()
    positive_without_strict_closure = 0
    censored_with_numeric_utility = 0
    censored_with_positive_bonus = 0
    evidence_upgrades = 0
    for record in records:
        if not isinstance(record, dict):
            raise UgiRouteCompletionUtilityQualificationError(
                "v6 product-value record is malformed"
            )
        value = ProductSynthesisValue.from_dict(record.get("value"))
        potential = exact_closure_potential_from_product_value(value)
        bridge = smc_utility_bridge_from_exact_closure(potential)
        counts[potential.disposition.value] += 1
        if bridge.support_bonus and not potential.strict_route_complete:
            positive_without_strict_closure += 1
        if bridge.censored and bridge.route_completion_utility is not None:
            censored_with_numeric_utility += 1
        if bridge.censored and bridge.support_bonus:
            censored_with_positive_bonus += 1
        if bridge.to_dict().get("evidence_upgraded") is not False:
            evidence_upgrades += 1
    expected = config.get("expected_census")
    observed = {
        "product_count": len(records),
        "support_bonus_count": counts[GuidanceDisposition.SUPPORT_BONUS.value],
        "neutral_count": counts[GuidanceDisposition.NEUTRAL.value],
        "censored_count": counts[GuidanceDisposition.CENSOR.value],
    }
    if observed != expected:
        raise UgiRouteCompletionUtilityQualificationError("route-completion utility census changed")
    censor_fraction = observed["censored_count"] / observed["product_count"]
    if censor_fraction > float(policy["censor_fraction_ceiling"]):
        raise UgiRouteCompletionUtilityQualificationError(
            "preregistered censor-fraction ceiling exceeded"
        )
    if any(
        (
            positive_without_strict_closure,
            censored_with_numeric_utility,
            censored_with_positive_bonus,
            evidence_upgrades,
        )
    ):
        raise UgiRouteCompletionUtilityQualificationError(
            "utility monotonicity or censoring bridge failed"
        )
    support_fraction = observed["support_bonus_count"] / observed["product_count"]
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "binary_exact_dossier_route_completion_utility_qualified",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "policy": policy,
        "implementation_policy": exact_closure_guidance_policy_manifest(),
        "census": {
            **observed,
            "support_bonus_fraction": support_fraction,
            "censor_fraction": censor_fraction,
            "censor_fraction_ceiling": float(policy["censor_fraction_ceiling"]),
        },
        "invariants": {
            "positive_without_strict_exact_l1_l2_l3_closure": 0,
            "censored_with_numeric_route_completion_utility": 0,
            "censored_with_positive_bonus": 0,
            "evidence_upgrades": 0,
            "success_probability_defined": False,
            "anti_collapse_safeguards_retained": expected_safeguards,
        },
        "risk": {
            "sparse_signal": support_fraction < 0.1,
            "interpretation": (
                "The binary bonus is deliberately sparse. Nonzero guidance may collapse "
                "toward the small exact-dossier subset; every run remains subject to the "
                "frozen diversity, novelty, concentration and coverage safeguards."
            ),
            "future_experiment": (
                "Proposal-augmented or graded guidance requires a separate policy, evidence "
                "qualification and preregistered experiment."
            ),
        },
        "adjudication": {
            "development_nonzero_guidance_readiness_may_advance": True,
            "candidate_selection_authorized": False,
            "sealed_holdout_accessed": False,
        },
    }
