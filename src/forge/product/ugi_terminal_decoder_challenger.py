"""Matched diagnostic for Ugi terminal-decoder chemistry collapse.

The audit compares terminal decoders on the same checkpoint, sampled global
programs, productive-flow seed and topology rows.  Only exact-L1-admissible
rows contribute component chemotype summaries; semantic failures remain in
the denominator and are reported separately.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_file
from forge.product.ugi_tail_chemotype_audit import summarize_component_cohort

CONFIG_SCHEMA_VERSION = "phase1_ugi_terminal_decoder_evaluation_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_terminal_decoder_evaluation.v1"
BOND_CONFIG_SCHEMA_VERSION = "phase1_ugi_bond_stochastic_decoder_evaluation_config.v1"
BOND_RESULT_SCHEMA_VERSION = "phase1_ugi_bond_stochastic_decoder_evaluation.v1"
CONFIRMATION_CONFIG_SCHEMA_VERSION = "phase1_ugi_bond_stochastic_confirmation_evaluation_config.v1"
CONFIRMATION_RESULT_SCHEMA_VERSION = "phase1_ugi_bond_stochastic_confirmation_evaluation.v1"
TAIL_ROLES = ("oxoester_aldehyde_body_tail", "isocyanide_tail")


class UgiTerminalDecoderChallengerError(ValueError):
    """Raised when a matched terminal-decoder audit is not interpretable."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise UgiTerminalDecoderChallengerError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise UgiTerminalDecoderChallengerError(f"{label} must be a JSON object")
    return value


def _compact_chemotype_summary(smiles: Sequence[str]) -> dict[str, Any]:
    summary = summarize_component_cohort(smiles)
    fields = (
        "component_occurrences",
        "unique_exact_components",
        "effective_exact_component_count",
        "unique_chemotype_signatures",
        "effective_occurrence_weighted_chemotype_count",
        "unique_architecture_signatures",
        "effective_occurrence_weighted_architecture_count",
        "feature_occurrence_fractions",
    )
    return {field: summary[field] for field in fields}


def summarize_terminal_decoder_arm(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize one frozen draw without silently dropping semantic failures."""

    rows = sample.get("samples")
    statistics = sample.get("statistics")
    if not isinstance(rows, list) or not isinstance(statistics, dict):
        raise UgiTerminalDecoderChallengerError("sample lacks rows or statistics")
    valid_rows = [row for row in rows if row.get("valid") is True]
    expected_valid = int(statistics.get("valid_molecules", -1))
    if len(valid_rows) != expected_valid:
        raise UgiTerminalDecoderChallengerError("valid-row denominator mismatch")
    eligible = [
        row
        for row in valid_rows
        if row.get("component_reconstruction_valid") is True
        and (row.get("l1_forward_verification") or {}).get("exact_product_reconstructed") is True
    ]
    reconstructed_nonexact = sum(
        row.get("component_reconstruction_valid") is True
        and (row.get("l1_forward_verification") or {}).get("exact_product_reconstructed") is False
        for row in valid_rows
    )
    unreconstructed = sum(
        row.get("component_reconstruction_valid") is not True for row in valid_rows
    )
    components: dict[str, list[str]] = {role: [] for role in TAIL_ROLES}
    for row in eligible:
        mapping = row.get("component_smiles_by_role")
        if not isinstance(mapping, dict):
            raise UgiTerminalDecoderChallengerError("eligible row lacks component mapping")
        for role in TAIL_ROLES:
            components[role].append(str(mapping[role]))
    return {
        "samples": len(rows),
        "valid_molecules": len(valid_rows),
        "valid_fraction": len(valid_rows) / len(rows),
        "exact_l1_eligible_molecules": len(eligible),
        "exact_l1_fraction_of_valid": len(eligible) / len(valid_rows),
        "semantic_failures_among_valid": {
            "component_reconstruction_failure": unreconstructed,
            "reconstructed_but_not_exact_l1": reconstructed_nonexact,
        },
        "tail_chemotypes_exact_l1_eligible_only": {
            role: _compact_chemotype_summary(values) for role, values in components.items()
        },
    }


def _assert_matched_rows(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    left_rows = left.get("samples")
    right_rows = right.get("samples")
    if not isinstance(left_rows, list) or not isinstance(right_rows, list):
        raise UgiTerminalDecoderChallengerError("matched samples lack rows")
    if len(left_rows) != len(right_rows):
        raise UgiTerminalDecoderChallengerError("matched arms have different row counts")
    matched_fields = (
        "structure_id",
        "product_id",
        "program",
        "offspring_by_role",
        "source_stratum",
        "branch_class",
    )
    for index, (left_row, right_row) in enumerate(zip(left_rows, right_rows, strict=True)):
        for field in matched_fields:
            if left_row.get(field) != right_row.get(field):
                raise UgiTerminalDecoderChallengerError(
                    f"matched topology row {index} differs at {field}"
                )


def _pair_comparison(
    baseline: Mapping[str, Any],
    challenger: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    role = "oxoester_aldehyde_body_tail"
    left_features = baseline["tail_chemotypes_exact_l1_eligible_only"][role][
        "feature_occurrence_fractions"
    ]
    right_features = challenger["tail_chemotypes_exact_l1_eligible_only"][role][
        "feature_occurrence_fractions"
    ]
    validity_delta = challenger["valid_fraction"] - baseline["valid_fraction"]
    ester_delta = (
        right_features["has_ester_like_carbonyl"] - left_features["has_ester_like_carbonyl"]
    )
    checks = {
        "exact_l1_terminal_admission_fraction": challenger["exact_l1_fraction_of_valid"]
        >= float(gates["exact_l1_terminal_admission_fraction"]),
        "maximum_absolute_validity_degradation": validity_delta
        >= -float(gates["maximum_absolute_validity_degradation"]),
        "aldehyde_carbon_carbon_double_bond_fraction_must_be_nonzero": (
            not gates["aldehyde_carbon_carbon_double_bond_fraction_must_be_nonzero"]
            or right_features["has_carbon_carbon_double_bond"] > 0
        ),
        "aldehyde_ester_absolute_fraction_improvement": ester_delta
        >= float(gates["aldehyde_ester_absolute_fraction_improvement"]),
    }
    return {
        "valid_fraction_delta": validity_delta,
        "exact_l1_fraction_of_valid_delta": challenger["exact_l1_fraction_of_valid"]
        - baseline["exact_l1_fraction_of_valid"],
        "aldehyde_feature_fraction_deltas": {
            feature: right_features[feature] - left_features[feature]
            for feature in sorted(left_features)
        },
        "aldehyde_effective_chemotype_count_delta": challenger[
            "tail_chemotypes_exact_l1_eligible_only"
        ][role]["effective_occurrence_weighted_chemotype_count"]
        - baseline["tail_chemotypes_exact_l1_eligible_only"][role][
            "effective_occurrence_weighted_chemotype_count"
        ],
        "minimum_interpretability_checks": checks,
        "all_minimum_interpretability_conditions_met": all(checks.values()),
    }


def build_terminal_decoder_evaluation(
    repo: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Build the hash-pinned matched decoder evaluation."""

    config = _load_json(config_path, label="terminal decoder evaluation config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiTerminalDecoderChallengerError("unexpected evaluation config schema")
    inputs: dict[str, Any] = {}
    loaded: dict[str, dict[str, Any]] = {}
    for name, specification in config["inputs"].items():
        path = (repo / specification["path"]).resolve()
        observed_hash = sha256_file(path)
        if observed_hash != specification["sha256"]:
            raise UgiTerminalDecoderChallengerError(f"input hash mismatch: {name}")
        inputs[name] = {"path": str(path.relative_to(repo)), "sha256": observed_hash}
        loaded[name] = _load_json(path, label=name)

    contract = loaded.pop("challenger_contract")
    gates = contract["decision_policy"]["minimum_interpretability_conditions"]
    arms: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for pair in config["matched_pairs"]:
        baseline_name = pair["argmax"]
        challenger_name = pair["stochastic"]
        baseline_raw = loaded[baseline_name]
        challenger_raw = loaded[challenger_name]
        _assert_matched_rows(baseline_raw, challenger_raw)
        for name, raw in ((baseline_name, baseline_raw), (challenger_name, challenger_raw)):
            arms.setdefault(name, summarize_terminal_decoder_arm(raw))
        comparisons[pair["name"]] = _pair_comparison(
            arms[baseline_name], arms[challenger_name], gates
        )

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_nonselecting_causal_decoder_diagnostic",
        "inputs": inputs,
        "matched_topology_rows_verified": True,
        "arms": arms,
        "comparisons": comparisons,
        "adjudication": {
            "naive_stochastic_decoder_qualified": all(
                value["all_minimum_interpretability_conditions_met"]
                for value in comparisons.values()
            ),
            "checkpoint_selection_changed": False,
            "prospective_candidate_selection_changed": False,
        },
    }


def _feature_l1_error(
    summary: Mapping[str, Any],
    reference: Mapping[str, float],
) -> float:
    observed = summary["feature_occurrence_fractions"]
    return float(
        sum(abs(float(observed[name]) - float(target)) for name, target in reference.items())
    )


def build_bond_stochastic_evaluation(repo: Path, config_path: Path) -> dict[str, Any]:
    """Evaluate the frozen bond-only stochastic decoder challenger."""

    config = _load_json(config_path, label="bond stochastic evaluation config")
    if config.get("schema_version") != BOND_CONFIG_SCHEMA_VERSION:
        raise UgiTerminalDecoderChallengerError("unexpected bond evaluation config schema")
    inputs: dict[str, Any] = {}
    loaded: dict[str, dict[str, Any]] = {}
    for name, specification in config["inputs"].items():
        path = (repo / specification["path"]).resolve()
        observed_hash = sha256_file(path)
        if observed_hash != specification["sha256"]:
            raise UgiTerminalDecoderChallengerError(f"input hash mismatch: {name}")
        inputs[name] = {"path": str(path.relative_to(repo)), "sha256": observed_hash}
        loaded[name] = _load_json(path, label=name)

    contract = loaded.pop("challenger_contract")
    policy = contract["decision_policy"]
    reference = contract["selection_visible_reference_feature_fractions"]
    comparisons: dict[str, Any] = {}
    arm_summaries: dict[str, Any] = {}
    passing: list[tuple[float, float, int]] = []
    for step in contract["matched_design"]["checkpoints"]:
        label = f"step{int(step)}"
        baseline_spec = contract["inputs"][f"{label}_argmax_control"]
        baseline_path = (repo / baseline_spec["path"]).resolve()
        if sha256_file(baseline_path) != baseline_spec["sha256"]:
            raise UgiTerminalDecoderChallengerError(f"input hash mismatch: {label} control")
        baseline_raw = _load_json(baseline_path, label=f"{label} control")
        challenger_raw = loaded[f"{label}_bond_stochastic"]
        _assert_matched_rows(baseline_raw, challenger_raw)
        baseline = summarize_terminal_decoder_arm(baseline_raw)
        challenger = summarize_terminal_decoder_arm(challenger_raw)
        arm_summaries[f"{label}_argmax"] = baseline
        arm_summaries[f"{label}_bond_stochastic"] = challenger

        feature_errors: dict[str, Any] = {}
        absent_feature_checks: dict[str, bool] = {}
        for role in TAIL_ROLES:
            baseline_role = baseline["tail_chemotypes_exact_l1_eligible_only"][role]
            challenger_role = challenger["tail_chemotypes_exact_l1_eligible_only"][role]
            feature_errors[role] = {
                "argmax": _feature_l1_error(baseline_role, reference[role]),
                "bond_stochastic": _feature_l1_error(challenger_role, reference[role]),
            }
            for feature, target in reference[role].items():
                if float(target) == 0.0:
                    absent_feature_checks[f"{role}:{feature}"] = challenger_role[
                        "feature_occurrence_fractions"
                    ][feature] <= float(
                        policy["maximum_fraction_for_reference_absent_tail_feature"]
                    )
        total_feature_error = sum(feature_errors[role]["bond_stochastic"] for role in TAIL_ROLES)
        exact_matches = int(
            challenger_raw["reference_comparison"]["all_frozen_ugi"]["exact_matches"]
        )
        reproduction_fraction = exact_matches / challenger["valid_molecules"]
        checks = {
            "exact_l1_fraction_of_valid": challenger["exact_l1_fraction_of_valid"]
            >= float(policy["exact_l1_fraction_of_valid"]),
            "maximum_absolute_validity_degradation_from_matched_argmax": challenger[
                "valid_fraction"
            ]
            - baseline["valid_fraction"]
            >= -float(policy["maximum_absolute_validity_degradation_from_matched_argmax"]),
            "maximum_exact_frozen_product_reproduction_fraction": reproduction_fraction
            <= float(policy["maximum_exact_frozen_product_reproduction_fraction"]),
            "maximum_fraction_for_reference_absent_tail_feature": all(
                absent_feature_checks.values()
            ),
            "aldehyde_feature_l1_error_must_improve_over_matched_argmax": feature_errors[
                "oxoester_aldehyde_body_tail"
            ]["bond_stochastic"]
            < feature_errors["oxoester_aldehyde_body_tail"]["argmax"],
        }
        passed = all(checks.values())
        if passed:
            passing.append((total_feature_error, reproduction_fraction, int(step)))
        comparisons[label] = {
            "valid_fraction_delta": challenger["valid_fraction"] - baseline["valid_fraction"],
            "exact_frozen_product_reproduction_fraction": reproduction_fraction,
            "feature_l1_errors": feature_errors,
            "total_declared_feature_l1_error": total_feature_error,
            "reference_absent_feature_checks": absent_feature_checks,
            "checks": checks,
            "all_checks_pass": passed,
        }

    selected = min(passing)[2] if passing else None
    return {
        "schema_version": BOND_RESULT_SCHEMA_VERSION,
        "status": "complete_nonselecting_bond_stochastic_decoder_evaluation",
        "inputs": inputs,
        "matched_topology_rows_verified": True,
        "arms": arm_summaries,
        "comparisons": comparisons,
        "adjudication": {
            "passing_checkpoints": sorted(item[2] for item in passing),
            "confirmation_checkpoint": selected,
            "production_selector_changed": False,
            "prospective_candidate_selection_changed": False,
        },
    }


def build_bond_stochastic_confirmation(repo: Path, config_path: Path) -> dict[str, Any]:
    """Evaluate an independent, preregistered matched decoder confirmation."""

    config = _load_json(config_path, label="bond stochastic confirmation evaluation config")
    if config.get("schema_version") != CONFIRMATION_CONFIG_SCHEMA_VERSION:
        raise UgiTerminalDecoderChallengerError("unexpected confirmation evaluation schema")
    loaded: dict[str, dict[str, Any]] = {}
    inputs: dict[str, Any] = {}
    for name, specification in config["inputs"].items():
        path = (repo / specification["path"]).resolve()
        observed_hash = sha256_file(path)
        if observed_hash != specification["sha256"]:
            raise UgiTerminalDecoderChallengerError(f"input hash mismatch: {name}")
        inputs[name] = {"path": str(path.relative_to(repo)), "sha256": observed_hash}
        loaded[name] = _load_json(path, label=name)

    contract = loaded["confirmation_contract"]
    argmax_raw = loaded["argmax_draw"]
    challenger_raw = loaded["bond_stochastic_draw"]
    _assert_matched_rows(argmax_raw, challenger_raw)
    argmax = summarize_terminal_decoder_arm(argmax_raw)
    challenger = summarize_terminal_decoder_arm(challenger_raw)
    gates = contract["confirmation_gates"]
    reference = contract["selection_visible_reference_feature_fractions"]

    feature_errors: dict[str, Any] = {}
    absent_feature_increases: dict[str, float] = {}
    for role in TAIL_ROLES:
        argmax_role = argmax["tail_chemotypes_exact_l1_eligible_only"][role]
        challenger_role = challenger["tail_chemotypes_exact_l1_eligible_only"][role]
        feature_errors[role] = {
            "argmax": _feature_l1_error(argmax_role, reference[role]),
            "bond_stochastic": _feature_l1_error(challenger_role, reference[role]),
        }
        for feature, target in reference[role].items():
            if float(target) == 0.0:
                absent_feature_increases[f"{role}:{feature}"] = (
                    challenger_role["feature_occurrence_fractions"][feature]
                    - argmax_role["feature_occurrence_fractions"][feature]
                )

    exact_matches = int(challenger_raw["reference_comparison"]["all_frozen_ugi"]["exact_matches"])
    reproduction_fraction = exact_matches / challenger["valid_molecules"]
    unique_fraction = (
        int(challenger_raw["statistics"]["unique_valid_molecules"]) / challenger["valid_molecules"]
    )
    aldehyde_features = challenger["tail_chemotypes_exact_l1_eligible_only"][
        "oxoester_aldehyde_body_tail"
    ]["feature_occurrence_fractions"]
    checks = {
        "minimum_valid_fraction": challenger["valid_fraction"]
        >= float(gates["minimum_valid_fraction"]),
        "minimum_unique_fraction_of_valid": unique_fraction
        >= float(gates["minimum_unique_fraction_of_valid"]),
        "exact_l1_fraction_of_valid": challenger["exact_l1_fraction_of_valid"]
        >= float(gates["exact_l1_fraction_of_valid"]),
        "maximum_exact_frozen_product_reproduction_fraction": reproduction_fraction
        <= float(gates["maximum_exact_frozen_product_reproduction_fraction"]),
        "minimum_aldehyde_carbon_carbon_double_bond_fraction": aldehyde_features[
            "has_carbon_carbon_double_bond"
        ]
        >= float(gates["minimum_aldehyde_carbon_carbon_double_bond_fraction"]),
        "minimum_aldehyde_ester_like_carbonyl_fraction": aldehyde_features[
            "has_ester_like_carbonyl"
        ]
        >= float(gates["minimum_aldehyde_ester_like_carbonyl_fraction"]),
        "aldehyde_feature_l1_error_must_improve_over_matched_argmax": feature_errors[
            "oxoester_aldehyde_body_tail"
        ]["bond_stochastic"]
        < feature_errors["oxoester_aldehyde_body_tail"]["argmax"],
        "maximum_isocyanide_feature_l1_error_increase_over_matched_argmax": feature_errors[
            "isocyanide_tail"
        ]["bond_stochastic"]
        - feature_errors["isocyanide_tail"]["argmax"]
        <= float(gates["maximum_isocyanide_feature_l1_error_increase_over_matched_argmax"]),
        "maximum_reference_absent_feature_increase_over_matched_argmax": max(
            absent_feature_increases.values(), default=0.0
        )
        <= float(gates["maximum_reference_absent_feature_increase_over_matched_argmax"]),
    }
    passed = all(checks.values())
    return {
        "schema_version": CONFIRMATION_RESULT_SCHEMA_VERSION,
        "status": "complete_independent_bond_stochastic_confirmation",
        "inputs": inputs,
        "matched_topology_rows_verified": True,
        "arms": {"argmax": argmax, "bond_stochastic": challenger},
        "comparison": {
            "unique_fraction_of_valid": unique_fraction,
            "exact_frozen_product_reproduction_fraction": reproduction_fraction,
            "feature_l1_errors": feature_errors,
            "reference_absent_feature_increases": absent_feature_increases,
            "checks": checks,
            "all_confirmation_gates_pass": passed,
        },
        "adjudication": {
            "production_sampling_policy_change_qualified": passed,
            "prospective_candidate_selection_changed": False,
            "no_retries_or_seed_search": True,
        },
    }
