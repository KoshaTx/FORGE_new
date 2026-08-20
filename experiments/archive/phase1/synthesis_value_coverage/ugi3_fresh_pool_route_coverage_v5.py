"""Apply one exact, fully closed C16 route to immutable fresh-pool v4.

Version 4 remains immutable.  This audit consumes only the separately
qualified exact four-step route to hexadec-15-ynal and changes one
role-qualified component record.  It independently rebuilds every component,
occurrence, product, provenance, and closure-bin count from the frozen v4
ledgers.  No reaction-family or homologue scope is promoted.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from forge.corpus.r1_prime_audit import sha256_bytes, sha256_file
from experiments.phase1.synthesis_guidance.sources.coverage.ugi3_fresh_pool_route_coverage import (
    _gzip_json_bytes,
    _load_gzip_json,
    _load_json,
)
from forge.synthesis.value.contracts import ComponentSynthesisValue, ProductSynthesisValue

CONFIG_SCHEMA_VERSION = "phase1_ugi3_fresh_pool_route_coverage_config.v5"
RESULT_SCHEMA_VERSION = "phase1_ugi3_fresh_pool_route_coverage.v5"
COMPONENT_LEDGER_SCHEMA_VERSION = "phase1_ugi3_fresh_pool_component_values.v5"
PRODUCT_LEDGER_SCHEMA_VERSION = "phase1_ugi3_fresh_pool_product_values.v5"

TARGET_ROLE = "oxoester_aldehyde_body_tail"
TARGET_SMILES = "C#CCCCCCCCCCCCCCC=O"
TARGET_SOURCE_CLASS = "exact_c16_four_step_route_two_current_terminals"


class Ugi3FreshPoolRouteCoverageV5Error(ValueError):
    """Raised when the isolated C16 coverage refresh violates its contract."""


def _validate_policy(config: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "purpose": (
            "Nonselecting development-pool refresh after exact C16 L2/L3 " "qualification."
        ),
        "base_version": "fresh_pool_route_coverage_v4",
        "delta_scope": (
            "One role-qualified exact identity only: "
            "oxoester_aldehyde_body_tail plus hexadec-15-ynal."
        ),
        "required_exact_route_steps": 4,
        "required_current_terminal_leaves": 2,
        "reaction_family_scope_promoted": False,
        "homologue_scope_promoted": False,
        "unknown_burden_imputed": False,
        "scalarization_authorized": False,
        "success_probability_authorized": False,
        "synthesis_guidance_authorized": False,
        "holdout_reveal_authorized": False,
    }
    observed = config.get("assessment_policy")
    if observed != expected:
        raise Ugi3FreshPoolRouteCoverageV5Error("fresh-pool route v5 assessment policy changed")
    return expected


def _validate_inputs(config: Mapping[str, Any], repo: Path) -> dict[str, Path]:
    declared = config.get("inputs")
    if not isinstance(declared, dict) or not declared:
        raise Ugi3FreshPoolRouteCoverageV5Error("v5 inputs are missing")
    paths: dict[str, Path] = {}
    for label, record in declared.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise Ugi3FreshPoolRouteCoverageV5Error(f"input {label} is malformed")
        path = Path(str(record["path"]))
        if not path.is_absolute():
            path = repo / path
        if sha256_file(path) != record["sha256"]:
            raise Ugi3FreshPoolRouteCoverageV5Error(f"input hash changed: {label}")
        paths[label] = path
    return paths


def _parse_frozen_utc(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise Ugi3FreshPoolRouteCoverageV5Error(f"{label} is not canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise Ugi3FreshPoolRouteCoverageV5Error(f"{label} is not canonical UTC") from error
    if parsed.tzinfo != timezone.utc or parsed.isoformat().replace("+00:00", "Z") != value:
        raise Ugi3FreshPoolRouteCoverageV5Error(f"{label} is not canonical UTC")
    return parsed


def _validate_base_v4(paths: Mapping[str, Path]) -> dict[str, Any]:
    base_result = _load_json(paths["base_result"], label="base route coverage v4")
    base_config = _load_json(paths["base_config"], label="base route coverage v4 config")
    if (
        base_result.get("schema_version") != "phase1_ugi3_fresh_pool_route_coverage.v4"
        or base_result.get("status") != "complete_nonselecting_exact_c18_route_refresh"
        or base_config.get("schema_version") != "phase1_ugi3_fresh_pool_route_coverage_config.v4"
        or base_result.get("config_sha256") != sha256_file(paths["base_config"])
        or base_result.get("policy") != base_config.get("assessment_policy")
    ):
        raise Ugi3FreshPoolRouteCoverageV5Error("immutable base v4 result/config ownership failed")
    artifacts = base_result.get("artifacts")
    if not isinstance(artifacts, dict) or (
        artifacts.get("component_synthesis_values.json.gz", {}).get("sha256")
        != sha256_file(paths["base_component_values"])
        or artifacts.get("product_synthesis_values.json.gz", {}).get("sha256")
        != sha256_file(paths["base_product_values"])
    ):
        raise Ugi3FreshPoolRouteCoverageV5Error("immutable base v4 ledger ownership failed")
    expected_policy = {
        "purpose": (
            "Nonselecting development-pool refresh after exact C18 L2/L3 " "qualification."
        ),
        "delta_scope": (
            "One role-qualified exact identity only: "
            "oxoester_aldehyde_body_tail plus octadec-17-ynal."
        ),
        "reaction_family_scope_promoted": False,
        "unknown_burden_imputed": False,
        "synthesis_guidance_authorized": False,
        "holdout_reveal_authorized": False,
    }
    adjudication = base_result.get("adjudication")
    summary = base_result.get("summary")
    if (
        base_result.get("policy") != expected_policy
        or not isinstance(adjudication, dict)
        or adjudication.get("reaction_family_promoted") is not False
        or adjudication.get("synthesis_guidance_authorized") is not False
        or adjudication.get("holdout_reveal_authorized") is not False
        or adjudication.get("prospective_candidate_selection_changed") is not False
        or not isinstance(summary, dict)
        or summary.get("exact_l1_products") != 3975
        or summary.get("product_outcomes_after", {}).get("complete") != 185
        or summary.get("products_by_route_complete_component_count")
        != {"0": 1252, "1": 1663, "2": 875, "3": 185}
        or summary.get("non_null_scalar_values") != 0
        or summary.get("non_null_success_probabilities") != 0
    ):
        raise Ugi3FreshPoolRouteCoverageV5Error("immutable base v4 policy changed")
    return base_result


def _load_exact_c16_value(paths: Mapping[str, Path]) -> ComponentSynthesisValue:
    result = _load_json(paths["exact_route_result"], label="exact C16 route result")
    exact_config = _load_json(paths["exact_route_config"], label="exact C16 route config")
    artifacts = result.get("artifacts")
    if (
        result.get("schema_version") != "phase1_ugi3_exact_c16_route_audit.v1"
        or result.get("status") != "complete_exact_l2_l3_closed_audit"
        or exact_config.get("schema_version") != "phase1_ugi3_exact_c16_route_config.v1"
        or result.get("config_sha256") != sha256_file(paths["exact_route_config"])
        or not isinstance(artifacts, dict)
        or artifacts.get("assessment.json.gz", {}).get("sha256")
        != sha256_file(paths["exact_route_assessment"])
    ):
        raise Ugi3FreshPoolRouteCoverageV5Error(
            "exact C16 result/config/assessment ownership failed"
        )
    summary = result.get("summary")
    adjudication = result.get("adjudication")
    rejected = result.get("rejected_evidence")
    exact_limitations = {
        "exact_substrate_product_pairs_only": True,
        "reaction_family_promotion_authorized": False,
        "substrate_scope_extrapolation_authorized": False,
        "analogue_promotion_authorized": False,
        "homologue_promotion_authorized": False,
        "general_enumeration_authorized": False,
        "terminal_closure_transfers_to_analogues": False,
        "rejected_source_conflict_can_support_route": False,
    }
    exact_verification_policy = {
        "randomized_smiles_trials": 100,
        "require_unique_forward_product": True,
        "require_exact_carbon_identity": True,
        "require_two_current_terminal_leaves": True,
    }
    if (
        not isinstance(summary, dict)
        or exact_config.get("scope") != "one_exact_c16_route_only"
        or exact_config.get("limitations") != exact_limitations
        or exact_config.get("verification_policy") != exact_verification_policy
        or exact_config.get("expected_summary") != summary
        or exact_config.get("assessment_as_of_utc") != summary.get("assessment_as_of_utc")
        or summary.get("target_role") != TARGET_ROLE
        or summary.get("target_smiles") != TARGET_SMILES
        or summary.get("target_carbon_count") != 16
        or summary.get("exact_steps") != 4
        or summary.get("route_step_count") != 4
        or summary.get("maximum_route_depth") != 4
        or summary.get("route_complete") is not True
        or summary.get("assessment_outcome") != "complete"
        or summary.get("forward_consistency") != "exact_unique"
        or summary.get("evidence_support") != "exact_identity"
        or summary.get("terminal_leaf_count") != 2
        or summary.get("terminal_availability") != "current_closed"
        or summary.get("unassessed_terminal_leaf_count") != 0
        or summary.get("reaction_family_admitted") is not False
        or summary.get("rejected_source_conflict_count") != 1
        or summary.get("rejected_source_conflict_used") is not False
        or not isinstance(adjudication, dict)
        or adjudication.get("independent_exact_l2_route_admitted") is not True
        or adjudication.get("current_l3_procurement_closed") is not True
        or adjudication.get("route_complete") is not True
        or adjudication.get("family_template_admitted") is not False
        or adjudication.get("homologue_template_admitted") is not False
        or adjudication.get("conflicted_source_admitted") is not False
        or adjudication.get("coverage_update_authorized") is not False
        or adjudication.get("synthesis_guidance_authorized") is not False
        or adjudication.get("holdout_reveal_authorized") is not False
        or not isinstance(rejected, list)
        or len(rejected) != 1
        or rejected[0].get("status") != "rejected_source_conflict"
        or rejected[0].get("used_in_route") is not False
    ):
        raise Ugi3FreshPoolRouteCoverageV5Error(
            "exact C16 result does not authorize exact-identity route closure"
        )
    assessment_as_of = _parse_frozen_utc(
        exact_config.get("assessment_as_of_utc"), label="exact C16 assessment timestamp"
    )
    terminals = exact_config.get("terminal_procurement")
    if not isinstance(terminals, list) or len(terminals) != 2:
        raise Ugi3FreshPoolRouteCoverageV5Error(
            "exact C16 frozen terminal freshness policy changed"
        )
    terminal_identities: set[str] = set()
    terminal_expiries: set[str] = set()
    for terminal in terminals:
        if not isinstance(terminal, dict):
            raise Ugi3FreshPoolRouteCoverageV5Error(
                "exact C16 frozen terminal freshness policy changed"
            )
        observed_at = _parse_frozen_utc(
            terminal.get("observed_at_utc"), label="exact C16 terminal observation timestamp"
        )
        expires_at = _parse_frozen_utc(
            terminal.get("expires_at_utc"), label="exact C16 terminal expiry timestamp"
        )
        molecule_id = terminal.get("molecule_id")
        if (
            not isinstance(molecule_id, str)
            or not molecule_id
            or molecule_id in terminal_identities
            or terminal.get("observation_source_input") != "terminal_observations"
            or terminal.get("identity_exact") is not True
            or not observed_at <= assessment_as_of <= expires_at
        ):
            raise Ugi3FreshPoolRouteCoverageV5Error(
                "exact C16 terminal evidence is not current at the frozen assessment"
            )
        terminal_identities.add(molecule_id)
        terminal_expiries.add(str(terminal.get("expires_at_utc")))
    if len(terminal_identities) != 2 or len(terminal_expiries) != 1:
        raise Ugi3FreshPoolRouteCoverageV5Error(
            "exact C16 frozen terminal freshness policy changed"
        )
    payload = _load_gzip_json(paths["exact_route_assessment"], label="exact C16 route assessment")
    if (
        payload.get("schema_version") != "phase1_ugi3_exact_c16_assessment.v1"
        or payload.get("assessment", {}).get("outcome") != "complete"
    ):
        raise Ugi3FreshPoolRouteCoverageV5Error("exact C16 assessment payload changed")
    value = ComponentSynthesisValue.from_dict(payload.get("synthesis_value"))
    if (
        value.target.role != TARGET_ROLE
        or value.target.canonical_smiles != TARGET_SMILES
        or not value.route_complete
        or value.route_step_count != 4
        or value.maximum_route_depth != 4
        or value.leaf_count != 2
        or value.current_terminal_leaf_count != 2
        or value.unassessed_terminal_leaf_count != 0
        or value.current_terminal_leaf_count != value.leaf_count
        or value.forward_consistency.value != "exact_unique"
        or value.evidence_support.value != "exact_identity"
        or value.protection_burden.knowledge.value != "unknown"
        or value.purification_burden.knowledge.value != "unknown"
    ):
        raise Ugi3FreshPoolRouteCoverageV5Error(
            "exact C16 synthesis value is not the qualified complete route"
        )
    return value


def _assert_product_metadata_unchanged(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    before_metadata = {key: value for key, value in before.items() if key != "value"}
    after_metadata = {key: value for key, value in after.items() if key != "value"}
    if before_metadata != after_metadata:
        raise Ugi3FreshPoolRouteCoverageV5Error("product metadata changed")
    before_value = before.get("value")
    after_value = after.get("value")
    if not isinstance(before_value, dict) or not isinstance(after_value, dict):
        raise Ugi3FreshPoolRouteCoverageV5Error("product value is malformed")
    invariant_fields = {
        "schema_version",
        "product_smiles",
        "l1_forward_consistent",
        "scalar_value",
        "success_probability",
    }
    if any(before_value.get(key) != after_value.get(key) for key in invariant_fields):
        raise Ugi3FreshPoolRouteCoverageV5Error("product molecular or scalar metadata changed")
    if (
        after_value.get("scalar_value") is not None
        or after_value.get("success_probability") is not None
    ):
        raise Ugi3FreshPoolRouteCoverageV5Error("pre-prospective product value became scalarized")


def build_fresh_pool_route_coverage_v5(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], bytes, bytes]:
    """Apply one exact C16 component value to immutable fresh-pool v4."""

    config = _load_json(config_path, label="fresh-pool route v5 config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3FreshPoolRouteCoverageV5Error("unsupported fresh-pool route v5 config")
    policy = _validate_policy(config)
    paths = _validate_inputs(config, repo)
    required = {
        "base_component_values",
        "base_product_values",
        "base_result",
        "base_config",
        "exact_route_result",
        "exact_route_assessment",
        "exact_route_config",
        "audit_source",
        "audit_runner",
        "audit_tests",
        "value_source",
    }
    if set(paths) != required:
        raise Ugi3FreshPoolRouteCoverageV5Error("v5 input set changed")
    _validate_base_v4(paths)
    target_value = _load_exact_c16_value(paths)
    target_key = (TARGET_ROLE, TARGET_SMILES)

    base_components = _load_gzip_json(paths["base_component_values"], label="base v4 components")
    if base_components.get("schema_version") != ("phase1_ugi3_fresh_pool_component_values.v4"):
        raise Ugi3FreshPoolRouteCoverageV5Error("base v4 component schema changed")
    records = base_components.get("records")
    if not isinstance(records, list):
        raise Ugi3FreshPoolRouteCoverageV5Error("base component records are missing")
    updated_values: dict[tuple[str, str], ComponentSynthesisValue] = {}
    source_by_key: dict[tuple[str, str], str] = {}
    component_records: list[dict[str, Any]] = []
    changed = 0
    unique_outcomes: Counter[str] = Counter()
    unique_role_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    unique_sources: Counter[str] = Counter()
    for raw in records:
        if not isinstance(raw, dict):
            raise Ugi3FreshPoolRouteCoverageV5Error("base component record is malformed")
        key = (str(raw.get("role")), str(raw.get("canonical_smiles")))
        if key in updated_values:
            raise Ugi3FreshPoolRouteCoverageV5Error("base component identities are not unique")
        value = ComponentSynthesisValue.from_dict(raw.get("value"))
        output = dict(raw)
        if key == target_key:
            if (
                value.route_complete
                or value.assessment_outcome.value != "missing_knowledge"
                or value.route_step_count != 0
            ):
                raise Ugi3FreshPoolRouteCoverageV5Error(
                    "exact C16 target was not an unresolved v4 component"
                )
            identity_metadata = {
                name: raw.get(name)
                for name in (
                    "role",
                    "canonical_smiles",
                    "registry_component_id",
                    "structural_provenance_stratum",
                    "catalog_provenance_substratum",
                )
            }
            value = target_value
            output["source_class"] = TARGET_SOURCE_CLASS
            output["value"] = value.to_dict()
            if identity_metadata != {name: output.get(name) for name in identity_metadata}:
                raise Ugi3FreshPoolRouteCoverageV5Error("C16 component identity metadata changed")
            changed += 1
        updated_values[key] = value
        source_by_key[key] = str(output.get("source_class"))
        component_records.append(output)
        unique_outcomes[value.assessment_outcome.value] += 1
        unique_role_outcomes[key[0]][value.assessment_outcome.value] += 1
        unique_sources[str(output.get("source_class"))] += 1
    if changed != 1:
        raise Ugi3FreshPoolRouteCoverageV5Error("exact C16 component delta did not match once")

    base_products = _load_gzip_json(paths["base_product_values"], label="base v4 products")
    if base_products.get("schema_version") != ("phase1_ugi3_fresh_pool_product_values.v4"):
        raise Ugi3FreshPoolRouteCoverageV5Error("base v4 product schema changed")
    product_rows = base_products.get("records")
    if not isinstance(product_rows, list):
        raise Ugi3FreshPoolRouteCoverageV5Error("base product records are missing")

    product_records: list[dict[str, Any]] = []
    occurrence_outcomes: Counter[str] = Counter()
    role_occurrence_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    source_occurrences: Counter[str] = Counter()
    product_outcomes: Counter[str] = Counter()
    product_outcomes_by_provenance: dict[str, Counter[str]] = defaultdict(Counter)
    complete_component_counts: Counter[int] = Counter()
    transition_counts: Counter[str] = Counter()
    target_prior_bins: Counter[int] = Counter()
    target_occurrences = 0
    newly_complete = 0
    base_complete = 0
    product_value_records_changed = 0
    product_metadata_records_changed = 0
    for raw in product_rows:
        if not isinstance(raw, dict):
            raise Ugi3FreshPoolRouteCoverageV5Error("base product record is malformed")
        prior = ProductSynthesisValue.from_dict(raw.get("value"))
        prior_complete_count = sum(value.route_complete for _, value in prior.components)
        base_complete += int(prior.route_complete)
        role_values: dict[str, ComponentSynthesisValue] = {}
        contains_target = False
        for role, old_value in prior.components:
            key = (role, old_value.target.canonical_smiles)
            value = updated_values.get(key)
            if value is None:
                raise Ugi3FreshPoolRouteCoverageV5Error(
                    "product component is absent from component ledger"
                )
            role_values[role] = value
            if key == target_key:
                contains_target = True
                target_occurrences += 1
                target_prior_bins[prior_complete_count] += 1
            occurrence_outcomes[value.assessment_outcome.value] += 1
            role_occurrence_outcomes[role][value.assessment_outcome.value] += 1
            source_occurrences[source_by_key[key]] += 1
        value = ProductSynthesisValue(
            product_smiles=prior.product_smiles,
            l1_forward_consistent=prior.l1_forward_consistent,
            components=tuple(sorted(role_values.items())),
        )
        after_complete_count = sum(item.route_complete for item in role_values.values())
        transition_counts[f"{prior_complete_count}_to_{after_complete_count}"] += 1
        newly_complete += int(value.route_complete and not prior.route_complete)
        outcome = "complete" if value.route_complete else "noncomplete"
        provenance = str(raw.get("product_structural_provenance_stratum"))
        product_outcomes[outcome] += 1
        product_outcomes_by_provenance[provenance][outcome] += 1
        complete_component_counts[after_complete_count] += 1
        output = dict(raw)
        output["value"] = value.to_dict()
        _assert_product_metadata_unchanged(raw, output)
        product_metadata_records_changed += int(
            {key: item for key, item in raw.items() if key != "value"}
            != {key: item for key, item in output.items() if key != "value"}
        )
        changed_value = output["value"] != raw["value"]
        product_value_records_changed += int(changed_value)
        if changed_value != contains_target:
            raise Ugi3FreshPoolRouteCoverageV5Error(
                "product-value delta escaped the exact C16 target occurrences"
            )
        product_records.append(output)

    component_ledger = _gzip_json_bytes(
        {
            "schema_version": COMPONENT_LEDGER_SCHEMA_VERSION,
            "records": component_records,
        }
    )
    product_ledger = _gzip_json_bytes(
        {
            "schema_version": PRODUCT_LEDGER_SCHEMA_VERSION,
            "records": product_records,
        }
    )
    summary = {
        "base_version": "fresh_pool_route_coverage_v4",
        "exact_route_components_changed": changed,
        "changed_component_keys": [f"{TARGET_ROLE}\t{TARGET_SMILES}"],
        "component_identity_metadata_records_changed": 0,
        "product_value_records_changed": product_value_records_changed,
        "product_metadata_records_changed": product_metadata_records_changed,
        "unique_components": len(component_records),
        "unique_component_outcomes": dict(sorted(unique_outcomes.items())),
        "unique_component_outcomes_by_role": {
            role: dict(sorted(counts.items()))
            for role, counts in sorted(unique_role_outcomes.items())
        },
        "unique_component_value_sources": dict(sorted(unique_sources.items())),
        "component_occurrence_outcomes": dict(sorted(occurrence_outcomes.items())),
        "component_occurrence_outcomes_by_role": {
            role: dict(sorted(counts.items()))
            for role, counts in sorted(role_occurrence_outcomes.items())
        },
        "component_occurrences_by_value_source": dict(sorted(source_occurrences.items())),
        "exact_l1_products": len(product_records),
        "product_outcomes_before": {
            "complete": base_complete,
            "noncomplete": len(product_records) - base_complete,
        },
        "product_outcomes_after": dict(sorted(product_outcomes.items())),
        "newly_complete_products": newly_complete,
        "target_occurrences": {f"{TARGET_ROLE}\t{TARGET_SMILES}": target_occurrences},
        "target_occurrences_by_prior_complete_component_count": {
            str(key): value for key, value in sorted(target_prior_bins.items())
        },
        "products_by_route_complete_component_count": {
            str(key): value for key, value in sorted(complete_component_counts.items())
        },
        "complete_component_count_transitions": dict(sorted(transition_counts.items())),
        "product_outcomes_by_structural_provenance": {
            key: dict(sorted(value.items()))
            for key, value in sorted(product_outcomes_by_provenance.items())
        },
        "non_null_scalar_values": sum(
            record["value"]["scalar_value"] is not None for record in product_records
        ),
        "non_null_success_probabilities": sum(
            record["value"]["success_probability"] is not None for record in product_records
        ),
    }
    if summary != config.get("expected_summary"):
        raise Ugi3FreshPoolRouteCoverageV5Error(
            "fresh-pool route v5 independently rebuilt summary changed"
        )

    return (
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "complete_nonselecting_exact_c16_route_refresh",
            "config_sha256": sha256_file(config_path),
            "summary": summary,
            "inputs": {
                label: {
                    "path": str(path.relative_to(repo)),
                    "sha256": sha256_file(path),
                }
                for label, path in sorted(paths.items())
            },
            "artifacts": {
                "component_synthesis_values.json.gz": {
                    "schema_version": COMPONENT_LEDGER_SCHEMA_VERSION,
                    "sha256": sha256_bytes(component_ledger),
                },
                "product_synthesis_values.json.gz": {
                    "schema_version": PRODUCT_LEDGER_SCHEMA_VERSION,
                    "sha256": sha256_bytes(product_ledger),
                },
            },
            "policy": policy,
            "adjudication": {
                "route_mining_priority_update_authorized": True,
                "synthesis_guidance_authorized": False,
                "prospective_candidate_selection_changed": False,
                "reaction_family_promoted": False,
                "homologue_scope_promoted": False,
                "holdout_reveal_authorized": False,
            },
            "nonclaims": [
                "One exact C16 route does not establish family-wide or homologue-wide substrate scope.",
                "Current precursor procurement does not establish Ugi product success.",
                "Exact graph reconstruction is not experimental synthesis success.",
                "Unknown burden remains unknown rather than zero.",
                "No scalar synthesis value or synthesis-success probability is introduced.",
                "This audit does not authorize synthesis-guided generation or holdout reveal.",
            ],
        },
        component_ledger,
        product_ledger,
    )
