"""Admit one exact, fully closed C18 route into fresh-pool coverage v3.

Version 3 is immutable.  This audit consumes the separately qualified exact
three-step route to octadec-17-ynal and changes only that role-qualified
component identity.  The route is not generalized to a neighboring homologue
or promoted to a reaction-family template.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.data.r1_prime_audit import sha256_bytes, sha256_file
from forge.route.evidence.ugi3_exact_c18_route import (
    Ugi3ExactC18RouteError,
    build_exact_c18_route_audit,
)
from forge.value.coverage.ugi3_fresh_pool_route_coverage import (
    _gzip_json_bytes,
    _load_gzip_json,
    _load_json,
)
from forge.value.synthesis.synthesis import ComponentSynthesisValue, ProductSynthesisValue

CONFIG_SCHEMA_VERSION = "phase1_ugi3_fresh_pool_route_coverage_config.v4"
RESULT_SCHEMA_VERSION = "phase1_ugi3_fresh_pool_route_coverage.v4"
COMPONENT_LEDGER_SCHEMA_VERSION = "phase1_ugi3_fresh_pool_component_values.v4"
PRODUCT_LEDGER_SCHEMA_VERSION = "phase1_ugi3_fresh_pool_product_values.v4"

TARGET_ROLE = "oxoester_aldehyde_body_tail"
TARGET_SMILES = "C#CCCCCCCCCCCCCCCCC=O"


class Ugi3FreshPoolRouteCoverageV4Error(ValueError):
    """Raised when the exact C18 coverage refresh violates its frozen contract."""


def _validate_policy(config: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
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
    observed = config.get("assessment_policy")
    if observed != expected:
        raise Ugi3FreshPoolRouteCoverageV4Error("fresh-pool route v4 assessment policy changed")
    return expected


def _validate_inputs(config: Mapping[str, Any], repo: Path) -> dict[str, Path]:
    declared = config.get("inputs")
    if not isinstance(declared, dict) or not declared:
        raise Ugi3FreshPoolRouteCoverageV4Error("v4 inputs are missing")
    paths: dict[str, Path] = {}
    for label, record in declared.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise Ugi3FreshPoolRouteCoverageV4Error(f"input {label} is malformed")
        path = repo / str(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise Ugi3FreshPoolRouteCoverageV4Error(f"input hash changed: {label}")
        paths[label] = path
    return paths


def _load_exact_c18_value(paths: Mapping[str, Path], *, repo: Path) -> ComponentSynthesisValue:
    exact_config_path = paths["exact_route_config"]
    exact_config = _load_json(exact_config_path, label="exact C18 route config")
    declared_inputs = exact_config.get("inputs")
    if not isinstance(declared_inputs, dict) or not declared_inputs:
        raise Ugi3FreshPoolRouteCoverageV4Error("exact C18 config inputs are missing")
    exact_input_paths: dict[str, Path] = {}
    for label, specification in declared_inputs.items():
        if not isinstance(specification, dict) or set(specification) != {"path", "sha256"}:
            raise Ugi3FreshPoolRouteCoverageV4Error(
                f"exact C18 config input {label!r} is malformed"
            )
        exact_input_paths[label] = repo / str(specification["path"])

    result = _load_json(paths["exact_route_result"], label="exact C18 route result")
    if result.get("config_sha256") != sha256_file(exact_config_path):
        raise Ugi3FreshPoolRouteCoverageV4Error("exact C18 result is not owned by its route config")
    try:
        reproduced_result, _, reproduced_assessment = build_exact_c18_route_audit(
            config_path=exact_config_path,
            input_paths=exact_input_paths,
        )
    except Ugi3ExactC18RouteError as exc:
        raise Ugi3FreshPoolRouteCoverageV4Error(
            f"exact C18 route config failed reproduction: {exc}"
        ) from exc
    if reproduced_result != result:
        raise Ugi3FreshPoolRouteCoverageV4Error(
            "exact C18 result does not reproduce from its owned config"
        )
    if reproduced_assessment != paths["exact_route_assessment"].read_bytes():
        raise Ugi3FreshPoolRouteCoverageV4Error(
            "exact C18 assessment does not reproduce from its owned config"
        )
    artifact = result.get("artifacts", {}).get("assessment.json.gz", {})
    if artifact.get("sha256") != sha256_file(paths["exact_route_assessment"]):
        raise Ugi3FreshPoolRouteCoverageV4Error("exact C18 assessment is not owned by its result")
    summary = result.get("summary")
    adjudication = result.get("adjudication")
    if (
        not isinstance(summary, dict)
        or summary.get("target_role") != TARGET_ROLE
        or summary.get("target_smiles") != TARGET_SMILES
        or summary.get("exact_steps") != 3
        or summary.get("route_complete") is not True
        or summary.get("terminal_availability") != "current_closed"
        or not isinstance(adjudication, dict)
        or adjudication.get("exact_l2_chain_admitted") is not True
        or adjudication.get("current_l3_procurement_closed") is not True
        or adjudication.get("route_complete") is not True
        or adjudication.get("family_template_admitted") is not False
    ):
        raise Ugi3FreshPoolRouteCoverageV4Error(
            "exact C18 result does not authorize exact-identity route closure"
        )
    payload = _load_gzip_json(paths["exact_route_assessment"], label="exact C18 route assessment")
    value = ComponentSynthesisValue.from_dict(payload.get("synthesis_value"))
    if (
        value.target.role != TARGET_ROLE
        or value.target.canonical_smiles != TARGET_SMILES
        or not value.route_complete
        or value.route_step_count != 3
        or value.maximum_route_depth != 3
        or value.current_terminal_leaf_count != value.leaf_count
    ):
        raise Ugi3FreshPoolRouteCoverageV4Error(
            "exact C18 synthesis value is not the qualified complete route"
        )
    return value


def build_fresh_pool_route_coverage_v4(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], bytes, bytes]:
    """Apply the exact complete C18 route to immutable fresh-pool v3."""

    config = _load_json(config_path, label="fresh-pool route v4 config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3FreshPoolRouteCoverageV4Error("unsupported fresh-pool route v4 config")
    policy = _validate_policy(config)
    paths = _validate_inputs(config, repo)
    required = {
        "base_component_values",
        "base_product_values",
        "base_result",
        "exact_route_result",
        "exact_route_assessment",
        "exact_route_config",
        "audit_source",
        "audit_runner",
        "audit_tests",
        "value_source",
    }
    if set(paths) != required:
        raise Ugi3FreshPoolRouteCoverageV4Error("v4 input set changed")

    base_result = _load_json(paths["base_result"], label="base route coverage v3")
    artifacts = base_result.get("artifacts", {})
    if artifacts.get("component_synthesis_values.json.gz", {}).get("sha256") != sha256_file(
        paths["base_component_values"]
    ) or artifacts.get("product_synthesis_values.json.gz", {}).get("sha256") != sha256_file(
        paths["base_product_values"]
    ):
        raise Ugi3FreshPoolRouteCoverageV4Error("base v3 route-coverage ownership failed")

    target_key = (TARGET_ROLE, TARGET_SMILES)
    target_value = _load_exact_c18_value(paths, repo=repo)
    source_class = "exact_c18_three_step_route_current_terminal"

    base_components = _load_gzip_json(paths["base_component_values"], label="base components")
    records = base_components.get("records")
    if not isinstance(records, list):
        raise Ugi3FreshPoolRouteCoverageV4Error("base component records are missing")
    updated_values: dict[tuple[str, str], ComponentSynthesisValue] = {}
    source_by_key: dict[tuple[str, str], str] = {}
    component_records: list[dict[str, Any]] = []
    changed = 0
    unique_outcomes: Counter[str] = Counter()
    unique_role_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    unique_sources: Counter[str] = Counter()
    for raw in records:
        if not isinstance(raw, dict):
            raise Ugi3FreshPoolRouteCoverageV4Error("base component record is malformed")
        key = (str(raw.get("role")), str(raw.get("canonical_smiles")))
        value = ComponentSynthesisValue.from_dict(raw.get("value"))
        output = dict(raw)
        if key == target_key:
            if value.route_complete:
                raise Ugi3FreshPoolRouteCoverageV4Error("exact C18 target was already complete")
            value = target_value
            output["source_class"] = source_class
            output["value"] = value.to_dict()
            changed += 1
        updated_values[key] = value
        source_by_key[key] = str(output.get("source_class"))
        component_records.append(output)
        unique_outcomes[value.assessment_outcome.value] += 1
        unique_role_outcomes[key[0]][value.assessment_outcome.value] += 1
        unique_sources[str(output.get("source_class"))] += 1
    if changed != 1:
        raise Ugi3FreshPoolRouteCoverageV4Error("exact C18 component delta did not match once")

    base_products = _load_gzip_json(paths["base_product_values"], label="base products")
    product_rows = base_products.get("records")
    if not isinstance(product_rows, list):
        raise Ugi3FreshPoolRouteCoverageV4Error("base product records are missing")
    product_records: list[dict[str, Any]] = []
    occurrence_outcomes: Counter[str] = Counter()
    role_occurrence_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    source_occurrences: Counter[str] = Counter()
    product_outcomes: Counter[str] = Counter()
    product_outcomes_by_provenance: dict[str, Counter[str]] = defaultdict(Counter)
    complete_component_counts: Counter[int] = Counter()
    transition_counts: Counter[str] = Counter()
    target_occurrences = 0
    newly_complete = 0
    base_complete = 0
    for raw in product_rows:
        if not isinstance(raw, dict):
            raise Ugi3FreshPoolRouteCoverageV4Error("base product record is malformed")
        prior = ProductSynthesisValue.from_dict(raw.get("value"))
        prior_complete_count = sum(value.route_complete for _, value in prior.components)
        base_complete += int(prior.route_complete)
        role_values: dict[str, ComponentSynthesisValue] = {}
        for role, old_value in prior.components:
            key = (role, old_value.target.canonical_smiles)
            value = updated_values.get(key)
            if value is None:
                raise Ugi3FreshPoolRouteCoverageV4Error(
                    "product component is absent from component ledger"
                )
            role_values[role] = value
            if key == target_key:
                target_occurrences += 1
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
        product_records.append(output)

    component_ledger = _gzip_json_bytes(
        {"schema_version": COMPONENT_LEDGER_SCHEMA_VERSION, "records": component_records}
    )
    product_ledger = _gzip_json_bytes(
        {"schema_version": PRODUCT_LEDGER_SCHEMA_VERSION, "records": product_records}
    )
    summary = {
        "base_version": "fresh_pool_route_coverage_v3",
        "exact_route_components_changed": changed,
        "changed_component_keys": [f"{TARGET_ROLE}\t{TARGET_SMILES}"],
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
        raise Ugi3FreshPoolRouteCoverageV4Error("fresh-pool route v4 summary changed")

    return (
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "complete_nonselecting_exact_c18_route_refresh",
            "config_sha256": sha256_file(config_path),
            "summary": summary,
            "inputs": {
                label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
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
                "holdout_reveal_authorized": False,
            },
            "nonclaims": [
                "One exact C18 route does not establish family-wide substrate scope.",
                "Current precursor procurement does not establish Ugi product success.",
                "Exact graph reconstruction is not experimental synthesis success.",
                "Unknown burden remains unknown rather than zero.",
                "This audit does not authorize synthesis-guided generation or holdout reveal.",
            ],
        },
        component_ledger,
        product_ledger,
    )
