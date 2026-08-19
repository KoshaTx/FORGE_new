"""Transfer frozen exact route values onto the fresh v2 Ugi product pool.

This audit is deliberately conservative.  It reuses only exact role plus
constitutional-component identities from the authenticated 424-component
hybrid assessment ledger and its frozen v3 exact-evidence overlays.  A fresh
component with no exact identity match is missing route knowledge, not
chemically incompatible.  No scalar route score or success probability is
created.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.bio.ugi_semantic_annotations import ROLE_NAMES
from forge.data.r1_prime_audit import sha256_bytes, sha256_file
from forge.product.ugi_postselection_provenance import (
    CATALOG_ABSENT,
    COMPONENT_STRATA,
    classify_component_provenance,
    classify_product_provenance,
)
from forge.route.planner import RouteTarget, SynthesisAssessment
from forge.value.synthesis import (
    ComponentSynthesisValue,
    ProductSynthesisValue,
    component_synthesis_value_from_assessment,
)
from forge.value.ugi3_synthesis_value_audit import _missing_generated_assessment

CONFIG_SCHEMA_VERSION = "phase1_ugi3_fresh_pool_route_coverage_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi3_fresh_pool_route_coverage.v1"
COMPONENT_LEDGER_SCHEMA_VERSION = "phase1_ugi3_fresh_pool_component_values.v1"
PRODUCT_LEDGER_SCHEMA_VERSION = "phase1_ugi3_fresh_pool_product_values.v1"


class Ugi3FreshPoolRouteCoverageError(ValueError):
    """Raised when a fresh-pool route audit cannot be reproduced exactly."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise Ugi3FreshPoolRouteCoverageError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Ugi3FreshPoolRouteCoverageError(f"{label} must be an object")
    return value


def _load_gzip_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with gzip.open(path, "rt") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise Ugi3FreshPoolRouteCoverageError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Ugi3FreshPoolRouteCoverageError(f"{label} must be an object")
    return value


def _read_csv(path: Path, *, label: str) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise Ugi3FreshPoolRouteCoverageError(f"{label} has no header")
            return list(reader)
    except (OSError, csv.Error) as exc:
        raise Ugi3FreshPoolRouteCoverageError(f"could not read {label}") from exc


def _gzip_json_bytes(value: Any) -> bytes:
    output = io.BytesIO()
    payload = (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        compressed.write(payload)
    return output.getvalue()


def _canonical_constitution(smiles: Any, *, label: str) -> str:
    if not isinstance(smiles, str) or not smiles:
        raise Ugi3FreshPoolRouteCoverageError(f"{label} must be non-empty SMILES")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise Ugi3FreshPoolRouteCoverageError(f"{label} is not one valid molecular graph")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _validate_inputs(config: Mapping[str, Any], repo: Path) -> dict[str, Path]:
    specifications = config.get("inputs")
    if not isinstance(specifications, dict) or not specifications:
        raise Ugi3FreshPoolRouteCoverageError("route audit inputs are missing")
    paths: dict[str, Path] = {}
    for label, specification in specifications.items():
        if not isinstance(specification, dict) or set(specification) != {"path", "sha256"}:
            raise Ugi3FreshPoolRouteCoverageError(f"input {label} must define path and sha256")
        path = (repo / specification["path"]).resolve()
        observed = sha256_file(path)
        if observed != specification["sha256"]:
            raise Ugi3FreshPoolRouteCoverageError(f"input hash changed: {label}")
        paths[label] = path
    return paths


def _exact_l1_rows(sample: Mapping[str, Any]) -> list[tuple[int, Mapping[str, Any]]]:
    rows = sample.get("samples")
    if not isinstance(rows, list):
        raise Ugi3FreshPoolRouteCoverageError("fresh sample lacks rows")
    output = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise Ugi3FreshPoolRouteCoverageError("fresh sample row is malformed")
        if (
            row.get("valid") is True
            and row.get("component_reconstruction_valid") is True
            and (row.get("l1_forward_verification") or {}).get("exact_product_reconstructed")
            is True
        ):
            components = row.get("component_smiles_by_role")
            if not isinstance(components, dict) or set(components) != set(ROLE_NAMES):
                raise Ugi3FreshPoolRouteCoverageError(
                    f"exact-L1 sample {index} lacks three component roles"
                )
            output.append((index, row))
    return output


def _hybrid_values(payload: Mapping[str, Any]) -> dict[tuple[str, str], ComponentSynthesisValue]:
    records = payload.get("records")
    if not isinstance(records, list):
        raise Ugi3FreshPoolRouteCoverageError("hybrid assessment ledger lacks records")
    output = {}
    for record in records:
        if not isinstance(record, dict):
            raise Ugi3FreshPoolRouteCoverageError("hybrid assessment record is malformed")
        assessment = SynthesisAssessment.from_dict(record.get("assessment"))
        key = (assessment.target.role, assessment.target.canonical_smiles)
        if key in output:
            raise Ugi3FreshPoolRouteCoverageError("duplicate hybrid assessment identity")
        output[key] = component_synthesis_value_from_assessment(assessment)
    return output


def _v3_values(payload: Mapping[str, Any]) -> dict[tuple[str, str], ComponentSynthesisValue]:
    records = payload.get("records")
    if not isinstance(records, list):
        raise Ugi3FreshPoolRouteCoverageError("v3 component ledger lacks records")
    output = {}
    for record in records:
        if not isinstance(record, dict):
            raise Ugi3FreshPoolRouteCoverageError("v3 component record is malformed")
        value = ComponentSynthesisValue.from_dict(record.get("value"))
        key = (str(record.get("role")), str(record.get("canonical_smiles")))
        if key != (value.target.role, value.target.canonical_smiles) or key in output:
            raise Ugi3FreshPoolRouteCoverageError("v3 component identity is inconsistent")
        output[key] = value
    return output


def _registry_index(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    output = {}
    for row in rows:
        if row.get("l1_structural_admission") != "true":
            continue
        role = row.get("role", "")
        if role not in ROLE_NAMES:
            raise Ugi3FreshPoolRouteCoverageError(f"unexpected registry role {role!r}")
        key = (
            role,
            _canonical_constitution(row.get("canonical_smiles"), label="registry component"),
        )
        if key in output:
            raise Ugi3FreshPoolRouteCoverageError("duplicate registry component identity")
        output[key] = dict(row)
    return output


def build_fresh_pool_route_coverage(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], bytes, bytes]:
    """Build a conservative exact-identity route audit for the fresh v2 pool."""

    config = _load_json(config_path, label="fresh-pool route config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3FreshPoolRouteCoverageError("unsupported fresh-pool route config")
    paths = _validate_inputs(config, repo)
    required = {
        "production_generator_v2",
        "fresh_pool_sample",
        "fresh_pool_audit",
        "component_registry",
        "hybrid_assessment_ledger",
        "hybrid_result",
        "synthesis_value_v3_components",
        "synthesis_value_v3_result",
        "value_source",
        "audit_source",
        "audit_runner",
        "audit_tests",
    }
    if set(paths) != required:
        raise Ugi3FreshPoolRouteCoverageError("fresh-pool route input set changed")

    manifest = _load_json(paths["production_generator_v2"], label="production v2 manifest")
    sample = _load_json(paths["fresh_pool_sample"], label="fresh pool sample")
    fresh_audit = _load_json(paths["fresh_pool_audit"], label="fresh pool audit")
    hybrid_result = _load_json(paths["hybrid_result"], label="hybrid result")
    v3_result = _load_json(paths["synthesis_value_v3_result"], label="v3 value result")
    if manifest.get("status") != "frozen_after_independent_decoder_confirmation":
        raise Ugi3FreshPoolRouteCoverageError("production v2 is not frozen")
    audit_inputs = fresh_audit.get("inputs", {})
    if (
        audit_inputs.get("fresh_pool_sample", {}).get("sha256")
        != sha256_file(paths["fresh_pool_sample"])
        or audit_inputs.get("production_generator_v2", {}).get("sha256")
        != sha256_file(paths["production_generator_v2"])
        or fresh_audit.get("adjudication", {}).get(
            "pool_may_advance_to_nonselecting_oracle_and_route_assessment"
        )
        is not True
    ):
        raise Ugi3FreshPoolRouteCoverageError("fresh-pool route assessment is not authorized")
    if hybrid_result.get("artifacts", {}).get("assessment_ledger_sha256") != sha256_file(
        paths["hybrid_assessment_ledger"]
    ) or v3_result.get("artifacts", {}).get("component_synthesis_values.json.gz", {}).get(
        "sha256"
    ) != sha256_file(
        paths["synthesis_value_v3_components"]
    ):
        raise Ugi3FreshPoolRouteCoverageError("frozen route-value ownership failed")

    hybrid = _load_gzip_json(paths["hybrid_assessment_ledger"], label="hybrid ledger")
    v3 = _load_gzip_json(paths["synthesis_value_v3_components"], label="v3 values")
    hybrid_values = _hybrid_values(hybrid)
    v3_values = _v3_values(v3)
    for key in set(hybrid_values) & set(v3_values):
        if hybrid_values[key].route_complete and not v3_values[key].route_complete:
            raise Ugi3FreshPoolRouteCoverageError("v3 exact overlay regresses route closure")
    known_values = {**hybrid_values, **v3_values}
    value_source = {
        **{key: "hybrid_registry_assessment" for key in hybrid_values},
        **{key: "v3_exact_overlay_or_replay" for key in v3_values},
    }
    registry = _registry_index(_read_csv(paths["component_registry"], label="component registry"))
    exact_rows = _exact_l1_rows(sample)
    expected_exact = int(fresh_audit.get("summary", {}).get("exact_l1_products", -1))
    if len(exact_rows) != expected_exact:
        raise Ugi3FreshPoolRouteCoverageError("fresh exact-L1 denominator changed")

    unique_values: dict[tuple[str, str], ComponentSynthesisValue] = {}
    component_metadata: dict[tuple[str, str], dict[str, Any]] = {}
    components_by_sample: dict[int, dict[str, ComponentSynthesisValue]] = {}
    component_occurrence_outcomes: Counter[str] = Counter()
    source_occurrences: Counter[str] = Counter()
    role_occurrence_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    component_provenance_occurrences: Counter[str] = Counter()

    for sample_index, row in exact_rows:
        role_values = {}
        for role in ROLE_NAMES:
            smiles = _canonical_constitution(
                row["component_smiles_by_role"][role],
                label=f"sample {sample_index} {role}",
            )
            key = (role, smiles)
            value = unique_values.get(key)
            if value is None:
                value = known_values.get(key)
                source_class = value_source.get(key)
                if value is None:
                    source_class = "fresh_component_missing_route_knowledge"
                    value = component_synthesis_value_from_assessment(
                        _missing_generated_assessment(
                            target=RouteTarget(role=role, canonical_smiles=smiles),
                            provenance_ledger_path=paths["fresh_pool_sample"],
                        )
                    )
                registry_row = registry.get(key)
                stratum, substratum = classify_component_provenance(
                    registry_row,
                    transfer_source_class="cross_platform_hydrophobic_transfer",
                )
                unique_values[key] = value
                component_metadata[key] = {
                    "source_class": source_class,
                    "structural_provenance_stratum": stratum,
                    "catalog_provenance_substratum": substratum,
                    "registry_component_id": (
                        registry_row.get("component_id") if registry_row is not None else None
                    ),
                }
            metadata = component_metadata[key]
            outcome = value.assessment_outcome.value
            component_occurrence_outcomes[outcome] += 1
            source_occurrences[metadata["source_class"]] += 1
            role_occurrence_outcomes[role][outcome] += 1
            component_provenance_occurrences[metadata["structural_provenance_stratum"]] += 1
            role_values[role] = value
        components_by_sample[sample_index] = role_values

    component_records = []
    unique_outcomes: Counter[str] = Counter()
    unique_sources: Counter[str] = Counter()
    unique_role_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    for key in sorted(unique_values):
        value = unique_values[key]
        metadata = component_metadata[key]
        outcome = value.assessment_outcome.value
        unique_outcomes[outcome] += 1
        unique_sources[metadata["source_class"]] += 1
        unique_role_outcomes[key[0]][outcome] += 1
        component_records.append(
            {
                "role": key[0],
                "canonical_smiles": key[1],
                **metadata,
                "value": value.to_dict(),
            }
        )

    product_records = []
    product_outcomes: Counter[str] = Counter()
    product_outcomes_by_provenance: dict[str, Counter[str]] = defaultdict(Counter)
    complete_component_counts: Counter[int] = Counter()
    for sample_index, row in exact_rows:
        role_values = components_by_sample[sample_index]
        component_strata = [
            component_metadata[(role, role_values[role].target.canonical_smiles)][
                "structural_provenance_stratum"
            ]
            for role in ROLE_NAMES
        ]
        if any(stratum not in COMPONENT_STRATA for stratum in component_strata):
            raise Ugi3FreshPoolRouteCoverageError("invalid component provenance stratum")
        product_stratum = classify_product_provenance(component_strata)
        value = ProductSynthesisValue(
            product_smiles=_canonical_constitution(
                row.get("smiles"), label=f"sample {sample_index} product"
            ),
            l1_forward_consistent=True,
            components=tuple(sorted(role_values.items())),
        )
        outcome = "complete" if value.route_complete else "noncomplete"
        product_outcomes[outcome] += 1
        product_outcomes_by_provenance[product_stratum][outcome] += 1
        complete_component_counts[sum(item.route_complete for item in role_values.values())] += 1
        product_records.append(
            {
                "sample_index": sample_index,
                "structure_id": row.get("structure_id"),
                "product_id": row.get("product_id"),
                "product_structural_provenance_stratum": product_stratum,
                "value": value.to_dict(),
            }
        )

    expected_products = fresh_audit.get("summary", {}).get("products_by_structural_provenance")
    observed_products = Counter(
        row["product_structural_provenance_stratum"] for row in product_records
    )
    if dict(sorted(observed_products.items())) != expected_products:
        raise Ugi3FreshPoolRouteCoverageError("fresh product provenance cross-check failed")
    component_payload = {
        "schema_version": COMPONENT_LEDGER_SCHEMA_VERSION,
        "records": component_records,
    }
    product_payload = {
        "schema_version": PRODUCT_LEDGER_SCHEMA_VERSION,
        "records": product_records,
    }
    component_ledger = _gzip_json_bytes(component_payload)
    product_ledger = _gzip_json_bytes(product_payload)
    summary = {
        "attempted_draws": len(sample.get("samples", [])),
        "exact_l1_products": len(product_records),
        "unique_components": len(component_records),
        "frozen_hybrid_component_values": len(hybrid_values),
        "frozen_v3_component_values": len(v3_values),
        "unique_component_outcomes": dict(sorted(unique_outcomes.items())),
        "unique_component_outcomes_by_role": {
            role: dict(sorted(unique_role_outcomes[role].items())) for role in ROLE_NAMES
        },
        "unique_component_value_sources": dict(sorted(unique_sources.items())),
        "component_occurrence_outcomes": dict(sorted(component_occurrence_outcomes.items())),
        "component_occurrence_outcomes_by_role": {
            role: dict(sorted(role_occurrence_outcomes[role].items())) for role in ROLE_NAMES
        },
        "component_occurrences_by_structural_provenance": dict(
            sorted(component_provenance_occurrences.items())
        ),
        "component_occurrences_by_value_source": dict(sorted(source_occurrences.items())),
        "product_outcomes": dict(sorted(product_outcomes.items())),
        "product_outcomes_by_structural_provenance": {
            stratum: dict(sorted(product_outcomes_by_provenance[stratum].items()))
            for stratum in sorted(product_outcomes_by_provenance)
        },
        "products_by_route_complete_component_count": {
            str(key): value for key, value in sorted(complete_component_counts.items())
        },
        "non_null_scalar_values": sum(
            record["value"]["scalar_value"] is not None for record in product_records
        ),
        "non_null_success_probabilities": sum(
            record["value"]["success_probability"] is not None for record in product_records
        ),
        "fresh_catalog_absent_unique_components": sum(
            record["structural_provenance_stratum"] == CATALOG_ABSENT
            for record in component_records
        ),
    }
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_nonselecting_exact_identity_route_coverage",
        "config": {"path": str(config_path.relative_to(repo)), "sha256": sha256_file(config_path)},
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "summary": summary,
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
        "adjudication": {
            "production_generator_changed": False,
            "prospective_candidate_selection_changed": False,
            "synthesis_guidance_authorized": False,
            "biological_guidance_authorized": False,
            "route_mining_priority_update_authorized": True,
        },
        "nonclaims": [
            "The values are not synthesis-success probabilities.",
            "Unknown protection and purification burdens remain unknown rather than zero.",
            "A missing exact route identity is missing knowledge, not chemical incompatibility.",
            "This exact-identity audit performs no new family projection or route planning.",
            "This audit does not authorize synthesis-guided generation or candidate lock.",
        ],
    }
    return result, component_ledger, product_ledger
