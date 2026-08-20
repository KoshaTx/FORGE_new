"""Apply authenticated exact-terminal deltas to fresh-v2 route coverage."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.data.r1_prime_audit import sha256_bytes, sha256_file
from forge.route.engine.planner import (
    AvailabilityState,
    EvidenceRecord,
    EvidenceTier,
    ForwardVerificationState,
    KnowledgeDisposition,
    KnowledgeResult,
    PlannerBudgetLedger,
    PlannerBudgetLimits,
    RecursiveRouteAssessor,
    RouteTarget,
)
from forge.route.terminals.ugi3_high_leverage_head_terminals import _parse_utc
from forge.value.coverage.ugi3_fresh_pool_route_coverage import (
    _canonical_constitution,
    _gzip_json_bytes,
    _load_gzip_json,
    _load_json,
)
from forge.value.synthesis.synthesis import (
    ComponentSynthesisValue,
    ProductSynthesisValue,
    component_synthesis_value_from_assessment,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi3_fresh_pool_route_coverage_config.v2"
RESULT_SCHEMA_VERSION = "phase1_ugi3_fresh_pool_route_coverage.v2"
COMPONENT_LEDGER_SCHEMA_VERSION = "phase1_ugi3_fresh_pool_component_values.v2"
PRODUCT_LEDGER_SCHEMA_VERSION = "phase1_ugi3_fresh_pool_product_values.v2"

HEAD_ROLE = "amine_head"
ALDEHYDE_ROLE = "oxoester_aldehyde_body_tail"
HEAD_SMILES = "NC1CCNCC1"
ALDEHYDE_SMILES = "CCCCCCCCCCCCCCCCC=O"


class Ugi3FreshPoolRouteCoverageV2Error(ValueError):
    """Raised when the exact-terminal delta cannot be reproduced safely."""


class _ExactTerminalSource:
    def __init__(self, results: dict[tuple[str, str], KnowledgeResult]):
        self._results = dict(results)

    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        key = (
            target.role,
            _canonical_constitution(target.canonical_smiles, label="route target"),
        )
        result = self._results.get(key)
        if result is not None:
            return result
        return KnowledgeResult(
            disposition=KnowledgeDisposition.MISSING_KNOWLEDGE,
            evidence=(),
            detail="identity is outside the exact-terminal delta",
        )


def _validate_inputs(config: Mapping[str, Any], repo: Path) -> dict[str, Path]:
    declared = config.get("inputs")
    if not isinstance(declared, dict) or not declared:
        raise Ugi3FreshPoolRouteCoverageV2Error("v2 inputs are missing")
    paths: dict[str, Path] = {}
    for label, record in declared.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise Ugi3FreshPoolRouteCoverageV2Error(f"input {label} is malformed")
        path = repo / str(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise Ugi3FreshPoolRouteCoverageV2Error(f"input hash changed: {label}")
        paths[label] = path
    return paths


def _find_head_record(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    records = payload.get("records")
    if not isinstance(records, list):
        raise Ugi3FreshPoolRouteCoverageV2Error("head procurement records are missing")
    matches = [record for record in records if record.get("canonical_smiles") == HEAD_SMILES]
    if len(matches) != 1:
        raise Ugi3FreshPoolRouteCoverageV2Error("exact head procurement record is not unique")
    record = matches[0]
    if (
        record.get("procurement_status") != "current_item_level_vendor_verified"
        or record.get("current_item_level_procurement_closed") is not True
        or record.get("identity", {}).get("inchi_key") != "BCIIMDOZSUCSEN-UHFFFAOYSA-N"
        or record.get("vendor_evidence", {}).get("product_code") != "A1591"
    ):
        raise Ugi3FreshPoolRouteCoverageV2Error("head procurement evidence is not exact/current")
    return record


def _find_tail_record(
    payload: Mapping[str, Any], *, identity_asset: Path, failed_archive: Path
) -> Mapping[str, Any]:
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 1:
        raise Ugi3FreshPoolRouteCoverageV2Error("tail evidence must contain one record")
    record = records[0]
    identity = _load_json(identity_asset, label="heptadecanal identity")
    properties = identity.get("PropertyTable", {}).get("Properties")
    if not isinstance(properties, list) or len(properties) != 1:
        raise Ugi3FreshPoolRouteCoverageV2Error("heptadecanal identity is malformed")
    exact = properties[0]
    if (
        record.get("role") != ALDEHYDE_ROLE
        or record.get("canonical_smiles") != ALDEHYDE_SMILES
        or record.get("component_id") != "ugi-component-ee515d932efe6740f313"
        or record.get("disposition") != "admit_exact_terminal"
        or record.get("availability") != "available_to_ship_same_day_us"
        or record.get("supplier") != "TCI America"
        or record.get("product_number") != "H1295"
        or record.get("source_url") != "https://www.tcichemicals.com/US/en/p/H1295"
        or record.get("identity_asset_sha256") != sha256_file(identity_asset)
        or record.get("failed_archive_asset_sha256") != sha256_file(failed_archive)
        or exact.get("CID") != 71552
        or exact.get("ConnectivitySMILES") != ALDEHYDE_SMILES
        or exact.get("MolecularFormula") != "C17H34O"
        or exact.get("InChIKey") != "PIYDVAYKYBWPPY-UHFFFAOYSA-N"
        or "Access Denied" not in failed_archive.read_text()
    ):
        raise Ugi3FreshPoolRouteCoverageV2Error("tail terminal evidence is not exact/current")
    snapshot = payload.get("snapshot")
    if not isinstance(snapshot, dict):
        raise Ugi3FreshPoolRouteCoverageV2Error("tail snapshot is missing")
    accessed = _parse_utc(snapshot.get("accessed_utc"), label="tail accessed_utc")
    expires = _parse_utc(snapshot.get("expires_utc"), label="tail expires_utc")
    if expires <= accessed:
        raise Ugi3FreshPoolRouteCoverageV2Error("tail availability snapshot does not expire")
    return record


def _terminal_value(
    *, target: RouteTarget, source_path: Path, evidence_id: str, locator: str
) -> ComponentSynthesisValue:
    result = KnowledgeResult(
        disposition=KnowledgeDisposition.TERMINAL,
        evidence=(
            EvidenceRecord(
                evidence_id=evidence_id,
                tier=EvidenceTier.ACCEPTED_TERMINAL,
                source_sha256=sha256_file(source_path),
                source_locator=locator,
                exact_substrate=True,
                forward_verification=ForwardVerificationState.NOT_APPLICABLE,
                availability=AvailabilityState.CURRENT_CLOSED,
            ),
        ),
        detail="exact current item-level procurement terminal",
    )
    limits = PlannerBudgetLimits(
        maximum_depth=1,
        maximum_expansions=1,
        maximum_product_candidates=1,
        maximum_verifier_calls=0,
        maximum_elapsed_milliseconds=0,
        maximum_logical_planner_calls=1,
    )
    assessment = RecursiveRouteAssessor(
        _ExactTerminalSource({(target.role, target.canonical_smiles): result})
    ).assess(target, PlannerBudgetLedger(limits))
    value = component_synthesis_value_from_assessment(assessment)
    if not value.route_complete:
        raise Ugi3FreshPoolRouteCoverageV2Error("exact terminal did not close")
    return value


def build_fresh_pool_route_coverage_v2(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], bytes, bytes]:
    """Apply two exact-current terminal deltas to immutable fresh-v2 values."""

    config = _load_json(config_path, label="fresh-pool route v2 config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3FreshPoolRouteCoverageV2Error("unsupported fresh-pool route v2 config")
    paths = _validate_inputs(config, repo)
    required = {
        "base_component_values",
        "base_product_values",
        "base_result",
        "head_procurement_evidence",
        "tail_terminal_evidence",
        "tail_identity_asset",
        "tail_failed_archive",
        "audit_source",
        "audit_runner",
        "audit_tests",
        "value_source",
    }
    if set(paths) != required:
        raise Ugi3FreshPoolRouteCoverageV2Error("v2 input set changed")
    base_result = _load_json(paths["base_result"], label="base route coverage")
    artifacts = base_result.get("artifacts", {})
    if artifacts.get("component_synthesis_values.json.gz", {}).get("sha256") != sha256_file(
        paths["base_component_values"]
    ) or artifacts.get("product_synthesis_values.json.gz", {}).get("sha256") != sha256_file(
        paths["base_product_values"]
    ):
        raise Ugi3FreshPoolRouteCoverageV2Error("base route-coverage ownership failed")

    head_payload = _load_json(paths["head_procurement_evidence"], label="head procurement")
    _find_head_record(head_payload)
    tail_payload = _load_json(paths["tail_terminal_evidence"], label="tail terminal evidence")
    _find_tail_record(
        tail_payload,
        identity_asset=paths["tail_identity_asset"],
        failed_archive=paths["tail_failed_archive"],
    )
    targets = {
        (HEAD_ROLE, HEAD_SMILES): _terminal_value(
            target=RouteTarget(role=HEAD_ROLE, canonical_smiles=HEAD_SMILES),
            source_path=paths["head_procurement_evidence"],
            evidence_id="existing-head-terminal:A20",
            locator=f"{paths['head_procurement_evidence']}#records[A20]",
        ),
        (ALDEHYDE_ROLE, ALDEHYDE_SMILES): _terminal_value(
            target=RouteTarget(role=ALDEHYDE_ROLE, canonical_smiles=ALDEHYDE_SMILES),
            source_path=paths["tail_terminal_evidence"],
            evidence_id="tail-terminal:H1295",
            locator=f"{paths['tail_terminal_evidence']}#records[0]",
        ),
    }
    target_sources = {
        (HEAD_ROLE, HEAD_SMILES): "existing_exact_current_head_terminal_refresh",
        (ALDEHYDE_ROLE, ALDEHYDE_SMILES): "new_exact_current_tail_terminal",
    }

    base_components = _load_gzip_json(paths["base_component_values"], label="base components")
    records = base_components.get("records")
    if not isinstance(records, list):
        raise Ugi3FreshPoolRouteCoverageV2Error("base component records are missing")
    updated_values: dict[tuple[str, str], ComponentSynthesisValue] = {}
    component_records = []
    source_by_key: dict[tuple[str, str], str] = {}
    changed: Counter[str] = Counter()
    unique_outcomes: Counter[str] = Counter()
    unique_role_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    unique_sources: Counter[str] = Counter()
    for raw in records:
        if not isinstance(raw, dict):
            raise Ugi3FreshPoolRouteCoverageV2Error("base component record is malformed")
        key = (str(raw.get("role")), str(raw.get("canonical_smiles")))
        value = ComponentSynthesisValue.from_dict(raw.get("value"))
        output = dict(raw)
        if key in targets:
            if value.route_complete:
                raise Ugi3FreshPoolRouteCoverageV2Error("delta target was already complete")
            value = targets[key]
            output["source_class"] = target_sources[key]
            output["value"] = value.to_dict()
            changed[f"{key[0]}\t{key[1]}"] += 1
        updated_values[key] = value
        source_by_key[key] = str(output.get("source_class"))
        component_records.append(output)
        unique_outcomes[value.assessment_outcome.value] += 1
        unique_role_outcomes[key[0]][value.assessment_outcome.value] += 1
        unique_sources[str(output.get("source_class"))] += 1
    if changed != Counter({f"{role}\t{smiles}": 1 for role, smiles in targets}):
        raise Ugi3FreshPoolRouteCoverageV2Error("terminal deltas did not match exactly once")

    base_products = _load_gzip_json(paths["base_product_values"], label="base products")
    product_rows = base_products.get("records")
    if not isinstance(product_rows, list):
        raise Ugi3FreshPoolRouteCoverageV2Error("base product records are missing")
    product_records = []
    occurrence_outcomes: Counter[str] = Counter()
    role_occurrence_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    source_occurrences: Counter[str] = Counter()
    product_outcomes: Counter[str] = Counter()
    product_outcomes_by_provenance: dict[str, Counter[str]] = defaultdict(Counter)
    complete_component_counts: Counter[int] = Counter()
    target_occurrences: Counter[str] = Counter()
    single_target_new_closures: Counter[str] = Counter()
    joint_target_new_closures = 0
    base_complete = 0
    for raw in product_rows:
        if not isinstance(raw, dict):
            raise Ugi3FreshPoolRouteCoverageV2Error("base product record is malformed")
        prior = ProductSynthesisValue.from_dict(raw.get("value"))
        base_complete += int(prior.route_complete)
        role_values = {}
        prior_gaps = set()
        target_hits = set()
        for role, old_value in prior.components:
            key = (role, old_value.target.canonical_smiles)
            value = updated_values.get(key)
            if value is None:
                raise Ugi3FreshPoolRouteCoverageV2Error("product component is absent from ledger")
            role_values[role] = value
            if not old_value.route_complete:
                prior_gaps.add(key)
            if key in targets:
                target_hits.add(key)
                target_occurrences[f"{key[0]}\t{key[1]}"] += 1
            occurrence_outcomes[value.assessment_outcome.value] += 1
            role_occurrence_outcomes[role][value.assessment_outcome.value] += 1
            source_occurrences[target_sources.get(key, source_by_key[key])] += 1
        value = ProductSynthesisValue(
            product_smiles=prior.product_smiles,
            l1_forward_consistent=prior.l1_forward_consistent,
            components=tuple(sorted(role_values.items())),
        )
        if value.route_complete and not prior.route_complete:
            closing_targets = prior_gaps & set(targets)
            if len(closing_targets) == 1:
                key = next(iter(closing_targets))
                single_target_new_closures[f"{key[0]}\t{key[1]}"] += 1
            elif len(closing_targets) > 1:
                joint_target_new_closures += 1
        outcome = "complete" if value.route_complete else "noncomplete"
        provenance = str(raw.get("product_structural_provenance_stratum"))
        product_outcomes[outcome] += 1
        product_outcomes_by_provenance[provenance][outcome] += 1
        complete_component_counts[sum(item.route_complete for item in role_values.values())] += 1
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
        "base_version": "fresh_pool_route_coverage_v1",
        "exact_terminal_components_changed": len(changed),
        "changed_component_keys": sorted(changed),
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
        "newly_complete_products": product_outcomes["complete"] - base_complete,
        "single_target_new_closures": dict(sorted(single_target_new_closures.items())),
        "joint_target_new_closures": joint_target_new_closures,
        "target_occurrences": dict(sorted(target_occurrences.items())),
        "products_by_route_complete_component_count": {
            str(key): value for key, value in sorted(complete_component_counts.items())
        },
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
        raise Ugi3FreshPoolRouteCoverageV2Error("fresh-pool route v2 summary changed")
    return (
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "complete_nonselecting_exact_terminal_delta",
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
            "policy": config.get("assessment_policy"),
            "adjudication": {
                "route_mining_priority_update_authorized": True,
                "synthesis_guidance_authorized": False,
                "prospective_candidate_selection_changed": False,
            },
            "nonclaims": [
                "Current procurement does not establish Ugi substrate compatibility or product success.",
                "No neighboring identity, reaction family or homologue is promoted.",
                "Unknown burden remains unknown rather than zero.",
                "This audit does not authorize synthesis-guided generation or candidate selection.",
            ],
        },
        component_ledger,
        product_ledger,
    )
