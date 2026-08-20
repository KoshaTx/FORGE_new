"""Nonselecting audit of exact and projected Ugi component evidence.

This module joins the frozen v5 component synthesis values to the final v5
route-gap triage.  It emits categorical evidence strata only.  It does not
define a scalar synthesis value, estimate synthesis-success probability, run
guidance, or select prospective candidates.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

from forge.corpus.r1_prime_audit import sha256_bytes, sha256_file
from forge.synthesis.engine.planner import AssessmentOutcome
from forge.synthesis.value.contracts import ComponentSynthesisValue

CONFIG_SCHEMA_VERSION = "phase1_ugi3_graded_family_evidence_audit_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi3_graded_family_evidence_audit.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi3_graded_family_evidence_ledger.v1"

EXACT_COMPLETE = "exact_complete_current"
FAMILY_ALL_CURRENT = "family_projected_all_current_leaves"
FAMILY_PARTIAL_CURRENT = "family_projected_partial_current_leaves"
FAMILY_NO_CURRENT = "family_projected_no_current_leaves"
MISSING_KNOWLEDGE = "missing_knowledge"
OUTSIDE_SUPPORT = "outside_support"
INCOMPATIBLE = "incompatible"
BUDGET_EXHAUSTED = "budget_exhausted"
INVALID_INPUT = "invalid_input"
EXECUTION_ERROR = "execution_error"

GRADED_CLASSES = (
    EXACT_COMPLETE,
    FAMILY_ALL_CURRENT,
    FAMILY_PARTIAL_CURRENT,
    FAMILY_NO_CURRENT,
    MISSING_KNOWLEDGE,
    OUTSIDE_SUPPORT,
    INCOMPATIBLE,
    BUDGET_EXHAUSTED,
    INVALID_INPUT,
    EXECUTION_ERROR,
)

FIELDS = (
    "schema_version",
    "role",
    "canonical_smiles",
    "structural_provenance_stratum",
    "catalog_provenance_substratum",
    "registry_component_id",
    "value_source_class",
    "exact_assessment_outcome",
    "exact_evidence_support",
    "exact_forward_consistency",
    "exact_route_complete",
    "exact_route_step_count",
    "exact_leaf_count",
    "exact_current_terminal_leaf_count",
    "in_final_one_gap_triage",
    "one_gap_priority_rank",
    "one_gap_product_count",
    "program_family",
    "projected_leaf_count",
    "current_projected_leaf_count",
    "historical_projected_leaf_count",
    "projected_leaves_json",
    "unresolved_projected_leaves_json",
    "projection_leaf_status",
    "family_scope_status",
    "graded_evidence_class",
    "evidence_relation",
    "exact_scope_still_required",
)


class Ugi3GradedFamilyEvidenceAuditError(ValueError):
    """Raised when the graded evidence audit cannot be reproduced safely."""


def _load_json(path: Path, *, compressed: bool, label: str) -> dict[str, Any]:
    try:
        if compressed:
            with gzip.open(path, "rt") as handle:
                value = json.load(handle)
        else:
            value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise Ugi3GradedFamilyEvidenceAuditError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Ugi3GradedFamilyEvidenceAuditError(f"{label} must be an object")
    return value


def _read_triage_rows(path: Path) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise Ugi3GradedFamilyEvidenceAuditError(f"invalid triage ledger: {path}") from exc
    required = {
        "schema_version",
        "priority_rank",
        "role",
        "canonical_smiles",
        "assessment_outcome",
        "one_gap_product_count",
        "target_current_terminal",
        "program_family",
        "projected_leaf_count",
        "current_projected_leaf_count",
        "historical_projected_leaf_count",
        "projected_leaves_json",
        "unresolved_projected_leaves_json",
        "triage_class",
    }
    if not rows or not required.issubset(rows[0]):
        raise Ugi3GradedFamilyEvidenceAuditError("triage ledger schema changed")
    return rows


def _gzip_csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        raise Ugi3GradedFamilyEvidenceAuditError("cannot serialize an empty evidence ledger")
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        compressed.write(text.getvalue().encode())
    return output.getvalue()


def _parse_bool(value: str, *, label: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise Ugi3GradedFamilyEvidenceAuditError(f"{label} must be True or False")


def _parse_nonnegative_int(value: str, *, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise Ugi3GradedFamilyEvidenceAuditError(f"{label} must be an integer") from exc
    if parsed < 0:
        raise Ugi3GradedFamilyEvidenceAuditError(f"{label} must be nonnegative")
    return parsed


def classify_graded_evidence(
    *,
    exact_value: ComponentSynthesisValue,
    program_family: str,
    projected_leaf_count: int,
    current_projected_leaf_count: int,
) -> tuple[str, str, str, bool]:
    """Return a categorical evidence class without changing the exact value."""

    if projected_leaf_count < 0 or current_projected_leaf_count < 0:
        raise Ugi3GradedFamilyEvidenceAuditError("projected leaf counts must be nonnegative")
    if current_projected_leaf_count > projected_leaf_count:
        raise Ugi3GradedFamilyEvidenceAuditError("current projected leaves exceed total leaves")
    if exact_value.route_complete:
        if program_family or projected_leaf_count or current_projected_leaf_count:
            raise Ugi3GradedFamilyEvidenceAuditError(
                "exact-complete component cannot also carry a route-gap projection"
            )
        if not exact_value.all_terminal_leaves_current:
            raise Ugi3GradedFamilyEvidenceAuditError(
                "exact-complete component lacks current terminal closure"
            )
        return EXACT_COMPLETE, "not_projected", "highest_exact_complete_evidence", False

    if program_family:
        if exact_value.assessment_outcome is not AssessmentOutcome.MISSING_KNOWLEDGE:
            raise Ugi3GradedFamilyEvidenceAuditError(
                "family projection cannot replace a typed non-missing outcome"
            )
        if projected_leaf_count == 0:
            raise Ugi3GradedFamilyEvidenceAuditError("family projection lacks projected leaves")
        if current_projected_leaf_count == projected_leaf_count:
            return (
                FAMILY_ALL_CURRENT,
                "all_current",
                "lower_confidence_family_projection_below_exact_complete",
                True,
            )
        if current_projected_leaf_count > 0:
            return (
                FAMILY_PARTIAL_CURRENT,
                "partial_current",
                "lower_confidence_family_projection_with_partial_leaf_closure",
                True,
            )
        return (
            FAMILY_NO_CURRENT,
            "no_current",
            "lower_confidence_family_projection_without_current_leaf_closure",
            True,
        )

    if projected_leaf_count or current_projected_leaf_count:
        raise Ugi3GradedFamilyEvidenceAuditError(
            "projected leaves cannot exist without a projected program family"
        )
    outcome = exact_value.assessment_outcome
    outcome_classes = {
        AssessmentOutcome.MISSING_KNOWLEDGE: (
            MISSING_KNOWLEDGE,
            "abstain_missing_knowledge",
        ),
        AssessmentOutcome.OUTSIDE_SUPPORT: (
            OUTSIDE_SUPPORT,
            "abstain_outside_declared_support",
        ),
        AssessmentOutcome.INCOMPATIBLE: (
            INCOMPATIBLE,
            "abstain_evidenced_incompatibility",
        ),
        AssessmentOutcome.BUDGET_EXHAUSTED: (
            BUDGET_EXHAUSTED,
            "abstain_budget_exhausted",
        ),
        AssessmentOutcome.INVALID_INPUT: (INVALID_INPUT, "abstain_invalid_input"),
        AssessmentOutcome.EXECUTION_ERROR: (
            EXECUTION_ERROR,
            "abstain_execution_error",
        ),
    }
    try:
        evidence_class, relation = outcome_classes[outcome]
    except KeyError as exc:
        raise Ugi3GradedFamilyEvidenceAuditError(
            f"unsupported exact assessment outcome: {outcome.value}"
        ) from exc
    return evidence_class, "not_projected", relation, True


def _validate_inputs(config: dict[str, Any], repo: Path) -> dict[str, Path]:
    specifications = config.get("inputs")
    required = {
        "coverage_result",
        "component_values",
        "triage_result",
        "triage_ledger",
        "audit_source",
        "audit_runner",
        "audit_tests",
    }
    if not isinstance(specifications, dict) or set(specifications) != required:
        raise Ugi3GradedFamilyEvidenceAuditError("graded evidence input set changed")
    paths: dict[str, Path] = {}
    for label, specification in specifications.items():
        if not isinstance(specification, dict) or set(specification) != {"path", "sha256"}:
            raise Ugi3GradedFamilyEvidenceAuditError(f"input {label} is malformed")
        path = (repo / specification["path"]).resolve()
        if sha256_file(path) != specification["sha256"]:
            raise Ugi3GradedFamilyEvidenceAuditError(f"input hash changed: {label}")
        paths[label] = path
    return paths


def _validate_policy(config: dict[str, Any]) -> None:
    policy = config.get("audit_policy")
    required_true = (
        "exact_complete_is_highest_evidence",
        "family_projection_all_current_is_lower_confidence",
        "partial_and_no_current_projected_leaves_are_distinct",
        "typed_failure_outcomes_remain_distinct",
        "exact_scope_required_before_candidate_lock",
    )
    required_false = (
        "family_projection_is_exact_route_evidence",
        "family_projection_changes_exact_assessment_outcome",
        "family_scope_qualification_performed",
        "scalar_synthesis_value_defined",
        "synthesis_success_probability_defined",
        "synthesis_guidance_run",
        "prospective_candidate_selection_changed",
        "holdout_reveal_authorized",
    )
    if not isinstance(policy, dict) or any(policy.get(key) is not True for key in required_true):
        raise Ugi3GradedFamilyEvidenceAuditError("positive graded-evidence safeguards changed")
    if any(policy.get(key) is not False for key in required_false):
        raise Ugi3GradedFamilyEvidenceAuditError("nonpromotion safeguards changed")


def build_graded_family_evidence_audit(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], bytes]:
    """Build a deterministic categorical ledger over the final v5 evidence."""

    config = _load_json(config_path, compressed=False, label="graded evidence config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3GradedFamilyEvidenceAuditError("unsupported graded evidence config")
    _validate_policy(config)
    paths = _validate_inputs(config, repo)

    coverage = _load_json(paths["coverage_result"], compressed=False, label="coverage result")
    components = _load_json(paths["component_values"], compressed=True, label="component values")
    triage_result = _load_json(paths["triage_result"], compressed=False, label="triage result")
    triage_rows = _read_triage_rows(paths["triage_ledger"])

    component_hash = sha256_file(paths["component_values"])
    if (
        coverage.get("artifacts", {}).get("component_synthesis_values.json.gz", {}).get("sha256")
        != component_hash
    ):
        raise Ugi3GradedFamilyEvidenceAuditError("coverage does not own component ledger")
    if components.get("schema_version") != "phase1_ugi3_fresh_pool_component_values.v5":
        raise Ugi3GradedFamilyEvidenceAuditError("component ledger is not final v5")
    triage_artifact = triage_result.get("artifacts", {}).get("triage_ledger", {})
    if (
        triage_artifact.get("sha256") != sha256_file(paths["triage_ledger"])
        or triage_result.get("decision", {}).get("holdout_remains_unrevealed") is not True
    ):
        raise Ugi3GradedFamilyEvidenceAuditError("final triage ownership or seal failed")
    expected_triage_rows = int(triage_result.get("summary", {}).get("one_gap_components", -1))
    if len(triage_rows) != expected_triage_rows:
        raise Ugi3GradedFamilyEvidenceAuditError("triage row denominator changed")

    triage_index: dict[tuple[str, str], dict[str, str]] = {}
    for row in triage_rows:
        if row["schema_version"] != "phase1_ugi3_route_gap_triage_ledger.v2":
            raise Ugi3GradedFamilyEvidenceAuditError("triage row schema changed")
        key = (row["role"], row["canonical_smiles"])
        if key in triage_index:
            raise Ugi3GradedFamilyEvidenceAuditError("duplicate triage component identity")
        triage_index[key] = row

    records = components.get("records")
    if not isinstance(records, list) or not records:
        raise Ugi3GradedFamilyEvidenceAuditError("component ledger records are missing")
    expected_components = int(coverage.get("summary", {}).get("unique_components", -1))
    if len(records) != expected_components:
        raise Ugi3GradedFamilyEvidenceAuditError("component denominator changed")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    class_counts: Counter[str] = Counter({name: 0 for name in GRADED_CLASSES})
    exact_outcomes: Counter[str] = Counter()
    exact_completion_basis: Counter[str] = Counter()
    projection_programs: Counter[str] = Counter()
    projection_leaf_states: Counter[str] = Counter()
    one_gap_class_counts: Counter[str] = Counter({name: 0 for name in GRADED_CLASSES})
    one_gap_product_counts: Counter[str] = Counter({name: 0 for name in GRADED_CLASSES})

    for record in records:
        exact_value = ComponentSynthesisValue.from_dict(record.get("value"))
        key = (str(record.get("role")), str(record.get("canonical_smiles")))
        if key in seen:
            raise Ugi3GradedFamilyEvidenceAuditError("duplicate component identity")
        seen.add(key)
        if key != (exact_value.target.role, exact_value.target.canonical_smiles):
            raise Ugi3GradedFamilyEvidenceAuditError("component metadata and value target differ")

        triage = triage_index.get(key)
        if triage is None:
            priority_rank = 0
            one_gap_product_count = 0
            program_family = ""
            projected_leaf_count = 0
            current_projected_leaf_count = 0
            historical_projected_leaf_count = 0
            projected_leaves: list[str] = []
            unresolved_leaves: list[str] = []
        else:
            if triage["assessment_outcome"] != exact_value.assessment_outcome.value:
                raise Ugi3GradedFamilyEvidenceAuditError(
                    "triage changed the exact component assessment outcome"
                )
            if _parse_bool(triage["target_current_terminal"], label="target_current_terminal"):
                raise Ugi3GradedFamilyEvidenceAuditError(
                    "final v5 triage retains an unapplied current exact target"
                )
            priority_rank = _parse_nonnegative_int(triage["priority_rank"], label="priority_rank")
            one_gap_product_count = _parse_nonnegative_int(
                triage["one_gap_product_count"], label="one_gap_product_count"
            )
            program_family = triage["program_family"]
            projected_leaf_count = _parse_nonnegative_int(
                triage["projected_leaf_count"], label="projected_leaf_count"
            )
            current_projected_leaf_count = _parse_nonnegative_int(
                triage["current_projected_leaf_count"],
                label="current_projected_leaf_count",
            )
            historical_projected_leaf_count = _parse_nonnegative_int(
                triage["historical_projected_leaf_count"],
                label="historical_projected_leaf_count",
            )
            try:
                projected_leaves = json.loads(triage["projected_leaves_json"])
                unresolved_leaves = json.loads(triage["unresolved_projected_leaves_json"])
            except json.JSONDecodeError as exc:
                raise Ugi3GradedFamilyEvidenceAuditError(
                    "triage projected leaves are malformed"
                ) from exc
            if (
                not isinstance(projected_leaves, list)
                or not all(isinstance(item, str) for item in projected_leaves)
                or not isinstance(unresolved_leaves, list)
                or not all(isinstance(item, str) for item in unresolved_leaves)
                or len(projected_leaves) != projected_leaf_count
                or len(unresolved_leaves) != projected_leaf_count - current_projected_leaf_count
            ):
                raise Ugi3GradedFamilyEvidenceAuditError(
                    "triage projected leaf counts do not match identities"
                )

        evidence_class, leaf_status, relation, exact_scope_required = classify_graded_evidence(
            exact_value=exact_value,
            program_family=program_family,
            projected_leaf_count=projected_leaf_count,
            current_projected_leaf_count=current_projected_leaf_count,
        )
        row = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "role": key[0],
            "canonical_smiles": key[1],
            "structural_provenance_stratum": record.get("structural_provenance_stratum", ""),
            "catalog_provenance_substratum": record.get("catalog_provenance_substratum", ""),
            "registry_component_id": record.get("registry_component_id") or "",
            "value_source_class": record.get("source_class", ""),
            "exact_assessment_outcome": exact_value.assessment_outcome.value,
            "exact_evidence_support": exact_value.evidence_support.value,
            "exact_forward_consistency": exact_value.forward_consistency.value,
            "exact_route_complete": exact_value.route_complete,
            "exact_route_step_count": exact_value.route_step_count,
            "exact_leaf_count": exact_value.leaf_count,
            "exact_current_terminal_leaf_count": exact_value.current_terminal_leaf_count,
            "in_final_one_gap_triage": triage is not None,
            "one_gap_priority_rank": priority_rank,
            "one_gap_product_count": one_gap_product_count,
            "program_family": program_family,
            "projected_leaf_count": projected_leaf_count,
            "current_projected_leaf_count": current_projected_leaf_count,
            "historical_projected_leaf_count": historical_projected_leaf_count,
            "projected_leaves_json": json.dumps(
                projected_leaves, separators=(",", ":"), sort_keys=True
            ),
            "unresolved_projected_leaves_json": json.dumps(
                unresolved_leaves, separators=(",", ":"), sort_keys=True
            ),
            "projection_leaf_status": leaf_status,
            "family_scope_status": (
                "not_qualified_by_this_audit" if program_family else "not_applicable"
            ),
            "graded_evidence_class": evidence_class,
            "evidence_relation": relation,
            "exact_scope_still_required": exact_scope_required,
        }
        rows.append(row)
        class_counts[evidence_class] += 1
        exact_outcomes[exact_value.assessment_outcome.value] += 1
        if evidence_class == EXACT_COMPLETE:
            basis = (
                "exact_route_current" if exact_value.route_step_count else "exact_terminal_current"
            )
            exact_completion_basis[basis] += 1
        if program_family:
            projection_programs[program_family] += 1
            projection_leaf_states[leaf_status] += 1
        if triage is not None:
            one_gap_class_counts[evidence_class] += 1
            one_gap_product_counts[evidence_class] += one_gap_product_count

    if set(triage_index) - seen:
        raise Ugi3GradedFamilyEvidenceAuditError("triage identity lacks an exact component value")
    rows.sort(key=lambda row: (str(row["role"]), str(row["canonical_smiles"])))
    ledger = _gzip_csv_bytes(rows)

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_nonselecting_graded_family_evidence_audit",
        "task": config.get("task"),
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "inputs": {
            label: {
                "path": str(path.relative_to(repo)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for label, path in sorted(paths.items())
        },
        "summary": {
            "components": len(rows),
            "components_by_graded_evidence_class": dict(class_counts),
            "components_by_exact_assessment_outcome": dict(sorted(exact_outcomes.items())),
            "exact_complete_components_by_basis": dict(sorted(exact_completion_basis.items())),
            "family_projected_components_by_program": dict(sorted(projection_programs.items())),
            "family_projected_components_by_leaf_status": dict(
                sorted(projection_leaf_states.items())
            ),
            "one_gap_components": len(triage_rows),
            "one_gap_components_by_graded_evidence_class": dict(one_gap_class_counts),
            "one_gap_product_occurrences": sum(one_gap_product_counts.values()),
            "one_gap_product_occurrences_by_graded_evidence_class": dict(one_gap_product_counts),
        },
        "declared_evidence_order": [
            EXACT_COMPLETE,
            FAMILY_ALL_CURRENT,
            FAMILY_PARTIAL_CURRENT,
            FAMILY_NO_CURRENT,
        ],
        "typed_abstentions": [
            MISSING_KNOWLEDGE,
            OUTSIDE_SUPPORT,
            INCOMPATIBLE,
            BUDGET_EXHAUSTED,
            INVALID_INPUT,
            EXECUTION_ERROR,
        ],
        "adjudication": {
            "graded_family_evidence_audit_completed": True,
            "family_projection_promoted_to_exact": False,
            "family_scope_qualification_performed": False,
            "scalar_synthesis_value_defined": False,
            "synthesis_success_probability_defined": False,
            "synthesis_guidance_run": False,
            "prospective_candidate_selection_changed": False,
            "candidate_lock_authorized": False,
            "holdout_revealed": False,
        },
        "nonclaims": [
            "The declared evidence order is not a synthesis-success probability.",
            "A family projection does not change the exact component assessment outcome.",
            "This audit does not qualify a projected family's bounded substrate scope.",
            "Current projected leaves do not establish exact-substrate reaction scope.",
            "Partial or absent current leaf evidence remains distinct from all-current projection.",
            "Missing knowledge, outside support and incompatibility are distinct outcomes.",
            "This audit does not run guidance, select candidates or reveal the holdout.",
        ],
        "artifacts": {
            "graded_family_evidence_ledger.csv.gz": {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "rows": len(rows),
                "sha256": sha256_bytes(ledger),
            }
        },
    }
    return result, ledger
