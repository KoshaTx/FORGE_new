"""Classify one-gap Ugi route targets before targeted evidence mining."""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_file
from forge.route.ugi3_agile_template_saturation_stress import projected_leaf_candidates
from forge.route.ugi3_precursor_leaf_closure import (
    ALDEHYDE_ROLE,
    HEAD_ROLE,
    ISOCYANIDE_ROLE,
    _parse_utc,
    _validated_inputs,
    constitutional_key,
    current_terminal_evidence,
    historical_leaf_source_use,
)
from forge.route.ugi3_virtual_programs import _aldehyde_program, _isocyanide_program

CONFIG_SCHEMA_VERSION = "phase1_ugi3_route_gap_triage_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi3_route_gap_triage.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi3_route_gap_triage_ledger.v1"


class Ugi3RouteGapTriageError(ValueError):
    """Raised when route-gap triage cannot be reproduced safely."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise Ugi3RouteGapTriageError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Ugi3RouteGapTriageError(f"{label} must be an object")
    return value


def _validate_inputs(config: Mapping[str, Any], repo: Path) -> dict[str, Path]:
    specifications = config.get("inputs")
    if not isinstance(specifications, dict) or not specifications:
        raise Ugi3RouteGapTriageError("triage inputs are missing")
    output: dict[str, Path] = {}
    for label, specification in specifications.items():
        if not isinstance(specification, dict) or set(specification) != {"path", "sha256"}:
            raise Ugi3RouteGapTriageError(f"input {label!r} is malformed")
        path = repo / str(specification["path"])
        if sha256_file(path) != str(specification["sha256"]):
            raise Ugi3RouteGapTriageError(f"input hash changed: {label}")
        output[label] = path
    return output


def _read_priority_rows(path: Path) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise Ugi3RouteGapTriageError("priority ledger has no header")
            return list(reader)
    except (OSError, csv.Error) as exc:
        raise Ugi3RouteGapTriageError("could not read priority ledger") from exc


def classify_route_gap(
    *,
    role: str,
    target_smiles: str,
    assessment_outcome: str,
    current_terminal_keys: set[str],
    historical_leaf_keys: set[tuple[str, str]],
) -> dict[str, Any]:
    """Classify one target without upgrading structural projections."""

    target_key = constitutional_key(target_smiles)
    base = {
        "target_current_terminal": target_key in current_terminal_keys,
        "program_family": "",
        "projected_leaf_count": 0,
        "current_projected_leaf_count": 0,
        "historical_projected_leaf_count": 0,
        "projected_leaves_json": "[]",
        "unresolved_projected_leaves_json": "[]",
        "new_reaction_family_indicated_now": False,
    }
    if assessment_outcome == "outside_support":
        return {
            **base,
            "triage_class": "outside_declared_route_support_review",
            "recommended_action": (
                "review declared support and chemistry semantics before any route mining"
            ),
        }
    if assessment_outcome != "missing_knowledge":
        return {
            **base,
            "triage_class": "typed_nonmissing_failure_review",
            "recommended_action": "review the typed assessment without changing evidence tier",
        }
    if target_key in current_terminal_keys:
        return {
            **base,
            "triage_class": "current_target_terminal_available_value_refresh",
            "recommended_action": (
                "verify role and handle identity, then admit through an exact terminal overlay"
            ),
        }
    if role == HEAD_ROLE:
        return {
            **base,
            "triage_class": "head_procurement_or_exact_upstream_route_missing",
            "recommended_action": (
                "seek current exact procurement first; otherwise retrieve an exact head route"
            ),
        }
    try:
        if role == ALDEHYDE_ROLE:
            family, steps = _aldehyde_program(target_key)
        elif role == ISOCYANIDE_ROLE:
            family, steps = _isocyanide_program(target_key)
        else:
            raise Ugi3RouteGapTriageError(f"unexpected component role: {role!r}")
        leaves = tuple(constitutional_key(value) for value in projected_leaf_candidates(steps))
    except (ValueError, Ugi3RouteGapTriageError):
        return {
            **base,
            "triage_class": "no_mechanical_program_family_discovery_candidate",
            "recommended_action": (
                "cluster recurrent unsupported targets before considering a new reaction family"
            ),
        }
    current = [leaf for leaf in leaves if leaf in current_terminal_keys]
    historical = [leaf for leaf in leaves if (role, leaf) in historical_leaf_keys]
    unresolved = [leaf for leaf in leaves if leaf not in current_terminal_keys]
    common = {
        **base,
        "program_family": family,
        "projected_leaf_count": len(leaves),
        "current_projected_leaf_count": len(current),
        "historical_projected_leaf_count": len(historical),
        "projected_leaves_json": json.dumps(sorted(leaves)),
        "unresolved_projected_leaves_json": json.dumps(sorted(unresolved)),
    }
    if not unresolved:
        return {
            **common,
            "triage_class": "projected_family_all_leaves_current_exact_scope_missing",
            "recommended_action": (
                "retrieve exact-substrate execution or qualify bounded substrate scope; do not add a family"
            ),
        }
    if current:
        return {
            **common,
            "triage_class": "projected_family_partial_leaf_and_exact_scope_gap",
            "recommended_action": (
                "resolve remaining terminal identities and exact-substrate evidence within the existing family"
            ),
        }
    return {
        **common,
        "triage_class": "projected_family_leaf_and_exact_scope_gap",
        "recommended_action": (
            "resolve terminal identities and exact-substrate evidence within the existing family"
        ),
    }


def build_route_gap_triage(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the deterministic one-gap triage ledger."""

    config = _load_json(config_path, label="route-gap triage config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3RouteGapTriageError("unsupported route-gap triage config")
    policy = config.get("triage_policy")
    required_true = (
        "current_exact_target_identity_precedes_upstream_planning",
        "existing_projection_family_precedes_new_family_mining",
    )
    required_false = (
        "all_projected_leaves_current_establishes_exact_l2_scope",
        "historical_leaf_use_establishes_current_procurement",
        "parent_component_observation_establishes_projected_leaf_use",
        "missing_knowledge_is_chemical_incompatibility",
        "outside_support_is_chemical_impossibility",
        "new_reaction_family_may_be_admitted_by_this_audit",
    )
    if not isinstance(policy, dict) or any(policy.get(key) is not True for key in required_true):
        raise Ugi3RouteGapTriageError("triage positive safeguards changed")
    if any(policy.get(key) is not False for key in required_false):
        raise Ugi3RouteGapTriageError("triage nonpromotion safeguards changed")
    paths = _validate_inputs(config, repo)
    priority = _load_json(paths["priority_result"], label="priority result")
    holdout_contract = _load_json(
        paths["route_saturation_holdout_contract"], label="holdout contract"
    )
    holdout = _load_json(paths["route_saturation_holdout_seal"], label="holdout seal")
    if priority.get("adjudication", {}).get("targeted_evidence_mining_authorized") is not True:
        raise Ugi3RouteGapTriageError("priority result does not authorize evidence mining")
    if holdout.get("status") != "program_draw_sealed_molecular_holdout_not_generated":
        raise Ugi3RouteGapTriageError("route-saturation holdout is not sealed")
    admission = holdout_contract.get("reaction_family_admission_policy")
    if not isinstance(admission, dict):
        raise Ugi3RouteGapTriageError("reaction-family admission policy is missing")
    minimum_family_components = admission.get("minimum_unique_development_components")
    minimum_family_occurrences = admission.get("minimum_one_gap_product_occurrences")
    if (
        not isinstance(minimum_family_components, int)
        or minimum_family_components <= 0
        or not isinstance(minimum_family_occurrences, int)
        or minimum_family_occurrences <= 0
    ):
        raise Ugi3RouteGapTriageError("reaction-family admission thresholds are malformed")

    leaf_config = _load_json(paths["precursor_leaf_audit_config"], label="leaf audit config")
    leaf_paths = _validated_inputs(leaf_config, repo)
    assessment_raw = leaf_config.get("assessment_as_of_utc")
    expiry_days = leaf_config.get("procurement_evidence_default_expiry_days")
    if not isinstance(assessment_raw, str) or not isinstance(expiry_days, int):
        raise Ugi3RouteGapTriageError("leaf audit time policy is malformed")
    assessment: datetime = _parse_utc(assessment_raw, label="leaf assessment time")
    terminals = current_terminal_evidence(
        leaf_paths,
        assessment_as_of=assessment,
        default_expiry_days=expiry_days,
        repo=repo,
    )
    historical = historical_leaf_source_use(leaf_paths, repo=repo)

    rows = []
    triage_components: Counter[str] = Counter()
    triage_products: Counter[str] = Counter()
    family_components: Counter[str] = Counter()
    family_products: Counter[str] = Counter()
    direct_family_discovery_components = 0
    direct_family_discovery_products = 0
    for source in _read_priority_rows(paths["priority_ledger"]):
        count = int(source["one_gap_product_count"])
        triage = classify_route_gap(
            role=source["role"],
            target_smiles=source["canonical_smiles"],
            assessment_outcome=source["assessment_outcome"],
            current_terminal_keys=set(terminals),
            historical_leaf_keys=set(historical),
        )
        row = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "priority_rank": int(source["priority_rank"]),
            "role": source["role"],
            "canonical_smiles": source["canonical_smiles"],
            "assessment_outcome": source["assessment_outcome"],
            "structural_provenance_stratum": source["structural_provenance_stratum"],
            "registry_component_id": source["registry_component_id"],
            "one_gap_product_count": count,
            **triage,
        }
        rows.append(row)
        triage_components[triage["triage_class"]] += 1
        triage_products[triage["triage_class"]] += count
        family = str(triage["program_family"])
        if family:
            family_components[family] += 1
            family_products[family] += count
        if triage["triage_class"] == "no_mechanical_program_family_discovery_candidate":
            direct_family_discovery_components += 1
            direct_family_discovery_products += count

    expected_rows = int(priority.get("summary", {}).get("unique_one_gap_components", -1))
    expected_products = int(priority.get("summary", {}).get("one_gap_products", -1))
    if (
        len(rows) != expected_rows
        or sum(row["one_gap_product_count"] for row in rows) != expected_products
    ):
        raise Ugi3RouteGapTriageError("one-gap triage denominator changed")
    rows.sort(key=lambda row: int(row["priority_rank"]))
    family_discovery_threshold_met = (
        direct_family_discovery_components >= minimum_family_components
        and direct_family_discovery_products >= minimum_family_occurrences
    )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_nonpromoting_route_gap_triage",
        "task": config.get("task"),
        "inputs": {
            label: {
                "path": str(path.relative_to(repo)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for label, path in sorted(paths.items())
        },
        "summary": {
            "one_gap_components": len(rows),
            "one_gap_products": expected_products,
            "triage_components": dict(sorted(triage_components.items())),
            "triage_product_occurrences": dict(sorted(triage_products.items())),
            "projected_program_family_components": dict(sorted(family_components.items())),
            "projected_program_family_product_occurrences": dict(sorted(family_products.items())),
            "direct_family_discovery_candidate_components": direct_family_discovery_components,
            "direct_family_discovery_candidate_product_occurrences": direct_family_discovery_products,
            "reaction_family_admission_thresholds": {
                "minimum_unique_development_components": minimum_family_components,
                "minimum_one_gap_product_occurrences": minimum_family_occurrences,
                "threshold_met": family_discovery_threshold_met,
            },
        },
        "decision": {
            "broad_reaction_family_mining_indicated_now": family_discovery_threshold_met,
            "first_action": (
                "resolve current-target, terminal-leaf and exact-substrate gaps within existing "
                "program families; review outside-support targets separately"
            ),
            "new_family_admission_requires_separate_evidence_audit": True,
            "holdout_remains_unrevealed": True,
        },
        "nonclaims": [
            "A mechanical family projection is not an exact L2 route.",
            "Current leaves do not validate the projected reaction on the exact target substrate.",
            "Missing procurement evidence is not proof of unavailability.",
            "Outside support is not proof of chemical impossibility.",
            "This audit does not authorize synthesis guidance or prospective selection.",
        ],
        "artifacts": {
            "config": {
                "path": str(config_path.relative_to(repo)),
                "sha256": sha256_file(config_path),
            },
            "audit_source": {
                "path": "src/forge/route/ugi3_route_gap_triage.py",
                "sha256": sha256_file(Path(__file__)),
            },
        },
    }
    return result, rows
