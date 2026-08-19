"""Corrected, family-aware triage for one-gap Ugi route targets.

Version 1 remains frozen.  This version fixes two fail-closed policy issues:
an exact current target terminal precedes a stale route-support label, and a
reaction-family recurrence threshold is evaluated per coherent candidate
family rather than across unrelated unsupported structures.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.data.r1_prime_audit import sha256_file
from forge.route.ugi3_precursor_leaf_closure import (
    _parse_utc,
    _validated_inputs,
    constitutional_key,
    current_terminal_evidence,
    historical_leaf_source_use,
)
from forge.route.ugi3_route_gap_triage import (
    Ugi3RouteGapTriageError,
    _load_json,
    _read_priority_rows,
    _validate_inputs,
)
from forge.route.ugi3_route_gap_triage import (
    classify_route_gap as classify_route_gap_v1,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi3_route_gap_triage_config.v2"
RESULT_SCHEMA_VERSION = "phase1_ugi3_route_gap_triage.v2"
LEDGER_SCHEMA_VERSION = "phase1_ugi3_route_gap_triage_ledger.v2"

_CARBONATE_SMARTS = Chem.MolFromSmarts("[O;X2][C;X3](=[O;X1])[O;X2]")
if _CARBONATE_SMARTS is None:  # pragma: no cover - import-time RDKit invariant
    raise RuntimeError("could not construct carbonate SMARTS")


def _family_discovery_cluster(role: str, target_smiles: str) -> str:
    """Return a conservative candidate-family key without admitting chemistry."""

    target = constitutional_key(target_smiles)
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(target)
    if molecule is None:
        raise Ugi3RouteGapTriageError("candidate-family target is invalid")
    if role == "oxoester_aldehyde_body_tail" and molecule.HasSubstructMatch(_CARBONATE_SMARTS):
        return "carbonate_linked_aldehyde_program_candidate"
    digest = hashlib.sha256(f"{role}\t{target}".encode()).hexdigest()[:16]
    return f"unclustered_exact_identity_{digest}"


def classify_route_gap(
    *,
    role: str,
    target_smiles: str,
    assessment_outcome: str,
    current_terminal_keys: set[str],
    historical_leaf_keys: set[tuple[str, str]],
) -> dict[str, Any]:
    """Classify one target with corrected exact-terminal precedence."""

    target_key = constitutional_key(target_smiles)
    if target_key in current_terminal_keys:
        return {
            "target_current_terminal": True,
            "program_family": "",
            "family_discovery_cluster": "",
            "projected_leaf_count": 0,
            "current_projected_leaf_count": 0,
            "historical_projected_leaf_count": 0,
            "projected_leaves_json": "[]",
            "unresolved_projected_leaves_json": "[]",
            "new_reaction_family_indicated_now": False,
            "triage_class": "current_target_terminal_available_value_refresh",
            "recommended_action": (
                "verify role and handle identity, then admit through an exact terminal overlay"
            ),
        }
    result = classify_route_gap_v1(
        role=role,
        target_smiles=target_key,
        assessment_outcome=assessment_outcome,
        current_terminal_keys=current_terminal_keys,
        historical_leaf_keys=historical_leaf_keys,
    )
    if result["triage_class"] == "no_mechanical_program_family_discovery_candidate":
        result["family_discovery_cluster"] = _family_discovery_cluster(role, target_key)
    else:
        result["family_discovery_cluster"] = ""
    return result


def build_route_gap_triage_v2(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the corrected deterministic one-gap triage ledger."""

    config = _load_json(config_path, label="route-gap triage v2 config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3RouteGapTriageError("unsupported route-gap triage v2 config")
    policy = config.get("triage_policy")
    required_true = (
        "current_exact_target_identity_precedes_upstream_planning",
        "existing_projection_family_precedes_new_family_mining",
        "family_thresholds_apply_per_coherent_cluster",
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
        raise Ugi3RouteGapTriageError("triage v2 positive safeguards changed")
    if any(policy.get(key) is not False for key in required_false):
        raise Ugi3RouteGapTriageError("triage v2 nonpromotion safeguards changed")

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
    minimum_components = admission.get("minimum_unique_development_components")
    minimum_occurrences = admission.get("minimum_one_gap_product_occurrences")
    if (
        not isinstance(minimum_components, int)
        or minimum_components <= 0
        or not isinstance(minimum_occurrences, int)
        or minimum_occurrences <= 0
    ):
        raise Ugi3RouteGapTriageError("reaction-family thresholds are malformed")

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

    rows: list[dict[str, Any]] = []
    triage_components: Counter[str] = Counter()
    triage_products: Counter[str] = Counter()
    family_components: Counter[str] = Counter()
    family_products: Counter[str] = Counter()
    discovery_components: Counter[str] = Counter()
    discovery_products: Counter[str] = Counter()
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
        cluster = str(triage["family_discovery_cluster"])
        if cluster:
            discovery_components[cluster] += 1
            discovery_products[cluster] += count

    expected_rows = int(priority.get("summary", {}).get("unique_one_gap_components", -1))
    expected_products = int(priority.get("summary", {}).get("one_gap_products", -1))
    if (
        len(rows) != expected_rows
        or sum(int(row["one_gap_product_count"]) for row in rows) != expected_products
    ):
        raise Ugi3RouteGapTriageError("one-gap triage denominator changed")
    rows.sort(key=lambda row: int(row["priority_rank"]))

    candidate_clusters = {}
    for cluster in sorted(discovery_components):
        eligible = (
            discovery_components[cluster] >= minimum_components
            and discovery_products[cluster] >= minimum_occurrences
        )
        candidate_clusters[cluster] = {
            "unique_development_components": discovery_components[cluster],
            "one_gap_product_occurrences": discovery_products[cluster],
            "threshold_met": eligible,
        }
    threshold_met = any(item["threshold_met"] for item in candidate_clusters.values())
    direct_components = sum(discovery_components.values())
    direct_products = sum(discovery_products.values())

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_corrected_nonpromoting_route_gap_triage",
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
            "direct_family_discovery_candidate_components": direct_components,
            "direct_family_discovery_candidate_product_occurrences": direct_products,
            "candidate_family_clusters": candidate_clusters,
            "reaction_family_admission_thresholds": {
                "minimum_unique_development_components_per_family": minimum_components,
                "minimum_one_gap_product_occurrences_per_family": minimum_occurrences,
                "any_candidate_family_threshold_met": threshold_met,
            },
        },
        "decision": {
            "broad_reaction_family_mining_indicated_now": threshold_met,
            "first_action": (
                "refresh current exact targets, then resolve terminal-leaf and exact-substrate "
                "gaps within existing programs; review outside-support targets separately"
            ),
            "new_family_admission_requires_separate_evidence_audit": True,
            "holdout_remains_unrevealed": True,
        },
        "correction_from_v1": {
            "current_terminal_precedence_applied": True,
            "family_thresholds_evaluated_per_coherent_cluster": True,
        },
        "nonclaims": [
            "A mechanical family projection is not an exact L2 route.",
            "Current leaves do not validate the projected reaction on the exact target substrate.",
            "Missing procurement evidence is not proof of unavailability.",
            "Outside support is not proof of chemical impossibility.",
            "A candidate-family cluster is not an admitted reaction family.",
            "This audit does not authorize synthesis guidance or prospective selection.",
        ],
        "artifacts": {
            "config": {
                "path": str(config_path.relative_to(repo)),
                "sha256": sha256_file(config_path),
            },
            "audit_source": {
                "path": "src/forge/route/ugi3_route_gap_triage_v2.py",
                "sha256": sha256_file(Path(__file__)),
            },
        },
    }
    return result, rows
