"""Intersect the exact expanded Ugi universe with frozen route evidence.

The census quantifies operational route support, not existential chemical
synthesizability. Family projections remain proposals, missing knowledge is not
an experimental failure, and absence from the current planner support is not
chemical impossibility.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

from forge.core.hashing import sha256_file

CONFIG_SCHEMA_VERSION = "phase1_ugi3_enumerated_routeability_census_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi3_enumerated_routeability_census.v1"

ROLE_COLUMNS = (
    ("amine_head", "amine_head_smiles"),
    ("oxoester_aldehyde_body_tail", "oxoester_aldehyde_body_tail_smiles"),
    ("isocyanide_tail", "isocyanide_tail_smiles"),
)

EXACT_COMPLETE = "exact_route_complete"
FAMILY_PROJECTED = "family_projected_only"
EXACT_L3_OPEN = "exact_route_l3_open"
MISSING_KNOWLEDGE = "missing_route_knowledge"
OUTSIDE_SUPPORT = "outside_current_route_support"

PRODUCT_CLASSES = (
    "all_exact_route_complete",
    "family_projected_no_other_gap",
    "exact_route_l3_open_no_family_projection",
    "family_projected_plus_l3_open",
    "missing_route_knowledge_present",
    "outside_current_route_support_present",
)


class Ugi3EnumeratedRouteabilityCensusError(ValueError):
    """Raised when the frozen census cannot be reproduced exactly."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise Ugi3EnumeratedRouteabilityCensusError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Ugi3EnumeratedRouteabilityCensusError(f"{label} must be an object")
    return value


def _load_gzip_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with gzip.open(path, "rt") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise Ugi3EnumeratedRouteabilityCensusError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Ugi3EnumeratedRouteabilityCensusError(f"{label} must be an object")
    return value


def _validated_inputs(config: Mapping[str, Any], repo: Path) -> dict[str, Path]:
    specifications = config.get("inputs")
    if not isinstance(specifications, dict) or not specifications:
        raise Ugi3EnumeratedRouteabilityCensusError("census inputs are missing")
    paths: dict[str, Path] = {}
    for label, specification in specifications.items():
        if not isinstance(specification, dict) or set(specification) != {"path", "sha256"}:
            raise Ugi3EnumeratedRouteabilityCensusError(
                f"input {label!r} must define path and sha256"
            )
        path = Path(str(specification["path"]))
        if not path.is_absolute():
            path = repo / path
        observed = sha256_file(path)
        if observed != str(specification["sha256"]):
            raise Ugi3EnumeratedRouteabilityCensusError(
                f"input hash changed for {label}: expected {specification['sha256']}, "
                f"observed {observed}"
            )
        paths[label] = path
    return paths


def component_route_tier(record: Mapping[str, Any]) -> str:
    """Map one frozen hybrid assessment to an evidence-bounded route tier."""

    outcome = record.get("hybrid_outcome")
    channel = record.get("search_channel")
    if outcome == "complete":
        return EXACT_COMPLETE
    if outcome == "missing_knowledge" and channel == "family_projection":
        return FAMILY_PROJECTED
    if outcome == "missing_knowledge" and channel == "exact_evidence_l3_open":
        return EXACT_L3_OPEN
    if outcome == "missing_knowledge":
        return MISSING_KNOWLEDGE
    if outcome == "outside_support":
        return OUTSIDE_SUPPORT
    raise Ugi3EnumeratedRouteabilityCensusError(
        f"unsupported hybrid outcome/channel: {outcome!r}/{channel!r}"
    )


def classify_product_routeability(component_tiers: tuple[str, str, str]) -> str:
    """Return the most conservative product-level routeability class."""

    tiers = set(component_tiers)
    if OUTSIDE_SUPPORT in tiers:
        return "outside_current_route_support_present"
    if MISSING_KNOWLEDGE in tiers:
        return "missing_route_knowledge_present"
    if FAMILY_PROJECTED in tiers and EXACT_L3_OPEN in tiers:
        return "family_projected_plus_l3_open"
    if EXACT_L3_OPEN in tiers:
        return "exact_route_l3_open_no_family_projection"
    if FAMILY_PROJECTED in tiers:
        return "family_projected_no_other_gap"
    if tiers == {EXACT_COMPLETE}:
        return "all_exact_route_complete"
    raise Ugi3EnumeratedRouteabilityCensusError(
        f"unclassified component-tier combination: {component_tiers!r}"
    )


def _component_tiers(
    hybrid_payload: Mapping[str, Any],
) -> tuple[dict[tuple[str, str], str], dict[str, Counter[str]]]:
    records = hybrid_payload.get("records")
    if not isinstance(records, list):
        raise Ugi3EnumeratedRouteabilityCensusError("hybrid ledger lacks records")
    tiers: dict[tuple[str, str], str] = {}
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    expected_roles = {role for role, _ in ROLE_COLUMNS}
    for record in records:
        if not isinstance(record, dict):
            raise Ugi3EnumeratedRouteabilityCensusError("malformed hybrid record")
        role = str(record.get("role"))
        smiles = str(record.get("canonical_smiles"))
        if role not in expected_roles or not smiles:
            raise Ugi3EnumeratedRouteabilityCensusError("invalid hybrid component identity")
        key = (role, smiles)
        if key in tiers:
            raise Ugi3EnumeratedRouteabilityCensusError("duplicate hybrid component identity")
        tier = component_route_tier(record)
        tiers[key] = tier
        counts[role][tier] += 1
    return tiers, counts


def _tier_combination_label(tiers: tuple[str, str, str]) -> str:
    return "|".join(f"{role}={tier}" for (role, _), tier in zip(ROLE_COLUMNS, tiers, strict=True))


def _fraction_payload(counts: Counter[str], denominator: int) -> dict[str, dict[str, Any]]:
    if denominator <= 0:
        raise Ugi3EnumeratedRouteabilityCensusError("product denominator must be positive")
    return {
        label: {"count": int(counts.get(label, 0)), "fraction": counts.get(label, 0) / denominator}
        for label in PRODUCT_CLASSES
    }


def _audit_product_rows(
    handle: TextIO,
    tiers: Mapping[tuple[str, str], str],
) -> dict[str, Any]:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
        raise Ugi3EnumeratedRouteabilityCensusError("expanded product ledger has no header")
    required = {
        *(column for _, column in ROLE_COLUMNS),
        "component_novelty_class",
        "canonical_product_smiles",
    }
    missing = required - set(reader.fieldnames)
    if missing:
        raise Ugi3EnumeratedRouteabilityCensusError(
            f"expanded product ledger lacks columns: {sorted(missing)}"
        )

    product_classes: Counter[str] = Counter()
    tier_combinations: Counter[str] = Counter()
    novelty_classes: dict[str, Counter[str]] = defaultdict(Counter)
    product_count = 0
    for row_number, row in enumerate(reader, start=2):
        component_tiers = []
        for role, column in ROLE_COLUMNS:
            key = (role, row[column])
            try:
                component_tiers.append(tiers[key])
            except KeyError as exc:
                raise Ugi3EnumeratedRouteabilityCensusError(
                    f"product row {row_number} has unmapped component {key!r}"
                ) from exc
        tier_tuple = tuple(component_tiers)
        if len(tier_tuple) != 3:
            raise AssertionError("three Ugi roles are required")
        product_class = classify_product_routeability(tier_tuple)  # type: ignore[arg-type]
        product_classes[product_class] += 1
        tier_combinations[_tier_combination_label(tier_tuple)] += 1  # type: ignore[arg-type]
        novelty_classes[row["component_novelty_class"]][product_class] += 1
        product_count += 1

    return {
        "product_count": product_count,
        "product_classes": product_classes,
        "tier_combinations": tier_combinations,
        "novelty_classes": novelty_classes,
    }


def build_enumerated_routeability_census(repo: Path, config_path: Path) -> dict[str, Any]:
    """Build the frozen routeability census over all enumerated products."""

    config = _load_json(config_path, label="routeability census config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3EnumeratedRouteabilityCensusError("unsupported routeability census config")
    policy = config.get("policy")
    if not isinstance(policy, dict) or any(
        policy.get(field) is not False
        for field in (
            "family_projection_is_exact_route_evidence",
            "missing_knowledge_is_incompatibility",
            "outside_current_route_support_is_chemical_impossibility",
            "enumerated_product_is_experimental_success",
            "similarity_can_transfer_route_status",
        )
    ):
        raise Ugi3EnumeratedRouteabilityCensusError("nonclaim safeguards changed")
    paths = _validated_inputs(config, repo)
    required_inputs = {
        "expanded_products",
        "expanded_enumeration_result",
        "hybrid_assessment_ledger",
        "hybrid_result",
    }
    if set(paths) != required_inputs:
        raise Ugi3EnumeratedRouteabilityCensusError("routeability input set changed")

    enumeration = _load_json(paths["expanded_enumeration_result"], label="enumeration result")
    hybrid_result = _load_json(paths["hybrid_result"], label="hybrid result")
    hybrid_payload = _load_gzip_json(paths["hybrid_assessment_ledger"], label="hybrid ledger")
    if enumeration.get("status") != "complete_exact_expanded_ugi_enumeration":
        raise Ugi3EnumeratedRouteabilityCensusError("expanded enumeration is not complete")
    if hybrid_result.get("artifacts", {}).get("assessment_ledger_sha256") != sha256_file(
        paths["hybrid_assessment_ledger"]
    ):
        raise Ugi3EnumeratedRouteabilityCensusError("hybrid ledger ownership failed")

    tiers, component_counts = _component_tiers(hybrid_payload)
    expected_components = enumeration.get("summary", {}).get("admitted_components")
    observed_components = {
        role: sum(component_counts.get(role, Counter()).values()) for role, _ in ROLE_COLUMNS
    }
    if observed_components != expected_components:
        raise Ugi3EnumeratedRouteabilityCensusError(
            f"hybrid component census changed: {observed_components} != {expected_components}"
        )

    try:
        with gzip.open(paths["expanded_products"], "rt", newline="") as handle:
            audit = _audit_product_rows(handle, tiers)
    except (OSError, csv.Error) as exc:
        raise Ugi3EnumeratedRouteabilityCensusError("could not audit expanded products") from exc

    product_count = int(audit["product_count"])
    expected_products = int(enumeration.get("summary", {}).get("unique_products", -1))
    if product_count != expected_products:
        raise Ugi3EnumeratedRouteabilityCensusError(
            f"expanded product denominator changed: {product_count} != {expected_products}"
        )
    product_classes: Counter[str] = audit["product_classes"]
    if sum(product_classes.values()) != product_count:
        raise Ugi3EnumeratedRouteabilityCensusError("product classes do not partition universe")

    exact = product_classes["all_exact_route_complete"]
    projected = product_classes["family_projected_no_other_gap"]
    l3_open = product_classes["exact_route_l3_open_no_family_projection"]
    projected_l3 = product_classes["family_projected_plus_l3_open"]
    operational_envelope = exact + projected + l3_open + projected_l3

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_evidence_bounded_enumerated_routeability_census",
        "task": config.get("task"),
        "inputs": {
            label: {
                "path": str(path.relative_to(repo)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for label, path in sorted(paths.items())
        },
        "universe": {
            "interpretation": policy.get("universe"),
            "raw_component_cartesian_upper_bound": (
                enumeration["summary"]["raw_cartesian_upper_bound"]
                if "raw_cartesian_upper_bound" in enumeration["summary"]
                else 264 * 107 * 53
            ),
            "attempted_single_site_triples": enumeration["summary"]["attempted_expanded_triples"],
            "actual_unique_products": product_count,
            "not_the_full_atom_level_support": True,
        },
        "component_route_tiers_by_role": {
            role: dict(sorted(component_counts.get(role, Counter()).items()))
            for role, _ in ROLE_COLUMNS
        },
        "product_routeability_classes": _fraction_payload(product_classes, product_count),
        "operational_envelope": {
            "strict_exact_route_complete_products": exact,
            "exact_or_family_projected_products_without_l3_open": exact + projected,
            "exact_or_family_projected_products_including_one_l3_open_component": operational_envelope,
            "fraction_including_l3_open": operational_envelope / product_count,
            "interpretation": (
                "This is an evidence-bounded lower envelope of currently encoded route programs, "
                "not a probability of successful synthesis and not an upper bound on chemistry."
            ),
        },
        "route_tier_combination_counts": dict(sorted(audit["tier_combinations"].items())),
        "product_classes_by_component_novelty": {
            novelty: _fraction_payload(counts, sum(counts.values()))
            for novelty, counts in sorted(audit["novelty_classes"].items())
        },
        "interpretation": {
            "synthesis_guidance_is_nonvacuous_under_current_evidence": exact < product_count,
            "existential_synthesizability_of_unresolved_products_is_determined": False,
            "why": (
                "Most unresolved products contain a component with missing route knowledge or no "
                "admitted route support. Those states justify uncertainty-aware route guidance but "
                "do not establish chemical impossibility."
            ),
        },
        "nonclaims": [
            "The 1.49-million-product universe is a finite exact component expansion, not every graph the atom-level generator can represent.",
            "Family-projected programs are not exact source-executed L2 routes.",
            "Missing route knowledge is not an experimental failure or chemical incompatibility.",
            "Outside current route support is not proof that a product is impossible to synthesize.",
            "Forward-consistent enumeration is not evidence that an exact lipid has been experimentally isolated.",
            "No synthesis-success probability is estimated by this census.",
        ],
        "artifacts": {
            "config": {
                "path": str(config_path.relative_to(repo)),
                "sha256": sha256_file(config_path),
            },
            "audit_source": {
                "path": "src/forge/route/ugi3_enumerated_routeability_census.py",
                "sha256": sha256_file(Path(__file__)),
            },
        },
    }
