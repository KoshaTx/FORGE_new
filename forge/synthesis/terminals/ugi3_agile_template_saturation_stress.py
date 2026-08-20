"""Stress-test whether AGILE structural templates can be attached by construction.

This deliberately optimistic audit asks whether every admitted tail graph can
be mechanically inverted through the published AGILE aldehyde or isocyanide
program. It does not treat the resulting leaves as available, qualify broader
substrate scope, or infer experimental success.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_file
from forge.core.io import read_json_object
from forge.synthesis.terminals.ugi3_virtual_programs import _aldehyde_program, _isocyanide_program

CONFIG_SCHEMA_VERSION = "phase1_ugi3_agile_template_saturation_stress_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi3_agile_template_saturation_stress.v1"

HEAD_ROLE = "amine_head"
ALDEHYDE_ROLE = "oxoester_aldehyde_body_tail"
ISOCYANIDE_ROLE = "isocyanide_tail"
ROLES = (HEAD_ROLE, ALDEHYDE_ROLE, ISOCYANIDE_ROLE)


class Ugi3AgileTemplateSaturationStressError(ValueError):
    """Raised when the optimistic template stress test cannot be reproduced."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    return read_json_object(path, error=Ugi3AgileTemplateSaturationStressError, label=label)


def _load_gzip_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with gzip.open(path, "rt") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise Ugi3AgileTemplateSaturationStressError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Ugi3AgileTemplateSaturationStressError(f"{label} must be an object")
    return value


def _read_csv(path: Path, *, label: str) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise Ugi3AgileTemplateSaturationStressError(f"{label} has no header")
            return list(reader)
    except (OSError, csv.Error) as exc:
        raise Ugi3AgileTemplateSaturationStressError(f"could not read {label}") from exc


def _validated_inputs(config: Mapping[str, Any], repo: Path) -> dict[str, Path]:
    specifications = config.get("inputs")
    if not isinstance(specifications, dict) or not specifications:
        raise Ugi3AgileTemplateSaturationStressError("stress-test inputs are missing")
    paths: dict[str, Path] = {}
    for label, specification in specifications.items():
        if not isinstance(specification, dict) or set(specification) != {"path", "sha256"}:
            raise Ugi3AgileTemplateSaturationStressError(
                f"input {label!r} must define path and sha256"
            )
        path = Path(str(specification["path"]))
        if not path.is_absolute():
            path = repo / path
        observed = sha256_file(path)
        if observed != str(specification["sha256"]):
            raise Ugi3AgileTemplateSaturationStressError(
                f"input hash changed for {label}: expected {specification['sha256']}, "
                f"observed {observed}"
            )
        paths[label] = path
    return paths


def projected_leaf_candidates(steps: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Return proposed upstream leaves without treating them as terminals."""

    products = {str(step["product"]) for step in steps}
    leaves = {
        str(reactant)
        for step in steps
        for reactant in step["reactants"]
        if str(reactant) not in products
    }
    return tuple(sorted(leaves))


def _accepted_heads(hybrid: Mapping[str, Any]) -> set[str]:
    records = hybrid.get("records")
    if not isinstance(records, list):
        raise Ugi3AgileTemplateSaturationStressError("hybrid ledger lacks records")
    return {
        str(record["canonical_smiles"])
        for record in records
        if isinstance(record, dict)
        and record.get("role") == HEAD_ROLE
        and record.get("hybrid_outcome") == "complete"
    }


def _project_tail_components(
    rows: Sequence[Mapping[str, str]],
) -> tuple[dict[str, set[str]], Counter[tuple[str, str]], dict[str, Counter[int]]]:
    projected: dict[str, set[str]] = {ALDEHYDE_ROLE: set(), ISOCYANIDE_ROLE: set()}
    families: Counter[tuple[str, str]] = Counter()
    leaf_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for row in rows:
        if row.get("l1_structural_admission") != "true":
            continue
        role = str(row.get("role"))
        target = str(row.get("canonical_smiles"))
        if role == ALDEHYDE_ROLE:
            family, steps = _aldehyde_program(target)
        elif role == ISOCYANIDE_ROLE:
            family, steps = _isocyanide_program(target)
        else:
            continue
        leaves = projected_leaf_candidates(steps)
        if not leaves:
            raise Ugi3AgileTemplateSaturationStressError(
                f"projected {role} component has no upstream leaf: {target}"
            )
        projected[role].add(target)
        families[(role, family)] += 1
        leaf_counts[role][len(leaves)] += 1
    return projected, families, leaf_counts


def _unique_projected_leaves(
    rows: Sequence[Mapping[str, str]],
) -> dict[str, set[str]]:
    leaves: dict[str, set[str]] = {ALDEHYDE_ROLE: set(), ISOCYANIDE_ROLE: set()}
    for row in rows:
        if row.get("l1_structural_admission") != "true":
            continue
        role = str(row.get("role"))
        target = str(row.get("canonical_smiles"))
        if role == ALDEHYDE_ROLE:
            _, steps = _aldehyde_program(target)
        elif role == ISOCYANIDE_ROLE:
            _, steps = _isocyanide_program(target)
        else:
            continue
        leaves[role].update(projected_leaf_candidates(steps))
    return leaves


def _product_stress_counts(
    path: Path,
    *,
    projected: Mapping[str, set[str]],
    accepted_heads: set[str],
) -> dict[str, int]:
    counts = Counter()
    try:
        with gzip.open(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "amine_head_smiles",
                "oxoester_aldehyde_body_tail_smiles",
                "isocyanide_tail_smiles",
            }
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise Ugi3AgileTemplateSaturationStressError(
                    "expanded product ledger lacks required component columns"
                )
            for row_number, row in enumerate(reader, start=2):
                if row["oxoester_aldehyde_body_tail_smiles"] not in projected[ALDEHYDE_ROLE]:
                    raise Ugi3AgileTemplateSaturationStressError(
                        f"row {row_number} aldehyde lacks a structural program"
                    )
                if row["isocyanide_tail_smiles"] not in projected[ISOCYANIDE_ROLE]:
                    raise Ugi3AgileTemplateSaturationStressError(
                        f"row {row_number} isocyanide lacks a structural program"
                    )
                counts["products_with_two_tail_structural_programs"] += 1
                if row["amine_head_smiles"] in accepted_heads:
                    counts["products_with_accepted_head_and_two_tail_structural_programs"] += 1
    except (OSError, csv.Error) as exc:
        raise Ugi3AgileTemplateSaturationStressError(
            "could not stream expanded product ledger"
        ) from exc
    return dict(counts)


def build_agile_template_saturation_stress(repo: Path, config_path: Path) -> dict[str, Any]:
    """Build the optimistic structural-template saturation stress test."""

    config = _load_json(config_path, label="AGILE template stress config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3AgileTemplateSaturationStressError("unsupported stress config")
    policy = config.get("stress_policy")
    if not isinstance(policy, dict) or any(
        policy.get(field) is not False
        for field in (
            "structural_program_is_route_closure",
            "family_projection_is_exact_evidence",
            "projected_leaf_is_terminal",
            "structural_program_establishes_substrate_scope",
            "structural_program_establishes_ugi_success",
            "unresolved_is_impossible",
        )
    ):
        raise Ugi3AgileTemplateSaturationStressError("stress-test safeguards changed")
    paths = _validated_inputs(config, repo)

    enumeration = _load_json(paths["expanded_enumeration_result"], label="enumeration result")
    hybrid_result = _load_json(paths["hybrid_result"], label="hybrid result")
    routeability = _load_json(paths["routeability_census"], label="routeability census")
    original_program = _load_json(
        paths["original_agile_program_result"], label="original AGILE program result"
    )
    hybrid = _load_gzip_json(paths["hybrid_assessment_ledger"], label="hybrid ledger")
    if hybrid_result.get("artifacts", {}).get("assessment_ledger_sha256") != sha256_file(
        paths["hybrid_assessment_ledger"]
    ):
        raise Ugi3AgileTemplateSaturationStressError("hybrid ledger ownership failed")
    if original_program.get("summary", {}).get("components_with_structural_program") != 71:
        raise Ugi3AgileTemplateSaturationStressError("original AGILE program anchor changed")

    rows = _read_csv(paths["component_registry"], label="component registry")
    admitted_counts = Counter(
        row["role"] for row in rows if row.get("l1_structural_admission") == "true"
    )
    if dict(admitted_counts) != enumeration.get("summary", {}).get("admitted_components"):
        raise Ugi3AgileTemplateSaturationStressError("admitted component census changed")

    projected, families, leaf_counts = _project_tail_components(rows)
    unique_leaves = _unique_projected_leaves(rows)
    accepted_heads = _accepted_heads(hybrid)
    product_counts = _product_stress_counts(
        paths["expanded_products"], projected=projected, accepted_heads=accepted_heads
    )
    product_count = int(enumeration["summary"]["unique_products"])
    if product_counts["products_with_two_tail_structural_programs"] != product_count:
        raise Ugi3AgileTemplateSaturationStressError("tail structural programs did not saturate")

    strict_exact = int(routeability["operational_envelope"]["strict_exact_route_complete_products"])
    family_payload = {
        f"{role}|{family}": count for (role, family), count in sorted(families.items())
    }
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_optimistic_agile_template_saturation_stress",
        "task": config.get("task"),
        "inputs": {
            label: {
                "path": str(path.relative_to(repo)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for label, path in sorted(paths.items())
        },
        "component_census": {
            "admitted_components_by_role": dict(admitted_counts),
            "aldehydes_with_mechanical_agile_program": len(projected[ALDEHYDE_ROLE]),
            "isocyanides_with_mechanical_agile_program": len(projected[ISOCYANIDE_ROLE]),
            "accepted_head_terminals": len(accepted_heads),
            "program_family_counts": family_payload,
            "projected_leaf_count_per_component": {
                role: {str(count): occurrences for count, occurrences in sorted(values.items())}
                for role, values in sorted(leaf_counts.items())
            },
            "unique_projected_tail_leaf_candidates": {
                role: len(values) for role, values in sorted(unique_leaves.items())
            },
            "total_unique_projected_tail_leaf_candidates_by_role_sum": sum(
                len(values) for values in unique_leaves.values()
            ),
        },
        "product_census": {
            "expanded_unique_products": product_count,
            **product_counts,
            "fraction_with_two_tail_structural_programs": (
                product_counts["products_with_two_tail_structural_programs"] / product_count
            ),
            "fraction_with_accepted_head_and_two_tail_structural_programs": (
                product_counts["products_with_accepted_head_and_two_tail_structural_programs"]
                / product_count
            ),
            "strict_exact_route_complete_products": strict_exact,
            "strict_exact_route_complete_fraction": strict_exact / product_count,
        },
        "counterfactual": {
            "condition": (
                "If all 264 head leaves and all 122 distinct projected tail leaves were genuinely "
                "terminally closed, every enumerated product would have a component-level "
                "AGILE-template program."
            ),
            "products_with_syntactic_component_program_under_condition": product_count,
            "establishes_exact_route_closure": False,
            "establishes_ugi_product_isolation": False,
        },
        "decision": {
            "route_template_assignment_alone_is_discriminative": False,
            "recursive_leaf_closure_is_load_bearing": True,
            "beyond_catalog_components_are_load_bearing_for_open_endedness": True,
            "required_baseline": (
                "AGILE-template-saturated post-hoc routing with identical terminal closure, "
                "uncertainty and compute accounting"
            ),
        },
        "nonclaims": [
            "Mechanical inversion through an AGILE structural program is not route closure.",
            "The 122 projected tail leaves are not assumed purchasable or synthesizable.",
            "The stress test does not qualify substrate scope for any unexecuted analogue.",
            "The stress test does not predict Ugi conversion, isolated yield or purification success.",
            "Failure to possess current evidence is not proof of chemical impossibility.",
            "The finite 424-component universe is not the full atom-level generator support.",
        ],
        "artifacts": {
            "config": {
                "path": str(config_path.relative_to(repo)),
                "sha256": sha256_file(config_path),
            },
            "audit_source": {
                "path": "src/forge/route/ugi3_agile_template_saturation_stress.py",
                "sha256": sha256_file(Path(__file__)),
            },
        },
    }
