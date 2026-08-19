"""Evidence-safe route-readiness audit for the matched Ugi panel.

The learned proposal engine broadens route *search*.  Independent exact and
family-projected evidence determines route readiness.  The ordinal utility in
this module is a frozen controller coordinate, not a probability of synthesis
success and not a promotion of family projections to exact route dossiers.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_SCHEMA_VERSION = "phase1_ugi_proposal_augmented_route_readiness_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_proposal_augmented_route_readiness.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_proposal_augmented_route_readiness_ledger.v1"

EXACT = "exact_complete_current"
FAMILY_ALL = "family_projected_all_current_leaves"
FAMILY_PARTIAL = "family_projected_partial_current_leaves"
UNRESOLVED = "unresolved_or_unsupported"
READINESS_ORDER = (UNRESOLVED, FAMILY_PARTIAL, FAMILY_ALL, EXACT)
READINESS_UTILITY = {
    UNRESOLVED: 0.0,
    FAMILY_PARTIAL: 0.25,
    FAMILY_ALL: 0.75,
    EXACT: 1.0,
}


class UgiProposalAugmentedRouteReadinessError(RuntimeError):
    """Raised when frozen route-readiness inputs or semantics change."""


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise UgiProposalAugmentedRouteReadinessError(f"JSON object required: {path}")
    return value


def _read_jsonl_gzip(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise UgiProposalAugmentedRouteReadinessError(f"JSONL rows are malformed: {path}")
    return rows


def _read_csv_gzip(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _gzip_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
        handle.write(b"".join((_stable_json(dict(row)) + "\n").encode() for row in rows))
    return raw.getvalue()


def component_readiness_class(graded_evidence_class: str) -> str:
    """Map typed evidence to the frozen ordinal controller class."""

    if graded_evidence_class == EXACT:
        return EXACT
    if graded_evidence_class == FAMILY_ALL:
        return FAMILY_ALL
    if graded_evidence_class == FAMILY_PARTIAL:
        return FAMILY_PARTIAL
    return UNRESOLVED


def product_readiness_class(component_classes: Sequence[str]) -> str:
    """Require every Ugi component to meet the returned readiness tier."""

    if len(component_classes) != 3 or any(
        value not in READINESS_ORDER for value in component_classes
    ):
        raise UgiProposalAugmentedRouteReadinessError(
            "product readiness requires three typed component classes"
        )
    return min(component_classes, key=READINESS_ORDER.index)


def _proposal_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    order = {
        "known_family_forward_consistent_projection": 3,
        "semantically_equivalent_known_family_projection": 3,
        "exact_known_route": 4,
        "new_family_hypothesis_retained": 1,
        "ambiguous": 0,
        "rejected": 0,
    }
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["role"]), str(row["target_smiles"]))
        semantic = row.get("semantic_equivalence")
        if not isinstance(semantic, Mapping):
            raise UgiProposalAugmentedRouteReadinessError("semantic proposal receipt is missing")
        status = str(semantic["semantic_resolution_status"])
        candidate = {
            "status": status,
            "rank": int(row["rank"]),
            "reactants": list(row["reactants"]),
            "proposal_sha256": str(row["proposal_sha256"]),
            "graph_consistent_discovery_hypothesis": bool(
                row["graph_consistent_discovery_hypothesis"]
            ),
        }
        current = output.get(key)
        if current is None or (order.get(status, 0), -candidate["rank"]) > (
            order.get(str(current["status"]), 0),
            -int(current["rank"]),
        ):
            output[key] = candidate
    return output


def build_proposal_augmented_route_readiness(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], bytes]:
    """Join panel terminals, independent evidence and proposal discovery."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiProposalAugmentedRouteReadinessError("unsupported config schema")
    paths: dict[str, Path] = {}
    for label, record in config["inputs"].items():
        path = repo / str(record["path"])
        if _sha256_file(path) != record["sha256"]:
            raise UgiProposalAugmentedRouteReadinessError(f"input hash changed: {label}")
        paths[str(label)] = path

    graded_rows = _read_csv_gzip(paths["graded_evidence_ledger"])
    graded = {(row["role"], row["canonical_smiles"]): row for row in graded_rows}
    if len(graded) != len(graded_rows):
        raise UgiProposalAugmentedRouteReadinessError("graded component identities collide")
    proposal_rows = _read_jsonl_gzip(paths["semantic_proposal_ledger"])
    proposals = _proposal_index(proposal_rows)
    route_rows = _read_jsonl_gzip(paths["route_assessment_ledger"])
    output_rows: list[dict[str, Any]] = []
    by_arm_class: dict[str, Counter[str]] = defaultdict(Counter)
    proposal_by_role: dict[str, Counter[str]] = defaultdict(Counter)
    unique_by_arm_class: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for source in route_rows:
        components = source.get("canonical_components")
        if not isinstance(components, Mapping) or len(components) != 3:
            raise UgiProposalAugmentedRouteReadinessError(
                "route row does not contain three canonical components"
            )
        strict_by_role = {
            str(record["role"]): bool(record["strict_complete"])
            for record in source.get("potential", {}).get("roles", [])
        }
        if set(strict_by_role) != set(components):
            raise UgiProposalAugmentedRouteReadinessError(
                "route row strict-role assessment is incomplete"
            )
        component_records = []
        component_classes = []
        for role, smiles in sorted(components.items()):
            evidence = graded.get((str(role), str(smiles)))
            if evidence is None:
                evidence_class = "not_in_graded_development_ledger"
                readiness = UNRESOLVED
            else:
                evidence_class = str(evidence["graded_evidence_class"])
                readiness = component_readiness_class(evidence_class)
            strict_assessment_override = strict_by_role[str(role)] and readiness != EXACT
            if strict_by_role[str(role)]:
                readiness = EXACT
            proposal = proposals.get((str(role), str(smiles)))
            proposal_status = "not_proposed" if proposal is None else str(proposal["status"])
            proposal_by_role[str(role)][proposal_status] += 1
            component_records.append(
                {
                    "role": role,
                    "canonical_smiles": smiles,
                    "graded_evidence_class": evidence_class,
                    "route_readiness_class": readiness,
                    "program_family": None if evidence is None else evidence["program_family"],
                    "projection_leaf_status": (
                        None if evidence is None else evidence["projection_leaf_status"]
                    ),
                    "proposal_discovery": proposal,
                    "proposal_source_sets_readiness": False,
                    "strict_candidate_assessment_overrides_older_graded_metadata": (
                        strict_assessment_override
                    ),
                }
            )
            component_classes.append(readiness)
        product_class = product_readiness_class(component_classes)
        if bool(source["route_complete"]) != (product_class == EXACT):
            raise UgiProposalAugmentedRouteReadinessError(
                "strict product assessment and joined readiness disagree"
            )
        product_smiles = str(source["canonical_product"])
        arm = str(source["arm_id"])
        by_arm_class[arm][product_class] += 1
        unique_by_arm_class[arm][product_class].add(product_smiles)
        output_rows.append(
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "arm_id": arm,
                "draw_index": source["draw_index"],
                "pattern_id": source["pattern_id"],
                "canonical_product": product_smiles,
                "terminal_sha256": source["terminal_sha256"],
                "strict_exact_route_complete": bool(source["route_complete"]),
                "route_readiness_class": product_class,
                "route_readiness_utility": READINESS_UTILITY[product_class],
                "components": component_records,
                "proposal_model_score_used": False,
                "synthesis_success_probability": None,
            }
        )
    output_rows.sort(
        key=lambda row: (str(row["arm_id"]), int(row["draw_index"]), str(row["terminal_sha256"]))
    )
    ledger = _gzip_jsonl(output_rows)
    arm_summary: dict[str, Any] = {}
    for arm in sorted(by_arm_class):
        unique_exact = unique_by_arm_class[arm][EXACT]
        unique_family_all = unique_exact | unique_by_arm_class[arm][FAMILY_ALL]
        unique_partial = unique_family_all | unique_by_arm_class[arm][FAMILY_PARTIAL]
        arm_summary[arm] = {
            "rows_by_route_readiness_class": dict(by_arm_class[arm]),
            "unique_products_exact_complete": len(unique_exact),
            "unique_products_family_all_current_or_better": len(unique_family_all),
            "unique_products_family_partial_or_better": len(unique_partial),
        }
    balanced_all = min(
        summary["unique_products_family_all_current_or_better"] for summary in arm_summary.values()
    )
    balanced_partial = min(
        summary["unique_products_family_partial_or_better"] for summary in arm_summary.values()
    )
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "proposal_augmented_evidence_safe_route_readiness_complete",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": _sha256_file(config_path),
        },
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": _sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "policy": {
            "declared_order": list(READINESS_ORDER),
            "ordinal_utility": READINESS_UTILITY,
            "product_aggregation": "minimum readiness across the three Ugi components",
            "proposal_model_score_used": False,
            "proposal_discovery_alone_may_set_readiness": False,
            "family_projected_is_exact_route_complete": False,
            "utility_is_synthesis_success_probability": False,
        },
        "summary": {
            "route_rows": len(output_rows),
            "arms": arm_summary,
            "balanced_unique_products_family_all_current_or_better": balanced_all,
            "balanced_unique_products_family_partial_or_better": balanced_partial,
            "proposal_status_by_role_occurrence": {
                role: dict(counts) for role, counts in sorted(proposal_by_role.items())
            },
        },
        "decision": {
            "graded_route_readiness_available_for_controller_qualification": True,
            "strict_exact_route_closure_replaced": False,
            "proposal_only_hypotheses_authorized_for_guidance": False,
            "synthesis_tilting_promoted": False,
            "next_gate": (
                "frozen matched graded-readiness guidance versus identical post-hoc routing"
            ),
        },
        "nonclaims": [
            "Family-projected all-current readiness is not an exact route dossier.",
            "The ordinal utility is not a probability of experimental synthesis success.",
            "A Graph2Edits proposal cannot create evidence or route closure.",
            "This audit does not select or lock prospective candidates.",
        ],
        "artifacts": {
            "route_readiness_ledger.jsonl.gz": {
                "path": "route_readiness_ledger.jsonl.gz",
                "rows": len(output_rows),
                "sha256": hashlib.sha256(ledger).hexdigest(),
                "schema_version": LEDGER_SCHEMA_VERSION,
            }
        },
    }
    content["result_sha256"] = hashlib.sha256(_stable_json(content).encode()).hexdigest()
    return content, ledger


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "EXACT",
    "FAMILY_ALL",
    "FAMILY_PARTIAL",
    "LEDGER_SCHEMA_VERSION",
    "READINESS_ORDER",
    "READINESS_UTILITY",
    "RESULT_SCHEMA_VERSION",
    "UNRESOLVED",
    "UgiProposalAugmentedRouteReadinessError",
    "build_proposal_augmented_route_readiness",
    "component_readiness_class",
    "product_readiness_class",
]
