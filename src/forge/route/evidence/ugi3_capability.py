"""Build a bounded, source-resolved Ugi-3 precursor capability audit."""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import platform
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.core.io import read_json_object
from forge.route.sources.supervision_inventory import sha256_file

CONFIG_SCHEMA_VERSION = "m0_09_ugi3_precursor_capability_config.v2"
RESULT_SCHEMA_VERSION = "m0_09_ugi3_precursor_capability.v2"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_EVIDENCE_AXES = {
    "component_observation",
    "operational_availability",
    "prospective_outcome",
    "route_closure",
    "transformation_evidence",
}


class Ugi3CapabilityError(ValueError):
    """Raised when Ugi-3 capability evidence violates its contract."""


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    return read_json_object(path, error=Ugi3CapabilityError, label=label)


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise Ugi3CapabilityError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _verify_hash(path: Path, expected: Any, *, label: str) -> str:
    expected_sha256 = _require_sha256(expected, label=f"{label} expected_sha256")
    if not path.is_file():
        raise Ugi3CapabilityError(f"{label} is missing: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise Ugi3CapabilityError(
            f"{label} hash mismatch for {path}: {observed} != {expected_sha256}"
        )
    return observed


def _canonical_smiles(smiles: Any, *, label: str) -> str:
    if not isinstance(smiles, str) or not smiles:
        raise Ugi3CapabilityError(f"{label} must be a nonempty SMILES string")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise Ugi3CapabilityError(f"{label} is not valid SMILES: {smiles!r}")
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def _require_count(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Ugi3CapabilityError(f"{label} must be a nonnegative integer")
    return value


def _expect_count(observed: int, expected: Any, *, label: str) -> None:
    expected_count = _require_count(expected, label=f"{label} expected count")
    if observed != expected_count:
        raise Ugi3CapabilityError(
            f"{label} count mismatch: observed {observed}, expected {expected_count}"
        )


def _generated_timestamp(generated_utc: str | None) -> str:
    if generated_utc is None:
        return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    try:
        parsed = dt.datetime.fromisoformat(generated_utc)
    except ValueError as exc:
        raise Ugi3CapabilityError(
            f"generated_utc is not valid ISO-8601: {generated_utc!r}"
        ) from exc
    if parsed.tzinfo is None:
        raise Ugi3CapabilityError("generated_utc must include a timezone")
    return parsed.isoformat()


def _validate_evidence_axes(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    axes = config.get("evidence_axes")
    if not isinstance(axes, dict):
        raise Ugi3CapabilityError("evidence_axes must be an object")
    if set(axes) != _REQUIRED_EVIDENCE_AXES:
        raise Ugi3CapabilityError(
            "evidence_axes must contain exactly "
            f"{sorted(_REQUIRED_EVIDENCE_AXES)}"
        )

    state_catalog: dict[str, set[str]] = {}
    for axis_name, axis in axes.items():
        if not isinstance(axis, dict):
            raise Ugi3CapabilityError(f"evidence axis {axis_name} must be an object")
        cardinality = axis.get("cardinality")
        if cardinality not in {"single", "multiple"}:
            raise Ugi3CapabilityError(
                f"evidence axis {axis_name} cardinality must be single or multiple"
            )
        definitions = axis.get("states")
        if not isinstance(definitions, list) or not definitions:
            raise Ugi3CapabilityError(
                f"evidence axis {axis_name} states must be a nonempty list"
            )
        if any(not isinstance(definition, dict) for definition in definitions):
            raise Ugi3CapabilityError(
                f"evidence axis {axis_name} state definitions must be objects"
            )
        if any("rank" in definition for definition in definitions):
            raise Ugi3CapabilityError(
                f"evidence axis {axis_name} must not encode a cross-axis rank"
            )
        states = [definition.get("state") for definition in definitions]
        if any(not isinstance(state, str) or not state for state in states):
            raise Ugi3CapabilityError(
                f"evidence axis {axis_name} states must be nonempty strings"
            )
        if len(set(states)) != len(states):
            raise Ugi3CapabilityError(
                f"evidence axis {axis_name} states must be unique"
            )
        if any(
            not isinstance(definition.get("meaning"), str)
            or not definition["meaning"]
            for definition in definitions
        ):
            raise Ugi3CapabilityError(
                f"evidence axis {axis_name} meanings must be nonempty strings"
            )
        state_catalog[axis_name] = set(states)
    return axes, state_catalog


def _validate_axis_value(
    value: Any,
    *,
    axis_name: str,
    state_catalog: dict[str, set[str]],
    cardinality: str,
    label: str,
) -> str | list[str]:
    allowed = state_catalog[axis_name]
    if cardinality == "single":
        if not isinstance(value, str) or value not in allowed:
            raise Ugi3CapabilityError(
                f"{label} must be one of {sorted(allowed)}"
            )
        return value
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(state, str) or state not in allowed for state in value)
        or len(set(value)) != len(value)
    ):
        raise Ugi3CapabilityError(
            f"{label} must be a nonempty unique list drawn from {sorted(allowed)}"
        )
    return sorted(value)


def _validate_scope_axes(
    scope: dict[str, Any],
    *,
    state_catalog: dict[str, set[str]],
    label: str,
) -> dict[str, Any]:
    validated = {
        "route_closure": _validate_axis_value(
            scope.get("route_closure"),
            axis_name="route_closure",
            state_catalog=state_catalog,
            cardinality="single",
            label=f"{label} route_closure",
        ),
        "operational_availability": _validate_axis_value(
            scope.get("operational_availability"),
            axis_name="operational_availability",
            state_catalog=state_catalog,
            cardinality="single",
            label=f"{label} operational_availability",
        ),
        "prospective_outcome": _validate_axis_value(
            scope.get("prospective_outcome"),
            axis_name="prospective_outcome",
            state_catalog=state_catalog,
            cardinality="single",
            label=f"{label} prospective_outcome",
        ),
    }
    forward_status = scope.get("forward_verification_status")
    if forward_status not in {"not_run", "passed", "failed"}:
        raise Ugi3CapabilityError(
            f"{label} forward_verification_status must be not_run, passed, or failed"
        )
    validated["forward_verification_status"] = forward_status
    return validated


def _load_component_ledger(path: Path) -> list[dict[str, Any]]:
    try:
        handle = path.open(newline="")
    except FileNotFoundError as exc:
        raise Ugi3CapabilityError(f"component source ledger not found: {path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        required = {
            "component_id",
            "role",
            "parse_status",
            "canonical_smiles",
            "source_pmids_json",
            "experiment_ids_json",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise Ugi3CapabilityError(
                f"component source ledger must contain columns {sorted(required)}"
            )
        records: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for row in reader:
            component_id = row["component_id"]
            if not component_id or component_id in seen_ids:
                raise Ugi3CapabilityError(
                    f"component source ledger has a missing or duplicate ID: {component_id!r}"
                )
            seen_ids.add(component_id)
            try:
                pmids = json.loads(row["source_pmids_json"])
                experiment_ids = json.loads(row["experiment_ids_json"])
            except json.JSONDecodeError as exc:
                raise Ugi3CapabilityError(
                    f"invalid JSON source fields for {component_id}"
                ) from exc
            if (
                not isinstance(pmids, list)
                or any(not isinstance(value, str) for value in pmids)
                or not isinstance(experiment_ids, list)
                or any(not isinstance(value, str) for value in experiment_ids)
            ):
                raise Ugi3CapabilityError(
                    f"source fields for {component_id} must be string lists"
                )
            canonical = None
            if row["parse_status"] == "parsed":
                canonical = _canonical_smiles(
                    row["canonical_smiles"],
                    label=f"component ledger {component_id}",
                )
            records.append(
                {
                    "component_id": component_id,
                    "role": row["role"],
                    "canonical_smiles": canonical,
                    "pmids": pmids,
                    "experiment_ids": experiment_ids,
                }
            )
    if not records:
        raise Ugi3CapabilityError("component source ledger contains no components")
    return records


def _validate_pool(
    pool: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    expected = config["expected_counts"]
    blocks = pool.get("blocks")
    if not isinstance(blocks, list):
        raise Ugi3CapabilityError("building block pool blocks must be a list")
    _expect_count(len(blocks), expected["building_blocks"], label="building blocks")
    if pool.get("total_blocks") != len(blocks):
        raise Ugi3CapabilityError("building block pool total_blocks does not match blocks")
    observed_by_handle = Counter()
    seen_ids: set[str] = set()
    for index, block in enumerate(blocks):
        block_id = block.get("block_id")
        if not isinstance(block_id, str) or not block_id or block_id in seen_ids:
            raise Ugi3CapabilityError(
                f"building block {index} has a missing or duplicate block_id"
            )
        seen_ids.add(block_id)
        handle = block.get("handle")
        if not isinstance(handle, str) or not handle:
            raise Ugi3CapabilityError(f"building block {block_id} has no handle")
        observed_by_handle[handle] += 1
        canonical = _canonical_smiles(
            block.get("canonical_smiles"),
            label=f"building block {block_id}",
        )
        if canonical != block["canonical_smiles"]:
            raise Ugi3CapabilityError(
                f"building block {block_id} canonical_smiles is not canonical"
            )
    if dict(sorted(observed_by_handle.items())) != pool.get("blocks_by_handle"):
        raise Ugi3CapabilityError("building block handle counts do not match pool metadata")
    for handle, expected_count in expected["blocks_by_handle"].items():
        _expect_count(
            observed_by_handle[handle],
            expected_count,
            label=f"{handle} building blocks",
        )
    return blocks, dict(sorted(observed_by_handle.items()))


def _agile_isocyanide_upstream_routes(
    agile: dict[str, Any],
    *,
    route_family_id: str,
    expected_count: int,
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    if agile.get("schema_version") != "m0_09_agile_component_routes.v1":
        raise Ugi3CapabilityError("AGILE route artifact has an unsupported schema")
    routes = agile.get("routes")
    if not isinstance(routes, list):
        raise Ugi3CapabilityError("AGILE route artifact routes must be a list")
    exact_routes = [
        route for route in routes if route.get("route_family_id") == route_family_id
    ]
    _expect_count(len(exact_routes), expected_count, label="direct AGILE isocyanide routes")
    by_target: dict[str, list[str]] = {}
    audit: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for route in exact_routes:
        route_id = route.get("route_id")
        if not isinstance(route_id, str) or not route_id or route_id in seen_ids:
            raise Ugi3CapabilityError("AGILE isocyanide routes have missing or duplicate IDs")
        seen_ids.add(route_id)
        target = _canonical_smiles(
            route.get("target", {}).get("canonical_smiles"),
            label=f"AGILE route {route_id} target",
        )
        by_target.setdefault(target, []).append(route_id)
        audit.append(
            {
                "route_id": route_id,
                "component_label": route.get("component_label"),
                "canonical_smiles": target,
                "reaction_steps": len(route.get("steps", [])),
                "route_evidence_status": route.get("route_evidence_status"),
                "execution_closure_status": route.get("execution_closure_status"),
                "forward_verification_status": route.get("forward_verification_status"),
            }
        )
    return by_target, sorted(audit, key=lambda record: record["route_id"])


def _candidate_audit(
    blocks: list[dict[str, Any]],
    agile_upstream_routes: dict[str, list[str]],
    config: dict[str, Any],
    state_catalog: dict[str, set[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scope = config["candidate_scope"]
    handle = scope["handle"]
    source = scope["required_provenance_source"]
    shared_axes = _validate_scope_axes(
        scope,
        state_catalog=state_catalog,
        label="candidate scope",
    )
    direct_transformation = _validate_axis_value(
        scope.get("exact_upstream_route_transformation_evidence"),
        axis_name="transformation_evidence",
        state_catalog=state_catalog,
        cardinality="multiple",
        label="candidate scope exact upstream route transformation evidence",
    )
    direct_observation = _validate_axis_value(
        scope.get("exact_upstream_route_component_observation"),
        axis_name="component_observation",
        state_catalog=state_catalog,
        cardinality="multiple",
        label="candidate scope exact upstream route component observation",
    )
    extrapolated_transformation = _validate_axis_value(
        scope.get("extrapolated_transformation_evidence"),
        axis_name="transformation_evidence",
        state_catalog=state_catalog,
        cardinality="multiple",
        label="candidate scope extrapolated transformation evidence",
    )
    extrapolated_observation = _validate_axis_value(
        scope.get("extrapolated_component_observation"),
        axis_name="component_observation",
        state_catalog=state_catalog,
        cardinality="multiple",
        label="candidate scope extrapolated component observation",
    )
    candidates = [
        block
        for block in blocks
        if block.get("handle") == handle
        and block.get("provenance", {}).get("source") == source
    ]
    _expect_count(
        len(candidates),
        config["expected_counts"]["rational_isocyanide_candidates"],
        label="rational isocyanide candidates",
    )
    records: list[dict[str, Any]] = []
    strata: Counter[tuple[int, int]] = Counter()
    covered_strata: Counter[tuple[int, int]] = Counter()
    for block in sorted(candidates, key=lambda value: value["block_id"]):
        block_id = block["block_id"]
        descriptors = block.get("descriptors")
        if not isinstance(descriptors, dict):
            raise Ugi3CapabilityError(f"candidate {block_id} descriptors must be an object")
        branch_count = _require_count(
            descriptors.get("n_branches"),
            label=f"candidate {block_id} n_branches",
        )
        unsaturation = _require_count(
            descriptors.get("unsaturation"),
            label=f"candidate {block_id} unsaturation",
        )
        chain_length = _require_count(
            descriptors.get("chain_length"),
            label=f"candidate {block_id} chain_length",
        )
        canonical = block["canonical_smiles"]
        route_ids = sorted(agile_upstream_routes.get(canonical, []))
        transformation_evidence = (
            direct_transformation if route_ids else extrapolated_transformation
        )
        component_observation = (
            direct_observation if route_ids else extrapolated_observation
        )
        stratum = (branch_count, unsaturation)
        strata[stratum] += 1
        if route_ids:
            covered_strata[stratum] += 1
        records.append(
            {
                "block_id": block_id,
                "canonical_smiles": canonical,
                "descriptors": {
                    "chain_length": chain_length,
                    "n_branches": branch_count,
                    "unsaturation": unsaturation,
                },
                "exact_agile_upstream_route_ids": route_ids,
                "transformation_evidence": transformation_evidence,
                "component_observation": component_observation,
                **shared_axes,
            }
        )
    stratum_records = [
        {
            "n_branches": branches,
            "unsaturation": unsaturation,
            "candidate_count": count,
            "exact_agile_upstream_route_count": covered_strata[
                (branches, unsaturation)
            ],
            "uncovered_count": count - covered_strata[(branches, unsaturation)],
        }
        for (branches, unsaturation), count in sorted(strata.items())
    ]
    return records, stratum_records


def _agile_head_audit(
    agile: dict[str, Any],
    config: dict[str, Any],
    state_catalog: dict[str, set[str]],
) -> tuple[list[dict[str, Any]], set[str]]:
    scope = config["agile_head_scope"]
    shared_axes = _validate_scope_axes(
        scope,
        state_catalog=state_catalog,
        label="AGILE head scope",
    )
    transformation_evidence = _validate_axis_value(
        scope.get("transformation_evidence"),
        axis_name="transformation_evidence",
        state_catalog=state_catalog,
        cardinality="multiple",
        label="AGILE head scope transformation evidence",
    )
    component_observation = _validate_axis_value(
        scope.get("component_observation"),
        axis_name="component_observation",
        state_catalog=state_catalog,
        cardinality="multiple",
        label="AGILE head scope component observation",
    )
    components = agile.get("components")
    if not isinstance(components, list):
        raise Ugi3CapabilityError("AGILE route artifact components must be a list")
    heads = [
        component
        for component in components
        if component.get("component_class") == scope["component_class"]
    ]
    _expect_count(
        len(heads),
        config["expected_counts"]["agile_amine_heads"],
        label="AGILE amine heads",
    )
    records: list[dict[str, Any]] = []
    canonical_set: set[str] = set()
    labels: set[str] = set()
    for head in heads:
        label = head.get("label")
        if not isinstance(label, str) or not label or label in labels:
            raise Ugi3CapabilityError("AGILE amine heads have missing or duplicate labels")
        labels.add(label)
        procurement = head.get("procurement_evidence_status")
        route_status = head.get("route_evidence_status")
        if procurement not in scope["allowed_procurement_evidence_status"]:
            raise Ugi3CapabilityError(f"AGILE head {label} has unsupported procurement evidence")
        if route_status not in scope["allowed_route_evidence_status"]:
            raise Ugi3CapabilityError(f"AGILE head {label} has unsupported route evidence")
        route_ids = head.get("route_ids")
        if not isinstance(route_ids, list):
            raise Ugi3CapabilityError(f"AGILE head {label} route_ids must be a list")
        canonical = _canonical_smiles(
            head.get("canonical_smiles"),
            label=f"AGILE head {label}",
        )
        canonical_set.add(canonical)
        records.append(
            {
                "label": label,
                "canonical_smiles": canonical,
                "procurement_evidence_status": procurement,
                "route_evidence_status": route_status,
                "upstream_route_ids": sorted(route_ids),
                "current_item_level_procurement_closed": False,
                "transformation_evidence": transformation_evidence,
                "component_observation": component_observation,
                **shared_axes,
            }
        )
    return sorted(records, key=lambda record: record["label"]), canonical_set


def _miao_audit(
    records: list[dict[str, Any]],
    agile_head_smiles: set[str],
    config: dict[str, Any],
    state_catalog: dict[str, set[str]],
) -> dict[str, Any]:
    scope = config["miao_cross_assembly_scope"]
    shared_axes = _validate_scope_axes(
        scope,
        state_catalog=state_catalog,
        label="Miao cross-assembly scope",
    )
    transformation_evidence = _validate_axis_value(
        scope.get("transformation_evidence"),
        axis_name="transformation_evidence",
        state_catalog=state_catalog,
        cardinality="multiple",
        label="Miao scope transformation evidence",
    )
    component_observation = _validate_axis_value(
        scope.get("component_observation"),
        axis_name="component_observation",
        state_catalog=state_catalog,
        cardinality="multiple",
        label="Miao scope component observation",
    )
    pmid = scope["pmid"]
    experiment_id = scope["experiment_id"]
    source_records = [
        record
        for record in records
        if pmid in record["pmids"] and experiment_id in record["experiment_ids"]
    ]
    heads = sorted(
        (record for record in source_records if record["role"] == scope["head_role"]),
        key=lambda record: record["component_id"],
    )
    configured_isocyanide_ids = set(scope["isocyanide_component_ids"])
    if len(configured_isocyanide_ids) != len(scope["isocyanide_component_ids"]):
        raise Ugi3CapabilityError("Miao isocyanide component IDs must be unique")
    source_by_id = {record["component_id"]: record for record in source_records}
    missing = sorted(configured_isocyanide_ids - source_by_id.keys())
    if missing:
        raise Ugi3CapabilityError(f"Miao isocyanide components missing from ledger: {missing}")
    isocyanides = sorted(
        (source_by_id[component_id] for component_id in configured_isocyanide_ids),
        key=lambda record: record["component_id"],
    )
    if any(record["role"] != scope["isocyanide_role"] for record in isocyanides):
        raise Ugi3CapabilityError("configured Miao isocyanides do not all have the declared role")
    if any(record["canonical_smiles"] is None for record in (*heads, *isocyanides)):
        raise Ugi3CapabilityError("Miao capability components must all have parsed structures")
    expected = config["expected_counts"]
    _expect_count(len(heads), expected["miao_heads"], label="Miao amine heads")
    _expect_count(
        len(isocyanides),
        expected["miao_isocyanides"],
        label="Miao isocyanides",
    )
    overlap = sorted(
        {
            record["canonical_smiles"]
            for record in heads
            if record["canonical_smiles"] in agile_head_smiles
        }
    )
    _expect_count(
        len(overlap),
        expected["miao_agile_head_overlap"],
        label="Miao and AGILE exact amine-head overlap",
    )
    return {
        "pmid": pmid,
        "doi": scope["doi"],
        "assembly_applicability": scope["assembly_applicability"],
        "procurement_evidence_status": scope["procurement_evidence_status"],
        "transformation_evidence": transformation_evidence,
        "component_observation": component_observation,
        **shared_axes,
        "heads": [
            {
                "component_id": record["component_id"],
                "canonical_smiles": record["canonical_smiles"],
                "exact_agile_head_overlap": record["canonical_smiles"] in agile_head_smiles,
                "transformation_evidence": transformation_evidence,
                "component_observation": component_observation,
                **shared_axes,
            }
            for record in heads
        ],
        "isocyanides": [
            {
                "component_id": record["component_id"],
                "canonical_smiles": record["canonical_smiles"],
                "transformation_evidence": transformation_evidence,
                "component_observation": component_observation,
                **shared_axes,
            }
            for record in isocyanides
        ],
        "exact_agile_head_overlap_count": len(overlap),
        "scope_note": scope["scope_note"],
    }


def _validate_agile_review(
    paper_reviews: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    if paper_reviews.get("schema_version") != "m0_09_lnpdb_paper_route_reviews.v2":
        raise Ugi3CapabilityError("paper route review artifact has an unsupported schema")
    expected = config["agile_paper_review"]
    reviews = paper_reviews.get("reviews")
    audit = paper_reviews.get("review_audit")
    if not isinstance(reviews, list) or not isinstance(audit, list):
        raise Ugi3CapabilityError("paper route review artifact is missing review records")
    review_matches = [
        review for review in reviews if review.get("review_id") == expected["review_id"]
    ]
    audit_matches = [
        record for record in audit if record.get("review_id") == expected["review_id"]
    ]
    if len(review_matches) != 1 or len(audit_matches) != 1:
        raise Ugi3CapabilityError("AGILE paper review must resolve to exactly one record")
    review = review_matches[0]
    record = audit_matches[0]
    for field in (
        "upstream_component_routes",
        "l2_reaction_instances",
        "structure_resolved_l2_route_instances",
    ):
        configured_field = f"expected_{field}"
        if record.get(field) != expected[configured_field]:
            raise Ugi3CapabilityError(
                f"AGILE review {field} mismatch: {record.get(field)} "
                f"!= {expected[configured_field]}"
            )
    source_isocyanide_routes = sum(
        group["upstream_route_count"]
        for group in review["building_block_groups"]
        if group["role"] == "ugi3_isocyanide_tail"
    )
    _expect_count(
        source_isocyanide_routes,
        config["expected_counts"]["agile_source_reported_isocyanide_routes"],
        label="source-reported AGILE isocyanide routes",
    )
    return {
        "review_id": record["review_id"],
        "upstream_component_routes": record["upstream_component_routes"],
        "l2_reaction_instances": record["l2_reaction_instances"],
        "structure_resolved_l2_route_instances": record[
            "structure_resolved_l2_route_instances"
        ],
        "source_reported_isocyanide_routes": source_isocyanide_routes,
    }


def _validate_source_evidence(
    config: dict[str, Any],
    *,
    axes: dict[str, Any],
    state_catalog: dict[str, set[str]],
) -> list[dict[str, Any]]:
    sources = config.get("evidence_sources")
    if not isinstance(sources, list) or not sources:
        raise Ugi3CapabilityError("evidence_sources must be a nonempty list")
    source_ids: set[str] = set()
    for source in sources:
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id or source_id in source_ids:
            raise Ugi3CapabilityError("evidence sources have missing or duplicate source IDs")
        source_ids.add(source_id)
        tags = source.get("evidence_tags")
        if not isinstance(tags, dict) or not tags:
            raise Ugi3CapabilityError(
                f"evidence source {source_id} evidence_tags must be a nonempty object"
            )
        unknown_axes = set(tags) - set(axes)
        if unknown_axes:
            raise Ugi3CapabilityError(
                f"evidence source {source_id} uses unknown axes: {sorted(unknown_axes)}"
            )
        for axis_name, values in tags.items():
            _validate_axis_value(
                values,
                axis_name=axis_name,
                state_catalog=state_catalog,
                cardinality=axes[axis_name]["cardinality"],
                label=f"evidence source {source_id} {axis_name}",
            )
        if "asset_sha256" in source:
            _require_sha256(
                source["asset_sha256"],
                label=f"evidence source {source_id} asset_sha256",
            )
    return sources


def build_ugi3_precursor_capability(
    config_path: Path,
    building_block_pool_path: Path,
    agile_component_routes_path: Path,
    component_ledger_path: Path,
    paper_route_reviews_path: Path,
    *,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic audit without claiming unverified route closure."""

    config = _load_json(config_path, label="Ugi-3 capability config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3CapabilityError(
            f"Ugi-3 capability config schema must be {CONFIG_SCHEMA_VERSION!r}"
        )
    axes, state_catalog = _validate_evidence_axes(config)
    source_evidence = _validate_source_evidence(
        config,
        axes=axes,
        state_catalog=state_catalog,
    )
    paths = {
        "building_block_pool": building_block_pool_path,
        "agile_component_routes": agile_component_routes_path,
        "component_source_ledger": component_ledger_path,
        "paper_route_reviews": paper_route_reviews_path,
    }
    for input_name, path in paths.items():
        _verify_hash(
            path,
            config["inputs"][input_name]["expected_sha256"],
            label=input_name.replace("_", " "),
        )

    pool = _load_json(building_block_pool_path, label="building block pool")
    agile = _load_json(agile_component_routes_path, label="AGILE component routes")
    paper_reviews = _load_json(paper_route_reviews_path, label="paper route reviews")
    component_records = _load_component_ledger(component_ledger_path)
    blocks, handle_counts = _validate_pool(pool, config)

    agile_upstream_routes, agile_upstream_route_audit = (
        _agile_isocyanide_upstream_routes(
            agile,
            route_family_id=config["candidate_scope"][
                "exact_agile_upstream_route_family_id"
            ],
            expected_count=config["expected_counts"][
                "direct_agile_isocyanide_routes"
            ],
        )
    )
    candidates, strata = _candidate_audit(
        blocks,
        agile_upstream_routes,
        config,
        state_catalog,
    )
    exact_upstream_count = sum(
        bool(record["exact_agile_upstream_route_ids"]) for record in candidates
    )
    _expect_count(
        exact_upstream_count,
        config["expected_counts"]["direct_agile_isocyanide_routes"],
        label="candidate exact AGILE upstream route overlap",
    )
    agile_heads, agile_head_smiles = _agile_head_audit(
        agile,
        config,
        state_catalog,
    )
    miao = _miao_audit(
        component_records,
        agile_head_smiles,
        config,
        state_catalog,
    )
    agile_review = _validate_agile_review(paper_reviews, config)

    aldehydes = handle_counts["aldehyde"]
    amines = handle_counts["amine"]
    isocyanides = handle_counts["isocyanide"]
    raw_upper_bound = aldehydes * amines * isocyanides
    generated = _generated_timestamp(generated_utc)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": config["task"],
        "generated_utc": generated,
        "randomness": {"seed": 0, "used": False},
        "inputs": [
            {
                "asset": _portable_path(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (
                config_path,
                building_block_pool_path,
                agile_component_routes_path,
                component_ledger_path,
                paper_route_reviews_path,
            )
        ],
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
        },
        "summary": {
            "rational_isocyanide_candidates": len(candidates),
            "exact_agile_isocyanide_upstream_route_candidates": exact_upstream_count,
            "family_supported_isocyanide_candidates_without_exact_route": (
                len(candidates) - exact_upstream_count
            ),
            "computationally_route_complete_isocyanide_candidates": sum(
                record["route_closure"] == "computationally_complete"
                for record in candidates
            ),
            "agile_source_reported_isocyanide_routes": agile_review[
                "source_reported_isocyanide_routes"
            ],
            "agile_amine_heads": len(agile_heads),
            "agile_heads_with_upstream_route": sum(
                bool(record["upstream_route_ids"]) for record in agile_heads
            ),
            "agile_heads_with_current_item_level_procurement_closure": sum(
                record["current_item_level_procurement_closed"] for record in agile_heads
            ),
            "miao_cross_assembly_isocyanides": len(miao["isocyanides"]),
            "miao_cross_assembly_heads": len(miao["heads"]),
            "miao_agile_exact_head_overlap": miao["exact_agile_head_overlap_count"],
            "raw_component_cartesian_upper_bound": raw_upper_bound,
            "isocyanide_candidates_with_prospective_outcome": sum(
                record["prospective_outcome"] != "not_attempted"
                for record in candidates
            ),
        },
        "claims_boundary": config["claims_boundary"],
        "evidence_axes": axes,
        "reaction_basis": config["reaction_basis"],
        "source_evidence": source_evidence,
        "agile_review_audit": agile_review,
        "agile_isocyanide_upstream_route_audit": agile_upstream_route_audit,
        "isocyanide_candidate_strata": strata,
        "isocyanide_candidates": candidates,
        "agile_amine_head_audit": agile_heads,
        "miao_cross_assembly_audit": miao,
    }


def write_ugi3_precursor_capability(result: dict[str, Any], output_path: Path) -> None:
    """Atomically write a validated Ugi-3 capability audit."""

    _atomic_write_bytes(
        output_path,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
