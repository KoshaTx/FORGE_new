"""Validate source-resolved LNPDB subcomponent route reviews for M0-09."""

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

CONFIG_SCHEMA_VERSION = "m0_09_lnpdb_paper_reviews_config.v2"
RESULT_SCHEMA_VERSION = "m0_09_lnpdb_paper_route_reviews.v2"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class PaperReviewError(ValueError):
    """Raised when a curated paper review violates its evidence contract."""


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
    return read_json_object(path, error=PaperReviewError, label=label)


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise PaperReviewError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _verify_hash(path: Path, expected: Any, *, label: str) -> str:
    expected_sha256 = _require_sha256(expected, label=f"{label} expected_sha256")
    if not path.is_file():
        raise PaperReviewError(f"{label} is missing: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise PaperReviewError(f"{label} hash mismatch for {path}: {observed} != {expected_sha256}")
    return observed


def _load_component_ledger(path: Path) -> dict[str, dict[str, Any]]:
    try:
        handle = path.open(newline="")
    except FileNotFoundError as exc:
        raise PaperReviewError(f"component source ledger not found: {path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        required = {"component_id", "role", "source_pmids_json"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise PaperReviewError(
                f"component source ledger must contain columns {sorted(required)}"
            )
        records: dict[str, dict[str, Any]] = {}
        for row in reader:
            component_id = row["component_id"]
            if not component_id or component_id in records:
                raise PaperReviewError(
                    f"component source ledger has a missing or duplicate ID: {component_id!r}"
                )
            try:
                pmids = json.loads(row["source_pmids_json"])
            except json.JSONDecodeError as exc:
                raise PaperReviewError(f"invalid source_pmids_json for {component_id}") from exc
            if not isinstance(pmids, list) or any(not isinstance(pmid, str) for pmid in pmids):
                raise PaperReviewError(
                    f"source_pmids_json must be a string list for {component_id}"
                )
            records[component_id] = {
                "role": row["role"],
                "pmids": set(pmids),
            }
    if not records:
        raise PaperReviewError("component source ledger contains no components")
    return records


def _canonical_smiles(smiles: Any, *, label: str) -> str:
    if not isinstance(smiles, str) or not smiles:
        raise PaperReviewError(f"{label} must be a nonempty SMILES string")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise PaperReviewError(f"{label} is not valid SMILES: {smiles!r}")
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def _validate_yield(value: Any, *, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PaperReviewError(f"{label} must be numeric")
    if not 0 <= value <= 100:
        raise PaperReviewError(f"{label} must be between 0 and 100")


def _validate_source_scope(
    review: dict[str, Any],
    component_records: dict[str, dict[str, Any]],
) -> dict[str, int]:
    pmid = review["pmid"]
    observed = Counter(
        component["role"] for component in component_records.values() if pmid in component["pmids"]
    )
    expected_by_role = review["source_component_scope"]["normalized_by_role"]
    if not isinstance(expected_by_role, dict):
        raise PaperReviewError(f"{review['review_id']} normalized_by_role must be an object")
    expected = {str(role): count for role, count in expected_by_role.items()}
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in expected.values()
    ):
        raise PaperReviewError(
            f"{review['review_id']} normalized_by_role counts must be nonnegative integers"
        )
    if dict(sorted(observed.items())) != dict(sorted(expected.items())):
        raise PaperReviewError(
            f"{review['review_id']} normalized role counts do not match the component ledger: "
            f"{dict(sorted(observed.items()))} != {dict(sorted(expected.items()))}"
        )
    total = review["source_component_scope"]["normalized_unique_components"]
    if total != sum(expected.values()):
        raise PaperReviewError(
            f"{review['review_id']} normalized_unique_components does not equal role counts"
        )
    return dict(sorted(observed.items()))


def _validate_building_block_groups(review: dict[str, Any]) -> dict[str, Any]:
    upstream_counts: Counter[str] = Counter()
    normalized_procurement_counts: Counter[str] = Counter()
    source_procurement_counts: Counter[str] = Counter()
    normalized_components = 0
    source_reported_components = 0
    for group_index, group in enumerate(review["building_block_groups"]):
        label = f"{review['review_id']} building_block_groups[{group_index}]"
        normalized_count = group["normalized_component_count"]
        source_count = group["source_reported_count"]
        route_count = group["upstream_route_count"]
        if (
            isinstance(normalized_count, bool)
            or not isinstance(normalized_count, int)
            or normalized_count < 0
        ):
            raise PaperReviewError(f"{label} normalized_component_count must be nonnegative")
        if isinstance(source_count, bool) or not isinstance(source_count, int) or source_count < 1:
            raise PaperReviewError(f"{label} source_reported_count must be positive")
        if isinstance(route_count, bool) or not isinstance(route_count, int) or route_count < 0:
            raise PaperReviewError(f"{label} upstream_route_count must be nonnegative")
        if route_count > source_count:
            raise PaperReviewError(f"{label} upstream_route_count exceeds source_reported_count")
        route_status = group["upstream_route_status"]
        if route_status == "not_reported" and route_count != 0:
            raise PaperReviewError(f"{label} cannot count routes when status is not_reported")
        if route_status == "route_extracted" and route_count < 1:
            raise PaperReviewError(f"{label} route_extracted requires a positive route count")
        if route_status not in {"not_reported", "route_extracted"}:
            raise PaperReviewError(f"{label} has unsupported upstream_route_status")
        role = group["role"]
        upstream_counts[role] += route_count
        procurement_status = group["procurement_evidence_status"]
        normalized_procurement_counts[procurement_status] += normalized_count
        source_procurement_counts[procurement_status] += source_count
        normalized_components += normalized_count
        source_reported_components += source_count
    return {
        "normalized_components": normalized_components,
        "source_reported_components": source_reported_components,
        "upstream_routes": sum(upstream_counts.values()),
        "upstream_routes_by_role": upstream_counts,
        "normalized_procurement_by_evidence": normalized_procurement_counts,
        "source_procurement_by_evidence": source_procurement_counts,
    }


def _validate_exact_route_family(family: dict[str, Any], *, label: str) -> tuple[int, int, int]:
    steps = family["steps"]
    if not steps:
        raise PaperReviewError(f"{label} must contain at least one step")
    if family["route_instance_count"] != 1:
        raise PaperReviewError(f"{label} exact route must have route_instance_count 1")
    start = _canonical_smiles(
        family["starting_material"]["smiles"],
        label=f"{label} starting material",
    )
    target = _canonical_smiles(family["target"]["smiles"], label=f"{label} target")
    previous_product = None
    for step_index, step in enumerate(steps, start=1):
        step_label = f"{label} step {step_index}"
        if step["step_index"] != step_index:
            raise PaperReviewError(f"{step_label} index is not contiguous")
        reactant = _canonical_smiles(
            step["reactant_smiles"],
            label=f"{step_label} reactant",
        )
        product = _canonical_smiles(
            step["product_smiles"],
            label=f"{step_label} product",
        )
        if step_index == 1 and reactant != start:
            raise PaperReviewError(f"{step_label} does not start from starting_material")
        if previous_product is not None and reactant != previous_product:
            raise PaperReviewError(f"{step_label} does not continue from the prior product")
        _validate_yield(
            step["isolated_yield_percent"],
            label=f"{step_label} isolated_yield_percent",
        )
        previous_product = product
    if previous_product != target:
        raise PaperReviewError(f"{label} final step does not produce target")
    if family["reaction_instance_count"] != len(steps):
        raise PaperReviewError(f"{label} reaction_instance_count does not match steps")
    return 1, len(steps), 1


def _validate_product_series_family(
    family: dict[str, Any],
    *,
    label: str,
) -> tuple[int, int, int]:
    members = family["members"]
    route_count = family["route_instance_count"]
    reaction_count = family["reaction_instance_count"]
    route_depth = family["route_depth"]
    if route_count != len(members):
        raise PaperReviewError(f"{label} route_instance_count does not match members")
    if isinstance(route_depth, bool) or not isinstance(route_depth, int) or route_depth < 1:
        raise PaperReviewError(f"{label} route_depth must be a positive integer")
    if reaction_count != route_count * route_depth:
        raise PaperReviewError(f"{label} reaction_instance_count must equal routes times depth")
    member_labels: set[str] = set()
    structure_resolved = 0
    for member_index, member in enumerate(members):
        member_label = f"{label} members[{member_index}]"
        source_label = member["label"]
        if not source_label or source_label in member_labels:
            raise PaperReviewError(f"{member_label} has a missing or duplicate label")
        member_labels.add(source_label)
        if "isolated_yield_percent" in member:
            _validate_yield(
                member["isolated_yield_percent"],
                label=f"{member_label} isolated_yield_percent",
            )
        elif member.get("yield_status") != "not_reported":
            raise PaperReviewError(
                f"{member_label} requires isolated_yield_percent or yield_status not_reported"
            )
        if "product_smiles" in member:
            _canonical_smiles(
                member["product_smiles"],
                label=f"{member_label} product",
            )
            structure_resolved += 1
    return route_count, reaction_count, structure_resolved


def _validate_l2_families(review: dict[str, Any]) -> dict[str, int]:
    family_ids: set[str] = set()
    route_instances = 0
    reaction_instances = 0
    structure_resolved_instances = 0
    for family_index, family in enumerate(review["l2_route_families"]):
        label = f"{review['review_id']} l2_route_families[{family_index}]"
        family_id = family["route_family_id"]
        if family["level"] != "L2":
            raise PaperReviewError(f"{label} must have level L2")
        if not family_id or family_id in family_ids:
            raise PaperReviewError(f"{label} has a missing or duplicate route_family_id")
        family_ids.add(family_id)
        status = family["route_evidence_status"]
        if status == "exact_route_extracted":
            counts = _validate_exact_route_family(family, label=label)
        elif status == "general_procedure_product_series_extracted":
            counts = _validate_product_series_family(family, label=label)
        else:
            raise PaperReviewError(f"{label} has unsupported route_evidence_status")
        route_instances += counts[0]
        reaction_instances += counts[1]
        structure_resolved_instances += counts[2]
    return {
        "families": len(family_ids),
        "route_instances": route_instances,
        "reaction_instances": reaction_instances,
        "structure_resolved_route_instances": structure_resolved_instances,
    }


def _validate_l1_families(
    review: dict[str, Any],
    component_records: dict[str, dict[str, Any]],
) -> int:
    family_ids: set[str] = set()
    pmid = review["pmid"]
    for family_index, family in enumerate(review["l1_assembly_families"]):
        label = f"{review['review_id']} l1_assembly_families[{family_index}]"
        family_id = family["assembly_family_id"]
        if family["level"] != "L1":
            raise PaperReviewError(f"{label} must have level L1")
        if not family_id or family_id in family_ids:
            raise PaperReviewError(f"{label} has a missing or duplicate assembly_family_id")
        family_ids.add(family_id)
        members = family.get("members", [])
        if "member_count" in family and family["member_count"] != len(members):
            raise PaperReviewError(f"{label} member_count does not match members")
        seen_members: set[str] = set()
        for member_index, member in enumerate(members):
            member_label = f"{label} members[{member_index}]"
            component_id = member["component_id"]
            if component_id in seen_members:
                raise PaperReviewError(f"{member_label} repeats component_id {component_id}")
            seen_members.add(component_id)
            component = component_records.get(component_id)
            if component is None:
                raise PaperReviewError(
                    f"{member_label} references unknown component {component_id}"
                )
            if component["role"] not in {"tail1", "tail2"}:
                raise PaperReviewError(f"{member_label} must reference a tail component")
            if pmid not in component["pmids"]:
                raise PaperReviewError(
                    f"{member_label} component {component_id} is not linked to PMID {pmid}"
                )
            _validate_yield(
                member["isolated_yield_percent"],
                label=f"{member_label} isolated_yield_percent",
            )
        if "representative_product_yield_percent" in family:
            _validate_yield(
                family["representative_product_yield_percent"],
                label=f"{label} representative_product_yield_percent",
            )
        if "theoretical_library_size" in family:
            theoretical = family["theoretical_library_size"]
            reported = family["reported_library_size"]
            unresolved = family["unresolved_combination_count"]
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (theoretical, reported, unresolved)
            ):
                raise PaperReviewError(f"{label} library counts must be nonnegative integers")
            if reported > theoretical or theoretical - reported != unresolved:
                raise PaperReviewError(f"{label} library counts do not reconcile")
    return len(family_ids)


def _validate_review_assets(
    review: dict[str, Any],
    source: dict[str, Any],
    cache_root: Path,
) -> list[dict[str, str]]:
    review_id = review["review_id"]
    reviewed_assets = []
    asset_config = review["source_asset"]
    matching_assets = [
        asset for asset in source["supplements"] if asset["filename"] == asset_config["filename"]
    ]
    if len(matching_assets) != 1:
        raise PaperReviewError(
            f"{review_id} source asset {asset_config['filename']!r} is not unique in source index"
        )
    asset = matching_assets[0]
    expected_asset_sha256 = _require_sha256(
        asset_config["expected_sha256"],
        label=f"{review_id} source asset expected_sha256",
    )
    if expected_asset_sha256 != asset["sha256"]:
        raise PaperReviewError(f"{review_id} source asset hash does not match source index")
    _verify_hash(
        cache_root / asset["cache_asset"],
        expected_asset_sha256,
        label=f"{review_id} reviewed source asset",
    )
    pages = asset_config["chemistry_pages"]
    if (
        not isinstance(pages, list)
        or not pages
        or any(isinstance(page, bool) or not isinstance(page, int) or page < 1 for page in pages)
        or pages != sorted(set(pages))
    ):
        raise PaperReviewError(
            f"{review_id} chemistry_pages must be sorted unique positive integers"
        )
    if asset_config["review_status"] != "chemistry_pages_visually_reviewed":
        raise PaperReviewError(f"{review_id} source asset has not been visually reviewed")
    reviewed_assets.append(
        {
            "asset_kind": "supplement",
            "cache_asset": asset["cache_asset"],
            "sha256": asset["sha256"],
        }
    )

    if "main_text_review" in review:
        main_config = review["main_text_review"]
        main_text = source["main_text"]
        expected_main_sha256 = _require_sha256(
            main_config["expected_sha256"],
            label=f"{review_id} main text expected_sha256",
        )
        if expected_main_sha256 != main_text["sha256"]:
            raise PaperReviewError(f"{review_id} main text hash does not match source index")
        _verify_hash(
            cache_root / main_text["cache_asset"],
            expected_main_sha256,
            label=f"{review_id} reviewed main text",
        )
        if main_config["review_status"] != "machine_readable_chemistry_sections_reviewed":
            raise PaperReviewError(f"{review_id} main text chemistry has not been reviewed")
        locators = main_config["section_locators"]
        if (
            not isinstance(locators, list)
            or not locators
            or any(not isinstance(locator, str) or not locator for locator in locators)
        ):
            raise PaperReviewError(f"{review_id} main text section_locators are invalid")
        reviewed_assets.append(
            {
                "asset_kind": "machine_readable_main_text",
                "cache_asset": main_text["cache_asset"],
                "sha256": main_text["sha256"],
            }
        )
    return reviewed_assets


def _validate_negative_outcomes(review: dict[str, Any]) -> int:
    outcome_ids: set[str] = set()
    total = 0
    for outcome_index, outcome in enumerate(review.get("negative_outcomes", [])):
        label = f"{review['review_id']} negative_outcomes[{outcome_index}]"
        outcome_id = outcome["outcome_family_id"]
        if not outcome_id or outcome_id in outcome_ids:
            raise PaperReviewError(f"{label} has a missing or duplicate outcome_family_id")
        outcome_ids.add(outcome_id)
        count = outcome["reported_outcome_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise PaperReviewError(f"{label} reported_outcome_count must be positive")
        entries = outcome.get("entries", [])
        if entries and len(entries) != count:
            raise PaperReviewError(f"{label} reported_outcome_count does not match entries")
        total += count
    return total


def _verify_supporting_route_artifacts(
    config_path: Path,
    inputs: dict[str, Any],
) -> list[Path]:
    repo_root = config_path.resolve().parents[2]
    verified_paths: list[Path] = []
    for artifact_index, artifact in enumerate(inputs.get("supporting_route_artifacts", [])):
        label = f"supporting_route_artifacts[{artifact_index}]"
        declared = Path(artifact["asset"])
        path = declared if declared.is_absolute() else repo_root / declared
        _verify_hash(path, artifact["expected_sha256"], label=label)
        payload = _load_json(path, label=label)
        if payload.get("schema_version") != artifact["expected_schema_version"]:
            raise PaperReviewError(f"{label} has an unexpected schema version")
        routes = payload.get("routes")
        if not isinstance(routes, list) or len(routes) != artifact["expected_route_count"]:
            raise PaperReviewError(f"{label} has an unexpected route count")
        expected_labels = artifact.get("expected_component_labels")
        if expected_labels is not None:
            if (
                not isinstance(expected_labels, list)
                or not expected_labels
                or any(not isinstance(value, str) or not value for value in expected_labels)
                or len(set(expected_labels)) != len(expected_labels)
            ):
                raise PaperReviewError(
                    f"{label} expected_component_labels must be unique nonempty strings"
                )
            observed_labels = [route.get("component_label") for route in routes]
            if any(not isinstance(value, str) or not value for value in observed_labels):
                raise PaperReviewError(f"{label} routes must have nonempty component_label values")
            if len(set(observed_labels)) != len(observed_labels):
                raise PaperReviewError(f"{label} routes contain duplicate component_label values")
            if sorted(observed_labels) != sorted(expected_labels):
                raise PaperReviewError(f"{label} component labels do not match the declared set")
        verified_paths.append(path)
    return verified_paths


def _validate_review(
    review: dict[str, Any],
    source_records: dict[str, dict[str, Any]],
    component_records: dict[str, dict[str, Any]],
    cache_root: Path,
) -> dict[str, Any]:
    review_id = review["review_id"]
    source = source_records.get(review["pmc_id"])
    if source is None:
        raise PaperReviewError(f"{review_id} references unknown PMCID {review['pmc_id']}")
    for field in ("pmid", "doi"):
        if review[field] != source[field]:
            raise PaperReviewError(
                f"{review_id} {field} does not match source review index: "
                f"{review[field]!r} != {source[field]!r}"
            )
    if source["source_package_status"] != "complete":
        raise PaperReviewError(f"{review_id} source package is not complete")
    reviewed_assets = _validate_review_assets(review, source, cache_root)

    role_counts = _validate_source_scope(review, component_records)
    building_blocks = _validate_building_block_groups(review)
    l2 = _validate_l2_families(review)
    l1_families = _validate_l1_families(review, component_records)
    negative_outcomes = _validate_negative_outcomes(review)
    conclusion = review["review_conclusion"]
    expected_conclusion = {
        "l2_route_families": l2["families"],
        "l2_route_instances": l2["route_instances"],
        "l2_reaction_instances": l2["reaction_instances"],
        "structure_resolved_l2_route_instances": l2["structure_resolved_route_instances"],
        "upstream_component_routes": building_blocks["upstream_routes"],
        "l1_assembly_families": l1_families,
        "reported_negative_outcomes": negative_outcomes,
    }
    for field, expected in expected_conclusion.items():
        if conclusion[field] != expected:
            raise PaperReviewError(
                f"{review_id} conclusion {field} is {conclusion[field]}, expected {expected}"
            )
    return {
        "review_id": review_id,
        "pmc_id": review["pmc_id"],
        "pmid": review["pmid"],
        "reviewed_assets": reviewed_assets,
        "normalized_components": sum(role_counts.values()),
        "normalized_components_by_role": role_counts,
        "normalized_building_block_components_reviewed": building_blocks["normalized_components"],
        "source_reported_building_blocks_reviewed": building_blocks["source_reported_components"],
        "upstream_component_routes": building_blocks["upstream_routes"],
        "upstream_component_routes_by_role": dict(
            sorted(building_blocks["upstream_routes_by_role"].items())
        ),
        "normalized_procurement_components_by_evidence": dict(
            sorted(building_blocks["normalized_procurement_by_evidence"].items())
        ),
        "source_procurement_components_by_evidence": dict(
            sorted(building_blocks["source_procurement_by_evidence"].items())
        ),
        "l2_route_families": l2["families"],
        "l2_route_instances": l2["route_instances"],
        "l2_reaction_instances": l2["reaction_instances"],
        "structure_resolved_l2_route_instances": l2["structure_resolved_route_instances"],
        "l1_assembly_families": l1_families,
        "reported_negative_outcomes": negative_outcomes,
    }


def build_paper_route_reviews(
    config_path: Path,
    source_review_index_path: Path,
    component_ledger_path: Path,
    cache_root: Path,
    *,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    """Validate curated reviews and build an aggregate, provenance-rich result."""

    config = _load_json(config_path, label="paper review config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise PaperReviewError(f"paper review config schema must be {CONFIG_SCHEMA_VERSION!r}")
    inputs = config["inputs"]
    _verify_hash(
        source_review_index_path,
        inputs["source_review_index"]["expected_sha256"],
        label="source review index",
    )
    _verify_hash(
        component_ledger_path,
        inputs["component_source_ledger"]["expected_sha256"],
        label="component source ledger",
    )
    supporting_route_artifacts = _verify_supporting_route_artifacts(config_path, inputs)
    source_index = _load_json(source_review_index_path, label="source review index")
    if source_index.get("schema_version") != "m0_09_source_review_index.v1":
        raise PaperReviewError("source review index has an unsupported schema")
    source_records = {record["pmc_id"]: record for record in source_index["records"]}
    component_records = _load_component_ledger(component_ledger_path)

    reviews = config["reviews"]
    if not isinstance(reviews, list) or not reviews:
        raise PaperReviewError("paper review config must contain at least one review")
    review_ids = [review["review_id"] for review in reviews]
    if len(set(review_ids)) != len(review_ids):
        raise PaperReviewError("paper review config contains duplicate review_id values")
    pmc_ids = [review["pmc_id"] for review in reviews]
    if len(set(pmc_ids)) != len(pmc_ids):
        raise PaperReviewError("paper review config contains duplicate PMCID reviews")

    audit_records = [
        _validate_review(review, source_records, component_records, cache_root)
        for review in reviews
    ]
    if generated_utc is None:
        generated = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    else:
        try:
            parsed = dt.datetime.fromisoformat(generated_utc)
        except ValueError as exc:
            raise PaperReviewError(
                f"generated_utc is not valid ISO-8601: {generated_utc!r}"
            ) from exc
        if parsed.tzinfo is None:
            raise PaperReviewError("generated_utc must include a timezone")
        generated = parsed.isoformat()

    aggregate_role_counts: Counter[str] = Counter()
    aggregate_upstream_role_counts: Counter[str] = Counter()
    normalized_procurement_counts: Counter[str] = Counter()
    source_procurement_counts: Counter[str] = Counter()
    for record in audit_records:
        aggregate_role_counts.update(record["normalized_components_by_role"])
        aggregate_upstream_role_counts.update(record["upstream_component_routes_by_role"])
        normalized_procurement_counts.update(
            record["normalized_procurement_components_by_evidence"]
        )
        source_procurement_counts.update(record["source_procurement_components_by_evidence"])
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": "M0-09 source-resolved LNPDB subcomponent route reviews",
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
                source_review_index_path,
                component_ledger_path,
                *supporting_route_artifacts,
            )
        ],
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
        },
        "summary": {
            "source_packages_chemistry_reviewed": len(audit_records),
            "normalized_components_in_reviewed_sources": sum(
                record["normalized_components"] for record in audit_records
            ),
            "normalized_components_by_role": dict(sorted(aggregate_role_counts.items())),
            "normalized_building_block_components_reviewed": sum(
                record["normalized_building_block_components_reviewed"] for record in audit_records
            ),
            "source_reported_building_blocks_reviewed": sum(
                record["source_reported_building_blocks_reviewed"] for record in audit_records
            ),
            "upstream_component_routes": sum(
                record["upstream_component_routes"] for record in audit_records
            ),
            "upstream_component_routes_by_role": dict(
                sorted(aggregate_upstream_role_counts.items())
            ),
            "l2_route_families": sum(record["l2_route_families"] for record in audit_records),
            "l2_route_instances": sum(record["l2_route_instances"] for record in audit_records),
            "l2_reaction_instances": sum(
                record["l2_reaction_instances"] for record in audit_records
            ),
            "structure_resolved_l2_route_instances": sum(
                record["structure_resolved_l2_route_instances"] for record in audit_records
            ),
            "l1_assembly_families": sum(record["l1_assembly_families"] for record in audit_records),
            "reported_negative_outcomes": sum(
                record["reported_negative_outcomes"] for record in audit_records
            ),
            "normalized_procurement_components_by_evidence": dict(
                sorted(normalized_procurement_counts.items())
            ),
            "source_procurement_components_by_evidence": dict(
                sorted(source_procurement_counts.items())
            ),
        },
        "evidence_contract": {
            "L1": "Final lipid assembly or immediate linker-tail subassembly.",
            "L2": "Upstream synthesis of a component or shared component precursor.",
            "not_reported": "The reviewed source does not report how the component itself was made.",
            "historical_blanket_vendor_claim": "The source states that a group was commercial but does not identify an exact item and current supplier.",
            "structure_resolved_route": "Every product structure required for the route instance is stored as validated SMILES.",
            "review_boundary": "A completed review applies only to the listed asset and chemistry pages.",
        },
        "review_audit": audit_records,
        "reviews": reviews,
    }


def write_paper_route_reviews(result: dict[str, Any], output_path: Path) -> None:
    """Atomically write validated paper route reviews."""

    _atomic_write_bytes(
        output_path,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
