"""Bounded M0-09 audit of Ugi-3 aldehydes and amine heads.

The audit attaches source routes by exact molecular identity, applies the literal
handle policy from the vendored reaction registry, and keeps historical
procurement claims separate from current item-level availability.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.chemistry import audit_reactive_site_multiplicity
from forge.core.io import read_json_object
from forge.route.supervision_inventory import sha256_file

CONFIG_SCHEMA_VERSION = "m0_09_ugi3_aldehyde_head_capability_config.v3"
RESULT_SCHEMA_VERSION = "m0_09_ugi3_aldehyde_head_capability.v3"
PROCUREMENT_SCHEMA_VERSION = "m0_09_ugi3_agile_head_procurement.v1"
ASSEMBLY_QUALIFICATION_SCHEMA_VERSION = "m0_09_ugi3_assembly_qualification.v1"


class Ugi3AldehydeHeadCapabilityError(ValueError):
    """Raised when aldehyde or head evidence violates the audit contract."""


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    return read_json_object(path, error=Ugi3AldehydeHeadCapabilityError, label=label)


def _verify_hash(path: Path, expected: Any, *, label: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise Ugi3AldehydeHeadCapabilityError(
            f"{label} expected_sha256 must be a 64-character string"
        )
    if not path.exists():
        raise Ugi3AldehydeHeadCapabilityError(f"{label} not found: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise Ugi3AldehydeHeadCapabilityError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )
    return observed


def _canonical_smiles(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Ugi3AldehydeHeadCapabilityError(f"{label} must be a nonempty SMILES string")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        raise Ugi3AldehydeHeadCapabilityError(f"{label} is not valid SMILES: {value!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _expect_count(observed: int, expected: Any, *, label: str) -> None:
    if isinstance(expected, bool) or not isinstance(expected, int):
        raise Ugi3AldehydeHeadCapabilityError(f"{label} expected count must be an integer")
    if observed != expected:
        raise Ugi3AldehydeHeadCapabilityError(
            f"{label} count mismatch: expected {expected}, observed {observed}"
        )


def _parse_timestamp(value: str, *, label: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Ugi3AldehydeHeadCapabilityError(
            f"{label} is not valid ISO-8601: {value!r}"
        ) from exc
    if parsed.tzinfo is None:
        raise Ugi3AldehydeHeadCapabilityError(f"{label} must include a timezone")
    return parsed


def _generated_timestamp(generated_utc: str | None) -> str:
    if generated_utc is None:
        return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    parsed = _parse_timestamp(generated_utc, label="generated_utc")
    return parsed.isoformat()


def _evidence_states(capability: dict[str, Any]) -> dict[str, set[str]]:
    axes = capability.get("evidence_axes")
    if not isinstance(axes, dict):
        raise Ugi3AldehydeHeadCapabilityError(
            "precursor capability artifact is missing evidence_axes"
        )
    states: dict[str, set[str]] = {}
    for axis_name, definition in axes.items():
        if not isinstance(definition, dict) or not isinstance(definition.get("states"), list):
            raise Ugi3AldehydeHeadCapabilityError(
                f"precursor capability evidence axis {axis_name} is malformed"
            )
        states[axis_name] = {
            entry["state"]
            for entry in definition["states"]
            if isinstance(entry, dict) and isinstance(entry.get("state"), str)
        }
    return states


def _validate_scope_states(
    config: dict[str, Any],
    state_catalog: dict[str, set[str]],
) -> None:
    checks = (
        (
            "aldehyde_scope",
            "exact_route_transformation_evidence",
            "transformation_evidence",
        ),
        (
            "aldehyde_scope",
            "unresolved_transformation_evidence",
            "transformation_evidence",
        ),
        ("aldehyde_scope", "route_closure", "route_closure"),
        ("aldehyde_scope", "operational_availability", "operational_availability"),
        ("aldehyde_scope", "prospective_outcome", "prospective_outcome"),
        ("head_scope", "transformation_evidence", "transformation_evidence"),
        ("head_scope", "route_closure", "route_closure"),
        (
            "head_scope",
            "verified_procurement_status",
            "operational_availability",
        ),
        ("head_scope", "prospective_outcome", "prospective_outcome"),
    )
    for section_name, field, axis_name in checks:
        value = config[section_name][field]
        values = value if isinstance(value, list) else [value]
        if not values or any(item not in state_catalog[axis_name] for item in values):
            raise Ugi3AldehydeHeadCapabilityError(
                f"{section_name}.{field} uses an unsupported {axis_name} state"
            )


def _load_registry_roles(
    registry: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    scope = config["registry_scope"]
    reactions = registry.get("reactions")
    if not isinstance(reactions, list):
        raise Ugi3AldehydeHeadCapabilityError("qualified reaction registry has no reactions")
    matches = [
        reaction
        for reaction in reactions
        if reaction.get("reaction_id") == scope["reaction_id"]
    ]
    if len(matches) != 1:
        raise Ugi3AldehydeHeadCapabilityError(
            f"registry reaction {scope['reaction_id']!r} must resolve exactly once"
        )
    roles = {
        role.get("name"): role
        for role in matches[0].get("reactant_roles", [])
        if isinstance(role, dict)
    }
    requested = (scope["aldehyde_role"], scope["amine_role"])
    if any(name not in roles for name in requested):
        raise Ugi3AldehydeHeadCapabilityError(
            f"registry reaction is missing requested roles {requested}"
        )
    return roles[requested[0]], roles[requested[1]]


def _compile_role(role: dict[str, Any], *, label: str) -> tuple[Chem.Mol, list[Chem.Mol]]:
    handle = Chem.MolFromSmarts(role.get("required_handle_smarts", ""))
    if handle is None:
        raise Ugi3AldehydeHeadCapabilityError(f"{label} required handle did not compile")
    forbidden = [
        Chem.MolFromSmarts(smarts) for smarts in role.get("forbidden_smarts", [])
    ]
    if any(pattern is None for pattern in forbidden):
        raise Ugi3AldehydeHeadCapabilityError(f"{label} forbidden SMARTS did not compile")
    return handle, [pattern for pattern in forbidden if pattern is not None]


def _handle_audit(
    canonical_smiles: str,
    role: dict[str, Any],
    handle: Chem.Mol,
    forbidden: list[Chem.Mol],
    *,
    multiplicity_semantics: str,
) -> dict[str, Any]:
    molecule = Chem.MolFromSmiles(canonical_smiles)
    if molecule is None:
        raise Ugi3AldehydeHeadCapabilityError(
            f"cannot audit invalid canonical SMILES {canonical_smiles!r}"
        )
    site_audit = audit_reactive_site_multiplicity(molecule, handle)
    forbidden_match = any(molecule.HasSubstructMatch(pattern) for pattern in forbidden)
    allowed = role.get("allowed_site_multiplicity")
    if (
        not isinstance(allowed, list)
        or not allowed
        or any(isinstance(value, bool) or not isinstance(value, int) for value in allowed)
    ):
        raise Ugi3AldehydeHeadCapabilityError(
            "registry allowed_site_multiplicity must be a nonempty integer list"
        )
    return {
        "required_handle_match_count": site_audit.raw_match_count,
        "raw_required_handle_match_count": site_audit.raw_match_count,
        "symmetry_distinct_required_handle_match_count": (
            site_audit.symmetry_distinct_match_count
        ),
        "site_multiplicity_semantics": multiplicity_semantics,
        "qualified_site_multiplicity": site_audit.count(multiplicity_semantics),
        "allowed_site_multiplicity": sorted(set(allowed)),
        "forbidden_handle_match": forbidden_match,
        "passes_literal_registry_handle_policy": (
            site_audit.raw_match_count in allowed and not forbidden_match
        ),
        "passes_qualified_registry_handle_policy": (
            site_audit.count(multiplicity_semantics) in allowed
            and not forbidden_match
        ),
    }


def _pool_candidates(
    pool: dict[str, Any],
    *,
    handle: str,
    expected_count: int,
) -> list[dict[str, Any]]:
    blocks = pool.get("blocks")
    if not isinstance(blocks, list):
        raise Ugi3AldehydeHeadCapabilityError("building block pool blocks must be a list")
    records = [block for block in blocks if block.get("handle") == handle]
    _expect_count(len(records), expected_count, label=f"{handle} candidates")
    seen: set[str] = set()
    for record in records:
        block_id = record.get("block_id")
        if not isinstance(block_id, str) or not block_id or block_id in seen:
            raise Ugi3AldehydeHeadCapabilityError(
                f"{handle} candidates contain a missing or duplicate block_id"
            )
        seen.add(block_id)
        canonical = _canonical_smiles(
            record.get("canonical_smiles"),
            label=f"building block {block_id}",
        )
        if canonical != record["canonical_smiles"]:
            raise Ugi3AldehydeHeadCapabilityError(
                f"building block {block_id} canonical_smiles is not canonical"
            )
    return sorted(records, key=lambda record: record["block_id"])


def _source_component_indices(
    agile: dict[str, Any],
) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    if agile.get("schema_version") != "m0_09_agile_component_routes.v1":
        raise Ugi3AldehydeHeadCapabilityError(
            "AGILE route artifact has an unsupported schema"
        )
    labels_by_canonical: dict[str, list[str]] = {}
    component_by_label: dict[str, dict[str, Any]] = {}
    components = agile.get("components")
    if not isinstance(components, list):
        raise Ugi3AldehydeHeadCapabilityError("AGILE route artifact has no components")
    for component in components:
        label = component.get("label")
        if not isinstance(label, str) or not label or label in component_by_label:
            raise Ugi3AldehydeHeadCapabilityError(
                "AGILE components contain a missing or duplicate label"
            )
        canonical = _canonical_smiles(
            component.get("canonical_smiles"),
            label=f"AGILE component {label}",
        )
        labels_by_canonical.setdefault(canonical, []).append(label)
        component_by_label[label] = component
    return labels_by_canonical, component_by_label


def _required_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Ugi3AldehydeHeadCapabilityError(f"{label} must be a nonempty string")
    return value


def _agile_head_procurement_index(
    procurement: dict[str, Any],
    *,
    component_by_label: dict[str, dict[str, Any]],
    head_candidates: list[dict[str, Any]],
    config: dict[str, Any],
    generated_utc: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if procurement.get("schema_version") != PROCUREMENT_SCHEMA_VERSION:
        raise Ugi3AldehydeHeadCapabilityError(
            "AGILE head procurement snapshot has an unsupported schema"
        )
    snapshot = procurement.get("snapshot")
    if not isinstance(snapshot, dict):
        raise Ugi3AldehydeHeadCapabilityError(
            "AGILE head procurement snapshot metadata is missing"
        )
    accessed_value = _required_string(
        snapshot.get("accessed_utc"),
        label="procurement snapshot accessed_utc",
    )
    accessed = _parse_timestamp(
        accessed_value,
        label="procurement snapshot accessed_utc",
    )
    generated = _parse_timestamp(generated_utc, label="generated_utc")
    expiry_days = snapshot.get("expiry_days")
    if (
        isinstance(expiry_days, bool)
        or not isinstance(expiry_days, int)
        or expiry_days <= 0
    ):
        raise Ugi3AldehydeHeadCapabilityError(
            "procurement snapshot expiry_days must be a positive integer"
        )
    valid_through = accessed + dt.timedelta(days=expiry_days)
    if generated < accessed:
        raise Ugi3AldehydeHeadCapabilityError(
            "generated_utc cannot predate the procurement snapshot"
        )
    if generated > valid_through:
        raise Ugi3AldehydeHeadCapabilityError(
            "AGILE head procurement snapshot expired before generated_utc"
        )
    region = _required_string(
        snapshot.get("region"),
        label="procurement snapshot region",
    )
    for policy_name in (
        "identity_policy",
        "closure_policy",
        "price_policy",
        "backup_policy",
    ):
        _required_string(
            snapshot.get(policy_name),
            label=f"procurement snapshot {policy_name}",
        )

    records = procurement.get("records")
    if not isinstance(records, list):
        raise Ugi3AldehydeHeadCapabilityError(
            "AGILE head procurement records must be a list"
        )
    expected = config["expected_counts"]
    _expect_count(
        len(records),
        expected["agile_head_procurement_records"],
        label="AGILE head procurement records",
    )
    procurement_expected = procurement.get("expected_counts")
    if not isinstance(procurement_expected, dict):
        raise Ugi3AldehydeHeadCapabilityError(
            "AGILE head procurement expected_counts is missing"
        )
    _expect_count(
        len(records),
        procurement_expected.get("records"),
        label="procurement snapshot records",
    )

    expected_labels = {
        label
        for label, component in component_by_label.items()
        if label.startswith("A") and component.get("component_class") == "amine"
    }
    pool_by_id = {candidate["block_id"]: candidate for candidate in head_candidates}
    indexed: dict[str, dict[str, Any]] = {}
    block_ids: set[str] = set()
    status_counts = {
        "current_item_level_vendor_verified": 0,
        "catalog_item_verified_availability_unresolved": 0,
    }
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            raise Ugi3AldehydeHeadCapabilityError(
                f"AGILE head procurement record {position} must be an object"
            )
        label = _required_string(
            record.get("component_label"),
            label=f"procurement record {position} component_label",
        )
        if label in indexed:
            raise Ugi3AldehydeHeadCapabilityError(
                f"duplicate procurement component label {label}"
            )
        if label not in expected_labels:
            raise Ugi3AldehydeHeadCapabilityError(
                f"procurement component label {label} is not an AGILE amine head"
            )
        block_id = _required_string(
            record.get("block_id"),
            label=f"procurement record {label} block_id",
        )
        if block_id in block_ids:
            raise Ugi3AldehydeHeadCapabilityError(
                f"duplicate procurement block_id {block_id}"
            )
        block_ids.add(block_id)
        pool_candidate = pool_by_id.get(block_id)
        if pool_candidate is None:
            raise Ugi3AldehydeHeadCapabilityError(
                f"procurement record {label} does not resolve to a pool head"
            )
        canonical = _canonical_smiles(
            record.get("canonical_smiles"),
            label=f"procurement record {label}",
        )
        if canonical != record["canonical_smiles"]:
            raise Ugi3AldehydeHeadCapabilityError(
                f"procurement record {label} canonical_smiles is not canonical"
            )
        source_canonical = _canonical_smiles(
            component_by_label[label].get("canonical_smiles"),
            label=f"AGILE component {label}",
        )
        if canonical != source_canonical:
            raise Ugi3AldehydeHeadCapabilityError(
                f"procurement record {label} identity does not match AGILE"
            )
        if canonical != pool_candidate["canonical_smiles"]:
            raise Ugi3AldehydeHeadCapabilityError(
                f"procurement record {label} identity does not match {block_id}"
            )

        identity = record.get("identity")
        if not isinstance(identity, dict):
            raise Ugi3AldehydeHeadCapabilityError(
                f"procurement record {label} identity metadata is missing"
            )
        pubchem_cid = identity.get("pubchem_cid")
        if (
            isinstance(pubchem_cid, bool)
            or not isinstance(pubchem_cid, int)
            or pubchem_cid <= 0
        ):
            raise Ugi3AldehydeHeadCapabilityError(
                f"procurement record {label} pubchem_cid must be positive"
            )
        for field in ("pubchem_title", "inchi_key", "cas_rn", "form"):
            _required_string(
                identity.get(field),
                label=f"procurement record {label} identity.{field}",
            )
        if not identity["form"].startswith("neutral_free_base"):
            raise Ugi3AldehydeHeadCapabilityError(
                f"procurement record {label} is not the neutral free-base form"
            )
        molecule = Chem.MolFromSmiles(canonical)
        if molecule is None or Chem.MolToInchiKey(molecule) != identity["inchi_key"]:
            raise Ugi3AldehydeHeadCapabilityError(
                f"procurement record {label} InChIKey does not match its structure"
            )

        vendor = record.get("vendor_evidence")
        if not isinstance(vendor, dict):
            raise Ugi3AldehydeHeadCapabilityError(
                f"procurement record {label} vendor evidence is missing"
            )
        for field in (
            "vendor",
            "product_title",
            "product_code",
            "url",
            "purity",
            "availability_observation",
        ):
            _required_string(
                vendor.get(field),
                label=f"procurement record {label} vendor_evidence.{field}",
            )
        if not vendor["url"].startswith("https://") or f"/{region}/" not in vendor["url"]:
            raise Ugi3AldehydeHeadCapabilityError(
                f"procurement record {label} URL is not region-specific HTTPS evidence"
            )

        status = record.get("procurement_status")
        if status not in status_counts:
            raise Ugi3AldehydeHeadCapabilityError(
                f"procurement record {label} has unsupported status {status!r}"
            )
        closed = record.get("current_item_level_procurement_closed")
        if not isinstance(closed, bool):
            raise Ugi3AldehydeHeadCapabilityError(
                f"procurement record {label} closure flag must be boolean"
            )
        if (status == config["head_scope"]["verified_procurement_status"]) != closed:
            raise Ugi3AldehydeHeadCapabilityError(
                f"procurement record {label} status and closure flag disagree"
            )
        if status == config["head_scope"]["unresolved_catalog_status"]:
            _required_string(
                record.get("required_followup"),
                label=f"procurement record {label} required_followup",
            )
        _required_string(
            record.get("backup_status"),
            label=f"procurement record {label} backup_status",
        )
        status_counts[status] += 1
        indexed[label] = {
            **record,
            "canonical_smiles": canonical,
        }

    if set(indexed) != expected_labels:
        missing = sorted(expected_labels - set(indexed))
        extra = sorted(set(indexed) - expected_labels)
        raise Ugi3AldehydeHeadCapabilityError(
            f"procurement snapshot does not cover every AGILE head; "
            f"missing={missing}, extra={extra}"
        )
    count_pairs = (
        (
            "current_item_level_vendor_verified",
            "agile_heads_current_item_level_vendor_verified",
        ),
        (
            "catalog_item_verified_availability_unresolved",
            "agile_heads_catalog_item_verified_availability_unresolved",
        ),
    )
    for status, config_key in count_pairs:
        observed = status_counts[status]
        _expect_count(
            observed,
            procurement_expected.get(status),
            label=f"procurement snapshot {status}",
        )
        _expect_count(
            observed,
            expected[config_key],
            label=config_key.replace("_", " "),
        )
    identity_discrepancies = 0
    _expect_count(
        identity_discrepancies,
        procurement_expected.get("identity_or_form_discrepancies"),
        label="procurement snapshot identity or form discrepancies",
    )
    _expect_count(
        identity_discrepancies,
        expected["agile_head_procurement_identity_or_form_discrepancies"],
        label="AGILE head procurement identity or form discrepancies",
    )
    return indexed, {
        "schema_version": procurement["schema_version"],
        "accessed_utc": accessed.isoformat(),
        "valid_through_utc": valid_through.isoformat(),
        "region": region,
        "expiry_days": expiry_days,
        "identity_policy": snapshot["identity_policy"],
        "closure_policy": snapshot["closure_policy"],
        "price_policy": snapshot["price_policy"],
        "backup_policy": snapshot["backup_policy"],
        "summary": {
            "records": len(records),
            **status_counts,
            "identity_or_form_discrepancies": identity_discrepancies,
        },
    }


def _validate_assembly_qualification(
    qualification: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    if qualification.get("schema_version") != ASSEMBLY_QUALIFICATION_SCHEMA_VERSION:
        raise Ugi3AldehydeHeadCapabilityError(
            "Ugi assembly qualification has an unsupported schema"
        )
    summary = qualification.get("summary")
    policy = qualification.get("multiplicity_policy")
    decision = qualification.get("decision")
    head_audit = qualification.get("head_audit")
    qa_flags = qualification.get("qa_flags")
    if not all(
        isinstance(value, expected_type)
        for value, expected_type in (
            (summary, dict),
            (policy, dict),
            (decision, dict),
            (head_audit, list),
            (qa_flags, list),
        )
    ):
        raise Ugi3AldehydeHeadCapabilityError(
            "Ugi assembly qualification is missing required sections"
        )
    scope = config["registry_scope"]
    if (
        policy.get("reaction_id") != scope["reaction_id"]
        or policy.get("role") != scope["amine_role"]
        or policy.get("semantics") != scope["multiplicity_semantics"]
    ):
        raise Ugi3AldehydeHeadCapabilityError(
            "Ugi assembly qualification policy disagrees with the capability config"
        )
    expected = config["expected_counts"]
    _expect_count(
        summary.get("products_passing_qualified_amine_multiplicity"),
        expected["agile_measured_products_passing_qualified_site_multiplicity"],
        label="AGILE products passing qualified site multiplicity",
    )
    if (
        summary.get("products_reconstructed_exactly")
        != summary.get("measured_products")
        or summary.get("products_failing_qualified_amine_multiplicity") != 0
        or decision.get("all_measured_products_pass") is not True
    ):
        raise Ugi3AldehydeHeadCapabilityError(
            "Ugi assembly qualification does not pass every measured product"
        )
    heads = {
        record.get("component_label"): record
        for record in head_audit
        if isinstance(record, dict)
        and isinstance(record.get("component_label"), str)
    }
    if len(heads) != expected["agile_head_candidate_overlap"]:
        raise Ugi3AldehydeHeadCapabilityError(
            "Ugi assembly qualification does not cover every AGILE head"
        )
    a5 = heads.get("A5")
    if (
        a5 is None
        or a5.get("raw_required_handle_matches") != 3
        or a5.get("symmetry_distinct_required_handle_matches") != 1
        or a5.get("passes_qualified_amine_multiplicity") is not True
        or a5.get("products_reconstructed_exactly") != a5.get("rows")
    ):
        raise Ugi3AldehydeHeadCapabilityError(
            "Ugi assembly qualification does not resolve A5 by symmetry and reconstruction"
        )
    explicit_site_heads = sorted(
        record["component_label"]
        for record in heads.values()
        if record.get("requires_explicit_site_selection") is True
    )
    required_explicit_site_heads = sorted(
        scope.get("required_explicit_site_selection_heads", [])
    )
    if explicit_site_heads != required_explicit_site_heads:
        raise Ugi3AldehydeHeadCapabilityError(
            "Ugi assembly qualification explicit-site heads disagree with config"
        )
    return {
        "schema_version": qualification["schema_version"],
        "summary": summary,
        "multiplicity_policy": policy,
        "row_qualification_digest_sha256": qualification.get(
            "row_qualification_digest_sha256"
        ),
        "failed_row_labels": qualification.get("failed_row_labels"),
        "qualified_head_findings": {
            label: heads[label]
            for label in sorted({"A5", *required_explicit_site_heads})
        },
        "qa_flags": qa_flags,
        "decision": decision,
    }


def _aldehyde_route_index(
    agile: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    family_ids = set(config["aldehyde_scope"]["route_family_ids"])
    routes = agile.get("routes")
    if not isinstance(routes, list):
        raise Ugi3AldehydeHeadCapabilityError("AGILE route artifact has no routes")
    selected = [route for route in routes if route.get("route_family_id") in family_ids]
    _expect_count(
        len(selected),
        config["expected_counts"]["structure_resolved_agile_aldehyde_routes"],
        label="structure-resolved AGILE aldehyde routes",
    )
    by_target: dict[str, list[str]] = {}
    audit: list[dict[str, Any]] = []
    route_ids: set[str] = set()
    for route in selected:
        route_id = route.get("route_id")
        if not isinstance(route_id, str) or not route_id or route_id in route_ids:
            raise Ugi3AldehydeHeadCapabilityError(
                "AGILE aldehyde routes contain a missing or duplicate route_id"
            )
        route_ids.add(route_id)
        target = _canonical_smiles(
            route.get("target", {}).get("canonical_smiles"),
            label=f"AGILE route {route_id} target",
        )
        by_target.setdefault(target, []).append(route_id)
        audit.append(
            {
                "route_id": route_id,
                "route_family_id": route["route_family_id"],
                "component_label": route.get("component_label"),
                "canonical_smiles": target,
                "reaction_steps": len(route.get("steps", [])),
                "route_evidence_status": route.get("route_evidence_status"),
                "forward_verification_status": route.get(
                    "forward_verification_status"
                ),
                "execution_closure_status": route.get("execution_closure_status"),
            }
        )
    return by_target, sorted(audit, key=lambda record: record["route_id"])


def _aldehyde_candidate_audit(
    candidates: list[dict[str, Any]],
    *,
    route_index: dict[str, list[str]],
    source_labels: dict[str, list[str]],
    component_by_label: dict[str, dict[str, Any]],
    role: dict[str, Any],
    handle: Chem.Mol,
    forbidden: list[Chem.Mol],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    scope = config["aldehyde_scope"]
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        canonical = candidate["canonical_smiles"]
        labels = sorted(
            label
            for label in source_labels.get(canonical, [])
            if label.startswith("B")
        )
        route_ids = sorted(route_index.get(canonical, []))
        source_statuses = sorted(
            {
                component_by_label[label].get("route_evidence_status", "not_assessed")
                for label in labels
            }
        )
        if route_ids:
            transformation_evidence = scope["exact_route_transformation_evidence"]
            source_route_status = "route_extracted"
        elif "source_discrepancy_unresolved" in source_statuses:
            transformation_evidence = scope["unresolved_transformation_evidence"]
            source_route_status = "source_discrepancy_unresolved"
        else:
            transformation_evidence = scope["unresolved_transformation_evidence"]
            source_route_status = "not_assessed"
        provenance = candidate.get("provenance", {})
        observation = (
            ["agile_measured_component"]
            if labels
            else (
                ["programmatic_candidate"]
                if provenance.get("source") == "programmatic_rational"
                else []
            )
        )
        records.append(
            {
                "block_id": candidate["block_id"],
                "canonical_smiles": canonical,
                "candidate_pool_provenance": provenance,
                "descriptors": candidate.get("descriptors", {}),
                "exact_agile_component_labels": labels,
                "exact_source_route_ids": route_ids,
                "source_route_evidence_status": source_route_status,
                "transformation_evidence": transformation_evidence,
                "component_observation": observation,
                **_handle_audit(
                    canonical,
                    role,
                    handle,
                    forbidden,
                    multiplicity_semantics=config["registry_scope"][
                        "multiplicity_semantics"
                    ],
                ),
                "route_closure": scope["route_closure"],
                "operational_availability": scope["operational_availability"],
                "prospective_outcome": scope["prospective_outcome"],
                "forward_verification_status": scope["forward_verification_status"],
            }
        )
    return records


def _head_candidate_audit(
    candidates: list[dict[str, Any]],
    *,
    agile_head_labels: dict[str, list[str]],
    procurement_by_label: dict[str, dict[str, Any]],
    procurement_snapshot_audit: dict[str, Any],
    miao_head_smiles: set[str],
    role: dict[str, Any],
    handle: Chem.Mol,
    forbidden: list[Chem.Mol],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    scope = config["head_scope"]
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        canonical = candidate["canonical_smiles"]
        labels = sorted(
            label
            for label in agile_head_labels.get(canonical, [])
            if label.startswith("A")
        )
        miao_observed = canonical in miao_head_smiles
        observations: list[str] = []
        if labels:
            observations.append("agile_measured_component")
        if miao_observed:
            observations.append("cross_assembly_component")
        historical_vendor = bool(labels or miao_observed)
        procurement_records = [procurement_by_label[label] for label in labels]
        if len(procurement_records) > 1:
            raise Ugi3AldehydeHeadCapabilityError(
                f"head candidate {candidate['block_id']} matches multiple procurement records"
            )
        procurement_record = procurement_records[0] if procurement_records else None
        procurement_closed = bool(
            procurement_record
            and procurement_record["current_item_level_procurement_closed"]
        )
        procurement_status = (
            procurement_record["procurement_status"]
            if procurement_record
            else scope["default_current_procurement_status"]
        )
        if procurement_closed:
            operational_availability = "current_item_level_vendor_verified"
        elif historical_vendor and procurement_record is None:
            operational_availability = "historical_blanket_vendor_claim"
        else:
            operational_availability = "route_or_procurement_resolution_required"
        procurement_evidence = None
        if procurement_record:
            procurement_evidence = {
                "component_label": procurement_record["component_label"],
                "snapshot_accessed_utc": procurement_snapshot_audit["accessed_utc"],
                "snapshot_valid_through_utc": procurement_snapshot_audit[
                    "valid_through_utc"
                ],
                "region": procurement_snapshot_audit["region"],
                "identity": procurement_record["identity"],
                "vendor_evidence": procurement_record["vendor_evidence"],
                "backup_status": procurement_record["backup_status"],
                "required_followup": procurement_record.get("required_followup"),
                "qa_note": procurement_record.get("qa_note"),
            }
        records.append(
            {
                "block_id": candidate["block_id"],
                "canonical_smiles": canonical,
                "candidate_pool_provenance": candidate.get("provenance", {}),
                "exact_agile_component_labels": labels,
                "miao_cross_assembly_observed": miao_observed,
                "transformation_evidence": scope["transformation_evidence"],
                "component_observation": observations,
                **_handle_audit(
                    canonical,
                    role,
                    handle,
                    forbidden,
                    multiplicity_semantics=config["registry_scope"][
                        "multiplicity_semantics"
                    ],
                ),
                "upstream_route_ids": [],
                "route_closure": scope["route_closure"],
                "operational_availability": operational_availability,
                "current_item_level_procurement_closed": procurement_closed,
                "current_procurement_status": procurement_status,
                "procurement_evidence": procurement_evidence,
                "prospective_outcome": scope["prospective_outcome"],
                "forward_verification_status": scope["forward_verification_status"],
                "procurement_queue_priority": (
                    "primary_agile_campaign"
                    if labels
                    else (
                        "secondary_cross_assembly"
                        if miao_observed
                        else "candidate_pool_review"
                    )
                ),
            }
        )
    return records


def build_ugi3_aldehyde_head_capability(
    config_path: Path,
    building_block_pool_path: Path,
    agile_component_routes_path: Path,
    qualified_reactions_path: Path,
    precursor_capability_path: Path,
    agile_head_procurement_path: Path,
    assembly_qualification_path: Path,
    *,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic, non-closing audit for all aldehydes and amine heads."""

    config = _load_json(config_path, label="aldehyde/head capability config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3AldehydeHeadCapabilityError(
            f"config schema must be {CONFIG_SCHEMA_VERSION!r}"
        )
    paths = {
        "building_block_pool": building_block_pool_path,
        "agile_component_routes": agile_component_routes_path,
        "qualified_reactions": qualified_reactions_path,
        "precursor_capability": precursor_capability_path,
        "agile_head_procurement": agile_head_procurement_path,
        "assembly_qualification": assembly_qualification_path,
    }
    for name, path in paths.items():
        _verify_hash(
            path,
            config["inputs"][name]["expected_sha256"],
            label=name.replace("_", " "),
        )

    pool = _load_json(building_block_pool_path, label="building block pool")
    agile = _load_json(agile_component_routes_path, label="AGILE component routes")
    registry = _load_json(qualified_reactions_path, label="qualified reaction registry")
    precursor = _load_json(precursor_capability_path, label="precursor capability")
    procurement = _load_json(
        agile_head_procurement_path,
        label="AGILE head procurement snapshot",
    )
    qualification = _load_json(
        assembly_qualification_path,
        label="Ugi assembly qualification",
    )
    if precursor.get("schema_version") != "m0_09_ugi3_precursor_capability.v2":
        raise Ugi3AldehydeHeadCapabilityError(
            "precursor capability artifact has an unsupported schema"
        )
    state_catalog = _evidence_states(precursor)
    _validate_scope_states(config, state_catalog)
    assembly_qualification_audit = _validate_assembly_qualification(
        qualification,
        config,
    )

    aldehyde_role, amine_role = _load_registry_roles(registry, config)
    aldehyde_handle, aldehyde_forbidden = _compile_role(
        aldehyde_role,
        label="aldehyde role",
    )
    amine_handle, amine_forbidden = _compile_role(amine_role, label="amine role")
    expected = config["expected_counts"]
    aldehyde_candidates = _pool_candidates(
        pool,
        handle="aldehyde",
        expected_count=expected["aldehyde_candidates"],
    )
    head_candidates = _pool_candidates(
        pool,
        handle="amine",
        expected_count=expected["amine_candidates"],
    )
    source_labels, component_by_label = _source_component_indices(agile)
    configured_generated_utc = config.get("generated_utc")
    if generated_utc is None and (
        not isinstance(configured_generated_utc, str)
        or not configured_generated_utc
    ):
        raise Ugi3AldehydeHeadCapabilityError(
            "config generated_utc must be a nonempty ISO-8601 timestamp"
        )
    generated = _generated_timestamp(
        generated_utc
        if generated_utc is not None
        else configured_generated_utc
    )
    procurement_by_label, procurement_snapshot_audit = (
        _agile_head_procurement_index(
            procurement,
            component_by_label=component_by_label,
            head_candidates=head_candidates,
            config=config,
            generated_utc=generated,
        )
    )
    aldehyde_routes, aldehyde_route_audit = _aldehyde_route_index(agile, config)
    aldehydes = _aldehyde_candidate_audit(
        aldehyde_candidates,
        route_index=aldehyde_routes,
        source_labels=source_labels,
        component_by_label=component_by_label,
        role=aldehyde_role,
        handle=aldehyde_handle,
        forbidden=aldehyde_forbidden,
        config=config,
    )
    miao_head_smiles = {
        _canonical_smiles(record["canonical_smiles"], label="Miao head")
        for record in precursor["miao_cross_assembly_audit"]["heads"]
    }
    heads = _head_candidate_audit(
        head_candidates,
        agile_head_labels=source_labels,
        procurement_by_label=procurement_by_label,
        procurement_snapshot_audit=procurement_snapshot_audit,
        miao_head_smiles=miao_head_smiles,
        role=amine_role,
        handle=amine_handle,
        forbidden=amine_forbidden,
        config=config,
    )

    rational_aldehydes = sum(
        record["candidate_pool_provenance"].get("source") == "programmatic_rational"
        for record in aldehydes
    )
    measured_aldehydes = sum(
        record["candidate_pool_provenance"].get("source") == "agile_measured"
        for record in aldehydes
    )
    exact_aldehyde_overlap = sum(bool(record["exact_source_route_ids"]) for record in aldehydes)
    unresolved_aldehydes = sum(
        record["source_route_evidence_status"] == "source_discrepancy_unresolved"
        for record in aldehydes
    )
    route_targets_in_pool = {
        route["canonical_smiles"]
        for route in aldehyde_route_audit
        if any(
            candidate["canonical_smiles"] == route["canonical_smiles"]
            for candidate in aldehydes
        )
    }
    outside_pool = len(aldehyde_route_audit) - len(route_targets_in_pool)
    agile_head_overlap = sum(bool(record["exact_agile_component_labels"]) for record in heads)
    miao_head_overlap = sum(record["miao_cross_assembly_observed"] for record in heads)
    amine_with_handle = sum(record["required_handle_match_count"] > 0 for record in heads)
    amine_policy_pass = sum(
        record["passes_literal_registry_handle_policy"] for record in heads
    )
    amine_policy_fail = len(heads) - amine_policy_pass
    agile_policy_fail = sum(
        bool(record["exact_agile_component_labels"])
        and not record["passes_literal_registry_handle_policy"]
        for record in heads
    )
    amine_qualified_policy_pass = sum(
        record["passes_qualified_registry_handle_policy"] for record in heads
    )
    amine_qualified_policy_fail = len(heads) - amine_qualified_policy_pass
    agile_qualified_policy_fail = sum(
        bool(record["exact_agile_component_labels"])
        and not record["passes_qualified_registry_handle_policy"]
        for record in heads
    )
    head_upstream_routes = sum(bool(record["upstream_route_ids"]) for record in heads)
    head_procurement_closed = sum(
        record["current_item_level_procurement_closed"] for record in heads
    )
    head_procurement_or_route_resolution_required = sum(
        not record["current_item_level_procurement_closed"]
        for record in heads
    )
    head_historical_vendor_only = sum(
        record["operational_availability"] == "historical_blanket_vendor_claim"
        for record in heads
    )
    head_catalog_availability_unresolved = sum(
        record["current_procurement_status"]
        == config["head_scope"]["unresolved_catalog_status"]
        for record in heads
    )
    route_complete = sum(
        record["route_closure"] == "computationally_complete"
        for record in [*aldehydes, *heads]
    )

    count_checks = {
        "rational_aldehyde_candidates": rational_aldehydes,
        "agile_measured_aldehyde_candidates": measured_aldehydes,
        "exact_route_aldehyde_candidate_overlap": exact_aldehyde_overlap,
        "source_resolved_aldehyde_routes_outside_pool": outside_pool,
        "unresolved_aldehyde_source_discrepancies": unresolved_aldehydes,
        "agile_head_candidate_overlap": agile_head_overlap,
        "miao_head_candidate_overlap": miao_head_overlap,
        "amine_candidates_with_required_handle": amine_with_handle,
        "amine_candidates_passing_raw_registry_multiplicity": amine_policy_pass,
        "amine_candidates_failing_raw_registry_multiplicity": amine_policy_fail,
        "agile_heads_failing_raw_registry_multiplicity": agile_policy_fail,
        "amine_candidates_passing_qualified_site_multiplicity": (
            amine_qualified_policy_pass
        ),
        "amine_candidates_failing_qualified_site_multiplicity": (
            amine_qualified_policy_fail
        ),
        "agile_heads_failing_qualified_site_multiplicity": (
            agile_qualified_policy_fail
        ),
        "agile_measured_products_passing_qualified_site_multiplicity": (
            assembly_qualification_audit["summary"][
                "products_passing_qualified_amine_multiplicity"
            ]
        ),
        "agile_head_procurement_records": procurement_snapshot_audit["summary"][
            "records"
        ],
        "agile_heads_current_item_level_vendor_verified": (
            procurement_snapshot_audit["summary"][
                "current_item_level_vendor_verified"
            ]
        ),
        "agile_heads_catalog_item_verified_availability_unresolved": (
            procurement_snapshot_audit["summary"][
                "catalog_item_verified_availability_unresolved"
            ]
        ),
        "agile_head_procurement_identity_or_form_discrepancies": (
            procurement_snapshot_audit["summary"]["identity_or_form_discrepancies"]
        ),
        "head_candidates_with_upstream_route": head_upstream_routes,
        "head_candidates_with_current_procurement_closure": head_procurement_closed,
        "head_candidates_requiring_procurement_or_route_resolution": (
            head_procurement_or_route_resolution_required
        ),
        "computationally_route_complete_candidates": route_complete,
    }
    for name, observed in count_checks.items():
        _expect_count(observed, expected[name], label=name.replace("_", " "))

    a5_matches = [
        record
        for record in heads
        if "A5" in record["exact_agile_component_labels"]
    ]
    if (
        len(a5_matches) != 1
        or a5_matches[0]["passes_literal_registry_handle_policy"]
        or not a5_matches[0]["passes_qualified_registry_handle_policy"]
        or a5_matches[0]["raw_required_handle_match_count"] != 3
        or a5_matches[0]["symmetry_distinct_required_handle_match_count"] != 1
    ):
        raise Ugi3AldehydeHeadCapabilityError(
            "A5 must fail the raw diagnostic and pass the symmetry-qualified gate"
        )
    b5 = component_by_label.get("B5")
    if b5 is None or b5.get("route_evidence_status") != "source_discrepancy_unresolved":
        raise Ugi3AldehydeHeadCapabilityError(
            "B5 must preserve its unresolved source discrepancy"
        )
    required_unresolved_labels = config["head_scope"].get(
        "required_unresolved_procurement_labels"
    )
    if (
        not isinstance(required_unresolved_labels, list)
        or any(
            not isinstance(label, str) or not label
            for label in required_unresolved_labels
        )
    ):
        raise Ugi3AldehydeHeadCapabilityError(
            "head_scope.required_unresolved_procurement_labels must be a string list"
        )
    unresolved_procurement_records = [
        record
        for record in procurement_by_label.values()
        if record["procurement_status"]
        == config["head_scope"]["unresolved_catalog_status"]
    ]
    if {record["component_label"] for record in unresolved_procurement_records} != set(
        required_unresolved_labels
    ):
        raise Ugi3AldehydeHeadCapabilityError(
            "required unresolved procurement labels do not match the snapshot"
        )

    isocyanides = precursor.get("summary", {}).get(
        "rational_isocyanide_candidates"
    )
    _expect_count(
        isocyanides,
        expected["isocyanide_candidates"],
        label="isocyanide candidates",
    )
    raw_cartesian = len(aldehydes) * len(heads) * isocyanides
    raw_registry_filtered = len(aldehydes) * amine_policy_pass * isocyanides
    qualified_registry_filtered = (
        len(aldehydes) * amine_qualified_policy_pass * isocyanides
    )
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
                qualified_reactions_path,
                precursor_capability_path,
                agile_head_procurement_path,
                assembly_qualification_path,
            )
        ],
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
        },
        "summary": {
            "aldehyde_candidates": len(aldehydes),
            "rational_aldehyde_candidates": rational_aldehydes,
            "agile_measured_aldehyde_candidates": measured_aldehydes,
            "structure_resolved_agile_aldehyde_routes": len(aldehyde_route_audit),
            "exact_route_aldehyde_candidate_overlap": exact_aldehyde_overlap,
            "source_resolved_aldehyde_routes_outside_pool": outside_pool,
            "unresolved_aldehyde_source_discrepancies": unresolved_aldehydes,
            "aldehyde_candidates_passing_literal_registry_handle_policy": sum(
                record["passes_literal_registry_handle_policy"] for record in aldehydes
            ),
            "amine_candidates": len(heads),
            "agile_head_candidate_overlap": agile_head_overlap,
            "miao_head_candidate_overlap": miao_head_overlap,
            "amine_candidates_with_required_handle": amine_with_handle,
            "amine_candidates_passing_raw_registry_multiplicity": amine_policy_pass,
            "amine_candidates_failing_raw_registry_multiplicity": amine_policy_fail,
            "agile_heads_failing_raw_registry_multiplicity": agile_policy_fail,
            "amine_candidates_passing_qualified_site_multiplicity": (
                amine_qualified_policy_pass
            ),
            "amine_candidates_failing_qualified_site_multiplicity": (
                amine_qualified_policy_fail
            ),
            "agile_heads_failing_qualified_site_multiplicity": (
                agile_qualified_policy_fail
            ),
            "agile_measured_products_passing_qualified_site_multiplicity": (
                assembly_qualification_audit["summary"][
                    "products_passing_qualified_amine_multiplicity"
                ]
            ),
            "agile_head_procurement_records": procurement_snapshot_audit["summary"][
                "records"
            ],
            "agile_heads_current_item_level_vendor_verified": (
                procurement_snapshot_audit["summary"][
                    "current_item_level_vendor_verified"
                ]
            ),
            "agile_heads_catalog_item_verified_availability_unresolved": (
                procurement_snapshot_audit["summary"][
                    "catalog_item_verified_availability_unresolved"
                ]
            ),
            "agile_head_procurement_identity_or_form_discrepancies": (
                procurement_snapshot_audit["summary"][
                    "identity_or_form_discrepancies"
                ]
            ),
            "head_candidates_with_historical_vendor_evidence_only": (
                head_historical_vendor_only
            ),
            "head_candidates_with_catalog_item_but_unresolved_availability": (
                head_catalog_availability_unresolved
            ),
            "head_candidates_with_upstream_route": head_upstream_routes,
            "head_candidates_with_current_procurement_closure": head_procurement_closed,
            "head_candidates_requiring_procurement_or_route_resolution": (
                head_procurement_or_route_resolution_required
            ),
            "computationally_route_complete_candidates": route_complete,
            "raw_component_cartesian_upper_bound": raw_cartesian,
            "raw_triples_remaining_under_literal_registry_head_filter": (
                raw_registry_filtered
            ),
            "raw_triples_remaining_under_qualified_registry_head_filter": (
                qualified_registry_filtered
            ),
        },
        "registry_scope": {
            **config["registry_scope"],
            "aldehyde_required_handle_smarts": aldehyde_role[
                "required_handle_smarts"
            ],
            "aldehyde_allowed_site_multiplicity": aldehyde_role[
                "allowed_site_multiplicity"
            ],
            "amine_required_handle_smarts": amine_role["required_handle_smarts"],
            "amine_allowed_site_multiplicity": amine_role[
                "allowed_site_multiplicity"
            ],
        },
        "claims_boundary": config["claims_boundary"],
        "qa_flags": [
            {
                "flag_id": "agile_b5_source_identity_conflict",
                "status": "unresolved_source_to_measured_identity_discrepancy",
                "component_label": "B5",
                "detail": b5.get("qa_reason", ""),
                "must_not_infer_route": True,
            },
            *assembly_qualification_audit["qa_flags"],
            *[
                {
                    "flag_id": (
                        f"agile_{record['component_label'].lower()}_"
                        "current_availability_unresolved"
                    ),
                    "status": config["head_scope"]["unresolved_catalog_status"],
                    "component_label": record["component_label"],
                    "canonical_smiles": record["canonical_smiles"],
                    "vendor_evidence": record["vendor_evidence"],
                    "required_followup": record["required_followup"],
                    "must_not_close_procurement": True,
                }
                for record in sorted(
                    unresolved_procurement_records,
                    key=lambda item: item["component_label"],
                )
            ],
        ],
        "agile_assembly_qualification_audit": assembly_qualification_audit,
        "agile_head_procurement_snapshot_audit": procurement_snapshot_audit,
        "aldehyde_source_route_audit": [
            {
                **route,
                "exact_candidate_pool_overlap": (
                    route["canonical_smiles"] in route_targets_in_pool
                ),
            }
            for route in aldehyde_route_audit
        ],
        "aldehyde_candidate_audit": aldehydes,
        "head_candidate_audit": heads,
        "head_procurement_queue": [
            {
                "block_id": record["block_id"],
                "canonical_smiles": record["canonical_smiles"],
                "exact_agile_component_labels": record[
                    "exact_agile_component_labels"
                ],
                "candidate_pool_provenance": record["candidate_pool_provenance"],
                "priority": record["procurement_queue_priority"],
                "current_procurement_status": record["current_procurement_status"],
                "procurement_evidence": record["procurement_evidence"],
                "required_followup": (
                    record["procurement_evidence"].get("required_followup")
                    if record["procurement_evidence"]
                    else None
                ),
                "required_review_fields": (
                    ["current stock or lead time", "backup supplier or upstream route"]
                    if record["procurement_evidence"]
                    else [
                        "exact constitutional identity",
                        "salt and protection state",
                        "region and supplier",
                        "catalog identifier",
                        "purity",
                        "current stock or lead time",
                        "backup supplier or upstream route",
                    ]
                ),
            }
            for record in sorted(
                heads,
                key=lambda record: (
                    {
                        "primary_agile_campaign": 0,
                        "secondary_cross_assembly": 1,
                        "candidate_pool_review": 2,
                    }[record["procurement_queue_priority"]],
                    record["block_id"],
                ),
            )
            if not record["current_item_level_procurement_closed"]
        ],
        "decision": {
            "model_built": False,
            "aldehyde_conclusion": (
                "Exact AGILE routes cover 11 of 51 current aldehyde candidates; "
                "the other source-resolved aldehydes broaden transformation "
                "evidence but do not close current pool candidates by analogy."
            ),
            "head_conclusion": (
                f"All {len(heads)} heads contain an N-H handle. "
                f"{head_procurement_closed} of the {len(procurement_by_label)} "
                "AGILE heads have time-stamped item-level procurement closure; "
                f"{', '.join(required_unresolved_labels)} has an exact catalog "
                "item but unresolved current availability. No head has an "
                f"extracted upstream route. {amine_qualified_policy_pass} of "
                f"{len(heads)} heads pass the symmetry-qualified site policy."
            ),
            "registry_action": (
                "Use symmetry-distinct reactive-site multiplicity for the Ugi "
                "amine gate. A5 is resolved without an identity exception; A19 "
                "and A20 require explicit reacting-site selection."
            ),
        },
    }


def write_ugi3_aldehyde_head_capability(
    result: dict[str, Any],
    output_path: Path,
) -> None:
    """Atomically write the validated aldehyde/head capability artifact."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write((json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
