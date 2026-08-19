"""Audit AGILE virtual products through exact Ugi decomposition and L2/L3 joins."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import platform
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.core.io import csv_gz_bytes as _csv_bytes
from forge.data.r1_prime_audit import (
    CompiledReaction,
    DecompositionCandidate,
    compile_reactions,
    decompose_structure,
    load_reaction_definitions,
    sha256_bytes,
    sha256_file,
)

CONFIG_SCHEMA_VERSION = "m0_09_agile_virtual_ugi3_capability_config.v1"
RESULT_SCHEMA_VERSION = "m0_09_agile_virtual_ugi3_capability.v1"
VIRTUAL_MANIFEST_SCHEMA_VERSION = "m0_09_agile_virtual_smiles_manifest.v1"
ASSEMBLY_SCHEMA_VERSION = "m0_09_ugi3_assembly_qualification.v1"
PRECURSOR_SCHEMA_VERSION = "m0_09_ugi3_precursor_capability.v2"
ALDEHYDE_HEAD_SCHEMA_VERSION = "m0_09_ugi3_aldehyde_head_capability.v3"
AGILE_COMPONENT_ROUTES_SCHEMA_VERSION = "m0_09_agile_component_routes.v1"
ROLE_ORDER = (
    "amine_head",
    "oxoester_aldehyde_body_tail",
    "isocyanide_tail",
)
SOURCE_ROUTE_ROLE_BY_FAMILY = {
    "agile_hexyl_aldehyde_two_step": "oxoester_aldehyde_body_tail",
    "agile_butyl_aldehyde_two_step": "oxoester_aldehyde_body_tail",
    "agile_direct_aldehyde_oxidation": "oxoester_aldehyde_body_tail",
    "agile_isocyanide_two_step": "isocyanide_tail",
    "agile_additional_isocyanide_two_step": "isocyanide_tail",
}
PRODUCT_LEDGER_FIELDS = (
    "source_row_index",
    "canonical_product_smiles",
    "decomposition_status",
    "candidate_count",
    "candidate_routes_json",
    "has_complete_l2_l3_candidate",
)
COMPONENT_LEDGER_FIELDS = (
    "component_id",
    "role",
    "canonical_smiles",
    "product_count",
    "candidate_occurrence_count",
    "evidence_match",
    "matched_record_ids_json",
    "transformation_evidence_json",
    "has_exact_source_route",
    "route_closure_states_json",
    "operational_availability_states_json",
    "accepted_terminal",
    "computationally_route_complete",
    "l2_l3_closed",
    "routing_action",
)


class Ugi3VirtualCapabilityError(ValueError):
    """Raised when the virtual capability census violates its frozen contract."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise Ugi3VirtualCapabilityError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise Ugi3VirtualCapabilityError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Ugi3VirtualCapabilityError(f"{label} must contain a JSON object")
    return value


def _verify_hash(path: Path, expected: Any, *, label: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise Ugi3VirtualCapabilityError(f"{label} expected_sha256 must be a 64-character string")
    if not path.exists():
        raise Ugi3VirtualCapabilityError(f"{label} not found: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise Ugi3VirtualCapabilityError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )
    return observed


def _canonicalize(molecule: Chem.Mol, *, label: str) -> str:
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(molecule)
        if len(Chem.GetMolFrags(molecule)) != 1:
            raise Ugi3VirtualCapabilityError(f"{label} is disconnected")
        return Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        )
    except Ugi3VirtualCapabilityError:
        raise
    except Exception as exc:
        raise Ugi3VirtualCapabilityError(f"{label} could not be sanitized") from exc


def _load_virtual_rows(
    virtual_smiles_path: Path,
    virtual_manifest: Mapping[str, Any],
) -> list[dict[str, str]]:
    if virtual_manifest.get("schema_version") != VIRTUAL_MANIFEST_SCHEMA_VERSION:
        raise Ugi3VirtualCapabilityError("virtual SMILES manifest has an unsupported schema")
    if virtual_manifest.get("output", {}).get("sha256") != sha256_file(virtual_smiles_path):
        raise Ugi3VirtualCapabilityError("virtual SMILES manifest does not match the derived table")
    required = {
        "source_row_index",
        "source_smiles",
        "canonical_isomeric_smiles",
    }
    try:
        handle = gzip.open(virtual_smiles_path, "rt", newline="")
        with handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or set(reader.fieldnames) != required:
                raise Ugi3VirtualCapabilityError("virtual SMILES table has an unexpected schema")
            rows = list(reader)
    except (gzip.BadGzipFile, csv.Error, OSError) as exc:
        raise Ugi3VirtualCapabilityError(f"virtual SMILES table could not be read: {exc}") from exc
    observed_indices = [int(row["source_row_index"]) for row in rows]
    if observed_indices != list(range(len(rows))):
        raise Ugi3VirtualCapabilityError(
            "virtual SMILES source_row_index must be contiguous and ordered"
        )
    if len({row["canonical_isomeric_smiles"] for row in rows}) != len(rows):
        raise Ugi3VirtualCapabilityError(
            "virtual SMILES table contains duplicate canonical products"
        )
    return rows


def _validate_assembly_policy(
    assembly: Mapping[str, Any],
    scope: Mapping[str, Any],
) -> None:
    if assembly.get("schema_version") != ASSEMBLY_SCHEMA_VERSION:
        raise Ugi3VirtualCapabilityError("assembly qualification has an unsupported schema")
    policy = assembly.get("multiplicity_policy")
    decision = assembly.get("decision")
    override = scope["role_policy_overrides"][0]
    if (
        not isinstance(policy, dict)
        or not isinstance(decision, dict)
        or policy.get("reaction_id") != override["reaction_id"]
        or policy.get("role") != override["role"]
        or policy.get("semantics") != override["site_multiplicity_semantics"]
        or decision.get("all_measured_products_pass") is not True
        or assembly.get("failed_row_labels") != []
    ):
        raise Ugi3VirtualCapabilityError(
            "assembly qualification does not support the role-policy override"
        )


def _reacting_site_record(
    amine: Chem.Mol,
    atom_index: int,
    symmetry_ranks: Sequence[int],
) -> dict[str, Any]:
    if atom_index < 0 or atom_index >= amine.GetNumAtoms():
        raise Ugi3VirtualCapabilityError(
            f"forward product reported invalid reacting atom index {atom_index}"
        )
    annotated = Chem.Mol(amine)
    annotated.GetAtomWithIdx(atom_index).SetAtomMapNum(1)
    return {
        "canonical_atom_index": atom_index,
        "symmetry_class": int(symmetry_ranks[atom_index]),
        "atom_mapped_amine_smiles": Chem.MolToSmiles(
            annotated,
            canonical=True,
            isomericSmiles=True,
        ),
    }


def _forward_site_audit(
    reaction: CompiledReaction,
    candidate: DecompositionCandidate,
    target: str,
    max_outcomes: int,
) -> dict[str, Any]:
    reactants = tuple(Chem.MolFromSmiles(smiles) for smiles in candidate.reactant_smiles)
    if any(molecule is None for molecule in reactants):
        raise Ugi3VirtualCapabilityError("decomposition candidate contains invalid reactant SMILES")
    typed_reactants: tuple[Chem.Mol, ...] = reactants  # type: ignore[assignment]
    with rdBase.BlockLogs():
        outcomes = reaction.forward.RunReactants(
            typed_reactants,
            maxProducts=max_outcomes,
        )
    if len(outcomes) >= max_outcomes:
        raise Ugi3VirtualCapabilityError(f"forward site audit reached max_outcomes={max_outcomes}")
    unique_products: set[str] = set()
    target_site_records: dict[str, dict[str, Any]] = {}
    target_matching_outcomes = 0
    amine = typed_reactants[0]
    symmetry_ranks = tuple(Chem.CanonicalRankAtoms(amine, breakTies=False))
    for outcome_index, outcome in enumerate(outcomes):
        if len(outcome) != 1:
            raise Ugi3VirtualCapabilityError(
                f"forward outcome {outcome_index} does not contain one product"
            )
        product = outcome[0]
        canonical = _canonicalize(
            product,
            label=f"forward outcome {outcome_index}",
        )
        unique_products.add(canonical)
        if canonical != target:
            continue
        target_matching_outcomes += 1
        reacting_atoms = [
            atom
            for atom in product.GetAtoms()
            if atom.HasProp("old_mapno")
            and atom.GetIntProp("old_mapno") == 1
            and atom.HasProp("react_idx")
            and atom.GetIntProp("react_idx") == 0
            and atom.HasProp("react_atom_idx")
        ]
        if len(reacting_atoms) != 1:
            raise Ugi3VirtualCapabilityError(
                "target-matching forward outcome does not identify one amine atom"
            )
        atom_index = reacting_atoms[0].GetIntProp("react_atom_idx")
        record = _reacting_site_record(
            amine,
            atom_index,
            symmetry_ranks,
        )
        target_site_records[record["atom_mapped_amine_smiles"]] = record
    if target_matching_outcomes == 0:
        raise Ugi3VirtualCapabilityError(
            "decomposition candidate failed the explicit forward-site round trip"
        )
    return {
        "raw_forward_outcomes": len(outcomes),
        "unique_forward_products": len(unique_products),
        "target_matching_forward_outcomes": target_matching_outcomes,
        "distinct_target_reacting_sites": len(target_site_records),
        "reacting_amine_sites": [target_site_records[key] for key in sorted(target_site_records)],
    }


def _evidence_index(
    precursor: Mapping[str, Any],
    aldehyde_head: Mapping[str, Any],
    agile_component_routes: Mapping[str, Any],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    if precursor.get("schema_version") != PRECURSOR_SCHEMA_VERSION:
        raise Ugi3VirtualCapabilityError("precursor capability has an unsupported schema")
    if aldehyde_head.get("schema_version") != ALDEHYDE_HEAD_SCHEMA_VERSION:
        raise Ugi3VirtualCapabilityError("aldehyde/head capability has an unsupported schema")
    if agile_component_routes.get("schema_version") != AGILE_COMPONENT_ROUTES_SCHEMA_VERSION:
        raise Ugi3VirtualCapabilityError("AGILE component routes have an unsupported schema")
    head_records = aldehyde_head.get("head_candidate_audit")
    aldehyde_records = aldehyde_head.get("aldehyde_candidate_audit")
    isocyanide_records = precursor.get("isocyanide_candidates")
    sources = {
        "amine_head": list(head_records) if isinstance(head_records, list) else head_records,
        "oxoester_aldehyde_body_tail": list(aldehyde_records)
        if isinstance(aldehyde_records, list)
        else aldehyde_records,
        "isocyanide_tail": list(isocyanide_records)
        if isinstance(isocyanide_records, list)
        else isocyanide_records,
    }
    routes = agile_component_routes.get("routes")
    if not isinstance(routes, list):
        raise Ugi3VirtualCapabilityError(
            "AGILE component route registry must contain a routes list"
        )
    for route in routes:
        if not isinstance(route, dict):
            raise Ugi3VirtualCapabilityError(
                "AGILE component route registry contains a malformed route"
            )
        family = route.get("route_family_id")
        role = SOURCE_ROUTE_ROLE_BY_FAMILY.get(family)
        route_id = route.get("route_id")
        target = route.get("target")
        if (
            role is None
            or not isinstance(route_id, str)
            or not route_id
            or not isinstance(target, dict)
            or not isinstance(target.get("canonical_smiles"), str)
            or route.get("route_evidence_status") != "route_extracted"
        ):
            raise Ugi3VirtualCapabilityError(
                "AGILE component route is outside the exact-route contract"
            )
        target_molecule = Chem.MolFromSmiles(target["canonical_smiles"])
        if target_molecule is None:
            raise Ugi3VirtualCapabilityError(
                f"AGILE component route {route_id!r} has an invalid target"
            )
        canonical_target = _canonicalize(
            target_molecule,
            label=f"AGILE component route {route_id!r} target",
        )
        if canonical_target != target["canonical_smiles"]:
            raise Ugi3VirtualCapabilityError(
                f"AGILE component route {route_id!r} target is not canonical"
            )
        role_records = sources[role]
        if not isinstance(role_records, list):
            raise Ugi3VirtualCapabilityError(f"capability artifact has malformed {role} records")
        role_records.append(
            {
                "block_id": f"source-route:{route_id}",
                "canonical_smiles": canonical_target,
                "transformation_evidence": ["exact_source_route"],
                "route_closure": "incomplete",
                "operational_availability": ("route_or_procurement_resolution_required"),
                "source_route_id": route_id,
                "source_route_family_id": family,
                "source_execution_closure_status": route.get("execution_closure_status"),
                "source_forward_verification_status": route.get("forward_verification_status"),
            }
        )
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for role, records in sources.items():
        if not isinstance(records, list) or any(
            not isinstance(record, dict) or not isinstance(record.get("canonical_smiles"), str)
            for record in records
        ):
            raise Ugi3VirtualCapabilityError(f"capability artifact has malformed {role} records")
        by_smiles: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_smiles[record["canonical_smiles"]].append(record)
        result[role] = dict(by_smiles)
    return result


def _component_projection(
    role: str,
    canonical_smiles: str,
    records: Sequence[Mapping[str, Any]],
    accepted_terminal_availability: set[str],
    complete_route_state: str,
) -> dict[str, Any]:
    record_ids = sorted(
        {str(record["block_id"]) for record in records if isinstance(record.get("block_id"), str)}
    )
    transformation_evidence = sorted(
        {
            str(state)
            for record in records
            for state in record.get("transformation_evidence", [])
            if isinstance(state, str)
        }
    )
    route_states = sorted(
        {
            str(record["route_closure"])
            for record in records
            if isinstance(record.get("route_closure"), str)
        }
    )
    availability_states = sorted(
        {
            str(record["operational_availability"])
            for record in records
            if isinstance(record.get("operational_availability"), str)
        }
    )
    accepted_terminal = any(
        state in accepted_terminal_availability for state in availability_states
    ) or any(record.get("current_item_level_procurement_closed") is True for record in records)
    route_complete = complete_route_state in route_states
    has_exact_source_route = "exact_source_route" in transformation_evidence
    closed = accepted_terminal or route_complete
    if closed:
        routing_action = "accepted_terminal_or_complete_route"
    elif records:
        routing_action = "continue_recursive_l2_l3_resolution"
    else:
        routing_action = "new_component_requires_evidence_and_route_search"
    component_key = f"{role}\0{canonical_smiles}".encode()
    return {
        "component_id": f"virtual-component-{hashlib.sha256(component_key).hexdigest()[:20]}",
        "role": role,
        "canonical_smiles": canonical_smiles,
        "evidence_match": bool(records),
        "matched_record_ids": record_ids,
        "transformation_evidence": transformation_evidence,
        "has_exact_source_route": has_exact_source_route,
        "route_closure_states": route_states,
        "operational_availability_states": availability_states,
        "accepted_terminal": accepted_terminal,
        "computationally_route_complete": route_complete,
        "l2_l3_closed": closed,
        "routing_action": routing_action,
    }


def _validate_frozen_counts(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    prefix: str = "summary",
) -> None:
    if set(observed) != set(expected):
        raise Ugi3VirtualCapabilityError(
            f"{prefix} fields mismatch: expected {sorted(expected)}, observed {sorted(observed)}"
        )
    for field, expected_value in expected.items():
        observed_value = observed[field]
        label = f"{prefix}.{field}"
        if isinstance(expected_value, dict):
            if not isinstance(observed_value, dict):
                raise Ugi3VirtualCapabilityError(f"{label} must be an object")
            _validate_frozen_counts(
                observed_value,
                expected_value,
                prefix=label,
            )
        elif observed_value != expected_value:
            raise Ugi3VirtualCapabilityError(
                f"{label} mismatch: expected {expected_value!r}, observed {observed_value!r}"
            )


def build_ugi3_virtual_capability(
    config_path: Path,
    virtual_smiles_path: Path,
    virtual_manifest_path: Path,
    qualified_reactions_path: Path,
    assembly_qualification_path: Path,
    precursor_capability_path: Path,
    aldehyde_head_capability_path: Path,
    agile_component_routes_path: Path,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Build product and component ledgers for all AGILE virtual candidates."""

    config = _load_json(config_path, label="virtual capability config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3VirtualCapabilityError(f"config schema must be {CONFIG_SCHEMA_VERSION!r}")
    inputs = {
        "virtual_smiles": virtual_smiles_path,
        "virtual_smiles_manifest": virtual_manifest_path,
        "qualified_reactions": qualified_reactions_path,
        "assembly_qualification": assembly_qualification_path,
        "precursor_capability": precursor_capability_path,
        "aldehyde_head_capability": aldehyde_head_capability_path,
        "agile_component_routes": agile_component_routes_path,
    }
    for name, path in inputs.items():
        _verify_hash(
            path,
            config["inputs"][name]["expected_sha256"],
            label=name.replace("_", " "),
        )
    scope = config.get("scope")
    if not isinstance(scope, dict):
        raise Ugi3VirtualCapabilityError("config scope must be an object")
    overrides = scope.get("role_policy_overrides")
    if not isinstance(overrides, list) or len(overrides) != 1:
        raise Ugi3VirtualCapabilityError(
            "virtual capability requires one qualified role-policy override"
        )
    assembly = _load_json(
        assembly_qualification_path,
        label="assembly qualification",
    )
    _validate_assembly_policy(assembly, scope)
    manifest = _load_json(
        virtual_manifest_path,
        label="virtual SMILES manifest",
    )
    rows = _load_virtual_rows(virtual_smiles_path, manifest)
    expected_source_products = config["expected_counts"].get("source_products")
    if len(rows) != expected_source_products:
        raise Ugi3VirtualCapabilityError(
            f"source product count mismatch: expected {expected_source_products}, "
            f"observed {len(rows)}"
        )

    definitions = load_reaction_definitions(
        (qualified_reactions_path,),
        expected_count=1,
        role_policy_overrides=overrides,
    )
    definition = definitions[0]
    if (
        definition.reaction_id != scope.get("reaction_id")
        or tuple(role.name for role in definition.reactant_roles) != ROLE_ORDER
    ):
        raise Ugi3VirtualCapabilityError(
            "qualified Ugi reaction does not match the declared role contract"
        )
    reaction = compile_reactions(definitions)[0]
    max_reverse = scope.get("max_reverse_outcomes_per_product")
    max_forward = scope.get("max_forward_outcomes_per_candidate")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (max_reverse, max_forward)
    ):
        raise Ugi3VirtualCapabilityError(
            "reverse and forward outcome limits must be positive integers"
        )
    precursor = _load_json(
        precursor_capability_path,
        label="precursor capability",
    )
    aldehyde_head = _load_json(
        aldehyde_head_capability_path,
        label="aldehyde/head capability",
    )
    agile_component_routes = _load_json(
        agile_component_routes_path,
        label="AGILE component routes",
    )
    evidence = _evidence_index(
        precursor,
        aldehyde_head,
        agile_component_routes,
    )
    configured_terminal_states = scope.get("accepted_terminal_availability")
    complete_route_state = scope.get("complete_route_state")
    if (
        not isinstance(configured_terminal_states, list)
        or not configured_terminal_states
        or any(not isinstance(state, str) or not state for state in configured_terminal_states)
        or not isinstance(complete_route_state, str)
        or not complete_route_state
    ):
        raise Ugi3VirtualCapabilityError(
            "closure policy must define terminal availability and route state"
        )
    accepted_terminal_availability = set(configured_terminal_states)

    rejection_counts: Counter[str] = Counter()
    product_rows: list[dict[str, Any]] = []
    component_products: dict[tuple[str, str], set[int]] = defaultdict(set)
    component_occurrences: Counter[tuple[str, str]] = Counter()
    component_projections: dict[tuple[str, str], dict[str, Any]] = {}
    component_tuples: set[tuple[str, ...]] = set()
    decomposition_profile: Counter[str] = Counter()
    forward_profile: Counter[str] = Counter()
    products_with_complete_candidate = 0
    for row in rows:
        source_row_index = int(row["source_row_index"])
        target = row["canonical_isomeric_smiles"]
        audit_row = {
            "r0_structure_id": f"AGILE-virtual-{source_row_index:05d}",
            "canonical_isomeric_smiles": target,
            "observed_source_ids": "agile_virtual12k",
            "study_split_groups_json": "{}",
        }
        candidates = decompose_structure(
            audit_row,
            "source_study",
            (reaction,),
            max_reverse,
            max_forward,
            rejection_counts,
        )
        candidate_records: list[dict[str, Any]] = []
        candidate_is_complete: list[bool] = []
        for candidate in candidates:
            component_tuples.add(candidate.reactant_smiles)
            site_audit = _forward_site_audit(
                reaction,
                candidate,
                target,
                max_forward,
            )
            forward_profile[
                "raw_"
                f"{site_audit['raw_forward_outcomes']}_"
                "unique_"
                f"{site_audit['unique_forward_products']}_"
                "target_sites_"
                f"{site_audit['distinct_target_reacting_sites']}"
            ] += 1
            components = {}
            component_closed = []
            for role, smiles in zip(
                ROLE_ORDER,
                candidate.reactant_smiles,
                strict=True,
            ):
                key = (role, smiles)
                component_products[key].add(source_row_index)
                component_occurrences[key] += 1
                projection = component_projections.setdefault(
                    key,
                    _component_projection(
                        role,
                        smiles,
                        evidence[role].get(smiles, []),
                        accepted_terminal_availability,
                        complete_route_state,
                    ),
                )
                components[role] = smiles
                component_closed.append(projection["l2_l3_closed"])
            is_complete = all(component_closed)
            candidate_is_complete.append(is_complete)
            candidate_records.append(
                {
                    "components": components,
                    **site_audit,
                    "l2_l3_closed": is_complete,
                }
            )
        if not candidates:
            decomposition_status = "no_exact_qualified_ugi_decomposition"
        elif len(candidates) == 1:
            decomposition_status = "one_exact_qualified_ugi_decomposition"
        else:
            decomposition_status = "multiple_exact_qualified_ugi_decompositions"
        decomposition_profile[decomposition_status] += 1
        has_complete_candidate = any(candidate_is_complete)
        products_with_complete_candidate += int(has_complete_candidate)
        product_rows.append(
            {
                "source_row_index": source_row_index,
                "canonical_product_smiles": target,
                "decomposition_status": decomposition_status,
                "candidate_count": len(candidates),
                "candidate_routes_json": json.dumps(
                    candidate_records,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "has_complete_l2_l3_candidate": str(has_complete_candidate).lower(),
            }
        )

    component_rows: list[dict[str, Any]] = []
    role_summary: dict[str, dict[str, int]] = {}
    for role in ROLE_ORDER:
        role_keys = sorted(key for key in component_projections if key[0] == role)
        role_summary[role] = {
            "unique_components": len(role_keys),
            "evidence_matched_components": sum(
                component_projections[key]["evidence_match"] for key in role_keys
            ),
            "unmatched_components": sum(
                not component_projections[key]["evidence_match"] for key in role_keys
            ),
            "components_with_exact_source_route": sum(
                component_projections[key]["has_exact_source_route"] for key in role_keys
            ),
            "accepted_terminal_components": sum(
                component_projections[key]["accepted_terminal"] for key in role_keys
            ),
            "computationally_route_complete_components": sum(
                component_projections[key]["computationally_route_complete"] for key in role_keys
            ),
            "l2_l3_closed_components": sum(
                component_projections[key]["l2_l3_closed"] for key in role_keys
            ),
        }
        for key in role_keys:
            projection = component_projections[key]
            component_rows.append(
                {
                    "component_id": projection["component_id"],
                    "role": role,
                    "canonical_smiles": key[1],
                    "product_count": len(component_products[key]),
                    "candidate_occurrence_count": component_occurrences[key],
                    "evidence_match": str(projection["evidence_match"]).lower(),
                    "matched_record_ids_json": json.dumps(projection["matched_record_ids"]),
                    "transformation_evidence_json": json.dumps(
                        projection["transformation_evidence"]
                    ),
                    "has_exact_source_route": str(projection["has_exact_source_route"]).lower(),
                    "route_closure_states_json": json.dumps(projection["route_closure_states"]),
                    "operational_availability_states_json": json.dumps(
                        projection["operational_availability_states"]
                    ),
                    "accepted_terminal": str(projection["accepted_terminal"]).lower(),
                    "computationally_route_complete": str(
                        projection["computationally_route_complete"]
                    ).lower(),
                    "l2_l3_closed": str(projection["l2_l3_closed"]).lower(),
                    "routing_action": projection["routing_action"],
                }
            )

    product_payload = _csv_bytes(product_rows, PRODUCT_LEDGER_FIELDS)
    component_payload = _csv_bytes(component_rows, COMPONENT_LEDGER_FIELDS)
    artifacts = {
        "agile_virtual_ugi3_product_ledger.csv.gz": product_payload,
        "agile_virtual_ugi3_component_ledger.csv.gz": component_payload,
    }
    cartesian_component_product = 1
    for role in ROLE_ORDER:
        cartesian_component_product *= role_summary[role]["unique_components"]
    if len(component_tuples) != cartesian_component_product:
        raise Ugi3VirtualCapabilityError(
            "recovered component tuples do not form a complete Cartesian product"
        )
    summary = {
        "source_products": len(rows),
        "products_with_exact_qualified_ugi_decomposition": sum(
            bool(row["candidate_count"]) for row in product_rows
        ),
        "products_without_exact_qualified_ugi_decomposition": sum(
            not bool(row["candidate_count"]) for row in product_rows
        ),
        "products_with_multiple_exact_decompositions": sum(
            row["candidate_count"] > 1 for row in product_rows
        ),
        "total_exact_decomposition_candidates": sum(row["candidate_count"] for row in product_rows),
        "products_with_complete_l2_l3_candidate": (products_with_complete_candidate),
        "unique_components_total": len(component_rows),
        "cartesian_component_tuples": len(component_tuples),
        "roles": role_summary,
    }
    _validate_frozen_counts(summary, config["expected_counts"])
    frozen_forward_profile = dict(sorted(forward_profile.items()))
    if frozen_forward_profile != config.get("expected_forward_site_profile"):
        raise Ugi3VirtualCapabilityError(
            "forward-site profile does not match the frozen expectation"
        )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": config["task"],
        "generated_utc": config["generated_utc"],
        "randomness": {"seed": 0, "used": False},
        "inputs": [
            {
                "asset": config_path.name,
                "role": "audit_config",
                "bytes": config_path.stat().st_size,
                "sha256": sha256_file(config_path),
            },
            *[
                {
                    "asset": config["inputs"][name]["asset"],
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for name, path in inputs.items()
            ],
        ],
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
        },
        "summary": summary,
        "decomposition_profile": dict(sorted(decomposition_profile.items())),
        "forward_site_profile": frozen_forward_profile,
        "rejected_reverse_outcomes": dict(sorted(rejection_counts.items())),
        "artifacts": {
            name: {
                "path": f"results/m0_09/{name}",
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
            for name, payload in artifacts.items()
        },
        "claims_boundary": config["claims_boundary"],
        "decision": {
            "product_structure_is_l1_route_evidence": False,
            "deduplicated_components_require_recursive_l2_l3_resolution": True,
            "complete_route_requires_every_component_branch_closed": True,
            "model_built": False,
        },
    }
    return result, artifacts


def write_ugi3_virtual_capability(
    result: Mapping[str, Any],
    artifacts: Mapping[str, bytes],
    output_dir: Path,
) -> None:
    """Atomically replace each deterministic capability artifact."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        **artifacts,
        "agile_virtual_ugi3_capability.json": (
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        ).encode(),
    }
    temporary_paths: dict[str, Path] = {}
    try:
        for name, payload in payloads.items():
            descriptor, temporary = tempfile.mkstemp(
                dir=output_dir,
                prefix=f".{name}.",
                suffix=".tmp",
            )
            temporary_path = Path(temporary)
            temporary_paths[name] = temporary_path
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        for name in sorted(payloads):
            os.replace(temporary_paths[name], output_dir / name)
    except Exception:
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)
        raise
