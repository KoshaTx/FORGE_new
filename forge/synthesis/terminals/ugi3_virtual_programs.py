"""Build bounded recursive synthesis programs for AGILE virtual components.

This M0 artifact turns the deduplicated component census into a structural
route-program queue. Exact source procedures and reaction-family projections
remain distinct, and no proposed leaf is treated as procurement closed.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import os
import platform
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.corpus.r1_prime_audit import sha256_bytes, sha256_file

CONFIG_SCHEMA_VERSION = "m0_09_agile_virtual_ugi3_component_programs_config.v1"
RESULT_SCHEMA_VERSION = "m0_09_agile_virtual_ugi3_component_programs.v1"
VIRTUAL_CAPABILITY_SCHEMA_VERSION = "m0_09_agile_virtual_ugi3_capability.v1"
AGILE_ROUTES_SCHEMA_VERSION = "m0_09_agile_component_routes.v1"
ALDEHYDE_ROLE = "oxoester_aldehyde_body_tail"
ISOCYANIDE_ROLE = "isocyanide_tail"
HEAD_ROLE = "amine_head"
ESTER_PROGRAM = "fatty_acid_diol_esterification_then_alcohol_oxidation"
DIRECT_ALDEHYDE_PROGRAM = "primary_alcohol_oxidation"
ISOCYANIDE_PROGRAM = "primary_amine_formylation_then_formamide_dehydration"
LEDGER_FIELDS = (
    "component_id",
    "role",
    "canonical_smiles",
    "scope_features_json",
    "program_status",
    "program_family",
    "evidence_level",
    "exact_source_route_ids_json",
    "program_steps_json",
    "proposed_intermediates_json",
    "proposed_leaf_candidates_json",
    "structural_program_status",
    "route_closure",
    "procurement_closure",
    "next_action",
)
ALDEHYDE_QUERY = Chem.MolFromSmarts("[CX3H1:1]=[OX1:2]")
ESTER_QUERY = Chem.MolFromSmarts("[CX3:1](=[OX1:2])[OX2:3][#6:4]")
ISOCYANIDE_QUERY = Chem.MolFromSmarts("[N+:1]#[C-:2]")


class Ugi3VirtualProgramError(ValueError):
    """Raised when the component-program census violates its frozen contract."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise Ugi3VirtualProgramError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise Ugi3VirtualProgramError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Ugi3VirtualProgramError(f"{label} must contain a JSON object")
    return value


def _verify_hash(path: Path, expected: Any, *, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise Ugi3VirtualProgramError(f"{label} expected_sha256 must be a 64-character string")
    observed = sha256_file(path)
    if observed != expected:
        raise Ugi3VirtualProgramError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )


def _canonicalize(molecule: Chem.Mol, *, label: str) -> str:
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(molecule)
        if len(Chem.GetMolFrags(molecule)) != 1:
            raise Ugi3VirtualProgramError(f"{label} is disconnected")
        return Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        )
    except Ugi3VirtualProgramError:
        raise
    except Exception as exc:
        raise Ugi3VirtualProgramError(f"{label} could not be sanitized") from exc


def _molecule(smiles: str, *, label: str) -> Chem.Mol:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise Ugi3VirtualProgramError(f"{label} contains invalid SMILES: {smiles!r}")
    canonical = _canonicalize(molecule, label=label)
    if canonical != smiles:
        raise Ugi3VirtualProgramError(f"{label} is not canonical: {smiles!r} != {canonical!r}")
    return molecule


def _single_match(
    molecule: Chem.Mol,
    query: Chem.Mol | None,
    *,
    label: str,
) -> tuple[int, ...]:
    if query is None:
        raise Ugi3VirtualProgramError(f"{label} SMARTS failed to compile")
    matches = molecule.GetSubstructMatches(query, uniquify=True)
    if len(matches) != 1:
        raise Ugi3VirtualProgramError(
            f"{label} requires exactly one match; observed {len(matches)}"
        )
    return matches[0]


def _scope_features(smiles: str, *, role: str) -> dict[str, Any]:
    """Return explicit, deterministic substrate-scope annotations."""

    molecule = _molecule(smiles, label=f"{role} scope component")
    carbon_carbon_double_bonds = 0
    carbon_carbon_triple_bonds = 0
    for bond in molecule.GetBonds():
        atomic_numbers = {
            bond.GetBeginAtom().GetAtomicNum(),
            bond.GetEndAtom().GetAtomicNum(),
        }
        if atomic_numbers != {6}:
            continue
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            carbon_carbon_double_bonds += 1
        elif bond.GetBondType() == Chem.BondType.TRIPLE:
            carbon_carbon_triple_bonds += 1
    if carbon_carbon_double_bonds and carbon_carbon_triple_bonds:
        carbon_unsaturation = "mixed_alkene_alkyne"
    elif carbon_carbon_triple_bonds:
        carbon_unsaturation = "alkyne"
    elif carbon_carbon_double_bonds == 1:
        carbon_unsaturation = "monoene"
    elif carbon_carbon_double_bonds > 1:
        carbon_unsaturation = "polyene"
    else:
        carbon_unsaturation = "saturated"
    carbon_branch_points = sum(
        atom.GetAtomicNum() == 6
        and sum(neighbor.GetAtomicNum() == 6 for neighbor in atom.GetNeighbors()) >= 3
        for atom in molecule.GetAtoms()
    )
    ring_count = molecule.GetRingInfo().NumRings()
    return {
        "aromatic_atom_count": sum(atom.GetIsAromatic() for atom in molecule.GetAtoms()),
        "carbon_branch_points": carbon_branch_points,
        "carbon_branching": ("branched" if carbon_branch_points else "unbranched"),
        "carbon_count": sum(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms()),
        "carbon_carbon_double_bonds": carbon_carbon_double_bonds,
        "carbon_carbon_triple_bonds": carbon_carbon_triple_bonds,
        "carbon_unsaturation": carbon_unsaturation,
        "ester_count": (
            len(molecule.GetSubstructMatches(ESTER_QUERY, uniquify=True))
            if ESTER_QUERY is not None
            else 0
        ),
        "heteroatom_count": sum(atom.GetAtomicNum() not in {1, 6} for atom in molecule.GetAtoms()),
        "nitrogen_count": sum(atom.GetAtomicNum() == 7 for atom in molecule.GetAtoms()),
        "ring_count": ring_count,
        "ring_topology": "cyclic" if ring_count else "acyclic",
    }


def _reduce_aldehyde(molecule: Chem.Mol, *, label: str) -> Chem.Mol:
    carbon, oxygen = _single_match(
        molecule,
        ALDEHYDE_QUERY,
        label=f"{label} aldehyde",
    )
    editable = Chem.RWMol(molecule)
    editable.RemoveBond(carbon, oxygen)
    editable.AddBond(carbon, oxygen, Chem.BondType.SINGLE)
    reduced = editable.GetMol()
    _canonicalize(reduced, label=f"{label} alcohol projection")
    return reduced


def _retain_fragment(
    molecule: Chem.Mol,
    *,
    bond: tuple[int, int],
    anchor: int,
    label: str,
) -> str:
    editable = Chem.RWMol(molecule)
    editable.RemoveBond(*bond)
    cut = editable.GetMol()
    fragments = Chem.GetMolFrags(cut)
    retained = next(
        (set(fragment) for fragment in fragments if anchor in fragment),
        None,
    )
    if retained is None:
        raise Ugi3VirtualProgramError(f"{label} anchor was not retained")
    fragment = Chem.RWMol(cut)
    for atom_index in sorted(
        set(range(cut.GetNumAtoms())) - retained,
        reverse=True,
    ):
        fragment.RemoveAtom(atom_index)
    return _canonicalize(fragment.GetMol(), label=label)


def _aldehyde_program(target: str) -> tuple[str, list[dict[str, Any]]]:
    molecule = _molecule(target, label="aldehyde component")
    ester_matches = (
        molecule.GetSubstructMatches(ESTER_QUERY, uniquify=True) if ESTER_QUERY is not None else ()
    )
    alcohol = _reduce_aldehyde(
        molecule,
        label="aldehyde component",
    )
    alcohol_smiles = _canonicalize(
        alcohol,
        label="aldehyde alcohol precursor",
    )
    if not ester_matches:
        return (
            DIRECT_ALDEHYDE_PROGRAM,
            [
                {
                    "step_index": 1,
                    "transformation": "alcohol_to_aldehyde_oxidation",
                    "reactants": [alcohol_smiles],
                    "product": target,
                }
            ],
        )
    if len(ester_matches) != 1:
        raise Ugi3VirtualProgramError("aldehyde component requires zero or one ester handle")
    carbonyl_carbon, _, ester_oxygen, spacer_carbon = ester_matches[0]
    acid = _retain_fragment(
        molecule,
        bond=(ester_oxygen, spacer_carbon),
        anchor=carbonyl_carbon,
        label="fatty-acid precursor",
    )
    diol = _retain_fragment(
        alcohol,
        bond=(carbonyl_carbon, ester_oxygen),
        anchor=ester_oxygen,
        label="diol precursor",
    )
    return (
        ESTER_PROGRAM,
        [
            {
                "step_index": 1,
                "transformation": "esterification",
                "reactants": sorted([acid, diol]),
                "product": alcohol_smiles,
            },
            {
                "step_index": 2,
                "transformation": "alcohol_to_aldehyde_oxidation",
                "reactants": [alcohol_smiles],
                "product": target,
            },
        ],
    )


def _isocyanide_program(target: str) -> tuple[str, list[dict[str, Any]]]:
    molecule = _molecule(target, label="isocyanide component")
    nitrogen, carbon = _single_match(
        molecule,
        ISOCYANIDE_QUERY,
        label="isocyanide component",
    )

    amine_editable = Chem.RWMol(molecule)
    amine_editable.RemoveAtom(carbon)
    amine_nitrogen = nitrogen - int(carbon < nitrogen)
    amine_atom = amine_editable.GetAtomWithIdx(amine_nitrogen)
    amine_atom.SetFormalCharge(0)
    amine_atom.SetNoImplicit(False)
    amine_atom.SetNumExplicitHs(0)
    amine = amine_editable.GetMol()
    amine.UpdatePropertyCache(strict=False)
    amine_smiles = _canonicalize(amine, label="primary-amine precursor")

    formamide_editable = Chem.RWMol(molecule)
    formamide_editable.RemoveBond(nitrogen, carbon)
    formamide_editable.AddBond(
        nitrogen,
        carbon,
        Chem.BondType.SINGLE,
    )
    for atom_index in (nitrogen, carbon):
        atom = formamide_editable.GetAtomWithIdx(atom_index)
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
    oxygen = formamide_editable.AddAtom(Chem.Atom(8))
    formamide_editable.AddBond(carbon, oxygen, Chem.BondType.DOUBLE)
    formamide = formamide_editable.GetMol()
    formamide.UpdatePropertyCache(strict=False)
    formamide_smiles = _canonicalize(
        formamide,
        label="formamide intermediate",
    )
    return (
        ISOCYANIDE_PROGRAM,
        [
            {
                "step_index": 1,
                "transformation": "amine_formylation",
                "reactants": [amine_smiles],
                "product": formamide_smiles,
            },
            {
                "step_index": 2,
                "transformation": "formamide_dehydration_to_isocyanide",
                "reactants": [formamide_smiles],
                "product": target,
            },
        ],
    )


def _load_component_rows(path: Path) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise Ugi3VirtualProgramError("component ledger has no header")
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise Ugi3VirtualProgramError(f"component ledger could not be read: {exc}") from exc
    required = {
        "component_id",
        "role",
        "canonical_smiles",
        "matched_record_ids_json",
        "has_exact_source_route",
        "accepted_terminal",
        "l2_l3_closed",
    }
    if not required.issubset(reader.fieldnames):
        raise Ugi3VirtualProgramError("component ledger is missing required fields")
    if len({row["component_id"] for row in rows}) != len(rows):
        raise Ugi3VirtualProgramError("component ledger contains duplicate component identifiers")
    return rows


def _source_routes_by_id(
    artifact: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if artifact.get("schema_version") != AGILE_ROUTES_SCHEMA_VERSION:
        raise Ugi3VirtualProgramError("AGILE component routes have an unsupported schema")
    routes = artifact.get("routes")
    if not isinstance(routes, list) or any(
        not isinstance(route, dict) or not isinstance(route.get("route_id"), str)
        for route in routes
    ):
        raise Ugi3VirtualProgramError("AGILE component routes contain malformed records")
    by_id = {route["route_id"]: route for route in routes}
    if len(by_id) != len(routes):
        raise Ugi3VirtualProgramError("AGILE component route identifiers are not unique")
    return by_id


def _canonical_step(step: Mapping[str, Any]) -> dict[str, Any]:
    reactants = step.get("reactants")
    product = step.get("product")
    if (
        not isinstance(reactants, list)
        or not isinstance(product, dict)
        or not isinstance(product.get("canonical_smiles"), str)
    ):
        raise Ugi3VirtualProgramError("source route contains a malformed reaction step")
    canonical_reactants = []
    for reactant in reactants:
        if not isinstance(reactant, dict) or not isinstance(
            reactant.get("canonical_smiles"),
            str,
        ):
            raise Ugi3VirtualProgramError("source route contains a malformed reactant")
        canonical_reactants.append(reactant["canonical_smiles"])
    return {
        "step_index": step.get("step_index"),
        "transformation": step.get("transformation"),
        "reactants": sorted(canonical_reactants),
        "product": product["canonical_smiles"],
    }


def _validate_exact_projection(
    route: Mapping[str, Any],
    program_steps: Sequence[Mapping[str, Any]],
    target: str,
) -> None:
    route_target = route.get("target")
    source_steps = route.get("steps")
    if (
        not isinstance(route_target, dict)
        or route_target.get("canonical_smiles") != target
        or not isinstance(source_steps, list)
    ):
        raise Ugi3VirtualProgramError("source route target does not match the projected component")
    normalized_source = [_canonical_step(step) for step in source_steps]
    normalized_program = [
        {
            **step,
            "reactants": sorted(step["reactants"]),
        }
        for step in program_steps
    ]
    if normalized_source != normalized_program:
        raise Ugi3VirtualProgramError(
            f"structural program did not reproduce route {route['route_id']!r}"
        )


def _program_projection(
    row: Mapping[str, str],
    source_routes: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    component_id = row["component_id"]
    role = row["role"]
    target = row["canonical_smiles"]
    try:
        matched_ids = json.loads(row["matched_record_ids_json"])
    except json.JSONDecodeError as exc:
        raise Ugi3VirtualProgramError(
            f"{component_id} has malformed matched route identifiers"
        ) from exc
    if not isinstance(matched_ids, list):
        raise Ugi3VirtualProgramError(f"{component_id} matched route identifiers must be a list")
    route_ids = sorted(
        identifier.removeprefix("source-route:")
        for identifier in matched_ids
        if isinstance(identifier, str) and identifier.startswith("source-route:")
    )
    exact = row["has_exact_source_route"] == "true"
    if exact != bool(route_ids):
        raise Ugi3VirtualProgramError(f"{component_id} exact-route state is inconsistent")

    if role == HEAD_ROLE:
        accepted = row["accepted_terminal"] == "true"
        if exact:
            raise Ugi3VirtualProgramError("head component unexpectedly carries a source route")
        return {
            "program_status": (
                "accepted_procurement_terminal"
                if accepted
                else "procurement_or_route_search_required"
            ),
            "program_family": "",
            "evidence_level": (
                "current_item_level_vendor_verified" if accepted else "no_assigned_upstream_program"
            ),
            "exact_source_route_ids": [],
            "program_steps": [],
            "proposed_intermediates": [],
            "proposed_leaf_candidates": [target],
            "structural_program_status": "not_applicable",
            "route_closure": ("accepted_terminal" if accepted else "incomplete"),
            "procurement_closure": ("accepted_terminal" if accepted else "unresolved"),
            "next_action": (
                "none" if accepted else "exact_identity_procurement_then_upstream_route_search"
            ),
        }
    if role == ALDEHYDE_ROLE:
        family, steps = _aldehyde_program(target)
    elif role == ISOCYANIDE_ROLE:
        family, steps = _isocyanide_program(target)
    else:
        raise Ugi3VirtualProgramError(f"{component_id} has unsupported role {role!r}")
    for route_id in route_ids:
        try:
            route = source_routes[route_id]
        except KeyError as exc:
            raise Ugi3VirtualProgramError(
                f"{component_id} references unknown route {route_id!r}"
            ) from exc
        _validate_exact_projection(route, steps, target)
    intermediates = [step["product"] for step in steps[:-1]]
    products = {step["product"] for step in steps}
    leaves = sorted(
        {reactant for step in steps for reactant in step["reactants"] if reactant not in products}
    )
    return {
        "program_status": (
            "exact_source_program" if exact else "reaction_family_projected_program"
        ),
        "program_family": family,
        "evidence_level": (
            "exact_source_route" if exact else "family_projection_from_exact_source_routes"
        ),
        "exact_source_route_ids": route_ids,
        "program_steps": steps,
        "proposed_intermediates": intermediates,
        "proposed_leaf_candidates": leaves,
        "structural_program_status": (
            "exact_source_structure_reproduced" if exact else "family_projection_only"
        ),
        "route_closure": "incomplete",
        "procurement_closure": "unresolved",
        "next_action": (
            "forward_verify_steps_and_close_proposed_leaves"
            if exact
            else "qualify_family_projection_then_close_proposed_leaves"
        ),
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=LEDGER_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        compressed.write(buffer.getvalue().encode())
    return output.getvalue()


def _validate_frozen_counts(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    prefix: str = "summary",
) -> None:
    if set(observed) != set(expected):
        raise Ugi3VirtualProgramError(
            f"{prefix} fields mismatch: expected {sorted(expected)}, "
            f"observed {sorted(observed)}"
        )
    for field, expected_value in expected.items():
        observed_value = observed[field]
        label = f"{prefix}.{field}"
        if isinstance(expected_value, dict):
            if not isinstance(observed_value, dict):
                raise Ugi3VirtualProgramError(f"{label} must be an object")
            _validate_frozen_counts(
                observed_value,
                expected_value,
                prefix=label,
            )
        elif observed_value != expected_value:
            raise Ugi3VirtualProgramError(
                f"{label} mismatch: expected {expected_value!r}, " f"observed {observed_value!r}"
            )


def _scope_group_counts(
    projections: Sequence[Mapping[str, Any]],
    *,
    feature: str,
) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter[str]] = {}
    for projection in projections:
        value = str(projection["scope_features"][feature])
        counts = grouped.setdefault(value, Counter())
        counts["components"] += 1
        status = projection["program_status"]
        if status == "exact_source_program":
            counts["exact_source_programs"] += 1
        elif status == "reaction_family_projected_program":
            counts["family_projected_programs"] += 1
        elif status == "accepted_procurement_terminal":
            counts["accepted_procurement_terminals"] += 1
        elif status == "procurement_or_route_search_required":
            counts["unresolved_components"] += 1
        else:
            raise Ugi3VirtualProgramError(f"unsupported program status in scope census: {status!r}")
    fields = (
        "components",
        "exact_source_programs",
        "family_projected_programs",
        "accepted_procurement_terminals",
        "unresolved_components",
    )
    return {
        value: {field: counts[field] for field in fields}
        for value, counts in sorted(grouped.items())
    }


def _substrate_scope_summary(
    role_projections: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    dimensions = (
        "carbon_unsaturation",
        "carbon_branching",
        "ring_topology",
    )
    summary: dict[str, Any] = {}
    for role, projections in role_projections.items():
        carbon_counts = [
            int(projection["scope_features"]["carbon_count"]) for projection in projections
        ]
        summary[role] = {
            "carbon_count": {
                "minimum": min(carbon_counts),
                "maximum": max(carbon_counts),
                "observed_values": sorted(set(carbon_counts)),
            },
            **{
                dimension: _scope_group_counts(
                    projections,
                    feature=dimension,
                )
                for dimension in dimensions
            },
            "ester_bearing": _scope_group_counts(
                [
                    {
                        **projection,
                        "scope_features": {
                            **projection["scope_features"],
                            "ester_bearing": (
                                "yes" if projection["scope_features"]["ester_count"] else "no"
                            ),
                        },
                    }
                    for projection in projections
                ],
                feature="ester_bearing",
            ),
        }
    return summary


def build_ugi3_virtual_component_programs(
    config_path: Path,
    component_ledger_path: Path,
    virtual_capability_path: Path,
    agile_component_routes_path: Path,
) -> tuple[dict[str, Any], bytes]:
    """Build and validate the recursive component-program work queue."""

    config = _load_json(config_path, label="component-program config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3VirtualProgramError(f"config schema must be {CONFIG_SCHEMA_VERSION!r}")
    inputs = {
        "component_ledger": component_ledger_path,
        "virtual_capability": virtual_capability_path,
        "agile_component_routes": agile_component_routes_path,
    }
    for name, path in inputs.items():
        _verify_hash(
            path,
            config["inputs"][name]["expected_sha256"],
            label=name.replace("_", " "),
        )
    capability = _load_json(
        virtual_capability_path,
        label="virtual capability",
    )
    if capability.get("schema_version") != VIRTUAL_CAPABILITY_SCHEMA_VERSION:
        raise Ugi3VirtualProgramError("virtual capability has an unsupported schema")
    source_routes = _source_routes_by_id(
        _load_json(
            agile_component_routes_path,
            label="AGILE component routes",
        )
    )
    rows = _load_component_rows(component_ledger_path)
    output_rows: list[dict[str, Any]] = []
    projections: list[dict[str, Any]] = []
    for row in rows:
        projection = _program_projection(row, source_routes)
        scope_features = _scope_features(
            row["canonical_smiles"],
            role=row["role"],
        )
        projections.append(
            {
                "role": row["role"],
                "scope_features": scope_features,
                **projection,
            }
        )
        output_rows.append(
            {
                "component_id": row["component_id"],
                "role": row["role"],
                "canonical_smiles": row["canonical_smiles"],
                "scope_features_json": json.dumps(
                    scope_features,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "program_status": projection["program_status"],
                "program_family": projection["program_family"],
                "evidence_level": projection["evidence_level"],
                "exact_source_route_ids_json": json.dumps(
                    projection["exact_source_route_ids"],
                    separators=(",", ":"),
                ),
                "program_steps_json": json.dumps(
                    projection["program_steps"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "proposed_intermediates_json": json.dumps(
                    projection["proposed_intermediates"],
                    separators=(",", ":"),
                ),
                "proposed_leaf_candidates_json": json.dumps(
                    projection["proposed_leaf_candidates"],
                    separators=(",", ":"),
                ),
                "structural_program_status": projection["structural_program_status"],
                "route_closure": projection["route_closure"],
                "procurement_closure": projection["procurement_closure"],
                "next_action": projection["next_action"],
            }
        )
    output_rows.sort(key=lambda row: (row["role"], row["canonical_smiles"]))
    statuses = Counter(projection["program_status"] for projection in projections)
    families = Counter(
        projection["program_family"] for projection in projections if projection["program_family"]
    )
    exact_reproduced = sum(
        projection["structural_program_status"] == "exact_source_structure_reproduced"
        for projection in projections
    )
    unique_intermediates = {
        intermediate
        for projection in projections
        if projection["program_family"]
        for intermediate in projection["proposed_intermediates"]
    }
    unique_leaf_candidates = {
        leaf
        for projection in projections
        if projection["program_family"]
        for leaf in projection["proposed_leaf_candidates"]
    }
    role_projections = {
        role: [projection for projection in projections if projection["role"] == role]
        for role in (HEAD_ROLE, ALDEHYDE_ROLE, ISOCYANIDE_ROLE)
    }
    summary = {
        "source_components": len(rows),
        "accepted_procurement_terminals": statuses["accepted_procurement_terminal"],
        "components_without_assigned_program": statuses["procurement_or_route_search_required"],
        "components_with_exact_source_program": statuses["exact_source_program"],
        "components_with_family_projected_program": statuses["reaction_family_projected_program"],
        "components_with_structural_program": sum(families.values()),
        "exact_source_programs_reproduced_structurally": exact_reproduced,
        "unique_proposed_intermediates": len(unique_intermediates),
        "unique_proposed_leaf_candidates": len(unique_leaf_candidates),
        "program_families": dict(sorted(families.items())),
        "roles": {
            HEAD_ROLE: {
                "components": len(role_projections[HEAD_ROLE]),
                "accepted_procurement_terminals": sum(
                    projection["program_status"] == "accepted_procurement_terminal"
                    for projection in role_projections[HEAD_ROLE]
                ),
                "without_assigned_program": sum(
                    projection["program_status"] == "procurement_or_route_search_required"
                    for projection in role_projections[HEAD_ROLE]
                ),
            },
            ALDEHYDE_ROLE: {
                "components": len(role_projections[ALDEHYDE_ROLE]),
                "exact_source_programs": sum(
                    projection["program_status"] == "exact_source_program"
                    for projection in role_projections[ALDEHYDE_ROLE]
                ),
                "family_projected_programs": sum(
                    projection["program_status"] == "reaction_family_projected_program"
                    for projection in role_projections[ALDEHYDE_ROLE]
                ),
                "route_complete_components": sum(
                    projection["route_closure"] == "computationally_complete"
                    for projection in role_projections[ALDEHYDE_ROLE]
                ),
            },
            ISOCYANIDE_ROLE: {
                "components": len(role_projections[ISOCYANIDE_ROLE]),
                "exact_source_programs": sum(
                    projection["program_status"] == "exact_source_program"
                    for projection in role_projections[ISOCYANIDE_ROLE]
                ),
                "family_projected_programs": sum(
                    projection["program_status"] == "reaction_family_projected_program"
                    for projection in role_projections[ISOCYANIDE_ROLE]
                ),
                "route_complete_components": sum(
                    projection["route_closure"] == "computationally_complete"
                    for projection in role_projections[ISOCYANIDE_ROLE]
                ),
            },
        },
        "substrate_scope": _substrate_scope_summary(role_projections),
    }
    _validate_frozen_counts(summary, config["expected_counts"])
    ledger = _csv_bytes(output_rows)
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
        "artifacts": {
            "agile_virtual_ugi3_component_program_ledger.csv.gz": {
                "path": ("results/m0_09/" "agile_virtual_ugi3_component_program_ledger.csv.gz"),
                "bytes": len(ledger),
                "sha256": sha256_bytes(ledger),
            }
        },
        "claims_boundary": config["claims_boundary"],
        "decision": {
            "project_at_unique_component_level": True,
            "source_exact_and_family_projected_programs_are_separate": True,
            "all_nonterminal_branches_require_forward_and_leaf_closure": True,
            "model_built": False,
        },
    }
    return result, ledger


def write_ugi3_virtual_component_programs(
    result: Mapping[str, Any],
    ledger: bytes,
    output_dir: Path,
) -> None:
    """Atomically replace deterministic component-program artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "agile_virtual_ugi3_component_program_ledger.csv.gz": ledger,
        "agile_virtual_ugi3_component_programs.json": (
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
