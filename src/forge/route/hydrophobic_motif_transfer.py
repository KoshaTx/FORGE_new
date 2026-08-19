"""Build the fixed M0-09 cross-platform hydrophobic-motif transfer pilot."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import platform
import tempfile
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdChemReactions, rdFingerprintGenerator

from forge.core.io import stable_json as _stable_json
from forge.data.r1_prime_audit import sha256_bytes, sha256_file

CONFIG_SCHEMA_VERSION = "m0_09_hydrophobic_motif_transfer_config.v3"
RESULT_SCHEMA_VERSION = "m0_09_hydrophobic_motif_transfer.v3"
RESULT_NAME = "hydrophobic_motif_transfer.json"
LEDGER_NAME = "hydrophobic_motif_transfer_ledger.csv.gz"
COMPLETE_ROUTE_CLOSURE = "computationally_complete"
ACCEPTED_TERMINAL_STATES = frozenset(
    {
        "current_item_level_procurement_closed",
        "internal_stock_verified",
    }
)
ALDEHYDE_QUERY = Chem.MolFromSmarts("[CX3H1]=[OX1]")
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
)
FIELDS = (
    "record_id",
    "source_id",
    "source_platform",
    "source_component_id",
    "source_locator",
    "source_reaction_id",
    "source_reaction_transformation",
    "source_reaction_conditions",
    "source_reaction_evidence_status",
    "source_reported_yield_percent",
    "common_precursor_name",
    "common_precursor_canonical_smiles",
    "common_precursor_inchikey",
    "motif_anchor_atom_map",
    "source_handle_atom_map",
    "source_handle_event",
    "source_attachment_mapping_status",
    "alcohol_class",
    "carbon_count",
    "carbon_carbon_double_bonds",
    "branch_carbon_count",
    "motif_classes_json",
    "risk_flags_json",
    "disposition",
    "proposed_ugi_role",
    "proposed_ugi_component_smiles",
    "proposed_program_id",
    "proposed_route_evidence_grade",
    "route_closure",
    "terminal_status",
    "route_outcome_category",
    "route_gap_class",
    "missing_encoded_route_knowledge",
    "observed_chemical_failure",
    "generative_support_status",
    "oracle_applicability_status",
    "route_support_status",
    "frozen_ugi_handle_matches",
    "frozen_ugi_forward_products",
    "exact_current_aldehyde_pool_match",
    "nearest_current_aldehyde_component_id",
    "nearest_current_aldehyde_smiles",
    "nearest_current_aldehyde_tanimoto",
    "computationally_route_complete",
    "biological_label_inherited",
    "defer_reason",
)


class HydrophobicMotifTransferError(ValueError):
    """Raised when the fixed transfer pilot violates its evidence contract."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise HydrophobicMotifTransferError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise HydrophobicMotifTransferError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HydrophobicMotifTransferError(f"{label} must contain a JSON object")
    return value


def _required_mapping(
    value: Any,
    *,
    label: str,
    fields: Sequence[str],
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise HydrophobicMotifTransferError(f"{label} must be an object")
    missing = set(fields) - set(value)
    if missing:
        raise HydrophobicMotifTransferError(f"{label} is missing fields: {sorted(missing)}")
    return value


def _required_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise HydrophobicMotifTransferError(f"{label} must be a nonempty string")
    return value


def _verify_hash(path: Path, expected: Any, *, label: str) -> dict[str, Any]:
    expected_digest = _required_string(
        expected,
        label=f"{label} expected_sha256",
    )
    if len(expected_digest) != 64:
        raise HydrophobicMotifTransferError(f"{label} expected_sha256 must contain 64 characters")
    try:
        observed = sha256_file(path)
        size = path.stat().st_size
    except FileNotFoundError as exc:
        raise HydrophobicMotifTransferError(f"{label} not found: {path}") from exc
    if observed != expected_digest:
        raise HydrophobicMotifTransferError(
            f"{label} hash mismatch: expected {expected_digest}, observed {observed}"
        )
    return {
        "asset": path.name,
        "bytes": size,
        "sha256": observed,
    }


def _verify_zip_member(
    archive_path: Path,
    *,
    member_name: Any,
    expected_digest: Any,
) -> dict[str, Any]:
    member = _required_string(member_name, label="source bundle member")
    digest = _required_string(
        expected_digest,
        label="source bundle member expected_sha256",
    )
    if len(digest) != 64:
        raise HydrophobicMotifTransferError(
            "source bundle member expected_sha256 must contain 64 characters"
        )
    try:
        with zipfile.ZipFile(archive_path) as archive:
            payload = archive.read(member)
    except (FileNotFoundError, KeyError, zipfile.BadZipFile) as exc:
        raise HydrophobicMotifTransferError(
            f"cannot read {member!r} from {archive_path}: {exc}"
        ) from exc
    observed = hashlib.sha256(payload).hexdigest()
    if observed != digest:
        raise HydrophobicMotifTransferError(
            f"source bundle member hash mismatch: expected {digest}, observed {observed}"
        )
    return {
        "asset": member,
        "bytes": len(payload),
        "sha256": observed,
    }


def _verify_source_assets(
    raw_assets: Any,
    *,
    source_root: Path,
) -> tuple[list[dict[str, Any]], set[str]]:
    if not isinstance(raw_assets, dict) or not raw_assets:
        raise HydrophobicMotifTransferError("source_assets must be a nonempty object")
    root = source_root.resolve()
    inputs: list[dict[str, Any]] = []
    verified_ids: set[str] = set()
    for asset_id, raw_spec in sorted(raw_assets.items()):
        asset_id = _required_string(asset_id, label="source asset id")
        spec = _required_mapping(
            raw_spec,
            label=f"source asset {asset_id}",
            fields=("asset", "expected_sha256", "role"),
        )
        relative = Path(
            _required_string(
                spec["asset"],
                label=f"source asset {asset_id} path",
            )
        )
        if relative.is_absolute() or ".." in relative.parts:
            raise HydrophobicMotifTransferError(
                f"source asset {asset_id} must use a repository-relative path"
            )
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise HydrophobicMotifTransferError(f"source asset {asset_id} escapes the source root")
        verified = _verify_hash(
            path,
            spec["expected_sha256"],
            label=f"source asset {asset_id}",
        )
        inputs.append(
            {
                **verified,
                "asset": relative.as_posix(),
                "asset_id": asset_id,
                "role": _required_string(
                    spec["role"],
                    label=f"source asset {asset_id} role",
                ),
            }
        )
        member = spec.get("member")
        member_digest = spec.get("expected_member_sha256")
        if (member is None) != (member_digest is None):
            raise HydrophobicMotifTransferError(
                f"source asset {asset_id} must specify both member and expected_member_sha256"
            )
        if member is not None:
            member_record = _verify_zip_member(
                path,
                member_name=member,
                expected_digest=member_digest,
            )
            inputs.append(
                {
                    **member_record,
                    "asset_id": f"{asset_id}:member",
                    "parent_asset_id": asset_id,
                    "role": "source_supplement_member",
                }
            )
        verified_ids.add(asset_id)
    return inputs, verified_ids


def _canonical_molecule(smiles: Any, *, label: str) -> tuple[Chem.Mol, str]:
    value = _required_string(smiles, label=label)
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        raise HydrophobicMotifTransferError(f"{label} is not valid SMILES: {value!r}")
    canonical = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )
    if canonical != value:
        raise HydrophobicMotifTransferError(f"{label} is not canonical: {value!r} != {canonical!r}")
    return molecule, canonical


def _mapped_attachment(
    mapped_smiles: Any,
    canonical_smiles: str,
    attachment: Mapping[str, Any],
    *,
    label: str,
) -> tuple[Chem.Mol, Chem.Atom, Chem.Atom]:
    value = _required_string(mapped_smiles, label=f"{label} mapped SMILES")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        raise HydrophobicMotifTransferError(f"{label} mapped SMILES is invalid")
    map_index: dict[int, Chem.Atom] = {}
    for atom in molecule.GetAtoms():
        atom_map = atom.GetAtomMapNum()
        if atom_map:
            if atom_map in map_index:
                raise HydrophobicMotifTransferError(f"{label} repeats atom map {atom_map}")
            map_index[atom_map] = atom
    anchor_map = attachment.get("motif_anchor_atom_map")
    handle_map = attachment.get("source_handle_atom_map")
    if not isinstance(anchor_map, int) or not isinstance(handle_map, int):
        raise HydrophobicMotifTransferError(f"{label} attachment maps must be integers")
    if set(map_index) != {anchor_map, handle_map}:
        raise HydrophobicMotifTransferError(
            f"{label} mapped atoms {sorted(map_index)} disagree with "
            f"declared maps {[anchor_map, handle_map]}"
        )
    anchor = map_index[anchor_map]
    handle = map_index[handle_map]
    if anchor.GetAtomicNum() != 6 or handle.GetAtomicNum() != 8:
        raise HydrophobicMotifTransferError(f"{label} must map a carbon anchor and oxygen handle")
    if molecule.GetBondBetweenAtoms(anchor.GetIdx(), handle.GetIdx()) is None:
        raise HydrophobicMotifTransferError(f"{label} mapped anchor and handle are not bonded")
    stripped = Chem.Mol(molecule)
    for atom in stripped.GetAtoms():
        atom.SetAtomMapNum(0)
    stripped_canonical = Chem.MolToSmiles(
        stripped,
        canonical=True,
        isomericSmiles=True,
    )
    if stripped_canonical != canonical_smiles:
        raise HydrophobicMotifTransferError(
            f"{label} mapped structure does not reconstruct {canonical_smiles!r}"
        )
    return molecule, anchor, handle


def _alcohol_class(anchor: Chem.Atom, handle: Chem.Atom) -> str:
    if handle.GetTotalNumHs() != 1:
        raise HydrophobicMotifTransferError("mapped source handle is not an alcohol oxygen")
    carbon_neighbors = sum(
        neighbor.GetAtomicNum() == 6
        for neighbor in anchor.GetNeighbors()
        if neighbor.GetIdx() != handle.GetIdx()
    )
    if anchor.GetTotalNumHs() == 2 and carbon_neighbors == 1:
        return "primary"
    if anchor.GetTotalNumHs() == 1 and carbon_neighbors == 2:
        return "secondary"
    raise HydrophobicMotifTransferError("mapped alcohol is neither primary nor secondary")


def _motif_descriptors(molecule: Chem.Mol) -> dict[str, int]:
    carbons = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 6]
    carbon_double_bonds = sum(
        bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetBeginAtom().GetAtomicNum() == 6
        and bond.GetEndAtom().GetAtomicNum() == 6
        for bond in molecule.GetBonds()
    )
    branch_carbons = sum(
        sum(neighbor.GetAtomicNum() == 6 for neighbor in atom.GetNeighbors()) > 2
        for atom in carbons
    )
    return {
        "carbon_count": len(carbons),
        "carbon_carbon_double_bonds": carbon_double_bonds,
        "branch_carbon_count": branch_carbons,
    }


def _read_component_pool(
    path: Path,
    *,
    role: str,
    expected_count: int,
) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"component_id", "role", "canonical_smiles"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise HydrophobicMotifTransferError(
                    "reference component ledger is missing required columns"
                )
            rows = [row for row in reader if row["role"] == role]
    except (OSError, csv.Error) as exc:
        raise HydrophobicMotifTransferError(
            f"cannot read reference component ledger: {exc}"
        ) from exc
    if len(rows) != expected_count:
        raise HydrophobicMotifTransferError(
            f"reference component ledger has {len(rows)} {role!r} rows; expected {expected_count}"
        )
    seen: set[str] = set()
    for index, row in enumerate(rows):
        _, canonical = _canonical_molecule(
            row["canonical_smiles"],
            label=f"reference component row {index}",
        )
        if canonical in seen:
            raise HydrophobicMotifTransferError(
                f"duplicate reference component structure: {canonical}"
            )
        seen.add(canonical)
    return rows


def _load_reaction(
    path: Path,
    *,
    reaction_id: str,
) -> tuple[dict[str, Any], rdChemReactions.ChemicalReaction]:
    registry = _load_json(path, label="qualified reaction registry")
    reactions = registry.get("reactions")
    if not isinstance(reactions, list):
        raise HydrophobicMotifTransferError("qualified reaction registry has no reactions list")
    matches = [
        reaction
        for reaction in reactions
        if isinstance(reaction, dict) and reaction.get("reaction_id") == reaction_id
    ]
    if len(matches) != 1:
        raise HydrophobicMotifTransferError(
            f"expected one reaction {reaction_id!r}, found {len(matches)}"
        )
    reaction_record = matches[0]
    smarts = _required_string(
        reaction_record.get("atom_mapped_reaction_smarts"),
        label=f"{reaction_id} atom_mapped_reaction_smarts",
    )
    with rdBase.BlockLogs():
        reaction = rdChemReactions.ReactionFromSmarts(smarts)
    if reaction is None:
        raise HydrophobicMotifTransferError(f"could not compile reaction {reaction_id!r}")
    roles = reaction_record.get("reactant_roles")
    role_names = (
        [role.get("name") for role in roles if isinstance(role, dict)]
        if isinstance(roles, list)
        else []
    )
    expected_roles = [
        "amine_head",
        "oxoester_aldehyde_body_tail",
        "isocyanide_tail",
    ]
    if role_names != expected_roles:
        raise HydrophobicMotifTransferError(f"{reaction_id} role order changed: {role_names}")
    return reaction_record, reaction


def _compile_programs(
    raw: Any,
    *,
    verified_asset_ids: set[str],
) -> dict[str, tuple[Mapping[str, Any], rdChemReactions.ChemicalReaction]]:
    if not isinstance(raw, dict) or not raw:
        raise HydrophobicMotifTransferError("programs must be a nonempty object")
    programs: dict[
        str,
        tuple[Mapping[str, Any], rdChemReactions.ChemicalReaction],
    ] = {}
    for program_id, record in sorted(raw.items()):
        spec = _required_mapping(
            record,
            label=f"program {program_id}",
            fields=(
                "reaction_smarts",
                "target_role",
                "evidence_grade",
                "route_closure",
                "terminal_status",
                "route_gap_class",
                "claim",
            ),
        )
        with rdBase.BlockLogs():
            reaction = rdChemReactions.ReactionFromSmarts(
                _required_string(
                    spec["reaction_smarts"],
                    label=f"program {program_id} reaction_smarts",
                )
            )
        if reaction is None:
            raise HydrophobicMotifTransferError(f"could not compile program {program_id!r}")
        route_closure = _required_string(
            spec["route_closure"],
            label=f"program {program_id} route_closure",
        )
        terminal_status = _required_string(
            spec["terminal_status"],
            label=f"program {program_id} terminal_status",
        )
        route_gap_class = spec["route_gap_class"]
        if not isinstance(route_gap_class, str):
            raise HydrophobicMotifTransferError(
                f"program {program_id} route_gap_class must be a string"
            )
        is_complete = (
            route_closure == COMPLETE_ROUTE_CLOSURE and terminal_status in ACCEPTED_TERMINAL_STATES
        )
        if is_complete:
            if route_gap_class:
                raise HydrophobicMotifTransferError(
                    f"complete program {program_id} cannot retain a route gap"
                )
            exact_evidence = _required_mapping(
                spec.get("exact_route_evidence"),
                label=f"program {program_id} exact_route_evidence",
                fields=(
                    "source_asset_ids",
                    "source_locator",
                    "reported_reactant",
                    "reported_product",
                    "conditions",
                    "reported_outcome",
                    "terminal_evidence",
                ),
            )
            asset_ids = exact_evidence["source_asset_ids"]
            if (
                not isinstance(asset_ids, list)
                or not asset_ids
                or any(not isinstance(asset_id, str) or not asset_id for asset_id in asset_ids)
            ):
                raise HydrophobicMotifTransferError(
                    f"program {program_id} exact route needs source assets"
                )
            unknown_assets = set(asset_ids) - verified_asset_ids
            if unknown_assets:
                raise HydrophobicMotifTransferError(
                    f"program {program_id} exact route references unknown "
                    f"assets: {sorted(unknown_assets)}"
                )
            for field in (
                "source_locator",
                "reported_reactant",
                "reported_product",
                "reported_outcome",
            ):
                _required_string(
                    exact_evidence[field],
                    label=f"program {program_id} exact route {field}",
                )
            if not isinstance(exact_evidence["conditions"], dict):
                raise HydrophobicMotifTransferError(
                    f"program {program_id} exact route conditions must be an object"
                )
            terminal_evidence = _required_mapping(
                exact_evidence["terminal_evidence"],
                label=f"program {program_id} terminal_evidence",
                fields=(
                    "vendor",
                    "product_code",
                    "url",
                    "region",
                    "purity",
                    "availability_observation",
                    "accessed_utc",
                ),
            )
            for field, value in terminal_evidence.items():
                _required_string(
                    value,
                    label=f"program {program_id} terminal evidence {field}",
                )
        elif route_closure == COMPLETE_ROUTE_CLOSURE or terminal_status in ACCEPTED_TERMINAL_STATES:
            raise HydrophobicMotifTransferError(
                f"program {program_id} has inconsistent route and terminal closure states"
            )
        elif not route_gap_class:
            raise HydrophobicMotifTransferError(
                f"incomplete program {program_id} must name a route gap"
            )
        programs[program_id] = (spec, reaction)
    return programs


def _validate_sources(
    raw_sources: Any,
    *,
    verified_asset_ids: set[str],
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(raw_sources, dict) or not raw_sources:
        raise HydrophobicMotifTransferError("sources must be a nonempty object")
    sources: dict[str, Mapping[str, Any]] = {}
    for source_id, raw_source in sorted(raw_sources.items()):
        source_id = _required_string(source_id, label="source id")
        source = _required_mapping(
            raw_source,
            label=f"source {source_id}",
            fields=(
                "platform_id",
                "source_asset_ids",
                "title",
                "source_final_assembly",
                "source_locator",
                "biological_provenance_policy",
                "reported_reactions",
            ),
        )
        asset_ids = source["source_asset_ids"]
        if (
            not isinstance(asset_ids, list)
            or not asset_ids
            or any(not isinstance(value, str) or not value for value in asset_ids)
        ):
            raise HydrophobicMotifTransferError(
                f"source {source_id} source_asset_ids must be nonempty strings"
            )
        unknown_assets = set(asset_ids) - verified_asset_ids
        if unknown_assets:
            raise HydrophobicMotifTransferError(
                f"source {source_id} references unknown assets: {sorted(unknown_assets)}"
            )
        for field in (
            "platform_id",
            "title",
            "source_final_assembly",
            "source_locator",
            "biological_provenance_policy",
        ):
            _required_string(
                source[field],
                label=f"source {source_id} {field}",
            )
        raw_reactions = source["reported_reactions"]
        if not isinstance(raw_reactions, dict) or not raw_reactions:
            raise HydrophobicMotifTransferError(
                f"source {source_id} reported_reactions must be nonempty"
            )
        for reaction_id, raw_reaction in sorted(raw_reactions.items()):
            reaction_id = _required_string(
                reaction_id,
                label=f"source {source_id} reaction id",
            )
            reaction = _required_mapping(
                raw_reaction,
                label=f"source {source_id} reaction {reaction_id}",
                fields=(
                    "level",
                    "transformation",
                    "source_locator",
                    "evidence_status",
                    "attachment_event",
                    "reactants",
                    "conditions",
                    "outcome_evidence",
                ),
            )
            for field in (
                "level",
                "transformation",
                "source_locator",
                "evidence_status",
                "attachment_event",
                "outcome_evidence",
            ):
                _required_string(
                    reaction[field],
                    label=(f"source {source_id} reaction {reaction_id} {field}"),
                )
            reactants = reaction["reactants"]
            if (
                not isinstance(reactants, list)
                or not reactants
                or any(not isinstance(value, str) or not value for value in reactants)
            ):
                raise HydrophobicMotifTransferError(
                    f"source {source_id} reaction {reaction_id} reactants must be nonempty strings"
                )
            if not isinstance(reaction["conditions"], dict):
                raise HydrophobicMotifTransferError(
                    f"source {source_id} reaction {reaction_id} conditions must be an object"
                )
        sources[source_id] = source
    return sources


def _unique_products(
    reaction: rdChemReactions.ChemicalReaction,
    reactants: Sequence[Chem.Mol],
    *,
    label: str,
) -> list[str]:
    products: set[str] = set()
    with rdBase.BlockLogs():
        outcomes = reaction.RunReactants(tuple(reactants))
    for outcome in outcomes:
        if len(outcome) != 1:
            raise HydrophobicMotifTransferError(f"{label} produced a multi-product outcome")
        product = Chem.Mol(outcome[0])
        try:
            Chem.SanitizeMol(product)
        except Exception as exc:
            raise HydrophobicMotifTransferError(
                f"{label} produced an unsanitizable product: {exc}"
            ) from exc
        for atom in product.GetAtoms():
            atom.SetAtomMapNum(0)
        products.add(
            Chem.MolToSmiles(
                product,
                canonical=True,
                isomericSmiles=True,
            )
        )
    return sorted(products)


def _nearest_component(
    candidate: Chem.Mol,
    reference_rows: Sequence[Mapping[str, str]],
) -> tuple[str, str, float, bool]:
    fingerprint = MORGAN_GENERATOR.GetFingerprint(candidate)
    best: tuple[float, str, str] | None = None
    candidate_smiles = Chem.MolToSmiles(
        candidate,
        canonical=True,
        isomericSmiles=True,
    )
    exact = False
    for row in reference_rows:
        molecule = Chem.MolFromSmiles(row["canonical_smiles"])
        if molecule is None:
            raise HydrophobicMotifTransferError("reference component became unparsable")
        similarity = float(
            DataStructs.TanimotoSimilarity(
                fingerprint,
                MORGAN_GENERATOR.GetFingerprint(molecule),
            )
        )
        key = (
            similarity,
            row["component_id"],
            row["canonical_smiles"],
        )
        if best is None or key > best:
            best = key
        exact = exact or row["canonical_smiles"] == candidate_smiles
    if best is None:
        raise HydrophobicMotifTransferError("reference component pool is empty")
    return best[1], best[2], best[0], exact


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(
        text,
        fieldnames=FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    raw = text.getvalue().encode()
    compressed = io.BytesIO()
    with gzip.GzipFile(
        fileobj=compressed,
        mode="wb",
        filename="",
        mtime=0,
    ) as handle:
        handle.write(raw)
    return compressed.getvalue()


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise HydrophobicMotifTransferError(f"config schema must be {CONFIG_SCHEMA_VERSION!r}")
    for field in ("task", "pilot_id", "generated_utc"):
        _required_string(config.get(field), label=field)
    policy = _required_mapping(
        config.get("scope_policy"),
        label="scope_policy",
        fields=(
            "transfer_target",
            "head_policy",
            "no_head_motif_conversion",
            "no_source_activity_label_inheritance",
            "no_source_route_inheritance",
            "no_procurement_inference_from_identity_reference",
            "forward_compatibility_is_not_experimental_success",
            "incomplete_routes_are_not_route_complete",
            "generative_oracle_and_route_support_are_independent",
            "missing_route_knowledge_is_not_chemical_failure",
            "operational_closure_not_exhaustive_reaction_mining",
            "reusable_synthesis_program_architecture",
            "chemistry_specific_evidence_required",
        ),
    )
    safeguards = [value for key, value in policy.items() if key.startswith("no_")] + [
        policy["forward_compatibility_is_not_experimental_success"],
        policy["incomplete_routes_are_not_route_complete"],
        policy["generative_oracle_and_route_support_are_independent"],
        policy["missing_route_knowledge_is_not_chemical_failure"],
        policy["operational_closure_not_exhaustive_reaction_mining"],
        policy["reusable_synthesis_program_architecture"],
        policy["chemistry_specific_evidence_required"],
    ]
    if any(value is not True for value in safeguards):
        raise HydrophobicMotifTransferError("all scope-policy safeguards must be true")
    if policy["transfer_target"] != "hydrophobic_tail_motifs":
        raise HydrophobicMotifTransferError("the pilot may transfer hydrophobic tail motifs only")
    if policy["head_policy"] != ("exact_site_defined_ugi_compatible_amine_only"):
        raise HydrophobicMotifTransferError("head policy must forbid motif conversion")
    taxonomy = config.get("route_outcome_taxonomy")
    expected_taxonomy = [
        "complete_route_found",
        "chemically_implausible_or_incompatible",
        "outside_declared_support",
        "missing_route_knowledge",
    ]
    if taxonomy != expected_taxonomy:
        raise HydrophobicMotifTransferError(
            "route_outcome_taxonomy must preserve the frozen four-way order"
        )


def _validate_expected_counts(
    summary: Mapping[str, int],
    expected: Any,
) -> None:
    expected_counts = _required_mapping(
        expected,
        label="expected_counts",
        fields=tuple(summary),
    )
    if set(expected_counts) != set(summary):
        raise HydrophobicMotifTransferError("expected_counts fields disagree with computed summary")
    for field, observed in summary.items():
        wanted = expected_counts[field]
        if not isinstance(wanted, int) or wanted != observed:
            raise HydrophobicMotifTransferError(
                f"{field}: expected {wanted!r}, observed {observed}"
            )


def build_hydrophobic_motif_transfer(
    config_path: Path,
    source_root: Path,
    component_ledger_path: Path,
    reaction_registry_path: Path,
) -> tuple[dict[str, Any], bytes]:
    """Build the deterministic transfer pilot and its row-level ledger."""

    config = _load_json(config_path, label="motif-transfer config")
    _validate_config(config)
    source_inputs, verified_asset_ids = _verify_source_assets(
        config.get("source_assets"),
        source_root=source_root,
    )
    sources = _validate_sources(
        config.get("sources"),
        verified_asset_ids=verified_asset_ids,
    )
    reference = _required_mapping(
        config.get("reference_component_ledger"),
        label="reference_component_ledger",
        fields=(
            "asset",
            "expected_sha256",
            "aldehyde_role",
            "expected_aldehyde_components",
        ),
    )
    registry_spec = _required_mapping(
        config.get("reaction_registry"),
        label="reaction_registry",
        fields=(
            "asset",
            "expected_sha256",
            "reaction_id",
            "anchor_reactants",
        ),
    )
    inputs = [
        {
            "asset": config_path.name,
            "role": "pilot_config",
            "bytes": config_path.stat().st_size,
            "sha256": sha256_file(config_path),
        },
        *source_inputs,
        {
            **_verify_hash(
                component_ledger_path,
                reference["expected_sha256"],
                label="reference component ledger",
            ),
            "role": "current_ugi_component_universe",
        },
        {
            **_verify_hash(
                reaction_registry_path,
                registry_spec["expected_sha256"],
                label="qualified reaction registry",
            ),
            "role": "frozen_l1_transform",
        },
    ]
    expected_pool_count = reference["expected_aldehyde_components"]
    if not isinstance(expected_pool_count, int) or expected_pool_count <= 0:
        raise HydrophobicMotifTransferError("expected_aldehyde_components must be positive")
    reference_rows = _read_component_pool(
        component_ledger_path,
        role=_required_string(
            reference["aldehyde_role"],
            label="reference aldehyde_role",
        ),
        expected_count=expected_pool_count,
    )
    _, ugi_reaction = _load_reaction(
        reaction_registry_path,
        reaction_id=_required_string(
            registry_spec["reaction_id"],
            label="reaction_registry reaction_id",
        ),
    )
    anchors = _required_mapping(
        registry_spec["anchor_reactants"],
        label="reaction_registry anchor_reactants",
        fields=("amine_head", "isocyanide_tail"),
    )
    anchor_amine, _ = _canonical_molecule(
        anchors["amine_head"],
        label="anchor amine",
    )
    anchor_isocyanide, _ = _canonical_molecule(
        anchors["isocyanide_tail"],
        label="anchor isocyanide",
    )
    programs = _compile_programs(
        config.get("programs"),
        verified_asset_ids=verified_asset_ids,
    )
    motifs = config.get("motifs")
    if not isinstance(motifs, list) or not motifs:
        raise HydrophobicMotifTransferError("motifs must be a nonempty list")

    rows: list[dict[str, Any]] = []
    seen_record_ids: set[str] = set()
    seen_source_components: set[tuple[str, str]] = set()
    class_counts: Counter[str] = Counter()
    for index, raw_record in enumerate(motifs):
        record = _required_mapping(
            raw_record,
            label=f"motifs[{index}]",
            fields=(
                "record_id",
                "source_id",
                "source_component_id",
                "source_reaction_id",
                "source_reported_yield_percent",
                "common_precursor_name",
                "common_precursor_smiles",
                "mapped_common_precursor_smiles",
                "identity_reference",
                "source_attachment",
                "expected_alcohol_class",
                "motif_classes",
                "risk_flags",
                "disposition",
                "program_id",
                "route_gap_class",
                "defer_reason",
            ),
        )
        record_id = _required_string(
            record["record_id"],
            label=f"motifs[{index}] record_id",
        )
        source_id = _required_string(
            record["source_id"],
            label=f"{record_id} source_id",
        )
        if source_id not in sources:
            raise HydrophobicMotifTransferError(f"{record_id}: unknown source {source_id!r}")
        source = sources[source_id]
        component_id = _required_string(
            record["source_component_id"],
            label=f"{record_id} source_component_id",
        )
        source_component_key = (source_id, component_id)
        if record_id in seen_record_ids or source_component_key in seen_source_components:
            raise HydrophobicMotifTransferError(
                f"{record_id}: duplicate record or source component"
            )
        seen_record_ids.add(record_id)
        seen_source_components.add(source_component_key)

        source_reaction_id = _required_string(
            record["source_reaction_id"],
            label=f"{record_id} source_reaction_id",
        )
        source_reactions = source["reported_reactions"]
        if source_reaction_id not in source_reactions:
            raise HydrophobicMotifTransferError(
                f"{record_id}: source reaction {source_reaction_id!r} "
                f"is not declared by {source_id}"
            )
        source_reaction = source_reactions[source_reaction_id]
        source_yield = record["source_reported_yield_percent"]
        if (
            isinstance(source_yield, bool)
            or not isinstance(source_yield, (int, float))
            or not 0 < float(source_yield) <= 100
        ):
            raise HydrophobicMotifTransferError(
                f"{record_id}: source reported yield must be in (0, 100]"
            )

        molecule, canonical = _canonical_molecule(
            record["common_precursor_smiles"],
            label=f"{record_id} common precursor",
        )
        identity = _required_mapping(
            record["identity_reference"],
            label=f"{record_id} identity_reference",
            fields=("kind", "identifier", "inchikey", "accessed_utc"),
        )
        if identity["kind"] not in {
            "pubchem",
            "source_reported_structure",
        }:
            raise HydrophobicMotifTransferError(f"{record_id}: unsupported identity reference")
        observed_inchikey = Chem.MolToInchiKey(molecule)
        if observed_inchikey != identity["inchikey"]:
            raise HydrophobicMotifTransferError(
                f"{record_id}: InChIKey mismatch; expected "
                f"{identity['inchikey']}, observed {observed_inchikey}"
            )
        attachment = _required_mapping(
            record["source_attachment"],
            label=f"{record_id} source_attachment",
            fields=(
                "motif_anchor_atom_map",
                "source_handle_atom_map",
                "event",
            ),
        )
        if attachment["event"] != source_reaction["attachment_event"]:
            raise HydrophobicMotifTransferError(
                f"{record_id}: source attachment event disagrees with {source_reaction_id}"
            )
        _, anchor, handle = _mapped_attachment(
            record["mapped_common_precursor_smiles"],
            canonical,
            attachment,
            label=record_id,
        )
        alcohol_class = _alcohol_class(anchor, handle)
        if alcohol_class != record["expected_alcohol_class"]:
            raise HydrophobicMotifTransferError(
                f"{record_id}: expected {record['expected_alcohol_class']} "
                f"alcohol, observed {alcohol_class}"
            )
        motif_classes = record["motif_classes"]
        if (
            not isinstance(motif_classes, list)
            or not motif_classes
            or any(not isinstance(item, str) or not item for item in motif_classes)
        ):
            raise HydrophobicMotifTransferError(
                f"{record_id}: motif_classes must be nonempty strings"
            )
        class_counts.update(motif_classes)
        risk_flags = record["risk_flags"]
        if not isinstance(risk_flags, list) or any(
            not isinstance(item, str) or not item for item in risk_flags
        ):
            raise HydrophobicMotifTransferError(f"{record_id}: risk_flags must contain strings")
        descriptors = _motif_descriptors(molecule)
        disposition = record["disposition"]

        candidate_smiles = ""
        program_id = ""
        evidence_grade = ""
        route_closure = "unresolved"
        terminal_status = "not_assessed"
        route_gap_class = ""
        route_support_status = "structure_only_unresolved"
        route_outcome_category = "missing_route_knowledge"
        missing_route_knowledge = True
        computationally_route_complete = False
        handle_matches = 0
        forward_products = 0
        exact_pool_match = False
        nearest_id = ""
        nearest_smiles = ""
        nearest_similarity: float | str = ""
        defer_reason = str(record["defer_reason"])
        proposed_role = ""
        if disposition == "propose_ugi_aldehyde":
            if alcohol_class != "primary" or defer_reason:
                raise HydrophobicMotifTransferError(
                    f"{record_id}: proposed aldehyde must be an undeferred primary alcohol"
                )
            program_id = _required_string(
                record["program_id"],
                label=f"{record_id} program_id",
            )
            if program_id not in programs:
                raise HydrophobicMotifTransferError(f"{record_id}: unknown program {program_id!r}")
            program, reaction = programs[program_id]
            realized = _unique_products(
                reaction,
                [molecule],
                label=f"{record_id} realization",
            )
            if len(realized) != 1:
                raise HydrophobicMotifTransferError(
                    f"{record_id}: expected one realization, found {len(realized)}"
                )
            candidate_smiles = realized[0]
            candidate = Chem.MolFromSmiles(candidate_smiles)
            if candidate is None or ALDEHYDE_QUERY is None:
                raise HydrophobicMotifTransferError(f"{record_id}: aldehyde realization is invalid")
            handle_matches = len(
                candidate.GetSubstructMatches(
                    ALDEHYDE_QUERY,
                    uniquify=True,
                )
            )
            if handle_matches != 1:
                raise HydrophobicMotifTransferError(
                    f"{record_id}: expected one aldehyde handle, found {handle_matches}"
                )
            forward = _unique_products(
                ugi_reaction,
                [anchor_amine, candidate, anchor_isocyanide],
                label=f"{record_id} frozen Ugi verification",
            )
            forward_products = len(forward)
            if forward_products != 1:
                raise HydrophobicMotifTransferError(
                    f"{record_id}: frozen Ugi transform produced {forward_products} unique products"
                )
            (
                nearest_id,
                nearest_smiles,
                nearest_similarity,
                exact_pool_match,
            ) = _nearest_component(candidate, reference_rows)
            proposed_role = _required_string(
                program["target_role"],
                label=f"{program_id} target_role",
            )
            evidence_grade = _required_string(
                program["evidence_grade"],
                label=f"{program_id} evidence_grade",
            )
            route_closure = _required_string(
                program["route_closure"],
                label=f"{program_id} route_closure",
            )
            terminal_status = _required_string(
                program["terminal_status"],
                label=f"{program_id} terminal_status",
            )
            raw_route_gap_class = program["route_gap_class"]
            if not isinstance(raw_route_gap_class, str):
                raise HydrophobicMotifTransferError(
                    f"{program_id} route_gap_class must be a string"
                )
            route_gap_class = raw_route_gap_class
            if record["route_gap_class"]:
                raise HydrophobicMotifTransferError(
                    f"{record_id}: proposed record must use the program's route-gap class"
                )
            computationally_route_complete = (
                route_closure == COMPLETE_ROUTE_CLOSURE
                and terminal_status in ACCEPTED_TERMINAL_STATES
            )
            if computationally_route_complete:
                route_support_status = "complete_route_to_accepted_terminal"
                route_outcome_category = "complete_route_found"
                missing_route_knowledge = False
            else:
                route_support_status = "forward_compatible_incomplete_route"
        elif disposition == "defer_structure_only":
            if record["program_id"] or not defer_reason:
                raise HydrophobicMotifTransferError(
                    f"{record_id}: deferred record needs an empty program and a reason"
                )
            route_gap_class = _required_string(
                record["route_gap_class"],
                label=f"{record_id} route_gap_class",
            )
        else:
            raise HydrophobicMotifTransferError(
                f"{record_id}: unsupported disposition {disposition!r}"
            )

        rows.append(
            {
                "record_id": record_id,
                "source_id": source_id,
                "source_platform": source["platform_id"],
                "source_component_id": component_id,
                "source_locator": source["source_locator"],
                "source_reaction_id": source_reaction_id,
                "source_reaction_transformation": source_reaction["transformation"],
                "source_reaction_conditions": _stable_json(source_reaction["conditions"]),
                "source_reaction_evidence_status": source_reaction["evidence_status"],
                "source_reported_yield_percent": f"{float(source_yield):g}",
                "common_precursor_name": record["common_precursor_name"],
                "common_precursor_canonical_smiles": canonical,
                "common_precursor_inchikey": observed_inchikey,
                "motif_anchor_atom_map": attachment["motif_anchor_atom_map"],
                "source_handle_atom_map": attachment["source_handle_atom_map"],
                "source_handle_event": attachment["event"],
                "source_attachment_mapping_status": "exact_mapped",
                "alcohol_class": alcohol_class,
                **descriptors,
                "motif_classes_json": _stable_json(sorted(motif_classes)),
                "risk_flags_json": _stable_json(sorted(risk_flags)),
                "disposition": disposition,
                "proposed_ugi_role": proposed_role,
                "proposed_ugi_component_smiles": candidate_smiles,
                "proposed_program_id": program_id,
                "proposed_route_evidence_grade": evidence_grade,
                "route_closure": route_closure,
                "terminal_status": terminal_status,
                "route_outcome_category": route_outcome_category,
                "route_gap_class": route_gap_class,
                "missing_encoded_route_knowledge": str(missing_route_knowledge).lower(),
                "observed_chemical_failure": "false",
                "generative_support_status": "not_evaluated_in_m0",
                "oracle_applicability_status": ("not_evaluated_no_label_transfer"),
                "route_support_status": route_support_status,
                "frozen_ugi_handle_matches": handle_matches,
                "frozen_ugi_forward_products": forward_products,
                "exact_current_aldehyde_pool_match": str(exact_pool_match).lower(),
                "nearest_current_aldehyde_component_id": nearest_id,
                "nearest_current_aldehyde_smiles": nearest_smiles,
                "nearest_current_aldehyde_tanimoto": (
                    f"{nearest_similarity:.6f}" if isinstance(nearest_similarity, float) else ""
                ),
                "computationally_route_complete": str(computationally_route_complete).lower(),
                "biological_label_inherited": "false",
                "defer_reason": defer_reason,
            }
        )

    rows.sort(key=lambda row: row["record_id"])
    source_platforms = sorted({str(row["source_platform"]) for row in rows})
    source_reaction_keys = {(str(row["source_id"]), str(row["source_reaction_id"])) for row in rows}
    summary = {
        "source_motifs": len(rows),
        "independent_source_platforms": len(source_platforms),
        "reported_source_reactions": len(source_reaction_keys),
        "motifs_linked_to_reported_source_reactions": sum(
            bool(row["source_reaction_id"]) for row in rows
        ),
        "unambiguous_source_attachment_mappings": sum(
            row["source_attachment_mapping_status"] == "exact_mapped" for row in rows
        ),
        "primary_alcohols": sum(row["alcohol_class"] == "primary" for row in rows),
        "secondary_alcohols": sum(row["alcohol_class"] == "secondary" for row in rows),
        "proposed_ugi_aldehydes": sum(row["disposition"] == "propose_ugi_aldehyde" for row in rows),
        "deferred_structure_only": sum(
            row["disposition"] == "defer_structure_only" for row in rows
        ),
        "exact_current_aldehyde_pool_matches": sum(
            row["exact_current_aldehyde_pool_match"] == "true" for row in rows
        ),
        "frozen_ugi_forward_verified": sum(row["frozen_ugi_forward_products"] == 1 for row in rows),
        "computationally_route_complete": sum(
            row["computationally_route_complete"] == "true" for row in rows
        ),
    }
    _validate_expected_counts(summary, config.get("expected_counts"))
    positive_signal = (
        summary["unambiguous_source_attachment_mappings"] == summary["source_motifs"]
        and summary["proposed_ugi_aldehydes"] > 0
        and summary["frozen_ugi_forward_verified"] == summary["proposed_ugi_aldehydes"]
        and summary["exact_current_aldehyde_pool_matches"] == 0
    )
    positive_platforms = sorted(
        platform
        for platform in source_platforms
        if any(
            row["source_platform"] == platform
            and row["disposition"] == "propose_ugi_aldehyde"
            and row["frozen_ugi_forward_products"] == 1
            for row in rows
        )
    )
    two_platform_signal = len(positive_platforms) >= 2
    expansion = _required_mapping(
        config.get("expansion_policy"),
        label="expansion_policy",
        fields=(
            "current_stage",
            "positive_signal",
            "next_stage",
            "broad_expansion_gate",
        ),
    )
    stopping_policy = _required_mapping(
        config.get("stopping_policy"),
        label="stopping_policy",
        fields=(
            "objective",
            "lnpdb_role",
            "tiers",
            "metrics",
            "numeric_threshold_status",
        ),
    )
    for field in ("tiers", "metrics"):
        values = stopping_policy[field]
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise HydrophobicMotifTransferError(
                f"stopping_policy {field} must contain nonempty strings"
            )
    ledger = _csv_bytes(rows)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": config["task"],
        "pilot_id": config["pilot_id"],
        "generated_utc": config["generated_utc"],
        "randomness": {"seed": 0, "used": False},
        "inputs": inputs,
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
        },
        "scope_policy": config["scope_policy"],
        "sources": config["sources"],
        "programs": config["programs"],
        "summary": summary,
        "route_outcome_counts": {
            category: sum(row["route_outcome_category"] == category for row in rows)
            for category in config["route_outcome_taxonomy"]
        },
        "motif_class_counts": dict(sorted(class_counts.items())),
        "decision": {
            "positive_transfer_signal": positive_signal,
            "positive_platforms": positive_platforms,
            "positive_two_platform_signal": two_platform_signal,
            "priority_queue_expansion_authorized": (
                two_platform_signal and summary["computationally_route_complete"] > 0
            ),
            "broad_lnpdb_expansion_authorized": False,
            "reason": expansion["positive_signal"],
            "next_stage": expansion["next_stage"],
            "broad_expansion_gate": expansion["broad_expansion_gate"],
            "transferred_components_are_observed_l2_supervision": False,
            "stopping_policy": dict(stopping_policy),
            "model_built": False,
        },
        "artifacts": {
            LEDGER_NAME: {
                "path": f"results/m0_09/{LEDGER_NAME}",
                "bytes": len(ledger),
                "sha256": sha256_bytes(ledger),
            }
        },
    }
    return result, ledger


def write_hydrophobic_motif_transfer(
    result: Mapping[str, Any],
    ledger: bytes,
    output_dir: Path,
) -> None:
    """Atomically replace deterministic motif-transfer artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        LEDGER_NAME: ledger,
        RESULT_NAME: (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
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
