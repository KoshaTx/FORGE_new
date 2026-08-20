"""Build the blinded M0-05 decomposition-precision review packet.

The pipeline scores only mechanically checkable properties. Chemical
plausibility, precision, specificity, and family disposition remain pending
until independent human chemists return the blinded annotations.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import html
import io
import json
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D

from forge.core.io import csv_bytes as _csv_bytes
from forge.corpus.r1_prime_audit import (
    AuditError,
    CompiledReaction,
    ReactionDefinition,
    compile_reactions,
    load_reaction_definitions,
    sha256_file,
)
from forge.corpus.r1_prime_audit import load_config as load_m0_04_config

CONFIG_SCHEMA_VERSION = "m0_05_decomposition_precision_audit_config.v1"
RESULT_SCHEMA_VERSION = "m0_05_decomposition_precision_audit.v1"
ALGORITHM_VERSION = "blinded_stratified_specificity_packet_v1"
UGI_REACTION_ID = "ugi_3cr_agile"
LX_REVIEW_ID = "lx_2024_aldehyde_tail_transfer_review"

BLIND_FIELDS = (
    "review_id",
    "proposed_reaction_family",
    "product_smiles",
    "proposed_roles_json",
    "proposed_components_json",
    "proposed_reactive_atoms_json",
    "reviewer_id",
    "plausible_final_assembly",
    "component_identities_and_roles_correct",
    "reactive_atoms_correct",
    "exact_forward_reconstruction",
    "plausible_alternative_decomposition",
    "confidence",
    "rejection_reason",
    "notes",
)

ANSWER_FIELDS = (
    "review_id",
    "sampling_frame",
    "case_origin",
    "control_type",
    "source_structure_id",
    "source_schemes_json",
    "source_decomposition_ids_json",
    "source_studies_json",
    "expected_reaction_family",
    "expected_roles_json",
    "expected_components_json",
    "expected_reactive_atoms_json",
    "mechanical_forward_reconstruction",
    "proposed_role_assignment_mechanically_correct",
    "proposed_reactive_atoms_mechanically_correct",
    "target_matching_forward_outcomes",
    "unique_target_site_signatures",
    "known_synthesis_alignment",
    "chemical_expectation",
    "sampling_strata_json",
    "anchor_distance",
    "component_frequencies_json",
    "handle_match_counts_json",
    "structural_flags_json",
)

STRATA_ORDER = (
    "rare_component",
    "common_component",
    "far_from_registry_anchor",
    "near_registry_anchor",
    "branching",
    "unsaturation",
    "embedded_linker",
    "multiple_reactive_sites",
    "degenerate_risk",
)

_REACTION_MAP_CACHE: dict[tuple[str, str], tuple[frozenset[int], ...]] = {}


@dataclass(frozen=True)
class SourceCandidate:
    """One deduplicated exact decomposition from the M0-04 ledger."""

    source_structure_id: str
    reaction_id: str
    product_smiles: str
    component_smiles: tuple[str, ...]
    schemes: tuple[str, ...]
    decomposition_ids: tuple[str, ...]
    source_studies: tuple[str, ...]


@dataclass(frozen=True)
class ReviewCase:
    """Internal unblinded representation of one review item."""

    case_key: str
    sampling_frame: str
    case_origin: str
    control_type: str
    proposed_reaction_id: str
    product_smiles: str
    components: tuple[str, ...]
    proposed_role_to_component: tuple[tuple[str, int], ...]
    proposed_reactive_atoms: tuple[tuple[str, tuple[int, ...]], ...]
    expected_reaction_id: str
    expected_role_to_component: tuple[tuple[str, int], ...]
    expected_components: tuple[str, ...]
    expected_reactive_atoms: tuple[tuple[str, tuple[int, ...]], ...]
    source_structure_id: str
    source_schemes: tuple[str, ...]
    source_decomposition_ids: tuple[str, ...]
    source_studies: tuple[str, ...]
    known_synthesis_alignment: str
    chemical_expectation: str
    sampling_strata: tuple[str, ...]
    anchor_distance: float | None
    component_frequencies: tuple[tuple[str, int], ...]
    handle_match_counts: tuple[tuple[str, int], ...]
    structural_flags: tuple[str, ...]
    mechanical_forward_reconstruction: bool
    proposed_role_assignment_mechanically_correct: bool
    proposed_reactive_atoms_mechanically_correct: bool
    target_matching_forward_outcomes: int
    target_site_signatures: tuple[tuple[tuple[str, tuple[int, ...]], ...], ...]
    review_id: str = ""


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise AuditError(f"{description} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AuditError(f"{description} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"{description} must contain a JSON object: {path}")
    return value


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the M0-05 packet configuration."""

    config = _load_json(path, "M0-05 config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise AuditError(
            f"unsupported M0-05 config schema {config.get('schema_version')!r}; "
            f"expected {CONFIG_SCHEMA_VERSION!r}"
        )
    if not isinstance(config.get("seed"), int):
        raise AuditError("M0-05 config seed must be an integer")
    if config.get("randomness_used") is not True:
        raise AuditError("M0-05 uses deterministic seeded sampling; randomness_used must be true")
    if config.get("sampling_method") != "seeded_sha256_stratified_round_robin_v1":
        raise AuditError("M0-05 sampling_method is unsupported")
    expected_inputs = config.get("expected_inputs")
    if not isinstance(expected_inputs, dict) or not expected_inputs:
        raise AuditError("M0-05 expected_inputs must be a nonempty object")
    for name, specification in expected_inputs.items():
        if (
            not isinstance(name, str)
            or not isinstance(specification, dict)
            or set(specification) != {"path", "sha256"}
            or not isinstance(specification["path"], str)
            or Path(specification["path"]).is_absolute()
            or ".." in Path(specification["path"]).parts
            or not isinstance(specification["sha256"], str)
            or len(specification["sha256"]) != 64
        ):
            raise AuditError(f"M0-05 expected input {name!r} is malformed")
    sampling = config.get("sampling")
    if not isinstance(sampling, dict):
        raise AuditError("M0-05 sampling must be an object")
    positive_integer_fields = (
        "observed_target_per_family",
        "virtual_ugi_stress_count",
        "lx_2024_transfer_count",
        "adversarial_count_per_type",
    )
    if any(
        not isinstance(sampling.get(field), int) or sampling[field] <= 0
        for field in positive_integer_fields
    ):
        raise AuditError("M0-05 sampling counts must be positive integers")
    adversarial_types = sampling.get("adversarial_types")
    required_types = {
        "role_swap",
        "cross_record_component_swap",
        "wrong_reaction_family",
        "non_ugi_as_ugi",
        "wrong_attachment_site",
        "degenerate_reconstructing",
    }
    if not isinstance(adversarial_types, list) or set(adversarial_types) != required_types:
        raise AuditError(
            "M0-05 adversarial_types must contain exactly the six declared control types"
        )
    fingerprint = config.get("fingerprint")
    if (
        not isinstance(fingerprint, dict)
        or fingerprint.get("kind") != "ECFP4"
        or fingerprint.get("radius") != 2
        or not isinstance(fingerprint.get("bits"), int)
        or fingerprint["bits"] <= 0
    ):
        raise AuditError("M0-05 fingerprint must define ECFP4 with radius 2 and positive bits")
    forward = config.get("forward_verification")
    if (
        not isinstance(forward, dict)
        or not isinstance(forward.get("max_products_per_case"), int)
        or forward["max_products_per_case"] <= 0
    ):
        raise AuditError("M0-05 max_products_per_case must be a positive integer")
    return config


def _resolve_and_verify_inputs(
    config: Mapping[str, Any], repo_root: Path
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    paths: dict[str, Path] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for name, specification in sorted(config["expected_inputs"].items()):
        path = repo_root / specification["path"]
        digest = sha256_file(path)
        if digest != specification["sha256"]:
            raise AuditError(
                f"M0-05 input hash mismatch for {name}: expected "
                f"{specification['sha256']}, found {digest}: {path}"
            )
        paths[name] = path
        provenance[name] = {
            "path": specification["path"],
            "sha256": digest,
            "bytes": path.stat().st_size,
        }
    return paths, provenance


def _canonical_smiles(smiles: str, description: str) -> str:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise AuditError(f"invalid SMILES for {description}: {smiles!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _molecule(smiles: str, description: str) -> Chem.Mol:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise AuditError(f"invalid SMILES for {description}: {smiles!r}")
    return molecule


def _stable_key(seed: int, *values: str) -> str:
    payload = "\x1f".join((str(seed), *values))
    return hashlib.sha256(payload.encode()).hexdigest()


def _json_cell(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _gzip_bytes(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed:
        compressed.write(payload)
    return buffer.getvalue()


def _read_r0(path: Path) -> dict[str, dict[str, str]]:
    required = {
        "r0_structure_id",
        "canonical_isomeric_smiles",
        "provenance_json",
    }
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise AuditError(
                f"R0 input is missing columns: {sorted(required.difference(reader.fieldnames or ()))}"
            )
        rows = list(reader)
    by_id = {row["r0_structure_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise AuditError("R0 input contains duplicate structure IDs")
    return by_id


def _read_source_candidates(
    path: Path, r0_rows: Mapping[str, Mapping[str, str]]
) -> tuple[SourceCandidate, ...]:
    required = {
        "scheme",
        "decomposition_id",
        "source_structure_id",
        "reaction_id",
        "reactant_smiles_json",
        "source_studies_json",
    }
    grouped: dict[tuple[str, str, tuple[str, ...]], dict[str, set[str]]] = {}
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise AuditError(
                "M0-04 decomposition ledger is missing columns: "
                f"{sorted(required.difference(reader.fieldnames or ()))}"
            )
        for row in reader:
            source_id = row["source_structure_id"]
            if source_id not in r0_rows:
                raise AuditError(f"decomposition source is missing from R0: {source_id}")
            try:
                components_raw = json.loads(row["reactant_smiles_json"])
                studies_raw = json.loads(row["source_studies_json"])
            except json.JSONDecodeError as exc:
                raise AuditError(
                    f"malformed JSON in decomposition {row['decomposition_id']}"
                ) from exc
            if not isinstance(components_raw, list) or not isinstance(studies_raw, list):
                raise AuditError(f"malformed decomposition arrays in {row['decomposition_id']}")
            components = tuple(
                _canonical_smiles(str(smiles), f"{row['decomposition_id']} component")
                for smiles in components_raw
            )
            key = (source_id, row["reaction_id"], components)
            entry = grouped.setdefault(
                key,
                {"schemes": set(), "decomposition_ids": set(), "source_studies": set()},
            )
            entry["schemes"].add(row["scheme"])
            entry["decomposition_ids"].add(row["decomposition_id"])
            entry["source_studies"].update(str(value) for value in studies_raw)
    candidates = []
    for (source_id, reaction_id, components), provenance in sorted(grouped.items()):
        candidates.append(
            SourceCandidate(
                source_structure_id=source_id,
                reaction_id=reaction_id,
                product_smiles=_canonical_smiles(
                    str(r0_rows[source_id]["canonical_isomeric_smiles"]),
                    source_id,
                ),
                component_smiles=components,
                schemes=tuple(sorted(provenance["schemes"])),
                decomposition_ids=tuple(sorted(provenance["decomposition_ids"])),
                source_studies=tuple(sorted(provenance["source_studies"])),
            )
        )
    return tuple(candidates)


def _reaction_map_numbers(
    reaction: CompiledReaction,
) -> tuple[frozenset[int], ...]:
    cache_key = (
        reaction.definition.reaction_id,
        reaction.definition.atom_mapped_reaction_smarts,
    )
    cached = _REACTION_MAP_CACHE.get(cache_key)
    if cached is not None:
        return cached
    with rdBase.BlockLogs():
        reaction.forward.Initialize()
        reacting_template_atoms = reaction.forward.GetReactingAtoms(mappedAtomsOnly=True)
    map_numbers = []
    for reactant_index, atom_indices in enumerate(reacting_template_atoms):
        template = reaction.forward.GetReactantTemplate(reactant_index)
        map_numbers.append(
            frozenset(
                template.GetAtomWithIdx(atom_index).GetAtomMapNum()
                for atom_index in atom_indices
                if template.GetAtomWithIdx(atom_index).GetAtomMapNum() > 0
            )
        )
    result = tuple(map_numbers)
    _REACTION_MAP_CACHE[cache_key] = result
    return result


def _forward_analysis(
    reaction: CompiledReaction,
    components: Sequence[str],
    role_to_component: Mapping[str, int],
    target_smiles: str,
    max_products: int,
) -> dict[str, Any]:
    roles = tuple(role.name for role in reaction.definition.reactant_roles)
    if set(role_to_component) != set(roles):
        return {
            "mechanical_forward_reconstruction": False,
            "raw_forward_outcomes": 0,
            "unique_forward_products": 0,
            "target_matching_forward_outcomes": 0,
            "target_site_signatures": (),
        }
    try:
        ordered = [
            _molecule(
                components[role_to_component[role]], f"{reaction.definition.reaction_id}/{role}"
            )
            for role in roles
        ]
    except (IndexError, TypeError):
        return {
            "mechanical_forward_reconstruction": False,
            "raw_forward_outcomes": 0,
            "unique_forward_products": 0,
            "target_matching_forward_outcomes": 0,
            "target_site_signatures": (),
        }
    with rdBase.BlockLogs():
        outcomes = reaction.forward.RunReactants(tuple(ordered), maxProducts=max_products)
    if len(outcomes) >= max_products:
        raise AuditError(
            f"{reaction.definition.reaction_id}: review-case forward run reached "
            f"max_products={max_products}"
        )
    target = _canonical_smiles(target_smiles, "review target")
    products: set[str] = set()
    signatures: set[tuple[tuple[str, tuple[int, ...]], ...]] = set()
    target_outcomes = 0
    reacting_maps = _reaction_map_numbers(reaction)
    for outcome in outcomes:
        if len(outcome) != 1:
            continue
        try:
            with rdBase.BlockLogs():
                Chem.SanitizeMol(outcome[0])
            product = Chem.MolToSmiles(outcome[0], canonical=True, isomericSmiles=True)
        except Exception:
            continue
        products.add(product)
        if product != target:
            continue
        target_outcomes += 1
        sites: dict[str, set[int]] = {role: set() for role in roles}
        for atom in outcome[0].GetAtoms():
            if not atom.HasProp("old_mapno"):
                continue
            properties = atom.GetPropsAsDict(includePrivate=True, includeComputed=False)
            reactant_index = int(properties.get("react_idx", -1))
            source_atom = int(properties.get("react_atom_idx", -1))
            map_number = int(properties["old_mapno"])
            if (
                0 <= reactant_index < len(roles)
                and source_atom >= 0
                and map_number in reacting_maps[reactant_index]
            ):
                sites[roles[reactant_index]].add(source_atom)
        signature = tuple((role, tuple(sorted(sites[role]))) for role in roles)
        signatures.add(signature)
    return {
        "mechanical_forward_reconstruction": target in products,
        "raw_forward_outcomes": len(outcomes),
        "unique_forward_products": len(products),
        "target_matching_forward_outcomes": target_outcomes,
        "target_site_signatures": tuple(sorted(signatures)),
    }


def _handle_match_counts(
    reaction: CompiledReaction, components: Sequence[str]
) -> tuple[tuple[str, int], ...]:
    counts = []
    for role, handle, smiles in zip(
        reaction.definition.reactant_roles,
        reaction.handles,
        components,
        strict=True,
    ):
        molecule = _molecule(smiles, f"{reaction.definition.reaction_id}/{role.name}")
        matches = molecule.GetSubstructMatches(handle, uniquify=True)
        counts.append((role.name, len(matches)))
    return tuple(counts)


def _structural_flags(smiles: str) -> tuple[str, ...]:
    molecule = _molecule(smiles, "structural feature")
    flags: set[str] = set()
    if any(
        atom.GetAtomicNum() == 6
        and sum(neighbor.GetAtomicNum() == 6 for neighbor in atom.GetNeighbors()) >= 3
        for atom in molecule.GetAtoms()
    ):
        flags.add("branching")
    if any(
        not bond.GetIsAromatic()
        and bond.GetBondType() in {Chem.BondType.DOUBLE, Chem.BondType.TRIPLE}
        and bond.GetBeginAtom().GetAtomicNum() == 6
        and bond.GetEndAtom().GetAtomicNum() == 6
        for bond in molecule.GetBonds()
    ):
        flags.add("unsaturation")
    linker_patterns = {
        "ester": "[CX3](=O)[OX2][#6]",
        "amide": "[CX3](=O)[NX3]",
        "carbonate": "[OX2][CX3](=O)[OX2]",
        "acetal": "[CX4]([OX2])([OX2])",
    }
    for name, pattern in linker_patterns.items():
        query = Chem.MolFromSmarts(pattern)
        if query is not None and molecule.HasSubstructMatch(query):
            flags.add(f"embedded_{name}")
    return tuple(sorted(flags))


def _degenerate_risk(components: Sequence[str]) -> bool:
    sizes = [
        sum(atom.GetAtomicNum() > 1 for atom in _molecule(smiles, "component size").GetAtoms())
        for smiles in components
    ]
    if not sizes:
        return True
    return min(sizes) <= 2 or max(sizes) >= 10 * max(1, min(sizes))


def _fingerprint(smiles: str, generator: Any) -> Any:
    return generator.GetFingerprint(_molecule(smiles, "fingerprint"))


def _known_positive_products(definition: ReactionDefinition) -> tuple[str, ...]:
    return tuple(
        _canonical_smiles(str(example["expected"]), f"{definition.reaction_id} positive example")
        for example in definition.known_positive_examples
    )


def _case_from_source(
    candidate: SourceCandidate,
    reaction: CompiledReaction,
    component_frequency: Mapping[tuple[str, str, str], int],
    anchor_distance: float,
    max_products: int,
) -> ReviewCase:
    roles = tuple(role.name for role in reaction.definition.reactant_roles)
    role_map = tuple((role, index) for index, role in enumerate(roles))
    analysis = _forward_analysis(
        reaction,
        candidate.component_smiles,
        dict(role_map),
        candidate.product_smiles,
        max_products,
    )
    if not analysis["mechanical_forward_reconstruction"]:
        raise AuditError(
            f"M0-04 candidate no longer reconstructs: {candidate.source_structure_id}/"
            f"{candidate.reaction_id}"
        )
    site_signatures = analysis["target_site_signatures"]
    if not site_signatures:
        raise AuditError(
            f"no reacting-site identity recovered for {candidate.source_structure_id}/"
            f"{candidate.reaction_id}"
        )
    reactive_atoms = site_signatures[0]
    handle_counts = _handle_match_counts(reaction, candidate.component_smiles)
    frequencies = tuple(
        (
            role,
            component_frequency[(candidate.reaction_id, role, smiles)],
        )
        for role, smiles in zip(roles, candidate.component_smiles, strict=True)
    )
    flags = set(_structural_flags(candidate.product_smiles))
    if any(count > 1 for _, count in handle_counts):
        flags.add("multiple_reactive_sites")
    if _degenerate_risk(candidate.component_smiles):
        flags.add("degenerate_risk")
    source_is_agile = any("AGILE" in source.upper() for source in candidate.source_studies)
    alignment = (
        "observed_lipid_with_agile_source_provenance"
        if source_is_agile
        else "observed_lipid_without_normalized_source_route"
    )
    return ReviewCase(
        case_key=(
            f"observed|{candidate.source_structure_id}|{candidate.reaction_id}|"
            f"{'|'.join(candidate.component_smiles)}"
        ),
        sampling_frame="observed_corpus_decomposition",
        case_origin="m0_04_exact_decomposition",
        control_type="",
        proposed_reaction_id=candidate.reaction_id,
        product_smiles=candidate.product_smiles,
        components=candidate.component_smiles,
        proposed_role_to_component=role_map,
        proposed_reactive_atoms=reactive_atoms,
        expected_reaction_id=candidate.reaction_id,
        expected_role_to_component=role_map,
        expected_components=candidate.component_smiles,
        expected_reactive_atoms=reactive_atoms,
        source_structure_id=candidate.source_structure_id,
        source_schemes=candidate.schemes,
        source_decomposition_ids=candidate.decomposition_ids,
        source_studies=candidate.source_studies,
        known_synthesis_alignment=alignment,
        chemical_expectation="human_review_required",
        sampling_strata=(),
        anchor_distance=anchor_distance,
        component_frequencies=frequencies,
        handle_match_counts=handle_counts,
        structural_flags=tuple(sorted(flags)),
        mechanical_forward_reconstruction=True,
        proposed_role_assignment_mechanically_correct=True,
        proposed_reactive_atoms_mechanically_correct=True,
        target_matching_forward_outcomes=analysis["target_matching_forward_outcomes"],
        target_site_signatures=site_signatures,
    )


def _sampling_tags(
    case: ReviewCase,
    family_median_distance: float,
) -> tuple[str, ...]:
    tags: set[str] = set()
    frequencies = [count for _, count in case.component_frequencies]
    if frequencies and min(frequencies) <= 1:
        tags.add("rare_component")
    if frequencies and min(frequencies) >= 5:
        tags.add("common_component")
    if case.anchor_distance is not None:
        if case.anchor_distance >= family_median_distance:
            tags.add("far_from_registry_anchor")
        if case.anchor_distance <= family_median_distance:
            tags.add("near_registry_anchor")
    if "branching" in case.structural_flags:
        tags.add("branching")
    if "unsaturation" in case.structural_flags:
        tags.add("unsaturation")
    if any(flag.startswith("embedded_") for flag in case.structural_flags):
        tags.add("embedded_linker")
    if "multiple_reactive_sites" in case.structural_flags:
        tags.add("multiple_reactive_sites")
    if "degenerate_risk" in case.structural_flags:
        tags.add("degenerate_risk")
    return tuple(sorted(tags))


def _stratified_round_robin(
    cases: Sequence[ReviewCase],
    target: int,
    seed: int,
) -> tuple[ReviewCase, ...]:
    if target >= len(cases):
        distances = [case.anchor_distance for case in cases if case.anchor_distance is not None]
        median = statistics.median(distances) if distances else 0.0
        return tuple(
            replace(case, sampling_strata=_sampling_tags(case, median))
            for case in sorted(cases, key=lambda item: _stable_key(seed, item.case_key))
        )
    distances = [case.anchor_distance for case in cases if case.anchor_distance is not None]
    median = statistics.median(distances) if distances else 0.0
    tagged = {
        case.case_key: replace(case, sampling_strata=_sampling_tags(case, median)) for case in cases
    }
    ordered = sorted(
        tagged.values(),
        key=lambda item: _stable_key(seed, item.case_key),
    )
    selected: list[ReviewCase] = []
    selected_keys: set[str] = set()
    while len(selected) < target:
        added = False
        for stratum in STRATA_ORDER:
            match = next(
                (
                    case
                    for case in ordered
                    if case.case_key not in selected_keys and stratum in case.sampling_strata
                ),
                None,
            )
            if match is not None:
                selected.append(match)
                selected_keys.add(match.case_key)
                added = True
                if len(selected) == target:
                    break
        if not added:
            break
    for case in ordered:
        if len(selected) == target:
            break
        if case.case_key not in selected_keys:
            selected.append(case)
            selected_keys.add(case.case_key)
    return tuple(selected)


def _prepare_observed_cases(
    candidates: Sequence[SourceCandidate],
    compiled: Mapping[str, CompiledReaction],
    target_per_family: int,
    seed: int,
    fingerprint_config: Mapping[str, Any],
    max_products: int,
    degenerate_challenge_count: int,
) -> tuple[tuple[ReviewCase, ...], tuple[ReviewCase, ...], dict[str, Any]]:
    component_frequency: Counter[tuple[str, str, str]] = Counter()
    for candidate in candidates:
        reaction = compiled[candidate.reaction_id]
        roles = tuple(role.name for role in reaction.definition.reactant_roles)
        if len(roles) != len(candidate.component_smiles):
            raise AuditError(f"{candidate.reaction_id}: component count does not match roles")
        component_frequency.update(
            (candidate.reaction_id, role, smiles)
            for role, smiles in zip(roles, candidate.component_smiles, strict=True)
        )

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(fingerprint_config["radius"]),
        fpSize=int(fingerprint_config["bits"]),
        includeChirality=bool(fingerprint_config.get("include_chirality", True)),
    )
    anchors = {
        reaction_id: tuple(
            _fingerprint(product, generator)
            for product in _known_positive_products(reaction.definition)
        )
        for reaction_id, reaction in compiled.items()
    }
    previews_by_family: dict[str, list[ReviewCase]] = defaultdict(list)
    candidate_by_key: dict[str, SourceCandidate] = {}
    for candidate in candidates:
        reaction = compiled.get(candidate.reaction_id)
        if reaction is None:
            raise AuditError(f"unknown reaction in M0-04 ledger: {candidate.reaction_id}")
        roles = tuple(role.name for role in reaction.definition.reactant_roles)
        candidate_fp = _fingerprint(candidate.product_smiles, generator)
        similarities = DataStructs.BulkTanimotoSimilarity(
            candidate_fp,
            list(anchors[candidate.reaction_id]),
        )
        distance = 1.0 - max(similarities)
        handle_counts = _handle_match_counts(reaction, candidate.component_smiles)
        flags = set(_structural_flags(candidate.product_smiles))
        if any(value > 1 for _, value in handle_counts):
            flags.add("multiple_reactive_sites")
        if _degenerate_risk(candidate.component_smiles):
            flags.add("degenerate_risk")
        frequencies = tuple(
            (
                role,
                component_frequency[(candidate.reaction_id, role, smiles)],
            )
            for role, smiles in zip(roles, candidate.component_smiles, strict=True)
        )
        case_key = (
            f"observed|{candidate.source_structure_id}|{candidate.reaction_id}|"
            f"{'|'.join(candidate.component_smiles)}"
        )
        preview = ReviewCase(
            case_key=case_key,
            sampling_frame="observed_corpus_decomposition",
            case_origin="m0_04_exact_decomposition",
            control_type="",
            proposed_reaction_id=candidate.reaction_id,
            product_smiles=candidate.product_smiles,
            components=candidate.component_smiles,
            proposed_role_to_component=tuple((role, index) for index, role in enumerate(roles)),
            proposed_reactive_atoms=(),
            expected_reaction_id=candidate.reaction_id,
            expected_role_to_component=(),
            expected_components=candidate.component_smiles,
            expected_reactive_atoms=(),
            source_structure_id=candidate.source_structure_id,
            source_schemes=candidate.schemes,
            source_decomposition_ids=candidate.decomposition_ids,
            source_studies=candidate.source_studies,
            known_synthesis_alignment="selection_preview_only",
            chemical_expectation="human_review_required",
            sampling_strata=(),
            anchor_distance=distance,
            component_frequencies=frequencies,
            handle_match_counts=handle_counts,
            structural_flags=tuple(sorted(flags)),
            mechanical_forward_reconstruction=False,
            proposed_role_assignment_mechanically_correct=True,
            proposed_reactive_atoms_mechanically_correct=False,
            target_matching_forward_outcomes=0,
            target_site_signatures=(),
        )
        previews_by_family[candidate.reaction_id].append(preview)
        candidate_by_key[case_key] = candidate

    selected: list[ReviewCase] = []
    selected_preview_keys: set[str] = set()
    family_summary: dict[str, Any] = {}
    for reaction_id in sorted(compiled):
        family_previews = previews_by_family.get(reaction_id, [])
        target = min(target_per_family, len(family_previews))
        selected_previews = _stratified_round_robin(
            family_previews,
            target,
            seed + int(hashlib.sha256(reaction_id.encode()).hexdigest()[:8], 16),
        )
        family_selected = tuple(
            replace(
                _case_from_source(
                    candidate_by_key[preview.case_key],
                    compiled[reaction_id],
                    component_frequency,
                    float(preview.anchor_distance),
                    max_products,
                ),
                sampling_strata=preview.sampling_strata,
            )
            for preview in selected_previews
        )
        selected.extend(family_selected)
        selected_preview_keys.update(preview.case_key for preview in selected_previews)
        family_summary[reaction_id] = {
            "deduplicated_exact_decompositions": len(family_previews),
            "target_rule": f"min({target_per_family}, available)",
            "sampled": len(family_selected),
            "status": "sampled" if family_selected else "zero_available_explicitly_reported",
            "sampled_strata": dict(
                sorted(
                    Counter(
                        stratum for case in family_selected for stratum in case.sampling_strata
                    ).items()
                )
            ),
        }
    challenge_previews = sorted(
        (
            preview
            for family_previews in previews_by_family.values()
            for preview in family_previews
            if preview.case_key not in selected_preview_keys
            and (
                "degenerate_risk" in preview.structural_flags
                or preview.proposed_reaction_id == "reductive_amination_amine_aldehyde"
            )
        ),
        key=lambda preview: (
            preview.proposed_reaction_id != "reductive_amination_amine_aldehyde",
            _stable_key(seed, "degenerate_challenge_pool", preview.case_key),
        ),
    )
    if len(challenge_previews) < degenerate_challenge_count:
        raise AuditError(
            "observed corpus cannot supply the requested unique degenerate challenge pool"
        )
    challenge_cases = tuple(
        replace(
            _case_from_source(
                candidate_by_key[preview.case_key],
                compiled[preview.proposed_reaction_id],
                component_frequency,
                float(preview.anchor_distance),
                max_products,
            ),
            sampling_strata=preview.sampling_strata,
        )
        for preview in challenge_previews[:degenerate_challenge_count]
    )
    return (
        tuple(selected),
        challenge_cases,
        {
            "deduplicated_exact_decompositions": len(candidates),
            "families": family_summary,
            "sampled": len(selected),
        },
    )


def _iter_virtual_ugi_records(path: Path) -> Iterable[dict[str, Any]]:
    required = {
        "source_row_index",
        "canonical_product_smiles",
        "decomposition_status",
        "candidate_count",
        "candidate_routes_json",
    }
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise AuditError(
                "AGILE virtual ledger is missing columns: "
                f"{sorted(required.difference(reader.fieldnames or ()))}"
            )
        for row in reader:
            if (
                row["decomposition_status"] != "one_exact_qualified_ugi_decomposition"
                or int(row["candidate_count"]) != 1
            ):
                raise AuditError(
                    "M0-05 requires every virtual stress candidate to have one exact "
                    f"decomposition; row {row['source_row_index']} does not"
                )
            try:
                routes = json.loads(row["candidate_routes_json"])
            except json.JSONDecodeError as exc:
                raise AuditError(
                    f"virtual row {row['source_row_index']} has malformed route JSON"
                ) from exc
            if not isinstance(routes, list) or len(routes) != 1:
                raise AuditError(
                    f"virtual row {row['source_row_index']} must contain exactly one route"
                )
            yield {
                "source_row_index": row["source_row_index"],
                "product_smiles": _canonical_smiles(
                    row["canonical_product_smiles"],
                    f"virtual row {row['source_row_index']}",
                ),
                "route": routes[0],
            }


def _case_from_virtual_record(
    record: Mapping[str, Any],
    reaction: CompiledReaction,
    max_products: int,
    generator: Any,
    anchor_fingerprints: Sequence[Any],
) -> ReviewCase:
    route = record["route"]
    components_raw = route.get("components")
    if not isinstance(components_raw, dict):
        raise AuditError(f"virtual row {record['source_row_index']} has no component mapping")
    roles = tuple(role.name for role in reaction.definition.reactant_roles)
    if set(components_raw) != set(roles):
        raise AuditError(
            f"virtual row {record['source_row_index']} component roles do not match registry"
        )
    components = tuple(
        _canonical_smiles(
            str(components_raw[role]),
            f"virtual row {record['source_row_index']}/{role}",
        )
        for role in roles
    )
    role_map = tuple((role, index) for index, role in enumerate(roles))
    analysis = _forward_analysis(
        reaction,
        components,
        dict(role_map),
        str(record["product_smiles"]),
        max_products,
    )
    if not analysis["mechanical_forward_reconstruction"] or not analysis["target_site_signatures"]:
        raise AuditError(
            f"virtual row {record['source_row_index']} failed frozen forward reconstruction"
        )
    handle_counts = _handle_match_counts(reaction, components)
    flags = set(_structural_flags(str(record["product_smiles"])))
    if any(count > 1 for _, count in handle_counts):
        flags.add("multiple_reactive_sites")
    if _degenerate_risk(components):
        flags.add("degenerate_risk")
    candidate_fp = _fingerprint(str(record["product_smiles"]), generator)
    distance = 1.0 - max(
        DataStructs.BulkTanimotoSimilarity(candidate_fp, list(anchor_fingerprints))
    )
    return ReviewCase(
        case_key=f"virtual|{record['source_row_index']}|{record['product_smiles']}",
        sampling_frame="ugi_applicability_stress",
        case_origin="agile_virtual_exact_decomposition",
        control_type="",
        proposed_reaction_id=UGI_REACTION_ID,
        product_smiles=str(record["product_smiles"]),
        components=components,
        proposed_role_to_component=role_map,
        proposed_reactive_atoms=analysis["target_site_signatures"][0],
        expected_reaction_id=UGI_REACTION_ID,
        expected_role_to_component=role_map,
        expected_components=components,
        expected_reactive_atoms=analysis["target_site_signatures"][0],
        source_structure_id=f"agile-virtual-{record['source_row_index']}",
        source_schemes=("agile_virtual_12k",),
        source_decomposition_ids=(),
        source_studies=("agile_virtual12k",),
        known_synthesis_alignment=(
            "virtual_product_component_pair_only_not_an_observed_route_or_outcome"
        ),
        chemical_expectation="human_review_required",
        sampling_strata=(),
        anchor_distance=distance,
        component_frequencies=tuple((role, 0) for role in roles),
        handle_match_counts=handle_counts,
        structural_flags=tuple(sorted(flags)),
        mechanical_forward_reconstruction=True,
        proposed_role_assignment_mechanically_correct=True,
        proposed_reactive_atoms_mechanically_correct=True,
        target_matching_forward_outcomes=analysis["target_matching_forward_outcomes"],
        target_site_signatures=analysis["target_site_signatures"],
    )


def _prepare_virtual_ugi_cases(
    path: Path,
    reaction: CompiledReaction,
    target: int,
    seed: int,
    fingerprint_config: Mapping[str, Any],
    max_products: int,
    excluded_product_components: set[tuple[str, tuple[str, ...]]],
) -> tuple[ReviewCase, ...]:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(fingerprint_config["radius"]),
        fpSize=int(fingerprint_config["bits"]),
        includeChirality=bool(fingerprint_config.get("include_chirality", True)),
    )
    anchors = tuple(
        _fingerprint(product, generator)
        for product in _known_positive_products(reaction.definition)
    )
    roles = tuple(role.name for role in reaction.definition.reactant_roles)
    records_by_key: dict[str, dict[str, Any]] = {}
    previews = []
    for record in _iter_virtual_ugi_records(path):
        components_raw = record["route"].get("components")
        if not isinstance(components_raw, dict) or set(components_raw) != set(roles):
            raise AuditError(
                f"virtual row {record['source_row_index']} component roles do not match registry"
            )
        components = tuple(
            _canonical_smiles(
                str(components_raw[role]),
                f"virtual row {record['source_row_index']}/{role}",
            )
            for role in roles
        )
        if (str(record["product_smiles"]), components) in excluded_product_components:
            continue
        handle_counts = _handle_match_counts(reaction, components)
        flags = set(_structural_flags(str(record["product_smiles"])))
        if any(value > 1 for _, value in handle_counts):
            flags.add("multiple_reactive_sites")
        if _degenerate_risk(components):
            flags.add("degenerate_risk")
        candidate_fp = _fingerprint(str(record["product_smiles"]), generator)
        distance = 1.0 - max(DataStructs.BulkTanimotoSimilarity(candidate_fp, list(anchors)))
        case_key = f"virtual|{record['source_row_index']}|{record['product_smiles']}"
        previews.append(
            ReviewCase(
                case_key=case_key,
                sampling_frame="ugi_applicability_stress",
                case_origin="agile_virtual_exact_decomposition",
                control_type="",
                proposed_reaction_id=UGI_REACTION_ID,
                product_smiles=str(record["product_smiles"]),
                components=components,
                proposed_role_to_component=tuple((role, index) for index, role in enumerate(roles)),
                proposed_reactive_atoms=(),
                expected_reaction_id=UGI_REACTION_ID,
                expected_role_to_component=(),
                expected_components=components,
                expected_reactive_atoms=(),
                source_structure_id=f"agile-virtual-{record['source_row_index']}",
                source_schemes=("agile_virtual_12k",),
                source_decomposition_ids=(),
                source_studies=("agile_virtual12k",),
                known_synthesis_alignment="selection_preview_only",
                chemical_expectation="human_review_required",
                sampling_strata=(),
                anchor_distance=distance,
                component_frequencies=tuple((role, 0) for role in roles),
                handle_match_counts=handle_counts,
                structural_flags=tuple(sorted(flags)),
                mechanical_forward_reconstruction=False,
                proposed_role_assignment_mechanically_correct=True,
                proposed_reactive_atoms_mechanically_correct=False,
                target_matching_forward_outcomes=0,
                target_site_signatures=(),
            )
        )
        records_by_key[case_key] = record
    selected_previews = _stratified_round_robin(previews, target, seed)
    return tuple(
        replace(
            _case_from_virtual_record(
                records_by_key[preview.case_key],
                reaction,
                max_products,
                generator,
                anchors,
            ),
            sampling_strata=preview.sampling_strata,
        )
        for preview in selected_previews
    )


def _find_review(value: Any, review_id: str) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if value.get("review_id") == review_id and isinstance(value.get("l2_route_families"), list):
            return value
        for child in value.values():
            match = _find_review(child, review_id)
            if match is not None:
                return match
    elif isinstance(value, list):
        for child in value:
            match = _find_review(child, review_id)
            if match is not None:
                return match
    return None


def _lx_members(review: Mapping[str, Any]) -> list[dict[str, str]]:
    members: list[dict[str, str]] = []
    families = review.get("l2_route_families")
    if not isinstance(families, list):
        raise AuditError(f"{LX_REVIEW_ID} has no L2 route families")
    for family in families:
        if not isinstance(family, dict):
            continue
        family_members = family.get("members")
        if not isinstance(family_members, list):
            continue
        for member in family_members:
            if (
                not isinstance(member, dict)
                or not isinstance(member.get("label"), str)
                or not isinstance(member.get("product_smiles"), str)
            ):
                raise AuditError(f"{LX_REVIEW_ID} has a malformed member")
            members.append(
                {
                    "label": member["label"],
                    "product_smiles": member["product_smiles"],
                }
            )
    return sorted(members, key=lambda member: member["label"])


def _prepare_lx_transfer_cases(
    reviews_path: Path,
    reaction: CompiledReaction,
    expected_count: int,
    max_products: int,
) -> tuple[ReviewCase, ...]:
    payload = _load_json(reviews_path, "M0-09 paper-route reviews")
    review = _find_review(payload, LX_REVIEW_ID)
    if review is None:
        raise AuditError(f"paper-route reviews do not contain {LX_REVIEW_ID}")
    members = _lx_members(review)
    if len(members) != expected_count:
        raise AuditError(
            f"{LX_REVIEW_ID} contains {len(members)} routed aldehydes; expected {expected_count}"
        )
    example = reaction.definition.known_positive_examples[0]
    roles = tuple(role.name for role in reaction.definition.reactant_roles)
    aldehyde_role = "oxoester_aldehyde_body_tail"
    reference_by_role = {
        role: _canonical_smiles(str(smiles), f"Ugi reference/{role}")
        for role, smiles in zip(roles, example["reactants"], strict=True)
    }
    cases = []
    for member in members:
        by_role = dict(reference_by_role)
        by_role[aldehyde_role] = _canonical_smiles(
            member["product_smiles"],
            f"LX transfer {member['label']}",
        )
        components = tuple(by_role[role] for role in roles)
        role_map = tuple((role, index) for index, role in enumerate(roles))
        with rdBase.BlockLogs():
            outcomes = reaction.forward.RunReactants(
                tuple(_molecule(smiles, f"LX {member['label']}") for smiles in components),
                maxProducts=max_products,
            )
        products = []
        for outcome in outcomes:
            if len(outcome) != 1:
                continue
            try:
                with rdBase.BlockLogs():
                    Chem.SanitizeMol(outcome[0])
                products.append(Chem.MolToSmiles(outcome[0], canonical=True, isomericSmiles=True))
            except Exception:
                continue
        unique_products = sorted(set(products))
        if len(unique_products) != 1:
            raise AuditError(
                f"LX transfer {member['label']} produced {len(unique_products)} "
                "unique sanitized Ugi products"
            )
        product = unique_products[0]
        analysis = _forward_analysis(
            reaction,
            components,
            dict(role_map),
            product,
            max_products,
        )
        if not analysis["target_site_signatures"]:
            raise AuditError(f"LX transfer {member['label']} has no reacting-site identity")
        handle_counts = _handle_match_counts(reaction, components)
        flags = set(_structural_flags(product))
        if any(count > 1 for _, count in handle_counts):
            flags.add("multiple_reactive_sites")
        if _degenerate_risk(components):
            flags.add("degenerate_risk")
        cases.append(
            ReviewCase(
                case_key=f"lx|{member['label']}|{product}",
                sampling_frame="ugi_applicability_stress",
                case_origin="lx_2024_source_routed_aldehyde_transfer",
                control_type="",
                proposed_reaction_id=UGI_REACTION_ID,
                product_smiles=product,
                components=components,
                proposed_role_to_component=role_map,
                proposed_reactive_atoms=analysis["target_site_signatures"][0],
                expected_reaction_id=UGI_REACTION_ID,
                expected_role_to_component=role_map,
                expected_components=components,
                expected_reactive_atoms=analysis["target_site_signatures"][0],
                source_structure_id=f"LX_2024-{member['label']}",
                source_schemes=("lx_2024_transfer",),
                source_decomposition_ids=(),
                source_studies=("lnpdb_v1:LX_2024",),
                known_synthesis_alignment=(
                    "exact_source_routed_aldehyde_but_proposed_ugi_product_not_observed"
                ),
                chemical_expectation="human_review_required",
                sampling_strata=tuple(
                    sorted(
                        {
                            "transferred_lx_2024_aldehyde",
                            *(("branching",) if "branching" in flags else ()),
                            *(
                                ("embedded_linker",)
                                if any(flag.startswith("embedded_") for flag in flags)
                                else ()
                            ),
                        }
                    )
                ),
                anchor_distance=None,
                component_frequencies=tuple((role, 0) for role in roles),
                handle_match_counts=handle_counts,
                structural_flags=tuple(sorted(flags)),
                mechanical_forward_reconstruction=True,
                proposed_role_assignment_mechanically_correct=True,
                proposed_reactive_atoms_mechanically_correct=True,
                target_matching_forward_outcomes=analysis["target_matching_forward_outcomes"],
                target_site_signatures=analysis["target_site_signatures"],
            )
        )
    return tuple(cases)


def _analysis_for_case(
    case: ReviewCase,
    compiled: Mapping[str, CompiledReaction],
    max_products: int,
) -> dict[str, Any]:
    return _forward_analysis(
        compiled[case.proposed_reaction_id],
        case.components,
        dict(case.proposed_role_to_component),
        case.product_smiles,
        max_products,
    )


def _with_analysis(
    case: ReviewCase,
    compiled: Mapping[str, CompiledReaction],
    max_products: int,
    *,
    role_correct: bool,
    sites_correct: bool,
) -> ReviewCase:
    analysis = _analysis_for_case(case, compiled, max_products)
    return replace(
        case,
        mechanical_forward_reconstruction=analysis["mechanical_forward_reconstruction"],
        proposed_role_assignment_mechanically_correct=role_correct,
        proposed_reactive_atoms_mechanically_correct=sites_correct,
        target_matching_forward_outcomes=analysis["target_matching_forward_outcomes"],
        target_site_signatures=analysis["target_site_signatures"],
    )


def _control_role_swaps(
    bases: Sequence[ReviewCase],
    count: int,
    seed: int,
    compiled: Mapping[str, CompiledReaction],
    max_products: int,
) -> tuple[ReviewCase, ...]:
    controls = []
    for base in sorted(bases, key=lambda item: _stable_key(seed, "role_swap", item.case_key)):
        if len(base.proposed_role_to_component) < 2:
            continue
        swapped = list(base.proposed_role_to_component)
        first_role, first_index = swapped[0]
        second_role, second_index = swapped[1]
        swapped[0] = (first_role, second_index)
        swapped[1] = (second_role, first_index)
        proposed_sites = dict(base.proposed_reactive_atoms)
        proposed_sites[first_role], proposed_sites[second_role] = (
            proposed_sites.get(second_role, ()),
            proposed_sites.get(first_role, ()),
        )
        control = replace(
            base,
            case_key=f"control|role_swap|{base.case_key}",
            sampling_frame="adversarial_control",
            case_origin="synthetic_adversarial_control",
            control_type="role_swap",
            proposed_role_to_component=tuple(swapped),
            proposed_reactive_atoms=tuple(
                (role, tuple(proposed_sites.get(role, ()))) for role, _ in swapped
            ),
            chemical_expectation="reject_incorrect_component_role_assignment",
            sampling_strata=("adversarial_role_swap",),
            review_id="",
        )
        control = _with_analysis(
            control,
            compiled,
            max_products,
            role_correct=False,
            sites_correct=False,
        )
        controls.append(control)
        if len(controls) == count:
            return tuple(controls)
    raise AuditError(f"could create only {len(controls)}/{count} role-swap controls")


def _control_cross_record_swaps(
    bases: Sequence[ReviewCase],
    count: int,
    seed: int,
    compiled: Mapping[str, CompiledReaction],
    max_products: int,
) -> tuple[ReviewCase, ...]:
    by_family: dict[str, list[ReviewCase]] = defaultdict(list)
    for base in bases:
        by_family[base.proposed_reaction_id].append(base)
    controls = []
    ordered = sorted(bases, key=lambda item: _stable_key(seed, "cross_swap", item.case_key))
    for base in ordered:
        family = sorted(
            by_family[base.proposed_reaction_id],
            key=lambda item: _stable_key(seed, "cross_donor", item.case_key),
        )
        for role, component_index in base.proposed_role_to_component:
            donor = next(
                (
                    candidate
                    for candidate in family
                    if candidate.components[dict(candidate.proposed_role_to_component)[role]]
                    != base.components[component_index]
                ),
                None,
            )
            if donor is None:
                continue
            components = list(base.components)
            donor_index = dict(donor.proposed_role_to_component)[role]
            components[component_index] = donor.components[donor_index]
            control = replace(
                base,
                case_key=f"control|cross_swap|{role}|{base.case_key}|{donor.case_key}",
                sampling_frame="adversarial_control",
                case_origin="synthetic_adversarial_control",
                control_type="cross_record_component_swap",
                components=tuple(components),
                proposed_reactive_atoms=(),
                chemical_expectation="reject_wrong_component_assignment",
                sampling_strata=("adversarial_cross_record_component_swap",),
                review_id="",
            )
            analysis = _analysis_for_case(control, compiled, max_products)
            if analysis["mechanical_forward_reconstruction"]:
                continue
            control = replace(
                control,
                mechanical_forward_reconstruction=False,
                proposed_role_assignment_mechanically_correct=True,
                proposed_reactive_atoms_mechanically_correct=False,
                target_matching_forward_outcomes=0,
                target_site_signatures=(),
            )
            controls.append(control)
            break
        if len(controls) == count:
            return tuple(controls)
    raise AuditError(f"could create only {len(controls)}/{count} cross-record component controls")


def _control_wrong_families(
    bases: Sequence[ReviewCase],
    count: int,
    seed: int,
    compiled: Mapping[str, CompiledReaction],
    max_products: int,
) -> tuple[ReviewCase, ...]:
    alternatives_by_count: dict[int, list[CompiledReaction]] = defaultdict(list)
    for reaction in compiled.values():
        alternatives_by_count[len(reaction.definition.reactant_roles)].append(reaction)
    controls = []
    for base in sorted(bases, key=lambda item: _stable_key(seed, "wrong_family", item.case_key)):
        alternatives = sorted(
            (
                reaction
                for reaction in alternatives_by_count[len(base.components)]
                if reaction.definition.reaction_id != base.proposed_reaction_id
            ),
            key=lambda reaction: _stable_key(
                seed,
                "wrong_family_choice",
                base.case_key,
                reaction.definition.reaction_id,
            ),
        )
        for alternative in alternatives:
            roles = tuple(role.name for role in alternative.definition.reactant_roles)
            role_map = tuple((role, index) for index, role in enumerate(roles))
            control = replace(
                base,
                case_key=(
                    f"control|wrong_family|{alternative.definition.reaction_id}|{base.case_key}"
                ),
                sampling_frame="adversarial_control",
                case_origin="synthetic_adversarial_control",
                control_type="wrong_reaction_family",
                proposed_reaction_id=alternative.definition.reaction_id,
                proposed_role_to_component=role_map,
                proposed_reactive_atoms=(),
                chemical_expectation="reject_wrong_reaction_family",
                sampling_strata=("adversarial_wrong_reaction_family",),
                review_id="",
            )
            analysis = _analysis_for_case(control, compiled, max_products)
            if analysis["mechanical_forward_reconstruction"]:
                continue
            controls.append(
                replace(
                    control,
                    mechanical_forward_reconstruction=False,
                    proposed_role_assignment_mechanically_correct=False,
                    proposed_reactive_atoms_mechanically_correct=False,
                    target_matching_forward_outcomes=0,
                    target_site_signatures=(),
                )
            )
            break
        if len(controls) == count:
            return tuple(controls)
    raise AuditError(f"could create only {len(controls)}/{count} wrong-family controls")


def _control_non_ugi_as_ugi(
    bases: Sequence[ReviewCase],
    count: int,
    seed: int,
    compiled: Mapping[str, CompiledReaction],
    max_products: int,
) -> tuple[ReviewCase, ...]:
    ugi_base = next(
        (
            base
            for base in sorted(
                bases, key=lambda item: _stable_key(seed, "ugi_donor", item.case_key)
            )
            if base.proposed_reaction_id == UGI_REACTION_ID
        ),
        None,
    )
    if ugi_base is None:
        raise AuditError("cannot build non-Ugi-as-Ugi controls without a Ugi base")
    controls = []
    non_ugi = sorted(
        (base for base in bases if base.expected_reaction_id != UGI_REACTION_ID),
        key=lambda item: _stable_key(seed, "non_ugi_target", item.case_key),
    )
    for target in non_ugi:
        control = replace(
            ugi_base,
            case_key=f"control|non_ugi_as_ugi|{target.case_key}",
            sampling_frame="adversarial_control",
            case_origin="synthetic_adversarial_control",
            control_type="non_ugi_as_ugi",
            product_smiles=target.product_smiles,
            expected_reaction_id=target.expected_reaction_id,
            expected_role_to_component=target.expected_role_to_component,
            expected_components=target.expected_components,
            expected_reactive_atoms=target.expected_reactive_atoms,
            source_structure_id=target.source_structure_id,
            source_schemes=target.source_schemes,
            source_decomposition_ids=target.source_decomposition_ids,
            source_studies=target.source_studies,
            known_synthesis_alignment=target.known_synthesis_alignment,
            chemical_expectation="reject_non_ugi_product_forced_through_ugi",
            sampling_strata=("adversarial_non_ugi_as_ugi",),
            review_id="",
        )
        analysis = _analysis_for_case(control, compiled, max_products)
        if analysis["mechanical_forward_reconstruction"]:
            continue
        controls.append(
            replace(
                control,
                mechanical_forward_reconstruction=False,
                proposed_role_assignment_mechanically_correct=False,
                proposed_reactive_atoms_mechanically_correct=False,
                target_matching_forward_outcomes=0,
                target_site_signatures=(),
            )
        )
        if len(controls) == count:
            return tuple(controls)
    raise AuditError(f"could create only {len(controls)}/{count} non-Ugi-as-Ugi controls")


def _alternative_handle_atoms(
    base: ReviewCase,
    reaction: CompiledReaction,
) -> tuple[str, tuple[int, ...]] | None:
    expected_sites = dict(base.expected_reactive_atoms)
    component_by_role = {
        role: base.components[index] for role, index in base.proposed_role_to_component
    }
    for role_definition, handle in zip(
        reaction.definition.reactant_roles,
        reaction.handles,
        strict=True,
    ):
        role = role_definition.name
        molecule = _molecule(component_by_role[role], f"alternative-site/{role}")
        selected = set(expected_sites.get(role, ()))
        matches = sorted(molecule.GetSubstructMatches(handle, uniquify=True))
        for match in matches:
            if not set(match).intersection(selected):
                return role, tuple(match)
    return None


def _control_wrong_sites(
    bases: Sequence[ReviewCase],
    count: int,
    seed: int,
    compiled: Mapping[str, CompiledReaction],
    max_products: int,
) -> tuple[ReviewCase, ...]:
    controls = []
    for base in sorted(bases, key=lambda item: _stable_key(seed, "wrong_site", item.case_key)):
        reaction = compiled[base.proposed_reaction_id]
        alternative = _alternative_handle_atoms(base, reaction)
        if alternative is None:
            continue
        role, wrong_atoms = alternative
        proposed_sites = dict(base.proposed_reactive_atoms)
        proposed_sites[role] = wrong_atoms
        control = replace(
            base,
            case_key=f"control|wrong_site|{role}|{base.case_key}",
            sampling_frame="adversarial_control",
            case_origin="synthetic_adversarial_control",
            control_type="wrong_attachment_site",
            proposed_reactive_atoms=tuple(
                (name, tuple(proposed_sites.get(name, ())))
                for name, _ in base.proposed_role_to_component
            ),
            chemical_expectation="reject_wrong_reactive_site_assignment",
            sampling_strata=("adversarial_wrong_attachment_site",),
            review_id="",
        )
        control = _with_analysis(
            control,
            compiled,
            max_products,
            role_correct=True,
            sites_correct=False,
        )
        if not control.mechanical_forward_reconstruction:
            raise AuditError("wrong-site control unexpectedly lost exact graph reconstruction")
        controls.append(control)
        if len(controls) == count:
            return tuple(controls)
    raise AuditError(f"could create only {len(controls)}/{count} wrong-site controls")


def _control_degenerate_reconstructing(
    bases: Sequence[ReviewCase],
    count: int,
    seed: int,
) -> tuple[ReviewCase, ...]:
    ranked = sorted(
        (
            base
            for base in bases
            if base.mechanical_forward_reconstruction
            and (
                "degenerate_risk" in base.structural_flags
                or base.proposed_reaction_id == "reductive_amination_amine_aldehyde"
            )
        ),
        key=lambda item: (
            item.proposed_reaction_id != "reductive_amination_amine_aldehyde",
            _stable_key(seed, "degenerate", item.case_key),
        ),
    )
    if len(ranked) < count:
        raise AuditError(
            f"could create only {len(ranked)}/{count} degenerate reconstruction controls"
        )
    return tuple(
        replace(
            base,
            case_key=f"control|degenerate|{base.case_key}",
            sampling_frame="adversarial_control",
            case_origin="synthetic_adversarial_control",
            control_type="degenerate_reconstructing",
            chemical_expectation=(
                "mechanically_reconstructing_chemical_challenge_human_review_required"
            ),
            sampling_strata=("adversarial_degenerate_reconstructing",),
            review_id="",
        )
        for base in ranked[:count]
    )


def _prepare_adversarial_controls(
    bases: Sequence[ReviewCase],
    degenerate_bases: Sequence[ReviewCase],
    control_types: Sequence[str],
    count_per_type: int,
    seed: int,
    compiled: Mapping[str, CompiledReaction],
    max_products: int,
) -> tuple[ReviewCase, ...]:
    builders = {
        "role_swap": lambda: _control_role_swaps(
            bases, count_per_type, seed, compiled, max_products
        ),
        "cross_record_component_swap": lambda: _control_cross_record_swaps(
            bases, count_per_type, seed, compiled, max_products
        ),
        "wrong_reaction_family": lambda: _control_wrong_families(
            bases, count_per_type, seed, compiled, max_products
        ),
        "non_ugi_as_ugi": lambda: _control_non_ugi_as_ugi(
            bases, count_per_type, seed, compiled, max_products
        ),
        "wrong_attachment_site": lambda: _control_wrong_sites(
            bases, count_per_type, seed, compiled, max_products
        ),
        "degenerate_reconstructing": lambda: _control_degenerate_reconstructing(
            degenerate_bases, count_per_type, seed
        ),
    }
    controls = []
    for control_type in control_types:
        controls.extend(builders[control_type]())
    return tuple(controls)


def _assign_review_ids(
    cases: Sequence[ReviewCase],
    seed: int,
) -> tuple[ReviewCase, ...]:
    assigned = []
    identifiers: set[str] = set()
    for case in cases:
        identifier = f"M005-{_stable_key(seed, 'review_id', case.case_key)[:12].upper()}"
        if identifier in identifiers:
            raise AuditError(f"review ID collision: {identifier}")
        identifiers.add(identifier)
        assigned.append(replace(case, review_id=identifier))
    return tuple(
        sorted(
            assigned,
            key=lambda case: _stable_key(seed, "blind_order", case.review_id),
        )
    )


def _role_payload(
    role_to_component: Sequence[tuple[str, int]],
) -> list[dict[str, Any]]:
    return [
        {"role": role, "component_index": component_index}
        for role, component_index in role_to_component
    ]


def _reactive_atom_payload(
    reactive_atoms: Sequence[tuple[str, tuple[int, ...]]],
) -> dict[str, list[int]]:
    return {role: list(atom_indices) for role, atom_indices in reactive_atoms}


def _blind_row(case: ReviewCase) -> dict[str, str]:
    return {
        "review_id": case.review_id,
        "proposed_reaction_family": case.proposed_reaction_id,
        "product_smiles": case.product_smiles,
        "proposed_roles_json": _json_cell(_role_payload(case.proposed_role_to_component)),
        "proposed_components_json": _json_cell(list(case.components)),
        "proposed_reactive_atoms_json": _json_cell(
            _reactive_atom_payload(case.proposed_reactive_atoms)
        ),
        "reviewer_id": "",
        "plausible_final_assembly": "",
        "component_identities_and_roles_correct": "",
        "reactive_atoms_correct": "",
        "exact_forward_reconstruction": "",
        "plausible_alternative_decomposition": "",
        "confidence": "",
        "rejection_reason": "",
        "notes": "",
    }


def _answer_row(case: ReviewCase) -> dict[str, str]:
    return {
        "review_id": case.review_id,
        "sampling_frame": case.sampling_frame,
        "case_origin": case.case_origin,
        "control_type": case.control_type,
        "source_structure_id": case.source_structure_id,
        "source_schemes_json": _json_cell(list(case.source_schemes)),
        "source_decomposition_ids_json": _json_cell(list(case.source_decomposition_ids)),
        "source_studies_json": _json_cell(list(case.source_studies)),
        "expected_reaction_family": case.expected_reaction_id,
        "expected_roles_json": _json_cell(_role_payload(case.expected_role_to_component)),
        "expected_components_json": _json_cell(list(case.expected_components)),
        "expected_reactive_atoms_json": _json_cell(
            _reactive_atom_payload(case.expected_reactive_atoms)
        ),
        "mechanical_forward_reconstruction": str(case.mechanical_forward_reconstruction).lower(),
        "proposed_role_assignment_mechanically_correct": str(
            case.proposed_role_assignment_mechanically_correct
        ).lower(),
        "proposed_reactive_atoms_mechanically_correct": str(
            case.proposed_reactive_atoms_mechanically_correct
        ).lower(),
        "target_matching_forward_outcomes": str(case.target_matching_forward_outcomes),
        "unique_target_site_signatures": str(len(case.target_site_signatures)),
        "known_synthesis_alignment": case.known_synthesis_alignment,
        "chemical_expectation": case.chemical_expectation,
        "sampling_strata_json": _json_cell(list(case.sampling_strata)),
        "anchor_distance": (
            "" if case.anchor_distance is None else format(case.anchor_distance, ".8f")
        ),
        "component_frequencies_json": _json_cell(dict(case.component_frequencies)),
        "handle_match_counts_json": _json_cell(dict(case.handle_match_counts)),
        "structural_flags_json": _json_cell(list(case.structural_flags)),
    }


def _svg(smiles: str, highlighted_atoms: Sequence[int] = ()) -> str:
    molecule = _molecule(smiles, "review rendering")
    drawer = rdMolDraw2D.MolDraw2DSVG(520, 300)
    options = drawer.drawOptions()
    options.addAtomIndices = True
    options.padding = 0.08
    drawer.DrawMolecule(
        molecule,
        highlightAtoms=[
            atom_index
            for atom_index in highlighted_atoms
            if 0 <= atom_index < molecule.GetNumAtoms()
        ],
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _html_packet(cases: Sequence[ReviewCase]) -> bytes:
    cards = []
    for case in cases:
        role_map = dict(case.proposed_role_to_component)
        reactive_atoms = dict(case.proposed_reactive_atoms)
        component_cards = []
        for role, component_index in case.proposed_role_to_component:
            smiles = case.components[component_index]
            component_cards.append(
                "<section class='component'>"
                f"<h3>Component {component_index + 1}: {html.escape(role)}</h3>"
                f"<div class='smiles'>{html.escape(smiles)}</div>"
                f"{_svg(smiles, reactive_atoms.get(role, ()))}"
                f"<p>Proposed reactive atom indices: "
                f"{html.escape(', '.join(str(value) for value in reactive_atoms.get(role, ())) or 'none')}</p>"
                "</section>"
            )
        cards.append(
            "<article class='review-card'>"
            f"<h2>{html.escape(case.review_id)} · "
            f"{html.escape(case.proposed_reaction_id)}</h2>"
            "<p class='instruction'>Judge the proposed final assembly, role assignment, "
            "reactive atoms, and exact forward reconstruction. Atom indices are shown in "
            "each drawing.</p>"
            "<section class='product'>"
            "<h3>Proposed product</h3>"
            f"<div class='smiles'>{html.escape(case.product_smiles)}</div>"
            f"{_svg(case.product_smiles)}"
            "</section>"
            f"<div class='components'>{''.join(component_cards)}</div>"
            "<div class='annotations'>"
            "<div>Plausible final assembly: □ yes □ no □ uncertain</div>"
            "<div>Identities and roles correct: □ yes □ no □ uncertain</div>"
            "<div>Reactive atoms correct: □ yes □ no □ uncertain</div>"
            "<div>Exact forward reconstruction: □ yes □ no □ uncertain</div>"
            "<div>Plausible alternative: □ yes □ no □ uncertain</div>"
            "<div>Confidence (1–5): ______</div>"
            "<div>Rejection reason / notes:</div><div class='notes'></div>"
            "</div>"
            "</article>"
        )
        if set(role_map.values()) != set(range(len(case.components))):
            raise AuditError(f"{case.review_id}: proposed component mapping is not bijective")
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FORGE M0-05 blinded decomposition review</title>
<style>
  :root {{ color-scheme: light; font-family: Arial, Helvetica, sans-serif; }}
  body {{ margin: 0; color: #17202a; background: #f3f5f7; }}
  header {{ padding: 24px 32px; background: #15324a; color: white; }}
  header h1 {{ margin: 0 0 8px; }}
  header p {{ margin: 4px 0; max-width: 1000px; }}
  main {{ padding: 24px; }}
  .review-card {{
    background: white; margin: 0 auto 24px; padding: 22px; max-width: 1120px;
    border: 1px solid #c8d1d9; border-radius: 8px; break-after: page;
  }}
  .review-card h2 {{ margin-top: 0; color: #15324a; }}
  .instruction {{ background: #eef5fa; border-left: 4px solid #2f6f9f; padding: 10px; }}
  .components {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 12px; }}
  .component, .product {{ border: 1px solid #d6dde3; padding: 10px; overflow: hidden; }}
  svg {{ width: 100%; height: auto; }}
  .smiles {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; overflow-wrap: anywhere; font-size: 12px; }}
  .annotations {{ margin-top: 16px; display: grid; gap: 9px; }}
  .notes {{ height: 80px; border: 1px solid #75808a; }}
  @media print {{
    body {{ background: white; }}
    header {{ color: black; background: white; border-bottom: 2px solid black; }}
    main {{ padding: 0; }}
    .review-card {{ border: none; margin: 0; max-width: none; }}
  }}
</style>
</head>
<body>
<header>
  <h1>FORGE M0-05 blinded decomposition review</h1>
  <p>Review independently. Do not consult the answer key until both reviewers have submitted frozen annotations.</p>
  <p>The packet mixes corpus decompositions, applicability stress cases, and controls. Their identities are intentionally hidden.</p>
</header>
<main>{"".join(cards)}</main>
</body>
</html>
"""
    return document.encode()


def _sampling_manifest(
    config: Mapping[str, Any],
    cases: Sequence[ReviewCase],
    observed_summary: Mapping[str, Any],
) -> dict[str, Any]:
    by_frame = Counter(case.sampling_frame for case in cases)
    by_control = Counter(case.control_type for case in cases if case.control_type)
    by_family = Counter(case.proposed_reaction_id for case in cases)
    return {
        "schema_version": "m0_05_sampling_manifest.v1",
        "algorithm_version": ALGORITHM_VERSION,
        "seed": config["seed"],
        "sampling_method": config["sampling_method"],
        "packet_rows": len(cases),
        "frames_before_blinding": dict(sorted(by_frame.items())),
        "adversarial_controls_before_blinding": dict(sorted(by_control.items())),
        "proposed_reaction_families_after_blinding": dict(sorted(by_family.items())),
        "observed_frame": observed_summary,
        "blind_order_review_ids": [case.review_id for case in cases],
        "human_review_status": "awaiting_human_review",
    }


def _artifact_record(path: str, payload: bytes) -> dict[str, Any]:
    return {
        "path": f"results/m0_05/{path}",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _result_payload(
    config: Mapping[str, Any],
    config_record: Mapping[str, Any],
    input_provenance: Mapping[str, Any],
    cases: Sequence[ReviewCase],
    observed_summary: Mapping[str, Any],
    source_m0_04_result: Mapping[str, Any],
    artifacts: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    controls = Counter(case.control_type for case in cases if case.control_type)
    frames = Counter(case.sampling_frame for case in cases)
    exact_mechanical = sum(case.mechanical_forward_reconstruction for case in cases)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": "M0-05",
        "generated_utc": config["generated_utc"],
        "status": "awaiting_human_chemist_review",
        "algorithm_version": ALGORITHM_VERSION,
        "seed": config["seed"],
        "randomness": {
            "used": True,
            "method": config["sampling_method"],
        },
        "inputs": {
            "config": dict(config_record),
            **input_provenance,
        },
        "source_m0_04_coverage": source_m0_04_result.get("schemes", {}),
        "packet_summary": {
            "rows": len(cases),
            "frames": dict(sorted(frames.items())),
            "adversarial_controls": dict(sorted(controls.items())),
            "mechanically_forward_reconstructing_rows": exact_mechanical,
            "observed_frame": observed_summary,
        },
        "human_review": {
            "status": "awaiting_human_review",
            "required_reviewers": config["human_review"]["required_reviewers"],
            "precision_per_family": None,
            "specificity_per_control_type": None,
            "confidence_intervals": None,
            "reviewer_agreement": None,
            "family_dispositions": {
                reaction_id: "pending_human_review" for reaction_id in observed_summary["families"]
            },
            "allowed_final_dispositions": config["human_review"]["family_dispositions"],
            "claim_boundary": (
                "The frozen procedure produced exact, forward-reconstructing "
                "decompositions under the declared transforms. Chemical precision "
                "and specificity have not yet been established."
            ),
        },
        "artifacts": artifacts,
    }


def run_audit(
    config_path: Path,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Generate the complete M0-05 review packet without overwriting results."""

    config = load_config(config_path)
    inputs, input_provenance = _resolve_and_verify_inputs(config, repo_root)
    if output_dir.exists():
        raise AuditError(f"M0-05 output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    m0_04_config = load_m0_04_config(inputs["m0_04_config"])
    definitions = load_reaction_definitions(
        (
            inputs["qualified_reaction_families"],
            inputs["qualified_reactions"],
        ),
        expected_count=int(config["expected_reaction_count"]),
        role_policy_overrides=m0_04_config["role_policy_overrides"],
    )
    compiled_tuple = compile_reactions(definitions)
    compiled = {reaction.definition.reaction_id: reaction for reaction in compiled_tuple}
    if UGI_REACTION_ID not in compiled:
        raise AuditError(f"qualified registries do not contain {UGI_REACTION_ID}")

    r0_rows = _read_r0(inputs["r0_structures"])
    candidates = _read_source_candidates(
        inputs["decomposition_candidates"],
        r0_rows,
    )
    max_products = int(config["forward_verification"]["max_products_per_case"])
    sampling = config["sampling"]
    observed_cases, degenerate_challenge_cases, observed_summary = _prepare_observed_cases(
        candidates,
        compiled,
        int(sampling["observed_target_per_family"]),
        int(config["seed"]),
        config["fingerprint"],
        max_products,
        int(sampling["adversarial_count_per_type"]),
    )
    virtual_cases = _prepare_virtual_ugi_cases(
        inputs["agile_virtual_ugi3_product_ledger"],
        compiled[UGI_REACTION_ID],
        int(sampling["virtual_ugi_stress_count"]),
        int(config["seed"]) + 1,
        config["fingerprint"],
        max_products,
        {(case.product_smiles, case.components) for case in observed_cases},
    )
    lx_cases = _prepare_lx_transfer_cases(
        inputs["lnpdb_paper_route_reviews"],
        compiled[UGI_REACTION_ID],
        int(sampling["lx_2024_transfer_count"]),
        max_products,
    )
    base_cases = (*observed_cases, *virtual_cases, *lx_cases)
    controls = _prepare_adversarial_controls(
        base_cases,
        degenerate_challenge_cases,
        tuple(sampling["adversarial_types"]),
        int(sampling["adversarial_count_per_type"]),
        int(config["seed"]) + 2,
        compiled,
        max_products,
    )
    cases = _assign_review_ids((*base_cases, *controls), int(config["seed"]))
    blind_rows = [_blind_row(case) for case in cases]
    answer_rows = [_answer_row(case) for case in cases]
    manifest = _sampling_manifest(config, cases, observed_summary)

    blind_payload = _csv_bytes(blind_rows, BLIND_FIELDS)
    html_payload = _gzip_bytes(_html_packet(cases))
    answer_payload = _gzip_bytes(_csv_bytes(answer_rows, ANSWER_FIELDS))
    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    artifact_payloads = {
        "review_packet.csv": blind_payload,
        "review_packet.html.gz": html_payload,
        "answer_key.csv.gz": answer_payload,
        "sampling_manifest.json": manifest_payload,
    }
    artifacts = {
        name: _artifact_record(name, payload) for name, payload in artifact_payloads.items()
    }
    source_m0_04_result = _load_json(inputs["m0_04_result"], "M0-04 result")
    try:
        config_display_path = str(config_path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        config_display_path = str(config_path)
    result = _result_payload(
        config,
        {
            "path": config_display_path,
            "sha256": sha256_file(config_path),
            "bytes": config_path.stat().st_size,
        },
        input_provenance,
        cases,
        observed_summary,
        source_m0_04_result,
        artifacts,
    )
    result_payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            dir=output_dir.parent,
        )
    )
    try:
        for name, payload in artifact_payloads.items():
            (staging / name).write_bytes(payload)
        (staging / "result.json").write_bytes(result_payload)
        os.replace(staging, output_dir)
    except Exception:
        for child in staging.iterdir():
            child.unlink()
        staging.rmdir()
        raise
    return result
