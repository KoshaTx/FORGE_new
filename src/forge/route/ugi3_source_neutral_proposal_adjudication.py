"""Source-neutral adjudication of proposal-derived Ugi component routes.

Graph2Edits supplies search hypotheses only.  This module independently asks
whether a hypothesis reconstructs the expected final step, remains inside a
source-derived structural scope, uses an evidenced reaction series, passes a
conservative operational chemistry screen, and closes every projected L3 leaf.
The proposal score is deliberately absent from every decision.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

CONFIG_SCHEMA_VERSION = "phase1_ugi3_source_neutral_proposal_adjudication_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi3_source_neutral_proposal_adjudication.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi3_source_neutral_proposal_adjudication_ledger.v1"

FATTY_ALDEHYDE = "fatty_acid_diol_esterification_then_alcohol_oxidation"
DIRECT_ALDEHYDE = "primary_alcohol_oxidation"
ISOCYANIDE = "primary_amine_formylation_then_formamide_dehydration"
SUPPORTED_PROGRAMS = (FATTY_ALDEHYDE, DIRECT_ALDEHYDE, ISOCYANIDE)

EXACT = "exact_complete_current"
FAMILY_ALL = "family_projected_all_current_leaves"
QUALIFIED_FAMILY = "source_bounded_family_route_all_current"
UNRESOLVED = "unresolved_or_unsupported"

_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_ALDEHYDE = Chem.MolFromSmarts("[CX3H1](=O)[#6]")
_ESTER = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
_ISOCYANIDE = Chem.MolFromSmarts("[N+]#[C-]")
_PRIMARY_ALCOHOL = Chem.MolFromSmarts("[CH2][OX2H1]")
_FORMAMIDE = Chem.MolFromSmarts("[NX3H1][CH1]=[OX1]")


class Ugi3ProposalAdjudicationError(RuntimeError):
    """Raised when a frozen input or adjudication contract changes."""


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_stable_json(dict(value)).encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise Ugi3ProposalAdjudicationError(f"JSON object required: {path}")
    return value


def _read_jsonl_gzip(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise Ugi3ProposalAdjudicationError(f"JSONL rows are malformed: {path}")
    return rows


def _read_csv_gzip(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _gzip_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
        zipped.write(b"".join((_stable_json(dict(row)) + "\n").encode() for row in rows))
    return raw.getvalue()


def _molecule(smiles: str) -> Chem.Mol:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise Ugi3ProposalAdjudicationError(f"invalid connected molecule: {smiles}")
    return molecule


def _canonical(smiles: str) -> str:
    return Chem.MolToSmiles(_molecule(smiles), canonical=True, isomericSmiles=False)


def _count(molecule: Chem.Mol, query: Chem.Mol | None) -> int:
    if query is None:  # pragma: no cover - static SMARTS contract
        raise Ugi3ProposalAdjudicationError("static chemistry query failed to compile")
    return len(molecule.GetSubstructMatches(query, uniquify=True))


def _aldehyde_to_ester_oxygen_distance(molecule: Chem.Mol) -> int | None:
    if _ALDEHYDE is None or _ESTER is None:  # pragma: no cover
        raise Ugi3ProposalAdjudicationError("static chemistry query failed to compile")
    aldehydes = molecule.GetSubstructMatches(_ALDEHYDE, uniquify=True)
    esters = molecule.GetSubstructMatches(_ESTER, uniquify=True)
    if len(aldehydes) != 1 or len(esters) != 1:
        return None
    aldehyde_carbon = aldehydes[0][0]
    alkoxy_oxygen = esters[0][2]
    return len(Chem.GetShortestPath(molecule, aldehyde_carbon, alkoxy_oxygen)) - 1


def structural_features(smiles: str) -> dict[str, Any]:
    """Return transparent features used only for source-envelope membership."""

    molecule = _molecule(smiles)
    carbon_atoms = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 6]
    carbon_branches = sum(
        sum(neighbor.GetAtomicNum() == 6 for neighbor in atom.GetNeighbors()) >= 3
        for atom in carbon_atoms
    )
    alkene_count = sum(
        bond.GetBondType() is Chem.BondType.DOUBLE
        and all(atom.GetAtomicNum() == 6 for atom in (bond.GetBeginAtom(), bond.GetEndAtom()))
        for bond in molecule.GetBonds()
    )
    alkyne_count = sum(
        bond.GetBondType() is Chem.BondType.TRIPLE
        and all(atom.GetAtomicNum() == 6 for atom in (bond.GetBeginAtom(), bond.GetEndAtom()))
        for bond in molecule.GetBonds()
    )
    return {
        "carbon_count": len(carbon_atoms),
        "heavy_atom_count": molecule.GetNumHeavyAtoms(),
        "ring_count": molecule.GetRingInfo().NumRings(),
        "net_formal_charge": sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()),
        "element_signature": tuple(sorted({atom.GetSymbol() for atom in molecule.GetAtoms()})),
        "carbon_branch_points": carbon_branches,
        "carbon_alkene_count": alkene_count,
        "carbon_alkyne_count": alkyne_count,
        "aldehyde_count": _count(molecule, _ALDEHYDE),
        "ester_count": _count(molecule, _ESTER),
        "isocyanide_count": _count(molecule, _ISOCYANIDE),
        "primary_alcohol_count": _count(molecule, _PRIMARY_ALCOHOL),
        "formamide_count": _count(molecule, _FORMAMIDE),
        "aldehyde_to_ester_oxygen_distance": _aldehyde_to_ester_oxygen_distance(molecule),
    }


@dataclass(frozen=True)
class SourcePair:
    program_family: str
    route_id: str
    precursor: str
    target: str


@dataclass(frozen=True)
class PairScopeProfile:
    program_family: str
    pairs: tuple[SourcePair, ...]
    similarity_threshold: float
    target_numeric_bounds: Mapping[str, tuple[int, int]]
    precursor_numeric_bounds: Mapping[str, tuple[int, int]]
    target_categorical_values: Mapping[str, tuple[Any, ...]]
    precursor_categorical_values: Mapping[str, tuple[Any, ...]]


def _program_for_route(route_family_id: str) -> str | None:
    if route_family_id in {"agile_hexyl_aldehyde_two_step", "agile_butyl_aldehyde_two_step"}:
        return FATTY_ALDEHYDE
    if route_family_id == "agile_direct_aldehyde_oxidation":
        return DIRECT_ALDEHYDE
    if route_family_id in {"agile_isocyanide_two_step", "agile_additional_isocyanide_two_step"}:
        return ISOCYANIDE
    return None


def extract_source_pairs(route_payload: Mapping[str, Any]) -> tuple[SourcePair, ...]:
    rows: list[SourcePair] = []
    for route in route_payload.get("routes", []):
        if not isinstance(route, Mapping):
            continue
        program = _program_for_route(str(route.get("route_family_id")))
        steps = route.get("steps")
        if program is None or not isinstance(steps, list) or not steps:
            continue
        final_step = steps[-1]
        reactants = final_step.get("reactants")
        product = final_step.get("product")
        if (
            not isinstance(reactants, list)
            or len(reactants) != 1
            or not isinstance(product, Mapping)
        ):
            raise Ugi3ProposalAdjudicationError("source final step is not unimolecular")
        precursor = reactants[0]
        if not isinstance(precursor, Mapping):
            raise Ugi3ProposalAdjudicationError("source precursor is malformed")
        rows.append(
            SourcePair(
                program_family=program,
                route_id=str(route["route_id"]),
                precursor=_canonical(str(precursor["canonical_smiles"])),
                target=_canonical(str(product["canonical_smiles"])),
            )
        )
    if Counter(row.program_family for row in rows) != Counter(
        {FATTY_ALDEHYDE: 15, DIRECT_ALDEHYDE: 2, ISOCYANIDE: 7}
    ):
        raise Ugi3ProposalAdjudicationError("exact-source route-family census changed")
    return tuple(rows)


def _fingerprint(smiles: str):
    return _GENERATOR.GetFingerprint(_molecule(smiles))


def _pair_similarity(left: SourcePair, right: SourcePair) -> float:
    target = DataStructs.TanimotoSimilarity(_fingerprint(left.target), _fingerprint(right.target))
    precursor = DataStructs.TanimotoSimilarity(
        _fingerprint(left.precursor), _fingerprint(right.precursor)
    )
    return min(float(target), float(precursor))


_NUMERIC_FEATURES = (
    "carbon_count",
    "heavy_atom_count",
    "carbon_branch_points",
    "carbon_alkene_count",
    "carbon_alkyne_count",
)
_CATEGORICAL_FEATURES = (
    "ring_count",
    "net_formal_charge",
    "element_signature",
    "aldehyde_count",
    "ester_count",
    "isocyanide_count",
    "primary_alcohol_count",
    "formamide_count",
    "aldehyde_to_ester_oxygen_distance",
)


def _feature_profile(
    rows: Sequence[str],
) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[Any, ...]]]:
    features = [structural_features(value) for value in rows]
    numeric = {
        name: (min(int(row[name]) for row in features), max(int(row[name]) for row in features))
        for name in _NUMERIC_FEATURES
    }
    categorical = {
        name: tuple(sorted({row[name] for row in features}, key=repr))
        for name in _CATEGORICAL_FEATURES
    }
    return numeric, categorical


def build_scope_profiles(
    pairs: Sequence[SourcePair], *, minimum_similarity_floor: float
) -> dict[str, PairScopeProfile]:
    output: dict[str, PairScopeProfile] = {}
    for program in SUPPORTED_PROGRAMS:
        members = tuple(row for row in pairs if row.program_family == program)
        if len(members) < 2:
            raise Ugi3ProposalAdjudicationError("source scope requires at least two route pairs")
        nearest = []
        for index, member in enumerate(members):
            nearest.append(
                max(
                    _pair_similarity(member, other) for j, other in enumerate(members) if j != index
                )
            )
        target_numeric, target_categorical = _feature_profile([row.target for row in members])
        precursor_numeric, precursor_categorical = _feature_profile(
            [row.precursor for row in members]
        )
        output[program] = PairScopeProfile(
            program_family=program,
            pairs=members,
            similarity_threshold=max(float(minimum_similarity_floor), min(nearest)),
            target_numeric_bounds=target_numeric,
            precursor_numeric_bounds=precursor_numeric,
            target_categorical_values=target_categorical,
            precursor_categorical_values=precursor_categorical,
        )
    return output


def _inside_feature_profile(
    smiles: str,
    *,
    numeric_bounds: Mapping[str, tuple[int, int]],
    categorical_values: Mapping[str, tuple[Any, ...]],
) -> bool:
    features = structural_features(smiles)
    return all(
        low <= int(features[name]) <= high for name, (low, high) in numeric_bounds.items()
    ) and all(features[name] in values for name, values in categorical_values.items())


def assess_pair_scope(
    *, program_family: str, precursor: str, target: str, profile: PairScopeProfile
) -> dict[str, Any]:
    if program_family != profile.program_family:
        raise Ugi3ProposalAdjudicationError("scope profile program mismatch")
    candidate = SourcePair(program_family, "candidate", _canonical(precursor), _canonical(target))
    exact_pair = any(
        candidate.precursor == row.precursor and candidate.target == row.target
        for row in profile.pairs
    )
    target_inside = _inside_feature_profile(
        candidate.target,
        numeric_bounds=profile.target_numeric_bounds,
        categorical_values=profile.target_categorical_values,
    )
    precursor_inside = _inside_feature_profile(
        candidate.precursor,
        numeric_bounds=profile.precursor_numeric_bounds,
        categorical_values=profile.precursor_categorical_values,
    )
    nearest = max(_pair_similarity(candidate, row) for row in profile.pairs)
    qualified = exact_pair or (
        target_inside and precursor_inside and nearest >= profile.similarity_threshold
    )
    return {
        "qualified": qualified,
        "exact_source_pair": exact_pair,
        "target_feature_envelope": target_inside,
        "precursor_feature_envelope": precursor_inside,
        "nearest_source_pair_similarity": nearest,
        "minimum_pair_similarity": profile.similarity_threshold,
        "source_pair_count": len(profile.pairs),
        "scope_is_success_probability": False,
    }


def _expected_reaction_fragment(program: str) -> str:
    if program in {FATTY_ALDEHYDE, DIRECT_ALDEHYDE}:
        return "primary_alcohol_oxidation"
    if program == ISOCYANIDE:
        return "formamide_dehydration"
    raise Ugi3ProposalAdjudicationError(f"unsupported program family: {program}")


def _forward_precursor(raw_row: Mapping[str, Any], *, program: str) -> str | None:
    expected = _expected_reaction_fragment(program)
    resolution = raw_row.get("raw_discovery_resolution")
    if not isinstance(resolution, Mapping):
        return None
    matches = []
    for trace in resolution.get("traces", []):
        if (
            isinstance(trace, Mapping)
            and bool(trace.get("target_reconstructed"))
            and expected in str(trace.get("reaction_id"))
        ):
            reactants = trace.get("role_ordered_reactants")
            if isinstance(reactants, list) and len(reactants) == 1:
                matches.append(_canonical(str(reactants[0])))
    return None if not matches else sorted(set(matches))[0]


def _best_semantic_by_proposal(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    output = {}
    for row in rows:
        proposal_sha = str(row.get("proposal_sha256"))
        if proposal_sha in output:
            raise Ugi3ProposalAdjudicationError("semantic proposal identities collide")
        output[proposal_sha] = row
    return output


def build_source_neutral_adjudication(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], bytes]:
    repo = repo.resolve()
    config = _read_json(config_path.resolve())
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3ProposalAdjudicationError("unsupported adjudication config")
    paths: dict[str, Path] = {}
    for label, record in config.get("inputs", {}).items():
        path = repo / str(record["path"])
        if _sha256_file(path) != record["sha256"]:
            raise Ugi3ProposalAdjudicationError(f"input hash changed: {label}")
        paths[str(label)] = path
    required = {
        "raw_proposal_ledger",
        "semantic_proposal_ledger",
        "graded_evidence_ledger",
        "route_assessment_ledger",
        "exact_source_routes",
    }
    if set(paths) != required:
        raise Ugi3ProposalAdjudicationError("adjudication input set changed")

    source_pairs = extract_source_pairs(_read_json(paths["exact_source_routes"]))
    floor = float(config["scope_policy"]["minimum_pair_similarity_floor"])
    profiles = build_scope_profiles(source_pairs, minimum_similarity_floor=floor)
    raw_rows = _read_jsonl_gzip(paths["raw_proposal_ledger"])
    semantic = _best_semantic_by_proposal(_read_jsonl_gzip(paths["semantic_proposal_ledger"]))
    graded_rows = _read_csv_gzip(paths["graded_evidence_ledger"])
    graded = {(row["role"], row["canonical_smiles"]): row for row in graded_rows}
    if len(graded) != len(graded_rows):
        raise Ugi3ProposalAdjudicationError("graded component identities collide")

    raw_by_component: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        key = (str(row["role"]), str(row["target_smiles"]))
        raw_by_component[key].append(row)

    adjudications: list[dict[str, Any]] = []
    qualified_components: set[tuple[str, str]] = set()
    disposition_counts: Counter[str] = Counter()
    for key, proposals in sorted(raw_by_component.items()):
        evidence = graded.get(key)
        program = "" if evidence is None else str(evidence["program_family"])
        chosen: Mapping[str, Any] | None = None
        precursor: str | None = None
        for proposal in sorted(proposals, key=lambda row: int(row["rank"])):
            semantic_row = semantic.get(str(proposal["proposal_sha256"]))
            if semantic_row is None or not bool(
                semantic_row.get("graph_consistent_discovery_hypothesis")
            ):
                continue
            if program in profiles:
                candidate_precursor = _forward_precursor(proposal, program=program)
                if candidate_precursor is not None:
                    chosen = proposal
                    precursor = candidate_precursor
                    break

        forward = chosen is not None and precursor is not None
        scope = (
            assess_pair_scope(
                program_family=program,
                precursor=str(precursor),
                target=key[1],
                profile=profiles[program],
            )
            if forward
            else {
                "qualified": False,
                "exact_source_pair": False,
                "target_feature_envelope": False,
                "precursor_feature_envelope": False,
                "nearest_source_pair_similarity": None,
                "minimum_pair_similarity": None,
                "source_pair_count": 0,
                "scope_is_success_probability": False,
            }
        )
        evidence_qualified = program in profiles and len(profiles[program].pairs) >= 2
        operational = bool(scope["target_feature_envelope"]) and bool(
            scope["precursor_feature_envelope"]
        )
        l3_closed = evidence is not None and evidence["projection_leaf_status"] == "all_current"
        qualified = bool(
            forward and scope["qualified"] and evidence_qualified and operational and l3_closed
        )
        if qualified:
            qualified_components.add(key)
            disposition = QUALIFIED_FAMILY
        elif forward and operational:
            disposition = "unqualified_missing_scope_evidence_or_l3"
        elif forward:
            disposition = "rejected_operational_or_scope_envelope"
        else:
            disposition = "no_expected_forward_consistent_proposal"
        disposition_counts[disposition] += 1
        adjudications.append(
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "role": key[0],
                "target_smiles": key[1],
                "program_family": program or None,
                "proposal_sha256": None if chosen is None else chosen["proposal_sha256"],
                "proposal_rank": None if chosen is None else int(chosen["rank"]),
                "precursor_smiles": precursor,
                "checks": {
                    "independent_expected_step_forward_consistent": forward,
                    "source_bounded_pair_scope": bool(scope["qualified"]),
                    "source_series_evidence_qualified": evidence_qualified,
                    "operationally_compatible": operational,
                    "l3_terminal_closed": l3_closed,
                },
                "scope": scope,
                "adjudicated_route_state": QUALIFIED_FAMILY if qualified else None,
                "adjudicated_route_value": 0.75 if qualified else None,
                "route_value_is_success_probability": False,
                "disposition": disposition,
                "proposal_model_score_used": False,
                "proposal_source_created_evidence": False,
                "route_closure_authorized": qualified,
            }
        )

    route_rows = _read_jsonl_gzip(paths["route_assessment_ledger"])
    arm_exact: dict[str, set[str]] = defaultdict(set)
    arm_qualified: dict[str, set[str]] = defaultdict(set)
    for row in route_rows:
        components = row.get("canonical_components")
        if not isinstance(components, Mapping) or len(components) != 3:
            raise Ugi3ProposalAdjudicationError("route row component contract changed")
        strict = {
            str(item["role"]): bool(item["strict_complete"])
            for item in row.get("potential", {}).get("roles", [])
        }
        if set(strict) != set(components):
            raise Ugi3ProposalAdjudicationError("strict route role assessment changed")
        product = str(row["canonical_product"])
        arm = str(row["arm_id"])
        if all(strict.values()):
            arm_exact[arm].add(product)
        if all(
            strict[str(role)] or (str(role), str(smiles)) in qualified_components
            for role, smiles in components.items()
        ):
            arm_qualified[arm].add(product)

    arm_summary = {
        arm: {
            "unique_products_strict_exact_complete": len(arm_exact[arm]),
            "unique_products_source_bounded_family_or_better": len(
                arm_exact[arm] | arm_qualified[arm]
            ),
            "incremental_unique_products": len(arm_qualified[arm] - arm_exact[arm]),
        }
        for arm in sorted(set(arm_exact) | set(arm_qualified))
    }
    content: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "source_neutral_proposal_adjudication_complete",
        "config": {
            "path": str(config_path.resolve().relative_to(repo)),
            "sha256": _sha256_file(config_path.resolve()),
        },
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": _sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "scope_profiles": {
            program: {
                "source_pairs": len(profile.pairs),
                "minimum_pair_similarity": profile.similarity_threshold,
                "target_numeric_bounds": profile.target_numeric_bounds,
                "precursor_numeric_bounds": profile.precursor_numeric_bounds,
                "target_categorical_values": profile.target_categorical_values,
                "precursor_categorical_values": profile.precursor_categorical_values,
            }
            for program, profile in profiles.items()
        },
        "summary": {
            "proposal_targets": len(raw_by_component),
            "qualified_component_routes": len(qualified_components),
            "component_dispositions": dict(sorted(disposition_counts.items())),
            "arms": arm_summary,
            "balanced_unique_products_source_bounded_family_or_better": min(
                (
                    row["unique_products_source_bounded_family_or_better"]
                    for row in arm_summary.values()
                ),
                default=0,
            ),
        },
        "decision": {
            "proposal_engine_qualified_as_route_evidence": False,
            "source_bounded_family_routes_available_for_value_contrast_audit": bool(
                qualified_components
            ),
            "production_evaluator_changed": False,
            "synthesis_tilting_promoted": False,
            "next_gate": "Measure nonuniform route-value contrast on the frozen matched guidance particles before authorizing any synthesis-tilt rerun.",
        },
        "scientific_authority": {
            "proposal_model_score_used": False,
            "proposal_reaction_class_used": False,
            "source_bounded_scope_is_success_probability": False,
            "raw_proposal_may_create_evidence": False,
            "new_family_hypothesis_may_close_route": False,
        },
        "artifacts": {},
    }
    ledger = _gzip_jsonl(adjudications)
    content["artifacts"]["adjudication_ledger.jsonl.gz"] = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "rows": len(adjudications),
        "sha256": hashlib.sha256(ledger).hexdigest(),
    }
    content["result_sha256"] = _content_sha256(content)
    return content, ledger


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "LEDGER_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "Ugi3ProposalAdjudicationError",
    "assess_pair_scope",
    "build_scope_profiles",
    "build_source_neutral_adjudication",
    "extract_source_pairs",
    "structural_features",
]
