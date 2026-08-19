"""Constitutional applicability-axis census for projected Ugi route families.

The census describes candidate-specific structure axes and recurrent buckets.
It does not qualify reaction-family scope, assign a scalar synthesis value,
estimate experimental success, run guidance, or select candidates.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.data.r1_prime_audit import sha256_bytes, sha256_file
from forge.route.ugi3_virtual_programs import (
    ALDEHYDE_QUERY,
    ALDEHYDE_ROLE,
    DIRECT_ALDEHYDE_PROGRAM,
    ESTER_PROGRAM,
    ISOCYANIDE_PROGRAM,
    ISOCYANIDE_QUERY,
    ISOCYANIDE_ROLE,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi3_family_applicability_census_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi3_family_applicability_census.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi3_family_applicability_census_ledger.v1"

RECURRENT_BUCKET = "recurrent_interpolation_bucket_candidate"
EXTRAPOLATION_REVIEW = "sparse_extrapolation_review"
MISSING_METADATA = "missing_axis_metadata"

FIELDS = (
    "schema_version",
    "role",
    "canonical_smiles",
    "program_family",
    "one_gap_product_count",
    "projected_leaf_currency",
    "projected_leaf_count",
    "current_projected_leaf_count",
    "reactive_handle_type",
    "reactive_handle_radius2_signature",
    "reactive_handle_radius2_sha256",
    "candidate_carbon_count",
    "maximum_reactive_center_to_carbon_distance_bonds",
    "longest_carbon_only_segment_atoms",
    "carbon_count_band",
    "carbon_carbon_double_bond_count",
    "carbon_carbon_triple_bond_count",
    "constitutional_unsaturation_positions_json",
    "unsaturation_position_zone_signature",
    "carbon_branch_point_count",
    "branch_point_distances_from_reactive_center_json",
    "branch_position_zone_signature",
    "axis_metadata_status",
    "applicability_bucket_id",
    "applicability_bucket_signature_json",
    "bucket_unique_component_count",
    "bucket_one_gap_product_occurrence_count",
    "applicability_census_class",
    "family_scope_qualified",
)

_EXPECTED_FAMILY_ROLES = {
    ESTER_PROGRAM: ALDEHYDE_ROLE,
    DIRECT_ALDEHYDE_PROGRAM: ALDEHYDE_ROLE,
    ISOCYANIDE_PROGRAM: ISOCYANIDE_ROLE,
}


class Ugi3FamilyApplicabilityCensusError(ValueError):
    """Raised when the applicability census cannot be reproduced safely."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise Ugi3FamilyApplicabilityCensusError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Ugi3FamilyApplicabilityCensusError(f"{label} must be an object")
    return value


def _read_graded_rows(path: Path) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise Ugi3FamilyApplicabilityCensusError(f"invalid graded ledger: {path}") from exc
    required = {
        "schema_version",
        "role",
        "canonical_smiles",
        "exact_assessment_outcome",
        "in_final_one_gap_triage",
        "one_gap_product_count",
        "program_family",
        "projected_leaf_count",
        "current_projected_leaf_count",
        "projection_leaf_status",
        "family_scope_status",
        "exact_scope_still_required",
    }
    if not rows or not required.issubset(rows[0]):
        raise Ugi3FamilyApplicabilityCensusError("graded ledger schema changed")
    return rows


def _gzip_csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        raise Ugi3FamilyApplicabilityCensusError("cannot serialize an empty census ledger")
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        compressed.write(text.getvalue().encode())
    return output.getvalue()


def _parse_positive_int(value: str, *, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise Ugi3FamilyApplicabilityCensusError(f"{label} must be an integer") from exc
    if parsed <= 0:
        raise Ugi3FamilyApplicabilityCensusError(f"{label} must be positive")
    return parsed


def _molecule(smiles: str) -> Chem.Mol:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise Ugi3FamilyApplicabilityCensusError("projected component is not one valid graph")
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    if canonical != smiles:
        raise Ugi3FamilyApplicabilityCensusError("projected component is not canonical")
    return molecule


def _bond_name(bond: Chem.Bond) -> str:
    if bond.GetIsAromatic():
        return "AROMATIC"
    names = {
        Chem.BondType.SINGLE: "SINGLE",
        Chem.BondType.DOUBLE: "DOUBLE",
        Chem.BondType.TRIPLE: "TRIPLE",
    }
    try:
        return names[bond.GetBondType()]
    except KeyError as exc:
        raise Ugi3FamilyApplicabilityCensusError("unsupported bond in handle neighborhood") from exc


def _atom_state(atom: Chem.Atom) -> dict[str, Any]:
    return {
        "aromatic": atom.GetIsAromatic(),
        "degree": atom.GetTotalDegree(),
        "element": atom.GetSymbol(),
        "formal_charge": atom.GetFormalCharge(),
        "hydrogens": atom.GetTotalNumHs(),
    }


def _rooted_radius2_signature(molecule: Chem.Mol, root: int) -> str:
    center = molecule.GetAtomWithIdx(root)
    neighbors = []
    for neighbor in center.GetNeighbors():
        second_shell = []
        for second in neighbor.GetNeighbors():
            if second.GetIdx() == root:
                continue
            second_shell.append(
                {
                    "atom": _atom_state(second),
                    "bond": _bond_name(
                        molecule.GetBondBetweenAtoms(neighbor.GetIdx(), second.GetIdx())
                    ),
                }
            )
        second_shell.sort(key=lambda value: json.dumps(value, sort_keys=True))
        neighbors.append(
            {
                "atom": _atom_state(neighbor),
                "bond": _bond_name(molecule.GetBondBetweenAtoms(root, neighbor.GetIdx())),
                "second_shell": second_shell,
            }
        )
    neighbors.sort(key=lambda value: json.dumps(value, sort_keys=True))
    return json.dumps(
        {"center": _atom_state(center), "neighbors": neighbors},
        separators=(",", ":"),
        sort_keys=True,
    )


def _reactive_center(molecule: Chem.Mol, role: str) -> tuple[int, str]:
    if role == ALDEHYDE_ROLE:
        query = ALDEHYDE_QUERY
        handle_type = "aldehyde_carbonyl"
        center_offset = 0
    elif role == ISOCYANIDE_ROLE:
        query = ISOCYANIDE_QUERY
        handle_type = "isocyanide_carbon"
        center_offset = 1
    else:
        raise Ugi3FamilyApplicabilityCensusError(f"unsupported projected role: {role}")
    if query is None:
        raise Ugi3FamilyApplicabilityCensusError("frozen reactive-handle query is unavailable")
    matches = molecule.GetSubstructMatches(query, uniquify=True)
    if len(matches) != 1:
        raise Ugi3FamilyApplicabilityCensusError(
            f"projected {role} requires exactly one reactive handle; observed {len(matches)}"
        )
    return matches[0][center_offset], handle_type


def _longest_carbon_segment_atoms(molecule: Chem.Mol) -> int:
    carbons = {atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 6}
    if not carbons:
        return 0
    adjacency = {
        index: {
            neighbor.GetIdx()
            for neighbor in molecule.GetAtomWithIdx(index).GetNeighbors()
            if neighbor.GetIdx() in carbons
        }
        for index in carbons
    }
    diameter = 0
    for start in carbons:
        distances = {start: 0}
        queue = [start]
        for current in queue:
            for neighbor in adjacency[current]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        diameter = max(diameter, max(distances.values()))
    return diameter + 1


def _relative_position_zone(distance: int, maximum_distance: int) -> str:
    if maximum_distance == 0:
        return "at_reactive_center"
    if 3 * distance <= maximum_distance:
        return "proximal"
    if 3 * distance < 2 * maximum_distance:
        return "middle"
    return "distal"


def constitutional_applicability_axes(role: str, smiles: str) -> dict[str, Any]:
    """Derive exact graph axes without inferring reaction scope or stereochemistry."""

    molecule = _molecule(smiles)
    reactive_center, handle_type = _reactive_center(molecule, role)
    distances = Chem.GetDistanceMatrix(molecule)[reactive_center]
    carbon_indices = {atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 6}
    maximum_distance = max(int(distances[index]) for index in carbon_indices)

    unsaturations = []
    double_count = 0
    triple_count = 0
    for bond in molecule.GetBonds():
        left = bond.GetBeginAtomIdx()
        right = bond.GetEndAtomIdx()
        if left not in carbon_indices or right not in carbon_indices or bond.GetIsAromatic():
            continue
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            bond_type = "double"
            double_count += 1
        elif bond.GetBondType() == Chem.BondType.TRIPLE:
            bond_type = "triple"
            triple_count += 1
        else:
            continue
        endpoint_distances = sorted((int(distances[left]), int(distances[right])))
        unsaturations.append(
            {
                "bond_type": bond_type,
                "far_endpoint_distance": endpoint_distances[1],
                "near_endpoint_distance": endpoint_distances[0],
                "position_zone": _relative_position_zone(endpoint_distances[0], maximum_distance),
            }
        )
    unsaturations.sort(
        key=lambda value: (
            value["near_endpoint_distance"],
            value["far_endpoint_distance"],
            value["bond_type"],
        )
    )

    branch_distances = []
    for index in carbon_indices:
        carbon_neighbors = sum(
            neighbor.GetIdx() in carbon_indices
            for neighbor in molecule.GetAtomWithIdx(index).GetNeighbors()
        )
        if carbon_neighbors >= 3:
            branch_distances.append(int(distances[index]))
    branch_distances.sort()

    handle_signature = _rooted_radius2_signature(molecule, reactive_center)
    unsaturation_zones = (
        "|".join(f"{item['bond_type']}:{item['position_zone']}" for item in unsaturations) or "none"
    )
    branch_zones = (
        "|".join(
            _relative_position_zone(distance, maximum_distance) for distance in branch_distances
        )
        or "none"
    )
    return {
        "reactive_handle_type": handle_type,
        "reactive_handle_radius2_signature": handle_signature,
        "reactive_handle_radius2_sha256": hashlib.sha256(handle_signature.encode()).hexdigest(),
        "candidate_carbon_count": len(carbon_indices),
        "maximum_reactive_center_to_carbon_distance_bonds": maximum_distance,
        "longest_carbon_only_segment_atoms": _longest_carbon_segment_atoms(molecule),
        "carbon_carbon_double_bond_count": double_count,
        "carbon_carbon_triple_bond_count": triple_count,
        "constitutional_unsaturation_positions_json": json.dumps(
            unsaturations, separators=(",", ":"), sort_keys=True
        ),
        "unsaturation_position_zone_signature": unsaturation_zones,
        "carbon_branch_point_count": len(branch_distances),
        "branch_point_distances_from_reactive_center_json": json.dumps(
            branch_distances, separators=(",", ":")
        ),
        "branch_position_zone_signature": branch_zones,
    }


def _band(value: int, width: int) -> str:
    lower = (value // width) * width
    return f"{lower}-{lower + width - 1}"


def _bucket_signature(row: dict[str, Any]) -> str:
    payload = {
        "branch_point_count": row["carbon_branch_point_count"],
        "branch_position_zones": row["branch_position_zone_signature"],
        "carbon_count_band": row["carbon_count_band"],
        "program_family": row["program_family"],
        "projected_leaf_currency": row["projected_leaf_currency"],
        "reactive_handle_radius2_sha256": row["reactive_handle_radius2_sha256"],
        "role": row["role"],
        "unsaturation_double_count": row["carbon_carbon_double_bond_count"],
        "unsaturation_position_zones": row["unsaturation_position_zone_signature"],
        "unsaturation_triple_count": row["carbon_carbon_triple_bond_count"],
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def classify_applicability_bucket(
    *, metadata_complete: bool, bucket_component_count: int, recurrence_threshold: int
) -> str:
    """Classify recurrence descriptively without qualifying interpolation."""

    if not metadata_complete:
        return MISSING_METADATA
    if bucket_component_count >= recurrence_threshold:
        return RECURRENT_BUCKET
    return EXTRAPOLATION_REVIEW


def _validate_inputs(config: dict[str, Any], repo: Path) -> dict[str, Path]:
    specifications = config.get("inputs")
    required = {
        "graded_evidence_result",
        "graded_evidence_ledger",
        "census_source",
        "census_runner",
        "census_tests",
    }
    if not isinstance(specifications, dict) or set(specifications) != required:
        raise Ugi3FamilyApplicabilityCensusError("applicability input set changed")
    paths: dict[str, Path] = {}
    for label, specification in specifications.items():
        if not isinstance(specification, dict) or set(specification) != {"path", "sha256"}:
            raise Ugi3FamilyApplicabilityCensusError(f"input {label} is malformed")
        path = (repo / specification["path"]).resolve()
        if sha256_file(path) != specification["sha256"]:
            raise Ugi3FamilyApplicabilityCensusError(f"input hash changed: {label}")
        paths[label] = path
    return paths


def _validate_policy(config: dict[str, Any]) -> tuple[int, int]:
    policy = config.get("census_policy")
    required_true = (
        "constitutional_axes_only",
        "reactive_handle_queries_reused_from_frozen_programs",
        "recurrence_is_descriptive_not_scope_qualification",
        "projected_leaf_currency_is_a_separate_axis",
    )
    required_false = (
        "family_scope_qualification_performed",
        "recurrent_bucket_is_validated_interpolation",
        "sparse_bucket_is_chemical_incompatibility",
        "scalar_synthesis_value_defined",
        "synthesis_guidance_run",
        "prospective_candidate_selection_changed",
        "holdout_reveal_authorized",
    )
    if not isinstance(policy, dict) or any(policy.get(key) is not True for key in required_true):
        raise Ugi3FamilyApplicabilityCensusError("positive applicability safeguards changed")
    if any(policy.get(key) is not False for key in required_false):
        raise Ugi3FamilyApplicabilityCensusError("nonqualification safeguards changed")
    width = policy.get("carbon_count_band_width")
    threshold = policy.get("minimum_unique_components_for_recurrent_bucket")
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or isinstance(threshold, bool)
        or not isinstance(threshold, int)
        or threshold < 2
    ):
        raise Ugi3FamilyApplicabilityCensusError("bucket parameters are malformed")
    return width, threshold


def _nested_counts(
    rows: list[dict[str, Any]], field: str, *, occurrence_weighted: bool = False
) -> dict[str, dict[str, int]]:
    output: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        weight = int(row["one_gap_product_count"]) if occurrence_weighted else 1
        output[str(row["program_family"])][str(row[field])] += weight
    return {family: dict(sorted(counter.items())) for family, counter in sorted(output.items())}


def build_family_applicability_census(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], bytes]:
    """Build the deterministic nonselecting census for all 209 projections."""

    config = _load_json(config_path, label="family applicability config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3FamilyApplicabilityCensusError("unsupported applicability config")
    band_width, recurrence_threshold = _validate_policy(config)
    paths = _validate_inputs(config, repo)
    graded_result = _load_json(paths["graded_evidence_result"], label="graded result")
    graded_rows = _read_graded_rows(paths["graded_evidence_ledger"])

    artifact = graded_result.get("artifacts", {}).get("graded_family_evidence_ledger.csv.gz", {})
    adjudication = graded_result.get("adjudication", {})
    if (
        graded_result.get("status") != "complete_nonselecting_graded_family_evidence_audit"
        or artifact.get("sha256") != sha256_file(paths["graded_evidence_ledger"])
        or adjudication.get("family_scope_qualification_performed") is not False
        or adjudication.get("synthesis_guidance_run") is not False
        or adjudication.get("prospective_candidate_selection_changed") is not False
        or adjudication.get("holdout_revealed") is not False
    ):
        raise Ugi3FamilyApplicabilityCensusError("graded evidence ownership failed")
    if len(graded_rows) != int(artifact.get("rows", -1)):
        raise Ugi3FamilyApplicabilityCensusError("graded evidence denominator changed")

    projected = [row for row in graded_rows if row["program_family"]]
    expected_projected = sum(
        int(
            graded_result.get("summary", {})
            .get("components_by_graded_evidence_class", {})
            .get(label, -1)
        )
        for label in (
            "family_projected_all_current_leaves",
            "family_projected_partial_current_leaves",
            "family_projected_no_current_leaves",
        )
    )
    if len(projected) != expected_projected:
        raise Ugi3FamilyApplicabilityCensusError("projected component denominator changed")

    preliminary: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for source in projected:
        key = (source["role"], source["canonical_smiles"])
        if key in seen:
            raise Ugi3FamilyApplicabilityCensusError("duplicate projected component identity")
        seen.add(key)
        program_family = source["program_family"]
        if _EXPECTED_FAMILY_ROLES.get(program_family) != source["role"]:
            raise Ugi3FamilyApplicabilityCensusError("program family and role are inconsistent")
        if (
            source["schema_version"] != "phase1_ugi3_graded_family_evidence_ledger.v1"
            or source["exact_assessment_outcome"] != "missing_knowledge"
            or source["in_final_one_gap_triage"] != "True"
            or source["family_scope_status"] != "not_qualified_by_this_audit"
            or source["exact_scope_still_required"] != "True"
        ):
            raise Ugi3FamilyApplicabilityCensusError(
                "projected row lost its nonqualified evidence semantics"
            )
        count = _parse_positive_int(source["one_gap_product_count"], label="one-gap product count")
        projected_leaf_count = _parse_positive_int(
            source["projected_leaf_count"], label="projected leaf count"
        )
        try:
            current_leaf_count = int(source["current_projected_leaf_count"])
        except ValueError as exc:
            raise Ugi3FamilyApplicabilityCensusError(
                "current projected leaf count must be an integer"
            ) from exc
        if not 0 <= current_leaf_count <= projected_leaf_count:
            raise Ugi3FamilyApplicabilityCensusError("projected leaf currency is malformed")
        leaf_currency = source["projection_leaf_status"]
        expected_currency = (
            "all_current"
            if current_leaf_count == projected_leaf_count
            else "partial_current" if current_leaf_count else "no_current"
        )
        if leaf_currency != expected_currency:
            raise Ugi3FamilyApplicabilityCensusError("projected leaf status changed")

        axes = constitutional_applicability_axes(*key)
        row = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "role": key[0],
            "canonical_smiles": key[1],
            "program_family": program_family,
            "one_gap_product_count": count,
            "projected_leaf_currency": leaf_currency,
            "projected_leaf_count": projected_leaf_count,
            "current_projected_leaf_count": current_leaf_count,
            **axes,
            "carbon_count_band": _band(axes["candidate_carbon_count"], band_width),
            "axis_metadata_status": "complete",
            "family_scope_qualified": False,
        }
        signature = _bucket_signature(row)
        row["applicability_bucket_signature_json"] = signature
        row["applicability_bucket_id"] = hashlib.sha256(signature.encode()).hexdigest()[:20]
        preliminary.append(row)

    bucket_components: Counter[str] = Counter(row["applicability_bucket_id"] for row in preliminary)
    bucket_occurrences: Counter[str] = Counter()
    bucket_signatures: dict[str, str] = {}
    for row in preliminary:
        bucket = row["applicability_bucket_id"]
        bucket_occurrences[bucket] += int(row["one_gap_product_count"])
        prior = bucket_signatures.setdefault(
            bucket, str(row["applicability_bucket_signature_json"])
        )
        if prior != row["applicability_bucket_signature_json"]:
            raise Ugi3FamilyApplicabilityCensusError("applicability bucket hash collision")

    rows = []
    for row in preliminary:
        bucket = row["applicability_bucket_id"]
        row["bucket_unique_component_count"] = bucket_components[bucket]
        row["bucket_one_gap_product_occurrence_count"] = bucket_occurrences[bucket]
        row["applicability_census_class"] = classify_applicability_bucket(
            metadata_complete=row["axis_metadata_status"] == "complete",
            bucket_component_count=bucket_components[bucket],
            recurrence_threshold=recurrence_threshold,
        )
        rows.append(row)
    rows.sort(key=lambda row: (str(row["program_family"]), str(row["canonical_smiles"])))
    ledger = _gzip_csv_bytes(rows)

    unique_buckets = sorted(bucket_components)
    recurring_buckets = [
        bucket for bucket in unique_buckets if bucket_components[bucket] >= recurrence_threshold
    ]
    sparse_buckets = [
        bucket for bucket in unique_buckets if bucket_components[bucket] < recurrence_threshold
    ]
    class_counts = Counter(row["applicability_census_class"] for row in rows)
    class_occurrences = Counter()
    for row in rows:
        class_occurrences[row["applicability_census_class"]] += int(row["one_gap_product_count"])
    recurrent_summaries = [
        {
            "applicability_bucket_id": bucket,
            "applicability_bucket_signature": json.loads(bucket_signatures[bucket]),
            "one_gap_product_occurrences": bucket_occurrences[bucket],
            "unique_components": bucket_components[bucket],
        }
        for bucket in recurring_buckets
    ]
    recurrent_summaries.sort(
        key=lambda row: (
            -int(row["unique_components"]),
            -int(row["one_gap_product_occurrences"]),
            str(row["applicability_bucket_id"]),
        )
    )

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_nonselecting_family_applicability_axis_census",
        "task": config.get("task"),
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "inputs": {
            label: {
                "path": str(path.relative_to(repo)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for label, path in sorted(paths.items())
        },
        "axis_definitions": {
            "chain_length": (
                "Exact carbon count, longest carbon-only graph segment and maximum graph-bond "
                "distance from the reactive center; bucket recurrence uses carbon-count bands."
            ),
            "unsaturation": (
                "Constitutional carbon-carbon double/triple counts and graph distances from the "
                "reactive center; alkene stereochemistry is not inferred."
            ),
            "branching": (
                "Carbon atoms with at least three carbon neighbors and their exact graph "
                "distances from the reactive center."
            ),
            "reactive_handle_neighborhood": (
                "Deterministic radius-two atom/bond environment rooted at the frozen aldehyde "
                "or isocyanide reactive center."
            ),
            "relative_position_zones": (
                "Proximal, middle and distal thirds of maximum reactive-center graph distance; "
                "exact distances remain in the ledger."
            ),
        },
        "summary": {
            "family_projected_components": len(rows),
            "one_gap_product_occurrences": sum(int(row["one_gap_product_count"]) for row in rows),
            "components_by_program_family": dict(
                sorted(Counter(row["program_family"] for row in rows).items())
            ),
            "components_by_role": dict(sorted(Counter(row["role"] for row in rows).items())),
            "components_by_projected_leaf_currency": dict(
                sorted(Counter(row["projected_leaf_currency"] for row in rows).items())
            ),
            "components_by_program_family_and_projected_leaf_currency": _nested_counts(
                rows, "projected_leaf_currency"
            ),
            "components_by_applicability_census_class": dict(sorted(class_counts.items())),
            "components_by_program_family_and_applicability_census_class": _nested_counts(
                rows, "applicability_census_class"
            ),
            "one_gap_product_occurrences_by_applicability_census_class": dict(
                sorted(class_occurrences.items())
            ),
            "one_gap_product_occurrences_by_program_family_and_applicability_census_class": (
                _nested_counts(rows, "applicability_census_class", occurrence_weighted=True)
            ),
            "unique_applicability_buckets": len(unique_buckets),
            "recurrent_interpolation_bucket_candidates": len(recurring_buckets),
            "sparse_extrapolation_review_buckets": len(sparse_buckets),
            "missing_axis_metadata_components": class_counts[MISSING_METADATA],
            "carbon_count_distribution_by_program_family": _nested_counts(
                rows, "candidate_carbon_count"
            ),
            "longest_carbon_segment_distribution_by_program_family": _nested_counts(
                rows, "longest_carbon_only_segment_atoms"
            ),
            "unsaturation_count_distribution_by_program_family": {
                family: dict(sorted(counter.items()))
                for family, counter in sorted(
                    {
                        family: Counter(
                            str(
                                int(row["carbon_carbon_double_bond_count"])
                                + int(row["carbon_carbon_triple_bond_count"])
                            )
                            for row in rows
                            if row["program_family"] == family
                        )
                        for family in {str(row["program_family"]) for row in rows}
                    }.items()
                )
            },
            "branch_point_count_distribution_by_program_family": _nested_counts(
                rows, "carbon_branch_point_count"
            ),
            "unsaturation_position_zone_distribution_by_program_family": _nested_counts(
                rows, "unsaturation_position_zone_signature"
            ),
            "branch_position_zone_distribution_by_program_family": _nested_counts(
                rows, "branch_position_zone_signature"
            ),
            "reactive_handle_neighborhood_distribution_by_program_family": _nested_counts(
                rows, "reactive_handle_radius2_sha256"
            ),
            "recurrent_bucket_summaries": recurrent_summaries,
        },
        "adjudication": {
            "family_scope_qualification_performed": False,
            "recurrent_bucket_is_validated_interpolation": False,
            "sparse_bucket_is_chemical_incompatibility": False,
            "scalar_synthesis_value_defined": False,
            "synthesis_success_probability_defined": False,
            "synthesis_guidance_run": False,
            "prospective_candidate_selection_changed": False,
            "holdout_revealed": False,
        },
        "nonclaims": [
            "A recurrent constitutional bucket is only an interpolation candidate.",
            "A sparse bucket requires extrapolation review but is not chemically incompatible.",
            "Constitutional alkene positions do not encode unmodeled stereochemistry.",
            "Reactive-handle compatibility does not establish bounded reaction-family scope.",
            "Projected leaf currency does not change the exact missing-knowledge outcome.",
            "This census does not define a scalar, run guidance, select candidates or reveal holdout.",
        ],
        "artifacts": {
            "family_applicability_census.csv.gz": {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "rows": len(rows),
                "sha256": sha256_bytes(ledger),
            }
        },
    }
    return result, ledger
