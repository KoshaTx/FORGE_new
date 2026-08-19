"""Nonselecting chemotype-diversity audit for Ugi tail-bearing components.

The audit distinguishes exact component diversity from a coarse, graph-derived
chemotype signature. It does not infer synthesis feasibility or biological
activity from structural diversity.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.data.r1_prime_audit import sha256_file
from forge.product.ugi_postselection_branching import carbon_branch_metrics

CONFIG_SCHEMA_VERSION = "phase1_ugi_tail_chemotype_audit_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_tail_chemotype_audit.v1"


class UgiTailChemotypeAuditError(ValueError):
    """Raised when a chemotype-audit input violates its frozen contract."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise UgiTailChemotypeAuditError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise UgiTailChemotypeAuditError(f"{label} must be a JSON object")
    return value


def _molecule(smiles: Any, *, label: str) -> Chem.Mol:
    if not isinstance(smiles, str) or not smiles:
        raise UgiTailChemotypeAuditError(f"{label} must be non-empty SMILES")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise UgiTailChemotypeAuditError(f"{label} must be one valid connected graph")
    return molecule


def _carbon_subgraph_diameter(molecule: Chem.Mol) -> int:
    carbon_indices = {atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 6}
    if not carbon_indices:
        return 0
    adjacency = {
        index: {
            neighbor.GetIdx()
            for neighbor in molecule.GetAtomWithIdx(index).GetNeighbors()
            if neighbor.GetIdx() in carbon_indices
        }
        for index in carbon_indices
    }
    diameter = 0
    for start in carbon_indices:
        distances = {start: 0}
        queue = [start]
        for current in queue:
            for neighbor in adjacency[current]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        diameter = max(diameter, max(distances.values()))
    return diameter


def component_chemotype_metrics(smiles: str) -> dict[str, int]:
    """Return graph-derived tail descriptors without a fragment vocabulary."""

    molecule = _molecule(smiles, label="component")
    element_counts = Counter(atom.GetSymbol() for atom in molecule.GetAtoms())
    carbon_double_bonds = 0
    carbon_triple_bonds = 0
    for bond in molecule.GetBonds():
        left = bond.GetBeginAtom()
        right = bond.GetEndAtom()
        if left.GetAtomicNum() != 6 or right.GetAtomicNum() != 6 or bond.GetIsAromatic():
            continue
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            carbon_double_bonds += 1
        elif bond.GetBondType() == Chem.BondType.TRIPLE:
            carbon_triple_bonds += 1

    carbonyl_count = 0
    ester_like_count = 0
    amide_like_count = 0
    carbonyl_adjacent_oxygens: set[int] = set()
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        bonds = [
            molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            for neighbor in atom.GetNeighbors()
        ]
        if not any(
            neighbor.GetAtomicNum() == 8 and bond.GetBondType() == Chem.BondType.DOUBLE
            for neighbor, bond in zip(atom.GetNeighbors(), bonds, strict=True)
        ):
            continue
        carbonyl_count += 1
        for neighbor, bond in zip(atom.GetNeighbors(), bonds, strict=True):
            if bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            if neighbor.GetAtomicNum() == 8:
                ester_like_count += 1
                carbonyl_adjacent_oxygens.add(neighbor.GetIdx())
            elif neighbor.GetAtomicNum() == 7:
                amide_like_count += 1

    ether_oxygen_count = 0
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != 8 or atom.GetIdx() in carbonyl_adjacent_oxygens:
            continue
        if atom.GetDegree() == 2 and all(
            neighbor.GetAtomicNum() == 6 for neighbor in atom.GetNeighbors()
        ):
            if all(
                molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType()
                == Chem.BondType.SINGLE
                for neighbor in atom.GetNeighbors()
            ):
                ether_oxygen_count += 1

    branch = carbon_branch_metrics(smiles)
    return {
        "heavy_atoms": molecule.GetNumHeavyAtoms(),
        "carbon_atoms": element_counts["C"],
        "nitrogen_atoms": element_counts["N"],
        "oxygen_atoms": element_counts["O"],
        "sulfur_atoms": element_counts["S"],
        "phosphorus_atoms": element_counts["P"],
        "halogen_atoms": sum(element_counts[symbol] for symbol in ("F", "Cl", "Br", "I")),
        "carbon_carbon_double_bonds": carbon_double_bonds,
        "carbon_carbon_triple_bonds": carbon_triple_bonds,
        "carbon_branch_atoms": int(branch["carbon_branch_atoms"] or 0),
        "adjacent_carbon_branch_edges": int(branch["adjacent_carbon_branch_edges"] or 0),
        "ring_count": molecule.GetRingInfo().NumRings(),
        "carbonyl_count": carbonyl_count,
        "ester_like_carbonyl_count": ester_like_count,
        "amide_like_carbonyl_count": amide_like_count,
        "ether_oxygen_count": ether_oxygen_count,
        "carbon_subgraph_diameter": _carbon_subgraph_diameter(molecule),
    }


SIGNATURE_FIELDS = (
    "carbon_atoms",
    "nitrogen_atoms",
    "oxygen_atoms",
    "sulfur_atoms",
    "phosphorus_atoms",
    "halogen_atoms",
    "carbon_carbon_double_bonds",
    "carbon_carbon_triple_bonds",
    "carbon_branch_atoms",
    "adjacent_carbon_branch_edges",
    "ring_count",
    "carbonyl_count",
    "ester_like_carbonyl_count",
    "amide_like_carbonyl_count",
    "ether_oxygen_count",
    "carbon_subgraph_diameter",
)

ARCHITECTURE_FIELDS = tuple(
    field for field in SIGNATURE_FIELDS if field not in {"carbon_atoms", "carbon_subgraph_diameter"}
)


def chemotype_signature(metrics: Mapping[str, int]) -> str:
    """Serialize the declared coarse chemotype signature deterministically."""

    return "|".join(f"{field}={int(metrics[field])}" for field in SIGNATURE_FIELDS)


def architecture_signature(metrics: Mapping[str, int]) -> str:
    """Serialize a length-independent functional architecture signature."""

    return "|".join(f"{field}={int(metrics[field])}" for field in ARCHITECTURE_FIELDS)


def _effective_count(counter: Mapping[str, int]) -> float:
    total = sum(counter.values())
    if total <= 0:
        raise UgiTailChemotypeAuditError("effective count requires a nonempty cohort")
    entropy = -sum((count / total) * math.log(count / total) for count in counter.values())
    return math.exp(entropy)


def _integer_distribution(values: Iterable[int]) -> dict[str, int]:
    return {str(key): count for key, count in sorted(Counter(values).items())}


def summarize_component_cohort(smiles_values: Sequence[str]) -> dict[str, Any]:
    """Summarize occurrence-weighted and exact-unique component diversity."""

    if not smiles_values:
        raise UgiTailChemotypeAuditError("cannot summarize an empty component cohort")
    exact = Counter(smiles_values)
    metrics_by_smiles = {smiles: component_chemotype_metrics(smiles) for smiles in exact}
    occurrence_signatures = Counter(
        {
            signature: sum(
                exact[smiles]
                for smiles, metrics in metrics_by_smiles.items()
                if chemotype_signature(metrics) == signature
            )
            for signature in {
                chemotype_signature(metrics) for metrics in metrics_by_smiles.values()
            }
        }
    )
    unique_signatures = Counter(
        chemotype_signature(metrics) for metrics in metrics_by_smiles.values()
    )
    occurrence_architectures = Counter(
        {
            signature: sum(
                exact[smiles]
                for smiles, metrics in metrics_by_smiles.items()
                if architecture_signature(metrics) == signature
            )
            for signature in {
                architecture_signature(metrics) for metrics in metrics_by_smiles.values()
            }
        }
    )
    unique_architectures = Counter(
        architecture_signature(metrics) for metrics in metrics_by_smiles.values()
    )
    occurrence_metrics = [metrics_by_smiles[smiles] for smiles in smiles_values]
    feature_tests = {
        "has_carbon_carbon_double_bond": lambda metrics: metrics["carbon_carbon_double_bonds"] > 0,
        "has_carbon_carbon_triple_bond": lambda metrics: metrics["carbon_carbon_triple_bonds"] > 0,
        "has_carbon_branch": lambda metrics: metrics["carbon_branch_atoms"] > 0,
        "has_adjacent_carbon_branches": lambda metrics: metrics["adjacent_carbon_branch_edges"] > 0,
        "has_ring": lambda metrics: metrics["ring_count"] > 0,
        "has_ester_like_carbonyl": lambda metrics: metrics["ester_like_carbonyl_count"] > 0,
        "has_ether_oxygen": lambda metrics: metrics["ether_oxygen_count"] > 0,
    }
    return {
        "component_occurrences": len(smiles_values),
        "unique_exact_components": len(exact),
        "effective_exact_component_count": _effective_count(exact),
        "unique_chemotype_signatures": len(unique_signatures),
        "effective_occurrence_weighted_chemotype_count": _effective_count(occurrence_signatures),
        "effective_unique_catalog_chemotype_count": _effective_count(unique_signatures),
        "unique_architecture_signatures": len(unique_architectures),
        "effective_occurrence_weighted_architecture_count": _effective_count(
            occurrence_architectures
        ),
        "effective_unique_catalog_architecture_count": _effective_count(unique_architectures),
        "signature_fields": list(SIGNATURE_FIELDS),
        "architecture_fields": list(ARCHITECTURE_FIELDS),
        "descriptor_distributions_unique_components": {
            field: _integer_distribution(metrics[field] for metrics in metrics_by_smiles.values())
            for field in SIGNATURE_FIELDS
        },
        "descriptor_distributions_occurrences": {
            field: _integer_distribution(metrics[field] for metrics in occurrence_metrics)
            for field in SIGNATURE_FIELDS
        },
        "feature_occurrence_fractions": {
            name: sum(test(metrics) for metrics in occurrence_metrics) / len(occurrence_metrics)
            for name, test in feature_tests.items()
        },
        "exact_component_counts": dict(sorted(exact.items())),
        "chemotype_signature_occurrence_counts": dict(sorted(occurrence_signatures.items())),
        "chemotype_signature_unique_component_counts": dict(sorted(unique_signatures.items())),
        "architecture_signature_occurrence_counts": dict(sorted(occurrence_architectures.items())),
        "architecture_signature_unique_component_counts": dict(
            sorted(unique_architectures.items())
        ),
    }


def _signature_counter(summary: Mapping[str, Any], *, view: str) -> Counter[str]:
    key = {
        "occurrence": "chemotype_signature_occurrence_counts",
        "unique": "chemotype_signature_unique_component_counts",
    }.get(view)
    if key is None:
        raise UgiTailChemotypeAuditError(f"unsupported signature view: {view}")
    return Counter({str(k): int(v) for k, v in summary[key].items()})


def _architecture_counter(summary: Mapping[str, Any], *, view: str) -> Counter[str]:
    key = {
        "occurrence": "architecture_signature_occurrence_counts",
        "unique": "architecture_signature_unique_component_counts",
    }.get(view)
    if key is None:
        raise UgiTailChemotypeAuditError(f"unsupported architecture view: {view}")
    return Counter({str(k): int(v) for k, v in summary[key].items()})


def _jensen_shannon_divergence(left: Mapping[str, int], right: Mapping[str, int]) -> float:
    left_total = sum(left.values())
    right_total = sum(right.values())
    if left_total <= 0 or right_total <= 0:
        raise UgiTailChemotypeAuditError("Jensen-Shannon divergence requires nonempty cohorts")
    keys = set(left) | set(right)
    value = 0.0
    for key in keys:
        p = left.get(key, 0) / left_total
        q = right.get(key, 0) / right_total
        midpoint = 0.5 * (p + q)
        if p > 0:
            value += 0.5 * p * math.log(p / midpoint, 2)
        if q > 0:
            value += 0.5 * q * math.log(q / midpoint, 2)
    return value


def compare_component_cohorts(
    generated: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, float | int]:
    """Compare generated occurrences against a reference component catalog."""

    generated_exact = Counter(
        {str(key): int(value) for key, value in generated["exact_component_counts"].items()}
    )
    reference_exact = set(reference["exact_component_counts"])
    generated_signatures = _signature_counter(generated, view="occurrence")
    reference_signatures = set(_signature_counter(reference, view="unique"))
    generated_total = sum(generated_exact.values())
    generated_signature_total = sum(generated_signatures.values())
    generated_unique_signatures = set(_signature_counter(generated, view="unique"))
    generated_architectures = _architecture_counter(generated, view="occurrence")
    generated_unique_architectures = set(_architecture_counter(generated, view="unique"))
    reference_unique_architectures = set(_architecture_counter(reference, view="unique"))
    return {
        "generated_occurrence_exact_identity_coverage_fraction": sum(
            count for smiles, count in generated_exact.items() if smiles in reference_exact
        )
        / generated_total,
        "generated_occurrence_chemotype_coverage_fraction": sum(
            count
            for signature, count in generated_signatures.items()
            if signature in reference_signatures
        )
        / generated_signature_total,
        "reference_unique_chemotype_recall_fraction": len(
            reference_signatures & generated_unique_signatures
        )
        / len(reference_signatures),
        "reference_unique_chemotypes": len(reference_signatures),
        "generated_unique_chemotypes": len(generated_unique_signatures),
        "unique_catalog_signature_jensen_shannon_divergence_bits": _jensen_shannon_divergence(
            _signature_counter(generated, view="unique"),
            _signature_counter(reference, view="unique"),
        ),
        "generated_occurrence_architecture_coverage_fraction": sum(
            count
            for signature, count in generated_architectures.items()
            if signature in reference_unique_architectures
        )
        / sum(generated_architectures.values()),
        "reference_unique_architecture_recall_fraction": len(
            reference_unique_architectures & generated_unique_architectures
        )
        / len(reference_unique_architectures),
        "reference_unique_architectures": len(reference_unique_architectures),
        "generated_unique_architectures": len(generated_unique_architectures),
        "unique_catalog_architecture_jensen_shannon_divergence_bits": _jensen_shannon_divergence(
            _architecture_counter(generated, view="unique"),
            _architecture_counter(reference, view="unique"),
        ),
    }


def _empty_roles(roles: Sequence[str]) -> dict[str, list[str]]:
    return {role: [] for role in roles}


def _read_generated(path: Path, roles: Sequence[str]) -> dict[str, list[str]]:
    selected = _load_json(path, label="selected sample")
    output = _empty_roles(roles)
    for row in selected.get("samples", []):
        if not row.get("component_reconstruction_valid"):
            continue
        components = row.get("component_smiles_by_role")
        if not isinstance(components, dict):
            raise UgiTailChemotypeAuditError("generated sample has no component mapping")
        for role in roles:
            output[role].append(str(components[role]))
    expected = int(selected["statistics"]["valid_molecules"])
    if any(len(values) != expected for values in output.values()):
        raise UgiTailChemotypeAuditError("generated component denominator mismatch")
    return output


def _read_original_ledger(path: Path, roles: Sequence[str]) -> dict[str, list[str]]:
    output = _empty_roles(roles)
    try:
        with gzip.open(path, "rt", newline="") as handle:
            for row in csv.DictReader(handle):
                routes = json.loads(row["candidate_routes_json"])
                if len(routes) != 1:
                    raise UgiTailChemotypeAuditError("original ledger row lacks one exact route")
                components = routes[0]["components"]
                for role in roles:
                    output[role].append(str(components[role]))
    except (OSError, csv.Error, KeyError, json.JSONDecodeError) as exc:
        raise UgiTailChemotypeAuditError("invalid original Ugi ledger") from exc
    return output


def _read_assignments(
    path: Path, roles: Sequence[str], folds: Sequence[str]
) -> dict[str, list[str]]:
    output = _empty_roles(roles)
    try:
        with gzip.open(path, "rt", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["primary_product_fold"] not in folds:
                    continue
                for role in roles:
                    output[role].append(row[f"{role}_smiles"])
    except (OSError, csv.Error, KeyError) as exc:
        raise UgiTailChemotypeAuditError("invalid selection assignments") from exc
    return output


def _read_registry(
    path: Path,
    roles: Sequence[str],
    folds: Sequence[str],
    transfer_source_class: str,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    admitted = _empty_roles(roles)
    transferred = _empty_roles(roles)
    try:
        with gzip.open(path, "rt", newline="") as handle:
            for row in csv.DictReader(handle):
                role = row["role"]
                if role not in admitted or row["family_fold"] not in folds:
                    continue
                if row["l1_structural_admission"].lower() != "true":
                    continue
                smiles = row["canonical_smiles"]
                admitted[role].append(smiles)
                source_classes = json.loads(row["source_classes_json"])
                if transfer_source_class in source_classes:
                    transferred[role].append(smiles)
    except (OSError, csv.Error, KeyError, json.JSONDecodeError) as exc:
        raise UgiTailChemotypeAuditError("invalid component registry") from exc
    return admitted, transferred


def build_tail_chemotype_audit(
    config_path: Path,
    production_manifest_path: Path,
    selected_sample_path: Path,
    original_ledger_path: Path,
    selection_assignments_path: Path,
    component_registry_path: Path,
) -> dict[str, Any]:
    """Build the hash-pinned nonselecting tail chemotype audit."""

    config = _load_json(config_path, label="tail chemotype audit config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiTailChemotypeAuditError("unsupported tail chemotype audit config schema")
    inputs = {
        "production_manifest": production_manifest_path,
        "selected_sample": selected_sample_path,
        "original_ugi_product_ledger": original_ledger_path,
        "selection_reference_assignments": selection_assignments_path,
        "component_registry": component_registry_path,
    }
    for name, path in inputs.items():
        expected = config["inputs"][name]["sha256"]
        observed = sha256_file(path)
        if observed != expected:
            raise UgiTailChemotypeAuditError(
                f"{name} hash mismatch: expected {expected}, observed {observed}"
            )

    roles = tuple(config["tail_roles"])
    folds = tuple(config["selection_visible_folds"])
    generated = _read_generated(selected_sample_path, roles)
    original = _read_original_ledger(original_ledger_path, roles)
    selection = _read_assignments(selection_assignments_path, roles, folds)
    expanded, transferred = _read_registry(
        component_registry_path,
        roles,
        folds,
        str(config["transfer_source_class"]),
    )
    raw_cohorts = {
        "generated_selected_sample": generated,
        "original_12276_product_occurrences_descriptive_only": original,
        "selection_visible_training_and_calibration_occurrences": selection,
        "selection_visible_expanded_component_catalog": expanded,
        "selection_visible_cross_platform_transfer_components": transferred,
    }
    cohorts: dict[str, dict[str, Any]] = {}
    for cohort_name, role_values in raw_cohorts.items():
        cohorts[cohort_name] = {}
        for role in roles:
            if role_values[role]:
                cohorts[cohort_name][role] = summarize_component_cohort(role_values[role])

    comparisons: dict[str, dict[str, Any]] = {}
    for reference_name in (
        "original_12276_product_occurrences_descriptive_only",
        "selection_visible_training_and_calibration_occurrences",
        "selection_visible_expanded_component_catalog",
        "selection_visible_cross_platform_transfer_components",
    ):
        comparisons[reference_name] = {}
        for role in roles:
            reference = cohorts[reference_name].get(role)
            if reference is not None:
                comparisons[reference_name][role] = compare_component_cohorts(
                    cohorts["generated_selected_sample"][role], reference
                )

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_nonselecting_tail_chemotype_audit",
        "task": "ugi_tail_chemotype_diversity",
        "scope": {
            "checkpoint_selection": False,
            "heldout_component_families_inspected": False,
            "synthesis_feasibility_assessed": False,
            "biological_activity_assessed": False,
            "theoretical_interpolative_space_counted": False,
        },
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)} for name, path in inputs.items()
        },
        "production_generator": _load_json(
            production_manifest_path, label="production manifest"
        ).get("selected_model"),
        "cohorts": cohorts,
        "comparisons_to_generated": comparisons,
        "safe_claim": (
            "The audit separates exact generated component novelty from coverage of graph-derived "
            "tail chemotypes in the observed and selection-visible Ugi support. It does not convert "
            "a large bounded interpolative design space into measured, route-closed or biologically "
            "validated evidence."
        ),
        "limitations": [
            "Chemotype signatures intentionally collapse positional and stereochemical detail.",
            "The full 12,276-product ledger is descriptive only and is not used for checkpoint selection.",
            "Heldout component families are excluded from all selection-visible reference cohorts.",
            "Structural similarity and diversity do not establish potency or synthesis success.",
        ],
    }
