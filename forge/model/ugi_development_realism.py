"""Role-aware development realism for exact Ugi products.

The generic whole-lipid assay is intentionally method-blind, but its small global descriptor vector
cannot resolve many head/tail changes.  This module adds a development-only view over exact Ugi
precursor roles.  Its reference is the group-balanced measured-train split constructed by
``common_lipid_realism``; calibration and held-out product structures are never consumed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

from forge.model.common_lipid_realism import (
    UGI_PRECURSOR_ROLES,
    RealismPolicy,
    ReferenceMolecule,
    UgiDevelopmentReference,
    fit_robust_descriptor_scale,
    grouped_classifier_two_sample,
)

ROLE_DESCRIPTOR_NAMES = (
    "amine_heavy_atoms",
    "amine_carbon_atoms",
    "amine_nitrogen_atoms",
    "amine_oxygen_atoms",
    "amine_hbd",
    "amine_hba",
    "amine_ring_count",
    "amine_largest_ring_size",
    "amine_branch_atoms",
    "amine_carbon_skeleton_diameter",
    "aldehyde_heavy_atoms",
    "aldehyde_carbon_atoms",
    "aldehyde_heteroatoms",
    "aldehyde_hbd",
    "aldehyde_hba",
    "aldehyde_ring_count",
    "aldehyde_largest_ring_size",
    "aldehyde_oxygen_ring_count",
    "aldehyde_branch_atoms",
    "aldehyde_unsaturated_bonds",
    "aldehyde_carbon_skeleton_diameter",
    "isocyanide_heavy_atoms",
    "isocyanide_carbon_atoms",
    "isocyanide_heteroatoms",
    "isocyanide_hbd",
    "isocyanide_hba",
    "isocyanide_ring_count",
    "isocyanide_largest_ring_size",
    "isocyanide_oxygen_ring_count",
    "isocyanide_branch_atoms",
    "isocyanide_unsaturated_bonds",
    "isocyanide_carbon_skeleton_diameter",
    "tail_total_carbon_atoms",
    "tail_carbon_asymmetry",
    "tail_skeleton_diameter_asymmetry",
    "total_component_heavy_atoms",
)

SANITY_SHIFT_FEATURES = (
    "amine_nitrogen_atoms",
    "aldehyde_carbon_atoms",
    "isocyanide_carbon_atoms",
)


class UgiDevelopmentRealismError(ValueError):
    """A role-aware realism input violates the development contract."""


def _parse_component(smiles: str, *, label: str) -> Chem.Mol:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0 or len(Chem.GetMolFrags(molecule)) != 1:
        raise UgiDevelopmentRealismError(f"{label} is not one valid component")
    return molecule


def _carbon_skeleton_diameter(molecule: Chem.Mol) -> int:
    carbon = {atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 6}
    if not carbon:
        return 0
    maximum_edges = 0
    for start in carbon:
        distances = {start: 0}
        queue = [start]
        for current in queue:
            for neighbor in molecule.GetAtomWithIdx(current).GetNeighbors():
                index = neighbor.GetIdx()
                if index in carbon and index not in distances:
                    distances[index] = distances[current] + 1
                    queue.append(index)
        maximum_edges = max(maximum_edges, max(distances.values()))
    return maximum_edges + 1


def _component_features(molecule: Chem.Mol) -> dict[str, float]:
    atoms = tuple(molecule.GetAtoms())
    rings = tuple(molecule.GetRingInfo().AtomRings())
    carbon = sum(atom.GetAtomicNum() == 6 for atom in atoms)
    heavy = molecule.GetNumHeavyAtoms()
    return {
        "heavy_atoms": float(heavy),
        "carbon_atoms": float(carbon),
        "heteroatoms": float(heavy - carbon),
        "nitrogen_atoms": float(sum(atom.GetAtomicNum() == 7 for atom in atoms)),
        "oxygen_atoms": float(sum(atom.GetAtomicNum() == 8 for atom in atoms)),
        "hbd": float(rdMolDescriptors.CalcNumHBD(molecule)),
        "hba": float(rdMolDescriptors.CalcNumHBA(molecule)),
        "ring_count": float(len(rings)),
        "largest_ring_size": float(max((len(ring) for ring in rings), default=0)),
        "oxygen_ring_count": float(
            sum(
                any(molecule.GetAtomWithIdx(index).GetAtomicNum() == 8 for index in ring)
                for ring in rings
            )
        ),
        "branch_atoms": float(
            sum(atom.GetAtomicNum() > 1 and atom.GetDegree() >= 3 for atom in atoms)
        ),
        "unsaturated_bonds": float(
            sum(
                not bond.GetIsAromatic() and bond.GetBondTypeAsDouble() > 1.0
                for bond in molecule.GetBonds()
            )
        ),
        "carbon_skeleton_diameter": float(_carbon_skeleton_diameter(molecule)),
    }


def ugi_role_descriptor_vector(components_by_role: Mapping[str, str]) -> np.ndarray:
    """Describe exact Ugi precursor roles without component identities or fragment labels."""

    if set(components_by_role) != set(UGI_PRECURSOR_ROLES):
        raise UgiDevelopmentRealismError("exact Ugi trace roles changed")
    features = {
        role: _component_features(
            _parse_component(str(components_by_role[role]), label=f"{role} component")
        )
        for role in UGI_PRECURSOR_ROLES
    }
    amine = features["amine_head"]
    aldehyde = features["oxoester_aldehyde_body_tail"]
    isocyanide = features["isocyanide_tail"]
    values = (
        amine["heavy_atoms"],
        amine["carbon_atoms"],
        amine["nitrogen_atoms"],
        amine["oxygen_atoms"],
        amine["hbd"],
        amine["hba"],
        amine["ring_count"],
        amine["largest_ring_size"],
        amine["branch_atoms"],
        amine["carbon_skeleton_diameter"],
        aldehyde["heavy_atoms"],
        aldehyde["carbon_atoms"],
        aldehyde["heteroatoms"],
        aldehyde["hbd"],
        aldehyde["hba"],
        aldehyde["ring_count"],
        aldehyde["largest_ring_size"],
        aldehyde["oxygen_ring_count"],
        aldehyde["branch_atoms"],
        aldehyde["unsaturated_bonds"],
        aldehyde["carbon_skeleton_diameter"],
        isocyanide["heavy_atoms"],
        isocyanide["carbon_atoms"],
        isocyanide["heteroatoms"],
        isocyanide["hbd"],
        isocyanide["hba"],
        isocyanide["ring_count"],
        isocyanide["largest_ring_size"],
        isocyanide["oxygen_ring_count"],
        isocyanide["branch_atoms"],
        isocyanide["unsaturated_bonds"],
        isocyanide["carbon_skeleton_diameter"],
        aldehyde["carbon_atoms"] + isocyanide["carbon_atoms"],
        abs(aldehyde["carbon_atoms"] - isocyanide["carbon_atoms"]),
        abs(aldehyde["carbon_skeleton_diameter"] - isocyanide["carbon_skeleton_diameter"]),
        amine["heavy_atoms"] + aldehyde["heavy_atoms"] + isocyanide["heavy_atoms"],
    )
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (len(ROLE_DESCRIPTOR_NAMES),) or not np.isfinite(vector).all():
        raise UgiDevelopmentRealismError("non-finite Ugi role descriptor vector")
    return vector


def _mean_pairwise_distance(left: np.ndarray, right: np.ndarray, *, chunk_size: int) -> float:
    if len(left) == 0 or len(right) == 0:
        raise UgiDevelopmentRealismError("continuous distance received an empty matrix")
    total = 0.0
    pairs = 0
    for start in range(0, len(left), chunk_size):
        block = left[start : start + chunk_size]
        distances = np.linalg.norm(block[:, None, :] - right[None, :, :], axis=2)
        total += float(distances.sum())
        pairs += int(distances.size)
    return total / pairs


def _energy_distance(
    generated: np.ndarray,
    reference: np.ndarray,
    *,
    chunk_size: int,
) -> float:
    value = (
        2.0 * _mean_pairwise_distance(generated, reference, chunk_size=chunk_size)
        - _mean_pairwise_distance(generated, generated, chunk_size=chunk_size)
        - _mean_pairwise_distance(reference, reference, chunk_size=chunk_size)
    )
    return max(0.0, float(value))


def _reference_bandwidth(reference: np.ndarray) -> float:
    distances = np.linalg.norm(reference[:, None, :] - reference[None, :, :], axis=2)
    positive = distances[distances > 1e-12]
    return float(np.median(positive)) if positive.size else 1.0


def _mean_rbf_kernel(
    left: np.ndarray,
    right: np.ndarray,
    *,
    bandwidth: float,
    chunk_size: int,
) -> float:
    denominator = 2.0 * bandwidth * bandwidth
    total = 0.0
    pairs = 0
    for start in range(0, len(left), chunk_size):
        block = left[start : start + chunk_size]
        squared = np.sum((block[:, None, :] - right[None, :, :]) ** 2, axis=2)
        kernel = np.exp(-squared / denominator)
        total += float(kernel.sum())
        pairs += int(kernel.size)
    return total / pairs


def _rbf_mmd2(
    generated: np.ndarray,
    reference: np.ndarray,
    *,
    chunk_size: int,
) -> dict[str, float]:
    bandwidth = _reference_bandwidth(reference)
    value = (
        _mean_rbf_kernel(generated, generated, bandwidth=bandwidth, chunk_size=chunk_size)
        + _mean_rbf_kernel(reference, reference, bandwidth=bandwidth, chunk_size=chunk_size)
        - 2.0 * _mean_rbf_kernel(generated, reference, bandwidth=bandwidth, chunk_size=chunk_size)
    )
    return {"mmd2": max(0.0, float(value)), "reference_median_bandwidth": bandwidth}


def _wasserstein(generated: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    grid = np.linspace(0.0, 1.0, 101)
    distances = np.mean(
        np.abs(np.quantile(generated, grid, axis=0) - np.quantile(reference, grid, axis=0)),
        axis=0,
    )
    return {
        "mean_across_role_descriptors": float(distances.mean()),
        "by_descriptor": {
            name: float(distances[index]) for index, name in enumerate(ROLE_DESCRIPTOR_NAMES)
        },
        "scale": "group-balanced measured-Ugi train-development robust standardized units",
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(values.min()),
        "q10": float(np.quantile(values, 0.1)),
        "median": float(np.quantile(values, 0.5)),
        "q90": float(np.quantile(values, 0.9)),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
    }


def _continuous_summary(
    generated: np.ndarray,
    reference: np.ndarray,
    *,
    chunk_size: int,
) -> dict[str, Any]:
    nearest = []
    for start in range(0, len(generated), chunk_size):
        block = generated[start : start + chunk_size]
        distances = np.linalg.norm(block[:, None, :] - reference[None, :, :], axis=2)
        nearest.extend(np.min(distances, axis=1).tolist())
    return {
        "normalized_wasserstein": _wasserstein(generated, reference),
        "energy_distance": _energy_distance(
            generated,
            reference,
            chunk_size=chunk_size,
        ),
        "rbf_mmd": _rbf_mmd2(generated, reference, chunk_size=chunk_size),
        "nearest_reference_distance": _quantiles(np.asarray(nearest, dtype=np.float64)),
    }


def assess_ugi_role_realism(
    assessed_attempts: Sequence[Mapping[str, Any]],
    reference: UgiDevelopmentReference,
    policy: RealismPolicy,
    *,
    sanity_shift_standardized_units: float,
) -> dict[str, Any]:
    """Assess exact Ugi roles with continuous, group-balanced development diagnostics."""

    if not assessed_attempts:
        raise UgiDevelopmentRealismError("assessed Ugi attempt ledger is empty")
    if not math.isfinite(sanity_shift_standardized_units) or sanity_shift_standardized_units <= 0:
        raise UgiDevelopmentRealismError("sanity shift must be finite and positive")
    scaling_raw = np.asarray(
        [ugi_role_descriptor_vector(row.components()) for row in reference.scaling],
        dtype=np.float64,
    )
    evaluation_raw = np.asarray(
        [ugi_role_descriptor_vector(row.components()) for row in reference.evaluation],
        dtype=np.float64,
    )
    scale = fit_robust_descriptor_scale(scaling_raw)
    scaling_matrix = scale.transform(scaling_raw)
    evaluation_matrix = scale.transform(evaluation_raw)
    evaluation_rows = tuple(
        ReferenceMolecule(row.structure_id, row.canonical_smiles, row.group_id)
        for row in reference.evaluation
    )
    evaluation_vectors = {
        row.structure_id: evaluation_matrix[index] for index, row in enumerate(reference.evaluation)
    }

    exact_vectors: list[np.ndarray] = []
    unique_vectors: dict[str, np.ndarray] = {}
    exact_attempts = 0
    ambiguous_exact_attempts = 0
    for index, row in enumerate(assessed_attempts):
        if row.get("attempt_index") != index:
            raise UgiDevelopmentRealismError("assessed Ugi attempts are not gap-free and ordered")
        if row.get("exact_l1_program") is not True:
            continue
        traces = row.get("exact_l1_traces")
        if not isinstance(traces, list) or not traces:
            raise UgiDevelopmentRealismError("exact-L1 attempt lacks its verified trace")
        if len(traces) != 1:
            ambiguous_exact_attempts += 1
            continue
        components = traces[0].get("components_by_role")
        if not isinstance(components, Mapping):
            raise UgiDevelopmentRealismError("exact-L1 trace lacks role components")
        vector = ugi_role_descriptor_vector(
            {str(role): str(smiles) for role, smiles in components.items()}
        )
        exact_vectors.append(vector)
        exact_attempts += 1
        canonical = row.get("canonical_smiles")
        if not isinstance(canonical, str) or not canonical:
            raise UgiDevelopmentRealismError("exact-L1 attempt lacks canonical product identity")
        unique_vectors.setdefault(canonical, vector)
    if not exact_vectors:
        raise UgiDevelopmentRealismError("no unambiguous exact-L1 products support role realism")
    attempt_matrix = scale.transform(np.asarray(exact_vectors, dtype=np.float64))
    unique_matrix = scale.transform(np.asarray(list(unique_vectors.values()), dtype=np.float64))
    standardized_unique = {
        smiles: unique_matrix[index] for index, smiles in enumerate(unique_vectors)
    }
    c2st = grouped_classifier_two_sample(
        standardized_unique,
        evaluation_rows,
        evaluation_vectors,
        policy,
        feature_names=ROLE_DESCRIPTOR_NAMES,
        generated_population="generated_ugi_role_c2st",
        reference_population="measured_ugi_development_role_c2st",
    )

    scaling_vectors = {
        row.canonical_smiles: scaling_matrix[index] for index, row in enumerate(reference.scaling)
    }
    measured_control = grouped_classifier_two_sample(
        scaling_vectors,
        evaluation_rows,
        evaluation_vectors,
        policy,
        feature_names=ROLE_DESCRIPTOR_NAMES,
        generated_population="measured_ugi_development_scaling_control",
        reference_population="measured_ugi_development_evaluation_control",
    )
    shifted_matrix = evaluation_matrix.copy()
    for feature in SANITY_SHIFT_FEATURES:
        shifted_matrix[:, ROLE_DESCRIPTOR_NAMES.index(feature)] += sanity_shift_standardized_units
    shifted_vectors = {
        row.canonical_smiles: shifted_matrix[index]
        for index, row in enumerate(reference.evaluation)
    }
    corrupted_control = grouped_classifier_two_sample(
        shifted_vectors,
        evaluation_rows,
        evaluation_vectors,
        policy,
        feature_names=ROLE_DESCRIPTOR_NAMES,
        generated_population="shifted_ugi_role_descriptor_control",
        reference_population="measured_ugi_development_evaluation_control",
    )
    measured_energy = _energy_distance(
        scaling_matrix,
        evaluation_matrix,
        chunk_size=policy.distance_chunk_size,
    )
    corrupted_energy = _energy_distance(
        shifted_matrix,
        evaluation_matrix,
        chunk_size=policy.distance_chunk_size,
    )
    sanity_gates = {
        "measured_control_uses_all_requested_folds": measured_control.get("folds")
        == policy.c2st_folds,
        "corrupted_control_uses_all_requested_folds": corrupted_control.get("folds")
        == policy.c2st_folds,
        "corrupted_control_exceeds_measured_energy": corrupted_energy > measured_energy,
        "corrupted_control_auc_exceeds_measured_auc": (
            measured_control.get("status") == "estimated"
            and corrupted_control.get("status") == "estimated"
            and float(corrupted_control["auc_mean"]) > float(measured_control["auc_mean"])
        ),
    }
    return {
        "schema_version": "forge.ugi_role_development_realism.v1",
        "status": "pass" if all(sanity_gates.values()) else "fail",
        "attempts": len(assessed_attempts),
        "exact_l1_unambiguous_attempts": exact_attempts,
        "exact_l1_unambiguous_fraction_per_attempt": exact_attempts / len(assessed_attempts),
        "ambiguous_exact_l1_attempts_excluded": ambiguous_exact_attempts,
        "unique_exact_l1_products": len(unique_vectors),
        "features": list(ROLE_DESCRIPTOR_NAMES),
        "attempt_weighted": _continuous_summary(
            attempt_matrix,
            evaluation_matrix,
            chunk_size=policy.distance_chunk_size,
        ),
        "unique_product_weighted": {
            **_continuous_summary(
                unique_matrix,
                evaluation_matrix,
                chunk_size=policy.distance_chunk_size,
            ),
            "classifier_two_sample": c2st,
        },
        "sanity_controls": {
            "measured_scaling_vs_measured_evaluation": {
                "classifier_two_sample": measured_control,
                "energy_distance": measured_energy,
            },
            "standardized_shift_vs_measured_evaluation": {
                "shifted_features": list(SANITY_SHIFT_FEATURES),
                "shift_standardized_units": sanity_shift_standardized_units,
                "classifier_two_sample": corrupted_control,
                "energy_distance": corrupted_energy,
            },
            "gates": sanity_gates,
        },
        "reference": reference.realism.audit,
        "candidate_selection": False,
        "route_or_oracle_calls": 0,
        "heldout_product_structures_accessed": False,
        "nonclaims": [
            "This is a train-fold-only development diagnostic, not a held-out model claim.",
            "Role-distribution proximity is not biological activity or synthesis success.",
            "Only unambiguous verified exact-L1 traces enter conditional role distances.",
        ],
    }


__all__ = [
    "ROLE_DESCRIPTOR_NAMES",
    "SANITY_SHIFT_FEATURES",
    "UgiDevelopmentRealismError",
    "assess_ugi_role_realism",
    "ugi_role_descriptor_vector",
]
