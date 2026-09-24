"""Independent structural support diagnostics for complete COMPOSE lipid graphs.

These are reference-support measurements, not chemical validity rules. An absent
environment or ring is unknown; presence does not establish stability, synthesis,
ionization or delivery. No SMARTS or reaction-specific chemistry is invented here.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator


def canonical_molecule(smiles: str | None) -> tuple[str, Chem.Mol] | None:
    """Sanitize and constitutionalize without repairing or discarding fragments."""
    if not smiles:
        return None
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False), mol


def atom_environment(atom: Chem.Atom) -> tuple:
    """The complete labeled radius-one star, including all incident bond orders."""
    return (
        atom.GetSymbol(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        atom.GetNumRadicalElectrons(),
        atom.GetIsAromatic(),
        tuple(
            sorted(
                (
                    neighbor.GetSymbol(),
                    neighbor.GetFormalCharge(),
                    neighbor.GetIsAromatic(),
                    bond.GetBondTypeAsDouble(),
                )
                for bond in atom.GetBonds()
                for neighbor in [bond.GetOtherAtom(atom)]
            )
        ),
    )


def representation_diagnostics(
    mol: Chem.Mol, *, elements: set[str], maximum_heavy_atoms: int, maximum_cycles: int
) -> dict[str, Any]:
    """Measure declared graph support without imposing a narrower lipid size prior."""
    fragments = len(Chem.GetMolFrags(mol))
    cycles = mol.GetNumBonds() - mol.GetNumAtoms() + fragments
    return {
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "independent_cycles": cycles,
        "within_declared_size_element_cycle_bounds": (
            mol.GetNumHeavyAtoms() <= maximum_heavy_atoms
            and cycles <= maximum_cycles
            and {atom.GetSymbol() for atom in mol.GetAtoms()} <= elements
        ),
        "radical_electrons": sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()),
    }


def ring_systems(mol: Chem.Mol) -> tuple[str, ...]:
    """Canonical complete connected ring-bond components, including spiro systems.

    This uses all ring bonds, rather than an arbitrary SSSR or a serialization's
    fundamental cycles. Exocyclic attachments are assessed by atom_environment.
    The ring-core signature itself does not encode attachment positions.
    """
    neighbors: dict[int, set[int]] = {}
    ring_bonds = []
    for bond in mol.GetBonds():
        if bond.IsInRing():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            neighbors.setdefault(a, set()).add(b)
            neighbors.setdefault(b, set()).add(a)
            ring_bonds.append(bond.GetIdx())
    pending = set(neighbors)
    signatures = []
    while pending:
        component = {min(pending)}
        frontier = list(component)
        while frontier:
            for other in neighbors[frontier.pop()]:
                if other not in component:
                    component.add(other)
                    frontier.append(other)
        pending.difference_update(component)
        bonds = [
            index
            for index in ring_bonds
            if mol.GetBondWithIdx(index).GetBeginAtomIdx() in component
        ]
        signatures.append(
            Chem.MolFragmentToSmiles(
                mol,
                atomsToUse=sorted(component),
                bondsToUse=bonds,
                canonical=True,
                isomericSmiles=False,
            )
        )
    return tuple(sorted(signatures))


@dataclass
class StructuralSupport:
    """Unweighted union over distinct source constitutions; no frequency prior."""

    environments: set[tuple] = field(default_factory=set)
    rings: set[str] = field(default_factory=set)
    identities: set[str] = field(default_factory=set)

    def add(self, smiles: str) -> None:
        parsed = canonical_molecule(smiles)
        if parsed is None or len(Chem.GetMolFrags(parsed[1])) != 1:
            raise ValueError("Reference contains an invalid or disconnected molecule")
        canonical, mol = parsed
        if canonical not in self.identities:
            self.identities.add(canonical)
            self.environments.update(atom_environment(atom) for atom in mol.GetAtoms())
            self.rings.update(ring_systems(mol))

    def assess(self, mol: Chem.Mol) -> dict[str, Any]:
        unknown = [
            {"atom_index": atom.GetIdx(), "environment": atom_environment(atom)}
            for atom in mol.GetAtoms()
            if atom_environment(atom) not in self.environments
        ]
        rings = ring_systems(mol)
        missing_rings = [signature for signature in rings if signature not in self.rings]
        return {
            "status": "assessed" if self.identities else "unassessed_no_reference",
            "unknown_atom_environments": unknown,
            "unknown_ring_systems": missing_rings,
            "atom_environment_coverage": (
                1 - len(unknown) / mol.GetNumAtoms() if self.identities else None
            ),
            "all_local_features_observed": (
                not unknown and not missing_rings if self.identities else None
            ),
            "reference_unique_constitutions": len(self.identities),
        }


def identity_diversity(identities: Iterable[str], *, requests: int) -> dict[str, Any]:
    """Exact constitution diversity with all-request and conditional denominators."""
    if requests < 0:
        raise ValueError("requests must be nonnegative")
    counts = Counter(identities)
    n = sum(counts.values())
    if n > requests:
        raise ValueError("one identity per request is required")
    probabilities = [count / n for count in counts.values()] if n else []
    entropy = -sum(p * math.log(p) for p in probabilities)
    return {
        "observations": n,
        "unique": len(counts),
        "unique_per_request": len(counts) / requests if requests else None,
        "unique_per_1000_requests": 1000 * len(counts) / requests if requests else None,
        "shannon_entropy_nats": entropy if n else None,
        "shannon_effective_count": math.exp(entropy) if n else 0.0,
        "simpson_effective_count": 1 / sum(p * p for p in probabilities) if n else 0.0,
        "largest_multiplicity": max(counts.values(), default=0),
    }


def source_role_mapping(
    components: Mapping[str, str], assessment: Mapping[str, Any], executors: Sequence[Mapping]
) -> dict[str, str]:
    """Resolve accepted registry roles using the exact ordered source executors.

    The caller must obtain executors from the request's pinned program/layout.
    No family-level synonym table or structural inference supplies the mapping.
    """
    checks = assessment.get("checks", [])
    if len(checks) != len(executors):
        raise ValueError("Saved assessment differs from matching source executor order")
    mappings = {
        tuple(sorted(executor["mapping"].items()))
        for executor, check in zip(executors, checks, strict=True)
        if dict(components) in check.get("accepted_components", [])
    }
    if len(mappings) != 1:
        raise ValueError("Accepted source-role mapping is absent or ambiguous")
    result = dict(next(iter(mappings)))
    if set(result) != set(components):
        raise ValueError("Accepted components differ from registered roles")
    return result


def fingerprint_distribution(
    generated: Sequence[str], reference: Sequence[str], *, requests: int, neighbors: int = 5
) -> dict[str, Any]:
    """ECFP4 set precision/coverage plus all-attempt distinct precision.

    Precision uses the union of reference k-nearest-neighbor balls. Coverage is
    the fraction of reference balls hit by a generated point. Recall separately
    measures reference points within generated balls; small cohorts make those
    generated balls broad. None of these metrics establishes chemical quality.
    """
    generated = sorted(set(generated))
    reference = sorted(set(reference))
    if len(reference) <= neighbors or not generated:
        return {"status": "unassessed_insufficient_molecules"}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def fingerprints(smiles: Sequence[str]) -> list[Any]:
        output = []
        for value in smiles:
            parsed = canonical_molecule(value)
            if parsed is None:
                raise ValueError("Fingerprint input must be valid")
            output.append(generator.GetFingerprint(parsed[1]))
        return output

    ref = fingerprints(reference)
    gen = fingerprints(generated)
    # At most the explicitly bounded reference and development set are materialized.
    rr = np.asarray([1 - np.asarray(DataStructs.BulkTanimotoSimilarity(f, ref)) for f in ref])
    radii = np.partition(rr, neighbors, axis=1)[:, neighbors]
    gr = np.asarray([1 - np.asarray(DataStructs.BulkTanimotoSimilarity(f, ref)) for f in gen])
    membership = np.any(gr <= radii[None, :], axis=1)
    covered = float(np.mean(np.any(gr <= radii[None, :], axis=0)))
    recall = None
    internal_diversity = None
    if len(gen) > 1:
        gg = np.asarray([1 - np.asarray(DataStructs.BulkTanimotoSimilarity(f, gen)) for f in gen])
        k = min(neighbors, len(gen) - 1)
        generated_radii = np.partition(gg, k, axis=1)[:, k]
        recall = float(np.mean(np.any(gr <= generated_radii[:, None], axis=0)))
        internal_diversity = float(gg.sum() / (len(gen) * (len(gen) - 1)))
    return {
        "status": "assessed_development_distribution_only",
        "unique_generated": len(gen),
        "unique_reference": len(ref),
        "reference_neighbors": neighbors,
        "generated_neighbors": min(neighbors, len(gen) - 1),
        "fingerprint_precision_among_unique": float(membership.mean()),
        "distinct_in_reference_manifold_per_request": float(membership.sum() / requests),
        "fingerprint_coverage": covered,
        "fingerprint_recall": recall,
        "nearest_reference_tanimoto_mean": float((1 - gr.min(axis=1)).mean()),
        "mean_pairwise_tanimoto_distance": internal_diversity,
    }


def descriptor_comparison(
    generated: np.ndarray, reference: np.ndarray, names: Sequence[str]
) -> Mapping[str, Any]:
    """Signed means and normalized Wasserstein distance, without an acceptance cutoff."""
    from scipy.stats import wasserstein_distance

    if not len(generated) or len(reference) < 2:
        return {"status": "unassessed_insufficient_molecules"}
    q25, q75 = np.quantile(reference, [0.25, 0.75], axis=0)
    scale = q75 - q25
    scale = np.where(scale > 1e-12, scale, np.maximum(reference.std(axis=0), 1.0))
    rows = {}
    for index, name in enumerate(names):
        rows[name] = {
            "generated_mean": float(generated[:, index].mean()),
            "reference_mean": float(reference[:, index].mean()),
            "reference_scale": float(scale[index]),
            "normalized_wasserstein": float(
                wasserstein_distance(generated[:, index], reference[:, index]) / scale[index]
            ),
        }
    return {"status": "assessed_development_distribution_only", "descriptors": rows}
