"""Output-identity guards for the CPU assessment optimizations.

Every function exercised here was rewritten for wall time only.  Each test therefore re-derives the
same quantity a second, deliberately naive way -- the breadth-first traversal the lipid context used
to run, the per-row canonicalization the Ugi identity reference used to run, the second evaluation
pass a single-program ledger used to run, a fresh parse for every fingerprint -- and requires exact
equality.  A future speed-up that changes a verdict, a metric, an abstention or an ambiguity
classification fails here rather than silently entering the evidence record.
"""

from __future__ import annotations

import gzip
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

import forge.model.common_lipid_realism as realism
import forge.model.reaction_program_evaluation as evaluation
from forge.model.common_ugi_benchmark import load_ugi_identity_references
from forge.model.defog_feasibility import FeasibilityError
from forge.model.lipid_context import (
    _is_amide_like_nitrogen,
    _is_carbonyl_carbon,
    _local_hetero_count,
    assign_lipid_regions,
    rooted_distances,
    select_lipid_polar_root,
)

REPO = Path(__file__).resolve().parents[1]

# Lipids spanning the shapes the root rule has to discriminate: several candidate nitrogens, a
# charged head, amide-only nitrogens, a saturated ring head, a phosphorus head, and a symmetric
# molecule whose tie-break has to reach the canonical ranking.
LIPIDS = (
    "CCCCCCCCCCCCCCCCCC(=O)OCCCCCC(NCCN1CCN(C)CC1)C(=O)NCCCCCCCCC",
    "CCCCCCCCCCCCCCNC(=O)C(CCCOC(=O)CCCCCCC)NCCN(CC)CC",
    "CCCCCCCCCC(=O)OCCCCCC(N)CN1CCCCC1",
    "CCCCNC(=O)C(CCCC)N(C)CC",
    "CCCCCCCCC[N+](C)(C)CCCCOP(=O)([O-])OCC",
    "CCCCCCNC(=O)CCC(=O)NCCCCCC",
    "OCCN(CCO)CCN(CCO)CCO",
    "CC(C)CCCCCCOC(=O)CCCCN(CCCCC(=O)OCCCCCCC(C)C)CCO",
    "C1CC1CCN(CC2CC2)CCOC(=O)CCCCCCCCCCC",
    "CCCCCCCCCCCCCCCCCCN",
)


def _reference_distances(molecule: Chem.Mol, start: int) -> list[int]:
    """The pre-optimization breadth-first traversal, kept verbatim as the oracle."""

    distances = [-1] * molecule.GetNumAtoms()
    distances[start] = 0
    queue = deque([start])
    while queue:
        atom_index = queue.popleft()
        for neighbor in molecule.GetAtomWithIdx(atom_index).GetNeighbors():
            neighbor_index = neighbor.GetIdx()
            if distances[neighbor_index] < 0:
                distances[neighbor_index] = distances[atom_index] + 1
                queue.append(neighbor_index)
    if any(distance < 0 for distance in distances):
        raise FeasibilityError("lipid-native root selection requires one connected molecule")
    return distances


def _reference_root(molecule: Chem.Mol) -> int:
    """The pre-optimization root rule: one eager lexicographic maximum over every candidate."""

    ranks = list(Chem.CanonicalRankAtoms(molecule, breakTies=True))
    primary_roles: list[int] = []
    for atom in molecule.GetAtoms():
        symbol = atom.GetSymbol()
        charge = atom.GetFormalCharge()
        non_amide_nitrogen = symbol == "N" and not _is_amide_like_nitrogen(atom)
        primary_roles.append(
            5
            if non_amide_nitrogen and charge > 0
            else (
                4
                if non_amide_nitrogen
                else (
                    3 if symbol == "P" else 2 if symbol == "N" else 1 if symbol in {"O", "S"} else 0
                )
            )
        )
    best_role = max(primary_roles)
    candidates = [index for index, role in enumerate(primary_roles) if role == best_role]

    def tie_break(index: int) -> tuple[int, int, int, int, int]:
        distances = _reference_distances(molecule, index)
        local_hetero = sum(
            distance <= 2 and molecule.GetAtomWithIdx(other).GetSymbol() not in {"C", "H", "F"}
            for other, distance in enumerate(distances)
        )
        atom = molecule.GetAtomWithIdx(index)
        return (
            local_hetero,
            int(atom.GetFormalCharge() != 0),
            atom.GetDegree(),
            -max(distances),
            -ranks[index],
        )

    return max(candidates, key=tie_break)


def _reference_regions(molecule: Chem.Mol, root: int) -> np.ndarray:
    distances = tuple(_reference_distances(molecule, root))
    head_ring_atoms: set[int] = set()
    for ring in molecule.GetRingInfo().AtomRings():
        if any(distances[index] <= 3 for index in ring):
            head_ring_atoms.update(ring)
    regions = np.full(molecule.GetNumAtoms(), 2, dtype=np.int64)
    for atom in molecule.GetAtoms():
        index = atom.GetIdx()
        hard_linker = (
            atom.GetSymbol() in {"O", "S", "P"}
            or _is_carbonyl_carbon(atom)
            or _is_amide_like_nitrogen(atom)
        )
        if (distances[index] <= 3 or index in head_ring_atoms) and not hard_linker:
            regions[index] = 0
        elif hard_linker or (4 <= distances[index] <= 7 and atom.GetDegree() >= 3):
            regions[index] = 1
    return regions


@pytest.mark.parametrize("smiles", LIPIDS)
def test_lipid_context_matches_the_eager_breadth_first_rule(smiles: str) -> None:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    expected_root = _reference_root(molecule)
    root = select_lipid_polar_root(molecule)

    assert root == expected_root
    assert list(rooted_distances(molecule, root)) == _reference_distances(molecule, expected_root)
    assert np.array_equal(
        assign_lipid_regions(molecule, root), _reference_regions(molecule, expected_root)
    )


@pytest.mark.parametrize("smiles", LIPIDS)
def test_supplied_rooted_distances_do_not_change_region_assignment(smiles: str) -> None:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    root = select_lipid_polar_root(molecule)
    distances = rooted_distances(molecule, root)

    assert np.array_equal(
        assign_lipid_regions(molecule, root, distances=distances),
        assign_lipid_regions(molecule, root),
    )


def test_supplied_rooted_distances_are_validated_not_trusted() -> None:
    molecule = Chem.MolFromSmiles(LIPIDS[3])
    assert molecule is not None
    root = select_lipid_polar_root(molecule)
    wrong_root = tuple(rooted_distances(molecule, (root + 1) % molecule.GetNumAtoms()))

    with pytest.raises(FeasibilityError):
        assign_lipid_regions(molecule, root, distances=wrong_root)
    with pytest.raises(FeasibilityError):
        assign_lipid_regions(molecule, root, distances=(0,))


@pytest.mark.parametrize("smiles", LIPIDS)
def test_two_hop_heteroatom_count_matches_the_traversal_definition(smiles: str) -> None:
    """The root rule's inlined neighborhood count must not drift from `_local_hetero_count`."""

    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    atoms = list(molecule.GetAtoms())
    hetero = [atom.GetSymbol() not in {"C", "H", "F"} for atom in atoms]

    for index in range(molecule.GetNumAtoms()):
        neighborhood = {index}
        for neighbor in atoms[index].GetNeighbors():
            neighborhood.add(neighbor.GetIdx())
            neighborhood.update(second.GetIdx() for second in neighbor.GetNeighbors())
        assert sum(hetero[member] for member in neighborhood) == _local_hetero_count(
            molecule, index
        )


def test_disconnected_input_still_fails_closed() -> None:
    molecule = Chem.MolFromSmiles("CCO.CCN")
    assert molecule is not None
    with pytest.raises(FeasibilityError):
        select_lipid_polar_root(molecule)
    with pytest.raises(FeasibilityError):
        rooted_distances(molecule, 0)


@pytest.mark.parametrize("smiles", LIPIDS)
def test_descriptor_vector_matches_a_per_descriptor_recomputation(smiles: str) -> None:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    vector = realism._descriptor_vector(molecule)

    atoms = list(molecule.GetAtoms())
    regions = _reference_regions(molecule, _reference_root(molecule))
    names = list(realism.DESCRIPTOR_NAMES)
    assert vector[names.index("formal_charge")] == sum(atom.GetFormalCharge() for atom in atoms)
    assert vector[names.index("absolute_formal_charge")] == sum(
        abs(atom.GetFormalCharge()) for atom in atoms
    )
    assert vector[names.index("branch_atoms")] == sum(
        atom.GetAtomicNum() > 1 and atom.GetDegree() >= 3 for atom in atoms
    )
    assert vector[names.index("tail_branch_atoms")] == sum(
        atom.GetAtomicNum() > 1 and atom.GetDegree() >= 3 and int(regions[atom.GetIdx()]) == 2
        for atom in atoms
    )
    heavy = molecule.GetNumHeavyAtoms()
    for element, descriptor in (
        ("N", "nitrogen_atoms"),
        ("O", "oxygen_atoms"),
        ("S", "sulfur_atoms"),
        ("P", "phosphorus_atoms"),
    ):
        assert vector[names.index(descriptor)] == sum(atom.GetSymbol() == element for atom in atoms)
    assert vector[names.index("carbon_fraction")] == (
        sum(atom.GetSymbol() == "C" for atom in atoms) / heavy
    )


def _write_assignments(path: Path, rows: list[dict[str, str]], roles: tuple[str, ...]) -> None:
    header = [
        "product_id",
        "canonical_product_smiles",
        "primary_product_fold",
        *(f"{role}_smiles" for role in roles),
        *(f"{role}_family_fold" for role in roles),
    ]
    lines = [",".join(header)]
    lines.extend(",".join(row[column] for column in header) for row in rows)
    path.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode()))


def test_ugi_identity_reference_memoization_matches_per_row_canonicalization(
    tmp_path: Path,
) -> None:
    roles = ("amine_head", "isocyanide_tail")
    # The same component constitution recurs across rows in several non-canonical spellings and in
    # both folds, which is exactly the situation the per-role memo has to reproduce.
    rows = [
        {
            "product_id": "P0",
            "canonical_product_smiles": "CCCCNC(=O)C(CCCC)N(C)CC",
            "primary_product_fold": "train",
            "amine_head_smiles": "NCCN(C)C",
            "isocyanide_tail_smiles": "[C-]#[N+]CCCC",
            "amine_head_family_fold": "train",
            "isocyanide_tail_family_fold": "train",
        },
        {
            "product_id": "P1",
            "canonical_product_smiles": "CCCCCCNC(=O)CCC(=O)NCCCCCC",
            "primary_product_fold": "train",
            "amine_head_smiles": "C(N)CN(C)C",
            "isocyanide_tail_smiles": "[C-]#[N+]CCCC",
            "amine_head_family_fold": "heldout",
            "isocyanide_tail_family_fold": "train",
        },
        {
            "product_id": "P2",
            "canonical_product_smiles": "OCCN(CCO)CCN(CCO)CCO",
            "primary_product_fold": "heldout",
            "amine_head_smiles": "NCCN(C)C",
            "isocyanide_tail_smiles": "[C-]#[N+]CCCCCC",
            "amine_head_family_fold": "train",
            "isocyanide_tail_family_fold": "heldout",
        },
    ]
    path = tmp_path / "assignments.csv.gz"
    _write_assignments(path, rows, roles)

    def canonical(smiles: str) -> str:
        molecule = Chem.MolFromSmiles(smiles)
        assert molecule is not None
        return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)

    expected_products = {
        canonical(row["canonical_product_smiles"])
        for row in rows
        if row["primary_product_fold"] == "train"
    }
    expected_training = {
        role: {
            canonical(row[f"{role}_smiles"])
            for row in rows
            if row["primary_product_fold"] == "train"
        }
        for role in roles
    }
    expected_held = {
        role: {
            canonical(row[f"{role}_smiles"])
            for row in rows
            if row[f"{role}_family_fold"] == "heldout"
        }
        for role in roles
    }

    products, training, held = load_ugi_identity_references(path, roles=roles)

    assert products == expected_products
    assert training == expected_training
    assert held == expected_held


def test_invalid_component_still_names_its_role(tmp_path: Path) -> None:
    roles = ("amine_head", "isocyanide_tail")
    rows = [
        {
            "product_id": "P0",
            "canonical_product_smiles": "CCCCNC(=O)C(CCCC)N(C)CC",
            "primary_product_fold": "train",
            "amine_head_smiles": "NCCN(C)C",
            "isocyanide_tail_smiles": "CCO.CCN",
            "amine_head_family_fold": "train",
            "isocyanide_tail_family_fold": "train",
        }
    ]
    path = tmp_path / "assignments.csv.gz"
    _write_assignments(path, rows, roles)

    with pytest.raises(Exception, match="invalid isocyanide_tail component"):
        load_ugi_identity_references(path, roles=roles)


def _evaluation_rows() -> list[dict[str, Any]]:
    return [
        {
            "program_id": "ugi_3cr_agile",
            "valid": True,
            "canonical_smiles": smiles,
            "exact_l1_program": False,
            "exact_l1_trace_count": 0,
            "forward_verified_trace_count": 0,
            "exact_l1_traces": [],
        }
        for smiles in LIPIDS
    ] + [
        {
            "program_id": "ugi_3cr_agile",
            "valid": False,
            "canonical_smiles": None,
            "exact_l1_program": False,
            "exact_l1_trace_count": 0,
            "forward_verified_trace_count": 0,
            "exact_l1_traces": [],
        }
    ]


def test_single_program_per_program_block_equals_a_second_evaluation_pass() -> None:
    rows = _evaluation_rows()
    references = {"ugi_3cr_agile": set()}
    components: dict[str, dict[str, set[str]]] = {"ugi_3cr_agile": {"amine_head": set()}}

    result = evaluation.evaluate_reaction_program_samples(
        rows, training_products=references, training_components=components
    )
    independent = evaluation._evaluate_rows(
        [row for row in rows if row["program_id"] == "ugi_3cr_agile"],
        training_products=references,
        training_components=components,
    )

    assert result["per_program"]["ugi_3cr_agile"] == independent
    assert result["overall"] == independent
    # The reused block must not alias the overall block: a caller that annotates one must not
    # silently annotate the other.
    assert result["per_program"]["ugi_3cr_agile"] is not result["overall"]


def test_mean_pairwise_distance_ignores_supplied_molecules_for_its_value() -> None:
    smiles = list(LIPIDS)
    molecules = {value: Chem.MolFromSmiles(value) for value in smiles}
    # Touch the shared molecules the way the assessors do before the fingerprints are taken.
    for value, molecule in molecules.items():
        Chem.GetMolFrags(molecule)
        Chem.CanonicalRankAtoms(molecule, breakTies=True)

    assert evaluation._mean_pairwise_distance(smiles, molecules) == (
        evaluation._mean_pairwise_distance(smiles)
    )


def test_mean_pairwise_distance_fails_when_a_supplied_molecule_is_missing() -> None:
    smiles = list(LIPIDS)
    molecules = {value: Chem.MolFromSmiles(value) for value in smiles[1:]}

    with pytest.raises(evaluation.ReactionProgramEvaluationError):
        evaluation._mean_pairwise_distance(smiles, molecules)


def test_internal_diversity_reuses_fingerprints_without_changing_the_value() -> None:
    policy = realism.RealismPolicy.from_mapping(
        json.loads((REPO / "configs/multireaction/common_lipid_realism_v1.json").read_text())[
            "policy"
        ]
    )
    canonical = [
        Chem.MolToSmiles(Chem.MolFromSmiles(value), canonical=True, isomericSmiles=False)
        for value in LIPIDS
    ]
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=policy.fingerprint_radius, fpSize=policy.fingerprint_bits
    )
    fingerprints = {
        value: generator.GetFingerprint(Chem.MolFromSmiles(value)) for value in canonical
    }
    value, rows = realism._internal_diversity(canonical, policy, fingerprints)

    selected = realism._ranked_sample(
        [realism.ReferenceMolecule(item, item, item) for item in sorted(set(canonical))],
        population="generated_internal_diversity",
        seed=policy.selection_seed,
        limit=policy.internal_diversity_limit,
    )
    fresh = [generator.GetFingerprint(Chem.MolFromSmiles(row.canonical_smiles)) for row in selected]
    distances: list[float] = []
    for index, fingerprint in enumerate(fresh[:-1]):
        distances.extend(
            1.0 - float(similarity)
            for similarity in DataStructs.BulkTanimotoSimilarity(fingerprint, fresh[index + 1 :])
        )

    assert rows == len(selected)
    assert value == float(np.mean(distances))


def test_c2st_thread_limit_is_a_wall_time_setting_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """The frozen classifier two-sample AUCs must not depend on the OpenMP pool size."""

    rng = np.random.default_rng(20260828)
    rows_per_class = 60
    features = np.vstack(
        (
            rng.normal(0.0, 1.0, size=(rows_per_class, len(realism.DESCRIPTOR_NAMES))),
            rng.normal(0.4, 1.0, size=(rows_per_class, len(realism.DESCRIPTOR_NAMES))),
        )
    )

    class _Reference:
        heldout = tuple(
            realism.ReferenceMolecule(f"R{index}", f"S{index}", f"{index % 7}")
            for index in range(rows_per_class)
        )
        heldout_descriptors = features[:rows_per_class]

    generated = {f"G{index}": features[rows_per_class + index] for index in range(rows_per_class)}

    def run(threads: int) -> dict[str, Any]:
        monkeypatch.setattr(realism, "C2ST_OPENMP_THREADS", threads)
        policy = realism.RealismPolicy.from_mapping(
            json.loads((REPO / "configs/multireaction/common_lipid_realism_v1.json").read_text())[
                "policy"
            ]
            | {"c2st_minimum_rows_per_class": 10, "c2st_maximum_rows_per_class": rows_per_class}
        )
        return realism._classifier_two_sample(generated, _Reference(), policy)

    single = run(1)
    many = run(4)

    assert single["status"] == "estimated"
    assert [repr(value) for value in single["auc_folds"]] == [
        repr(value) for value in many["auc_folds"]
    ]
    assert repr(single["auc_mean"]) == repr(many["auc_mean"])
    assert single["folds"] == many["folds"] > 1
