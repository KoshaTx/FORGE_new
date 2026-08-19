"""Corpus-wide selection gate for a lipid-native morphology support skeleton."""

from __future__ import annotations

import csv
import gzip
import json
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, rdBase

from forge.core.io import atomic_write as _atomic_write
from forge.product.defog_feasibility import sha256_file
from forge.product.lipid_context import (
    HEAD_REGION,
    TAIL_REGION,
    assign_lipid_regions,
    select_lipid_polar_root,
)
from forge.product.lipid_support_skeleton import (
    FUNCTIONAL_SUPPORT,
    SKELETON_VARIANTS,
    encode_lipid_support_skeleton,
    protected_core_indices,
    selected_component_count,
    skeleton_signature,
    skeleton_statistics,
)


class LipidSupportSkeletonAuditError(RuntimeError):
    """Raised when the support-skeleton selection gate cannot be evaluated."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _quantiles(values: Sequence[int | float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "q00": float(array.min()),
        "q10": float(np.quantile(array, 0.10)),
        "q50": float(np.quantile(array, 0.50)),
        "q90": float(np.quantile(array, 0.90)),
        "q100": float(array.max()),
    }


def _stable_seed(seed: int, value: str) -> int:
    import hashlib

    digest = hashlib.sha256(f"{seed}\0{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _nonidentity_permutation(node_count: int, seed: int) -> list[int]:
    if node_count < 2:
        return list(range(node_count))
    rng = np.random.default_rng(seed)
    identity = np.arange(node_count)
    for _ in range(100):
        order = rng.permutation(node_count)
        if not np.array_equal(order, identity):
            return order.tolist()
    return np.roll(identity, 1).tolist()


def _skeleton_degrees(encoding: Any) -> dict[int, int]:
    degrees = {index: 0 for index in encoding.retained_atoms}
    for left, right, _ in encoding.skeleton_bonds:
        degrees[left] += 1
        degrees[right] += 1
    return degrees


def _regional_statistics(encoding: Any) -> dict[str, Any]:
    molecule = Chem.MolFromSmiles(encoding.canonical_smiles)
    if molecule is None:
        raise LipidSupportSkeletonAuditError(
            f"invalid canonical skeleton molecule: {encoding.structure_id}"
        )
    root = select_lipid_polar_root(molecule)
    regions = assign_lipid_regions(molecule, root)
    retained = set(encoding.retained_atoms)
    degrees = _skeleton_degrees(encoding)
    head_atoms = {index for index in retained if int(regions[index]) == HEAD_REGION}
    tail_atoms = {index for index in retained if int(regions[index]) == TAIL_REGION}
    return {
        "polar_root_retained": root in retained,
        "head_components": selected_component_count(encoding, head_atoms),
        "tail_components": selected_component_count(encoding, tail_atoms),
        "tail_junctions": sum(degrees[index] >= 3 for index in tail_atoms),
        "terminal_tail_carbons": sum(
            degrees[index] <= 1 and encoding.atom_states[index].symbol == "C"
            for index in tail_atoms
        ),
    }


def _audit_molecule(
    molecule: Chem.Mol,
    *,
    structure_id: str,
    corpus: str,
    fold: str,
    protected_atoms: set[int] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    protected = protected_atoms or set()
    full_cycle_rank = molecule.GetNumBonds() - molecule.GetNumAtoms() + 1
    for variant in SKELETON_VARIANTS:
        encoding = encode_lipid_support_skeleton(
            molecule,
            structure_id=structure_id,
            variant=variant,
            protected_atoms=protected,
        )
        statistics = skeleton_statistics(encoding)
        regional = _regional_statistics(encoding)
        rows.append(
            {
                "corpus": corpus,
                "fold": fold,
                "structure_id": structure_id,
                "variant": variant,
                "protected_atoms": len(protected),
                **statistics,
                **regional,
                "full_cycle_rank": full_cycle_rank,
                "cycle_rank_preserved": statistics["cycle_rank"] == full_cycle_rank,
                "removed_symbols_json": json.dumps(
                    statistics["removed_symbols"], sort_keys=True, separators=(",", ":")
                ),
                "removed_bond_types_json": json.dumps(
                    statistics["removed_bond_types"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    return rows


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant"])].append(row)
    result = {}
    for variant, values in sorted(grouped.items()):
        result[variant] = {
            "molecules": len(values),
            "exact_roundtrip_fraction": sum(bool(row["roundtrip_exact"]) for row in values)
            / len(values),
            "connected_fraction": sum(bool(row["connected"]) for row in values) / len(values),
            "cycle_rank_preserved_fraction": sum(
                bool(row["cycle_rank_preserved"]) for row in values
            )
            / len(values),
            "polar_root_retained_fraction": sum(bool(row["polar_root_retained"]) for row in values)
            / len(values),
            "retained_fraction": _quantiles([row["retained_fraction"] for row in values]),
            "removed_atoms": _quantiles([row["removed_atoms"] for row in values]),
            "components": _quantiles([row["components"] for row in values]),
            "full_leaves": _quantiles([row["full_leaves"] for row in values]),
            "skeleton_leaves": _quantiles([row["skeleton_leaves"] for row in values]),
            "resolved_false_junctions": _quantiles(
                [row["resolved_false_junctions"] for row in values]
            ),
            "skeleton_junctions": _quantiles([row["skeleton_junctions"] for row in values]),
            "head_components": _quantiles([row["head_components"] for row in values]),
            "tail_components": _quantiles([row["tail_components"] for row in values]),
            "tail_junctions": _quantiles([row["tail_junctions"] for row in values]),
            "terminal_tail_carbons": _quantiles([row["terminal_tail_carbons"] for row in values]),
        }
    return result


def _origin_metrics(
    products_path: Path,
    atoms_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    products = _read_csv(products_path)
    atom_rows: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(atoms_path):
        atom_rows[str(row["product_id"])].append(row)
    ledger = []
    for product in products:
        product_id = str(product["product_id"])
        molecule = Chem.MolFromSmiles(str(product["product_smiles"]))
        if molecule is None:
            raise LipidSupportSkeletonAuditError(f"invalid semantic Ugi product: {product_id}")
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
        if canonical != str(product["product_smiles"]):
            raise LipidSupportSkeletonAuditError(
                f"semantic Ugi atom order is not canonical: {product_id}"
            )
        rows = atom_rows[product_id]
        if sorted(int(row["product_atom_index"]) for row in rows) != list(
            range(molecule.GetNumAtoms())
        ):
            raise LipidSupportSkeletonAuditError(
                f"semantic origin rows do not cover product: {product_id}"
            )
        protected = protected_core_indices(rows)
        encoding = encode_lipid_support_skeleton(
            molecule,
            structure_id=product_id,
            variant=FUNCTIONAL_SUPPORT,
            protected_atoms=protected,
        )
        support_statistics = skeleton_statistics(encoding)
        retained = set(encoding.retained_atoms)
        roles: defaultdict[str, set[int]] = defaultdict(set)
        for row in rows:
            roles[str(row["origin_role"])].add(int(row["product_atom_index"]))
        role_components = {
            role: selected_component_count(encoding, indices & retained)
            for role, indices in roles.items()
        }
        tail_roles = ("oxoester_aldehyde_body_tail", "isocyanide_tail")
        ledger.append(
            {
                "product_id": product_id,
                "support_connected": selected_component_count(encoding) == 1,
                "core_atoms": len(protected),
                "core_atoms_retained": len(protected & retained),
                "amine_head_components": role_components.get("amine_head", 0),
                "aldehyde_tail_components": role_components.get(tail_roles[0], 0),
                "isocyanide_tail_components": role_components.get(tail_roles[1], 0),
                "two_tail_origins_present": all(roles.get(role) for role in tail_roles),
                "two_tail_origins_connected": all(
                    role_components.get(role, 0) == 1 for role in tail_roles
                ),
                "removed_atoms": support_statistics["removed_atoms"],
                "full_leaves": support_statistics["full_leaves"],
                "support_leaves": support_statistics["skeleton_leaves"],
                "full_junctions": support_statistics["full_junctions"],
                "support_junctions": support_statistics["skeleton_junctions"],
                "roundtrip_exact": support_statistics["roundtrip_exact"],
            }
        )
    if not ledger:
        raise LipidSupportSkeletonAuditError("semantic Ugi audit selected no products")
    summary = {
        "products": len(ledger),
        "support_connected_fraction": sum(row["support_connected"] for row in ledger) / len(ledger),
        "core_fully_retained_fraction": sum(
            row["core_atoms"] == row["core_atoms_retained"] for row in ledger
        )
        / len(ledger),
        "amine_head_connected_fraction": sum(row["amine_head_components"] == 1 for row in ledger)
        / len(ledger),
        "two_tail_origins_present_fraction": sum(row["two_tail_origins_present"] for row in ledger)
        / len(ledger),
        "two_tail_origins_connected_fraction": sum(
            row["two_tail_origins_connected"] for row in ledger
        )
        / len(ledger),
        "exact_roundtrip_fraction": sum(row["roundtrip_exact"] for row in ledger) / len(ledger),
        "protected_core_support_morphology": {
            "removed_atoms": _quantiles([row["removed_atoms"] for row in ledger]),
            "full_leaves": _quantiles([row["full_leaves"] for row in ledger]),
            "support_leaves": _quantiles([row["support_leaves"] for row in ledger]),
            "full_junctions": _quantiles([row["full_junctions"] for row in ledger]),
            "support_junctions": _quantiles([row["support_junctions"] for row in ledger]),
        },
        "components_by_origin": {
            "amine_head": _quantiles([row["amine_head_components"] for row in ledger]),
            "aldehyde_tail": _quantiles([row["aldehyde_tail_components"] for row in ledger]),
            "isocyanide_tail": _quantiles([row["isocyanide_tail_components"] for row in ledger]),
        },
    }
    return summary, ledger


def audit_lipid_support_skeletons(
    r0_path: Path,
    assignments_path: Path,
    ugi_l1_path: Path,
    ugi_products_path: Path,
    ugi_atoms_path: Path,
    *,
    seed: int,
    permutation_records: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare support definitions on all broad and Ugi product structures."""

    paths = {
        "r0_constitutional": r0_path,
        "r0_assignments": assignments_path,
        "ugi_l1_assignments": ugi_l1_path,
        "ugi_semantic_products": ugi_products_path,
        "ugi_semantic_atoms": ugi_atoms_path,
    }
    for label, path in paths.items():
        if not path.is_file():
            raise LipidSupportSkeletonAuditError(f"{label} not found: {path}")
    if seed < 0 or permutation_records < 1:
        raise LipidSupportSkeletonAuditError("invalid permutation-stability policy")

    started = time.monotonic()
    assignments = {
        str(row["r0_structure_id"]): str(row["source_study_fold"])
        for row in _read_csv(assignments_path)
    }
    broad_rows = []
    r0_source = _read_csv(r0_path)
    for row in r0_source:
        structure_id = str(row["r0_structure_id"])
        if str(row["r0_pretraining_eligible"]).lower() != "true":
            continue
        molecule = Chem.MolFromSmiles(str(row["canonical_constitutional_smiles"]))
        if molecule is None:
            raise LipidSupportSkeletonAuditError(f"invalid R0 molecule: {structure_id}")
        broad_rows.extend(
            _audit_molecule(
                molecule,
                structure_id=structure_id,
                corpus="R0",
                fold=assignments[structure_id],
            )
        )

    ugi_rows = []
    for row in _read_csv(ugi_l1_path):
        product_id = str(row["product_id"])
        molecule = Chem.MolFromSmiles(str(row["canonical_product_smiles"]))
        if molecule is None:
            raise LipidSupportSkeletonAuditError(f"invalid Ugi L1 product: {product_id}")
        fold = (
            "measured"
            if str(row["is_source_adjudicated_measured_product"]).lower() == "true"
            else "virtual"
        )
        ugi_rows.extend(
            _audit_molecule(
                molecule,
                structure_id=product_id,
                corpus="Ugi_L1",
                fold=fold,
            )
        )

    permutation_candidates = sorted(
        (row for row in r0_source if str(row["r0_pretraining_eligible"]).lower() == "true"),
        key=lambda row: _stable_seed(seed, str(row["r0_structure_id"])),
    )[:permutation_records]
    stability_counts: Counter[str] = Counter()
    for row in permutation_candidates:
        structure_id = str(row["r0_structure_id"])
        molecule = Chem.MolFromSmiles(str(row["canonical_constitutional_smiles"]))
        order = _nonidentity_permutation(molecule.GetNumAtoms(), _stable_seed(seed, structure_id))
        permuted = Chem.RenumberAtoms(molecule, order)
        for variant in SKELETON_VARIANTS:
            original = encode_lipid_support_skeleton(
                molecule,
                structure_id=structure_id,
                variant=variant,
            )
            shuffled = encode_lipid_support_skeleton(
                permuted,
                structure_id=f"{structure_id}-permuted",
                variant=variant,
            )
            stability_counts[f"{variant}:tested"] += 1
            stability_counts[f"{variant}:stable"] += int(
                skeleton_signature(original) == skeleton_signature(shuffled)
            )

    origin_summary, origin_ledger = _origin_metrics(ugi_products_path, ugi_atoms_path)
    broad_summary = _aggregate(broad_rows)
    ugi_summary = _aggregate(ugi_rows)
    selected = FUNCTIONAL_SUPPORT
    gates = {
        "all_broad_roundtrips_exact": broad_summary[selected]["exact_roundtrip_fraction"] == 1.0,
        "all_ugi_roundtrips_exact": ugi_summary[selected]["exact_roundtrip_fraction"] == 1.0,
        "all_broad_support_skeletons_connected": broad_summary[selected]["connected_fraction"]
        == 1.0,
        "all_ugi_support_skeletons_connected": ugi_summary[selected]["connected_fraction"] == 1.0,
        "all_broad_cycle_ranks_preserved": broad_summary[selected]["cycle_rank_preserved_fraction"]
        == 1.0,
        "all_ugi_cycle_ranks_preserved": ugi_summary[selected]["cycle_rank_preserved_fraction"]
        == 1.0,
        "permutation_stable": all(
            stability_counts[f"{variant}:stable"] == stability_counts[f"{variant}:tested"]
            for variant in SKELETON_VARIANTS
        ),
        "ugi_core_fully_retained": origin_summary["core_fully_retained_fraction"] == 1.0,
        "ugi_two_tail_origins_connected": origin_summary["two_tail_origins_connected_fraction"]
        == 1.0,
    }
    result = {
        "schema_version": "phase1_lipid_support_skeleton_audit.v1",
        "status": "pass" if all(gates.values()) else "fail",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rdkit_version": rdBase.rdkitVersion,
        "elapsed_seconds": time.monotonic() - started,
        "inputs": {
            label: {"path": str(path), "sha256": sha256_file(path)} for label, path in paths.items()
        },
        "policy": {
            "selected_candidate": selected,
            "functional_support_definition": (
                "all carbon; all nitrogen and phosphorus; all ring, charged, protected, "
                "and heavy-degree-at-least-two atoms; terminal neutral non-carbon "
                "decorations are generated during chemistry realization"
            ),
            "carbon_induced_is_a_negative_control": True,
            "no_fragment_vocabulary": True,
            "chemistry_realization_remains_atom_level": True,
            "permutation_records": permutation_records,
            "seed": seed,
        },
        "corpora": {
            "R0": {
                "molecules": len(broad_rows) // len(SKELETON_VARIANTS),
                "by_variant": broad_summary,
            },
            "Ugi_L1": {
                "molecules": len(ugi_rows) // len(SKELETON_VARIANTS),
                "by_variant": ugi_summary,
            },
            "Ugi_semantic_origins": origin_summary,
        },
        "permutation_stability": {
            variant: {
                "tested": stability_counts[f"{variant}:tested"],
                "stable": stability_counts[f"{variant}:stable"],
            }
            for variant in SKELETON_VARIANTS
        },
        "gates": gates,
        "interpretation": {
            "full_heavy": (
                "exact molecular graph baseline; terminal heteroatoms can inflate apparent "
                "morphology leaves and junctions"
            ),
            "carbon_induced": (
                "diagnostic only; disconnection quantifies why a literal carbon skeleton "
                "cannot represent ester, ether, amine, and heterocycle connectivity"
            ),
            "functional_support": (
                "candidate generated morphology graph; connector chemistry is retained and "
                "terminal decorations remain explicit conditional atom-level targets"
            ),
        },
    }
    return result, [*broad_rows, *ugi_rows], origin_ledger


def write_audit_artifacts(
    output_path: Path,
    ledger_path: Path,
    origin_ledger_path: Path,
    result: Mapping[str, Any],
    ledger: Sequence[Mapping[str, Any]],
    origin_ledger: Sequence[Mapping[str, Any]],
) -> None:
    """Atomically write the hash-pinnable decision artifact and ledgers."""

    _atomic_write(
        output_path,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    for path, rows in ((ledger_path, ledger), (origin_ledger_path, origin_ledger)):
        if not rows:
            raise LipidSupportSkeletonAuditError(f"cannot write empty ledger: {path}")
        import io

        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        _atomic_write(path, gzip.compress(buffer.getvalue().encode(), mtime=0))
