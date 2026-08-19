"""Build a provenance-stratified chemistry corpus from expanded Ugi enumeration."""

from __future__ import annotations

import csv
import gzip
import hashlib
import heapq
import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.bio.ugi_semantic_annotations import (
    ATOM_FIELDS,
    BOND_FIELDS,
    COMPONENT_MAPPING_FIELDS,
    PRODUCT_FIELDS,
    ROLE_NAMES,
    annotate_qualified_ugi_product,
)
from forge.data.r0_splits import sha256_bytes, sha256_file
from forge.product.ugi_expanded_enumeration import PRODUCT_FIELDS as ENUMERATED_PRODUCT_FIELDS
from forge.product.ugi_expanded_exemplars import (
    _atomic_write,
    _gzip_csv,
    _read_csv,
    _select_fold_clean_chemistry_exemplars,
)
from forge.route.qualified_forward import load_qualified_forward_reaction


class UgiBalancedChemistryCorpusError(RuntimeError):
    """Raised when balanced selection or exact annotation violates policy."""


def _resolve_inputs(
    config: Mapping[str, Any], repo: Path
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    paths = {}
    records = {}
    for label, value in config["inputs"].items():
        path = Path(value["path"])
        if not path.is_absolute():
            path = repo / path
        observed = sha256_file(path)
        if observed != value["sha256"]:
            raise UgiBalancedChemistryCorpusError(f"{label} hash changed")
        paths[label] = path
        records[label] = {
            "path": str(path.relative_to(repo)),
            "sha256": observed,
            "bytes": path.stat().st_size,
        }
    return paths, records


def _priority(seed: int, product_id: str, weight: float) -> float:
    digest = hashlib.sha256(f"{seed}|{product_id}".encode()).digest()
    integer = int.from_bytes(digest[:8], "big")
    uniform = (integer + 1) / (2**64 + 1)
    return -math.log(uniform) / weight


def _select_rows(
    products_path: Path,
    components: Mapping[tuple[str, str], Mapping[str, str]],
    *,
    seed: int,
    expanded_quota_by_fold: Mapping[str, int],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    mandatory, _, _ = _select_fold_clean_chemistry_exemplars(products_path, components)
    # Current-union rows are included wholesale below.  Only mandatory rows
    # drawn from the expanded stratum consume the additional expanded quota.
    mandatory_by_id = {
        row["product_id"]: row
        for row in mandatory
        if row["source_stratum"] != "current_phase1_union"
    }
    mandatory_counts = Counter(
        row["primary_product_fold"] for row in mandatory_by_id.values()
    )
    capacities = {
        fold: int(expanded_quota_by_fold[fold]) - mandatory_counts[fold]
        for fold in ("train", "calibration", "heldout")
    }
    if any(value < 0 for value in capacities.values()):
        raise UgiBalancedChemistryCorpusError("expanded quota is below mandatory coverage")
    reservoirs: dict[str, list[tuple[float, str, dict[str, str]]]] = {
        fold: [] for fold in capacities
    }
    current_rows: list[dict[str, str]] = []
    census = Counter()
    weighted_total = Counter()
    weighted_triple = Counter()
    with gzip.open(products_path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ENUMERATED_PRODUCT_FIELDS:
            raise UgiBalancedChemistryCorpusError("expanded product columns changed")
        for source in reader:
            row = dict(source)
            fold = row["primary_product_fold"]
            stratum = row["source_stratum"]
            census[(fold, stratum)] += 1
            weight = float(row["family_balance_weight_raw"])
            if not math.isfinite(weight) or weight <= 0:
                raise UgiBalancedChemistryCorpusError("invalid family-balance weight")
            weighted_total[fold] += weight
            weighted_triple[fold] += weight * ("#" in row["canonical_product_smiles"])
            if stratum == "current_phase1_union":
                current_rows.append(row)
                continue
            if row["product_id"] in mandatory_by_id:
                continue
            capacity = capacities[fold]
            if capacity == 0:
                continue
            priority = _priority(seed, row["product_id"], weight)
            item = (-priority, row["product_id"], row)
            heap = reservoirs[fold]
            if len(heap) < capacity:
                heapq.heappush(heap, item)
            elif priority < -heap[0][0]:
                heapq.heapreplace(heap, item)
    selected = [*current_rows, *mandatory_by_id.values()]
    for heap in reservoirs.values():
        selected.extend(item[2] for item in heap)
    selected_by_id = {row["product_id"]: row for row in selected}
    if len(selected_by_id) != len(selected):
        raise UgiBalancedChemistryCorpusError("balanced selection contains duplicate products")
    selected = sorted(
        selected,
        key=lambda row: (
            ("train", "calibration", "heldout").index(row["primary_product_fold"]),
            row["source_stratum"],
            row["product_id"],
        ),
    )
    covered = {
        (role, row[f"{role}_smiles"])
        for row in selected
        for role in ROLE_NAMES
        if components[(role, row[f"{role}_smiles"])] ["family_fold"]
        == row["primary_product_fold"]
    }
    if covered != set(components):
        raise UgiBalancedChemistryCorpusError("balanced selection lost component-fold coverage")
    selected_counts = Counter(
        (row["primary_product_fold"], row["source_stratum"]) for row in selected
    )
    return selected, {
        "full_census_by_fold_and_stratum": {
            f"{fold}|{stratum}": count
            for (fold, stratum), count in sorted(census.items())
        },
        "selected_by_fold_and_stratum": {
            f"{fold}|{stratum}": count
            for (fold, stratum), count in sorted(selected_counts.items())
        },
        "mandatory_component_cover_products": len(mandatory_by_id),
        "full_weighted_triple_fraction_by_fold": {
            fold: weighted_triple[fold] / weighted_total[fold]
            for fold in ("train", "calibration", "heldout")
        },
    }


def build_balanced_ugi_chemistry_corpus(
    config_path: Path,
    repo: Path,
) -> dict[str, Any]:
    """Select and exactly annotate the production Ugi chemistry mixture."""

    config = json.loads(config_path.read_text())
    if config.get("schema_version") != "phase1_ugi_balanced_chemistry_corpus_config.v1":
        raise UgiBalancedChemistryCorpusError("unsupported balanced chemistry config")
    policy = config["policy"]
    if policy.get("uniform_product_row_sampling_allowed") is not False:
        raise UgiBalancedChemistryCorpusError("uniform enumeration sampling remains prohibited")
    if policy.get("source_activity_labels_inherited") is not False:
        raise UgiBalancedChemistryCorpusError("enumerated products cannot inherit activity")
    paths, inputs = _resolve_inputs(config, repo)
    registry = [
        row
        for row in _read_csv(paths["component_registry"])
        if row["l1_structural_admission"] == "true"
    ]
    components = {(row["role"], row["canonical_smiles"]): row for row in registry}
    if len(components) != len(registry):
        raise UgiBalancedChemistryCorpusError("component registry identities are not unique")
    selected, selection = _select_rows(
        paths["products"],
        components,
        seed=int(config["seed"]),
        expanded_quota_by_fold=config["expanded_quota_by_fold"],
    )
    compiled = load_qualified_forward_reaction(
        paths["qualified_reactions"],
        paths["ugi_variant"],
        reaction_id=config["reaction_id"],
    )
    product_rows = []
    atom_rows = []
    bond_rows = []
    mapping_rows = []
    for row in selected:
        product, atoms, bonds, mappings = annotate_qualified_ugi_product(
            compiled,
            product_id=row["product_id"],
            target_smiles=row["canonical_product_smiles"],
            component_smiles_by_role={role: row[f"{role}_smiles"] for role in ROLE_NAMES},
            source_evidence_record_id=(
                "current_phase1_union"
                if row["source_stratum"] == "current_phase1_union"
                else "expanded_family_weighted_enumeration"
            ),
            max_outcomes=int(config["maximum_forward_outcomes"]),
        )
        product_rows.append(product)
        atom_rows.extend(atoms)
        bond_rows.extend(bonds)
        mapping_rows.extend(mappings)
    output_dir = repo / config["outputs"]["directory"]
    payloads = {
        "assignments.csv.gz": _gzip_csv(selected, ENUMERATED_PRODUCT_FIELDS),
        "semantic_products.csv.gz": _gzip_csv(product_rows, PRODUCT_FIELDS),
        "semantic_atoms.csv.gz": _gzip_csv(atom_rows, ATOM_FIELDS),
        "semantic_bonds.csv.gz": _gzip_csv(bond_rows, BOND_FIELDS),
        "semantic_component_mappings.csv.gz": _gzip_csv(
            mapping_rows, COMPONENT_MAPPING_FIELDS
        ),
    }
    artifacts = {}
    for name, payload in payloads.items():
        path = output_dir / name
        _atomic_write(path, payload)
        artifacts[name] = {
            "path": str(path.relative_to(repo)),
            "sha256": sha256_bytes(payload),
            "bytes": len(payload),
        }
    fold_counts = Counter(row["primary_product_fold"] for row in selected)
    source_counts = Counter(row["source_stratum"] for row in selected)
    result = {
        "schema_version": "phase1_ugi_balanced_chemistry_corpus_result.v1",
        "status": "pass",
        "inputs": inputs,
        "policy": policy,
        "selection": selection,
        "summary": {
            "selected_products": len(selected),
            "fold_counts": dict(fold_counts),
            "source_counts": dict(source_counts),
            "semantic_atom_rows": len(atom_rows),
            "semantic_bond_rows": len(bond_rows),
        },
        "artifacts": artifacts,
    }
    _atomic_write(
        output_dir / "result.json",
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    return result
