"""Exact Ugi atom-origin annotations for the complete Phase 1 L1 union."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdkit import rdBase

from forge.core.io import atomic_write as _atomic_write
from forge.potency.ugi_semantic_annotations import (
    ATOM_FIELDS,
    BOND_FIELDS,
    COMPONENT_MAPPING_FIELDS,
    PRODUCT_FIELDS,
    ROLE_NAMES,
    UgiSemanticAnnotationError,
    annotate_qualified_ugi_product,
)
from forge.product.defog_feasibility import sha256_file
from forge.route.qualified_forward import load_qualified_forward_reaction

RESULT_SCHEMA_VERSION = "phase1_ugi_l1_semantics.v1"
_IMPLEMENTATION_PATHS = {
    "phase1_ugi_l1_semantics": Path(__file__).resolve(),
    "qualified_ugi_semantics": Path(__file__)
    .parents[1]
    .joinpath("bio", "ugi_semantic_annotations.py")
    .resolve(),
}


class Phase1UgiOriginError(RuntimeError):
    """Raised when complete-union Ugi origin supervision is not exact."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise Phase1UgiOriginError(f"invalid Ugi L1 semantic config: {path}") from exc
    if config.get("schema_version") != "phase1_ugi_l1_semantics_config.v1":
        raise Phase1UgiOriginError("Ugi L1 semantic config schema changed")
    return config


def _verify_inputs(config: Mapping[str, Any], repo: Path) -> dict[str, Path]:
    resolved = {}
    for label, specification in config["inputs"].items():
        path = repo / str(specification["path"])
        if not path.is_file() or sha256_file(path) != str(specification["sha256"]):
            raise Phase1UgiOriginError(f"hash-qualified input changed: {label}: {path}")
        resolved[label] = path
    return resolved


def _normalized_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _semantic_rows_signature(
    rows: Sequence[Mapping[str, Any]],
    *,
    excluded: set[str],
) -> tuple[tuple[tuple[str, str], ...], ...]:
    return tuple(
        sorted(
            tuple(
                (key, _normalized_value(value)) for key, value in row.items() if key not in excluded
            )
            for row in rows
        )
    )


def _gzip_csv(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return gzip.compress(buffer.getvalue().encode(), mtime=0)


def _artifact(payload: bytes) -> dict[str, Any]:
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _measured_index(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    products = _read_csv(paths["measured_products"])
    atoms: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    bonds: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    mappings: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(paths["measured_atoms"]):
        atoms[row["product_id"]].append(row)
    for row in _read_csv(paths["measured_bonds"]):
        bonds[row["product_id"]].append(row)
    for row in _read_csv(paths["measured_component_mappings"]):
        mappings[row["product_id"]].append(row)
    result = {}
    for product in products:
        product_id = product["product_id"]
        result[product["product_smiles"]] = {
            "product": product,
            "atoms": atoms[product_id],
            "bonds": bonds[product_id],
            "mappings": mappings[product_id],
        }
    return result


def _assert_measured_match(
    target_smiles: str,
    product: Mapping[str, Any],
    atoms: Sequence[Mapping[str, Any]],
    bonds: Sequence[Mapping[str, Any]],
    mappings: Sequence[Mapping[str, Any]],
    measured: Mapping[str, Mapping[str, Any]],
) -> None:
    reference = measured.get(target_smiles)
    if reference is None:
        raise Phase1UgiOriginError(f"measured semantic product missing: {target_smiles}")
    product_excluded = {"product_id", "source_evidence_record_id", "annotation_sha256"}
    if _semantic_rows_signature([product], excluded=product_excluded) != _semantic_rows_signature(
        [reference["product"]], excluded=product_excluded
    ):
        raise Phase1UgiOriginError("measured product semantic fields changed")
    for label, observed, expected in (
        ("atom", atoms, reference["atoms"]),
        ("bond", bonds, reference["bonds"]),
        ("component mapping", mappings, reference["mappings"]),
    ):
        if _semantic_rows_signature(observed, excluded={"product_id"}) != _semantic_rows_signature(
            expected, excluded={"product_id"}
        ):
            raise Phase1UgiOriginError(f"measured {label} semantics changed")


def build_phase1_ugi_l1_semantics(
    config_path: Path,
    output_dir: Path,
    repo: Path,
) -> dict[str, Any]:
    """Annotate every unique Phase 1 Ugi product through the qualified transform."""

    config = _load_config(config_path)
    paths = _verify_inputs(config, repo)
    assignments = _read_csv(paths["ugi_l1_assignments"])
    expected = config["expected"]
    if len(assignments) != int(expected["unique_products"]):
        raise Phase1UgiOriginError("Ugi L1 union row count changed")
    products = [row["canonical_product_smiles"] for row in assignments]
    if len(set(products)) != len(products):
        raise Phase1UgiOriginError("Ugi L1 union contains duplicate products")

    compiled = load_qualified_forward_reaction(
        paths["qualified_reactions"],
        paths["ugi_variant"],
        reaction_id=str(config["reaction_id"]),
    )
    if tuple(compiled.role_names) != ROLE_NAMES:
        raise Phase1UgiOriginError("qualified Ugi role order changed")
    measured = _measured_index(paths)
    measured_matches = 0
    product_rows: list[dict[str, Any]] = []
    atom_rows: list[dict[str, Any]] = []
    bond_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    origin_counts: Counter[str] = Counter()
    core_origin_counts: Counter[str] = Counter()
    core_position_origin_counts: Counter[tuple[str, str]] = Counter()
    role_anchor_origin_counts: Counter[tuple[str, str]] = Counter()
    role_anchor_core_counts: Counter[str] = Counter()
    connected_precursor_origin_products = 0
    component_sets = {role: set() for role in ROLE_NAMES}

    for row in sorted(assignments, key=lambda item: item["product_id"]):
        product_id = row["product_id"]
        is_measured = str(row["is_source_adjudicated_measured_product"]).lower() == "true"
        components = {role: row[f"{role}_smiles"] for role in ROLE_NAMES}
        for role, smiles in components.items():
            component_sets[role].add(smiles)
        try:
            product, atoms, bonds, mappings = annotate_qualified_ugi_product(
                compiled,
                product_id=product_id,
                target_smiles=row["canonical_product_smiles"],
                component_smiles_by_role=components,
                source_evidence_record_id=(
                    "source_adjudicated_measured"
                    if is_measured
                    else "transform_consistency_virtual"
                ),
                max_outcomes=int(config["max_forward_outcomes_per_product"]),
            )
        except UgiSemanticAnnotationError as exc:
            raise Phase1UgiOriginError(f"semantic annotation failed: {product_id}") from exc
        if is_measured:
            _assert_measured_match(
                row["canonical_product_smiles"],
                product,
                atoms,
                bonds,
                mappings,
                measured,
            )
            measured_matches += 1
        product_rows.append(product)
        atom_rows.extend(atoms)
        bond_rows.extend(bonds)
        mapping_rows.extend(mappings)
        origin_counts.update(atom["origin_role"] for atom in atoms)
        connected_precursor_origin_products += 1
        atom_by_index = {int(atom["product_atom_index"]): atom for atom in atoms}
        for atom in atoms:
            if bool(atom["is_ugi_core"]):
                core_origin_counts[str(atom["origin_role"])] += 1
                core_position_origin_counts[
                    (str(atom["core_position"]), str(atom["origin_role"]))
                ] += 1
        for role, anchor_index in json.loads(product["role_anchor_indices_json"]).items():
            anchor = atom_by_index[int(anchor_index)]
            role_anchor_origin_counts[(str(role), str(anchor["origin_role"]))] += 1
            if bool(anchor["is_ugi_core"]):
                role_anchor_core_counts[str(role)] += 1

    measured_count = sum(
        str(row["is_source_adjudicated_measured_product"]).lower() == "true" for row in assignments
    )
    virtual_only = len(assignments) - measured_count
    core_counts = Counter(row["product_id"] for row in atom_rows if bool(row["is_ugi_core"]))
    expected_core_position_counts = Counter(
        {
            (str(position), str(origin)): len(product_rows)
            for position, origin in expected["core_position_origins"].items()
        }
    )
    expected_anchor_counts = Counter({(role, role): len(product_rows) for role in ROLE_NAMES})
    gates = {
        "all_products_annotated": len(product_rows) == int(expected["unique_products"]),
        "measured_count_exact": measured_count == int(expected["measured_products"]),
        "virtual_only_count_exact": virtual_only == int(expected["virtual_only_products"]),
        "all_measured_semantics_reproduced": measured_matches == measured_count,
        "all_products_have_exact_core": len(core_counts) == len(product_rows)
        and set(core_counts.values()) == {int(expected["core_atoms_per_product"])},
        "all_semantic_signatures_unique": all(
            int(row["semantic_signature_multiplicity"]) == 1 for row in product_rows
        ),
        "all_precursor_origin_regions_connected": (
            connected_precursor_origin_products == len(product_rows)
        ),
        "origin_and_core_annotations_orthogonal": (
            core_position_origin_counts == expected_core_position_counts
        ),
        "all_products_have_one_core_role_anchor_per_precursor": (
            role_anchor_origin_counts == expected_anchor_counts
            and role_anchor_core_counts == Counter({role: len(product_rows) for role in ROLE_NAMES})
        ),
        "component_counts_exact": {role: len(component_sets[role]) for role in ROLE_NAMES}
        == {role: int(expected["component_counts"][role]) for role in ROLE_NAMES},
    }
    payloads = {
        "ugi_l1_semantic_products.csv.gz": _gzip_csv(product_rows, PRODUCT_FIELDS),
        "ugi_l1_semantic_atoms.csv.gz": _gzip_csv(atom_rows, ATOM_FIELDS),
        "ugi_l1_semantic_bonds.csv.gz": _gzip_csv(bond_rows, BOND_FIELDS),
        "ugi_l1_semantic_component_mappings.csv.gz": _gzip_csv(
            mapping_rows, COMPONENT_MAPPING_FIELDS
        ),
    }
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "pass" if all(gates.values()) else "fail",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "software": {"rdkit": rdBase.rdkitVersion},
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "inputs": {
            label: {"path": str(path), "sha256": sha256_file(path)} for label, path in paths.items()
        },
        "implementation": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in sorted(_IMPLEMENTATION_PATHS.items())
        },
        "summary": {
            "products": len(product_rows),
            "measured_products": measured_count,
            "virtual_only_products": virtual_only,
            "product_atoms": len(atom_rows),
            "product_bonds": len(bond_rows),
            "component_mapping_rows": len(mapping_rows),
            "origin_atom_counts": dict(sorted(origin_counts.items())),
            "core_origin_atom_counts": dict(sorted(core_origin_counts.items())),
            "core_position_origin_counts": {
                f"{position}|{origin}": count
                for (position, origin), count in sorted(core_position_origin_counts.items())
            },
            "role_anchor_origin_counts": {
                f"{anchor_role}|{atom_origin}": count
                for (anchor_role, atom_origin), count in sorted(role_anchor_origin_counts.items())
            },
            "products_with_connected_precursor_origins": (connected_precursor_origin_products),
            "component_counts": {role: len(component_sets[role]) for role in ROLE_NAMES},
        },
        "gates": gates,
        "policy": config["policy"],
        "artifacts": {name: _artifact(payload) for name, payload in payloads.items()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        _atomic_write(output_dir / name, payload)
    result_payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    _atomic_write(output_dir / "ugi_l1_semantics_result.json", result_payload)
    if result["status"] != "pass":
        raise Phase1UgiOriginError("Ugi L1 semantic annotation gates failed")
    return result
