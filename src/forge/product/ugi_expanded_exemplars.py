"""Materialize one exact semantic exemplar per expanded Ugi component."""

from __future__ import annotations

import csv
import gzip
import io
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.core.io import atomic_write as _atomic_write
from forge.core.io import read_json_object
from forge.data.r0_splits import sha256_bytes, sha256_file
from forge.potency.ugi_semantic_annotations import (
    ATOM_FIELDS,
    BOND_FIELDS,
    COMPONENT_MAPPING_FIELDS,
    PRODUCT_FIELDS,
    ROLE_NAMES,
    UgiSemanticAnnotationError,
    annotate_qualified_ugi_product,
)
from forge.product.ugi_expanded_enumeration import PRODUCT_FIELDS as ENUMERATED_PRODUCT_FIELDS
from forge.route.qualified_forward import load_qualified_forward_reaction

CONFIG_SCHEMA_VERSION = "phase1_ugi_expanded_exemplars_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_expanded_exemplars_result.v1"
CHEMISTRY_CONFIG_SCHEMA_VERSION = "phase1_ugi_expanded_chemistry_exemplars_config.v1"
CHEMISTRY_RESULT_SCHEMA_VERSION = "phase1_ugi_expanded_chemistry_exemplars_result.v1"
LEDGER_FIELDS = (
    "role",
    "component_id",
    "component_smiles",
    "family_id",
    "family_fold",
    "exemplar_product_id",
    "exemplar_product_smiles",
    "exemplar_source_stratum",
)


class ExpandedExemplarError(ValueError):
    """Raised when compact semantic coverage is incomplete or inconsistent."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    return read_json_object(path, error=ExpandedExemplarError, label=label)


def _resolve_inputs(
    config: Mapping[str, Any], repo: Path
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    paths: dict[str, Path] = {}
    records: dict[str, dict[str, Any]] = {}
    for label, specification in sorted(config["inputs"].items()):
        if not isinstance(specification, dict) or set(specification) != {"path", "sha256"}:
            raise ExpandedExemplarError(f"input {label!r} must define path and sha256")
        path = Path(str(specification["path"]))
        if not path.is_absolute():
            path = repo / path
        observed = sha256_file(path)
        if observed != specification["sha256"]:
            raise ExpandedExemplarError(
                f"input {label!r} SHA-256 mismatch: expected {specification['sha256']}, "
                f"observed {observed}"
            )
        paths[label] = path
        records[label] = {
            "path": str(path.relative_to(repo)),
            "bytes": path.stat().st_size,
            "sha256": observed,
        }
    return paths, records


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _select_exemplars(
    products_path: Path,
    components: Mapping[tuple[str, str], Mapping[str, str]],
) -> tuple[list[dict[str, str]], dict[tuple[str, str], dict[str, str]], int]:
    unseen = set(components)
    selected: list[dict[str, str]] = []
    exemplar_by_component: dict[tuple[str, str], dict[str, str]] = {}
    scanned = 0
    with gzip.open(products_path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scanned += 1
            local = {(role, row[f"{role}_smiles"]) for role in ROLE_NAMES}
            newly_covered = local.intersection(unseen)
            if not newly_covered:
                continue
            selected.append(dict(row))
            for key in newly_covered:
                exemplar_by_component[key] = dict(row)
            unseen.difference_update(newly_covered)
            if not unseen:
                break
    if unseen:
        counts = Counter(role for role, _ in unseen)
        raise ExpandedExemplarError(f"expanded products do not cover components: {dict(counts)}")
    return selected, exemplar_by_component, scanned


def _gzip_csv(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in fields})
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as archive:
        archive.write(text.getvalue().encode())
    return output.getvalue()


def build_expanded_ugi_exemplars(config_path: Path, repo: Path) -> dict[str, Any]:
    """Select a minimal cover and annotate it with the exact atom-mapped transform."""

    config = _load_json(config_path, "expanded-exemplar config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ExpandedExemplarError(
            f"unsupported expanded-exemplar schema {config.get('schema_version')!r}"
        )
    policy = config.get("policy")
    if not isinstance(policy, dict):
        raise ExpandedExemplarError("policy must be an object")
    if policy.get("source_activity_labels_inherited") is not False:
        raise ExpandedExemplarError("source activity labels must not be inherited")
    if policy.get("substructure_origin_inference_allowed") is not False:
        raise ExpandedExemplarError("origin semantics must come from the forward transform")
    paths, input_records = _resolve_inputs(config, repo)
    registry_rows = [
        row
        for row in _read_csv(paths["component_registry"])
        if row["l1_structural_admission"] == "true"
    ]
    components = {(row["role"], row["canonical_smiles"]): row for row in registry_rows}
    if len(components) != len(registry_rows):
        raise ExpandedExemplarError("admitted component registry contains duplicate identities")
    selected, exemplar_by_component, scanned = _select_exemplars(paths["products"], components)
    compiled = load_qualified_forward_reaction(
        paths["qualified_reactions"],
        paths["ugi_variant"],
        reaction_id=str(config["reaction_id"]),
    )
    product_rows: list[dict[str, Any]] = []
    atom_rows: list[dict[str, Any]] = []
    bond_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    product_ids: set[str] = set()
    for row in selected:
        product_id = row["product_id"]
        if product_id in product_ids:
            raise ExpandedExemplarError(f"duplicate selected product ID: {product_id}")
        components_by_role = {role: row[f"{role}_smiles"] for role in ROLE_NAMES}
        try:
            product, atoms, bonds, mappings = annotate_qualified_ugi_product(
                compiled,
                product_id=product_id,
                target_smiles=row["canonical_product_smiles"],
                component_smiles_by_role=components_by_role,
                source_evidence_record_id="expanded_exact_forward_semantic_exemplar",
                max_outcomes=int(config["maximum_forward_outcomes"]),
            )
        except UgiSemanticAnnotationError as exc:
            raise ExpandedExemplarError(f"semantic annotation failed: {product_id}") from exc
        product_ids.add(product_id)
        product_rows.append(product)
        atom_rows.extend(atoms)
        bond_rows.extend(bonds)
        mapping_rows.extend(mappings)

    ledger_rows = []
    for key, component in sorted(components.items()):
        exemplar = exemplar_by_component[key]
        ledger_rows.append(
            {
                "role": key[0],
                "component_id": component["component_id"],
                "component_smiles": key[1],
                "family_id": component["family_id"],
                "family_fold": component["family_fold"],
                "exemplar_product_id": exemplar["product_id"],
                "exemplar_product_smiles": exemplar["canonical_product_smiles"],
                "exemplar_source_stratum": exemplar["source_stratum"],
            }
        )
    expected_component_counts = dict(Counter(row["role"] for row in registry_rows))
    mapped_component_counts = dict(Counter(row["role"] for row in ledger_rows))
    gates = {
        "all_components_covered": len(ledger_rows) == len(registry_rows),
        "component_counts_exact": mapped_component_counts == expected_component_counts,
        "all_selected_products_annotated": len(product_rows) == len(selected),
        "all_semantic_signatures_unique": all(
            int(row["semantic_signature_multiplicity"]) == 1 for row in product_rows
        ),
        "all_component_origin_regions_connected": all(
            int(row["component_mapping_rows"]) > 0 for row in product_rows
        ),
    }
    payloads = {
        "semantic_products.csv.gz": _gzip_csv(product_rows, PRODUCT_FIELDS),
        "semantic_atoms.csv.gz": _gzip_csv(atom_rows, ATOM_FIELDS),
        "semantic_bonds.csv.gz": _gzip_csv(bond_rows, BOND_FIELDS),
        "semantic_component_mappings.csv.gz": _gzip_csv(mapping_rows, COMPONENT_MAPPING_FIELDS),
        "component_exemplar_ledger.csv.gz": _gzip_csv(ledger_rows, LEDGER_FIELDS),
    }
    output_dir = repo / config["outputs"]["directory"]
    artifacts = {}
    for name, payload in payloads.items():
        path = output_dir / name
        _atomic_write(path, payload)
        artifacts[name] = {
            "path": str(path.relative_to(repo)),
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
        }
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "pass" if all(gates.values()) else "fail",
        "task": config["task"],
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "inputs": input_records,
        "policy": policy,
        "summary": {
            "admitted_components": len(registry_rows),
            "component_counts": expected_component_counts,
            "selected_exemplar_products": len(selected),
            "product_rows_scanned_until_complete_cover": scanned,
            "semantic_atom_rows": len(atom_rows),
            "semantic_bond_rows": len(bond_rows),
            "semantic_component_mapping_rows": len(mapping_rows),
        },
        "gates": gates,
        "artifacts": artifacts,
        "randomness": {"used": False},
    }
    _atomic_write(
        output_dir / "result.json",
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    if result["status"] != "pass":
        raise ExpandedExemplarError("expanded semantic exemplar gates failed")
    return result


def _select_fold_clean_chemistry_exemplars(
    products_path: Path,
    components: Mapping[tuple[str, str], Mapping[str, str]],
) -> tuple[list[dict[str, str]], dict[str, int], dict[str, int]]:
    """Cover every component in its own frozen fold without training leakage."""

    unseen = {
        fold: {key for key, row in components.items() if row["family_fold"] == fold}
        for fold in ("train", "calibration", "heldout")
    }
    selected: list[dict[str, str]] = []
    selected_ids: set[str] = set()
    scanned_by_fold = Counter()
    with gzip.open(products_path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ENUMERATED_PRODUCT_FIELDS:
            raise ExpandedExemplarError("expanded product columns changed")
        for row in reader:
            fold = row["primary_product_fold"]
            if fold not in unseen or not unseen[fold]:
                continue
            scanned_by_fold[fold] += 1
            local = {(role, row[f"{role}_smiles"]) for role in ROLE_NAMES}
            newly_covered = local.intersection(unseen[fold])
            if not newly_covered:
                continue
            if row["product_id"] not in selected_ids:
                selected.append(dict(row))
                selected_ids.add(row["product_id"])
            unseen[fold].difference_update(newly_covered)
            if all(not values for values in unseen.values()):
                break
    if any(unseen.values()):
        missing = {
            fold: dict(Counter(role for role, _ in values))
            for fold, values in unseen.items()
            if values
        }
        raise ExpandedExemplarError(f"fold-clean chemistry cover is incomplete: {missing}")
    return (
        selected,
        dict(Counter(row["primary_product_fold"] for row in selected)),
        dict(scanned_by_fold),
    )


def build_expanded_ugi_chemistry_exemplars(
    config_path: Path,
    repo: Path,
) -> dict[str, Any]:
    """Build a compact fold-clean product set covering all admitted chemistry."""

    config = _load_json(config_path, "expanded-chemistry-exemplar config")
    if config.get("schema_version") != CHEMISTRY_CONFIG_SCHEMA_VERSION:
        raise ExpandedExemplarError(
            f"unsupported expanded-chemistry-exemplar schema {config.get('schema_version')!r}"
        )
    policy = config.get("policy")
    if not isinstance(policy, dict):
        raise ExpandedExemplarError("expanded chemistry policy must be an object")
    if policy.get("uniform_product_row_sampling_allowed") is not False:
        raise ExpandedExemplarError("uniform product-row sampling must remain prohibited")
    if policy.get("source_activity_labels_inherited") is not False:
        raise ExpandedExemplarError("source activity labels must not be inherited")
    paths, input_records = _resolve_inputs(config, repo)
    registry_rows = [
        row
        for row in _read_csv(paths["component_registry"])
        if row["l1_structural_admission"] == "true"
    ]
    components = {(row["role"], row["canonical_smiles"]): row for row in registry_rows}
    if len(components) != len(registry_rows):
        raise ExpandedExemplarError("admitted component registry contains duplicate identities")
    selected, selected_counts, scanned_counts = _select_fold_clean_chemistry_exemplars(
        paths["products"],
        components,
    )
    compiled = load_qualified_forward_reaction(
        paths["qualified_reactions"],
        paths["ugi_variant"],
        reaction_id=str(config["reaction_id"]),
    )
    product_rows: list[dict[str, Any]] = []
    atom_rows: list[dict[str, Any]] = []
    bond_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    for row in selected:
        components_by_role = {role: row[f"{role}_smiles"] for role in ROLE_NAMES}
        try:
            product, atoms, bonds, mappings = annotate_qualified_ugi_product(
                compiled,
                product_id=row["product_id"],
                target_smiles=row["canonical_product_smiles"],
                component_smiles_by_role=components_by_role,
                source_evidence_record_id="expanded_fold_clean_chemistry_exemplar",
                max_outcomes=int(config["maximum_forward_outcomes"]),
            )
        except UgiSemanticAnnotationError as exc:
            raise ExpandedExemplarError(
                f"chemistry semantic annotation failed: {row['product_id']}"
            ) from exc
        product_rows.append(product)
        atom_rows.extend(atoms)
        bond_rows.extend(bonds)
        mapping_rows.extend(mappings)

    covered = {
        (role, row[f"{role}_smiles"])
        for row in selected
        for role in ROLE_NAMES
        if components[(role, row[f"{role}_smiles"])]["family_fold"] == row["primary_product_fold"]
    }
    gates = {
        "all_components_covered_in_own_fold": covered == set(components),
        "all_selected_products_annotated": len(product_rows) == len(selected),
        "all_semantic_signatures_unique": all(
            int(row["semantic_signature_multiplicity"]) == 1 for row in product_rows
        ),
        "no_train_product_contains_nontrain_family": all(
            row["primary_product_fold"] != "train"
            or all(row[f"{role}_family_fold"] == "train" for role in ROLE_NAMES)
            for row in selected
        ),
    }
    payloads = {
        "assignments.csv.gz": _gzip_csv(selected, ENUMERATED_PRODUCT_FIELDS),
        "semantic_products.csv.gz": _gzip_csv(product_rows, PRODUCT_FIELDS),
        "semantic_atoms.csv.gz": _gzip_csv(atom_rows, ATOM_FIELDS),
        "semantic_bonds.csv.gz": _gzip_csv(bond_rows, BOND_FIELDS),
        "semantic_component_mappings.csv.gz": _gzip_csv(
            mapping_rows,
            COMPONENT_MAPPING_FIELDS,
        ),
    }
    output_dir = repo / config["outputs"]["directory"]
    artifacts = {}
    for name, payload in payloads.items():
        path = output_dir / name
        _atomic_write(path, payload)
        artifacts[name] = {
            "path": str(path.relative_to(repo)),
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
        }
    result = {
        "schema_version": CHEMISTRY_RESULT_SCHEMA_VERSION,
        "status": "pass" if all(gates.values()) else "fail",
        "task": config["task"],
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "inputs": input_records,
        "policy": policy,
        "summary": {
            "admitted_components": len(components),
            "selected_products": len(selected),
            "selected_products_by_fold": selected_counts,
            "product_rows_considered_by_fold": scanned_counts,
            "semantic_atom_rows": len(atom_rows),
            "semantic_bond_rows": len(bond_rows),
            "semantic_component_mapping_rows": len(mapping_rows),
        },
        "gates": gates,
        "artifacts": artifacts,
        "randomness": {"used": False},
    }
    _atomic_write(
        output_dir / "result.json",
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    if result["status"] != "pass":
        raise ExpandedExemplarError("expanded chemistry exemplar gates failed")
    return result
