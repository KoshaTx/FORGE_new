"""Enumerate the family-split Phase 1 Ugi structural training corpus."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import platform
import sqlite3
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.data.r0_splits import sha256_file
from forge.data.r1_prime_audit import compile_reactions, load_reaction_definitions
from forge.product.ugi_component_expansion import FOLDS, ROLES

CONFIG_SCHEMA_VERSION = "phase1_ugi_expanded_enumeration_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_expanded_enumeration_result.v1"
ALGORITHM_VERSION = "exact_single_site_family_split_enumeration_v1"
PRODUCT_FIELDS = (
    "product_id",
    "canonical_product_smiles",
    "amine_head_smiles",
    "amine_head_family_id",
    "amine_head_family_fold",
    "oxoester_aldehyde_body_tail_smiles",
    "oxoester_aldehyde_body_tail_family_id",
    "oxoester_aldehyde_body_tail_family_fold",
    "isocyanide_tail_smiles",
    "isocyanide_tail_family_id",
    "isocyanide_tail_family_fold",
    "primary_product_fold",
    "source_stratum",
    "is_source_adjudicated_measured_product",
    "component_novelty_class",
    "family_balance_weight_raw",
)


class ExpandedEnumerationError(ValueError):
    """Raised when the exact expanded enumeration contract is violated."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ExpandedEnumerationError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ExpandedEnumerationError(f"{label} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExpandedEnumerationError(f"{label} must contain a JSON object")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise ExpandedEnumerationError(f"could not read {path}: {exc}") from exc


def _resolve_inputs(
    config: Mapping[str, Any], repo: Path
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    paths: dict[str, Path] = {}
    records: dict[str, dict[str, Any]] = {}
    for label, specification in sorted(config["inputs"].items()):
        if not isinstance(specification, dict) or set(specification) != {"path", "sha256"}:
            raise ExpandedEnumerationError(f"input {label!r} must define path and sha256")
        path = Path(str(specification["path"]))
        if not path.is_absolute():
            path = repo / path
        observed = sha256_file(path)
        if observed != specification["sha256"]:
            raise ExpandedEnumerationError(
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


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ExpandedEnumerationError(
            f"unsupported expanded-enumeration schema {config.get('schema_version')!r}"
        )
    policy = config.get("policy")
    if not isinstance(policy, dict) or policy.get("reaction_id") != "ugi_3cr_agile":
        raise ExpandedEnumerationError("policy must select ugi_3cr_agile")
    if policy.get("identity") != "canonical_constitutional_smiles":
        raise ExpandedEnumerationError("enumeration identity must remain constitutional")
    if policy.get("uniform_product_row_sampling_allowed") is not False:
        raise ExpandedEnumerationError("uniform product-row sampling must remain prohibited")
    if policy.get("source_activity_labels_inherited") is not False:
        raise ExpandedEnumerationError("source biological labels must not be inherited")
    if policy.get("enumerated_product_is_synthesis_success") is not False:
        raise ExpandedEnumerationError("enumerated products are not synthesis outcomes")
    if policy.get("enumeration_requires_one_symmetry_distinct_site_per_role") is not True:
        raise ExpandedEnumerationError("expanded enumeration requires one unambiguous site")
    if (
        not isinstance(policy.get("gzip_compresslevel"), int)
        or not 1 <= int(policy["gzip_compresslevel"]) <= 9
    ):
        raise ExpandedEnumerationError("gzip_compresslevel must be an integer from one to nine")


def _canonical_product(molecule: Chem.Mol) -> tuple[str, Chem.Mol] | None:
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(molecule)
        if len(Chem.GetMolFrags(molecule)) != 1:
            return None
        smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
        with rdBase.BlockLogs():
            normalized = Chem.MolFromSmiles(smiles)
        if normalized is None:
            return None
        return smiles, normalized
    except Exception:
        return None


def _product_fold(component_folds: Sequence[str]) -> str:
    if any(fold not in FOLDS for fold in component_folds):
        raise ExpandedEnumerationError(f"invalid component fold tuple: {component_folds}")
    if "heldout" in component_folds:
        return "heldout"
    if "calibration" in component_folds:
        return "calibration"
    return "train"


def _component_novelty(components: Sequence[Mapping[str, str]]) -> str:
    if all(component["is_current_catalog"] == "true" for component in components):
        return "familiar_components"
    transferred_markers = {
        "cross_platform_hydrophobic_transfer",
        "lnpdb_head_census",
        "cross_assembly_observed_component",
    }
    for component in components:
        classes = set(json.loads(component["source_classes_json"]))
        if classes.intersection(transferred_markers):
            return "transferred_known_component"
    return "bounded_structural_expansion_component"


def _product_record(
    product_smiles: str,
    components: Sequence[Mapping[str, str]],
    *,
    source_stratum: str,
    measured: bool,
) -> tuple[Any, ...]:
    folds = [component["family_fold"] for component in components]
    family_weight = 1.0
    for component in components:
        size = int(component["family_size"])
        if size <= 0:
            raise ExpandedEnumerationError("admitted component has nonpositive family size")
        family_weight /= size
    product_id = "EUGI-" + hashlib.sha256(product_smiles.encode()).hexdigest()[:20]
    values: dict[str, Any] = {
        "product_id": product_id,
        "canonical_product_smiles": product_smiles,
        "primary_product_fold": _product_fold(folds),
        "source_stratum": source_stratum,
        "is_source_adjudicated_measured_product": str(measured).lower(),
        "component_novelty_class": _component_novelty(components),
        "family_balance_weight_raw": f"{family_weight:.12g}",
    }
    for role, component in zip(ROLES, components, strict=True):
        values[f"{role}_smiles"] = component["canonical_smiles"]
        values[f"{role}_family_id"] = component["family_id"]
        values[f"{role}_family_fold"] = component["family_fold"]
    return tuple(values[field] for field in PRODUCT_FIELDS)


def _create_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    # The database is an ephemeral deterministic deduplication index.  It is
    # never published and is rebuilt after any failure, so journaling adds
    # latency without improving the artifact contract.
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    columns = ",".join(f'"{field}" TEXT NOT NULL' for field in PRODUCT_FIELDS)
    connection.execute(
        f'CREATE TABLE products ({columns}, PRIMARY KEY ("canonical_product_smiles")) '
        "WITHOUT ROWID"
    )
    return connection


def _insert_sql() -> str:
    columns = ",".join(f'"{field}"' for field in PRODUCT_FIELDS)
    placeholders = ",".join("?" for _ in PRODUCT_FIELDS)
    return f"INSERT OR IGNORE INTO products ({columns}) VALUES ({placeholders})"


def _export_database(
    connection: sqlite3.Connection, output_path: Path, compresslevel: int
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    os.close(descriptor)
    try:
        with open(temporary, "wb") as compressed_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=compressed_handle,
                compresslevel=compresslevel,
                mtime=0,
            ) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle, lineterminator="\n")
                    writer.writerow(PRODUCT_FIELDS)
                    query = (
                        "SELECT "
                        + ",".join(f'"{field}"' for field in PRODUCT_FIELDS)
                        + ' FROM products ORDER BY "canonical_product_smiles"'
                    )
                    for row in connection.execute(query):
                        writer.writerow(row)
        os.replace(temporary, output_path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return {
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
    }


def enumerate_expanded_ugi_corpus(config_path: Path, repo: Path) -> dict[str, Any]:
    """Enumerate all exact single-site products into a deterministic deduplicated corpus."""

    config = _load_json(config_path, "expanded-enumeration config")
    _validate_config(config)
    paths, input_records = _resolve_inputs(config, repo)
    policy = config["policy"]
    registry_rows = _read_csv(paths["component_registry"])
    admitted = [row for row in registry_rows if row["l1_structural_admission"] == "true"]
    by_role = {
        role: sorted(
            [row for row in admitted if row["role"] == role],
            key=lambda row: row["canonical_smiles"],
        )
        for role in ROLES
    }
    unambiguous = {
        role: [row for row in by_role[role] if row["symmetry_distinct_handle_sites"] == "1"]
        for role in ROLES
    }
    component_index = {(row["role"], row["canonical_smiles"]): row for row in admitted}
    if len(component_index) != len(admitted):
        raise ExpandedEnumerationError("component registry contains duplicate admitted identities")

    registry = _load_json(paths["qualified_reactions"], "reaction registry")
    definitions = load_reaction_definitions(
        [paths["qualified_reactions"]], expected_count=len(registry["reactions"])
    )
    compiled = {item.definition.reaction_id: item for item in compile_reactions(definitions)}
    reaction = compiled[policy["reaction_id"]]
    role_order = tuple(role.name for role in reaction.definition.reactant_roles)
    if role_order != ROLES:
        raise ExpandedEnumerationError(f"unexpected Ugi role order: {role_order}")
    molecules = {
        role: [Chem.MolFromSmiles(row["canonical_smiles"]) for row in unambiguous[role]]
        for role in ROLES
    }
    if any(molecule is None for values in molecules.values() for molecule in values):
        raise ExpandedEnumerationError("an admitted component no longer parses")

    output_path = repo / config["outputs"]["products"]
    result_path = repo / config["outputs"]["result"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ugi-enumeration-", dir=output_path.parent) as temp:
        connection = _create_database(Path(temp) / "products.sqlite")
        insert_sql = _insert_sql()
        current_rows = _read_csv(paths["current_assignments"])
        base_inserted = 0
        for row in current_rows:
            components = [component_index[(role, row[f"{role}_smiles"])] for role in ROLES]
            record = _product_record(
                row["canonical_product_smiles"],
                components,
                source_stratum="current_phase1_union",
                measured=row["is_source_adjudicated_measured_product"] == "true",
            )
            before = connection.total_changes
            connection.execute(insert_sql, record)
            base_inserted += connection.total_changes - before
        connection.commit()

        attempted = 0
        valid_unique_outcome = 0
        invalid_or_ambiguous = 0
        duplicate_products = 0
        batch: list[tuple[Any, ...]] = []
        maximum_outcomes = int(policy["maximum_forward_outcomes"])
        maximum_heavy_atoms = int(policy["maximum_heavy_atoms"])
        allowed_elements = set(policy["allowed_elements"])
        for head_index, head in enumerate(unambiguous[ROLES[0]]):
            head_molecule = molecules[ROLES[0]][head_index]
            for aldehyde_index, aldehyde in enumerate(unambiguous[ROLES[1]]):
                aldehyde_molecule = molecules[ROLES[1]][aldehyde_index]
                for isocyanide_index, isocyanide in enumerate(unambiguous[ROLES[2]]):
                    isocyanide_molecule = molecules[ROLES[2]][isocyanide_index]
                    attempted += 1
                    with rdBase.BlockLogs():
                        outcomes = reaction.forward.RunReactants(
                            (head_molecule, aldehyde_molecule, isocyanide_molecule),
                            maxProducts=maximum_outcomes,
                        )
                    if len(outcomes) >= maximum_outcomes:
                        raise ExpandedEnumerationError(
                            f"forward enumeration reached maximum_outcomes={maximum_outcomes}"
                        )
                    products: dict[str, Chem.Mol] = {}
                    for outcome in outcomes:
                        if len(outcome) != 1:
                            continue
                        normalized = _canonical_product(outcome[0])
                        if normalized is not None:
                            products[normalized[0]] = normalized[1]
                    if len(products) != 1:
                        invalid_or_ambiguous += 1
                        continue
                    product_smiles, product = next(iter(products.items()))
                    if product.GetNumHeavyAtoms() > maximum_heavy_atoms or not {
                        atom.GetSymbol() for atom in product.GetAtoms()
                    }.issubset(allowed_elements):
                        invalid_or_ambiguous += 1
                        continue
                    valid_unique_outcome += 1
                    components = (head, aldehyde, isocyanide)
                    batch.append(
                        _product_record(
                            product_smiles,
                            components,
                            source_stratum="expanded_exact_forward_enumeration",
                            measured=False,
                        )
                    )
                    if len(batch) >= 10_000:
                        before = connection.total_changes
                        connection.executemany(insert_sql, batch)
                        inserted = connection.total_changes - before
                        duplicate_products += len(batch) - inserted
                        connection.commit()
                        batch.clear()
        if batch:
            before = connection.total_changes
            connection.executemany(insert_sql, batch)
            inserted = connection.total_changes - before
            duplicate_products += len(batch) - inserted
            connection.commit()

        total_products = connection.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        fold_counts = dict(
            connection.execute(
                'SELECT "primary_product_fold", COUNT(*) FROM products '
                'GROUP BY "primary_product_fold"'
            ).fetchall()
        )
        stratum_counts = dict(
            connection.execute(
                'SELECT "source_stratum", COUNT(*) FROM products GROUP BY "source_stratum"'
            ).fetchall()
        )
        novelty_counts = dict(
            connection.execute(
                'SELECT "component_novelty_class", COUNT(*) FROM products '
                'GROUP BY "component_novelty_class"'
            ).fetchall()
        )
        artifact = _export_database(
            connection, output_path, compresslevel=int(policy["gzip_compresslevel"])
        )
        connection.close()

    artifact["path"] = str(output_path.relative_to(repo))
    artifact["rows"] = total_products
    artifact["columns"] = list(PRODUCT_FIELDS)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "status": "complete_exact_expanded_ugi_enumeration",
        "task": config["task"],
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "inputs": input_records,
        "policy": policy,
        "summary": {
            "admitted_components": {role: len(by_role[role]) for role in ROLES},
            "single_site_enumeration_components": {role: len(unambiguous[role]) for role in ROLES},
            "preserved_current_products": base_inserted,
            "attempted_expanded_triples": attempted,
            "valid_unique_outcome_triples": valid_unique_outcome,
            "invalid_or_ambiguous_triples": invalid_or_ambiguous,
            "enumeration_duplicates_against_existing_or_expanded": duplicate_products,
            "unique_products": total_products,
            "product_fold_counts": fold_counts,
            "source_stratum_counts": stratum_counts,
            "component_novelty_class_counts": novelty_counts,
        },
        "claims_boundary": {
            "exact_forward_enumeration_is_observed_synthesis": False,
            "exact_forward_enumeration_is_route_closure": False,
            "source_activity_labels_inherited": False,
            "family_balance_weight_must_be_used_by_training_sampler": True,
        },
        "randomness": {"used": False},
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "artifacts": {"products": artifact},
    }
    result_payload = (
        json.dumps(result, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"
    ).encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{result_path.name}.", dir=result_path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(result_payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, result_path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return result
