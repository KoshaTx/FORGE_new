"""Reconcile AGILE-derived provenance and freeze constitutional R0 graphs."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, rdMolDescriptors

from forge.core.hashing import sha256_file
from forge.core.io import atomic_write as _atomic_write

CONFIG_SCHEMA_VERSION = "m0_03_r0_reconciliation_config.v1"
RESULT_SCHEMA_VERSION = "m0_03_r0_reconciliation_result.v1"
CORPUS_NAME = "r0_constitutional.csv.gz"
LEDGER_NAME = "r0_reconciliation_ledger.csv.gz"
RESULT_NAME = "r0_reconciliation.json"
CORPUS_FIELDS = (
    "r0_structure_id",
    "canonical_isomeric_smiles",
    "canonical_constitutional_smiles",
    "structure_sha256",
    "source_r0_structure_ids_json",
    "source_isomeric_smiles_json",
    "original_isomeric_smiles_json",
    "identity_reconciliation_json",
    "excluded_mixture_associations_json",
    "observed_source_ids",
    "all_available_source_ids",
    "observed_source_count",
    "observed_occurrence_count",
    "provenance_json",
    "region_annotations_json",
    "leakage_group_id",
    "study_split_groups_json",
    "component_holdout_groups_json",
    "reaction_family_holdout_groups",
    "prospective_lock_status",
    "biological_label_policy",
    "heavy_atoms",
    "formal_charge",
    "fragment_count",
    "ring_count",
    "aromatic_ring_count",
    "rotatable_bonds",
    "stereogenic_atom_count",
    "molecular_weight",
    "elements",
    "r0_pretraining_eligible",
)
LEDGER_FIELDS = (
    "source_r0_structure_id",
    "source_canonical_isomeric_smiles",
    "reconciled_single_compound_isomeric_smiles",
    "constitutional_smiles",
    "reconciled_r0_structure_id",
    "agile_labels_json",
    "agile_action",
    "single_compound_structure_evidence",
    "retained_as_structure_only_mixture_association",
    "source_observed_occurrence_count",
)


class R0ReconciliationError(ValueError):
    """Raised when constitutional R0 reconciliation violates its contract."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise R0ReconciliationError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise R0ReconciliationError(f"{label} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise R0ReconciliationError(f"{label} must be a JSON object")
    return value


def _read_csv(path: Path, required: set[str], label: str) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise R0ReconciliationError(f"{label} lacks fields: {sorted(missing)}")
            return [dict(row) for row in reader]
    except FileNotFoundError as exc:
        raise R0ReconciliationError(f"{label} not found: {path}") from exc


def _canonical(smiles: str, *, isomeric: bool, label: str) -> tuple[str, Chem.Mol]:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise R0ReconciliationError(f"{label} has invalid SMILES: {smiles!r}")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise R0ReconciliationError(f"{label} is disconnected")
    return (
        Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=isomeric,
        ),
        molecule,
    )


def _json_object(value: str, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise R0ReconciliationError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise R0ReconciliationError(f"{label} must be a JSON object")
    return parsed


def _merge_string_lists(
    rows: Sequence[Mapping[str, str]],
    field: str,
    *,
    separator: str,
) -> str:
    values = {item for row in rows for item in str(row[field]).split(separator) if item}
    return separator.join(sorted(values))


def _merge_annotation_objects(
    rows: Sequence[Mapping[str, str]],
    field: str,
) -> dict[str, dict[str, list[str]]]:
    merged: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        value = _json_object(row[field], f"{row['r0_structure_id']} {field}")
        for source, source_annotations in value.items():
            if not isinstance(source_annotations, Mapping):
                raise R0ReconciliationError(
                    f"{row['r0_structure_id']} {field}/{source} must be an object"
                )
            for name, items in source_annotations.items():
                if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
                    raise R0ReconciliationError(
                        f"{row['r0_structure_id']} {field}/{source}/{name} must be a string list"
                    )
                merged[str(source)][str(name)].update(items)
    return {
        source: {name: sorted(items) for name, items in sorted(annotations.items())}
        for source, annotations in sorted(merged.items())
    }


def _merge_study_objects(
    rows: Sequence[Mapping[str, str]],
) -> dict[str, list[str]]:
    merged: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        value = _json_object(
            row["study_split_groups_json"],
            f"{row['r0_structure_id']} study_split_groups_json",
        )
        for source, items in value.items():
            if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
                raise R0ReconciliationError("study split groups must be string lists")
            merged[str(source)].update(items)
    return {source: sorted(items) for source, items in sorted(merged.items())}


def _merge_provenance(
    rows: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    reaction_families: dict[str, set[str]] = defaultdict(set)
    occurrence_counts: Counter[str] = Counter()
    record_ids: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        value = _json_object(
            row["provenance_json"],
            f"{row['r0_structure_id']} provenance_json",
        )
        for source, record in value.items():
            if not isinstance(record, Mapping):
                raise R0ReconciliationError("provenance source record must be an object")
            source_name = str(source)
            source_metadata = record.get("metadata", {})
            if not isinstance(source_metadata, Mapping):
                raise R0ReconciliationError("provenance metadata must be an object")
            for name, items in source_metadata.items():
                if not isinstance(items, list):
                    raise R0ReconciliationError("provenance metadata must contain lists")
                metadata[source_name][str(name)].update(str(item) for item in items)
            reaction_family = record.get("reaction_family")
            if isinstance(reaction_family, str) and reaction_family:
                reaction_families[source_name].add(reaction_family)
            occurrence_counts[source_name] += int(record.get("source_occurrence_count", 0))
            ids = record.get("source_record_ids", [])
            if not isinstance(ids, list):
                raise R0ReconciliationError("provenance source_record_ids must be a list")
            record_ids[source_name].update(str(identifier) for identifier in ids)
    output = {}
    for source in sorted(
        set(metadata) | set(reaction_families) | set(occurrence_counts) | set(record_ids)
    ):
        families = sorted(reaction_families[source])
        output[source] = {
            "metadata": {name: sorted(items) for name, items in sorted(metadata[source].items())},
            "reaction_family": "|".join(families),
            "source_occurrence_count": occurrence_counts[source],
            "source_record_ids": sorted(record_ids[source]),
        }
    return output


def _agile_labels(row: Mapping[str, str]) -> list[str]:
    provenance = _json_object(
        row["provenance_json"],
        f"{row['r0_structure_id']} provenance_json",
    )
    agile = provenance.get("agile_measured1200")
    if agile is None:
        return []
    if not isinstance(agile, Mapping):
        raise R0ReconciliationError("AGILE provenance must be an object")
    metadata = agile.get("metadata", {})
    labels = metadata.get("label", []) if isinstance(metadata, Mapping) else []
    if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
        raise R0ReconciliationError("AGILE provenance labels must be a string list")
    return sorted(set(labels))


def _descriptors(molecule: Chem.Mol) -> dict[str, str]:
    potential_atoms = sum(
        item.type == Chem.StereoType.Atom_Tetrahedral for item in Chem.FindPotentialStereo(molecule)
    )
    elements = sorted({atom.GetSymbol() for atom in molecule.GetAtoms()})
    return {
        "heavy_atoms": str(molecule.GetNumHeavyAtoms()),
        "formal_charge": str(Chem.GetFormalCharge(molecule)),
        "fragment_count": str(len(Chem.GetMolFrags(molecule))),
        "ring_count": str(rdMolDescriptors.CalcNumRings(molecule)),
        "aromatic_ring_count": str(rdMolDescriptors.CalcNumAromaticRings(molecule)),
        "rotatable_bonds": str(rdMolDescriptors.CalcNumRotatableBonds(molecule)),
        "stereogenic_atom_count": str(potential_atoms),
        "molecular_weight": f"{Descriptors.MolWt(molecule):.6f}".rstrip("0").rstrip("."),
        "elements": "|".join(elements),
    }


def _verified_inputs(
    config: Mapping[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    specifications = config.get("inputs")
    expected_names = {
        "r0",
        "agile_reconciliation",
        "agile_reconciliation_ledger",
        "training_corpus_manifest",
    }
    if not isinstance(specifications, Mapping) or set(specifications) != expected_names:
        raise R0ReconciliationError("R0 reconciliation inputs are incomplete")
    paths = {}
    verified = {}
    for name, specification in sorted(specifications.items()):
        if not isinstance(specification, Mapping):
            raise R0ReconciliationError(f"{name} input specification is invalid")
        path = repo_root / str(specification.get("path", ""))
        observed = sha256_file(path)
        if observed != specification.get("sha256"):
            raise R0ReconciliationError(
                f"{name} hash mismatch: expected {specification.get('sha256')}, observed {observed}"
            )
        paths[name] = path
        verified[name] = {
            "path": str(path.resolve().relative_to(repo_root.resolve())),
            "sha256": observed,
            "bytes": path.stat().st_size,
        }
    return paths, verified


def _artifact_metadata(payload: bytes) -> dict[str, Any]:
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def _gzip_csv(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", compresslevel=9, mtime=0) as archive:
        archive.write(text.getvalue().encode())
    return output.getvalue()


def build_r0_reconciliation(
    config_path: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], bytes, bytes]:
    """Build reconciled constitutional R0 and a row-level identity ledger."""

    config = _load_json(config_path, "R0 reconciliation config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise R0ReconciliationError("unsupported R0 reconciliation config schema")
    if config.get("seed") != 20260730:
        raise R0ReconciliationError("R0 reconciliation seed must remain 20260730")
    required_policy = {
        "model_graph_identity": "canonical_constitutional_smiles",
        "stereochemistry_in_product_flow": False,
        "preserve_isomeric_source_provenance": True,
        "preserve_formal_charge": True,
        "preserve_protonation": True,
        "preserve_tautomers": True,
        "strip_fragments": False,
        "require_connected": True,
        "deduplicate_by_constitution": True,
        "b4_mixture_contributes_single_structure_weight": False,
        "b4_mixture_retained_in_exclusion_ledger": True,
        "b5_uses_reconciled_pure_trans_provenance": True,
    }
    if config.get("identity_policy") != required_policy:
        raise R0ReconciliationError("R0 identity policy changed from the frozen contract")
    paths, verified = _verified_inputs(config, repo_root)
    r0 = _read_csv(
        paths["r0"],
        set(CORPUS_FIELDS)
        - {
            "canonical_constitutional_smiles",
            "source_r0_structure_ids_json",
            "source_isomeric_smiles_json",
            "original_isomeric_smiles_json",
            "identity_reconciliation_json",
            "excluded_mixture_associations_json",
        },
        "R0",
    )
    agile_reconciliation = _load_json(
        paths["agile_reconciliation"],
        "AGILE reconciliation",
    )
    agile_ledger = _read_csv(
        paths["agile_reconciliation_ledger"],
        {
            "label",
            "original_isomeric_smiles",
            "reconciled_isomeric_smiles",
            "action",
        },
        "AGILE reconciliation ledger",
    )
    manifest = _load_json(paths["training_corpus_manifest"], "training corpus manifest")
    expected = config.get("expected")
    if not isinstance(expected, Mapping):
        raise R0ReconciliationError("R0 expected counts are absent")
    if len(r0) != expected["input_rows"]:
        raise R0ReconciliationError("R0 input count changed")
    if (
        manifest.get("layers", {}).get("r0_observed_real", {}).get("sha256")
        != verified["r0"]["sha256"]
    ):
        raise R0ReconciliationError("training manifest does not authenticate R0")
    if (
        agile_reconciliation.get("summary", {}).get("nominal_experimental_measurements")
        != expected["agile_derived_rows"]
    ):
        raise R0ReconciliationError("AGILE reconciliation count changed")
    by_label = {row["label"]: row for row in agile_ledger}
    if len(by_label) != len(agile_ledger):
        raise R0ReconciliationError("AGILE reconciliation labels are not unique")

    prepared = []
    action_counts: Counter[str] = Counter()
    agile_derived = 0
    for row in r0:
        labels = _agile_labels(row)
        if labels and len(labels) != 1:
            raise R0ReconciliationError(f"{row['r0_structure_id']} has {len(labels)} AGILE labels")
        action = "retain_source_single_compound_graph"
        reconciled_isomeric = row["canonical_isomeric_smiles"]
        single_compound = True
        if labels:
            agile_derived += 1
            source = by_label.get(labels[0])
            if source is None:
                raise R0ReconciliationError(
                    f"{row['r0_structure_id']} AGILE label is absent from reconciliation"
                )
            action = source["action"]
            if action == "exclude_cis_trans_mixture_from_single_graph_oracle":
                single_compound = False
                reconciled_isomeric = ""
            elif action == "correct_B5_to_pure_trans_graph":
                reconciled_isomeric = source["reconciled_isomeric_smiles"]
            elif action != "retain_source_graph":
                raise R0ReconciliationError(f"unsupported AGILE action {action!r}")
        structure_for_constitution = reconciled_isomeric or row["canonical_isomeric_smiles"]
        constitution, _ = _canonical(
            structure_for_constitution,
            isomeric=False,
            label=f"R0 {row['r0_structure_id']}",
        )
        action_counts[action] += 1
        prepared.append(
            {
                "row": row,
                "labels": labels,
                "action": action,
                "reconciled_isomeric": reconciled_isomeric,
                "single_compound": single_compound,
                "constitution": constitution,
            }
        )
    if agile_derived != expected["agile_derived_rows"]:
        raise R0ReconciliationError("AGILE-derived R0 count changed")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in prepared:
        groups[record["constitution"]].append(record)
    duplicate_groups = [records for records in groups.values() if len(records) > 1]
    duplicate_distribution = Counter(str(len(records)) for records in duplicate_groups)
    observed_group_contract = {
        "output_constitutions": len(groups),
        "input_duplicate_groups": len(duplicate_groups),
        "rows_collapsed": len(r0) - len(groups),
        "duplicate_group_size_distribution": dict(sorted(duplicate_distribution.items())),
    }
    for field, observed in observed_group_contract.items():
        if observed != expected[field]:
            raise R0ReconciliationError(
                f"{field} changed: expected {expected[field]}, observed {observed}"
            )

    corpus = []
    ledger = []
    b4_with_b5 = 0
    mixture_only = 0
    excluded_mixture_occurrences = 0
    for constitution, records in sorted(groups.items()):
        source_digest = hashlib.sha256(constitution.encode()).hexdigest()
        structure_id = f"R0C-{source_digest[:20]}"
        exact_records = [record for record in records if record["single_compound"]]
        mixture_records = [record for record in records if not record["single_compound"]]
        if mixture_records and not exact_records:
            mixture_only += 1
            raise R0ReconciliationError(
                f"{structure_id} is supported only by a mixture association"
            )
        if mixture_records:
            b5_actions = {record["action"] for record in exact_records if record["labels"]}
            if "correct_B5_to_pure_trans_graph" in b5_actions:
                b4_with_b5 += 1
        exact_rows = [record["row"] for record in exact_records]
        molecule = Chem.MolFromSmiles(constitution)
        excluded = []
        for record in mixture_records:
            row = record["row"]
            excluded_mixture_occurrences += int(row["observed_occurrence_count"])
            excluded.append(
                {
                    "source_r0_structure_id": row["r0_structure_id"],
                    "agile_labels": record["labels"],
                    "original_isomeric_smiles": row["canonical_isomeric_smiles"],
                    "observed_source_ids": row["observed_source_ids"].split("|"),
                    "observed_occurrence_count": int(row["observed_occurrence_count"]),
                    "evidence_state": "mixture_associated_structure_only",
                }
            )
        source_ids = _merge_string_lists(
            exact_rows,
            "observed_source_ids",
            separator="|",
        )
        all_sources = _merge_string_lists(
            exact_rows,
            "all_available_source_ids",
            separator="|",
        )
        identity_actions = Counter(record["action"] for record in records)
        corpus.append(
            {
                "r0_structure_id": structure_id,
                "canonical_isomeric_smiles": constitution,
                "canonical_constitutional_smiles": constitution,
                "structure_sha256": source_digest,
                "source_r0_structure_ids_json": json.dumps(
                    sorted(record["row"]["r0_structure_id"] for record in records),
                    separators=(",", ":"),
                ),
                "source_isomeric_smiles_json": json.dumps(
                    sorted(
                        {
                            record["reconciled_isomeric"]
                            for record in exact_records
                            if record["reconciled_isomeric"]
                        }
                    ),
                    separators=(",", ":"),
                ),
                "original_isomeric_smiles_json": json.dumps(
                    sorted({record["row"]["canonical_isomeric_smiles"] for record in records}),
                    separators=(",", ":"),
                ),
                "identity_reconciliation_json": json.dumps(
                    {
                        "actions": dict(sorted(identity_actions.items())),
                        "model_graph_identity": "constitutional",
                        "stereochemical_training_weight": 0,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "excluded_mixture_associations_json": json.dumps(
                    excluded,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "observed_source_ids": source_ids,
                "all_available_source_ids": all_sources,
                "observed_source_count": str(len(source_ids.split("|"))),
                "observed_occurrence_count": str(
                    sum(int(row["observed_occurrence_count"]) for row in exact_rows)
                ),
                "provenance_json": json.dumps(
                    _merge_provenance(exact_rows),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "region_annotations_json": json.dumps(
                    _merge_annotation_objects(exact_rows, "region_annotations_json"),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "leakage_group_id": f"mol-{source_digest}",
                "study_split_groups_json": json.dumps(
                    _merge_study_objects(exact_rows),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "component_holdout_groups_json": json.dumps(
                    _merge_annotation_objects(
                        exact_rows,
                        "component_holdout_groups_json",
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "reaction_family_holdout_groups": _merge_string_lists(
                    exact_rows,
                    "reaction_family_holdout_groups",
                    separator="|",
                ),
                "prospective_lock_status": "retrospective_pretraining_only",
                "biological_label_policy": ("no_label_inheritance_in_structural_pretraining"),
                **_descriptors(molecule),
                "r0_pretraining_eligible": "True",
            }
        )
        for record in records:
            row = record["row"]
            ledger.append(
                {
                    "source_r0_structure_id": row["r0_structure_id"],
                    "source_canonical_isomeric_smiles": row["canonical_isomeric_smiles"],
                    "reconciled_single_compound_isomeric_smiles": record["reconciled_isomeric"],
                    "constitutional_smiles": constitution,
                    "reconciled_r0_structure_id": structure_id,
                    "agile_labels_json": json.dumps(
                        record["labels"],
                        separators=(",", ":"),
                    ),
                    "agile_action": record["action"],
                    "single_compound_structure_evidence": record["single_compound"],
                    "retained_as_structure_only_mixture_association": (
                        not record["single_compound"]
                    ),
                    "source_observed_occurrence_count": row["observed_occurrence_count"],
                }
            )

    if (
        action_counts["exclude_cis_trans_mixture_from_single_graph_oracle"]
        != expected["b4_mixture_rows"]
    ):
        raise R0ReconciliationError("B4 mixture row count changed")
    if action_counts["correct_B5_to_pure_trans_graph"] != expected["b5_corrected_rows"]:
        raise R0ReconciliationError("B5 corrected row count changed")
    if b4_with_b5 != expected["b4_constitutions_with_exact_b5_support"]:
        raise R0ReconciliationError("B4/B5 constitutional support pairing changed")
    if mixture_only != expected["mixture_only_constitutions"]:
        raise R0ReconciliationError("mixture-only constitutional group count changed")
    if len({row["canonical_constitutional_smiles"] for row in corpus}) != len(corpus):
        raise R0ReconciliationError("reconciled R0 constitutions are not unique")
    if len({row["r0_structure_id"] for row in corpus}) != len(corpus):
        raise R0ReconciliationError("reconciled R0 IDs are not unique")

    corpus_payload = _gzip_csv(corpus, CORPUS_FIELDS)
    ledger_payload = _gzip_csv(
        sorted(ledger, key=lambda row: row["source_r0_structure_id"]),
        LEDGER_FIELDS,
    )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "completed_constitutional_r0_reconciliation",
        "task": config["task"],
        "inputs": verified,
        "configuration": {
            "path": str(config_path.resolve().relative_to(repo_root.resolve())),
            "sha256": sha256_file(config_path),
            "seed": config["seed"],
        },
        "identity_policy": config["identity_policy"],
        "summary": {
            "input_rows": len(r0),
            "output_constitutions": len(corpus),
            "input_duplicate_groups": len(duplicate_groups),
            "rows_collapsed": len(r0) - len(corpus),
            "duplicate_group_size_distribution": dict(sorted(duplicate_distribution.items())),
            "agile_derived_rows": agile_derived,
            "b4_mixture_rows_excluded_from_single_structure_weight": action_counts[
                "exclude_cis_trans_mixture_from_single_graph_oracle"
            ],
            "b5_rows_corrected_to_pure_trans_provenance": action_counts[
                "correct_B5_to_pure_trans_graph"
            ],
            "b4_constitutions_with_exact_b5_support": b4_with_b5,
            "mixture_only_constitutions": mixture_only,
            "excluded_mixture_source_occurrences": excluded_mixture_occurrences,
            "model_graph_stereochemical_labels": 0,
        },
        "artifacts": {
            CORPUS_NAME: _artifact_metadata(corpus_payload),
            LEDGER_NAME: _artifact_metadata(ledger_payload),
        },
        "decision": {
            "r0_ready_for_split_regeneration": True,
            "old_r0_and_splits_remain_quarantined": True,
            "product_flow_identity": "constitutional_graph",
            "isomeric_provenance_preserved": True,
            "b4_mixture_used_as_single_compound": False,
            "b5_identity_corrected": True,
        },
        "limitations": [
            (
                "Constitutional product-flow training does not distinguish alkene "
                "geometry or tetrahedral stereoisomers."
            ),
            (
                "The source isomeric structures remain provenance and chemistry "
                "records rather than generated graph labels."
            ),
            (
                "This artifact must receive new leakage-safe splits before any "
                "product-prior training."
            ),
        ],
        "runtime": {
            "python": sys.version,
            "rdkit": rdBase.rdkitVersion,
        },
    }
    return result, corpus_payload, ledger_payload


def write_r0_reconciliation(
    result: Mapping[str, Any],
    corpus_payload: bytes,
    ledger_payload: bytes,
    output_dir: Path,
) -> None:
    """Write deterministic R0 reconciliation artifacts atomically."""

    _atomic_write(output_dir / CORPUS_NAME, corpus_payload)
    _atomic_write(output_dir / LEDGER_NAME, ledger_payload)
    _atomic_write(
        output_dir / RESULT_NAME,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
