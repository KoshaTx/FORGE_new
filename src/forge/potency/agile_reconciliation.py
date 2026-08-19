"""Reconcile AGILE assay labels and molecular identities before oracle training.

The published AGILE library contains 1,200 nominal formulation measurements.
LANTERN established that the B4 reagent was a cis/trans mixture while B5 was a
pure trans reagent. A single-graph oracle cannot represent the B4 mixtures.
This module therefore:

* verifies HeLa and RAW labels against the official AGILE source workbook;
* excludes the 100 B4 mixture observations from single-graph supervision;
* corrects the 100 B5 product and component structures to the pure-trans graph;
* cross-validates the resulting 1,100 HeLa records against LANTERN; and
* emits deterministic audit, exclusion, and oracle-training artifacts.

The correction is a representation-policy gate. It does not add stereochemical
generation to FORGE. The model-facing SMILES are constitutional canonical
SMILES, while isomeric identities are retained for provenance and chemistry.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import platform
import re
import shutil
import tempfile
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree as ET

from rdkit import Chem, rdBase

from forge.core.hashing import sha256_file

CONFIG_SCHEMA_VERSION = "m0_07_agile_reconciliation_config.v1"
RESULT_SCHEMA_VERSION = "m0_07_agile_reconciliation.v1"

CURATED_FIELDS = (
    "source_row_id",
    "label",
    "model_smiles",
    "isomeric_smiles",
    "A_smiles",
    "B_smiles",
    "C_smiles",
    "expt_Hela",
    "expt_Raw",
    "structure_policy",
    "hela_label_source",
    "raw_label_source",
)
EXCLUSION_FIELDS = (
    "source_row_id",
    "label",
    "model_smiles",
    "reported_mixture_smiles",
    "mixture_members_json",
    "A_smiles",
    "original_B_smiles",
    "C_smiles",
    "expt_Hela",
    "expt_Raw",
    "exclusion_reason",
)
LEDGER_FIELDS = (
    "source_row_id",
    "label",
    "component_A_id",
    "component_B_id",
    "component_C_id",
    "original_model_smiles",
    "original_isomeric_smiles",
    "reconciled_model_smiles",
    "reconciled_isomeric_smiles",
    "original_B_smiles",
    "reconciled_B_smiles",
    "action",
    "source_hela",
    "source_raw",
    "lantern_hela",
)

_LABEL_RE = re.compile(r"^A(?P<a>\d+)(?:B(?P<b>\d+)C(?P<c>\d+)|C(?P<c2>\d+)B(?P<b2>\d+))$")
_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


class AgileReconciliationError(ValueError):
    """Raised when an input violates the frozen reconciliation contract."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise AgileReconciliationError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AgileReconciliationError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AgileReconciliationError(f"{label} must be a JSON object")
    return payload


def _canonical(smiles: str, *, isomeric: bool, label: str) -> str:
    if not isinstance(smiles, str) or not smiles:
        raise AgileReconciliationError(f"{label} must be a nonempty SMILES string")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise AgileReconciliationError(f"{label} is invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=isomeric)


def _parse_component_label(value: str) -> tuple[int, int, int]:
    match = _LABEL_RE.fullmatch(value)
    if match is None:
        raise AgileReconciliationError(f"invalid AGILE component label: {value!r}")
    a = int(match.group("a"))
    b = int(match.group("b") or match.group("b2"))
    c = int(match.group("c") or match.group("c2"))
    return a, b, c


def _normalized_label(value: str) -> str:
    a, b, c = _parse_component_label(value)
    return f"A{a}B{b}C{c}"


def _load_csv(path: Path, required_fields: Sequence[str], label: str) -> list[dict[str, str]]:
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            missing = set(required_fields) - fields
            if missing:
                raise AgileReconciliationError(f"{label} is missing columns: {sorted(missing)}")
            return [dict(row) for row in reader]
    except FileNotFoundError as exc:
        raise AgileReconciliationError(f"{label} not found: {path}") from exc


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    strings: list[str] = []
    for item in root.findall(f"{{{_MAIN_NS}}}si"):
        strings.append("".join(node.text or "" for node in item.iter(f"{{{_MAIN_NS}}}t")))
    return strings


def _xlsx_sheet_target(archive: zipfile.ZipFile, sheet_name: str) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall(f"{{{_PACKAGE_REL_NS}}}Relationship")
    }
    for sheet in workbook.findall(f".//{{{_MAIN_NS}}}sheet"):
        if sheet.attrib.get("name") == sheet_name:
            relationship_id = sheet.attrib.get(f"{{{_OFFICE_REL_NS}}}id")
            if relationship_id not in targets:
                break
            target = PurePosixPath(targets[relationship_id])
            if target.is_absolute() or ".." in target.parts:
                raise AgileReconciliationError(
                    f"unsafe worksheet target for {sheet_name!r}: {target}"
                )
            return str(PurePosixPath("xl") / target)
    raise AgileReconciliationError(f"worksheet {sheet_name!r} not found")


def _column_number(reference: str) -> int:
    letters = "".join(character for character in reference if character.isalpha())
    if not letters:
        raise AgileReconciliationError(f"invalid worksheet cell reference: {reference!r}")
    value = 0
    for character in letters:
        value = value * 26 + ord(character.upper()) - ord("A") + 1
    return value


def _load_xlsx_cells(path: Path, sheet_name: str) -> dict[tuple[int, int], str | float]:
    try:
        with zipfile.ZipFile(path) as archive:
            shared = _xlsx_shared_strings(archive)
            target = _xlsx_sheet_target(archive, sheet_name)
            root = ET.fromstring(archive.read(target))
    except (FileNotFoundError, zipfile.BadZipFile, KeyError, ET.ParseError) as exc:
        raise AgileReconciliationError(f"cannot parse workbook {path}: {exc}") from exc

    cells: dict[tuple[int, int], str | float] = {}
    for cell in root.findall(f".//{{{_MAIN_NS}}}c"):
        reference = cell.attrib.get("r", "")
        row_digits = "".join(character for character in reference if character.isdigit())
        if not row_digits:
            raise AgileReconciliationError(f"invalid worksheet cell reference: {reference!r}")
        row = int(row_digits)
        column = _column_number(reference)
        kind = cell.attrib.get("t")
        value_node = cell.find(f"{{{_MAIN_NS}}}v")
        if kind == "inlineStr":
            value = "".join(node.text or "" for node in cell.iter(f"{{{_MAIN_NS}}}t"))
        elif value_node is None or value_node.text is None:
            continue
        elif kind == "s":
            try:
                value = shared[int(value_node.text)]
            except (IndexError, ValueError) as exc:
                raise AgileReconciliationError(
                    f"invalid shared-string index at {reference}"
                ) from exc
        elif kind == "str":
            value = value_node.text
        else:
            try:
                value = float(value_node.text)
            except ValueError:
                value = value_node.text
        cells[(row, column)] = value
    return cells


def _source_matrix(
    workbook: Path,
    *,
    sheet_name: str,
    header_row: int,
    first_data_row: int,
    first_head_column: int,
) -> dict[str, float]:
    cells = _load_xlsx_cells(workbook, sheet_name)
    heads: dict[int, int] = {}
    for column in range(first_head_column, first_head_column + 20):
        value = cells.get((header_row, column))
        if not isinstance(value, str) or not value.startswith("A"):
            raise AgileReconciliationError(
                f"{sheet_name} missing A-head header at row {header_row}, column {column}"
            )
        heads[column] = int(value[1:])
    if set(heads.values()) != set(range(1, 21)):
        raise AgileReconciliationError(f"{sheet_name} does not contain A1 through A20")

    output: dict[str, float] = {}
    for offset in range(60):
        row = first_data_row + offset
        c = offset // 12 + 1
        b = offset % 12 + 1
        for column, a in heads.items():
            value = cells.get((row, column))
            if not isinstance(value, float):
                raise AgileReconciliationError(
                    f"{sheet_name} has no numeric value for A{a}B{b}C{c}"
                )
            output[f"A{a}B{b}C{c}"] = value
    if len(output) != 1200:
        raise AgileReconciliationError(f"{sheet_name} produced {len(output)} labels, expected 1200")
    return output


def _verify_input(path: Path, expected_sha256: Any, label: str) -> dict[str, Any]:
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise AgileReconciliationError(f"{label} has no valid expected sha256")
    if not path.is_file():
        raise AgileReconciliationError(f"{label} not found: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise AgileReconciliationError(
            f"{label} hash mismatch: expected {expected_sha256}, observed {observed}"
        )
    return {"path": str(path), "sha256": observed, "bytes": path.stat().st_size}


def _portable(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _float_text(value: float) -> str:
    return format(value, ".12g")


def _render_csv_gzip(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    text_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(text_buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return gzip.compress(text_buffer.getvalue().encode(), compresslevel=9, mtime=0)


def _artifact_record(payload: bytes) -> dict[str, Any]:
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _load_config(config_path: Path) -> dict[str, Any]:
    config = _load_json(config_path, "M0-07 AGILE reconciliation config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise AgileReconciliationError(
            f"unsupported config schema: {config.get('schema_version')!r}"
        )
    if config.get("randomness") != {"used": False, "seed": 0}:
        raise AgileReconciliationError("reconciliation must be deterministic with seed 0")
    if not isinstance(config.get("inputs"), dict) or len(config["inputs"]) != 4:
        raise AgileReconciliationError("config must declare exactly four source inputs")
    policy = config.get("policy")
    if not isinstance(policy, dict):
        raise AgileReconciliationError("config policy must be an object")
    required_policy = {
        "model_stereochemistry": "not_encoded",
        "mixture_handling": "exclude_from_single_graph_supervision_preserve_in_ledger",
        "pure_trans_handling": "correct_graph_retain_source_labels",
        "missing_properties": "never_impute",
    }
    for key, expected in required_policy.items():
        if policy.get(key) != expected:
            raise AgileReconciliationError(
                f"policy {key!r} must be {expected!r}, found {policy.get(key)!r}"
            )
    return config


def reconcile_agile_records(
    *,
    original_rows: Sequence[Mapping[str, str]],
    lantern_rows: Sequence[Mapping[str, str]],
    cis_trans_rows: Sequence[Mapping[str, str]],
    source_hela: Mapping[str, float],
    source_raw: Mapping[str, float],
    tolerance: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Reconcile already loaded records under the frozen B4/B5 policy."""

    if len(original_rows) != 1200:
        raise AgileReconciliationError(
            f"AGILE library has {len(original_rows)} rows, expected 1200"
        )
    by_label: dict[str, Mapping[str, str]] = {}
    for row in original_rows:
        label = _normalized_label(row["label"])
        if label in by_label:
            raise AgileReconciliationError(f"duplicate original label: {label}")
        by_label[label] = row
    if set(by_label) != set(source_hela) or set(by_label) != set(source_raw):
        raise AgileReconciliationError("AGILE labels and official source matrices do not agree")

    label_disagreements = {"HeLa": [], "RAW": []}
    for label, row in by_label.items():
        for endpoint, column, source in (
            ("HeLa", "expt_Hela", source_hela),
            ("RAW", "expt_Raw", source_raw),
        ):
            observed = float(row[column])
            if abs(observed - source[label]) > tolerance:
                label_disagreements[endpoint].append(label)
    if any(label_disagreements.values()):
        raise AgileReconciliationError(
            "vendored AGILE assay labels disagree with the official source workbook: "
            f"{label_disagreements}"
        )

    cis_trans_by_label: dict[str, Mapping[str, str]] = {}
    for row in cis_trans_rows:
        label = _normalized_label(row["Lipid"])
        if label in cis_trans_by_label:
            raise AgileReconciliationError(f"duplicate LANTERN cis/trans label: {label}")
        cis_trans_by_label[label] = row
    if len(cis_trans_by_label) != 200:
        raise AgileReconciliationError(
            f"LANTERN cis/trans audit has {len(cis_trans_by_label)} rows, expected 200"
        )

    b4_labels = {
        label
        for label, row in cis_trans_by_label.items()
        if _parse_component_label(label)[1] == 4 and row["Isometry"] == "C+T"
    }
    b5_labels = {
        label
        for label, row in cis_trans_by_label.items()
        if _parse_component_label(label)[1] == 5 and row["Isometry"] == ""
    }
    if len(b4_labels) != 100 or len(b5_labels) != 100:
        raise AgileReconciliationError(
            "LANTERN audit must contain 100 B4 mixtures and 100 B5 pure structures"
        )

    lantern_by_isomeric: dict[str, float] = {}
    for row in lantern_rows:
        canonical = _canonical(row["SMILES"], isomeric=True, label="LANTERN AGILE SMILES")
        if canonical in lantern_by_isomeric:
            raise AgileReconciliationError(
                f"LANTERN curated data contains duplicate structure: {canonical}"
            )
        lantern_by_isomeric[canonical] = float(row["Target"])
    if len(lantern_by_isomeric) != 1100:
        raise AgileReconciliationError(
            f"LANTERN curated data has {len(lantern_by_isomeric)} structures, expected 1100"
        )

    curated: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    lantern_mismatch_labels: list[str] = []

    def order_key(label: str) -> tuple[int, int, int]:
        return _parse_component_label(label)

    for label in sorted(by_label, key=order_key):
        row = by_label[label]
        a, b, c = _parse_component_label(label)
        original_isomeric = _canonical(
            row["combined_mol_SMILES"],
            isomeric=True,
            label=f"{label} original product",
        )
        original_model = _canonical(
            row["combined_mol_SMILES"],
            isomeric=False,
            label=f"{label} original product",
        )
        source_row_id = int(row["id"])
        action = "retain_source_graph"
        reconciled_isomeric = original_isomeric
        reconciled_model = original_model
        reconciled_b = _canonical(
            row["B_smiles"], isomeric=True, label=f"{label} original B component"
        )
        lantern_target = ""

        if label in b4_labels:
            mixture = cis_trans_by_label[label]["SMILES"]
            members = [
                _canonical(member.strip(), isomeric=True, label=f"{label} mixture member")
                for member in mixture.split(" + ")
            ]
            if len(members) != 2 or len(set(members)) != 2:
                raise AgileReconciliationError(
                    f"{label} mixture does not contain two distinct structures"
                )
            action = "exclude_cis_trans_mixture_from_single_graph_oracle"
            excluded.append(
                {
                    "source_row_id": source_row_id,
                    "label": label,
                    "model_smiles": original_model,
                    "reported_mixture_smiles": mixture,
                    "mixture_members_json": json.dumps(sorted(members), separators=(",", ":")),
                    "A_smiles": _canonical(
                        row["A_smiles"], isomeric=True, label=f"{label} A component"
                    ),
                    "original_B_smiles": reconciled_b,
                    "C_smiles": _canonical(
                        row["C_smiles"], isomeric=True, label=f"{label} C component"
                    ),
                    "expt_Hela": _float_text(source_hela[label]),
                    "expt_Raw": _float_text(source_raw[label]),
                    "exclusion_reason": (
                        "measured_cis_trans_mixture_cannot_be_represented_as_one_molecular_graph"
                    ),
                }
            )
        else:
            if label in b5_labels:
                cis_trans = cis_trans_by_label[label]
                reconciled_isomeric = _canonical(
                    cis_trans["SMILES"], isomeric=True, label=f"{label} LANTERN B5 product"
                )
                reconciled_model = _canonical(
                    cis_trans["SMILES"], isomeric=False, label=f"{label} LANTERN B5 product"
                )
                paired_b4 = by_label[f"A{a}B4C{c}"]
                reconciled_b = _canonical(
                    paired_b4["B_smiles"],
                    isomeric=True,
                    label=f"{label} corrected trans B component",
                )
                action = "correct_B5_to_pure_trans_graph"
                if reconciled_model != original_model:
                    raise AgileReconciliationError(
                        f"{label} B5 correction changed constitutional identity"
                    )
            try:
                lantern_value = lantern_by_isomeric[reconciled_isomeric]
            except KeyError as exc:
                raise AgileReconciliationError(
                    f"{label} reconciled structure is absent from LANTERN curated data"
                ) from exc
            lantern_target = _float_text(lantern_value)
            if abs(lantern_value - source_hela[label]) > tolerance:
                lantern_mismatch_labels.append(label)
            curated.append(
                {
                    "source_row_id": source_row_id,
                    "label": label,
                    "model_smiles": reconciled_model,
                    "isomeric_smiles": reconciled_isomeric,
                    "A_smiles": _canonical(
                        row["A_smiles"], isomeric=True, label=f"{label} A component"
                    ),
                    "B_smiles": reconciled_b,
                    "C_smiles": _canonical(
                        row["C_smiles"], isomeric=True, label=f"{label} C component"
                    ),
                    "expt_Hela": _float_text(source_hela[label]),
                    "expt_Raw": _float_text(source_raw[label]),
                    "structure_policy": (
                        "single_compound_constitutional_graph_with_isomeric_provenance"
                    ),
                    "hela_label_source": "AGILE_official_source_data_cross_validated_by_LANTERN",
                    "raw_label_source": "AGILE_official_source_data",
                }
            )

        ledger.append(
            {
                "source_row_id": source_row_id,
                "label": label,
                "component_A_id": f"A{a}",
                "component_B_id": f"B{b}",
                "component_C_id": f"C{c}",
                "original_model_smiles": original_model,
                "original_isomeric_smiles": original_isomeric,
                "reconciled_model_smiles": ("" if label in b4_labels else reconciled_model),
                "reconciled_isomeric_smiles": ("" if label in b4_labels else reconciled_isomeric),
                "original_B_smiles": _canonical(
                    row["B_smiles"], isomeric=True, label=f"{label} original B component"
                ),
                "reconciled_B_smiles": "" if label in b4_labels else reconciled_b,
                "action": action,
                "source_hela": _float_text(source_hela[label]),
                "source_raw": _float_text(source_raw[label]),
                "lantern_hela": lantern_target,
            }
        )

    if lantern_mismatch_labels:
        raise AgileReconciliationError(
            f"reconciled HeLa labels disagree with LANTERN: {lantern_mismatch_labels[:10]}"
        )
    if len(curated) != 1100 or len(excluded) != 100 or len(ledger) != 1200:
        raise AgileReconciliationError(
            f"unexpected outputs: curated={len(curated)}, excluded={len(excluded)}, "
            f"ledger={len(ledger)}"
        )

    model_counts = Counter(row["model_smiles"] for row in curated)
    isomeric_counts = Counter(row["isomeric_smiles"] for row in curated)
    if any(count != 1 for count in model_counts.values()):
        raise AgileReconciliationError("curated model-facing structures are not unique")
    if any(count != 1 for count in isomeric_counts.values()):
        raise AgileReconciliationError("curated isomeric structures are not unique")

    summary = {
        "nominal_experimental_measurements": 1200,
        "curated_single_structure_records": len(curated),
        "excluded_mixture_measurements": len(excluded),
        "B5_pure_trans_graph_corrections": sum(
            row["action"] == "correct_B5_to_pure_trans_graph" for row in ledger
        ),
        "official_hela_label_mismatches": 0,
        "official_raw_label_mismatches": 0,
        "lantern_hela_cross_validation_mismatches": 0,
        "unique_model_graphs": len(model_counts),
        "unique_isomeric_graphs": len(isomeric_counts),
        "component_counts": {
            "A": len({row["component_A_id"] for row in ledger}),
            "B_in_curated": len(
                {
                    row["component_B_id"]
                    for row in ledger
                    if row["action"] != "exclude_cis_trans_mixture_from_single_graph_oracle"
                }
            ),
            "C": len({row["component_C_id"] for row in ledger}),
        },
    }
    return curated, excluded, ledger, summary


def _publish_atomically(output_dir: Path, payloads: Mapping[str, bytes]) -> None:
    """Write every artifact, or none of them, with a single directory rename.

    Publishing file by file can leave new data beside a previous run's `result.json`, whose
    `artifacts` sha256 manifest then describes bytes no longer on disk -- a result set that still
    parses and is silently self-inconsistent. One rename makes the whole set appear at once.

    An existing output directory is moved aside first, because a rename cannot overwrite a
    non-empty directory, and is restored if the publish then fails.

    Two rules keep the failure path from causing the damage it exists to prevent. Cleanup never
    raises, so it cannot replace the exception that caused it -- the previous `finally: rmdir()`
    did exactly that, reporting `Directory not empty` instead of the real error. And the superseded
    copy is deleted only after a *successful* publish: if restoring it fails, it stays on disk
    under its temporary name, because a recoverable directory with an awkward name is a far better
    outcome than a deleted one.
    """
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    superseded = Path(f"{staging}.superseded")
    try:
        for name, payload in payloads.items():
            (staging / name).write_bytes(payload)
        if output_dir.exists():
            os.replace(output_dir, superseded)
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if superseded.exists() and not output_dir.exists():
            try:
                os.replace(superseded, output_dir)
            except OSError:
                pass
        raise
    shutil.rmtree(superseded, ignore_errors=True)


def run_agile_reconciliation(
    config_path: Path,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Run the deterministic, hash-pinned reconciliation gate."""

    config = _load_config(config_path)
    verified_inputs: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for name, spec in config["inputs"].items():
        if not isinstance(spec, dict) or not isinstance(spec.get("path"), str):
            raise AgileReconciliationError(f"input {name!r} is malformed")
        path = repo_root / spec["path"]
        record = _verify_input(path, spec.get("sha256"), f"input {name}")
        record["path"] = _portable(path, repo_root)
        verified_inputs[name] = record
        paths[name] = path

    original_rows = _load_csv(
        paths["agile_library"],
        (
            "id",
            "label",
            "combined_mol_SMILES",
            "A_smiles",
            "B_smiles",
            "C_smiles",
            "expt_Hela",
            "expt_Raw",
        ),
        "AGILE measured library",
    )
    lantern_rows = _load_csv(
        paths["lantern_curated_hela"],
        ("SMILES", "Target"),
        "LANTERN curated AGILE data",
    )
    cis_trans_rows = _load_csv(
        paths["lantern_cis_trans_audit"],
        ("Lipid", "Isometry", "SMILES", "Target"),
        "LANTERN cis/trans audit",
    )
    source_hela = _source_matrix(
        paths["agile_official_source_data"],
        sheet_name="Figure 2",
        header_row=3,
        first_data_row=4,
        first_head_column=2,
    )
    source_raw = _source_matrix(
        paths["agile_official_source_data"],
        sheet_name="Supplementary Figure 21",
        header_row=2,
        first_data_row=3,
        first_head_column=3,
    )
    tolerance = float(config["expected"]["label_tolerance"])
    curated, excluded, ledger, summary = reconcile_agile_records(
        original_rows=original_rows,
        lantern_rows=lantern_rows,
        cis_trans_rows=cis_trans_rows,
        source_hela=source_hela,
        source_raw=source_raw,
        tolerance=tolerance,
    )

    expected = config["expected"]
    required_counts = {
        "nominal_experimental_measurements": expected["nominal_measurements"],
        "curated_single_structure_records": expected["curated_records"],
        "excluded_mixture_measurements": expected["excluded_mixtures"],
        "B5_pure_trans_graph_corrections": expected["corrected_B5_records"],
    }
    for key, value in required_counts.items():
        if summary[key] != value:
            raise AgileReconciliationError(f"{key} is {summary[key]}, expected {value}")

    curated_payload = _render_csv_gzip(curated, CURATED_FIELDS)
    exclusion_payload = _render_csv_gzip(excluded, EXCLUSION_FIELDS)
    ledger_payload = _render_csv_gzip(ledger, LEDGER_FIELDS)
    artifacts = {
        "agile_oracle_curated.csv.gz": _artifact_record(curated_payload),
        "agile_mixture_exclusions.csv.gz": _artifact_record(exclusion_payload),
        "agile_reconciliation_ledger.csv.gz": _artifact_record(ledger_payload),
    }
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": "M0-07",
        "status": "completed_blocking_source_reconciliation",
        "generated_utc": config["generated_utc"],
        "randomness": config["randomness"],
        "inputs": verified_inputs,
        "policy": config["policy"],
        "summary": summary,
        "artifacts": artifacts,
        "adversarial_checks": {
            "official_hela_values_match_all_1200_labels": True,
            "official_raw_values_match_all_1200_labels": True,
            "B4_mixtures_absent_from_single_graph_training": True,
            "B4_mixtures_preserved_in_exclusion_ledger": True,
            "B5_graphs_corrected_to_LANTERN_pure_trans_identity": True,
            "LANTERN_hela_matches_all_1100_curated_records": True,
            "model_facing_constitutional_graphs_are_unique": True,
            "missing_formulation_properties_not_imputed": True,
        },
        "downstream_contract": {
            "oracle_training_input": "results/m0_07/agile_oracle_curated.csv.gz",
            "raw_1200_row_file_allowed_for_oracle_fit": False,
            "raw_1200_row_file_allowed_for_nominal_chemistry_audit": True,
            "lantern_scope": (
                "HeLa curation and B4/B5 identity audit only; RAW labels are independently "
                "reconciled to AGILE official source data"
            ),
            "requires_rebuild_before_model_training": [
                "R0 AGILE-derived single-structure records",
                "M0-03 splits derived from the unreconciled R0 artifact",
                "M0-04 AGILE-specific denominators derived from the unreconciled R0 artifact",
                "M0-06 corpus counts derived from the unreconciled R0 artifact",
            ],
            "not_invalidated": [
                "non-AGILE M0-04 conclusions",
                "Ugi transform mechanics for nominal library rows",
                "M0-09 upstream route evidence unrelated to B4/B5 identity",
            ],
        },
        "environment": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
        },
    }
    result_payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()

    _publish_atomically(
        output_dir,
        {
            "agile_oracle_curated.csv.gz": curated_payload,
            "agile_mixture_exclusions.csv.gz": exclusion_payload,
            "agile_reconciliation_ledger.csv.gz": ledger_payload,
            "agile_label_reconciliation.json": result_payload,
        },
    )
    return result
