"""Freeze leak-aware M0-07 oracle evaluation splits.

The split artifact separates:

* the exact LANTERN random split, used only as a reproduction diagnostic;
* LANTERN's scaffold-balanced split, accepted after a scaffold-leakage audit;
* LANTERN's nominal Murcko split, retained only as an invalid-source audit;
* five-fold held-component and held-component-pair evaluations; and
* the unlabeled AGILE virtual library, used only for applicability analysis.

No model is trained here. This module freezes the evaluation contract before
the oracle matrix is fitted.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import platform
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

from forge.potency.oracle.agile_reconciliation import sha256_file

CONFIG_SCHEMA_VERSION = "m0_07_oracle_splits_config.v1"
RESULT_SCHEMA_VERSION = "m0_07_oracle_splits.v1"
ASSIGNMENT_FIELDS = ("scheme", "fold", "label", "stage", "group_id")
STAGES = ("train", "calibration", "test")
LABEL_ABS_TOLERANCE = 1e-7


class OracleSplitError(ValueError):
    """Raised when an oracle split violates the frozen evaluation contract."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise OracleSplitError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise OracleSplitError(f"{label} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise OracleSplitError(f"{label} must be a JSON object")
    return value


def _verify_hash(path: Path, expected: Any, label: str) -> dict[str, Any]:
    if not isinstance(expected, str) or len(expected) != 64:
        raise OracleSplitError(f"{label} expected sha256 must contain 64 characters")
    if not path.is_file():
        raise OracleSplitError(f"{label} not found: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise OracleSplitError(f"{label} hash mismatch: expected {expected}, observed {observed}")
    return {"path": str(path), "sha256": observed, "bytes": path.stat().st_size}


def _portable(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _canonical(smiles: str, label: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise OracleSplitError(f"{label} contains invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _read_csv(
    path: Path,
    required_fields: Sequence[str],
    label: str,
    *,
    compressed: bool = False,
) -> list[dict[str, str]]:
    try:
        handle = gzip.open(path, "rt", newline="") if compressed else path.open(newline="")
    except FileNotFoundError as exc:
        raise OracleSplitError(f"{label} not found: {path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        missing = set(required_fields) - set(reader.fieldnames or ())
        if missing:
            raise OracleSplitError(f"{label} is missing fields: {sorted(missing)}")
        return [dict(row) for row in reader]


def _load_config(path: Path) -> dict[str, Any]:
    config = _load_json(path, "M0-07 oracle split config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise OracleSplitError(f"unsupported config schema: {config.get('schema_version')!r}")
    randomness = config.get("randomness")
    if not isinstance(randomness, dict) or randomness.get("seed") != 1729:
        raise OracleSplitError("oracle split config must freeze seed 1729")
    policy = config.get("policy")
    if not isinstance(policy, dict):
        raise OracleSplitError("oracle split policy must be an object")
    required_policy = {
        "random_split_role": "reproduction_diagnostic_only",
        "murcko_source_split_role": "audit_only_reject_if_scaffold_leakage",
        "scaffold_balanced_role": "selection_eligible_if_scaffold_disjoint",
        "candidate_library_role": "unlabeled_applicability_only",
        "model_selection_uses_random_split": False,
    }
    for field, expected in required_policy.items():
        if policy.get(field) != expected:
            raise OracleSplitError(
                f"policy {field!r} must be {expected!r}, found {policy.get(field)!r}"
            )
    return config


def _load_numpy_split(path: Path, expected_rows: int) -> tuple[list[int], list[int], list[int]]:
    """Load a hash-pinned LANTERN object-array split.

    The upstream files require pickle because they are NumPy object arrays. The
    caller verifies their pinned SHA256 digests before this function runs.
    """

    try:
        raw = np.load(path, allow_pickle=True)
    except (OSError, ValueError) as exc:
        raise OracleSplitError(f"cannot load LANTERN split {path}: {exc}") from exc
    if raw.shape != (3,) or raw.dtype != object:
        raise OracleSplitError(
            f"LANTERN split must be an object array with shape (3,), found {raw.shape}/{raw.dtype}"
        )
    splits: list[list[int]] = []
    for stage_index, values in enumerate(raw):
        output: list[int] = []
        for value in values:
            if not isinstance(value, (int, np.integer)):
                raise OracleSplitError(
                    f"LANTERN split stage {stage_index} contains a non-integer index"
                )
            integer = int(value)
            if integer < 0 or integer >= expected_rows:
                raise OracleSplitError(
                    f"LANTERN split index {integer} is outside [0, {expected_rows})"
                )
            output.append(integer)
        if len(output) != len(set(output)):
            raise OracleSplitError(f"LANTERN split stage {stage_index} contains duplicates")
        splits.append(output)
    flattened = [index for split in splits for index in split]
    if len(flattened) != expected_rows or set(flattened) != set(range(expected_rows)):
        raise OracleSplitError("LANTERN split does not cover each source row exactly once")
    return splits[0], splits[1], splits[2]


def _scaffold(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise OracleSplitError(f"cannot compute scaffold for invalid SMILES: {smiles!r}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)


def _audit_scaffold_split(
    split: tuple[list[int], list[int], list[int]],
    scaffolds: Sequence[str],
) -> dict[str, Any]:
    stage_scaffolds = [{scaffolds[index] for index in stage_indices} for stage_indices in split]
    scaffold_stages: defaultdict[str, set[int]] = defaultdict(set)
    for stage_index, values in enumerate(stage_scaffolds):
        for value in values:
            scaffold_stages[value].add(stage_index)
    leaking = sorted(scaffold for scaffold, stages in scaffold_stages.items() if len(stages) > 1)
    return {
        "stage_rows": [len(values) for values in split],
        "stage_unique_scaffolds": [len(values) for values in stage_scaffolds],
        "total_unique_scaffolds": len(set(scaffolds)),
        "leaking_scaffold_groups": len(leaking),
        "leaking_scaffolds": leaking,
        "scaffold_disjoint": not leaking,
        "acyclic_rows_by_stage": [
            sum(scaffolds[index] == "" for index in stage_indices) for stage_indices in split
        ],
    }


def _hash_order(value: str, *, seed: int, namespace: str) -> str:
    return hashlib.sha256(f"{seed}\x1f{namespace}\x1f{value}".encode()).hexdigest()


def assign_groups_to_folds(
    groups: Sequence[str],
    *,
    n_folds: int,
    seed: int,
    namespace: str,
) -> dict[str, int]:
    """Assign unique equal-weight groups to balanced folds without using labels."""

    unique = sorted(
        set(groups), key=lambda value: _hash_order(value, seed=seed, namespace=namespace)
    )
    if len(unique) < n_folds:
        raise OracleSplitError(f"scheme {namespace!r} has {len(unique)} groups for {n_folds} folds")
    return {group: index % n_folds for index, group in enumerate(unique)}


def build_group_cv_assignments(
    records: Sequence[Mapping[str, str]],
    *,
    scheme: str,
    groups: Mapping[str, str],
    n_folds: int,
    seed: int,
    calibration_fraction_of_non_test: float,
) -> list[dict[str, Any]]:
    labels = {record["label"] for record in records}
    if set(groups) != labels:
        raise OracleSplitError(f"scheme {scheme!r} group labels do not match the curated data")
    group_folds = assign_groups_to_folds(
        list(groups.values()),
        n_folds=n_folds,
        seed=seed,
        namespace=scheme,
    )
    output: list[dict[str, Any]] = []
    for fold in range(n_folds):
        test_labels = {label for label, group in groups.items() if group_folds[group] == fold}
        non_test = sorted(labels - test_labels)
        calibration_count = round(len(non_test) * calibration_fraction_of_non_test)
        calibration_labels = set(
            sorted(
                non_test,
                key=lambda label: _hash_order(
                    label,
                    seed=seed,
                    namespace=f"{scheme}:fold-{fold}:calibration",
                ),
            )[:calibration_count]
        )
        for label in sorted(labels):
            if label in test_labels:
                stage = "test"
            elif label in calibration_labels:
                stage = "calibration"
            else:
                stage = "train"
            output.append(
                {
                    "scheme": scheme,
                    "fold": fold,
                    "label": label,
                    "stage": stage,
                    "group_id": groups[label],
                }
            )
    return output


def _validate_assignment_partition(
    assignments: Sequence[Mapping[str, Any]],
    labels: set[str],
) -> dict[str, Any]:
    by_fold: defaultdict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in assignments:
        by_fold[(str(row["scheme"]), int(row["fold"]))].append(row)
    summaries: dict[str, Any] = {}
    for (scheme, fold), rows in sorted(by_fold.items()):
        row_labels = [str(row["label"]) for row in rows]
        if len(row_labels) != len(labels) or set(row_labels) != labels:
            raise OracleSplitError(f"{scheme} fold {fold} does not assign every label once")
        stages = Counter(str(row["stage"]) for row in rows)
        if set(stages) != set(STAGES):
            raise OracleSplitError(f"{scheme} fold {fold} is missing one or more stages")
        test_groups = {str(row["group_id"]) for row in rows if row["stage"] == "test"}
        fit_groups = {
            str(row["group_id"]) for row in rows if row["stage"] in {"train", "calibration"}
        }
        group_overlap = test_groups & fit_groups
        if scheme.startswith("held_") and group_overlap:
            raise OracleSplitError(
                f"{scheme} fold {fold} leaks test groups into fitting: {sorted(group_overlap)}"
            )
        summaries[f"{scheme}:fold-{fold}"] = {
            "stage_rows": {stage: stages[stage] for stage in STAGES},
            "test_groups": len(test_groups),
            "test_group_leakage": len(group_overlap),
        }
    return summaries


def _render_assignments(rows: Sequence[Mapping[str, Any]]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=ASSIGNMENT_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        {
            "scheme": row["scheme"],
            "fold": row["fold"],
            "label": row["label"],
            "stage": row["stage"],
            "group_id": row["group_id"],
        }
        for row in sorted(
            rows,
            key=lambda row: (
                str(row["scheme"]),
                int(row["fold"]),
                str(row["label"]),
            ),
        )
    )
    return gzip.compress(text.getvalue().encode(), compresslevel=9, mtime=0)


def _component_ids(label: str) -> tuple[str, str, str]:
    try:
        a_part, remainder = label.split("B", maxsplit=1)
        b_part, c_part = remainder.split("C", maxsplit=1)
        return a_part, f"B{int(b_part)}", f"C{int(c_part)}"
    except (ValueError, TypeError) as exc:
        raise OracleSplitError(f"invalid curated component label: {label!r}") from exc


def build_oracle_splits(
    config_path: Path,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Build and atomically write the M0-07 split contract."""

    config = _load_config(config_path)
    config_record = {
        "path": _portable(config_path, repo_root),
        "sha256": sha256_file(config_path),
        "bytes": config_path.stat().st_size,
    }
    inputs: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for name, spec in config["inputs"].items():
        if not isinstance(spec, dict) or not isinstance(spec.get("path"), str):
            raise OracleSplitError(f"input {name!r} is malformed")
        path = repo_root / spec["path"]
        record = _verify_hash(path, spec.get("sha256"), f"input {name}")
        record["path"] = _portable(path, repo_root)
        inputs[name] = record
        paths[name] = path

    curated = _read_csv(
        paths["curated_oracle_data"],
        (
            "label",
            "model_smiles",
            "isomeric_smiles",
            "A_smiles",
            "B_smiles",
            "C_smiles",
            "expt_Hela",
        ),
        "curated oracle data",
        compressed=True,
    )
    expected_rows = int(config["expected"]["curated_records"])
    if len(curated) != expected_rows:
        raise OracleSplitError(
            f"curated oracle data has {len(curated)} rows, expected {expected_rows}"
        )
    labels = {row["label"] for row in curated}
    if len(labels) != expected_rows:
        raise OracleSplitError("curated oracle labels are not unique")

    lantern = _read_csv(
        paths["lantern_curated_hela"],
        ("SMILES", "Target"),
        "LANTERN curated HeLa data",
    )
    if len(lantern) != expected_rows:
        raise OracleSplitError(
            f"LANTERN curated data has {len(lantern)} rows, expected {expected_rows}"
        )
    curated_by_isomeric = {
        _canonical(row["isomeric_smiles"], f"{row['label']} isomeric SMILES"): row["label"]
        for row in curated
    }
    if len(curated_by_isomeric) != expected_rows:
        raise OracleSplitError("curated isomeric structures are not unique")
    try:
        curated_hela_by_label = {row["label"]: float(row["expt_Hela"]) for row in curated}
    except ValueError as exc:
        raise OracleSplitError("curated oracle contains a nonnumeric HeLa value") from exc
    source_index_to_label: dict[int, str] = {}
    source_scaffolds: list[str] = []
    max_hela_abs_difference = 0.0
    for index, row in enumerate(lantern):
        canonical = _canonical(row["SMILES"], f"LANTERN row {index}")
        try:
            label = curated_by_isomeric[canonical]
        except KeyError as exc:
            raise OracleSplitError(
                f"LANTERN row {index} is absent from the reconciled oracle data"
            ) from exc
        source_index_to_label[index] = label
        curated_hela = curated_hela_by_label[label]
        try:
            lantern_hela = float(row["Target"])
        except ValueError as exc:
            raise OracleSplitError(f"LANTERN row {index} contains a nonnumeric HeLa value") from exc
        hela_abs_difference = abs(curated_hela - lantern_hela)
        max_hela_abs_difference = max(max_hela_abs_difference, hela_abs_difference)
        if hela_abs_difference > LABEL_ABS_TOLERANCE:
            raise OracleSplitError(
                f"LANTERN row {index} HeLa value differs from reconciled label {label}: "
                f"{lantern_hela} versus {curated_hela}"
            )
        source_scaffolds.append(_scaffold(canonical))
    if len(set(source_index_to_label.values())) != expected_rows:
        raise OracleSplitError("LANTERN rows do not map one-to-one to curated labels")

    numpy_splits = {
        "lantern_random": _load_numpy_split(paths["lantern_random_split"], expected_rows),
        "lantern_murcko_source": _load_numpy_split(paths["lantern_murcko_split"], expected_rows),
        "lantern_scaffold_balanced": _load_numpy_split(
            paths["lantern_scaffold_balanced_split"], expected_rows
        ),
    }
    source_split_audits = {
        name: _audit_scaffold_split(split, source_scaffolds) for name, split in numpy_splits.items()
    }
    if source_split_audits["lantern_murcko_source"]["scaffold_disjoint"]:
        raise OracleSplitError(
            "the frozen contract expects the LANTERN Murcko source artifact to fail leakage audit"
        )
    if not source_split_audits["lantern_scaffold_balanced"]["scaffold_disjoint"]:
        raise OracleSplitError("LANTERN scaffold-balanced split leaks scaffold groups")

    assignments: list[dict[str, Any]] = []
    for scheme in ("lantern_random", "lantern_scaffold_balanced"):
        split = numpy_splits[scheme]
        for stage, indices in zip(STAGES, split, strict=True):
            for source_index in indices:
                label = source_index_to_label[source_index]
                assignments.append(
                    {
                        "scheme": scheme,
                        "fold": 0,
                        "label": label,
                        "stage": stage,
                        "group_id": (
                            label
                            if scheme == "lantern_random"
                            else source_scaffolds[source_index] or "<acyclic>"
                        ),
                    }
                )

    a_ids: dict[str, str] = {}
    b_ids: dict[str, str] = {}
    c_ids: dict[str, str] = {}
    for row in curated:
        a_id, b_id, c_id = _component_ids(row["label"])
        a_ids[row["label"]] = a_id
        b_ids[row["label"]] = b_id
        c_ids[row["label"]] = c_id
    component_cardinality = {
        "A": len(set(a_ids.values())),
        "B": len(set(b_ids.values())),
        "C": len(set(c_ids.values())),
    }
    expected_component_cardinality = {
        "A": int(config["expected"]["A_components"]),
        "B": int(config["expected"]["B_components"]),
        "C": int(config["expected"]["C_components"]),
    }
    if component_cardinality != expected_component_cardinality:
        raise OracleSplitError(
            "curated component cardinality differs from the frozen contract: "
            f"{component_cardinality} versus {expected_component_cardinality}"
        )
    group_schemes = {
        "held_head_5fold": a_ids,
        "held_aldehyde_5fold": b_ids,
        "held_isocyanide_5fold": c_ids,
        "held_head_aldehyde_pair_5fold": {
            label: f"{a_ids[label]}|{b_ids[label]}" for label in labels
        },
        "held_head_isocyanide_pair_5fold": {
            label: f"{a_ids[label]}|{c_ids[label]}" for label in labels
        },
        "held_aldehyde_isocyanide_pair_5fold": {
            label: f"{b_ids[label]}|{c_ids[label]}" for label in labels
        },
    }
    for scheme, groups in group_schemes.items():
        assignments.extend(
            build_group_cv_assignments(
                curated,
                scheme=scheme,
                groups=groups,
                n_folds=int(config["split_design"]["group_cv_folds"]),
                seed=int(config["randomness"]["seed"]),
                calibration_fraction_of_non_test=float(
                    config["split_design"]["calibration_fraction_of_non_test"]
                ),
            )
        )
    assignment_summary = _validate_assignment_partition(assignments, labels)

    virtual = _read_csv(
        paths["agile_virtual_candidate_library"],
        ("source_row_index", "canonical_isomeric_smiles"),
        "AGILE virtual candidate library",
        compressed=True,
    )
    if len(virtual) != int(config["expected"]["virtual_candidate_records"]):
        raise OracleSplitError(
            f"virtual candidate library has {len(virtual)} rows, expected "
            f"{config['expected']['virtual_candidate_records']}"
        )

    payload = _render_assignments(assignments)
    payload_record = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    selection_schemes = [
        "lantern_scaffold_balanced",
        "held_head_5fold",
        "held_aldehyde_5fold",
        "held_isocyanide_5fold",
        "held_head_aldehyde_pair_5fold",
        "held_head_isocyanide_pair_5fold",
        "held_aldehyde_isocyanide_pair_5fold",
    ]
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": "M0-07",
        "status": "oracle_evaluation_contract_frozen",
        "generated_utc": config["generated_utc"],
        "configuration": config_record,
        "randomness": config["randomness"],
        "inputs": inputs,
        "summary": {
            "curated_records": len(curated),
            "virtual_candidate_records": len(virtual),
            "component_cardinality": component_cardinality,
            "assignment_rows": len(assignments),
            "evaluation_schemes": len({row["scheme"] for row in assignments}),
            "selection_eligible_schemes": len(selection_schemes),
        },
        "source_label_alignment": {
            "rows_mapped_one_to_one": len(source_index_to_label),
            "max_hela_absolute_difference": max_hela_abs_difference,
            "absolute_tolerance": LABEL_ABS_TOLERANCE,
        },
        "source_split_audits": source_split_audits,
        "assignment_summary": assignment_summary,
        "artifacts": {"oracle_split_assignments.csv.gz": payload_record},
        "evaluation_contract": {
            "random_diagnostic_scheme": "lantern_random",
            "invalid_source_scheme": "lantern_murcko_source",
            "valid_structural_scheme": "lantern_scaffold_balanced",
            "selection_eligible_schemes": selection_schemes,
            "component_pair_shift_is_required": True,
            "candidate_library_evaluation": (
                "applicability and distance only because the 12,276 structures have no "
                "wet-lab endpoint labels"
            ),
            "model_selection_uses_random_split": False,
            "endpoints": ["expt_Hela", "expt_Raw"],
            "metrics": ["R2", "RMSE", "MAE", "Pearson_r", "Spearman_rho"],
            "calibration": {
                "method": "split_conformal_absolute_residual",
                "nominal_coverages": [0.8, 0.9, 0.95],
                "report": ["empirical_coverage", "coverage_gap", "mean_interval_width"],
            },
            "unavailable_properties": [
                "apparent_pKa",
                "particle_size",
                "polydispersity",
                "encapsulation_efficiency",
                "formulation_robustness",
            ],
        },
        "decision": {
            "lantern_random_is_selection_eligible": False,
            "lantern_murcko_source_is_selection_eligible": False,
            "lantern_scaffold_balanced_is_selection_eligible": True,
            "oracle_model_frozen": False,
            "reason": "models have not yet been fitted or calibrated",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
        },
    }
    result_payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.splits.", dir=output_dir.parent))
    try:
        (staging / "oracle_split_assignments.csv.gz").write_bytes(payload)
        (staging / "oracle_split_manifest.json").write_bytes(result_payload)
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(staging.iterdir()):
            os.replace(path, output_dir / path.name)
    finally:
        staging.rmdir()
    return result
