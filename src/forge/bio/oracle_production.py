"""Deterministically refit the frozen M0-07 AGILE oracle for inference."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import pickle
import statistics
import tempfile
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import rdChemReactions

from forge.bio import oracle_graph_matrix as graph_matrix
from forge.bio import oracle_graph_transfer as graph_transfer
from forge.bio.oracle_classical import (
    build_estimator,
    build_feature_bundle,
    load_classical_config,
    regression_metrics,
)
from forge.bio.oracle_freeze import RESULT_SCHEMA_VERSION as FREEZE_SCHEMA_VERSION
from forge.bio.oracle_freeze import sha256_file
from forge.bio.oracle_graph import (
    DMPNNEncoder,
    GraphFeatureVocabulary,
    OracleGraphRecord,
    WholeGraphRegressor,
    tensorize_smiles,
)
from forge.route.qualified_forward import (
    QualifiedForwardError,
    QualifiedForwardReaction,
    load_qualified_forward_reaction,
    unique_forward_products,
)

CONFIG_SCHEMA_VERSION = "m0_07_oracle_production_config.v1"
RESULT_SCHEMA_VERSION = "m0_07_oracle_production.v1"
CHECKPOINT_SCHEMA_VERSION = "m0_07_oracle_production_checkpoint.v1"
SUPPORTED_LANES = frozenset(
    {
        "classical",
        "supervised_graph",
        "label_free_r0_transfer",
    }
)


class OracleProductionError(ValueError):
    """Raised when the production refit violates the frozen oracle contract."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise OracleProductionError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise OracleProductionError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise OracleProductionError(f"{label} must contain an object")
    return value


def _stable_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise OracleProductionError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise OracleProductionError(f"{label} must be finite")
    return number


def _portable(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _is_repository_relative_path(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    path = Path(value)
    return bool(path.parts) and not path.is_absolute() and ".." not in path.parts


def _resolve_pinned_json_artifact(
    record: Any,
    *,
    repo_root: Path,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    """Resolve and authenticate a repository-local JSON artifact record."""

    if not isinstance(record, Mapping):
        raise OracleProductionError(f"{label} record is missing")
    raw_path = record.get("path")
    relative = Path(str(raw_path))
    expected_sha256 = str(record.get("sha256", ""))
    expected_bytes = record.get("bytes")
    if (
        not _is_repository_relative_path(raw_path)
        or len(expected_sha256) != 64
        or not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes < 0
    ):
        raise OracleProductionError(f"{label} record is unsafe")
    path = (repo_root / relative).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise OracleProductionError(f"{label} path escapes the repository") from exc
    try:
        observed_bytes = path.stat().st_size
        observed_sha256 = sha256_file(path)
    except FileNotFoundError as exc:
        raise OracleProductionError(f"{label} not found: {path}") from exc
    if observed_bytes != expected_bytes or observed_sha256 != expected_sha256:
        raise OracleProductionError(f"{label} does not match its frozen artifact record")
    return path, _load_json(path, label)


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION or config.get("seed") != 1729:
        raise OracleProductionError("unsupported production config schema or seed")
    inputs = config.get("inputs")
    chemistry = config.get("chemistry")
    training = config.get("training")
    uncertainty = config.get("uncertainty")
    checkpoint = config.get("checkpoint")
    if not all(
        isinstance(value, Mapping)
        for value in (inputs, chemistry, training, uncertainty, checkpoint)
    ):
        raise OracleProductionError("production config sections are incomplete")
    if set(inputs) != {
        "freeze_result",
        "classical_config",
        "supervised_graph_config",
        "label_free_r0_transfer_config",
    }:
        raise OracleProductionError("production input set changed")
    if not all(_is_repository_relative_path(value) for value in inputs.values()):
        raise OracleProductionError("production input path is unsafe")
    if (
        chemistry.get("reaction_id") != "ugi_3cr_agile"
        or chemistry.get("identity") != "canonical_constitutional_smiles"
        or chemistry.get("maximum_forward_products") != 128
    ):
        raise OracleProductionError("production chemistry contract changed")
    for key in ("qualified_reaction_registry", "reaction_variant"):
        specification = chemistry.get(key)
        if (
            not isinstance(specification, Mapping)
            or not _is_repository_relative_path(specification.get("path"))
            or not isinstance(specification.get("sha256"), str)
            or len(specification["sha256"]) != 64
        ):
            raise OracleProductionError("production chemistry source is invalid")
    if training.get("endpoints") != ["expt_Hela", "expt_Raw"]:
        raise OracleProductionError("production endpoints changed")
    if training.get("ensemble_seeds") != [1729, 11729, 21729]:
        raise OracleProductionError("production ensemble seeds changed")
    if training.get("records") != "all_1100_reconciled_single_structure_records":
        raise OracleProductionError("production refit must use all reconciled records")
    if training.get("guided_generation_labels_used") is not False:
        raise OracleProductionError("guided-generation outcomes cannot enter the M0-07 refit")
    if training.get("virtual_candidate_labels_used") is not False:
        raise OracleProductionError("the virtual applicability set has no labels")
    if uncertainty.get("coverages") != [0.8, 0.9, 0.95]:
        raise OracleProductionError("uncertainty coverages changed")
    if uncertainty.get("raw_mean_maximization_allowed") is not False:
        raise OracleProductionError("raw oracle-mean maximization is prohibited")
    if (
        uncertainty.get("guidance_score")
        != "ensemble_mean_minus_max_held_domain_empirical_residual_radius"
        or uncertainty.get("interpretation")
        != "conservative_empirical_held_domain_radius_not_a_finite_sample_coverage_guarantee_and_not_in_vivo_confidence"
    ):
        raise OracleProductionError("production uncertainty interpretation changed")
    if checkpoint.get("require_hash_before_inference") is not True:
        raise OracleProductionError("production inference must require a checkpoint hash")
    if checkpoint.get("stereochemistry_used") is not False:
        raise OracleProductionError("M0-07 production identity is constitutional")
    if checkpoint.get("graph_truncation_allowed") is not False:
        raise OracleProductionError("production graph truncation is prohibited")
    if not all(
        _is_repository_relative_path(checkpoint.get(key)) for key in ("path", "result_path")
    ):
        raise OracleProductionError("production checkpoint path is unsafe")


def _derive_seed(repeat_seed: int, lane: str, representation: str, endpoint: str) -> int:
    key = "\x1f".join(("m0-07-production", str(repeat_seed), lane, representation, endpoint))
    value = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    return value % (2**31 - 1) or 1


def _state_dict_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _read_metrics(path: Path) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except (FileNotFoundError, OSError) as exc:
        raise OracleProductionError(f"cannot read selected metrics: {path}") from exc


def build_domain_radii(
    *,
    metrics_path: Path,
    selected: Mapping[str, Any],
    freeze_result: Mapping[str, Any],
    coverages: Sequence[float],
) -> dict[str, Any]:
    """Use the largest foldwise split-conformal radius in each held-out domain."""

    rows = [
        row
        for row in _read_metrics(metrics_path)
        if row.get("representation") == selected["representation"]
        and row.get("model") == selected["model"]
        and row.get("scheme") in set(freeze_result["selection_contract"]["eligible_schemes"])
    ]
    if not rows:
        raise OracleProductionError("selected model has no uncertainty metric rows")
    endpoints = set(freeze_result["selection_contract"]["endpoints"])
    schemes = set(freeze_result["selection_contract"]["eligible_schemes"])
    by_endpoint_scheme: defaultdict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        by_endpoint_scheme[(str(row["endpoint"]), str(row["scheme"]))].append(row)
    if set(endpoint for endpoint, _ in by_endpoint_scheme) != endpoints:
        raise OracleProductionError("selected uncertainty rows lack an endpoint")
    scheme_radii: dict[str, dict[str, dict[str, float]]] = {}
    for endpoint in sorted(endpoints):
        scheme_radii[endpoint] = {}
        for scheme in sorted(schemes):
            group = by_endpoint_scheme.get((endpoint, scheme), [])
            if not group:
                raise OracleProductionError(f"selected uncertainty rows lack {endpoint}/{scheme}")
            radii = {}
            for coverage in coverages:
                percent = int(round(100 * coverage))
                field = f"conformal_q{percent}"
                values = [
                    _finite_float(row.get(field), f"{endpoint}/{scheme}/{field}") for row in group
                ]
                if any(value < 0 for value in values):
                    raise OracleProductionError("conformal radius cannot be negative")
                radii[f"q{percent}"] = max(values)
            scheme_radii[endpoint][scheme] = radii
    domain_radii: dict[str, dict[str, Any]] = {}
    for endpoint, endpoint_policy in freeze_result["applicability_policy"]["endpoints"].items():
        domain_radii[endpoint] = {}
        for domain, policy in endpoint_policy["domains"].items():
            required = list(policy["required_schemes"])
            radii = {
                f"q{int(round(100 * coverage))}": max(
                    scheme_radii[endpoint][scheme][f"q{int(round(100 * coverage))}"]
                    for scheme in required
                )
                for coverage in coverages
            }
            domain_radii[endpoint][domain] = {
                "authorized": bool(policy["passes"]),
                "action": policy["action"],
                "required_schemes": required,
                "radii": radii,
            }
        for domain, policy in freeze_result["applicability_policy"]["unsupported_domains"].items():
            domain_radii[endpoint][domain] = {
                "authorized": False,
                "action": "abstain",
                "required_schemes": [],
                "radii": {},
                "reason": str(policy["reason"]),
            }
    return {
        "scheme_radii": scheme_radii,
        "domain_radii": domain_radii,
        "rule": (
            "maximum foldwise split-conformal radius within each required "
            "scheme, then maximum across required schemes"
        ),
        "production_interpretation": (
            "conservative empirical held-domain radius transferred to the "
            "full-data refit; not a finite-sample coverage guarantee"
        ),
    }


def _median_epoch(values: Sequence[int]) -> int:
    if not values or any(value <= 0 for value in values):
        raise OracleProductionError("selected epoch set is empty or invalid")
    return max(1, int(math.floor(statistics.median(values) + 0.5)))


def collect_selected_graph_epochs(
    *,
    repo_root: Path,
    fit_root: Path,
    fit_source_index: Mapping[Path, Mapping[str, str]],
    fit_schema_version: str,
    expected_config_sha256: str,
    expected_input_hashes: Mapping[str, str],
    maximum_epoch: int,
    representation: str,
    endpoints: Sequence[str],
    repeat_seeds: Sequence[int],
    eligible_schemes: Sequence[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Collect train-only selected epochs without consulting predictions."""

    output: dict[str, dict[str, dict[str, Any]]] = {}
    for endpoint in endpoints:
        output[endpoint] = {}
        for repeat_seed in repeat_seeds:
            values: list[int] = []
            source_hashes: list[str] = []
            for scheme in eligible_schemes:
                scheme_root = fit_root / representation / endpoint / scheme
                paths = sorted(scheme_root.glob(f"fold-*/seed-{repeat_seed}.json"))
                expected_folds = {0} if scheme == "lantern_scaffold_balanced" else set(range(5))
                if len(paths) != len(expected_folds):
                    raise OracleProductionError(
                        f"expected {len(expected_folds)} epoch sources for "
                        f"{representation}/{endpoint}/{scheme}/seed-{repeat_seed}, "
                        f"found {len(paths)}"
                    )
                observed_folds: set[int] = set()
                for path in paths:
                    source = fit_source_index.get(path.resolve())
                    if not isinstance(source, Mapping) or source.get("sha256") != sha256_file(path):
                        raise OracleProductionError(
                            f"epoch source is not hash-pinned by the selected lane: {path}"
                        )
                    fit = _load_json(path, "selected graph fit")
                    job = fit.get("job", {})
                    try:
                        fold = int(job.get("fold", -1))
                        best_epoch = int(fit.get("epoch_selection", {}).get("best_epoch", -1))
                    except (TypeError, ValueError) as exc:
                        raise OracleProductionError(
                            f"epoch source contains non-integer fit metadata: {path}"
                        ) from exc
                    try:
                        expected_relative_path = str(
                            path.resolve().relative_to(repo_root.resolve())
                        )
                    except ValueError as exc:
                        raise OracleProductionError(
                            f"epoch source escapes the repository: {path}"
                        ) from exc
                    if (
                        fit.get("schema_version") != fit_schema_version
                        or fit.get("status") != "completed"
                        or job.get("job_id") != source.get("job_id")
                        or job.get("architecture") != representation
                        or job.get("endpoint") != endpoint
                        or job.get("scheme") != scheme
                        or int(job.get("seed", -1)) != repeat_seed
                        or fold not in expected_folds
                        or fold in observed_folds
                        or path.parent.name != f"fold-{fold}"
                        or job.get("output_relative_path") != expected_relative_path
                        or job.get("selection_eligible") != "true"
                        or fit.get("config_sha256") != expected_config_sha256
                        or fit.get("input_hashes") != dict(sorted(expected_input_hashes.items()))
                        or fit.get("epoch_selection", {}).get(
                            "calibration_or_test_targets_accessed"
                        )
                        is not False
                        or not 1 <= best_epoch <= maximum_epoch
                        or fit.get("boundary")
                        != {
                            "calibration_or_test_used_for_scaling": False,
                            "calibration_or_test_used_for_epoch_selection": False,
                            "calibration_or_test_used_for_weight_updates": False,
                            "calibration_and_test_accessed_after_checkpoint_hash": True,
                        }
                    ):
                        raise OracleProductionError(f"invalid epoch source: {path}")
                    observed_folds.add(fold)
                    values.append(best_epoch)
                    source_hashes.append(sha256_file(path))
                if observed_folds != expected_folds:
                    raise OracleProductionError(
                        f"{representation}/{endpoint}/{scheme}/seed-{repeat_seed} "
                        f"has folds {sorted(observed_folds)}, expected "
                        f"{sorted(expected_folds)}"
                    )
            output[endpoint][str(repeat_seed)] = {
                "epoch_count": _median_epoch(values),
                "source_fits": len(values),
                "source_hashes": sorted(source_hashes),
                "selection_targets_accessed": False,
            }
    return output


def _fit_source_index(
    selected_result: Mapping[str, Any],
    repo_root: Path,
) -> dict[Path, dict[str, str]]:
    """Resolve the aggregate result's immutable per-fit source manifest."""

    source_block = selected_result.get("fit_sources")
    if not isinstance(source_block, Mapping):
        raise OracleProductionError("selected graph lane lacks a fit-source manifest")
    entries = source_block.get("entries")
    expected_count = selected_result.get("summary", {}).get("fits")
    if (
        not isinstance(entries, list)
        or int(source_block.get("count", -1)) != len(entries)
        or not isinstance(expected_count, int)
        or isinstance(expected_count, bool)
        or expected_count != len(entries)
        or not entries
    ):
        raise OracleProductionError("selected graph fit-source manifest is incomplete")
    output: dict[Path, dict[str, str]] = {}
    job_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise OracleProductionError("selected graph fit-source entry is invalid")
        relative = Path(str(entry.get("path", "")))
        job_id = str(entry.get("job_id", ""))
        digest = str(entry.get("sha256", ""))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or len(job_id) != 64
            or len(digest) != 64
        ):
            raise OracleProductionError("selected graph fit-source entry is unsafe")
        path = (repo_root / relative).resolve()
        try:
            path.relative_to(repo_root.resolve())
        except ValueError as exc:
            raise OracleProductionError(
                "selected graph fit-source path escapes the repository"
            ) from exc
        if path in output or job_id in job_ids:
            raise OracleProductionError("selected graph fit-source manifest has duplicates")
        output[path] = {
            "job_id": job_id,
            "sha256": digest,
        }
        job_ids.add(job_id)
    return output


def _load_freeze_result(
    production_config: Mapping[str, Any],
    repo_root: Path,
) -> tuple[Path, dict[str, Any]]:
    relative = production_config["inputs"]["freeze_result"]
    if not isinstance(relative, str):
        raise OracleProductionError("freeze result path is invalid")
    path = repo_root / relative
    result = _load_json(path, "oracle freeze result")
    if (
        result.get("schema_version") != FREEZE_SCHEMA_VERSION
        or result.get("status") != "oracle_architecture_and_applicability_policy_frozen"
        or result.get("decision", {}).get("one_representation_and_model_frozen_across_endpoints")
        is not True
        or result.get("decision", {}).get("production_refit_required") is not True
    ):
        raise OracleProductionError("oracle freeze result is not production-ready")
    selected = result.get("selected_model")
    if not isinstance(selected, Mapping) or selected.get("lane") not in SUPPORTED_LANES:
        raise OracleProductionError("freeze result selected an unsupported lane")
    return path, result


def _selected_lane_paths(
    freeze_result: Mapping[str, Any],
    selected: Mapping[str, Any],
    repo_root: Path,
) -> tuple[Path, Path]:
    metadata = freeze_result["inputs"][selected["lane"]]
    result_path = repo_root / metadata["result"]["path"]
    metrics_path = repo_root / metadata["metrics"]["path"]
    if sha256_file(result_path) != metadata["result"]["sha256"]:
        raise OracleProductionError("selected lane result changed after oracle freeze")
    if sha256_file(metrics_path) != metadata["metrics"]["sha256"]:
        raise OracleProductionError("selected lane metrics changed after oracle freeze")
    return result_path, metrics_path


def _selected_lane_config_hash(
    selected_result: Mapping[str, Any],
    lane: str,
) -> str:
    record = (
        selected_result.get("configuration")
        if lane == "classical"
        else selected_result.get("inputs", {}).get("config")
    )
    if not isinstance(record, Mapping) or not isinstance(record.get("sha256"), str):
        raise OracleProductionError("selected lane result lacks its source config hash")
    return str(record["sha256"])


def _canonical_constitutional_smiles(smiles: str, label: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise OracleProductionError(f"{label} is invalid SMILES")
    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=False,
    )


def _build_applicability_index(
    source_config_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Pin the measured component universe used for domain classification."""

    source_config = _load_json(source_config_path, "selected lane config")
    specification = source_config.get("inputs", {}).get("curated_oracle_data")
    if not isinstance(specification, Mapping):
        raise OracleProductionError("selected lane config lacks curated oracle data")
    path = repo_root / str(specification.get("path", ""))
    if sha256_file(path) != specification.get("sha256"):
        raise OracleProductionError("curated oracle data changed before production refit")
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError) as exc:
        raise OracleProductionError("could not read curated oracle data") from exc
    required = {"label", "A_smiles", "B_smiles", "C_smiles"}
    if len(frame) != 1100 or not required.issubset(frame.columns):
        raise OracleProductionError("curated oracle component universe changed")
    role_fields = {
        "amine": "A_smiles",
        "aldehyde": "B_smiles",
        "isocyanide": "C_smiles",
    }
    components: dict[str, list[str]] = {}
    canonical_rows = []
    for row_index, row in frame.iterrows():
        canonical = {
            role: _canonical_constitutional_smiles(
                str(row[field]),
                f"curated row {row_index} {role}",
            )
            for role, field in role_fields.items()
        }
        canonical_rows.append(
            (
                str(row["label"]),
                canonical["amine"],
                canonical["aldehyde"],
                canonical["isocyanide"],
            )
        )
    for role_index, role in enumerate(role_fields, start=1):
        components[role] = sorted({row[role_index] for row in canonical_rows})
    triples = sorted({"\x1f".join((row[1], row[2], row[3])) for row in canonical_rows})
    digest = hashlib.sha256()
    for row in sorted(canonical_rows):
        digest.update("\x1f".join(row).encode())
        digest.update(b"\n")
    return {
        "identity": "canonical_constitutional_smiles",
        "source": {
            "path": _portable(path, repo_root),
            "sha256": sha256_file(path),
            "records": len(canonical_rows),
            "canonical_ledger_sha256": digest.hexdigest(),
        },
        "components": components,
        "measured_component_triples": triples,
    }


def _build_chemistry_contract(
    production_config: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Hash, compile, and serialize the frozen Ugi forward transform."""

    chemistry = production_config["chemistry"]
    registry_spec = chemistry["qualified_reaction_registry"]
    variant_spec = chemistry["reaction_variant"]
    registry_path = repo_root / registry_spec["path"]
    variant_path = repo_root / variant_spec["path"]
    for path, specification, label in (
        (registry_path, registry_spec, "qualified reaction registry"),
        (variant_path, variant_spec, "reaction variant"),
    ):
        if sha256_file(path) != specification["sha256"]:
            raise OracleProductionError(f"{label} changed before production refit")
    compiled = load_qualified_forward_reaction(
        registry_path,
        variant_path,
        reaction_id=str(chemistry["reaction_id"]),
    )
    return {
        "reaction_id": compiled.reaction_id,
        "role_names": list(compiled.role_names),
        "reaction_smarts": rdChemReactions.ReactionToSmarts(compiled.reaction),
        "maximum_forward_products": int(chemistry["maximum_forward_products"]),
        "identity": str(chemistry["identity"]),
        "sources": {
            "qualified_reaction_registry": {
                "path": _portable(registry_path, repo_root),
                "sha256": sha256_file(registry_path),
            },
            "reaction_variant": {
                "path": _portable(variant_path, repo_root),
                "sha256": sha256_file(variant_path),
            },
        },
    }


def _fit_supervised_graph(
    *,
    source_config_path: Path,
    selected_result: Mapping[str, Any],
    selected: Mapping[str, Any],
    freeze_result: Mapping[str, Any],
    production_config: Mapping[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    (
        source_config,
        source_config_hash,
        verified,
        records,
        _assignments,
        vocabulary,
    ) = graph_matrix._load_worker_inputs(source_config_path, repo_root)
    architecture = str(selected["representation"])
    if architecture not in source_config["matrix"]["architectures"]:
        raise OracleProductionError("selected supervised graph architecture is unavailable")
    seeds = [int(value) for value in production_config["training"]["ensemble_seeds"]]
    endpoints = [str(value) for value in production_config["training"]["endpoints"]]
    fit_root = repo_root / source_config["execution"]["fit_output_root"]
    epochs = collect_selected_graph_epochs(
        repo_root=repo_root,
        fit_root=fit_root,
        fit_source_index=_fit_source_index(selected_result, repo_root),
        fit_schema_version=graph_matrix.FIT_SCHEMA_VERSION,
        expected_config_sha256=source_config_hash,
        expected_input_hashes={
            name: str(record["sha256"]) for name, record in sorted(verified.items())
        },
        maximum_epoch=int(source_config["epoch_selection"]["maximum_epochs"]),
        representation=architecture,
        endpoints=endpoints,
        repeat_seeds=seeds,
        eligible_schemes=freeze_result["selection_contract"]["eligible_schemes"],
    )
    all_indices = list(range(len(records)))
    if len(all_indices) != 1100:
        raise OracleProductionError("supervised production record count changed")
    role_aware = architecture == "ugi_component_role_aware_dmpnn"
    endpoint_models: dict[str, Any] = {}
    training_metrics: dict[str, Any] = {}
    for endpoint_index, endpoint in enumerate(endpoints):
        target_mean, target_scale = graph_matrix._target_scaler(
            records,
            all_indices,
            endpoint_index,
        )
        ensemble = []
        predictions = []
        for repeat_seed in seeds:
            fit_seed = _derive_seed(
                repeat_seed,
                str(selected["lane"]),
                architecture,
                endpoint,
            )
            epoch_count = int(epochs[endpoint][str(repeat_seed)]["epoch_count"])
            model = graph_matrix._new_model(
                architecture,
                vocabulary,
                source_config["model"],
                fit_seed,
            )
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(source_config["model"]["learning_rate"]),
                weight_decay=float(source_config["model"]["weight_decay"]),
            )
            losses = []
            for epoch in range(1, epoch_count + 1):
                losses.append(
                    graph_matrix._train_epoch(
                        model,
                        records,
                        all_indices,
                        endpoint_index=endpoint_index,
                        mean=target_mean,
                        scale=target_scale,
                        optimizer=optimizer,
                        role_aware=role_aware,
                        model_config=source_config["model"],
                        order_seed=fit_seed + 100_000 + epoch,
                    )
                )
            prediction = graph_matrix._predict(
                model,
                records,
                all_indices,
                mean=target_mean,
                scale=target_scale,
                role_aware=role_aware,
                model_config=source_config["model"],
            )
            predictions.append(prediction)
            ensemble.append(
                {
                    "repeat_seed": repeat_seed,
                    "fit_seed": fit_seed,
                    "epoch_count": epoch_count,
                    "final_training_loss": losses[-1],
                    "model_sha256": graph_matrix._model_digest(model),
                    "state_dict": _state_dict_cpu(model),
                }
            )
        truth = np.asarray(
            [float(record.targets[endpoint_index]) for record in records],
            dtype=np.float64,
        )
        mean_prediction = np.mean(np.asarray(predictions, dtype=np.float64), axis=0)
        training_metrics[endpoint] = regression_metrics(truth, mean_prediction)
        endpoint_models[endpoint] = {
            "target_mean": target_mean,
            "target_scale": target_scale,
            "ensemble": ensemble,
        }
    checkpoint_lane = {
        "lane": "supervised_graph",
        "architecture": architecture,
        "model_config": source_config["model"],
        "feature_vocabulary": vocabulary.to_dict(),
        "role_aware": role_aware,
        "endpoint_models": endpoint_models,
    }
    audit = {
        "source_config_sha256": source_config_hash,
        "source_inputs": verified,
        "selected_epochs": epochs,
        "training_metrics_resubstitution_only": training_metrics,
    }
    return checkpoint_lane, audit


def _fit_transfer_graph(
    *,
    source_config_path: Path,
    selected_result: Mapping[str, Any],
    selected: Mapping[str, Any],
    freeze_result: Mapping[str, Any],
    production_config: Mapping[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    (
        source_config,
        source_config_hash,
        verified,
        records,
        _assignments,
        source_checkpoint,
        vocabulary,
    ) = graph_transfer._load_worker_inputs(source_config_path, repo_root)
    variant = str(selected["representation"])
    if variant not in source_config["matrix"]["variants"]:
        raise OracleProductionError("selected transfer variant is unavailable")
    seeds = [int(value) for value in production_config["training"]["ensemble_seeds"]]
    endpoints = [str(value) for value in production_config["training"]["endpoints"]]
    fit_root = repo_root / source_config["execution"]["fit_output_root"]
    epochs = collect_selected_graph_epochs(
        repo_root=repo_root,
        fit_root=fit_root,
        fit_source_index=_fit_source_index(selected_result, repo_root),
        fit_schema_version=graph_transfer.FIT_SCHEMA_VERSION,
        expected_config_sha256=source_config_hash,
        expected_input_hashes={
            name: str(record["sha256"]) for name, record in sorted(verified.items())
        },
        maximum_epoch=int(source_config["epoch_selection"]["maximum_epochs"]),
        representation=variant,
        endpoints=endpoints,
        repeat_seeds=seeds,
        eligible_schemes=freeze_result["selection_contract"]["eligible_schemes"],
    )
    all_indices = list(range(len(records)))
    if len(all_indices) != 1100:
        raise OracleProductionError("transfer production record count changed")
    frozen_encoder = variant == "r0_pretrained_frozen_linear"
    endpoint_models: dict[str, Any] = {}
    training_metrics: dict[str, Any] = {}
    for endpoint_index, endpoint in enumerate(endpoints):
        target_mean, target_scale = graph_transfer._target_scaler(
            records,
            all_indices,
            endpoint_index,
        )
        ensemble = []
        predictions = []
        for repeat_seed in seeds:
            fit_seed = _derive_seed(
                repeat_seed,
                str(selected["lane"]),
                variant,
                endpoint,
            )
            epoch_count = int(epochs[endpoint][str(repeat_seed)]["epoch_count"])
            model = graph_transfer.build_transfer_model(
                variant,
                vocabulary=vocabulary,
                checkpoint=source_checkpoint,
                seed=fit_seed,
            )
            optimizer = graph_transfer._optimizer(
                model,
                variant=variant,
                model_config=source_config["model"],
            )
            losses = []
            for epoch in range(1, epoch_count + 1):
                losses.append(
                    graph_transfer._train_epoch(
                        model,
                        records,
                        all_indices,
                        endpoint_index=endpoint_index,
                        mean=target_mean,
                        scale=target_scale,
                        optimizer=optimizer,
                        frozen_encoder=frozen_encoder,
                        model_config=source_config["model"],
                        order_seed=fit_seed + 100_000 + epoch,
                    )
                )
            prediction = graph_transfer._predict(
                model,
                records,
                all_indices,
                mean=target_mean,
                scale=target_scale,
                model_config=source_config["model"],
            )
            predictions.append(prediction)
            ensemble.append(
                {
                    "repeat_seed": repeat_seed,
                    "fit_seed": fit_seed,
                    "epoch_count": epoch_count,
                    "final_training_loss": losses[-1],
                    "model_sha256": graph_transfer._model_digest(model),
                    "encoder_sha256": graph_transfer._model_digest(model.encoder),
                    "state_dict": _state_dict_cpu(model),
                }
            )
        truth = np.asarray(
            [float(record.targets[endpoint_index]) for record in records],
            dtype=np.float64,
        )
        mean_prediction = np.mean(np.asarray(predictions, dtype=np.float64), axis=0)
        training_metrics[endpoint] = regression_metrics(truth, mean_prediction)
        endpoint_models[endpoint] = {
            "target_mean": target_mean,
            "target_scale": target_scale,
            "ensemble": ensemble,
        }
    checkpoint_lane = {
        "lane": "label_free_r0_transfer",
        "variant": variant,
        "model_config": source_config["model"],
        "encoder_architecture": source_checkpoint["architecture"],
        "feature_vocabulary": vocabulary.to_dict(),
        "pretraining_checkpoint_sha256": verified["pretraining_checkpoint"]["sha256"],
        "endpoint_models": endpoint_models,
    }
    audit = {
        "source_config_sha256": source_config_hash,
        "source_inputs": verified,
        "selected_epochs": epochs,
        "training_metrics_resubstitution_only": training_metrics,
    }
    return checkpoint_lane, audit


def _fit_classical(
    *,
    source_config_path: Path,
    selected: Mapping[str, Any],
    production_config: Mapping[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_config = load_classical_config(source_config_path)
    curated_spec = source_config["inputs"]["curated_oracle_data"]
    curated_path = repo_root / curated_spec["path"]
    if sha256_file(curated_path) != curated_spec["sha256"]:
        raise OracleProductionError("classical curated data changed")
    curated = pd.read_csv(curated_path)
    if len(curated) != 1100:
        raise OracleProductionError("classical production record count changed")
    bundle = build_feature_bundle(curated, source_config["representations"])
    representation = str(selected["representation"])
    model_name = str(selected["model"])
    if representation not in bundle.matrices or model_name not in source_config["models"]:
        raise OracleProductionError("selected classical candidate is unavailable")
    features = bundle.matrices[representation]
    seeds = [int(value) for value in production_config["training"]["ensemble_seeds"]]
    endpoints = [str(value) for value in production_config["training"]["endpoints"]]
    endpoint_models: dict[str, Any] = {}
    training_metrics: dict[str, Any] = {}
    fit_warnings: dict[str, list[str]] = {}
    for endpoint in endpoints:
        targets = curated[endpoint].to_numpy(dtype=np.float64)
        ensemble = []
        predictions = []
        endpoint_warnings: list[str] = []
        for repeat_seed in seeds:
            fit_seed = _derive_seed(
                repeat_seed,
                str(selected["lane"]),
                representation,
                endpoint,
            )
            estimator = build_estimator(
                model_name,
                source_config["models"][model_name],
                seed=fit_seed,
            )
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always")
                estimator.fit(features, targets)
            endpoint_warnings.extend(sorted({warning.category.__name__ for warning in observed}))
            prediction = np.asarray(estimator.predict(features), dtype=np.float64)
            if not np.all(np.isfinite(prediction)):
                raise OracleProductionError("classical production prediction is nonfinite")
            predictions.append(prediction)
            payload = pickle.dumps(estimator, protocol=5)
            ensemble.append(
                {
                    "repeat_seed": repeat_seed,
                    "fit_seed": fit_seed,
                    "pickle_sha256": hashlib.sha256(payload).hexdigest(),
                    "trusted_local_pickle": payload,
                }
            )
        mean_prediction = np.mean(np.asarray(predictions), axis=0)
        training_metrics[endpoint] = regression_metrics(targets, mean_prediction)
        endpoint_models[endpoint] = {"ensemble": ensemble}
        fit_warnings[endpoint] = sorted(set(endpoint_warnings))
    checkpoint_lane = {
        "lane": "classical",
        "representation": representation,
        "model": model_name,
        "representation_config": source_config["representations"],
        "model_config": source_config["models"][model_name],
        "feature_names": list(bundle.feature_names[representation]),
        "endpoint_models": endpoint_models,
    }
    audit = {
        "source_config_sha256": sha256_file(source_config_path),
        "curated_data_sha256": curated_spec["sha256"],
        "training_metrics_resubstitution_only": training_metrics,
        "fit_warning_categories": fit_warnings,
    }
    return checkpoint_lane, audit


def run_oracle_production_refit(
    config_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Fit, hash, and record the selected model package."""

    config = _load_json(config_path, "oracle production config")
    _validate_config(config)
    freeze_path, freeze_result = _load_freeze_result(config, repo_root)
    selected = freeze_result["selected_model"]
    selected_result_path, metrics_path = _selected_lane_paths(
        freeze_result,
        selected,
        repo_root,
    )
    uncertainty = build_domain_radii(
        metrics_path=metrics_path,
        selected=selected,
        freeze_result=freeze_result,
        coverages=[float(value) for value in config["uncertainty"]["coverages"]],
    )
    source_key = {
        "classical": "classical_config",
        "supervised_graph": "supervised_graph_config",
        "label_free_r0_transfer": "label_free_r0_transfer_config",
    }[selected["lane"]]
    source_config_path = repo_root / config["inputs"][source_key]
    selected_result = _load_json(selected_result_path, "selected lane result")
    expected_source_config_hash = _selected_lane_config_hash(
        selected_result,
        str(selected["lane"]),
    )
    if sha256_file(source_config_path) != expected_source_config_hash:
        raise OracleProductionError("selected lane source config changed after evaluation")
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.use_deterministic_algorithms(True)
    if selected["lane"] == "supervised_graph":
        lane_checkpoint, training_audit = _fit_supervised_graph(
            source_config_path=source_config_path,
            selected_result=selected_result,
            selected=selected,
            freeze_result=freeze_result,
            production_config=config,
            repo_root=repo_root,
        )
    elif selected["lane"] == "label_free_r0_transfer":
        lane_checkpoint, training_audit = _fit_transfer_graph(
            source_config_path=source_config_path,
            selected_result=selected_result,
            selected=selected,
            freeze_result=freeze_result,
            production_config=config,
            repo_root=repo_root,
        )
    else:
        lane_checkpoint, training_audit = _fit_classical(
            source_config_path=source_config_path,
            selected=selected,
            production_config=config,
            repo_root=repo_root,
        )
    checkpoint_payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "selected_model": dict(selected),
        "freeze_result_sha256": sha256_file(freeze_path),
        "production_config_sha256": sha256_file(config_path),
        "data_policy": {
            "records": config["training"]["records"],
            "virtual_candidate_labels_used": False,
            "guided_generation_labels_used": False,
            "stereochemistry_used": False,
            "graph_truncation_allowed": False,
        },
        "uncertainty": uncertainty,
        "applicability_index": _build_applicability_index(
            source_config_path,
            repo_root,
        ),
        "chemistry": _build_chemistry_contract(config, repo_root),
        "software": {
            "rdkit": rdBase.rdkitVersion,
            "scikit_learn": sklearn.__version__,
            "torch": str(torch.__version__),
        },
        "model": lane_checkpoint,
    }
    checkpoint_path = repo_root / config["checkpoint"]["path"]
    result_path = repo_root / config["checkpoint"]["result_path"]
    _atomic_torch_save(checkpoint_path, checkpoint_payload)
    checkpoint_hash = sha256_file(checkpoint_path)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "production_oracle_checkpoint_frozen",
        "configuration": {
            "path": _portable(config_path, repo_root),
            "sha256": sha256_file(config_path),
            "bytes": config_path.stat().st_size,
        },
        "freeze_result": {
            "path": _portable(freeze_path, repo_root),
            "sha256": sha256_file(freeze_path),
            "bytes": freeze_path.stat().st_size,
        },
        "selected_model": dict(selected),
        "training": training_audit,
        "uncertainty": uncertainty,
        "checkpoint": {
            "path": _portable(checkpoint_path, repo_root),
            "sha256": checkpoint_hash,
            "bytes": checkpoint_path.stat().st_size,
            "format": config["checkpoint"]["format"],
            "hash_required_before_inference": True,
        },
        "decision": {
            "production_checkpoint_frozen": True,
            "oracle_tilting_authorized_only_in_passing_domains": True,
            "raw_mean_maximization_allowed": False,
            "in_vivo_endpoint_oracle": False,
            "unavailable_formulation_properties_imputed": False,
        },
    }
    _atomic_write(result_path, _stable_json(result))
    return result


def load_production_checkpoint(
    production_result_path: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Load only a checkpoint authenticated by its frozen production result."""

    result = _load_json(production_result_path, "oracle production result")
    if (
        result.get("schema_version") != RESULT_SCHEMA_VERSION
        or result.get("status") != "production_oracle_checkpoint_frozen"
        or result.get("decision", {}).get("production_checkpoint_frozen") is not True
    ):
        raise OracleProductionError("oracle production result is not frozen")
    config_path, config = _resolve_pinned_json_artifact(
        result.get("configuration"),
        repo_root=repo_root,
        label="oracle production config",
    )
    _validate_config(config)
    expected_result_path = (repo_root / config["checkpoint"]["result_path"]).resolve()
    if production_result_path.resolve() != expected_result_path:
        raise OracleProductionError(
            "oracle production result is not the path frozen by its production config"
        )
    freeze_path, freeze_result = _resolve_pinned_json_artifact(
        result.get("freeze_result"),
        repo_root=repo_root,
        label="oracle freeze result",
    )
    expected_freeze_path = (repo_root / config["inputs"]["freeze_result"]).resolve()
    if freeze_path != expected_freeze_path:
        raise OracleProductionError(
            "oracle freeze result is not the path frozen by the production config"
        )
    if (
        freeze_result.get("schema_version") != FREEZE_SCHEMA_VERSION
        or freeze_result.get("status") != "oracle_architecture_and_applicability_policy_frozen"
        or freeze_result.get("decision", {}).get(
            "one_representation_and_model_frozen_across_endpoints"
        )
        is not True
        or freeze_result.get("decision", {}).get("production_refit_required") is not True
        or freeze_result.get("selected_model") != result.get("selected_model")
    ):
        raise OracleProductionError("oracle freeze result is not production-ready")
    selected = result["selected_model"]
    selected_lane = str(selected["lane"])
    lane_metadata = freeze_result.get("inputs", {}).get(selected_lane)
    if not isinstance(lane_metadata, Mapping):
        raise OracleProductionError("oracle freeze result lacks the selected lane provenance")
    selected_result_path, selected_result = _resolve_pinned_json_artifact(
        lane_metadata.get("result"),
        repo_root=repo_root,
        label="selected oracle lane result",
    )
    expected_lane_schema = lane_metadata["result"].get("schema_version")
    expected_lane_status = lane_metadata["result"].get("status")
    if (
        selected_result.get("schema_version") != expected_lane_schema
        or selected_result.get("status") != expected_lane_status
    ):
        raise OracleProductionError("selected oracle lane result contract changed")
    source_key = {
        "classical": "classical_config",
        "supervised_graph": "supervised_graph_config",
        "label_free_r0_transfer": "label_free_r0_transfer_config",
    }.get(selected_lane)
    if source_key is None:
        raise OracleProductionError("oracle freeze result selected an unsupported lane")
    source_config_path = (repo_root / config["inputs"][source_key]).resolve()
    try:
        source_config_path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise OracleProductionError("selected lane config escapes the repository") from exc
    expected_source_config_hash = _selected_lane_config_hash(
        selected_result,
        selected_lane,
    )
    if sha256_file(source_config_path) != expected_source_config_hash:
        raise OracleProductionError(
            "selected lane source config changed after scientific evaluation"
        )
    expected_applicability = _build_applicability_index(source_config_path, repo_root)
    expected_chemistry = _build_chemistry_contract(config, repo_root)
    checkpoint_record = result.get("checkpoint")
    if not isinstance(checkpoint_record, Mapping):
        raise OracleProductionError("oracle production result lacks a checkpoint")
    relative = Path(str(checkpoint_record.get("path", "")))
    expected_sha256 = str(checkpoint_record.get("sha256", ""))
    expected_bytes = checkpoint_record.get("bytes")
    if (
        not _is_repository_relative_path(checkpoint_record.get("path"))
        or len(expected_sha256) != 64
        or not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes < 0
        or checkpoint_record.get("hash_required_before_inference") is not True
    ):
        raise OracleProductionError("oracle production checkpoint record is unsafe")
    checkpoint_path = (repo_root / relative).resolve()
    try:
        checkpoint_path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise OracleProductionError("production checkpoint path escapes the repository") from exc
    expected_checkpoint_path = (repo_root / config["checkpoint"]["path"]).resolve()
    if checkpoint_path != expected_checkpoint_path:
        raise OracleProductionError(
            "production checkpoint is not the path frozen by the production config"
        )
    if checkpoint_record.get("format") != config["checkpoint"]["format"]:
        raise OracleProductionError("production checkpoint format changed")
    observed = sha256_file(checkpoint_path)
    if observed != expected_sha256 or checkpoint_path.stat().st_size != expected_bytes:
        raise OracleProductionError(
            "production checkpoint hash mismatch or byte-count mismatch: "
            f"expected {expected_sha256}, observed {observed}"
        )
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise OracleProductionError(
            f"could not load production checkpoint: {checkpoint_path}"
        ) from exc
    expected_policy = {
        "records": "all_1100_reconciled_single_structure_records",
        "virtual_candidate_labels_used": False,
        "guided_generation_labels_used": False,
        "stereochemistry_used": False,
        "graph_truncation_allowed": False,
    }
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or checkpoint.get("data_policy") != expected_policy
        or checkpoint.get("freeze_result_sha256") != sha256_file(freeze_path)
        or checkpoint.get("production_config_sha256") != sha256_file(config_path)
        or checkpoint.get("selected_model") != result.get("selected_model")
    ):
        raise OracleProductionError("production checkpoint contract changed")
    applicability = checkpoint.get("applicability_index")
    chemistry = checkpoint.get("chemistry")
    if (
        not isinstance(applicability, Mapping)
        or applicability != expected_applicability
        or not isinstance(chemistry, Mapping)
        or chemistry != expected_chemistry
    ):
        raise OracleProductionError("production applicability or chemistry contract changed")
    if checkpoint.get("software") != {
        "rdkit": rdBase.rdkitVersion,
        "scikit_learn": sklearn.__version__,
        "torch": str(torch.__version__),
    }:
        raise OracleProductionError("production inference software versions differ from the refit")
    selected = checkpoint["selected_model"]
    model = checkpoint.get("model")
    if not isinstance(selected, Mapping) or not isinstance(model, Mapping):
        raise OracleProductionError("production checkpoint model identity is incomplete")
    representation_key = {
        "classical": "representation",
        "supervised_graph": "architecture",
        "label_free_r0_transfer": "variant",
    }.get(str(selected.get("lane")))
    if (
        representation_key is None
        or model.get("lane") != selected.get("lane")
        or model.get(representation_key) != selected.get("representation")
    ):
        raise OracleProductionError("production checkpoint selected-model identity changed")
    checkpoint["_runtime_hash_verified"] = True
    checkpoint["_runtime_production_result_sha256"] = sha256_file(production_result_path)
    return checkpoint


def _require_verified_checkpoint(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    if (
        checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or checkpoint.get("_runtime_hash_verified") is not True
    ):
        raise OracleProductionError(
            "inference requires load_production_checkpoint with an expected hash"
        )
    model = checkpoint.get("model")
    if not isinstance(model, Mapping):
        raise OracleProductionError("production checkpoint lacks a model package")
    return model


def _vocabulary_from_checkpoint(value: Any) -> GraphFeatureVocabulary:
    if not isinstance(value, Mapping):
        raise OracleProductionError("production graph vocabulary is invalid")
    fields = (
        "elements",
        "formal_charges",
        "total_degrees",
        "total_hydrogens",
        "hybridizations",
        "bond_types",
    )
    if not all(isinstance(value.get(field), list) and value[field] for field in fields):
        raise OracleProductionError("production graph vocabulary is incomplete")
    return GraphFeatureVocabulary(
        elements=tuple(str(item) for item in value["elements"]),
        formal_charges=tuple(str(item) for item in value["formal_charges"]),
        total_degrees=tuple(str(item) for item in value["total_degrees"]),
        total_hydrogens=tuple(str(item) for item in value["total_hydrogens"]),
        hybridizations=tuple(str(item) for item in value["hybridizations"]),
        bond_types=tuple(str(item) for item in value["bond_types"]),
    )


def predict_production_graph_records(
    checkpoint: Mapping[str, Any],
    records: Sequence[OracleGraphRecord],
    *,
    endpoint: str,
) -> dict[str, Any]:
    """Predict records already tensorized under the checkpoint vocabulary.

    Generator integration should normally use ``predict_production_ugi_smiles``
    so tensorization is owned by the verified checkpoint.
    """

    model_package = _require_verified_checkpoint(checkpoint)
    lane = str(model_package.get("lane"))
    if lane not in {"supervised_graph", "label_free_r0_transfer"}:
        raise OracleProductionError("selected production oracle is not graph-based")
    endpoint_package = model_package.get("endpoint_models", {}).get(endpoint)
    if not isinstance(endpoint_package, Mapping):
        raise OracleProductionError(f"production checkpoint lacks endpoint {endpoint!r}")
    target_mean = _finite_float(endpoint_package.get("target_mean"), "target_mean")
    target_scale = _finite_float(endpoint_package.get("target_scale"), "target_scale")
    if target_scale <= 0:
        raise OracleProductionError("production target scale must be positive")
    ensemble = endpoint_package.get("ensemble")
    if not isinstance(ensemble, list) or len(ensemble) != 3:
        raise OracleProductionError("production graph ensemble must contain three models")
    vocabulary = _vocabulary_from_checkpoint(model_package.get("feature_vocabulary"))
    indices = list(range(len(records)))
    predictions: list[list[float]] = []
    for member in ensemble:
        if not isinstance(member, Mapping) or not isinstance(member.get("state_dict"), Mapping):
            raise OracleProductionError("production graph ensemble member is invalid")
        if lane == "supervised_graph":
            architecture = str(model_package["architecture"])
            role_aware = bool(model_package["role_aware"])
            model = graph_matrix._new_model(
                architecture,
                vocabulary,
                model_package["model_config"],
                int(member["fit_seed"]),
            )
            model.load_state_dict(member["state_dict"], strict=True)
            prediction = graph_matrix._predict(
                model,
                records,
                indices,
                mean=target_mean,
                scale=target_scale,
                role_aware=role_aware,
                model_config=model_package["model_config"],
            )
        else:
            architecture = model_package.get("encoder_architecture")
            if not isinstance(architecture, Mapping):
                raise OracleProductionError("transfer encoder architecture is invalid")
            encoder = DMPNNEncoder(
                vocabulary.atom_feature_dim,
                vocabulary.bond_feature_dim,
                int(architecture["hidden_dim"]),
                int(architecture["depth"]),
                float(architecture["dropout"]),
            )
            model = WholeGraphRegressor(
                encoder,
                int(architecture["hidden_dim"]),
                outputs=1,
            )
            model.load_state_dict(member["state_dict"], strict=True)
            prediction = graph_transfer._predict(
                model,
                records,
                indices,
                mean=target_mean,
                scale=target_scale,
                model_config=model_package["model_config"],
            )
        predictions.append(prediction)
    array = np.asarray(predictions, dtype=np.float64)
    if array.shape != (3, len(records)) or not np.all(np.isfinite(array)):
        raise OracleProductionError("production graph ensemble prediction is invalid")
    return {
        "endpoint": endpoint,
        "records": len(records),
        "ensemble_mean": np.mean(array, axis=0).tolist(),
        "ensemble_standard_deviation": np.std(array, axis=0).tolist(),
    }


def predict_production_ugi_smiles(
    checkpoint: Mapping[str, Any],
    candidates: Sequence[Mapping[str, str]],
    *,
    endpoint: str,
) -> dict[str, Any]:
    """Tensorize complete Ugi products and components under the checkpoint."""

    model_package = _require_verified_checkpoint(checkpoint)
    if model_package.get("lane") not in {
        "supervised_graph",
        "label_free_r0_transfer",
    }:
        raise OracleProductionError("selected production oracle is not graph-based")
    if not candidates:
        raise OracleProductionError("graph production inference received no candidates")
    required = {
        "label",
        "product_smiles",
        "amine_smiles",
        "aldehyde_smiles",
        "isocyanide_smiles",
    }
    vocabulary = _vocabulary_from_checkpoint(model_package.get("feature_vocabulary"))
    records: list[OracleGraphRecord] = []
    for index, candidate in enumerate(candidates):
        missing = required - set(candidate)
        if missing:
            raise OracleProductionError(f"graph candidate {index} lacks fields: {sorted(missing)}")
        label = str(candidate["label"])
        records.append(
            OracleGraphRecord(
                label=label,
                product=tensorize_smiles(
                    str(candidate["product_smiles"]),
                    vocabulary,
                    label=f"{label} product",
                ),
                amine=tensorize_smiles(
                    str(candidate["amine_smiles"]),
                    vocabulary,
                    label=f"{label} amine",
                ),
                aldehyde=tensorize_smiles(
                    str(candidate["aldehyde_smiles"]),
                    vocabulary,
                    label=f"{label} aldehyde",
                ),
                isocyanide=tensorize_smiles(
                    str(candidate["isocyanide_smiles"]),
                    vocabulary,
                    label=f"{label} isocyanide",
                ),
                targets=torch.zeros(2, dtype=torch.float32),
            )
        )
    return predict_production_graph_records(
        checkpoint,
        records,
        endpoint=endpoint,
    )


def predict_production_smiles(
    checkpoint: Mapping[str, Any],
    smiles: Sequence[str],
    *,
    endpoint: str,
) -> dict[str, Any]:
    """Predict constitutional SMILES with a hash-verified classical bundle."""

    model_package = _require_verified_checkpoint(checkpoint)
    if model_package.get("lane") != "classical":
        raise OracleProductionError("selected production oracle is not classical")
    endpoint_package = model_package.get("endpoint_models", {}).get(endpoint)
    if not isinstance(endpoint_package, Mapping):
        raise OracleProductionError(f"production checkpoint lacks endpoint {endpoint!r}")
    ensemble = endpoint_package.get("ensemble")
    if not isinstance(ensemble, list) or len(ensemble) != 3:
        raise OracleProductionError("production classical ensemble must contain three models")
    if not smiles:
        raise OracleProductionError("classical production inference received no structures")
    frame = pd.DataFrame(
        {
            "label": [f"production-{index}" for index in range(len(smiles))],
            "model_smiles": [str(value) for value in smiles],
        }
    )
    bundle = build_feature_bundle(frame, model_package["representation_config"])
    representation = str(model_package["representation"])
    observed_feature_names = list(bundle.feature_names[representation])
    if observed_feature_names != model_package.get("feature_names"):
        raise OracleProductionError("production classical feature names or ordering changed")
    features = bundle.matrices[representation]
    predictions = []
    for member in ensemble:
        if not isinstance(member, Mapping) or not isinstance(
            member.get("trusted_local_pickle"), bytes
        ):
            raise OracleProductionError("production classical ensemble member is invalid")
        payload = member["trusted_local_pickle"]
        if hashlib.sha256(payload).hexdigest() != member.get("pickle_sha256"):
            raise OracleProductionError("production estimator payload hash changed")
        estimator = pickle.loads(payload)  # noqa: S301 - outer checkpoint hash is required.
        prediction = np.asarray(estimator.predict(features), dtype=np.float64)
        predictions.append(prediction)
    array = np.asarray(predictions, dtype=np.float64)
    if array.shape != (3, len(smiles)) or not np.all(np.isfinite(array)):
        raise OracleProductionError("production classical ensemble prediction is invalid")
    return {
        "endpoint": endpoint,
        "records": len(smiles),
        "ensemble_mean": np.mean(array, axis=0).tolist(),
        "ensemble_standard_deviation": np.std(array, axis=0).tolist(),
    }


def _conservative_domain_scores(
    checkpoint: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    domain: str,
    coverage: float = 0.9,
) -> dict[str, Any]:
    """Apply the frozen domain radius or return an explicit abstention."""

    _require_verified_checkpoint(checkpoint)
    endpoint = str(prediction.get("endpoint"))
    domain_policy = (
        checkpoint.get("uncertainty", {}).get("domain_radii", {}).get(endpoint, {}).get(domain)
    )
    if not isinstance(domain_policy, Mapping):
        raise OracleProductionError(f"production uncertainty lacks {endpoint}/{domain}")
    if domain_policy.get("authorized") is not True:
        return {
            "endpoint": endpoint,
            "domain": domain,
            "action": "abstain",
            "scores": None,
        }
    percent = int(round(100 * coverage))
    radius = _finite_float(
        domain_policy.get("radii", {}).get(f"q{percent}"),
        f"{endpoint}/{domain}/q{percent}",
    )
    means = prediction.get("ensemble_mean")
    if not isinstance(means, list):
        raise OracleProductionError("prediction lacks ensemble means")
    scores = [_finite_float(value, "ensemble mean") - radius for value in means]
    return {
        "endpoint": endpoint,
        "domain": domain,
        "action": "empirical_held_domain_radius_adjusted_guidance",
        "nominal_coverage_reference": coverage,
        "radius": radius,
        "scores": scores,
    }


def classify_production_ugi_candidate(
    checkpoint: Mapping[str, Any],
    candidate: Mapping[str, str],
) -> dict[str, Any]:
    """Verify Ugi reconstruction and infer novelty from pinned component sets."""

    _require_verified_checkpoint(checkpoint)
    required = {
        "label",
        "product_smiles",
        "amine_smiles",
        "aldehyde_smiles",
        "isocyanide_smiles",
    }
    missing = required - set(candidate)
    if missing:
        raise OracleProductionError(f"candidate lacks applicability fields: {sorted(missing)}")
    label = str(candidate["label"])
    canonical = {
        "product": _canonical_constitutional_smiles(
            str(candidate["product_smiles"]),
            f"{label} product",
        ),
        "amine": _canonical_constitutional_smiles(
            str(candidate["amine_smiles"]),
            f"{label} amine",
        ),
        "aldehyde": _canonical_constitutional_smiles(
            str(candidate["aldehyde_smiles"]),
            f"{label} aldehyde",
        ),
        "isocyanide": _canonical_constitutional_smiles(
            str(candidate["isocyanide_smiles"]),
            f"{label} isocyanide",
        ),
    }
    chemistry = checkpoint.get("chemistry")
    if not isinstance(chemistry, Mapping):
        raise OracleProductionError("production checkpoint lacks chemistry")
    reaction = rdChemReactions.ReactionFromSmarts(str(chemistry.get("reaction_smarts", "")))
    if reaction is None:
        raise OracleProductionError("production checkpoint reaction did not compile")
    compiled = QualifiedForwardReaction(
        reaction_id=str(chemistry["reaction_id"]),
        role_names=tuple(str(value) for value in chemistry["role_names"]),
        reaction=reaction,
    )
    try:
        products = unique_forward_products(
            compiled,
            (
                canonical["amine"],
                canonical["aldehyde"],
                canonical["isocyanide"],
            ),
            max_products=int(chemistry["maximum_forward_products"]),
            isomeric_smiles=False,
        )
    except QualifiedForwardError as exc:
        return {
            "label": label,
            "domain": "non_ugi_final_assembly",
            "exact_forward_verified": False,
            "reason": str(exc),
            "canonical": canonical,
        }
    if canonical["product"] not in set(products):
        return {
            "label": label,
            "domain": "non_ugi_final_assembly",
            "exact_forward_verified": False,
            "forward_product_count": len(products),
            "canonical": canonical,
        }
    index = checkpoint.get("applicability_index")
    if not isinstance(index, Mapping):
        raise OracleProductionError("production checkpoint lacks applicability index")
    component_sets = {
        role: set(str(value) for value in index["components"][role])
        for role in ("amine", "aldehyde", "isocyanide")
    }
    unseen = tuple(
        role
        for role in ("amine", "aldehyde", "isocyanide")
        if canonical[role] not in component_sets[role]
    )
    domain_by_unseen = {
        (): "known_components_novel_combination",
        ("amine",): "unseen_head",
        ("aldehyde",): "unseen_aldehyde",
        ("isocyanide",): "unseen_isocyanide",
        ("amine", "aldehyde"): "unseen_head_and_aldehyde",
        ("amine", "isocyanide"): "unseen_head_and_isocyanide",
        ("aldehyde", "isocyanide"): "unseen_aldehyde_and_isocyanide",
        ("amine", "aldehyde", "isocyanide"): "three_unseen_components",
    }
    domain = domain_by_unseen[unseen]
    triple = "\x1f".join(
        (
            canonical["amine"],
            canonical["aldehyde"],
            canonical["isocyanide"],
        )
    )
    return {
        "label": label,
        "domain": domain,
        "exact_forward_verified": True,
        "forward_product_count": len(products),
        "unseen_component_roles": list(unseen),
        "combination_seen_in_measured_training": triple in set(index["measured_component_triples"]),
        "canonical": canonical,
    }


def score_production_ugi_candidates(
    checkpoint: Mapping[str, Any],
    candidates: Sequence[Mapping[str, str]],
    *,
    endpoint: str,
    nominal_coverage_reference: float = 0.9,
) -> dict[str, Any]:
    """Predict and apply only automatically inferred, verified Ugi domains."""

    classifications = [
        classify_production_ugi_candidate(checkpoint, candidate) for candidate in candidates
    ]
    model_package = _require_verified_checkpoint(checkpoint)
    if model_package.get("lane") == "classical":
        prediction = predict_production_smiles(
            checkpoint,
            [str(candidate["product_smiles"]) for candidate in candidates],
            endpoint=endpoint,
        )
    else:
        prediction = predict_production_ugi_smiles(
            checkpoint,
            candidates,
            endpoint=endpoint,
        )
    means = prediction["ensemble_mean"]
    standard_deviations = prediction["ensemble_standard_deviation"]
    scored = []
    for index, classification in enumerate(classifications):
        one_prediction = {
            "endpoint": endpoint,
            "ensemble_mean": [float(means[index])],
        }
        adjusted = _conservative_domain_scores(
            checkpoint,
            one_prediction,
            domain=str(classification["domain"]),
            coverage=nominal_coverage_reference,
        )
        scored.append(
            {
                **classification,
                "ensemble_mean": float(means[index]),
                "ensemble_standard_deviation": float(standard_deviations[index]),
                "guidance_action": adjusted["action"],
                "guidance_score": (
                    None if adjusted["scores"] is None else float(adjusted["scores"][0])
                ),
                "empirical_held_domain_radius": adjusted.get("radius"),
            }
        )
    return {
        "endpoint": endpoint,
        "records": len(scored),
        "domain_source": "checkpoint_owned_component_identity_and_exact_forward_verification",
        "candidates": scored,
    }
