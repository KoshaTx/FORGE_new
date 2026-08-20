"""Distribution-aware applicability audit for generated Ugi lipids.

The earlier fresh-pool oracle audit records whether exact precursor graphs were
present in the measured AGILE corpus.  Exact identity is provenance, not an
applicability domain.  This additive audit keeps that provenance field and
independently measures chemical distance in four views: complete product,
amine, aldehyde and isocyanide.

Applicability thresholds are learned without targets, predictions or errors.
For each held-component split, calibration structures are compared with that
fold's training structures.  Pooled calibration-distance quantiles then define
``interpolative``, ``boundary`` and ``extrapolative`` bins.  Only after those
thresholds are frozen in memory are the selected oracle's outer-test
predictions summarized within bins.  Generated candidates are compared with
the full measured corpus.  The audit is descriptive and cannot authorize
biological guidance or candidate selection.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors

from forge.core.hashing import sha256_bytes, sha256_file
from forge.core.io import csv_gz_bytes as _csv_bytes

CONFIG_SCHEMA_VERSION = "phase1_ugi_distributional_applicability_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_distributional_applicability.v1"
GENERATED_LEDGER_SCHEMA_VERSION = "phase1_ugi_generated_distributional_applicability.v1"
HELDOUT_LEDGER_SCHEMA_VERSION = "phase1_ugi_heldout_distributional_applicability.v1"

SELECTED_REPRESENTATION = "ugi_component_role_aware_dmpnn"
SELECTED_MODEL = "neural_3seed_ensemble"
ENDPOINT = "expt_Hela"

ROLE_FIELDS = {
    "amine": "A_smiles",
    "aldehyde": "B_smiles",
    "isocyanide": "C_smiles",
}
GENERATED_ROLE_FIELDS = {
    "amine": "amine_smiles",
    "aldehyde": "aldehyde_smiles",
    "isocyanide": "isocyanide_smiles",
}
ROLE_SCHEMES = {
    "amine": "held_head_5fold",
    "aldehyde": "held_aldehyde_5fold",
    "isocyanide": "held_isocyanide_5fold",
}
EVALUATION_SCHEMES = (
    "held_head_5fold",
    "held_aldehyde_5fold",
    "held_isocyanide_5fold",
    "held_head_aldehyde_pair_5fold",
    "held_head_isocyanide_pair_5fold",
    "held_aldehyde_isocyanide_pair_5fold",
    "lantern_scaffold_balanced",
)
VIEWS = ("product", "amine", "aldehyde", "isocyanide")
DISTANCE_TYPES = ("fingerprint", "descriptor")
BIN_ORDER = {"interpolative": 0, "boundary": 1, "extrapolative": 2}

DESCRIPTOR_NAMES = (
    "heavy_atoms",
    "carbon_atoms",
    "hetero_atoms",
    "molecular_weight",
    "logp",
    "tpsa",
    "rotatable_bonds",
    "ring_count",
    "formal_charge_abs",
    "double_bonds",
    "triple_bonds",
    "branch_excess",
    "ester_count",
    "amide_count",
    "ether_count",
)

GENERATED_FIELDS = (
    "sample_index",
    "product_id",
    "product_smiles",
    "amine_smiles",
    "aldehyde_smiles",
    "isocyanide_smiles",
    "exact_identity_provenance",
    "exact_unseen_roles_json",
    "product_fingerprint_distance",
    "product_descriptor_distance",
    "amine_fingerprint_distance",
    "amine_descriptor_distance",
    "aldehyde_fingerprint_distance",
    "aldehyde_descriptor_distance",
    "isocyanide_fingerprint_distance",
    "isocyanide_descriptor_distance",
    "product_distribution_bin",
    "amine_distribution_bin",
    "aldehyde_distribution_bin",
    "isocyanide_distribution_bin",
    "overall_distribution_bin",
    "ensemble_mean_descriptive_only",
    "ensemble_standard_deviation",
    "guidance_action",
)

HELDOUT_FIELDS = (
    "scheme",
    "fold",
    "label",
    "distribution_bin",
    "product_fingerprint_distance",
    "product_descriptor_distance",
    "amine_fingerprint_distance",
    "amine_descriptor_distance",
    "aldehyde_fingerprint_distance",
    "aldehyde_descriptor_distance",
    "isocyanide_fingerprint_distance",
    "isocyanide_descriptor_distance",
    "y_true",
    "y_pred",
    "absolute_error",
    "conformal_q90",
    "covered90",
)


class UgiDistributionalApplicabilityError(ValueError):
    """Raised when the distribution-aware audit violates its frozen contract."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiDistributionalApplicabilityError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiDistributionalApplicabilityError(f"{label} must contain one JSON object")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        if path.suffix == ".gz":
            handle = gzip.open(path, "rt", newline="")
        else:
            handle = path.open("r", newline="")
        with handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise UgiDistributionalApplicabilityError(f"invalid CSV: {path}") from error
    if not rows:
        raise UgiDistributionalApplicabilityError(f"CSV has no rows: {path}")
    return rows


def _verified_inputs(config: Mapping[str, Any], repo: Path) -> dict[str, Path]:
    expected = {
        "curated_agile",
        "fresh_pool_oracle_predictions",
        "oracle_graph_metrics",
        "oracle_graph_predictions",
        "oracle_split_assignments",
        "source",
        "runner",
        "tests",
    }
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != expected:
        raise UgiDistributionalApplicabilityError("applicability input set changed")
    output: dict[str, Path] = {}
    for label, specification in inputs.items():
        if not isinstance(specification, Mapping) or set(specification) != {"path", "sha256"}:
            raise UgiDistributionalApplicabilityError(f"input {label} pin is malformed")
        path = (repo / str(specification["path"])).resolve()
        try:
            path.relative_to(repo)
        except ValueError as error:
            raise UgiDistributionalApplicabilityError(
                f"input {label} escapes repository"
            ) from error
        if sha256_file(path) != specification["sha256"]:
            raise UgiDistributionalApplicabilityError(f"input {label} hash changed")
        output[label] = path
    return output


@cache
def _molecule(smiles: str) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise UgiDistributionalApplicabilityError(f"invalid molecular graph: {smiles!r}")
    return molecule


@cache
def _canonical(smiles: str) -> str:
    return Chem.MolToSmiles(_molecule(smiles), canonical=True, isomericSmiles=False)


_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_ESTER = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
_AMIDE = Chem.MolFromSmarts("[CX3](=O)[NX3]")
_ETHER = Chem.MolFromSmarts("[#6][OX2][#6]")


@cache
def _fingerprint(smiles: str):
    return _MORGAN.GetFingerprint(_molecule(smiles))


@cache
def _descriptors(smiles: str) -> tuple[float, ...]:
    molecule = _molecule(smiles)
    atoms = list(molecule.GetAtoms())
    bonds = list(molecule.GetBonds())
    heavy_atoms = sum(atom.GetAtomicNum() > 1 for atom in atoms)
    carbon_atoms = sum(atom.GetAtomicNum() == 6 for atom in atoms)
    hetero_atoms = sum(atom.GetAtomicNum() not in (1, 6) for atom in atoms)
    double_bonds = sum(bond.GetBondType() == Chem.BondType.DOUBLE for bond in bonds)
    triple_bonds = sum(bond.GetBondType() == Chem.BondType.TRIPLE for bond in bonds)
    branch_excess = sum(max(0, atom.GetDegree() - 2) for atom in atoms if atom.GetAtomicNum() > 1)
    return (
        float(heavy_atoms),
        float(carbon_atoms),
        float(hetero_atoms),
        float(Descriptors.MolWt(molecule)),
        float(Crippen.MolLogP(molecule)),
        float(rdMolDescriptors.CalcTPSA(molecule)),
        float(Lipinski.NumRotatableBonds(molecule)),
        float(rdMolDescriptors.CalcNumRings(molecule)),
        float(sum(abs(atom.GetFormalCharge()) for atom in atoms)),
        float(double_bonds),
        float(triple_bonds),
        float(branch_excess),
        float(len(molecule.GetSubstructMatches(_ESTER))),
        float(len(molecule.GetSubstructMatches(_AMIDE))),
        float(len(molecule.GetSubstructMatches(_ETHER))),
    )


@dataclass(frozen=True)
class DistancePair:
    fingerprint: float
    descriptor: float
    exact_identity_seen: bool


class ChemicalReference:
    """Nearest-neighbor chemical reference with robust descriptor scaling."""

    def __init__(self, smiles: Iterable[str]) -> None:
        canonical = tuple(sorted({_canonical(str(value)) for value in smiles}))
        if not canonical:
            raise UgiDistributionalApplicabilityError("chemical reference is empty")
        self.canonical = canonical
        self.canonical_set = frozenset(canonical)
        self.fingerprints = tuple(_fingerprint(value) for value in canonical)
        matrix = np.asarray([_descriptors(value) for value in canonical], dtype=np.float64)
        self.descriptors = matrix
        self.center = np.median(matrix, axis=0)
        q25 = np.quantile(matrix, 0.25, axis=0)
        q75 = np.quantile(matrix, 0.75, axis=0)
        scale = q75 - q25
        standard = np.std(matrix, axis=0)
        self.scale = np.where(scale > 1e-12, scale, np.where(standard > 1e-12, standard, 1.0))
        self.standardized = (matrix - self.center) / self.scale

    def distance(self, smiles: str) -> DistancePair:
        canonical = _canonical(smiles)
        similarities = DataStructs.BulkTanimotoSimilarity(
            _fingerprint(canonical), self.fingerprints
        )
        fingerprint = 1.0 - float(max(similarities))
        query = (np.asarray(_descriptors(canonical), dtype=np.float64) - self.center) / self.scale
        descriptor = float(
            np.min(np.linalg.norm(self.standardized - query, axis=1))
            / math.sqrt(len(DESCRIPTOR_NAMES))
        )
        if not (math.isfinite(fingerprint) and math.isfinite(descriptor)):
            raise UgiDistributionalApplicabilityError("chemical distance is not finite")
        return DistancePair(
            fingerprint=fingerprint,
            descriptor=descriptor,
            exact_identity_seen=canonical in self.canonical_set,
        )


def _quantile(values: Sequence[float], probability: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.all(np.isfinite(array)):
        raise UgiDistributionalApplicabilityError("distance threshold input is invalid")
    return float(np.quantile(array, probability, method="linear"))


def _distance_fields(
    references: Mapping[str, ChemicalReference],
    row: Mapping[str, str],
    *,
    generated: bool,
) -> dict[str, DistancePair]:
    output = {"product": references["product"].distance(row["product_smiles"])}
    fields = GENERATED_ROLE_FIELDS if generated else ROLE_FIELDS
    for role, field in fields.items():
        output[role] = references[role].distance(row[field])
    return output


def _references(rows: Sequence[Mapping[str, str]]) -> dict[str, ChemicalReference]:
    return {
        "product": ChemicalReference(row["product_smiles"] for row in rows),
        **{
            role: ChemicalReference(row[field] for row in rows)
            for role, field in ROLE_FIELDS.items()
        },
    }


def _training_rows(
    curated: Mapping[str, Mapping[str, str]],
    assignments: Mapping[tuple[str, int, str], str],
    scheme: str,
    fold: int,
    stage: str,
) -> list[Mapping[str, str]]:
    labels = [
        label
        for (candidate_scheme, candidate_fold, label), candidate_stage in assignments.items()
        if candidate_scheme == scheme and candidate_fold == fold and candidate_stage == stage
    ]
    if not labels:
        raise UgiDistributionalApplicabilityError(f"split has no {stage} records: {scheme}/{fold}")
    try:
        return [curated[label] for label in sorted(labels)]
    except KeyError as error:
        raise UgiDistributionalApplicabilityError(
            "split label is absent from curated AGILE"
        ) from error


def _split_index(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, int, str], str]:
    output: dict[tuple[str, int, str], str] = {}
    for row in rows:
        key = (str(row["scheme"]), int(row["fold"]), str(row["label"]))
        if key in output:
            raise UgiDistributionalApplicabilityError("duplicate split assignment")
        output[key] = str(row["stage"])
    return output


def _calibration_thresholds(
    curated: Mapping[str, Mapping[str, str]],
    assignments: Mapping[tuple[str, int, str], str],
    *,
    lower_quantile: float,
    upper_quantile: float,
) -> dict[str, dict[str, dict[str, float]]]:
    collected: dict[str, dict[str, list[float]]] = {
        view: {kind: [] for kind in DISTANCE_TYPES} for view in VIEWS
    }
    for role, scheme in ROLE_SCHEMES.items():
        folds = sorted(
            {fold for candidate_scheme, fold, _ in assignments if candidate_scheme == scheme}
        )
        for fold in folds:
            train = _training_rows(curated, assignments, scheme, fold, "train")
            calibration = _training_rows(curated, assignments, scheme, fold, "calibration")
            references = _references(train)
            for row in calibration:
                distances = _distance_fields(references, row, generated=False)
                for view in ("product", role):
                    collected[view]["fingerprint"].append(distances[view].fingerprint)
                    collected[view]["descriptor"].append(distances[view].descriptor)
    thresholds: dict[str, dict[str, dict[str, float]]] = {}
    for view in VIEWS:
        thresholds[view] = {}
        for kind in DISTANCE_TYPES:
            values = collected[view][kind]
            if not values:
                raise UgiDistributionalApplicabilityError(
                    f"calibration produced no {view}/{kind} distances"
                )
            thresholds[view][kind] = {
                "interpolative_max": _quantile(values, lower_quantile),
                "boundary_max": _quantile(values, upper_quantile),
                "calibration_records": len(values),
            }
    return thresholds


def _view_bin(distance: DistancePair, thresholds: Mapping[str, Mapping[str, float]]) -> str:
    if (
        distance.fingerprint <= thresholds["fingerprint"]["interpolative_max"]
        and distance.descriptor <= thresholds["descriptor"]["interpolative_max"]
    ):
        return "interpolative"
    if (
        distance.fingerprint <= thresholds["fingerprint"]["boundary_max"]
        and distance.descriptor <= thresholds["descriptor"]["boundary_max"]
    ):
        return "boundary"
    return "extrapolative"


def _bins(
    distances: Mapping[str, DistancePair],
    thresholds: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> tuple[dict[str, str], str]:
    views = {view: _view_bin(distances[view], thresholds[view]) for view in VIEWS}
    overall = max(views.values(), key=BIN_ORDER.__getitem__)
    return views, overall


def _metric_rows(
    rows: Sequence[Mapping[str, str]],
) -> dict[tuple[str, int], Mapping[str, str]]:
    output = {}
    for row in rows:
        if (
            row["representation"] == SELECTED_REPRESENTATION
            and row["model"] == SELECTED_MODEL
            and row["endpoint"] == ENDPOINT
            and row["scheme"] in EVALUATION_SCHEMES
        ):
            key = (row["scheme"], int(row["fold"]))
            if key in output:
                raise UgiDistributionalApplicabilityError("duplicate selected metric row")
            output[key] = row
    if not output:
        raise UgiDistributionalApplicabilityError("selected oracle metric rows are absent")
    return output


def _selected_predictions(rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    output = [
        row
        for row in rows
        if row["representation"] == SELECTED_REPRESENTATION
        and row["model"] == SELECTED_MODEL
        and row["endpoint"] == ENDPOINT
        and row["scheme"] in EVALUATION_SCHEMES
    ]
    if not output:
        raise UgiDistributionalApplicabilityError("selected oracle predictions are absent")
    return output


def _heldout_ledger(
    curated: Mapping[str, Mapping[str, str]],
    assignments: Mapping[tuple[str, int, str], str],
    predictions: Sequence[Mapping[str, str]],
    metrics: Mapping[tuple[str, int], Mapping[str, str]],
    thresholds: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> list[dict[str, Any]]:
    references: dict[tuple[str, int], dict[str, ChemicalReference]] = {}
    output = []
    for prediction in predictions:
        scheme = str(prediction["scheme"])
        fold = int(prediction["fold"])
        label = str(prediction["label"])
        if assignments.get((scheme, fold, label)) != "test":
            raise UgiDistributionalApplicabilityError("prediction is not an outer-test record")
        key = (scheme, fold)
        if key not in references:
            references[key] = _references(
                _training_rows(curated, assignments, scheme, fold, "train")
            )
        row = curated[label]
        distances = _distance_fields(references[key], row, generated=False)
        _, overall = _bins(distances, thresholds)
        q90 = float(metrics[key]["conformal_q90"])
        absolute_error = float(prediction["absolute_error"])
        record: dict[str, Any] = {
            "scheme": scheme,
            "fold": fold,
            "label": label,
            "distribution_bin": overall,
            "y_true": float(prediction["y_true"]),
            "y_pred": float(prediction["ensemble_y_pred"]),
            "absolute_error": absolute_error,
            "conformal_q90": q90,
            "covered90": absolute_error <= q90,
        }
        for view in VIEWS:
            record[f"{view}_fingerprint_distance"] = distances[view].fingerprint
            record[f"{view}_descriptor_distance"] = distances[view].descriptor
        output.append(record)
    return sorted(output, key=lambda row: (row["scheme"], row["fold"], row["label"]))


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _performance(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "records": 0,
            "r2": None,
            "rmse": None,
            "spearman_rho": None,
            "coverage90": None,
            "absolute_coverage90_gap": None,
        }
    truth = np.asarray([float(row["y_true"]) for row in rows], dtype=np.float64)
    prediction = np.asarray([float(row["y_pred"]) for row in rows], dtype=np.float64)
    residual = truth - prediction
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    r2 = None if denominator <= 0 else float(1.0 - np.sum(residual**2) / denominator)
    rho = None
    if len(rows) > 1:
        truth_rank = _rank(truth)
        prediction_rank = _rank(prediction)
        if np.std(truth_rank) > 0 and np.std(prediction_rank) > 0:
            rho = float(np.corrcoef(truth_rank, prediction_rank)[0, 1])
    coverage = float(np.mean([bool(row["covered90"]) for row in rows]))
    return {
        "records": len(rows),
        "r2": r2,
        "rmse": float(math.sqrt(np.mean(residual**2))),
        "spearman_rho": rho,
        "coverage90": coverage,
        "absolute_coverage90_gap": abs(coverage - 0.9),
    }


def _heldout_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_bin: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_scheme_bin: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        bin_name = str(row["distribution_bin"])
        by_bin[bin_name].append(row)
        by_scheme_bin[(str(row["scheme"]), bin_name)].append(row)
    return {
        "overall_by_distribution_bin": {
            name: _performance(by_bin.get(name, [])) for name in BIN_ORDER
        },
        "by_scheme_and_distribution_bin": {
            scheme: {
                name: _performance(by_scheme_bin.get((scheme, name), [])) for name in BIN_ORDER
            }
            for scheme in EVALUATION_SCHEMES
        },
    }


def _generated_ledger(
    rows: Sequence[Mapping[str, str]],
    full_reference: Mapping[str, ChemicalReference],
    thresholds: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if row.get("guidance_action") != "abstain" or row.get("guidance_score") not in ("", None):
            raise UgiDistributionalApplicabilityError("frozen generated row was not abstained")
        distances = _distance_fields(full_reference, row, generated=True)
        view_bins, overall = _bins(distances, thresholds)
        unseen_roles = json.loads(row["unseen_component_roles_json"])
        if not isinstance(unseen_roles, list):
            raise UgiDistributionalApplicabilityError("unseen role provenance is malformed")
        if row["combination_seen_in_measured_training"].lower() == "true":
            identity = "exact_measured_combination"
        elif unseen_roles:
            identity = "exact_new_component_identity"
        else:
            identity = "exact_seen_components_novel_combination"
        record: dict[str, Any] = {
            "sample_index": int(row["sample_index"]),
            "product_id": row["product_id"],
            "product_smiles": row["product_smiles"],
            "amine_smiles": row["amine_smiles"],
            "aldehyde_smiles": row["aldehyde_smiles"],
            "isocyanide_smiles": row["isocyanide_smiles"],
            "exact_identity_provenance": identity,
            "exact_unseen_roles_json": json.dumps(unseen_roles, separators=(",", ":")),
            "overall_distribution_bin": overall,
            "ensemble_mean_descriptive_only": float(row["ensemble_mean"]),
            "ensemble_standard_deviation": float(row["ensemble_standard_deviation"]),
            "guidance_action": "abstain",
        }
        for view in VIEWS:
            record[f"{view}_fingerprint_distance"] = distances[view].fingerprint
            record[f"{view}_descriptor_distance"] = distances[view].descriptor
            record[f"{view}_distribution_bin"] = view_bins[view]
        output.append(record)
    return output


def _generated_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    bins = Counter(str(row["overall_distribution_bin"]) for row in rows)
    provenance = Counter(str(row["exact_identity_provenance"]) for row in rows)
    cross = Counter(
        (str(row["exact_identity_provenance"]), str(row["overall_distribution_bin"]))
        for row in rows
    )
    return {
        "records": len(rows),
        "distribution_bins": {name: bins[name] for name in BIN_ORDER},
        "exact_identity_provenance": dict(sorted(provenance.items())),
        "identity_by_distribution_bin": {
            identity: {name: cross[(identity, name)] for name in BIN_ORDER}
            for identity in sorted(provenance)
        },
        "all_guidance_actions_abstain": all(row["guidance_action"] == "abstain" for row in rows),
    }


def build_distributional_applicability_audit(
    repo: Path,
    config_path: Path,
) -> tuple[dict[str, Any], bytes, bytes]:
    """Build the nonselecting chemical-distribution audit."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _read_json(config_path, "distributional-applicability config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiDistributionalApplicabilityError("unsupported applicability config schema")
    policy = config.get("policy")
    if not isinstance(policy, Mapping):
        raise UgiDistributionalApplicabilityError("applicability policy is missing")
    if policy.get("endpoint") != ENDPOINT:
        raise UgiDistributionalApplicabilityError("only the HeLa campaign endpoint is allowed")
    if policy.get("threshold_inputs") != "structures_and_split_stage_only":
        raise UgiDistributionalApplicabilityError("threshold input policy changed")
    if policy.get("targets_predictions_and_errors_used_for_thresholds") is not False:
        raise UgiDistributionalApplicabilityError("outcomes cannot set applicability thresholds")
    if policy.get("biological_guidance_authorized") is not False:
        raise UgiDistributionalApplicabilityError("this audit cannot authorize guidance")
    if policy.get("candidate_selection_changed") is not False:
        raise UgiDistributionalApplicabilityError("this audit cannot change candidate selection")
    lower = float(policy.get("interpolative_calibration_quantile"))
    upper = float(policy.get("boundary_calibration_quantile"))
    if not 0.0 < lower < upper < 1.0:
        raise UgiDistributionalApplicabilityError("applicability quantiles are invalid")

    paths = _verified_inputs(config, repo)
    curated_rows = _read_csv(paths["curated_agile"])
    curated: dict[str, dict[str, str]] = {}
    for source in curated_rows:
        label = source["label"]
        if label in curated:
            raise UgiDistributionalApplicabilityError("curated AGILE label is duplicated")
        curated[label] = {
            **source,
            "product_smiles": source["model_smiles"],
        }
    assignments = _split_index(_read_csv(paths["oracle_split_assignments"]))
    thresholds = _calibration_thresholds(
        curated,
        assignments,
        lower_quantile=lower,
        upper_quantile=upper,
    )

    metric_rows = _metric_rows(_read_csv(paths["oracle_graph_metrics"]))
    selected_predictions = _selected_predictions(_read_csv(paths["oracle_graph_predictions"]))
    heldout = _heldout_ledger(
        curated,
        assignments,
        selected_predictions,
        metric_rows,
        thresholds,
    )
    generated_source = _read_csv(paths["fresh_pool_oracle_predictions"])
    generated = _generated_ledger(
        generated_source,
        _references(list(curated.values())),
        thresholds,
    )
    generated_bytes = _csv_bytes(generated, GENERATED_FIELDS)
    heldout_bytes = _csv_bytes(heldout, HELDOUT_FIELDS)

    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_distributional_applicability_audit_guidance_still_abstained",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "definitions": {
            "exact_identity_provenance": (
                "whether exact precursor graphs or the exact measured combination were observed"
            ),
            "distributional_applicability": (
                "multi-view chemical proximity calibrated without potency targets or predictions"
            ),
            "exact_new_does_not_imply_extrapolative": True,
            "interpolative_does_not_itself_authorize_guidance": True,
            "descriptor_names": list(DESCRIPTOR_NAMES),
            "fingerprint": "Morgan radius 2, 2048 bits, constitutional graph",
        },
        "thresholds": thresholds,
        "heldout_oracle_evidence": _heldout_summary(heldout),
        "generated_pool": _generated_summary(generated),
        "artifacts": {
            "generated_applicability.csv.gz": {
                "schema_version": GENERATED_LEDGER_SCHEMA_VERSION,
                "records": len(generated),
                "sha256": sha256_bytes(generated_bytes),
            },
            "heldout_applicability.csv.gz": {
                "schema_version": HELDOUT_LEDGER_SCHEMA_VERSION,
                "records": len(heldout),
                "sha256": sha256_bytes(heldout_bytes),
            },
        },
        "adjudication": {
            "biological_guidance_authorized": False,
            "candidate_selection_changed": False,
            "raw_oracle_mean_used_for_ranking": False,
            "thresholds_used_targets_predictions_or_errors": False,
            "next_gate": (
                "independent review of bin-specific outer-test performance and a versioned "
                "guidance authorization decision"
            ),
        },
    }
    logical = json.loads(json.dumps(result, sort_keys=True))
    result["result_sha256"] = hashlib.sha256(
        json.dumps(logical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result, generated_bytes, heldout_bytes


__all__ = [
    "ChemicalReference",
    "DistancePair",
    "UgiDistributionalApplicabilityError",
    "build_distributional_applicability_audit",
]
