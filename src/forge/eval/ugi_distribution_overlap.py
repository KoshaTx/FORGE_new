"""Directional chemical-support audit for generated Ugi lipids.

Exact constitutional identity, marginal descriptor range, directional
support-set proximity and biological-oracle applicability are deliberately
separate.  This module compares query and reference populations in both
directions.  A query's nonconformity percentile is calibrated against
identity-excluded draws from the reference population, so an exact-new
molecule can remain in a dense structural neighborhood.  The audit does not
estimate a probability-density ratio or authorize biological guidance.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from rdkit import DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from sklearn.neighbors import NearestNeighbors

from forge.bio import ugi_distributional_applicability as chemistry
from forge.data.r1_prime_audit import sha256_bytes, sha256_file
from forge.product.ugi_morphology_corpus import source_stratified_family_weights

CONFIG_SCHEMA_VERSION = "phase1_ugi_distribution_overlap_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_distribution_overlap.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_distribution_overlap_generated_ledger.v1"

VIEWS = ("product", "amine", "aldehyde", "isocyanide")
ROLE_FIELDS = {
    "amine": "amine_smiles",
    "aldehyde": "aldehyde_smiles",
    "isocyanide": "isocyanide_smiles",
}
DISTANCE_KINDS = ("fingerprint", "descriptor")
COUNT_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


class UgiDistributionOverlapError(ValueError):
    """Raised when the overlap audit violates its frozen contract."""


@cache
def _canonical(smiles: str) -> str:
    return chemistry._canonical(smiles)


@cache
def _fingerprint(smiles: str):
    return COUNT_MORGAN.GetCountFingerprint(chemistry._molecule(_canonical(smiles)))


@cache
def _descriptor(smiles: str) -> tuple[float, ...]:
    return chemistry._descriptors(_canonical(smiles))


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        if path.suffix == ".gz":
            handle = gzip.open(path, "rt", newline="")
        else:
            handle = path.open("rt", newline="")
        with handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise UgiDistributionOverlapError(f"invalid CSV: {path}") from error
    if not rows:
        raise UgiDistributionOverlapError(f"CSV has no rows: {path}")
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiDistributionOverlapError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise UgiDistributionOverlapError(f"JSON must contain an object: {path}")
    return value


def _verified_inputs(config: Mapping[str, Any], repo: Path) -> dict[str, Path]:
    expected = {
        "balanced_assignments",
        "broad_r0",
        "canonicalization_and_descriptors",
        "curated_agile",
        "generated_pool",
        "runner",
        "sampling_implementation",
        "source",
        "tests",
        "training_config",
    }
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != expected:
        raise UgiDistributionOverlapError("support-audit input set changed")
    output = {}
    for name, record in inputs.items():
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise UgiDistributionOverlapError(f"invalid input record: {name}")
        path = (repo / str(record["path"])).resolve()
        try:
            path.relative_to(repo)
        except ValueError as error:
            raise UgiDistributionOverlapError(f"input escapes repository: {name}") from error
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise UgiDistributionOverlapError(f"input hash mismatch: {name}")
        output[str(name)] = path
    return output


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted({_canonical(value) for value in values}))


def _sample_rows(
    rows: Sequence[Mapping[str, str]], count: int, seed: int
) -> list[Mapping[str, str]]:
    if count >= len(rows):
        return list(rows)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(rows), size=count, replace=False))
    return [rows[int(index)] for index in indices]


def _weighted_sample_rows(
    rows: Sequence[Mapping[str, str]],
    weights: np.ndarray,
    count: int,
    seed: int,
) -> list[Mapping[str, str]]:
    if weights.shape != (len(rows),) or np.any(weights < 0):
        raise UgiDistributionOverlapError("invalid weighted query distribution")
    normalized = weights / weights.sum()
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(rows), size=count, replace=True, p=normalized)
    return [rows[int(index)] for index in indices]


@dataclass(frozen=True)
class DistancePair:
    fingerprint: float
    descriptor: float


class ViewReference:
    """Identity-excluded nearest-neighbor reference in two chemical views."""

    def __init__(
        self,
        smiles: Sequence[str],
        *,
        fingerprint_cap: int,
        seed: int,
    ) -> None:
        canonical = _unique(smiles)
        if len(canonical) < 2:
            raise UgiDistributionOverlapError("a view reference needs at least two identities")
        self.canonical = canonical
        self.canonical_set = frozenset(canonical)
        descriptor_matrix = np.asarray([_descriptor(value) for value in canonical])
        self.q01 = np.quantile(descriptor_matrix, 0.01, axis=0)
        self.q99 = np.quantile(descriptor_matrix, 0.99, axis=0)
        self.minimum = np.min(descriptor_matrix, axis=0)
        self.maximum = np.max(descriptor_matrix, axis=0)
        center = np.median(descriptor_matrix, axis=0)
        q25 = np.quantile(descriptor_matrix, 0.25, axis=0)
        q75 = np.quantile(descriptor_matrix, 0.75, axis=0)
        standard = np.std(descriptor_matrix, axis=0)
        scale = np.where(q75 - q25 > 1e-12, q75 - q25, standard)
        self.scale = np.where(scale > 1e-12, scale, 1.0)
        self.center = center
        standardized = (descriptor_matrix - center) / self.scale
        self.neighbors = NearestNeighbors(
            n_neighbors=min(3, len(canonical)), algorithm="auto", metric="euclidean"
        ).fit(standardized)

        if len(canonical) > fingerprint_cap:
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(len(canonical), fingerprint_cap, replace=False))
            fingerprint_canonical = tuple(canonical[int(index)] for index in indices)
        else:
            fingerprint_canonical = canonical
        self.fingerprint_canonical = fingerprint_canonical
        self.fingerprints = tuple(_fingerprint(value) for value in fingerprint_canonical)
        self.fingerprint_list = list(self.fingerprints)

    def distance(self, smiles: str) -> DistancePair:
        query = _canonical(smiles)
        query_descriptor = (np.asarray(_descriptor(query)) - self.center) / self.scale
        distances, indices = self.neighbors.kneighbors(query_descriptor.reshape(1, -1))
        descriptor_distance = None
        for distance, index in zip(distances[0], indices[0], strict=True):
            if self.canonical[int(index)] != query:
                descriptor_distance = float(distance / math.sqrt(len(chemistry.DESCRIPTOR_NAMES)))
                break
        if descriptor_distance is None:
            raise UgiDistributionOverlapError("descriptor identity exclusion exhausted neighbors")

        similarities = DataStructs.BulkTanimotoSimilarity(
            _fingerprint(query), self.fingerprint_list
        )
        fingerprint_similarity = max(
            similarity
            for candidate, similarity in zip(self.fingerprint_canonical, similarities, strict=True)
            if candidate != query
        )
        return DistancePair(
            fingerprint=1.0 - float(fingerprint_similarity),
            descriptor=descriptor_distance,
        )

    def marginal_support(self, smiles: str) -> tuple[bool, bool, list[bool]]:
        values = np.asarray(_descriptor(_canonical(smiles)))
        within_full = bool(np.all((values >= self.minimum) & (values <= self.maximum)))
        flags = list((values >= self.q01) & (values <= self.q99))
        return within_full, bool(all(flags)), flags


def _references(
    rows: Sequence[Mapping[str, str]],
    views: Sequence[str],
    *,
    product_fingerprint_cap: int,
    seed: int,
) -> dict[str, ViewReference]:
    output = {}
    for offset, view in enumerate(views):
        field = "product_smiles" if view == "product" else ROLE_FIELDS[view]
        cap = product_fingerprint_cap if view == "product" else 100_000
        output[view] = ViewReference(
            [row[field] for row in rows], fingerprint_cap=cap, seed=seed + offset
        )
    return output


def _distance_rows(
    rows: Sequence[Mapping[str, str]],
    references: Mapping[str, ViewReference],
    views: Sequence[str],
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        record: dict[str, Any] = {}
        for view in views:
            field = "product_smiles" if view == "product" else ROLE_FIELDS[view]
            value = row[field]
            distance = references[view].distance(value)
            within_full, within_central, flags = references[view].marginal_support(value)
            record[f"{view}_fingerprint"] = distance.fingerprint
            record[f"{view}_descriptor"] = distance.descriptor
            record[f"{view}_exact_seen"] = _canonical(value) in references[view].canonical_set
            record[f"{view}_within_full_range"] = within_full
            record[f"{view}_within_q01_q99_all"] = within_central
            record[f"{view}_descriptor_flags"] = flags
        output.append(record)
    return output


def _midrank_percentile(reference: np.ndarray, value: float) -> float:
    """Return the deterministic empirical midrank for tied nonconformity values."""

    if reference.ndim != 1 or len(reference) < 1:
        raise UgiDistributionOverlapError(
            "percentile reference must be nonempty and one-dimensional"
        )
    left = int(np.searchsorted(reference, value, side="left"))
    right = int(np.searchsorted(reference, value, side="right"))
    return float((left + right) / (2.0 * len(reference)))


def _joint_percentiles(
    query: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    views: Sequence[str],
) -> list[float]:
    return [
        float(record["joint_reference_nonconformity_percentile"])
        for record in _joint_nonconformity_records(query, baseline, views)
    ]


def _joint_nonconformity_records(
    query: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    views: Sequence[str],
) -> list[dict[str, Any]]:
    keys = [f"{view}_{kind}" for view in views for kind in DISTANCE_KINDS]
    sorted_values = {
        key: np.sort(np.asarray([float(row[key]) for row in baseline])) for key in keys
    }
    baseline_joint = np.sort(
        np.asarray(
            [
                max(_midrank_percentile(sorted_values[key], float(row[key])) for key in keys)
                for row in baseline
            ]
        )
    )
    output = []
    for row in query:
        axis_percentiles = {
            key: _midrank_percentile(sorted_values[key], float(row[key])) for key in keys
        }
        maximum = max(axis_percentiles.values())
        dominant = min(
            key for key, value in axis_percentiles.items() if abs(value - maximum) <= 1e-12
        )
        output.append(
            {
                "joint_reference_nonconformity_percentile": _midrank_percentile(
                    baseline_joint, maximum
                ),
                "dominant_nonconformity_axis": dominant,
                "axis_nonconformity_percentiles": axis_percentiles,
            }
        )
    return output


def _comparison_summary(
    rows: Sequence[Mapping[str, Any]],
    nonconformity: Sequence[Mapping[str, Any]],
    views: Sequence[str],
) -> dict[str, Any]:
    if len(rows) != len(nonconformity):
        raise UgiDistributionOverlapError("query rows and nonconformity records differ")
    percentiles = [
        float(record["joint_reference_nonconformity_percentile"]) for record in nonconformity
    ]
    per_view = {}
    for view in views:
        per_view[view] = {
            "exact_identity_fraction": float(
                np.mean([bool(row[f"{view}_exact_seen"]) for row in rows])
            ),
            "within_full_marginal_range_fraction": float(
                np.mean([bool(row[f"{view}_within_full_range"]) for row in rows])
            ),
            "within_joint_q01_q99_descriptor_box_fraction": float(
                np.mean([bool(row[f"{view}_within_q01_q99_all"]) for row in rows])
            ),
            "median_identity_excluded_fingerprint_distance": float(
                np.median([float(row[f"{view}_fingerprint"]) for row in rows])
            ),
            "median_identity_excluded_descriptor_distance": float(
                np.median([float(row[f"{view}_descriptor"]) for row in rows])
            ),
            "marginal_q01_q99_coverage_by_descriptor": {
                name: float(np.mean([bool(row[f"{view}_descriptor_flags"][index]) for row in rows]))
                for index, name in enumerate(chemistry.DESCRIPTOR_NAMES)
            },
        }
    bins = Counter(
        (
            "central_reference_neighborhood"
            if value <= 0.5
            else (
                "extended_reference_neighborhood"
                if value <= 0.9
                else "above_reference_90th_nonconformity"
            )
        )
        for value in percentiles
    )
    return {
        "records": len(rows),
        "joint_reference_nonconformity_percentile": {
            "median": float(np.median(percentiles)),
            "q90": float(np.quantile(percentiles, 0.9)),
            "bins": {
                "central_reference_neighborhood": bins["central_reference_neighborhood"],
                "extended_reference_neighborhood": bins["extended_reference_neighborhood"],
                "above_reference_90th_nonconformity": bins["above_reference_90th_nonconformity"],
            },
            "dominant_axis_counts": dict(
                sorted(
                    Counter(
                        str(record["dominant_nonconformity_axis"]) for record in nonconformity
                    ).items()
                )
            ),
            "median_axis_nonconformity_percentile": {
                key: float(
                    np.median(
                        [
                            float(record["axis_nonconformity_percentiles"][key])
                            for record in nonconformity
                        ]
                    )
                )
                for key in sorted(nonconformity[0]["axis_nonconformity_percentiles"])
            },
        },
        "per_view": per_view,
    }


def _identity_reference(
    rows: Sequence[Mapping[str, str]],
) -> tuple[frozenset[str], frozenset[tuple[str, str, str]], dict[str, frozenset[str]]]:
    products = frozenset(_canonical(row["product_smiles"]) for row in rows)
    triples = frozenset(
        tuple(_canonical(row[ROLE_FIELDS[role]]) for role in ROLE_FIELDS) for row in rows
    )
    roles = {
        role: frozenset(_canonical(row[field]) for row in rows)
        for role, field in ROLE_FIELDS.items()
    }
    return products, triples, roles


def _identity_record(
    row: Mapping[str, str],
    reference: tuple[
        frozenset[str],
        frozenset[tuple[str, str, str]],
        dict[str, frozenset[str]],
    ],
) -> tuple[bool, bool, list[str]]:
    products, triples, roles = reference
    triple = tuple(_canonical(row[ROLE_FIELDS[role]]) for role in ROLE_FIELDS)
    unseen = [
        role for role, value in zip(ROLE_FIELDS, triple, strict=True) if value not in roles[role]
    ]
    return _canonical(row["product_smiles"]) in products, triple in triples, unseen


def _generated_ledger_bytes(
    rows: Sequence[Mapping[str, str]],
    train_nonconformity: Sequence[Mapping[str, Any]],
    measured_nonconformity: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, str]],
    measured_rows: Sequence[Mapping[str, str]],
) -> bytes:
    fields = (
        "sample_index",
        "product_id",
        "source_pool_unseen_component_roles_json",
        "generator_train_product_constitution_seen",
        "generator_train_exact_component_triple_seen",
        "generator_train_unseen_component_roles_json",
        "measured_agile_product_constitution_seen",
        "measured_agile_exact_component_triple_seen",
        "measured_agile_unseen_component_roles_json",
        "generator_reference_nonconformity_percentile",
        "generator_dominant_nonconformity_axis",
        "measured_reference_nonconformity_percentile",
        "measured_dominant_nonconformity_axis",
    )
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    train_reference = _identity_reference(train_rows)
    measured_reference = _identity_reference(measured_rows)
    for row, train_record, measured_record in zip(
        rows, train_nonconformity, measured_nonconformity, strict=True
    ):
        train_product, train_triple, train_unseen = _identity_record(row, train_reference)
        measured_product, measured_triple, measured_unseen = _identity_record(
            row, measured_reference
        )
        writer.writerow(
            {
                "sample_index": row["sample_index"],
                "product_id": row["product_id"],
                "source_pool_unseen_component_roles_json": row["unseen_component_roles_json"],
                "generator_train_product_constitution_seen": str(train_product).lower(),
                "generator_train_exact_component_triple_seen": str(train_triple).lower(),
                "generator_train_unseen_component_roles_json": json.dumps(
                    train_unseen, separators=(",", ":")
                ),
                "measured_agile_product_constitution_seen": str(measured_product).lower(),
                "measured_agile_exact_component_triple_seen": str(measured_triple).lower(),
                "measured_agile_unseen_component_roles_json": json.dumps(
                    measured_unseen, separators=(",", ":")
                ),
                "generator_reference_nonconformity_percentile": (
                    f"{float(train_record['joint_reference_nonconformity_percentile']):.12g}"
                ),
                "generator_dominant_nonconformity_axis": train_record[
                    "dominant_nonconformity_axis"
                ],
                "measured_reference_nonconformity_percentile": (
                    f"{float(measured_record['joint_reference_nonconformity_percentile']):.12g}"
                ),
                "measured_dominant_nonconformity_axis": measured_record[
                    "dominant_nonconformity_axis"
                ],
            }
        )
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as handle:
        handle.write(text.getvalue().encode())
    return output.getvalue()


def _train_rows(source: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    output = []
    for row in source:
        if row["primary_product_fold"] != "train":
            continue
        output.append(
            {
                "record_id": row["product_id"],
                "product_smiles": row["canonical_product_smiles"],
                "amine_smiles": row["amine_head_smiles"],
                "aldehyde_smiles": row["oxoester_aldehyde_body_tail_smiles"],
                "isocyanide_smiles": row["isocyanide_tail_smiles"],
            }
        )
    return output


def _measured_rows(source: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "record_id": row["label"],
            "product_smiles": row["model_smiles"],
            "amine_smiles": row["A_smiles"],
            "aldehyde_smiles": row["B_smiles"],
            "isocyanide_smiles": row["C_smiles"],
        }
        for row in source
    ]


def _generated_rows(source: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [
        {
            **row,
            "record_id": row["product_id"],
            "product_smiles": row["product_smiles"],
            "amine_smiles": row["amine_smiles"],
            "aldehyde_smiles": row["aldehyde_smiles"],
            "isocyanide_smiles": row["isocyanide_smiles"],
        }
        for row in source
    ]


def _non_agile_broad_rows(source: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "record_id": row["r0_structure_id"],
            "product_smiles": row["canonical_constitutional_smiles"],
        }
        for row in source
        if row["r0_pretraining_eligible"].lower() == "true"
        and not any(
            value.startswith("agile_") for value in row["all_available_source_ids"].split("|")
        )
    ]


def build_ugi_distribution_overlap_audit(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], bytes]:
    """Build the directional generator, measured and broad overlap matrix."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiDistributionOverlapError("unsupported overlap config schema")
    policy = config.get("policy")
    if not isinstance(policy, Mapping):
        raise UgiDistributionOverlapError("overlap policy is missing")
    required = {
        "exact_identity_is_not_distribution_status": True,
        "distance_calibration_excludes_exact_identity": True,
        "biological_targets_used_for_geometry": False,
        "biological_guidance_authorized": False,
        "candidate_selection_changed": False,
    }
    for key, expected in required.items():
        if policy.get(key) != expected:
            raise UgiDistributionOverlapError(f"overlap policy changed: {key}")
    seed = int(policy["seed"])
    query_count = int(policy["maximum_query_count"])
    fingerprint_cap = int(policy["product_fingerprint_reference_cap"])
    if query_count != 3975 or fingerprint_cap < query_count:
        raise UgiDistributionOverlapError("overlap sample-size policy changed")

    paths = _verified_inputs(config, repo)
    training_config = _read_json(paths["training_config"])
    sampling = training_config.get("sampling")
    if not isinstance(sampling, Mapping) or sampling.get("mode") != (
        "source_stratified_role_family_raked"
    ):
        raise UgiDistributionOverlapError("production training sampler changed")
    if sampling.get("uniform_product_row_sampling_allowed") is not False:
        raise UgiDistributionOverlapError("uniform product-row sampling remains prohibited")
    assignment_rows = _read_csv(paths["balanced_assignments"])
    train_source = [row for row in assignment_rows if row["primary_product_fold"] == "train"]
    train = _train_rows(train_source)
    source_mass = sampling.get("source_mass")
    if not isinstance(source_mass, Mapping):
        raise UgiDistributionOverlapError("production source-mass contract is missing")
    train_weights = source_stratified_family_weights(
        train_source,
        source_mass={str(key): float(value) for key, value in source_mass.items()},
        uniform_row_mixture=float(sampling["uniform_row_mixture"]),
    )
    measured = _measured_rows(_read_csv(paths["curated_agile"]))
    generated = _generated_rows(_read_csv(paths["generated_pool"]))
    broad = _non_agile_broad_rows(_read_csv(paths["broad_r0"]))
    if len(train) != 66_464 or len(measured) != 1_100 or len(generated) != 3_975:
        raise UgiDistributionOverlapError("frozen distribution census changed")
    if len(broad) != 13_919:
        raise UgiDistributionOverlapError("non-AGILE broad-R0 census changed")

    populations = {
        "generator_train": train,
        "measured_agile": measured,
        "generated": generated,
        "non_agile_broad_r0": broad,
    }
    query_populations = {
        "generator_train": _weighted_sample_rows(train, train_weights, query_count, seed + 1),
        "measured_agile": measured,
        "generated": generated,
        "non_agile_broad_r0": _sample_rows(broad, query_count, seed + 2),
    }
    reference_views = {
        "generator_train": VIEWS,
        "measured_agile": VIEWS,
        "generated": VIEWS,
        "non_agile_broad_r0": ("product",),
    }

    references = {
        name: _references(
            rows,
            reference_views[name],
            product_fingerprint_cap=fingerprint_cap,
            seed=seed + 100 * index,
        )
        for index, (name, rows) in enumerate(populations.items())
    }
    baselines = {
        name: _distance_rows(query_populations[name], references[name], reference_views[name])
        for name in populations
    }

    comparison_specs = (
        ("generated", "generator_train", VIEWS),
        ("generator_train", "generated", VIEWS),
        ("generated", "measured_agile", VIEWS),
        ("measured_agile", "generated", VIEWS),
        ("measured_agile", "generator_train", VIEWS),
        ("generator_train", "measured_agile", VIEWS),
        ("generated", "non_agile_broad_r0", ("product",)),
        ("non_agile_broad_r0", "generated", ("product",)),
    )
    comparisons = {}
    nonconformity_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for query_name, reference_name, views in comparison_specs:
        query_rows = query_populations[query_name]
        distances = _distance_rows(query_rows, references[reference_name], views)
        nonconformity = _joint_nonconformity_records(distances, baselines[reference_name], views)
        key = (query_name, reference_name)
        nonconformity_cache[key] = nonconformity
        comparisons[f"{query_name}_to_{reference_name}"] = _comparison_summary(
            distances, nonconformity, views
        )

    ledger_bytes = _generated_ledger_bytes(
        generated,
        nonconformity_cache[("generated", "generator_train")],
        nonconformity_cache[("generated", "measured_agile")],
        train,
        measured,
    )
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_directional_chemical_support_audit_guidance_abstained",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "inputs": {
            name: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for name, path in sorted(paths.items())
        },
        "definitions": {
            "exact_identity": (
                "constitutional provenance only; recomputed per reference and excluded "
                "from neighborhood distance"
            ),
            "marginal_support": "within each reference descriptor range",
            "joint_reference_nonconformity_percentile": (
                "worst multiview fingerprint/descriptor nonconformity, with tied distances "
                "assigned empirical midranks, calibrated against identity-excluded queries "
                "drawn from the named reference population"
            ),
            "central_reference_neighborhood": "nonconformity percentile <= 0.50",
            "extended_reference_neighborhood": ("0.50 < nonconformity percentile <= 0.90"),
            "above_reference_90th_nonconformity": "nonconformity percentile > 0.90",
            "support_is_directional": True,
            "reference_set_deduplicated": True,
            "density_ratio_estimated": False,
            "product_fingerprint_reference_is_capped": True,
            "broad_reference_scope": (
                "R0 structures with direct AGILE source records excluded; descriptive pooled "
                "non-AGILE support, not evidence of cross-platform generalization"
            ),
        },
        "census": {name: len(rows) for name, rows in populations.items()},
        "query_census": {name: len(rows) for name, rows in query_populations.items()},
        "policy": dict(policy),
        "training_sampler": {
            "mode": sampling["mode"],
            "source_mass": dict(source_mass),
            "uniform_row_mixture": float(sampling["uniform_row_mixture"]),
            "weighted_query_sampling_with_replacement": True,
        },
        "software": {
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
            "scikit_learn": sklearn.__version__,
        },
        "comparisons": comparisons,
        "artifacts": {
            "generated_overlap_ledger.csv.gz": {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "records": len(generated),
                "sha256": sha256_bytes(ledger_bytes),
            }
        },
        "adjudication": {
            "biological_guidance_authorized": False,
            "candidate_selection_changed": False,
            "v3_strict_core_reinterpreted_not_rewritten": True,
            "next_gate": (
                "calibrate oracle residual risk against the measured-reference "
                "nonconformity percentile before any potency guidance"
            ),
        },
    }
    logical = json.loads(json.dumps(result, sort_keys=True))
    result["result_sha256"] = hashlib.sha256(
        json.dumps(logical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result, ledger_bytes


__all__ = [
    "UgiDistributionOverlapError",
    "ViewReference",
    "_joint_percentiles",
    "_midrank_percentile",
    "build_ugi_distribution_overlap_audit",
]
