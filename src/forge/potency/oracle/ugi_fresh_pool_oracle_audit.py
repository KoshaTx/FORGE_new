"""Nonselecting HeLa-oracle audit for the frozen fresh Ugi product pool.

The frozen M0-07 oracle abstains in every evaluated guidance domain.  This
module therefore records endpoint predictions and automatically inferred Ugi
domains only as descriptive evidence.  It never ranks candidates, converts a
prediction into a guidance score, or reopens biological tilting.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_bytes, sha256_file
from forge.potency.audit.ugi_semantic_annotations import ROLE_NAMES
from forge.potency.oracle.oracle_production import (
    load_production_checkpoint,
    score_production_ugi_candidates,
)
from forge.value.synthesis.synthesis import ProductSynthesisValue

CONFIG_SCHEMA_VERSION = "phase1_ugi_fresh_pool_oracle_audit_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_fresh_pool_oracle_audit.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_fresh_pool_oracle_predictions.v1"

ROLE_TO_FIELD = {
    "amine_head": "amine_smiles",
    "oxoester_aldehyde_body_tail": "aldehyde_smiles",
    "isocyanide_tail": "isocyanide_smiles",
}

LEDGER_FIELDS = (
    "sample_index",
    "product_id",
    "structure_id",
    "product_smiles",
    "amine_smiles",
    "aldehyde_smiles",
    "isocyanide_smiles",
    "product_structural_provenance_stratum",
    "route_complete",
    "route_complete_component_count",
    "oracle_domain",
    "unseen_component_roles_json",
    "combination_seen_in_measured_training",
    "exact_forward_verified",
    "ensemble_mean",
    "ensemble_standard_deviation",
    "guidance_action",
    "guidance_score",
    "empirical_held_domain_radius",
)


class UgiFreshPoolOracleAuditError(ValueError):
    """Raised when the nonselecting oracle audit is not reproducible."""


def _load_json(path: Path, *, compressed: bool, label: str) -> dict[str, Any]:
    try:
        if compressed:
            with gzip.open(path, "rt") as handle:
                value = json.load(handle)
        else:
            value = json.loads(path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        raise UgiFreshPoolOracleAuditError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise UgiFreshPoolOracleAuditError(f"{label} must be an object")
    return value


def _validate_inputs(config: Mapping[str, Any], repo: Path) -> dict[str, Path]:
    specifications = config.get("inputs")
    required = {
        "production_generator_v2",
        "fresh_pool_sample",
        "fresh_pool_audit",
        "route_coverage_result",
        "route_product_values",
        "oracle_campaign_selection",
        "oracle_production_result",
        "oracle_production_checkpoint",
        "oracle_audit_source",
        "oracle_audit_runner",
        "oracle_audit_tests",
    }
    if not isinstance(specifications, dict) or set(specifications) != required:
        raise UgiFreshPoolOracleAuditError("oracle-audit input set changed")
    paths: dict[str, Path] = {}
    for label, specification in specifications.items():
        if not isinstance(specification, dict) or set(specification) != {"path", "sha256"}:
            raise UgiFreshPoolOracleAuditError(f"input {label} is malformed")
        path = (repo / str(specification["path"])).resolve()
        if sha256_file(path) != specification["sha256"]:
            raise UgiFreshPoolOracleAuditError(f"input hash changed: {label}")
        paths[label] = path
    return paths


def _exact_l1_candidates(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = sample.get("samples")
    if not isinstance(records, list):
        raise UgiFreshPoolOracleAuditError("fresh pool has no sample rows")
    candidates = []
    for sample_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise UgiFreshPoolOracleAuditError("fresh-pool sample row is malformed")
        if not (
            record.get("valid") is True
            and record.get("component_reconstruction_valid") is True
            and (record.get("l1_forward_verification") or {}).get("exact_product_reconstructed")
            is True
        ):
            continue
        components = record.get("component_smiles_by_role")
        if not isinstance(components, dict) or set(components) != set(ROLE_NAMES):
            raise UgiFreshPoolOracleAuditError(
                f"exact-L1 sample {sample_index} lacks three component roles"
            )
        candidate = {
            "sample_index": sample_index,
            "label": str(record["product_id"]),
            "product_id": str(record["product_id"]),
            "structure_id": str(record["structure_id"]),
            "product_smiles": str(record["smiles"]),
        }
        for role, field in ROLE_TO_FIELD.items():
            candidate[field] = str(components[role])
        candidates.append(candidate)
    return candidates


def _route_index(payload: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    records = payload.get("records")
    if not isinstance(records, list):
        raise UgiFreshPoolOracleAuditError("route product ledger lacks records")
    output = {}
    for record in records:
        if not isinstance(record, dict):
            raise UgiFreshPoolOracleAuditError("route product record is malformed")
        sample_index = int(record.get("sample_index", -1))
        if sample_index < 0 or sample_index in output:
            raise UgiFreshPoolOracleAuditError("route product sample index is invalid")
        value = ProductSynthesisValue.from_dict(record.get("value"))
        output[sample_index] = {
            "product_id": str(record.get("product_id")),
            "structure_id": str(record.get("structure_id")),
            "product_structural_provenance_stratum": str(
                record.get("product_structural_provenance_stratum")
            ),
            "route_complete": value.route_complete,
            "route_complete_component_count": sum(
                component.route_complete for _, component in value.components
            ),
        }
    return output


def _quantiles(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if len(finite) != len(values):
        raise UgiFreshPoolOracleAuditError("oracle output contains a nonfinite value")
    if not finite:
        return {
            "count": 0,
            "minimum": None,
            "q05": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q95": None,
            "maximum": None,
        }

    def select(probability: float) -> float:
        position = probability * (len(finite) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return finite[lower]
        fraction = position - lower
        return finite[lower] * (1.0 - fraction) + finite[upper] * fraction

    return {
        "count": len(finite),
        "minimum": finite[0],
        "q05": select(0.05),
        "q25": select(0.25),
        "median": select(0.5),
        "q75": select(0.75),
        "q95": select(0.95),
        "maximum": finite[-1],
    }


def summarize_oracle_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize predictions without ranking or authorizing any candidate."""

    domains: Counter[str] = Counter()
    route_complete_domains: Counter[str] = Counter()
    one_gap_domains: Counter[str] = Counter()
    route_component_counts: Counter[int] = Counter()
    provenance_counts: Counter[str] = Counter()
    domain_means: dict[str, list[float]] = defaultdict(list)
    means = []
    deviations = []
    measured_combinations = 0
    for row in rows:
        if row.get("guidance_action") != "abstain" or row.get("guidance_score") not in (None, ""):
            raise UgiFreshPoolOracleAuditError(
                "the nonselecting audit encountered an authorized guidance score"
            )
        if row.get("exact_forward_verified") is not True:
            raise UgiFreshPoolOracleAuditError("oracle candidate failed exact forward verification")
        domain = str(row["oracle_domain"])
        complete_count = int(row["route_complete_component_count"])
        mean = float(row["ensemble_mean"])
        deviation = float(row["ensemble_standard_deviation"])
        domains[domain] += 1
        route_component_counts[complete_count] += 1
        provenance_counts[str(row["product_structural_provenance_stratum"])] += 1
        if row.get("route_complete") is True:
            route_complete_domains[domain] += 1
        if complete_count == 2:
            one_gap_domains[domain] += 1
        if row.get("combination_seen_in_measured_training") is True:
            measured_combinations += 1
        means.append(mean)
        deviations.append(deviation)
        domain_means[domain].append(mean)
    return {
        "records": len(rows),
        "oracle_domains": dict(sorted(domains.items())),
        "route_complete_products_by_oracle_domain": dict(sorted(route_complete_domains.items())),
        "one_gap_products_by_oracle_domain": dict(sorted(one_gap_domains.items())),
        "products_by_route_complete_component_count": {
            str(key): value for key, value in sorted(route_component_counts.items())
        },
        "products_by_structural_provenance": dict(sorted(provenance_counts.items())),
        "measured_training_component_combinations": measured_combinations,
        "ensemble_mean_distribution_descriptive_only": _quantiles(means),
        "ensemble_standard_deviation_distribution": _quantiles(deviations),
        "ensemble_mean_by_domain_descriptive_only": {
            domain: _quantiles(values) for domain, values in sorted(domain_means.items())
        },
        "all_guidance_actions_abstain": True,
    }


def _gzip_csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=LEDGER_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        compressed.write(text.getvalue().encode())
    return output.getvalue()


def build_fresh_pool_oracle_audit(repo: Path, config_path: Path) -> tuple[dict[str, Any], bytes]:
    """Score the frozen fresh pool while preserving the frozen abstention policy."""

    config = _load_json(config_path, compressed=False, label="oracle audit config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiFreshPoolOracleAuditError("unsupported oracle-audit config")
    policy = config.get("policy")
    if (
        not isinstance(policy, dict)
        or policy.get("endpoint") != "expt_Hela"
        or policy.get("descriptive_nonselecting_only") is not True
        or policy.get("raw_mean_ranking_allowed") is not False
        or policy.get("biological_guidance_authorized") is not False
    ):
        raise UgiFreshPoolOracleAuditError("oracle-audit policy changed")
    batch_size = int(policy.get("batch_size", 0))
    if batch_size <= 0:
        raise UgiFreshPoolOracleAuditError("oracle batch size must be positive")

    paths = _validate_inputs(config, repo)
    manifest = _load_json(
        paths["production_generator_v2"], compressed=False, label="production generator v2"
    )
    sample = _load_json(paths["fresh_pool_sample"], compressed=False, label="fresh-pool sample")
    fresh_audit = _load_json(paths["fresh_pool_audit"], compressed=False, label="fresh-pool audit")
    route_coverage = _load_json(
        paths["route_coverage_result"], compressed=False, label="route coverage"
    )
    route_products = _load_json(
        paths["route_product_values"], compressed=True, label="route product values"
    )
    campaign = _load_json(
        paths["oracle_campaign_selection"], compressed=False, label="oracle campaign selection"
    )
    production_result = _load_json(
        paths["oracle_production_result"], compressed=False, label="oracle production result"
    )
    if manifest.get("status") != "frozen_after_independent_decoder_confirmation":
        raise UgiFreshPoolOracleAuditError("production generator v2 is not frozen")
    if fresh_audit.get("adjudication", {}).get(
        "pool_may_advance_to_nonselecting_oracle_and_route_assessment"
    ) is not True or fresh_audit.get("inputs", {}).get("fresh_pool_sample", {}).get(
        "sha256"
    ) != sha256_file(
        paths["fresh_pool_sample"]
    ):
        raise UgiFreshPoolOracleAuditError("fresh pool is not authorized for oracle audit")
    route_artifact = route_coverage.get("artifacts", {}).get("product_synthesis_values.json.gz", {})
    if route_artifact.get("sha256") != sha256_file(paths["route_product_values"]):
        raise UgiFreshPoolOracleAuditError("route product ledger ownership failed")
    if (
        campaign.get("status") != "hela_campaign_oracle_selected_but_guidance_still_abstained"
        or campaign.get("campaign", {}).get("endpoint") != "expt_Hela"
        or campaign.get("guidance_policy", {}).get("authorized") is not False
        or campaign.get("selected_model", {}).get("candidate_id")
        != production_result.get("selected_model", {}).get("candidate_id")
    ):
        raise UgiFreshPoolOracleAuditError("HeLa campaign oracle contract changed")
    checkpoint_record = production_result.get("checkpoint")
    if not isinstance(checkpoint_record, dict) or checkpoint_record.get("sha256") != sha256_file(
        paths["oracle_production_checkpoint"]
    ):
        raise UgiFreshPoolOracleAuditError("production checkpoint ownership failed")

    candidates = _exact_l1_candidates(sample)
    expected_exact = int(fresh_audit.get("summary", {}).get("exact_l1_products", -1))
    if len(candidates) != expected_exact:
        raise UgiFreshPoolOracleAuditError("fresh-pool exact-L1 denominator changed")
    routes = _route_index(route_products)
    if {candidate["sample_index"] for candidate in candidates} != set(routes):
        raise UgiFreshPoolOracleAuditError("route and oracle candidate sets differ")

    checkpoint = load_production_checkpoint(paths["oracle_production_result"], repo_root=repo)
    scored = []
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        result = score_production_ugi_candidates(
            checkpoint,
            batch,
            endpoint="expt_Hela",
            nominal_coverage_reference=float(policy["nominal_coverage_reference"]),
        )
        rows = result.get("candidates")
        if not isinstance(rows, list) or len(rows) != len(batch):
            raise UgiFreshPoolOracleAuditError("oracle batch output length changed")
        scored.extend(rows)
    if len(scored) != len(candidates):
        raise UgiFreshPoolOracleAuditError("oracle output denominator changed")

    ledger_rows = []
    for candidate, prediction in zip(candidates, scored, strict=True):
        if prediction.get("label") != candidate["label"]:
            raise UgiFreshPoolOracleAuditError("oracle output order changed")
        route = routes[candidate["sample_index"]]
        if (
            route["product_id"] != candidate["product_id"]
            or route["structure_id"] != candidate["structure_id"]
        ):
            raise UgiFreshPoolOracleAuditError("route/oracle candidate identity changed")
        ledger_rows.append(
            {
                "sample_index": candidate["sample_index"],
                "product_id": candidate["product_id"],
                "structure_id": candidate["structure_id"],
                "product_smiles": candidate["product_smiles"],
                "amine_smiles": candidate["amine_smiles"],
                "aldehyde_smiles": candidate["aldehyde_smiles"],
                "isocyanide_smiles": candidate["isocyanide_smiles"],
                "product_structural_provenance_stratum": route[
                    "product_structural_provenance_stratum"
                ],
                "route_complete": route["route_complete"],
                "route_complete_component_count": route["route_complete_component_count"],
                "oracle_domain": prediction["domain"],
                "unseen_component_roles_json": json.dumps(
                    prediction.get("unseen_component_roles", []), separators=(",", ":")
                ),
                "combination_seen_in_measured_training": prediction.get(
                    "combination_seen_in_measured_training", False
                ),
                "exact_forward_verified": prediction["exact_forward_verified"],
                "ensemble_mean": format(float(prediction["ensemble_mean"]), ".12g"),
                "ensemble_standard_deviation": format(
                    float(prediction["ensemble_standard_deviation"]), ".12g"
                ),
                "guidance_action": prediction["guidance_action"],
                "guidance_score": (
                    "" if prediction["guidance_score"] is None else prediction["guidance_score"]
                ),
                "empirical_held_domain_radius": (
                    ""
                    if prediction["empirical_held_domain_radius"] is None
                    else prediction["empirical_held_domain_radius"]
                ),
            }
        )

    summary = summarize_oracle_rows(ledger_rows)
    ledger = _gzip_csv_bytes(ledger_rows)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_nonselecting_hela_oracle_audit",
        "config": {"path": str(config_path.relative_to(repo)), "sha256": sha256_file(config_path)},
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "summary": summary,
        "artifacts": {
            "oracle_predictions.csv.gz": {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "sha256": sha256_bytes(ledger),
            }
        },
        "adjudication": {
            "endpoint": "expt_Hela",
            "predictions_are_descriptive_only": True,
            "raw_mean_used_for_ranking": False,
            "biological_guidance_authorized": False,
            "prospective_candidate_selection_changed": False,
            "route_assessments_changed": False,
        },
    }
    return result, ledger
