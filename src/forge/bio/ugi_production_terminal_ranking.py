"""Frozen matched terminal ranking for the first FORGE production pools.

The broad-prior and support-enriched arms receive the same applicability
policy, the same lane-specific oracle budgets and the same conservative
ranking rule.  This stage neither changes generation nor locks a prospective
panel; it produces the authenticated pool that may enter route assessment.
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
from pathlib import Path
from typing import Any

from forge.bio import ugi_distributional_applicability as applicability
from forge.bio.ugi_hela_potency_diagnostic import (
    ROLE_MAP,
    FrozenHeLaOracleWorker,
    HeLaBatchPredictor,
    HeLaPotencyDiagnosticPolicy,
)
from forge.bio.ugi_morphology_potency_matched_adjudication import _candidate, _exact_l1
from forge.core.hashing import sha256_json as _sha256_payload
from forge.data.r1_prime_audit import sha256_file

CONFIG_SCHEMA_VERSION = "phase1_ugi_production_terminal_ranking_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_production_terminal_ranking.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_production_terminal_ranking_ledger.v1"
EXPECTED_ARMS = ("broad_prior", "support_enriched")


class UgiProductionTerminalRankingError(RuntimeError):
    """Raised when the production ranking contract changes."""


def _load(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiProductionTerminalRankingError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiProductionTerminalRankingError(f"{label} must be an object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiProductionTerminalRankingError(f"malformed input pin: {label}")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiProductionTerminalRankingError(f"input escapes repository: {label}") from error
    if not path.is_file() or path.is_symlink() or sha256_file(path) != record["sha256"]:
        raise UgiProductionTerminalRankingError(f"input changed: {label}")
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise UgiProductionTerminalRankingError("terminal ledger is malformed")
    return rows


def _gzip_csv(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
        handle.write(buffer.getvalue().encode())
    return raw.getvalue()


def _classify(
    rows: Sequence[Mapping[str, Any]],
    policy: HeLaPotencyDiagnosticPolicy,
    lane_by_roles: Mapping[tuple[str, ...], str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        record: dict[str, Any] = {
            "arm_id": str(source.get("arm_id", "")),
            "draw_index": int(source.get("draw_index", -1)),
            "program_sha256": str(source.get("program_sha256", "")),
            "exact_l1": False,
            "lane_id": "",
            "unseen_roles": "",
            "reason": "invalid_or_nonexact_l1",
            "overall_bin": "",
            "view_bins": "",
            "eligible": False,
            "oracle_selected": False,
            "canonical_product": "",
            "canonical_amine": "",
            "canonical_aldehyde": "",
            "canonical_isocyanide": "",
            "oracle_mean": "",
            "oracle_sd": "",
            "conformal_q90": "",
            "lcb90": "",
            "calibration_ecdf": "",
            "potency_utility": 0.0,
            "conservative_high_potency": False,
            "worker_receipt_sha256": "",
        }
        terminal = source.get("native_terminal")
        if not isinstance(terminal, Mapping) or not _exact_l1(terminal):
            output.append(record)
            continue
        candidate = _candidate(source)
        canonical = {
            "product": applicability._canonical(candidate["product_smiles"]),
            **{role: applicability._canonical(candidate[f"{role}_smiles"]) for role in ROLE_MAP},
        }
        unseen = tuple(
            role for role in ROLE_MAP if canonical[role] not in policy.component_sets[role]
        )
        lane_id = lane_by_roles.get(unseen)
        exact_measured = (
            "\x1f".join(canonical[role] for role in ROLE_MAP) in policy.measured_triples
        )
        record.update(
            {
                "exact_l1": True,
                "lane_id": lane_id or "",
                "unseen_roles": ",".join(unseen),
                "canonical_product": canonical["product"],
                "canonical_amine": canonical["amine"],
                "canonical_aldehyde": canonical["aldehyde"],
                "canonical_isocyanide": canonical["isocyanide"],
            }
        )
        if exact_measured:
            record["reason"] = "exact_measured_product_neutral"
        elif lane_id is None:
            record["reason"] = "novelty_pattern_without_frozen_calibration"
        else:
            classification = policy.classify_candidate_mapping(candidate)
            if (
                classification["canonical"] != canonical
                or classification["unseen_roles"] != unseen
                or classification["exact_measured_combination"] is not False
            ):
                raise UgiProductionTerminalRankingError("authenticated classification changed")
            record.update(
                {
                    "overall_bin": classification["overall_bin"],
                    "view_bins": ";".join(
                        f"{name}:{value}" for name, value in classification["view_bins"]
                    ),
                }
            )
            if classification["overall_bin"] != "interpolative" or any(
                value != "interpolative" for _, value in classification["view_bins"]
            ):
                record["reason"] = f"distribution_{classification['overall_bin']}"
            else:
                record["eligible"] = True
                record["reason"] = f"eligible_{lane_id}"
                record["candidate"] = candidate
        output.append(record)
    return output


def _deduplicated(values: Sequence[dict[str, Any]], *, lane_id: str) -> list[dict[str, Any]]:
    ordered = sorted(
        values,
        key=lambda row: hashlib.sha256(
            (
                f"production-ranking-v1|{lane_id}|{row['draw_index']}|{row['canonical_product']}"
            ).encode()
        ).digest(),
    )
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in ordered:
        product = str(row["canonical_product"])
        if product not in seen:
            seen.add(product)
            output.append(row)
    return output


def _select_matched_budgets(
    rows: Sequence[dict[str, Any]], lanes: Sequence[str]
) -> tuple[dict[str, int], dict[str, list[dict[str, Any]]]]:
    selected = {arm: [] for arm in EXPECTED_ARMS}
    budgets: dict[str, int] = {}
    for lane_id in lanes:
        candidates = {
            arm: _deduplicated(
                [
                    row
                    for row in rows
                    if row["arm_id"] == arm and row["lane_id"] == lane_id and row["eligible"]
                ],
                lane_id=lane_id,
            )
            for arm in EXPECTED_ARMS
        }
        budget = min(len(values) for values in candidates.values())
        budgets[lane_id] = budget
        for arm, values in candidates.items():
            chosen = values[:budget]
            for row in chosen:
                row["oracle_selected"] = True
            selected[arm].extend(chosen)
    return budgets, selected


def _score(
    selected: Mapping[str, Sequence[dict[str, Any]]],
    policy: HeLaPotencyDiagnosticPolicy,
    scale_by_lane: Mapping[str, str],
    predictor: HeLaBatchPredictor,
) -> None:
    rows = [row for arm in EXPECTED_ARMS for row in selected[arm]]
    if not rows:
        return
    response = predictor.predict([row["candidate"] for row in rows])
    classifications = response.get("classifications")
    prediction = response.get("prediction")
    receipt = response.get("receipt_sha256")
    if (
        response.get("status") != "complete"
        or not isinstance(classifications, list)
        or len(classifications) != len(rows)
        or not isinstance(prediction, Mapping)
        or prediction.get("records") != len(rows)
        or prediction.get("endpoint") != "expt_Hela"
        or not isinstance(receipt, str)
        or len(receipt) != 64
    ):
        raise UgiProductionTerminalRankingError("oracle response changed")
    means = prediction.get("ensemble_mean")
    deviations = prediction.get("ensemble_standard_deviation")
    if not isinstance(means, list) or not isinstance(deviations, list):
        raise UgiProductionTerminalRankingError("oracle predictions are absent")
    for row, classification, raw_mean, raw_sd in zip(
        rows, classifications, means, deviations, strict=True
    ):
        if not isinstance(classification, Mapping):
            raise UgiProductionTerminalRankingError("worker classification is malformed")
        expected_unseen = tuple(filter(None, str(row["unseen_roles"]).split(",")))
        if (
            classification.get("exact_forward_verified") is not True
            or tuple(classification.get("unseen_component_roles", [])) != expected_unseen
            or classification.get("combination_seen_in_measured_training") is not False
            or classification.get("canonical")
            != {
                "product": row["canonical_product"],
                "amine": row["canonical_amine"],
                "aldehyde": row["canonical_aldehyde"],
                "isocyanide": row["canonical_isocyanide"],
            }
        ):
            raise UgiProductionTerminalRankingError("worker and generator identities disagree")
        mean = float(raw_mean)
        deviation = float(raw_sd)
        if not math.isfinite(mean) or not math.isfinite(deviation) or deviation < 0.0:
            raise UgiProductionTerminalRankingError("oracle prediction is invalid")
        scale = policy.scales[scale_by_lane[str(row["lane_id"])]]
        lcb90, cdf, utility = scale.score(mean)
        row.update(
            {
                "oracle_mean": mean,
                "oracle_sd": deviation,
                "conformal_q90": scale.max_q90,
                "lcb90": lcb90,
                "calibration_ecdf": cdf,
                "potency_utility": utility,
                "conservative_high_potency": cdf > 0.5,
                "worker_receipt_sha256": receipt,
            }
        )


def _arm_metrics(rows: Sequence[dict[str, Any]], arm: str, lanes: Sequence[str]) -> dict[str, Any]:
    arm_rows = [row for row in rows if row["arm_id"] == arm]
    output: dict[str, Any] = {
        "generator_calls": len(arm_rows),
        "exact_l1_terminals": sum(bool(row["exact_l1"]) for row in arm_rows),
        "unique_exact_l1_products": len(
            {str(row["canonical_product"]) for row in arm_rows if row["exact_l1"]}
        ),
        "abstention_reasons": dict(sorted(Counter(str(row["reason"]) for row in arm_rows).items())),
        "lanes": {},
    }
    for lane_id in lanes:
        eligible = [row for row in arm_rows if row["lane_id"] == lane_id and row["eligible"]]
        scored = [row for row in eligible if row["oracle_selected"]]
        high = [row for row in scored if row["conservative_high_potency"]]
        output["lanes"][lane_id] = {
            "eligible_terminals": len(eligible),
            "unique_eligible_products": len({row["canonical_product"] for row in eligible}),
            "oracle_calls": len(scored),
            "conservative_high_potency_terminals": len(high),
            "unique_conservative_high_potency_products": len(
                {row["canonical_product"] for row in high}
            ),
        }
    return output


def build_production_terminal_ranking(
    repo: Path,
    config_path: Path,
    *,
    predictor: HeLaBatchPredictor | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Classify and conservatively rank both matched production arms."""

    repo = repo.resolve()
    config = _load(config_path.resolve(), label="production terminal ranking config")
    if (
        config.get("schema_version") != CONFIG_SCHEMA_VERSION
        or config.get("status")
        != "frozen_after_matched_generation_before_terminal_classification_or_oracle_scoring"
        or tuple(config.get("arms", [])) != EXPECTED_ARMS
    ):
        raise UgiProductionTerminalRankingError("unsupported production ranking config")
    paths = {
        label: _pin(repo, record, label=label) for label, record in config.get("inputs", {}).items()
    }
    if set(paths) != {
        "generation_result",
        "terminal_ledger",
        "hela_diagnostic_policy",
        "potency_tilt_adjudication",
        "continuous_novelty_diagnostic",
    }:
        raise UgiProductionTerminalRankingError("production input set changed")
    generation = _load(paths["generation_result"], label="production generation result")
    potency = _load(paths["potency_tilt_adjudication"], label="potency adjudication")
    novelty = _load(paths["continuous_novelty_diagnostic"], label="continuous novelty result")
    if (
        generation.get("status") != "complete_matched_production_candidate_generation"
        or generation.get("artifacts", {}).get("terminal_ledger.jsonl.gz", {}).get("sha256")
        != sha256_file(paths["terminal_ledger"])
        or potency.get("decision", {}).get("potency_tilting_promoted") is not False
        or novelty.get("interpretation", {}).get("potency_tilting_promoted") is not False
    ):
        raise UgiProductionTerminalRankingError("production prerequisites changed")
    rows = _read_jsonl(paths["terminal_ledger"])
    expected = int(config["matched_design"]["generator_calls_per_arm"])
    if Counter(str(row.get("arm_id")) for row in rows) != Counter(
        {arm: expected for arm in EXPECTED_ARMS}
    ):
        raise UgiProductionTerminalRankingError("production arm budgets changed")
    policy = HeLaPotencyDiagnosticPolicy(repo, paths["hela_diagnostic_policy"])
    lanes = tuple(str(value) for value in config["lanes"])
    lane_by_roles = {
        tuple(str(role) for role in config["lanes"][lane]["unseen_roles"]): lane for lane in lanes
    }
    scale_by_lane = {lane: str(config["lanes"][lane]["calibration_scale"]) for lane in lanes}
    if not set(scale_by_lane.values()).issubset(policy.scales):
        raise UgiProductionTerminalRankingError("lane calibration scale changed")
    classified = _classify(rows, policy, lane_by_roles)
    budgets, selected = _select_matched_budgets(classified, lanes)
    if predictor is None:
        with FrozenHeLaOracleWorker(policy) as worker:
            _score(selected, policy, scale_by_lane, worker)
            worker_receipt = worker.ready_receipt_sha256
    else:
        _score(selected, policy, scale_by_lane, predictor)
        worker_receipt = "test_predictor"
    fields = (
        "arm_id",
        "draw_index",
        "program_sha256",
        "exact_l1",
        "lane_id",
        "unseen_roles",
        "reason",
        "overall_bin",
        "view_bins",
        "eligible",
        "oracle_selected",
        "canonical_product",
        "canonical_amine",
        "canonical_aldehyde",
        "canonical_isocyanide",
        "oracle_mean",
        "oracle_sd",
        "conformal_q90",
        "lcb90",
        "calibration_ecdf",
        "potency_utility",
        "conservative_high_potency",
        "worker_receipt_sha256",
    )
    ledger = _gzip_csv(classified, fields)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_matched_production_terminal_ranking",
        "scope": dict(config["scope"]),
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "lanes": dict(config["lanes"]),
        "matched_oracle_budgets_per_arm": budgets,
        "worker_ready_receipt_sha256": worker_receipt,
        "arms": {arm: _arm_metrics(classified, arm, lanes) for arm in EXPECTED_ARMS},
        "next_gate": "identical_diversity_balanced_route_shortlists_before_panel_lock",
        "artifacts": {
            "terminal_ranking.csv.gz": {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "rows": len(classified),
                "sha256": hashlib.sha256(ledger).hexdigest(),
            }
        },
        "nonclaims": [
            "Potency did not alter molecular generation.",
            "A conservative ranking is not an absolute potency guarantee.",
            "New-head results remain exploratory.",
            "No prospective panel is locked before route assessment.",
        ],
    }
    result["result_sha256"] = _sha256_payload(result)
    return result, ledger


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "UgiProductionTerminalRankingError",
    "build_production_terminal_ranking",
]
