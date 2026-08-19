"""Adjudicate the frozen broad/support/potency morphology experiment.

The terminal policy in this module is intentionally narrower than the older
HeLa diagnostic wrapper.  The held aldehyde--isocyanide split tested a new
pairing of component identities that were each observed elsewhere in the fit
population; it did not test two exact-new components.  This module therefore
uses the older artifact only for its authenticated oracle, four-view chemical
distances, and held-pair calibration scale.
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

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from forge.core.hashing import sha256_json as _sha256_payload
from forge.data.r1_prime_audit import sha256_file
from forge.potency import ugi_distributional_applicability as applicability
from forge.potency.ugi_hela_potency_diagnostic import (
    ROLE_MAP,
    FrozenHeLaOracleWorker,
    HeLaBatchPredictor,
    HeLaPotencyDiagnosticPolicy,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi_morphology_potency_matched_adjudication_config.v1"
POLICY_SCHEMA_VERSION = "phase1_ugi_morphology_potency_matched_adjudication_policy.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_morphology_potency_matched_adjudication.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_morphology_potency_matched_adjudication_ledger.v1"
BROAD_ARM = "broad_prior"
SUPPORT_ARM = "support_enriched"
POTENCY_ARM = "nested_potency"
EXPECTED_ARMS = (BROAD_ARM, SUPPORT_ARM, POTENCY_ARM)
PAIR_SCALE = "aldehyde_isocyanide_pair"


class UgiMorphologyPotencyMatchedAdjudicationError(RuntimeError):
    """Raised when the frozen matched diagnostic cannot be reproduced."""


def _load(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiMorphologyPotencyMatchedAdjudicationError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiMorphologyPotencyMatchedAdjudicationError(f"{label} must be an object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiMorphologyPotencyMatchedAdjudicationError(f"malformed pin: {label}")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiMorphologyPotencyMatchedAdjudicationError(
            f"input path escapes repository: {label}"
        ) from error
    if not path.is_file() or path.is_symlink() or sha256_file(path) != record["sha256"]:
        raise UgiMorphologyPotencyMatchedAdjudicationError(f"input changed: {label}")
    return path


def _read_jsonl_gzip(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise UgiMorphologyPotencyMatchedAdjudicationError("terminal ledger is malformed")
    return rows


def _exact_l1(terminal: Mapping[str, Any]) -> bool:
    verification = terminal.get("l1_forward_verification")
    return bool(
        terminal.get("valid") is True
        and terminal.get("terminal_valid") is True
        and terminal.get("component_reconstruction_valid") is True
        and isinstance(verification, Mapping)
        and verification.get("exact_product_reconstructed") is True
        and verification.get("maximum_outcomes_saturated") is False
    )


def _candidate(row: Mapping[str, Any]) -> dict[str, str]:
    terminal = row.get("native_terminal")
    if not isinstance(terminal, Mapping) or not _exact_l1(terminal):
        raise UgiMorphologyPotencyMatchedAdjudicationError(
            "candidate requested from a nonexact terminal"
        )
    components = terminal.get("component_smiles_by_role")
    if not isinstance(components, Mapping):
        raise UgiMorphologyPotencyMatchedAdjudicationError("terminal components are absent")
    try:
        return {
            "label": _sha256_payload(
                [str(row["arm_id"]), int(row["draw_index"]), str(terminal["smiles"])]
            ),
            "product_smiles": str(terminal["smiles"]),
            **{
                f"{role}_smiles": str(components[native_role])
                for role, native_role in ROLE_MAP.items()
            },
        }
    except KeyError as error:
        raise UgiMorphologyPotencyMatchedAdjudicationError(
            "terminal component roles changed"
        ) from error


def _effective_count(values: Sequence[str]) -> float:
    if not values:
        return 0.0
    counts = np.asarray(tuple(Counter(values).values()), dtype=np.float64)
    probabilities = counts / counts.sum()
    return float(1.0 / np.square(probabilities).sum())


def _maximum_fraction(values: Sequence[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    return float(max(counts.values()) / len(values))


def _internal_diversity(smiles: Sequence[str], *, maximum: int = 512) -> float:
    ordered = sorted(set(smiles))
    if len(ordered) < 2:
        return 0.0
    if len(ordered) > maximum:
        ranks = sorted(
            range(len(ordered)),
            key=lambda index: hashlib.sha256(ordered[index].encode()).digest(),
        )[:maximum]
        ordered = [ordered[index] for index in sorted(ranks)]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = []
    for value in ordered:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise UgiMorphologyPotencyMatchedAdjudicationError(
                "canonical product failed fingerprinting"
            )
        fingerprints.append(generator.GetFingerprint(molecule))
    similarities: list[float] = []
    for index in range(1, len(fingerprints)):
        similarities.extend(
            DataStructs.BulkTanimotoSimilarity(fingerprints[index], fingerprints[:index])
        )
    return float(1.0 - np.mean(similarities))


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


def _classify_rows(
    rows: Sequence[Mapping[str, Any]], policy: HeLaPotencyDiagnosticPolicy
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        arm = str(source.get("arm_id"))
        draw_index = int(source.get("draw_index", -1))
        terminal = source.get("native_terminal")
        record: dict[str, Any] = {
            "arm_id": arm,
            "draw_index": draw_index,
            "program_sha256": str(source.get("program_sha256", "")),
            "exact_l1": False,
            "reason": "invalid_or_nonexact_l1",
            "eligible": False,
            "oracle_selected": False,
            "oracle_mean": "",
            "oracle_sd": "",
            "conformal_q90": "",
            "lcb90": "",
            "calibration_ecdf": "",
            "potency_utility": 0.0,
            "conservative_high_potency": False,
            "canonical_product": "",
            "canonical_amine": "",
            "canonical_aldehyde": "",
            "canonical_isocyanide": "",
            "unseen_roles": "",
            "exact_measured_combination": "",
            "overall_bin": "",
            "view_bins": "",
            "worker_receipt_sha256": "",
        }
        if not isinstance(terminal, Mapping) or not _exact_l1(terminal):
            output.append(record)
            continue
        candidate = _candidate(source)
        canonical = {
            "product": applicability._canonical(candidate["product_smiles"]),
            **{role: applicability._canonical(candidate[f"{role}_smiles"]) for role in ROLE_MAP},
        }
        unseen_roles = tuple(
            role for role in ROLE_MAP if canonical[role] not in policy.component_sets[role]
        )
        triple = "\x1f".join(canonical[role] for role in ROLE_MAP)
        exact_measured = triple in policy.measured_triples
        record["exact_l1"] = True
        record.update(
            {
                "canonical_product": canonical["product"],
                "canonical_amine": canonical["amine"],
                "canonical_aldehyde": canonical["aldehyde"],
                "canonical_isocyanide": canonical["isocyanide"],
                "unseen_roles": ",".join(unseen_roles),
                "exact_measured_combination": exact_measured,
            }
        )
        # Rejected identity classes cannot become authorized by their geometric
        # bin. Short-circuit them before the substantially costlier multiview
        # radius calculation without changing the frozen scientific policy.
        if exact_measured:
            record["reason"] = "exact_measured_combination_neutral"
            output.append(record)
            continue
        if unseen_roles:
            record["reason"] = "exact_new_component_biological_abstention"
            output.append(record)
            continue
        classification = policy.classify_candidate_mapping(candidate)
        if classification["canonical"] != canonical:
            raise UgiMorphologyPotencyMatchedAdjudicationError(
                "fast and authenticated canonicalization disagree"
            )
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
            record["reason"] = "eligible_familiar_components_unseen_combination"
            record["candidate"] = candidate
        output.append(record)
    return output


def _select_equal_oracle_budget(
    rows: Sequence[dict[str, Any]], arms: Sequence[str]
) -> tuple[int, dict[str, list[dict[str, Any]]]]:
    eligible = {
        arm: [row for row in rows if row["arm_id"] == arm and row["eligible"]] for arm in arms
    }
    budget = min((len(values) for values in eligible.values()), default=0)
    selected: dict[str, list[dict[str, Any]]] = {}
    for arm, values in eligible.items():
        selected[arm] = sorted(
            values,
            key=lambda row: hashlib.sha256(
                f"matched-oracle-budget-v1|{row['draw_index']}|{row['canonical_product']}".encode()
            ).digest(),
        )[:budget]
        for row in selected[arm]:
            row["oracle_selected"] = True
    return budget, selected


def _score_selected(
    selected: Mapping[str, Sequence[dict[str, Any]]],
    policy: HeLaPotencyDiagnosticPolicy,
    predictor: HeLaBatchPredictor,
) -> None:
    scale = policy.scales[PAIR_SCALE]
    flattened = [row for arm in EXPECTED_ARMS for row in selected[arm]]
    if not flattened:
        return
    response = predictor.predict([row["candidate"] for row in flattened])
    classifications = response.get("classifications")
    prediction = response.get("prediction")
    receipt = response.get("receipt_sha256")
    if (
        response.get("status") != "complete"
        or not isinstance(classifications, list)
        or len(classifications) != len(flattened)
        or not isinstance(prediction, Mapping)
        or prediction.get("endpoint") != "expt_Hela"
        or prediction.get("records") != len(flattened)
        or not isinstance(receipt, str)
        or len(receipt) != 64
    ):
        raise UgiMorphologyPotencyMatchedAdjudicationError("oracle response changed")
    means = prediction.get("ensemble_mean")
    deviations = prediction.get("ensemble_standard_deviation")
    if not isinstance(means, list) or not isinstance(deviations, list):
        raise UgiMorphologyPotencyMatchedAdjudicationError("oracle predictions absent")
    if len(means) != len(flattened) or len(deviations) != len(flattened):
        raise UgiMorphologyPotencyMatchedAdjudicationError("oracle predictions misaligned")
    for row, classification, raw_mean, raw_sd in zip(
        flattened, classifications, means, deviations, strict=True
    ):
        if not isinstance(classification, Mapping):
            raise UgiMorphologyPotencyMatchedAdjudicationError("worker classification malformed")
        mean = float(raw_mean)
        deviation = float(raw_sd)
        if not math.isfinite(mean) or not math.isfinite(deviation) or deviation < 0.0:
            raise UgiMorphologyPotencyMatchedAdjudicationError("invalid oracle prediction")
        if (
            classification.get("exact_forward_verified") is not True
            or classification.get("canonical")
            != {
                "product": row["canonical_product"],
                "amine": row["canonical_amine"],
                "aldehyde": row["canonical_aldehyde"],
                "isocyanide": row["canonical_isocyanide"],
            }
            or classification.get("unseen_component_roles") != []
            or classification.get("combination_seen_in_measured_training") is not False
        ):
            raise UgiMorphologyPotencyMatchedAdjudicationError(
                "worker and generator identities disagree"
            )
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


def _arm_metrics(rows: Sequence[dict[str, Any]], arm: str, oracle_budget: int) -> dict[str, Any]:
    arm_rows = [row for row in rows if row["arm_id"] == arm]
    exact = [row for row in arm_rows if row["exact_l1"]]
    eligible = [row for row in arm_rows if row["eligible"]]
    scored = [row for row in arm_rows if row["oracle_selected"]]
    high = [row for row in scored if row["conservative_high_potency"]]
    unique_exact = {row["canonical_product"]: row for row in exact}
    unique_high = {row["canonical_product"]: row for row in high}
    roles = ("amine", "aldehyde", "isocyanide")
    role_values = {role: [row[f"canonical_{role}"] for row in exact] for role in roles}
    return {
        "generator_calls": len(arm_rows),
        "exact_l1_terminals": len(exact),
        "exact_l1_fraction": len(exact) / len(arm_rows),
        "unique_exact_l1_products": len(unique_exact),
        "uniqueness_fraction_among_exact_l1": len(unique_exact) / max(1, len(exact)),
        "internal_product_diversity": _internal_diversity(list(unique_exact)),
        "eligible_terminals_before_oracle_budget": len(eligible),
        "eligible_fraction_per_generator_call": len(eligible) / len(arm_rows),
        "oracle_calls": len(scored),
        "matched_oracle_budget": oracle_budget,
        "conservative_high_potency_terminals": len(high),
        "unique_conservative_high_potency_products": len(unique_high),
        "unique_high_potency_yield_per_generator_call": len(unique_high) / len(arm_rows),
        "unique_high_potency_yield_per_oracle_call": len(unique_high) / max(1, len(scored)),
        "mean_potency_utility": float(
            np.mean([float(row["potency_utility"]) for row in scored]) if scored else 0.0
        ),
        "role_effective_component_counts": {
            role: _effective_count(role_values[role]) for role in roles
        },
        "role_maximum_component_fractions": {
            role: _maximum_fraction(role_values[role]) for role in roles
        },
        "abstention_reasons": dict(sorted(Counter(row["reason"] for row in arm_rows).items())),
    }


def _paired_unique_bootstrap(
    rows: Sequence[dict[str, Any]], *, replicates: int, seed: int
) -> dict[str, float]:
    indexed: dict[str, dict[int, str | None]] = {arm: {} for arm in EXPECTED_ARMS}
    for row in rows:
        product = (
            str(row["canonical_product"])
            if row["oracle_selected"] and row["conservative_high_potency"]
            else None
        )
        indexed[str(row["arm_id"])][int(row["draw_index"])] = product
    draw_count = len(indexed[SUPPORT_ARM])
    if draw_count == 0 or any(set(values) != set(range(draw_count)) for values in indexed.values()):
        raise UgiMorphologyPotencyMatchedAdjudicationError("draw indices are not paired")
    rng = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        draws = rng.integers(0, draw_count, size=draw_count)
        support = {indexed[SUPPORT_ARM][int(index)] for index in draws}
        potency = {indexed[POTENCY_ARM][int(index)] for index in draws}
        support.discard(None)
        potency.discard(None)
        differences[replicate] = (len(potency) - len(support)) / draw_count
    return {
        "point_difference_per_generator_call": (
            len({value for value in indexed[POTENCY_ARM].values() if value is not None})
            - len({value for value in indexed[SUPPORT_ARM].values() if value is not None})
        )
        / draw_count,
        "confidence_interval_low": float(np.quantile(differences, 0.025)),
        "confidence_interval_high": float(np.quantile(differences, 0.975)),
        "replicates": replicates,
        "seed": seed,
    }


def _promotion_checks(
    metrics: Mapping[str, Mapping[str, Any]],
    bootstrap: Mapping[str, float],
    thresholds: Mapping[str, Any],
) -> dict[str, bool]:
    support = metrics[SUPPORT_ARM]
    potency = metrics[POTENCY_ARM]
    checks = {
        "primary_endpoint_positive": bootstrap["point_difference_per_generator_call"] > 0.0,
        "bootstrap_lower_nonnegative": bootstrap["confidence_interval_low"]
        >= float(thresholds["paired_bootstrap_interval_lower_bound_at_least"]),
        "unique_high_potency_yield": potency["unique_conservative_high_potency_products"]
        >= float(thresholds["minimum_unique_high_potency_yield_ratio_vs_applicability"])
        * support["unique_conservative_high_potency_products"],
        "exact_l1_validity": potency["exact_l1_fraction"]
        >= float(thresholds["minimum_exact_l1_validity_fraction_vs_applicability"])
        * support["exact_l1_fraction"],
        "unique_product_count": potency["unique_exact_l1_products"]
        >= float(thresholds["minimum_unique_product_count_fraction_vs_applicability"])
        * support["unique_exact_l1_products"],
        "product_diversity": potency["internal_product_diversity"]
        >= float(thresholds["minimum_product_fingerprint_diversity_fraction_vs_applicability"])
        * support["internal_product_diversity"],
    }
    for role in ("amine", "aldehyde", "isocyanide"):
        checks[f"effective_{role}_count"] = (
            potency["role_effective_component_counts"][role]
            >= float(thresholds[f"minimum_effective_{role}_count_fraction_vs_applicability"])
            * support["role_effective_component_counts"][role]
        )
        checks[f"maximum_{role}_fraction"] = potency["role_maximum_component_fractions"][
            role
        ] <= float(thresholds["no_single_component_probability_above"])
    return checks


def build_matched_potency_adjudication(
    repo: Path,
    config_path: Path,
    *,
    predictor: HeLaBatchPredictor | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Score the preregistered matched terminal ledger and apply fail-closed gates."""

    repo = repo.resolve()
    config = _load(config_path.resolve(), label="matched adjudication config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiMorphologyPotencyMatchedAdjudicationError("unsupported config schema")
    paths = {
        label: _pin(repo, record, label=label) for label, record in config.get("inputs", {}).items()
    }
    required = {
        "matched_generation_result",
        "matched_terminal_ledger",
        "adjudication_policy",
        "hela_diagnostic_policy",
    }
    if set(paths) != required:
        raise UgiMorphologyPotencyMatchedAdjudicationError("input set changed")
    frozen = _load(paths["adjudication_policy"], label="adjudication policy")
    if (
        frozen.get("schema_version") != POLICY_SCHEMA_VERSION
        or frozen.get("status") != "frozen_before_matched_terminal_outcomes"
    ):
        raise UgiMorphologyPotencyMatchedAdjudicationError("adjudication policy changed")
    generation = _load(paths["matched_generation_result"], label="generation result")
    artifact = generation.get("artifacts", {}).get("terminal_ledger.jsonl.gz", {})
    if artifact.get("sha256") != sha256_file(paths["matched_terminal_ledger"]):
        raise UgiMorphologyPotencyMatchedAdjudicationError("generation ledger pin disagrees")
    rows = _read_jsonl_gzip(paths["matched_terminal_ledger"])
    expected_per_arm = int(frozen["budgets"]["generator_calls_per_arm"])
    if len(rows) != expected_per_arm * len(EXPECTED_ARMS):
        raise UgiMorphologyPotencyMatchedAdjudicationError("matched row count changed")
    counts = Counter(str(row.get("arm_id")) for row in rows)
    if counts != Counter({arm: expected_per_arm for arm in EXPECTED_ARMS}):
        raise UgiMorphologyPotencyMatchedAdjudicationError("arm allocation changed")
    for arm in EXPECTED_ARMS:
        indices = sorted(int(row["draw_index"]) for row in rows if row["arm_id"] == arm)
        if indices != list(range(expected_per_arm)):
            raise UgiMorphologyPotencyMatchedAdjudicationError("draw schedule changed")

    policy = HeLaPotencyDiagnosticPolicy(repo, paths["hela_diagnostic_policy"])
    classified = _classify_rows(rows, policy)
    oracle_budget, selected = _select_equal_oracle_budget(classified, EXPECTED_ARMS)
    if predictor is None:
        with FrozenHeLaOracleWorker(policy) as worker:
            _score_selected(selected, policy, worker)
            worker_receipt = worker.ready_receipt_sha256
    else:
        _score_selected(selected, policy, predictor)
        worker_receipt = "test_predictor"
    arm_metrics = {arm: _arm_metrics(classified, arm, oracle_budget) for arm in EXPECTED_ARMS}
    bootstrap = _paired_unique_bootstrap(
        classified,
        replicates=int(frozen["primary_endpoint"]["replicates"]),
        seed=int(config["bootstrap_seed"]),
    )
    checks = _promotion_checks(arm_metrics, bootstrap, frozen["promotion_gates"])
    promoted = oracle_budget > 0 and all(checks.values())
    fields = (
        "arm_id",
        "draw_index",
        "program_sha256",
        "exact_l1",
        "reason",
        "eligible",
        "oracle_selected",
        "oracle_mean",
        "oracle_sd",
        "conformal_q90",
        "lcb90",
        "calibration_ecdf",
        "potency_utility",
        "conservative_high_potency",
        "canonical_product",
        "canonical_amine",
        "canonical_aldehyde",
        "canonical_isocyanide",
        "unseen_roles",
        "exact_measured_combination",
        "overall_bin",
        "view_bins",
        "worker_receipt_sha256",
    )
    ledger = _gzip_csv(classified, fields)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "matched_terminal_potency_adjudication_complete",
        "scope": dict(frozen["scope"]),
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "scientific_lane": dict(frozen["scientific_lane"]),
        "matched_budgets": {
            "generator_calls_per_arm": expected_per_arm,
            "oracle_calls_per_arm": oracle_budget,
            "worker_ready_receipt_sha256": worker_receipt,
        },
        "arms": arm_metrics,
        "primary_endpoint": bootstrap,
        "promotion_checks": checks,
        "decision": {
            "potency_tilting_promoted": promoted,
            "selected_production_method": (
                "nested_morphology_potency_proposal_plus_terminal_ranking"
                if promoted
                else "promoted_applicability_proposal_plus_terminal_ranking"
            ),
            "reason": (
                "all preregistered matched-comparison gates passed"
                if promoted
                else "one or more preregistered matched-comparison gates failed"
            ),
            "mh_authorized": False,
            "partial_state_smc_authorized": False,
            "prospective_candidate_selection_authorized": False,
        },
        "artifacts": {
            "terminal_adjudication.csv.gz": {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "rows": len(classified),
                "sha256": hashlib.sha256(ledger).hexdigest(),
            }
        },
        "nonclaims": [
            "The held-pair calibration does not validate exact-new aldehyde or isocyanide identities.",
            "A morphology-level potency tilt is not atom- or bond-level biological guidance.",
            "This diagnostic does not authorize prospective candidate selection.",
        ],
    }
    result["result_sha256"] = _sha256_payload(result)
    return result, ledger


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "LEDGER_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "UgiMorphologyPotencyMatchedAdjudicationError",
    "build_matched_potency_adjudication",
]
