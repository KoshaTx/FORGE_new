"""Audit what component novelty is actually tested by the Ugi potency split."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_SCHEMA_VERSION = "phase1_ugi_potency_novelty_lane_audit_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_potency_novelty_lane_audit.v1"
PAIR_SCHEME = "held_aldehyde_isocyanide_pair_5fold"
LABEL_PATTERN = re.compile(r"^(A\d+)(B\d+)(C\d+)$")


class UgiPotencyNoveltyLaneAuditError(RuntimeError):
    """Raised when the frozen novelty audit contract is violated."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise UgiPotencyNoveltyLaneAuditError(f"JSON object required: {path}")
    return value


def _read_csv_gzip(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _components(label: str) -> tuple[str, str, str]:
    match = LABEL_PATTERN.fullmatch(label)
    if match is None:
        raise UgiPotencyNoveltyLaneAuditError(f"invalid AGILE label: {label}")
    return match.group(1), match.group(2), match.group(3)


def audit_pair_split_novelty(
    assignments: Sequence[Mapping[str, str]], *, folds: Sequence[int]
) -> dict[str, Any]:
    """Determine whether pair-held-out rows also hold out either component identity."""

    output: dict[str, Any] = {}
    total_patterns: Counter[str] = Counter()
    for fold in folds:
        rows = [
            row
            for row in assignments
            if row.get("scheme") == PAIR_SCHEME and int(row.get("fold", "-1")) == fold
        ]
        fit = [row for row in rows if row.get("stage") in {"train", "calibration"}]
        test = [row for row in rows if row.get("stage") == "test"]
        if not fit or not test:
            raise UgiPotencyNoveltyLaneAuditError(f"missing fit/test rows for fold {fold}")
        fit_b = {_components(row["label"])[1] for row in fit}
        fit_c = {_components(row["label"])[2] for row in fit}
        fit_pairs = {(_components(row["label"])[1], _components(row["label"])[2]) for row in fit}
        patterns: Counter[str] = Counter()
        leaked_pairs = 0
        for row in test:
            _, b_id, c_id = _components(row["label"])
            leaked_pairs += int((b_id, c_id) in fit_pairs)
            pattern = (
                ("aldehyde_seen" if b_id in fit_b else "aldehyde_exact_new")
                + "__"
                + ("isocyanide_seen" if c_id in fit_c else "isocyanide_exact_new")
            )
            patterns[pattern] += 1
            total_patterns[pattern] += 1
        output[str(fold)] = {
            "fit_rows": len(fit),
            "test_rows": len(test),
            "fit_aldehyde_identities": len(fit_b),
            "fit_isocyanide_identities": len(fit_c),
            "fit_pair_identities": len(fit_pairs),
            "test_pair_leakage_rows": leaked_pairs,
            "test_identity_novelty_patterns": dict(sorted(patterns.items())),
        }
    return {
        "folds": output,
        "total_test_rows": sum(int(row["test_rows"]) for row in output.values()),
        "total_identity_novelty_patterns": dict(sorted(total_patterns.items())),
        "all_test_pairs_unseen": all(
            int(row["test_pair_leakage_rows"]) == 0 for row in output.values()
        ),
        "all_test_component_identities_seen_individually": set(total_patterns)
        == {"aldehyde_seen__isocyanide_seen"},
    }


def build_potency_novelty_lane_audit(repo: Path, config_path: Path) -> dict[str, Any]:
    """Build the frozen read-only novelty-scope audit."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiPotencyNoveltyLaneAuditError("unsupported config schema")
    raw_inputs = config.get("inputs")
    if not isinstance(raw_inputs, Mapping):
        raise UgiPotencyNoveltyLaneAuditError("inputs are missing")
    paths: dict[str, Path] = {}
    for label, record in raw_inputs.items():
        if not isinstance(record, Mapping):
            raise UgiPotencyNoveltyLaneAuditError(f"invalid input pin: {label}")
        path = repo / str(record["path"])
        if _sha256_file(path) != record["sha256"]:
            raise UgiPotencyNoveltyLaneAuditError(f"input hash changed: {label}")
        paths[str(label)] = path

    assignments = _read_csv_gzip(paths["oracle_split_assignments"])
    pair_audit = audit_pair_split_novelty(assignments, folds=tuple(config["analysis"]["folds"]))
    oracle_audit = _read_json(paths["morphology_enriched_oracle_audit"])
    promoted = _read_json(paths["promoted_applicability_proposal"])
    old_policy = _read_json(paths["legacy_potency_policy"])
    pair_metrics = (
        oracle_audit.get("morphology_enriched_supported_distribution", {})
        .get("per_scheme", {})
        .get(PAIR_SCHEME, {})
        .get("metrics")
    )
    if not isinstance(pair_metrics, Mapping):
        raise UgiPotencyNoveltyLaneAuditError("held-pair oracle metrics are missing")
    if promoted.get("decision", {}).get("applicability_proposal_promoted") is not True:
        raise UgiPotencyNoveltyLaneAuditError("applicability proposal is not promoted")
    legacy_patterns = old_policy.get("eligible_patterns", {})
    if not isinstance(legacy_patterns, Mapping) or "amine_only" not in legacy_patterns:
        raise UgiPotencyNoveltyLaneAuditError("legacy policy no longer exposes the superseded lane")

    split_supports_unseen_pairs_only = bool(
        pair_audit["all_test_pairs_unseen"]
        and pair_audit["all_test_component_identities_seen_individually"]
    )
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "potency_novelty_scope_audited",
        "scope": dict(config["scope"]),
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": _sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "pair_split_audit": pair_audit,
        "existing_pair_lane_oracle_metrics": dict(pair_metrics),
        "interpretation": {
            "held_pair_split_validates": (
                "unseen aldehyde-isocyanide pairings whose individual component identities "
                "remain represented in fitting data"
            ),
            "held_pair_split_does_not_validate": [
                "an exact-new aldehyde identity",
                "an exact-new isocyanide identity",
                "two simultaneously exact-new tail-component identities",
            ],
            "continuous_support_is_required_but_does_not_upgrade_the_split_claim": True,
            "legacy_amine_only_lane_superseded_by_negative_held_head_absolute_fit": True,
        },
        "authorized_potency_signal_cohort": {
            "head": "familiar measured amine identity and inside frozen amine support",
            "tails": (
                "individual aldehyde and isocyanide identities represented in the oracle fitting "
                "domain, with their exact pair held out and all continuous views inside R<=1"
            ),
            "target": "observed expt_Hela only",
            "exact_new_aldehyde": False,
            "exact_new_isocyanide": False,
            "two_exact_new_tails": False,
        },
        "decision": {
            "split_supports_unseen_pairs_only": split_supports_unseen_pairs_only,
            "morphology_potency_signal_gate_may_proceed": split_supports_unseen_pairs_only,
            "potency_tilting_authorized": False,
            "mh_authorized": False,
            "partial_state_smc_authorized": False,
            "next_gate": "morphology_to_observed_hela_signal_under_group_disjoint_validation",
        },
        "nonclaims": [
            "This audit does not establish potency prediction for either exact-new tail identity.",
            "The previously reported oracle shift used the earlier support proposal; production-proposal shift is re-evaluated in the potency signal gate.",
            "No generated molecule or oracle prediction was produced by this audit.",
        ],
    }
    content["result_sha256"] = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return content


__all__ = [
    "UgiPotencyNoveltyLaneAuditError",
    "audit_pair_split_novelty",
    "build_potency_novelty_lane_audit",
]
