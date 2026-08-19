"""Read-only applicability census for fully generated Ugi component roles.

The census reuses one frozen, unguided terminal pool.  It measures chemical
distance for the complete product and the generated amine, aldehyde and
isocyanide views under the already-frozen version-3 applicability thresholds.
It never calls the oracle, advances a generator, computes a potency utility,
or authorizes biological or synthesis guidance.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.bio import ugi_distributional_applicability as applicability_v1
from forge.data.r1_prime_audit import sha256_bytes, sha256_file

CONFIG_SCHEMA_VERSION = "phase1_ugi_generated_role_applicability_census_readiness_config.v1"
POLICY_SCHEMA_VERSION = "phase1_ugi_generated_role_applicability_census_policy.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_generated_role_applicability_census_readiness.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_generated_role_applicability_census_ledger.v1"

VIEWS = ("product", "amine", "aldehyde", "isocyanide")
ROLES = ("amine", "aldehyde", "isocyanide")
ROLE_TO_SAMPLE_ROLE = {
    "amine": "amine_head",
    "aldehyde": "oxoester_aldehyde_body_tail",
    "isocyanide": "isocyanide_tail",
}
ROLE_TO_SMILES_FIELD = {
    "amine": "amine_smiles",
    "aldehyde": "aldehyde_smiles",
    "isocyanide": "isocyanide_smiles",
}
BIN_NAMES = ("interpolative", "boundary", "extrapolative")
EXPECTED_INPUTS = {
    "applicability_result",
    "bio_policy",
    "fresh_pool_audit",
    "fresh_pool_config",
    "fresh_pool_sample",
    "generated_applicability",
    "prior_bio_policy",
    "runner",
    "source",
    "tests",
}
EXPECTED_SUPPORTED_PATTERNS = {
    ("amine",): "amine_only",
    ("aldehyde", "isocyanide"): "aldehyde_isocyanide_pair",
}
EXPECTED_SCOPE = {
    "read_only": True,
    "existing_frozen_terminal_pool_only": True,
    "valid_exact_l1_terminals_only": True,
    "all_three_component_roles_generated": True,
    "fixed_head_clamp_primary_path": False,
    "fixed_head_clamp_diagnostic_ablation_only": True,
    "oracle_calls": 0,
    "oracle_selection": False,
    "potency_predictions_consumed": False,
    "biological_guidance": False,
    "synthesis_calls": 0,
    "synthesis_guidance": False,
    "generator_trajectories_advanced": False,
    "candidate_selection": False,
    "prospective_candidate_lock": False,
    "nonzero_guidance_execution": False,
}

LEDGER_FIELDS = (
    "sample_index",
    "product_id",
    "exact_identity_provenance",
    "exact_unseen_roles_json",
    "branch_class",
    "source_stratum",
    "current_policy_pattern_id",
    "current_policy_action",
    "current_policy_reason",
    "noninterpolative_views_json",
    "extrapolative_views_json",
    *(
        field
        for view in VIEWS
        for field in (
            f"{view}_distribution_bin",
            f"{view}_fingerprint_distance",
            f"{view}_descriptor_distance",
            f"{view}_interpolative_radius_ratio",
            f"{view}_boundary_radius_ratio",
        )
    ),
)


class UgiGeneratedRoleApplicabilityCensusError(RuntimeError):
    """Raised when the read-only generated-role census changes contract."""


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiGeneratedRoleApplicabilityCensusError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiGeneratedRoleApplicabilityCensusError(f"{label} must contain one object")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise UgiGeneratedRoleApplicabilityCensusError(
            f"invalid applicability ledger: {path}"
        ) from error
    if not rows:
        raise UgiGeneratedRoleApplicabilityCensusError("applicability ledger is empty")
    return rows


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiGeneratedRoleApplicabilityCensusError(f"{label} pin is malformed")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiGeneratedRoleApplicabilityCensusError(
            f"{label} path escapes repository"
        ) from error
    if not path.is_file() or path.is_symlink() or sha256_file(path) != record["sha256"]:
        raise UgiGeneratedRoleApplicabilityCensusError(f"{label} hash changed")
    return path


def _quantiles(values: Sequence[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
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
    if not all(math.isfinite(value) and value >= 0.0 for value in ordered):
        raise UgiGeneratedRoleApplicabilityCensusError(
            "applicability census encountered an invalid distance"
        )

    def select(probability: float) -> float:
        position = probability * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "q05": select(0.05),
        "q25": select(0.25),
        "median": select(0.5),
        "q75": select(0.75),
        "q95": select(0.95),
        "maximum": ordered[-1],
    }


def _effective_count(values: Sequence[str]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return 1.0 / sum((count / total) ** 2 for count in counts.values())


def _concentration(values: Sequence[str]) -> dict[str, float | int]:
    counts = sorted(Counter(values).values(), reverse=True)
    total = sum(counts)
    if total == 0:
        return {
            "occurrences": 0,
            "unique": 0,
            "effective_inverse_simpson": 0.0,
            "largest_component_fraction": 0.0,
            "top_five_component_fraction": 0.0,
        }
    return {
        "occurrences": total,
        "unique": len(counts),
        "effective_inverse_simpson": _effective_count(values),
        "largest_component_fraction": counts[0] / total,
        "top_five_component_fraction": sum(counts[:5]) / total,
    }


def _radius_ratio(
    row: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    view: str,
    radius: str,
) -> float:
    if radius not in {"interpolative_max", "boundary_max"}:
        raise UgiGeneratedRoleApplicabilityCensusError("unknown applicability radius")
    ratios = []
    for kind in ("fingerprint", "descriptor"):
        threshold = float(thresholds[view][kind][radius])
        distance = float(row[f"{view}_{kind}_distance"])
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise UgiGeneratedRoleApplicabilityCensusError("invalid v3 applicability threshold")
        if not math.isfinite(distance) or distance < 0.0:
            raise UgiGeneratedRoleApplicabilityCensusError("invalid generated distance")
        ratios.append(distance / threshold)
    return max(ratios)


def _supported_patterns(policy: Mapping[str, Any]) -> dict[tuple[str, ...], str]:
    raw = policy.get("supported_exact_unseen_role_patterns")
    if not isinstance(raw, list) or not raw:
        raise UgiGeneratedRoleApplicabilityCensusError("supported pattern policy is missing")
    output: dict[tuple[str, ...], str] = {}
    for record in raw:
        if not isinstance(record, Mapping):
            raise UgiGeneratedRoleApplicabilityCensusError("supported pattern is malformed")
        roles = tuple(str(value) for value in record.get("unseen_roles", ()))
        pattern_id = str(record.get("pattern_id", ""))
        if not roles or not pattern_id or roles in output:
            raise UgiGeneratedRoleApplicabilityCensusError("supported pattern is ambiguous")
        output[roles] = pattern_id
    return output


def _validate_policy(policy: Mapping[str, Any]) -> dict[tuple[str, ...], str]:
    if policy.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise UgiGeneratedRoleApplicabilityCensusError("unsupported census policy schema")
    if policy.get("scope") != EXPECTED_SCOPE:
        raise UgiGeneratedRoleApplicabilityCensusError("census policy scope changed")
    restrictions = policy.get("role_pattern_policy")
    if not isinstance(restrictions, Mapping):
        raise UgiGeneratedRoleApplicabilityCensusError("role-pattern policy is missing")
    if restrictions.get("all_three_new_action") != "abstain":
        raise UgiGeneratedRoleApplicabilityCensusError("all-three-new must remain abstained")
    if restrictions.get("unsupported_pattern_action") != "abstain":
        raise UgiGeneratedRoleApplicabilityCensusError("unsupported patterns must abstain")
    if restrictions.get("exact_measured_combination_action") != "neutral":
        raise UgiGeneratedRoleApplicabilityCensusError(
            "exact measured combinations must remain neutral"
        )
    supported = _supported_patterns(restrictions)
    if supported != EXPECTED_SUPPORTED_PATTERNS:
        raise UgiGeneratedRoleApplicabilityCensusError(
            "current conditional role restrictions changed"
        )
    return supported


def _validate_prior_policy(
    policy: Mapping[str, Any],
    prior: Mapping[str, Any],
    prior_path: Path,
    repo: Path,
) -> None:
    expected_source = {
        "path": str(prior_path.relative_to(repo)),
        "sha256": sha256_file(prior_path),
    }
    if policy.get("source_policy") != expected_source:
        raise UgiGeneratedRoleApplicabilityCensusError("source policy binding changed")
    eligible = prior.get("eligible_patterns")
    if not isinstance(eligible, Mapping):
        raise UgiGeneratedRoleApplicabilityCensusError("prior potency policy is malformed")
    observed = {
        tuple(str(value) for value in record.get("unseen_roles", ())): str(pattern_id)
        for pattern_id, record in eligible.items()
        if isinstance(record, Mapping)
    }
    if observed != EXPECTED_SUPPORTED_PATTERNS:
        raise UgiGeneratedRoleApplicabilityCensusError(
            "census role restrictions diverge from the frozen potency policy"
        )


def _validate_thresholds(
    result: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> Mapping[str, Any]:
    if result.get("schema_version") != "phase1_ugi_distributional_applicability.v3":
        raise UgiGeneratedRoleApplicabilityCensusError("applicability result is not v3")
    thresholds = result.get("thresholds")
    if not isinstance(thresholds, Mapping) or set(thresholds) != set(VIEWS):
        raise UgiGeneratedRoleApplicabilityCensusError("v3 thresholds are incomplete")
    expected = policy.get("threshold_identity")
    observed = _sha256_payload(thresholds)
    if not isinstance(expected, Mapping) or expected.get("thresholds_sha256") != observed:
        raise UgiGeneratedRoleApplicabilityCensusError("v3 threshold identity changed")
    if expected.get("applicability_result_sha256") != result.get("result_sha256"):
        raise UgiGeneratedRoleApplicabilityCensusError("v3 result identity changed")
    return thresholds


def _sample_index(sample: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    records = sample.get("samples")
    if not isinstance(records, list) or not records:
        raise UgiGeneratedRoleApplicabilityCensusError("fresh terminal sample is empty")
    return {index: record for index, record in enumerate(records) if isinstance(record, Mapping)}


def _validate_terminal_row(
    row: Mapping[str, str],
    sample_record: Mapping[str, Any],
) -> None:
    forward = sample_record.get("l1_forward_verification")
    components = sample_record.get("component_smiles_by_role")
    if not (
        sample_record.get("valid") is True
        and sample_record.get("component_reconstruction_valid") is True
        and isinstance(forward, Mapping)
        and forward.get("exact_product_reconstructed") is True
        and isinstance(components, Mapping)
        and set(components) == set(ROLE_TO_SAMPLE_ROLE.values())
    ):
        raise UgiGeneratedRoleApplicabilityCensusError(
            "applicability row is not a valid exact-L1 terminal"
        )
    if str(sample_record.get("product_id")) != row["product_id"]:
        raise UgiGeneratedRoleApplicabilityCensusError("terminal product identity changed")
    if applicability_v1._canonical(str(sample_record.get("smiles"))) != applicability_v1._canonical(
        row["product_smiles"]
    ):
        raise UgiGeneratedRoleApplicabilityCensusError("terminal product graph changed")
    for role, field in ROLE_TO_SMILES_FIELD.items():
        source = str(components[ROLE_TO_SAMPLE_ROLE[role]])
        if applicability_v1._canonical(source) != applicability_v1._canonical(row[field]):
            raise UgiGeneratedRoleApplicabilityCensusError(f"terminal {role} graph changed")


def _policy_classification(
    row: Mapping[str, str],
    supported: Mapping[tuple[str, ...], str],
) -> tuple[str, str, str]:
    try:
        unseen = tuple(str(value) for value in json.loads(row["exact_unseen_roles_json"]))
    except (json.JSONDecodeError, TypeError) as error:
        raise UgiGeneratedRoleApplicabilityCensusError(
            "unseen-role pattern is malformed"
        ) from error
    if any(role not in ROLES for role in unseen) or len(set(unseen)) != len(unseen):
        raise UgiGeneratedRoleApplicabilityCensusError("unseen-role pattern is invalid")
    if row["exact_identity_provenance"] == "exact_measured_combination":
        return "", "neutral", "exact_measured_combination"
    if unseen == ROLES:
        return "", "abstain", "unsupported_all_three_new"
    pattern = supported.get(unseen)
    if pattern is None:
        return "", "abstain", "unsupported_exact_unseen_role_pattern"
    if any(row[f"{view}_distribution_bin"] != "interpolative" for view in VIEWS):
        return pattern, "abstain", "one_or_more_views_not_interpolative"
    return pattern, "applicability_supported_no_score", "supported_pattern_all_views_interpolative"


def _derived_rows(
    rows: Sequence[Mapping[str, str]],
    samples: Mapping[int, Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    supported: Mapping[tuple[str, ...], str],
) -> list[dict[str, Any]]:
    output = []
    seen_indices: set[int] = set()
    for row in rows:
        sample_index = int(row["sample_index"])
        if sample_index in seen_indices or sample_index not in samples:
            raise UgiGeneratedRoleApplicabilityCensusError("sample index is invalid or duplicated")
        seen_indices.add(sample_index)
        sample_record = samples[sample_index]
        _validate_terminal_row(row, sample_record)
        if row.get("guidance_action") != "abstain":
            raise UgiGeneratedRoleApplicabilityCensusError("input pool was not fully abstained")
        for view in VIEWS:
            if row.get(f"{view}_distribution_bin") not in BIN_NAMES:
                raise UgiGeneratedRoleApplicabilityCensusError(
                    "generated applicability bin changed"
                )
        pattern_id, action, reason = _policy_classification(row, supported)
        record: dict[str, Any] = {
            "sample_index": sample_index,
            "product_id": row["product_id"],
            "exact_identity_provenance": row["exact_identity_provenance"],
            "exact_unseen_roles_json": row["exact_unseen_roles_json"],
            "branch_class": str(sample_record.get("branch_class")),
            "source_stratum": str(sample_record.get("source_stratum")),
            "current_policy_pattern_id": pattern_id,
            "current_policy_action": action,
            "current_policy_reason": reason,
            "noninterpolative_views_json": json.dumps(
                [view for view in VIEWS if row[f"{view}_distribution_bin"] != "interpolative"],
                separators=(",", ":"),
            ),
            "extrapolative_views_json": json.dumps(
                [view for view in VIEWS if row[f"{view}_distribution_bin"] == "extrapolative"],
                separators=(",", ":"),
            ),
        }
        for view in VIEWS:
            record[f"{view}_distribution_bin"] = row[f"{view}_distribution_bin"]
            for kind in ("fingerprint", "descriptor"):
                record[f"{view}_{kind}_distance"] = float(row[f"{view}_{kind}_distance"])
            record[f"{view}_interpolative_radius_ratio"] = _radius_ratio(
                row, thresholds, view, "interpolative_max"
            )
            record[f"{view}_boundary_radius_ratio"] = _radius_ratio(
                row, thresholds, view, "boundary_max"
            )
        output.append(record)
    return sorted(output, key=lambda record: int(record["sample_index"]))


def _subset_summary(rows: Sequence[Mapping[str, Any]], view: str) -> dict[str, Any]:
    return {
        "records": len(rows),
        "distribution_bins": dict(
            sorted(Counter(str(row[f"{view}_distribution_bin"]) for row in rows).items())
        ),
        "fingerprint_distance": _quantiles(
            [float(row[f"{view}_fingerprint_distance"]) for row in rows]
        ),
        "descriptor_distance": _quantiles(
            [float(row[f"{view}_descriptor_distance"]) for row in rows]
        ),
        "interpolative_radius_ratio": _quantiles(
            [float(row[f"{view}_interpolative_radius_ratio"]) for row in rows]
        ),
        "boundary_radius_ratio": _quantiles(
            [float(row[f"{view}_boundary_radius_ratio"]) for row in rows]
        ),
    }


def summarize_applicability_rows(
    rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Summarize generated roles without consuming potency predictions."""

    if len(rows) != len(source_rows):
        raise UgiGeneratedRoleApplicabilityCensusError("derived/source row counts differ")
    source_by_index = {int(row["sample_index"]): row for row in source_rows}
    role_summaries: dict[str, Any] = {}
    for role in ROLES:
        known = []
        novel = []
        all_smiles = []
        known_smiles = []
        novel_smiles = []
        field = ROLE_TO_SMILES_FIELD[role]
        for row in rows:
            source = source_by_index[int(row["sample_index"])]
            unseen = tuple(json.loads(str(row["exact_unseen_roles_json"])))
            canonical = applicability_v1._canonical(source[field])
            all_smiles.append(canonical)
            if role in unseen:
                novel.append(row)
                novel_smiles.append(canonical)
            else:
                known.append(row)
                known_smiles.append(canonical)
        role_summaries[role] = {
            "all": {
                **_subset_summary(rows, role),
                "component_diversity": _concentration(all_smiles),
            },
            "exact_known": {
                **_subset_summary(known, role),
                "component_diversity": _concentration(known_smiles),
            },
            "exact_new": {
                **_subset_summary(novel, role),
                "component_diversity": _concentration(novel_smiles),
            },
        }

    pattern_bins: defaultdict[str, Counter[str]] = defaultdict(Counter)
    actions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    noninterpolative: Counter[str] = Counter()
    extrapolative: Counter[str] = Counter()
    branch_by_bin: defaultdict[str, Counter[str]] = defaultdict(Counter)
    exact_identity: Counter[str] = Counter()
    product_smiles = []
    for row in rows:
        unseen = tuple(json.loads(str(row["exact_unseen_roles_json"])))
        pattern = "+".join(unseen) if unseen else "none"
        overall = max(
            (str(row[f"{view}_distribution_bin"]) for view in VIEWS),
            key=applicability_v1.BIN_ORDER.__getitem__,
        )
        pattern_bins[pattern][overall] += 1
        actions[str(row["current_policy_action"])] += 1
        reasons[str(row["current_policy_reason"])] += 1
        noninterpolative[str(row["noninterpolative_views_json"])] += 1
        extrapolative[str(row["extrapolative_views_json"])] += 1
        branch_by_bin[str(row["branch_class"])][overall] += 1
        exact_identity[str(row["exact_identity_provenance"])] += 1
        product_smiles.append(
            applicability_v1._canonical(source_by_index[int(row["sample_index"])]["product_smiles"])
        )

    all_four_interpolative = sum(
        all(row[f"{view}_distribution_bin"] == "interpolative" for view in VIEWS) for row in rows
    )
    all_three_new = [
        row for row in rows if tuple(json.loads(str(row["exact_unseen_roles_json"]))) == ROLES
    ]
    if any(row["current_policy_action"] != "abstain" for row in all_three_new):
        raise UgiGeneratedRoleApplicabilityCensusError("all-three-new row escaped abstention")
    return {
        "records": len(rows),
        "product": {
            **_subset_summary(rows, "product"),
            "product_diversity": _concentration(product_smiles),
            "exact_identity_provenance": dict(sorted(exact_identity.items())),
        },
        "roles": role_summaries,
        "joint": {
            "all_four_views_interpolative": all_four_interpolative,
            "all_three_roles_exact_new": len(all_three_new),
            "all_three_roles_exact_new_all_four_views_interpolative": sum(
                all(row[f"{view}_distribution_bin"] == "interpolative" for view in VIEWS)
                for row in all_three_new
            ),
            "current_policy_actions": dict(sorted(actions.items())),
            "current_policy_reasons": dict(sorted(reasons.items())),
            "exact_unseen_role_pattern_by_overall_bin": {
                pattern: dict(sorted(counts.items()))
                for pattern, counts in sorted(pattern_bins.items())
            },
            "noninterpolative_view_failure_matrix": dict(sorted(noninterpolative.items())),
            "extrapolative_view_failure_matrix": dict(sorted(extrapolative.items())),
            "branch_class_by_overall_bin": {
                branch: dict(sorted(counts.items()))
                for branch, counts in sorted(branch_by_bin.items())
            },
        },
    }


def _ledger_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=LEDGER_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in LEDGER_FIELDS})
    raw = buffer.getvalue().encode()
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", filename="", mtime=0) as handle:
        handle.write(raw)
    return compressed.getvalue()


def build_generated_role_applicability_census(
    repo: Path,
    config_path: Path,
) -> tuple[dict[str, Any], bytes]:
    """Build a nonselecting readiness receipt from frozen terminal outputs."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load_json(config_path, label="generated-role census readiness config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiGeneratedRoleApplicabilityCensusError("unsupported readiness config schema")
    if config.get("scope") != EXPECTED_SCOPE:
        raise UgiGeneratedRoleApplicabilityCensusError("readiness scope changed")
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != EXPECTED_INPUTS:
        raise UgiGeneratedRoleApplicabilityCensusError("readiness input set changed")
    paths = {name: _pin(repo, record, label=name) for name, record in inputs.items()}

    policy = _load_json(paths["bio_policy"], label="generated-role census policy")
    supported = _validate_policy(policy)
    prior_policy = _load_json(paths["prior_bio_policy"], label="prior potency policy")
    _validate_prior_policy(policy, prior_policy, paths["prior_bio_policy"], repo)
    applicability_result = _load_json(
        paths["applicability_result"], label="v3 applicability result"
    )
    thresholds = _validate_thresholds(applicability_result, policy)
    source_rows = _read_csv(paths["generated_applicability"])
    expected_ledger = applicability_result.get("artifacts", {}).get(
        "generated_applicability.csv.gz", {}
    )
    if expected_ledger.get("records") != len(source_rows) or expected_ledger.get(
        "sha256"
    ) != sha256_bytes(paths["generated_applicability"].read_bytes()):
        raise UgiGeneratedRoleApplicabilityCensusError("v3 generated ledger identity changed")

    fresh_config = _load_json(paths["fresh_pool_config"], label="fresh-pool config")
    fresh_audit = _load_json(paths["fresh_pool_audit"], label="fresh-pool audit")
    fresh_sample = _load_json(paths["fresh_pool_sample"], label="fresh-pool sample")
    samples = _sample_index(fresh_sample)
    summary = fresh_audit.get("summary")
    if not isinstance(summary, Mapping):
        raise UgiGeneratedRoleApplicabilityCensusError("fresh-pool summary is missing")
    attempted = int(summary.get("attempted_draws", -1))
    valid = int(summary.get("valid_products", -1))
    invalid_types = summary.get("invalid_failure_types")
    if (
        attempted != len(samples)
        or valid != len(source_rows)
        or summary.get("exact_l1_products") != valid
        or summary.get("exact_l1_fraction_of_valid") != 1.0
        or not isinstance(invalid_types, Mapping)
        or valid + sum(int(value) for value in invalid_types.values()) != attempted
    ):
        raise UgiGeneratedRoleApplicabilityCensusError("terminal attempt partition changed")
    if fresh_config.get("purpose") != (
        "Generate an independent nonselecting v2 product-plus-L1 pool for novelty, "
        "oracle-applicability and recursive-route coverage evaluation."
    ):
        raise UgiGeneratedRoleApplicabilityCensusError("fresh-pool purpose changed")

    rows = _derived_rows(source_rows, samples, thresholds, supported)
    census = summarize_applicability_rows(rows, source_rows)
    ledger = _ledger_bytes(rows)
    content: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_read_only_generated_role_census_nonzero_guidance_blocked",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "scope": EXPECTED_SCOPE,
        "terminal_accounting": {
            "scheduled_attempts": attempted,
            "valid_exact_l1_terminals": valid,
            "invalid_or_unsupported_terminals": attempted - valid,
            "invalid_failure_types": {
                str(key): int(value) for key, value in sorted(invalid_types.items())
            },
            "every_scheduled_attempt_accounted_for": True,
        },
        "generation_contract": {
            "primary_path": "fully_generated_amine_aldehyde_isocyanide_graphs",
            "fixed_head_path": "diagnostic_ablation_only",
            "terminal_source_strata": dict(
                sorted(
                    Counter(
                        str(record.get("source_stratum")) for record in samples.values()
                    ).items()
                )
            ),
            "component_novelty_classes": dict(
                sorted(
                    Counter(
                        str(record.get("component_novelty_class")) for record in samples.values()
                    ).items()
                )
            ),
        },
        "applicability_contract": {
            "version": 3,
            "thresholds_sha256": _sha256_payload(thresholds),
            "threshold_inputs": "structures_and_split_stage_only",
            "views": list(VIEWS),
            "both_fingerprint_and_descriptor_views_required": True,
            "worst_view_defines_overall_bin": True,
            "supported_exact_unseen_role_patterns": {
                "+".join(roles): pattern for roles, pattern in sorted(supported.items())
            },
            "all_three_new_action": "abstain",
            "unsupported_pattern_action": "abstain",
            "exact_measured_combination_action": "neutral",
        },
        "census": census,
        "artifacts": {
            "ledger.csv.gz": {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "records": len(rows),
                "sha256": sha256_bytes(ledger),
                "contains_oracle_predictions": False,
                "contains_potency_scores": False,
            }
        },
        "adjudication": {
            "readiness_for_trust_region_controller_design": True,
            "readiness_for_nonzero_guidance_execution": False,
            "biological_guidance_authorized": False,
            "synthesis_guidance_authorized": False,
            "candidate_selection_changed": False,
            "next_gate": (
                "review role-specific radius distributions and freeze a fully generated-role "
                "controller plus matched lambda-zero identity before any nonzero execution"
            ),
            "controller_constraints": [
                "generate all three precursor roles without identity clamping",
                "treat applicability as a gate or risk penalty, never a potency bonus",
                "preserve current conditional role-pattern restrictions",
                "abstain on unsupported all-three-new candidates until separately calibrated",
                "freeze separate component-novelty and diversity safeguards",
            ],
        },
        "inputs": {
            name: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for name, path in sorted(paths.items())
        },
    }
    return {**content, "result_sha256": _sha256_payload(content)}, ledger


__all__ = [
    "UgiGeneratedRoleApplicabilityCensusError",
    "_radius_ratio",
    "build_generated_role_applicability_census",
    "summarize_applicability_rows",
]
