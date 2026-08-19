"""Per-platform addendum to the M0-04 reaction-registry coverage audit."""

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import rdBase

from forge.data.r1_prime_audit import (
    AuditError,
    ReactionDefinition,
    compile_reactions,
    decompose_structure,
    load_reaction_definitions,
    sha256_file,
)
from forge.data.r1_prime_audit import load_config as load_m0_04_config

CONFIG_SCHEMA_VERSION = "m0_04_source_platform_registry_audit_config.v1"
RESULT_SCHEMA_VERSION = "m0_04_source_platform_registry_audit.v1"
LEDGER_NAME = "source_platform_registry_coverage.csv.gz"
RESULT_NAME = "source_platform_registry_audit.json"


class SourcePlatformAuditError(RuntimeError):
    """Raised when the bounded source-platform audit cannot be trusted."""


@dataclass(frozen=True)
class PlatformSpec:
    """One frozen source-platform cohort and its reported chemistry evidence."""

    platform_id: str
    aliases: frozenset[str]
    expected_structures: int
    reported_final_assembly: Mapping[str, str]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise SourcePlatformAuditError(f"cannot load {description} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SourcePlatformAuditError(f"{description} must be a JSON object")
    return value


def _all_strings(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        result: set[str] = set()
        for item in value:
            result.update(_all_strings(item))
        return result
    if isinstance(value, dict):
        result = set()
        for item in value.values():
            result.update(_all_strings(item))
        return result
    return set()


def _load_platform_specs(raw: Any) -> tuple[PlatformSpec, ...]:
    if not isinstance(raw, list) or not raw:
        raise SourcePlatformAuditError("platforms must be a nonempty list")
    specs: list[PlatformSpec] = []
    seen_ids: set[str] = set()
    seen_aliases: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise SourcePlatformAuditError(f"platforms[{index}] must be an object")
        platform_id = item.get("platform_id")
        aliases = item.get("aliases")
        expected = item.get("expected_structures")
        reported = item.get("reported_final_assembly")
        if not isinstance(platform_id, str) or not platform_id:
            raise SourcePlatformAuditError(f"platforms[{index}] has invalid platform_id")
        if (
            not isinstance(aliases, list)
            or not aliases
            or any(not isinstance(alias, str) or not alias for alias in aliases)
        ):
            raise SourcePlatformAuditError(f"{platform_id}: aliases must be nonempty strings")
        if not isinstance(expected, int) or expected <= 0:
            raise SourcePlatformAuditError(f"{platform_id}: expected_structures must be positive")
        if not isinstance(reported, dict) or not {
            "status",
            "description",
        }.issubset(reported):
            raise SourcePlatformAuditError(
                f"{platform_id}: reported_final_assembly is incomplete"
            )
        alias_set = frozenset(aliases)
        if platform_id in seen_ids or seen_aliases.intersection(alias_set):
            raise SourcePlatformAuditError(f"{platform_id}: duplicate platform or alias")
        seen_ids.add(platform_id)
        seen_aliases.update(alias_set)
        specs.append(
            PlatformSpec(
                platform_id=platform_id,
                aliases=alias_set,
                expected_structures=expected,
                reported_final_assembly={
                    str(key): str(value) for key, value in sorted(reported.items())
                },
            )
        )
    return tuple(specs)


def load_config(path: Path) -> tuple[dict[str, Any], tuple[PlatformSpec, ...]]:
    """Load and validate the source-platform audit configuration."""

    config = _load_json(path, "source-platform audit config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise SourcePlatformAuditError(
            f"config schema must be {CONFIG_SCHEMA_VERSION!r}"
        )
    expected_inputs = config.get("expected_inputs")
    if (
        not isinstance(expected_inputs, dict)
        or not expected_inputs
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or len(digest) != 64
            for name, digest in expected_inputs.items()
        )
    ):
        raise SourcePlatformAuditError("expected_inputs must map assets to SHA-256 digests")
    expected_total = config.get("expected_source_study_heldout")
    if not isinstance(expected_total, int) or expected_total <= 0:
        raise SourcePlatformAuditError("expected_source_study_heldout must be positive")
    for field in (
        "max_reverse_outcomes_per_structure_reaction",
        "max_forward_outcomes_per_reconstruction",
    ):
        if not isinstance(config.get(field), int) or config[field] <= 0:
            raise SourcePlatformAuditError(f"{field} must be a positive integer")
    claims = config.get("claims_boundary")
    if not isinstance(claims, dict) or any(value is not True for value in claims.values() if isinstance(value, bool)):
        raise SourcePlatformAuditError("claims_boundary boolean safeguards must be true")
    return config, _load_platform_specs(config.get("platforms"))


def _verify_inputs(paths: Mapping[str, Path], expected: Mapping[str, str]) -> list[dict[str, Any]]:
    if set(paths) != set(expected):
        raise SourcePlatformAuditError(
            f"input asset set mismatch: expected {sorted(expected)}, received {sorted(paths)}"
        )
    records = []
    for asset in sorted(paths):
        path = paths[asset]
        try:
            digest = sha256_file(path)
            size = path.stat().st_size
        except FileNotFoundError as exc:
            raise SourcePlatformAuditError(f"required input is missing: {path}") from exc
        if digest != expected[asset]:
            raise SourcePlatformAuditError(
                f"{asset}: hash mismatch; expected {expected[asset]}, observed {digest}"
            )
        records.append(
            {
                "asset": asset,
                "bytes": size,
                "sha256": digest,
            }
        )
    return records


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        handle = path.open(newline="")
    except FileNotFoundError as exc:
        raise SourcePlatformAuditError(f"CSV input is missing: {path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SourcePlatformAuditError(f"CSV input has no header: {path}")
        return list(reader)


def _source_heldout_ids(path: Path) -> set[str]:
    rows = _read_csv_rows(path)
    required = {"r0_structure_id", "source_study_fold"}
    if not rows or not required.issubset(rows[0]):
        raise SourcePlatformAuditError(f"{path.name}: missing {sorted(required)}")
    return {
        row["r0_structure_id"]
        for row in rows
        if row["source_study_fold"] == "R0_heldout"
    }


def _classify_platform(
    row: Mapping[str, str],
    specs: Sequence[PlatformSpec],
) -> str:
    try:
        groups = json.loads(row["study_split_groups_json"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise SourcePlatformAuditError(
            f"{row.get('r0_structure_id', '<unknown>')}: invalid study_split_groups_json"
        ) from exc
    tokens = _all_strings(groups)
    matches = [spec.platform_id for spec in specs if spec.aliases.intersection(tokens)]
    if len(matches) != 1:
        raise SourcePlatformAuditError(
            f"{row['r0_structure_id']}: expected one platform match, observed {matches}"
        )
    return matches[0]


def _load_heldout_rows(
    r0_path: Path,
    split_path: Path,
    specs: Sequence[PlatformSpec],
    expected_total: int,
) -> dict[str, list[dict[str, str]]]:
    heldout_ids = _source_heldout_ids(split_path)
    if len(heldout_ids) != expected_total:
        raise SourcePlatformAuditError(
            f"source-study heldout has {len(heldout_ids)} rows; expected {expected_total}"
        )
    r0_rows = _read_csv_rows(r0_path)
    required = {
        "r0_structure_id",
        "canonical_isomeric_smiles",
        "observed_source_ids",
        "study_split_groups_json",
    }
    if not r0_rows or not required.issubset(r0_rows[0]):
        raise SourcePlatformAuditError(f"{r0_path.name}: missing {sorted(required)}")
    by_platform: dict[str, list[dict[str, str]]] = defaultdict(list)
    found_ids: set[str] = set()
    for row in r0_rows:
        structure_id = row["r0_structure_id"]
        if structure_id not in heldout_ids:
            continue
        by_platform[_classify_platform(row, specs)].append(row)
        found_ids.add(structure_id)
    if found_ids != heldout_ids:
        missing = sorted(heldout_ids.difference(found_ids))
        raise SourcePlatformAuditError(f"heldout R0 row is missing: {missing[0]}")
    for spec in specs:
        observed = len(by_platform[spec.platform_id])
        if observed != spec.expected_structures:
            raise SourcePlatformAuditError(
                f"{spec.platform_id}: observed {observed}; expected {spec.expected_structures}"
            )
        by_platform[spec.platform_id].sort(key=lambda row: row["r0_structure_id"])
    return dict(by_platform)


def _load_source_pool(path: Path) -> set[tuple[str, str, str]]:
    try:
        handle = gzip.open(path, "rt", newline="")
        rows = list(csv.DictReader(handle))
        handle.close()
    except (FileNotFoundError, gzip.BadGzipFile, OSError) as exc:
        raise SourcePlatformAuditError(f"cannot read component pool {path}: {exc}") from exc
    required = {"scheme", "reaction_id", "role", "canonical_smiles"}
    if not rows or not required.issubset(rows[0]):
        raise SourcePlatformAuditError(f"{path.name}: missing {sorted(required)}")
    return {
        (row["reaction_id"], row["role"], row["canonical_smiles"])
        for row in rows
        if row["scheme"] == "source_study"
    }


def _role_index(
    definitions: Sequence[ReactionDefinition],
) -> dict[str, tuple[str, ...]]:
    return {
        definition.reaction_id: tuple(role.name for role in definition.reactant_roles)
        for definition in definitions
    }


def _audit_platform(
    spec: PlatformSpec,
    rows: Sequence[Mapping[str, str]],
    definitions: Sequence[ReactionDefinition],
    source_pool: set[tuple[str, str, str]],
    max_reverse_outcomes: int,
    max_forward_outcomes: int,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    compiled = compile_reactions(definitions)
    roles = _role_index(definitions)
    reaction_candidates: Counter[str] = Counter()
    products_by_reaction: dict[str, set[str]] = defaultdict(set)
    rejection_counts: Counter[str] = Counter()
    unique_missing: set[tuple[str, str, str]] = set()
    products_with_candidates = 0
    products_with_pool_supported_candidate = 0
    ledger: list[dict[str, str]] = []
    for row in rows:
        candidates = decompose_structure(
            row,
            "source_study",
            compiled,
            max_reverse_outcomes,
            max_forward_outcomes,
            rejection_counts,
        )
        supported_reactions: set[str] = set()
        missing_for_product: set[tuple[str, str, str]] = set()
        for candidate in candidates:
            candidate_roles = roles[candidate.reaction_id]
            if len(candidate_roles) != len(candidate.reactant_smiles):
                raise SourcePlatformAuditError(
                    f"{candidate.reaction_id}: candidate role count mismatch"
                )
            missing = {
                (candidate.reaction_id, role, smiles)
                for role, smiles in zip(
                    candidate_roles,
                    candidate.reactant_smiles,
                    strict=True,
                )
                if (candidate.reaction_id, role, smiles) not in source_pool
            }
            if missing:
                missing_for_product.update(missing)
                unique_missing.update(missing)
            else:
                supported_reactions.add(candidate.reaction_id)
            reaction_candidates[candidate.reaction_id] += 1
            products_by_reaction[candidate.reaction_id].add(row["r0_structure_id"])
        if candidates:
            products_with_candidates += 1
        if supported_reactions:
            products_with_pool_supported_candidate += 1
        ledger.append(
            {
                "platform_id": spec.platform_id,
                "r0_structure_id": row["r0_structure_id"],
                "exact_registered_decomposition_count": str(len(candidates)),
                "exact_registered_reactions_json": json.dumps(
                    sorted({candidate.reaction_id for candidate in candidates}),
                    separators=(",", ":"),
                ),
                "pool_supported_reactions_json": json.dumps(
                    sorted(supported_reactions),
                    separators=(",", ":"),
                ),
                "missing_component_count": str(len(missing_for_product)),
                "missing_components_json": json.dumps(
                    [
                        {
                            "reaction_id": reaction_id,
                            "role": role,
                            "canonical_smiles": smiles,
                        }
                        for reaction_id, role, smiles in sorted(missing_for_product)
                    ],
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            }
        )
    if products_with_pool_supported_candidate:
        raise SourcePlatformAuditError(
            f"{spec.platform_id}: {products_with_pool_supported_candidate} heldout products "
            "have a fully training-pool-supported exact decomposition despite zero reported "
            "source-study recovery"
        )
    summary = {
        "heldout_structures": len(rows),
        "products_with_exact_registered_decomposition": products_with_candidates,
        "products_without_exact_registered_decomposition": (
            len(rows) - products_with_candidates
        ),
        "products_with_fully_training_pool_supported_decomposition": (
            products_with_pool_supported_candidate
        ),
        "total_exact_registered_decompositions": sum(reaction_candidates.values()),
        "exact_decomposition_candidates_by_reaction": dict(
            sorted(reaction_candidates.items())
        ),
        "products_with_exact_decomposition_by_reaction": {
            reaction_id: len(structure_ids)
            for reaction_id, structure_ids in sorted(products_by_reaction.items())
        },
        "unique_missing_training_components": len(unique_missing),
        "unique_missing_training_components_by_reaction_role": dict(
            sorted(
                Counter(
                    f"{reaction_id}:{role}"
                    for reaction_id, role, _ in unique_missing
                ).items()
            )
        ),
        "rejected_reverse_outcomes": dict(sorted(rejection_counts.items())),
        "reported_final_assembly": dict(spec.reported_final_assembly),
        "registry_interpretation": (
            "no_exact_registered_decomposition"
            if not products_with_candidates
            else "alternative_registered_decomposition_exists_but_training_components_are_missing"
        ),
    }
    return summary, ledger


def _ledger_payload(rows: Sequence[Mapping[str, str]]) -> bytes:
    fields = (
        "platform_id",
        "r0_structure_id",
        "exact_registered_decomposition_count",
        "exact_registered_reactions_json",
        "pool_supported_reactions_json",
        "missing_component_count",
        "missing_components_json",
    )
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return gzip.compress(text.getvalue().encode(), compresslevel=9, mtime=0)


def build_source_platform_audit(
    config_path: Path,
    r0_path: Path,
    split_path: Path,
    registry_paths: Sequence[Path],
    m0_04_config_path: Path,
    m0_04_result_path: Path,
    component_pool_path: Path,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Build the exact per-platform registry-coverage addendum."""

    config, specs = load_config(config_path)
    input_paths = {
        "r0_observed_real_structures.csv": r0_path,
        "r0_fold_assignments.csv": split_path,
        "qualified_reaction_families_v1.json": registry_paths[0],
        "qualified_reactions_v1.json": registry_paths[1],
        "m0_04_r1_prime_audit.json": m0_04_config_path,
        "m0_04_result.json": m0_04_result_path,
        "component_pool.csv.gz": component_pool_path,
    }
    input_records = _verify_inputs(input_paths, config["expected_inputs"])
    m0_04_config = load_m0_04_config(m0_04_config_path)
    m0_04_result = _load_json(m0_04_result_path, "M0-04 result")
    try:
        source_recovery = m0_04_result["schemes"]["source_study"][
            "r1_prime_exact_recovery"
        ]["all_heldout"]
    except (KeyError, TypeError) as exc:
        raise SourcePlatformAuditError("M0-04 result lacks source-study recovery") from exc
    if (
        source_recovery.get("denominator") != config["expected_source_study_heldout"]
        or source_recovery.get("recovered") != 0
    ):
        raise SourcePlatformAuditError(
            "source-platform addendum requires the frozen 0/2,333 source-study result"
        )
    try:
        definitions = load_reaction_definitions(
            registry_paths,
            expected_count=m0_04_config["expected_reaction_count"],
            role_policy_overrides=m0_04_config["role_policy_overrides"],
        )
    except AuditError as exc:
        raise SourcePlatformAuditError(str(exc)) from exc
    heldout = _load_heldout_rows(
        r0_path,
        split_path,
        specs,
        config["expected_source_study_heldout"],
    )
    source_pool = _load_source_pool(component_pool_path)
    summaries: dict[str, Any] = {}
    ledger_rows: list[dict[str, str]] = []
    for spec in specs:
        summary, rows = _audit_platform(
            spec,
            heldout[spec.platform_id],
            definitions,
            source_pool,
            config["max_reverse_outcomes_per_structure_reaction"],
            config["max_forward_outcomes_per_reconstruction"],
        )
        summaries[spec.platform_id] = summary
        ledger_rows.extend(rows)
    ledger_rows.sort(key=lambda row: (row["platform_id"], row["r0_structure_id"]))
    ledger_payload = _ledger_payload(ledger_rows)
    artifacts = {LEDGER_NAME: ledger_payload}
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": config["task"],
        "generated_utc": config["generated_utc"],
        "randomness": config["randomness"],
        "inputs": [
            {
                "asset": config_path.name,
                "bytes": config_path.stat().st_size,
                "sha256": sha256_file(config_path),
            },
            *input_records,
        ],
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
        },
        "summary": {
            "source_study_heldout": sum(
                details["heldout_structures"] for details in summaries.values()
            ),
            "products_with_exact_registered_decomposition": sum(
                details["products_with_exact_registered_decomposition"]
                for details in summaries.values()
            ),
            "products_with_fully_training_pool_supported_decomposition": sum(
                details["products_with_fully_training_pool_supported_decomposition"]
                for details in summaries.values()
            ),
            "platforms": summaries,
        },
        "artifacts": {
            LEDGER_NAME: {
                "path": f"results/m0_04/{LEDGER_NAME}",
                "bytes": len(ledger_payload),
                "sha256": _sha256_bytes(ledger_payload),
            }
        },
        "claims_boundary": config["claims_boundary"],
        "decision": {
            "source_holdout_is_registry_coverage_and_leakage_audit": True,
            "source_holdout_is_model_generalization_benchmark": False,
            "heldout_components_added_to_pool": False,
            "model_built": False,
        },
    }
    return result, artifacts


def write_source_platform_audit(
    result: Mapping[str, Any],
    artifacts: Mapping[str, bytes],
    output_dir: Path,
) -> None:
    """Atomically replace the deterministic audit artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        **artifacts,
        RESULT_NAME: (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    }
    temporary_paths: dict[str, Path] = {}
    try:
        for name, payload in payloads.items():
            descriptor, temporary = tempfile.mkstemp(
                dir=output_dir,
                prefix=f".{name}.",
                suffix=".tmp",
            )
            temporary_path = Path(temporary)
            temporary_paths[name] = temporary_path
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        for name in sorted(payloads):
            os.replace(temporary_paths[name], output_dir / name)
    except Exception:
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)
        raise
