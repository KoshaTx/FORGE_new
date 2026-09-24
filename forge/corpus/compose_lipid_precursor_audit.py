"""Audit precursor identities across authenticated TRAIN program ledgers.

This consumes existing computed-program evidence, without upgrading it or parsing provider
evaluation graphs. A source label can resolve only within its observed family and role. Missing
or conflicting identities remain unresolved; this audit never admits a training row.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from forge.assembly.compose_lipid import ComposeLipidError, role_metadata
from forge.assembly.families import constitutional_molecule
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid import _frozen_guards
from forge.corpus.compose_lipid_protection import verify_product_protection
from forge.corpus.library_splits import FrozenIdentityFolds

CONFIG_SCHEMA = "forge.compose_lipid_precursor_audit_config.v1"
RESULT_SCHEMA = "forge.compose_lipid_precursor_audit.v1"
POLICY = {
    "seed": 0,
    "random_sampling_used": False,
    "provider_evaluation_graphs_parsed": False,
    "source_splits_changed": False,
    "training_calls": 0,
    "training_rows_admitted": 0,
    "identity_scope": "exact_preparation_components_family_role_scoped_labels",
    "unknown_labels": "unresolved_never_assumed_novel_or_disjoint",
    "chemistry_scope": "consume_pinned_program_evidence_no_new_chemistry_qualification",
}
# These are persisted record formats, not chemical definitions or admission policies.
LEDGER_FORMATS = {
    "forge.compose_lipid_source_program.v1": ("program_checks.jsonl.gz", "qualified"),
    "forge.compose_lipid_source_event.v1": ("programs.jsonl.gz", "computed_consistency_pass"),
    "forge.compose_lipid_repeated_inverse.v1": ("programs.jsonl.gz", "computed_consistency_pass"),
    "forge.compose_lipid_scaffold_event.v1": ("programs.jsonl.gz", "computed_consistency_pass"),
    "forge.compose_lipid_sequential.v1": ("programs.jsonl.gz", "computed_consistency_pass"),
}
IMPLEMENTATION = (
    "forge/corpus/compose_lipid_precursor_audit.py",
    "forge/corpus/compose_lipid_protection.py",
    "forge/corpus/compose_lipid.py",
    "forge/corpus/library_splits.py",
    "forge/assembly/compose_lipid.py",
    "forge/assembly/families.py",
    "forge/core/hashing.py",
)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read(path: Path) -> Iterable[dict]:
    with gzip.open(path, "rt") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ComposeLipidError(f"non-object ledger row: {path}")
            yield value


def _pin(repo: Path, path: Path) -> dict:
    return {"path": path.relative_to(repo).as_posix(), "sha256": str(sha256_file(path))}


def _labels(row: dict, catalogue: dict) -> dict[str, str]:
    metadata = role_metadata(row, catalogue[row["family"]])
    if metadata["missing_role_metadata"]:
        raise ComposeLipidError(f"{row['target_id']}: missing source role labels")
    return {role: _json(value["metadata"]) for role, value in metadata["roles"].items()}


def audit_precursors(
    preparation: Mapping[str, dict],
    program_rows: Iterable[dict],
    evaluation_metadata: Iterable[dict],
    catalogue: dict,
    guards: FrozenIdentityFolds,
) -> dict:
    """Join exact TRAIN evidence, propagate known exclusions and audit metadata-only panels.

    Evaluation labels may query the frozen TRAIN dictionary but never extend it. Canonical
    identities are recomputed once per distinct precursor, independent of its role or family.
    """
    if {row["family"] for row in preparation.values()} != set(catalogue):
        raise ComposeLipidError("preparation must account for every imported family")
    exact: dict[str, list[dict]] = {}
    seen: set[str] = set()
    labels_to_ids: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    canonical: dict[str, str] = {}
    identities: dict[str, str] = {}
    uses: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    counts = {family: Counter() for family in sorted(catalogue)}
    for target, row in preparation.items():
        if target != row["target_id"] or type(row["eligible_for_program_preparation"]) is not bool:
            raise ComposeLipidError("invalid preparation identity or disposition")
        count = counts[row["family"]]
        count["inspection_rows"] += 1
        count["prior_product_exclusions"] += not row["eligible_for_program_preparation"]
        count["preparation_rows"] += row["eligible_for_program_preparation"]

    for row in program_rows:
        target, family = row["target_id"], row["family"]
        if target in seen or target not in preparation:
            raise ComposeLipidError(f"duplicate or non-TRAIN program target: {target}")
        seen.add(target)
        source = preparation[target]
        if family != source["family"] or row.get("training_admitted") is not False:
            raise ComposeLipidError(f"program family or training scope changed: {target}")
        if type(row.get("exact_program")) is not bool:
            raise ComposeLipidError(f"non-boolean exact-program status: {target}")
        if not source["eligible_for_program_preparation"]:
            counts[family]["program_rows_excluded_by_prior_product"] += 1
            continue
        counts[family]["program_rows_evaluated"] += 1
        if not row["exact_program"]:
            counts[family]["program_rows_not_exact"] += 1
            continue
        if row.get("component_label_conflict", False) is not False:
            raise ComposeLipidError(f"exact-program evidence carries a label conflict: {target}")
        source_labels = _labels(source, catalogue)
        components = row.get("components")
        if (
            not isinstance(components, list)
            or len(components) != len(source_labels)
            or {c["role"] for c in components} != set(source_labels)
        ):
            raise ComposeLipidError(f"incomplete exact-program components: {target}")
        normalized = []
        for component in components:
            role = component["role"]
            label = source_labels[role]
            if "source_label" in component:
                matches = _json(json.loads(component["source_label"])) == label
            else:
                values = list(json.loads(label).values())
                matches = len(values) == 1 and component.get("source_id") == values[0]
            if not matches:
                raise ComposeLipidError(f"component label disagrees with source: {target}/{role}")
            smiles = component["canonical_smiles"]
            if smiles not in canonical:
                canonical[smiles] = constitutional_molecule(smiles)[0]
            identity = hashlib.sha256(canonical[smiles].encode()).hexdigest()
            if component["constitution_id"] != identity or canonical[smiles] != smiles:
                raise ComposeLipidError(
                    f"component constitutional identity changed: {target}/{role}"
                )
            fold = guards.effective(identity)
            if component.get("historical_fold") != fold:
                raise ComposeLipidError(f"component historical protection changed: {target}/{role}")
            identities[identity] = smiles
            key = (family, role, label)
            labels_to_ids[key].add(identity)
            uses[identity].append((target, family, role, label))
            normalized.append(
                {"identity": identity, "key": key, "protected": fold in {"calibration", "heldout"}}
            )
        if row.get("historical_protected_precursor") is not any(c["protected"] for c in normalized):
            raise ComposeLipidError(f"program protection summary changed: {target}")
        exact[target] = normalized
        counts[family]["exact_program_rows"] += 1

    conflicts = {key for key, values in labels_to_ids.items() if len(values) != 1}
    row_ledger = []
    potential_ids: set[str] = set()
    for target, source in sorted(preparation.items()):
        family = source["family"]
        components = exact.get(target)
        reasons = []
        if not source["eligible_for_program_preparation"]:
            reasons.append("prior_protected_product")
        elif components is None:
            reasons.append(
                "no_exact_program_evidence" if target not in seen else "program_not_exact"
            )
        else:
            if any(c["protected"] for c in components):
                reasons.append("historical_protected_precursor")
                counts[family]["exact_rows_with_known_protected_precursor"] += 1
            if any(c["key"] in conflicts for c in components):
                reasons.append("conflicting_scoped_component_label")
                counts[family]["exact_rows_with_conflicting_labels"] += 1
        clear = not reasons
        if clear:
            counts[family]["exact_rows_clear_of_known_exclusions"] += 1
            potential_ids.update(c["identity"] for c in components)
        row_ledger.append(
            {
                "target_id": target,
                "family": family,
                "constitution_id": source["constitution_id"],
                "exact_program_evidence": components is not None,
                "component_ids": sorted(c["identity"] for c in components or []),
                "known_exclusion_reasons": reasons,
                "clear_of_known_exclusions": clear,
                "training_admitted": False,
            }
        )

    panels = []
    panel_seen: set[str] = set()
    for row in evaluation_metadata:
        target, family = row["target_id"], row["family"]
        if target in panel_seen or target in preparation:
            raise ComposeLipidError(f"duplicate or TRAIN evaluation target: {target}")
        panel_seen.add(target)
        if row["split"] not in {"calibration", "heldout"} or family not in catalogue:
            raise ComposeLipidError(f"unknown evaluation split or family: {target}")
        malformed = None
        try:
            labels = _labels(row, catalogue)
        except ComposeLipidError as exc:
            labels, malformed = {}, str(exc)
        resolved = {}
        unresolved = []
        for role, label in labels.items():
            observed = labels_to_ids.get((family, role, label), set())
            if len(observed) == 1:
                resolved[role] = next(iter(observed))
            else:
                unresolved.append(role)
        complete = bool(labels) and not unresolved and malformed is None
        all_seen = complete and all(identity in potential_ids for identity in resolved.values())
        panels.append(
            {
                "target_id": target,
                "family": family,
                "split": row["split"],
                "test_panels": sorted(row["test_panels"]),
                "resolved_role_identities": resolved,
                "unresolved_roles": sorted(unresolved),
                "malformed_metadata": malformed,
                "all_roles_resolved_from_preparation": complete,
                "all_role_identities_present_in_clear_exact_rows": all_seen,
                "unseen_precursor_label_claim_contradicted": all_seen
                and "unseen_precursor_identity" in row["test_panels"],
                "chemical_identity_holdout_qualified": False,
            }
        )
    panels.sort(key=lambda row: row["target_id"])
    panel_counts: dict[str, Counter] = defaultdict(Counter)
    for row in panels:
        for panel in ["all_evaluation_rows", *row["test_panels"]]:
            count = panel_counts[panel]
            count["rows"] += 1
            for field in (
                "all_roles_resolved_from_preparation",
                "all_role_identities_present_in_clear_exact_rows",
                "unseen_precursor_label_claim_contradicted",
            ):
                count[field] += row[field]
            count["malformed_metadata"] += row["malformed_metadata"] is not None
    identity_ledger = [
        {
            "constitution_id": identity,
            "canonical_smiles": identities[identity],
            "historical_fold": guards.effective(identity),
            "families": sorted({family for _, family, _, _ in records}),
            "roles": sorted({role for _, _, role, _ in records}),
            "source_labels": [
                {"family": family, "role": role, "label": label}
                for family, role, label in sorted({record[1:] for record in records})
            ],
            "exact_program_target_count": len({target for target, _, _, _ in records}),
            "present_in_clear_exact_rows": identity in potential_ids,
        }
        for identity, records in sorted(uses.items())
    ]
    for count in counts.values():
        count["preparation_rows_without_program_evidence"] = (
            count["preparation_rows"] - count["program_rows_evaluated"]
        )
    return {
        "rows": row_ledger,
        "identities": identity_ledger,
        "panels": panels,
        "summary": {
            "by_family": {family: dict(sorted(count.items())) for family, count in counts.items()},
            "totals": dict(sorted(sum(counts.values(), Counter()).items())),
            "unique_exact_precursor_identities": len(identities),
            "identities_shared_across_families": sum(
                len(r["families"]) > 1 for r in identity_ledger
            ),
            "identities_shared_across_roles": sum(len(r["roles"]) > 1 for r in identity_ledger),
            "conflicting_scoped_labels": len(conflicts),
            "panels": {
                name: dict(sorted(count.items())) for name, count in sorted(panel_counts.items())
            },
            "training_ready": False,
            "training_rows_admitted": 0,
            "global_precursor_holdout_qualified": False,
            "decomposition_precision": "not_established_by_this_identity_audit",
        },
    }


def _evaluate(repo: Path, config: dict) -> dict:
    if (
        set(config) != {"schema_version", "policy", "protection_result", "program_results"}
        or config["schema_version"] != CONFIG_SCHEMA
        or config["policy"] != POLICY
    ):
        raise ComposeLipidError("precursor audit configuration or scope changed")
    protection = verify_product_protection(
        repo, resolve_pin(config["protection_result"], repo, label="protection")
    )
    imported = json.loads(
        resolve_pin(protection["inputs"]["import_result"], repo, label="import").read_text()
    )
    import_config = json.loads(
        resolve_pin(imported["config"], repo, label="import config").read_text()
    )
    paths = {name: resolve_pin(pin, repo, label=name) for name, pin in imported["inputs"].items()}
    guards = _frozen_guards(import_config["historical_identity_guards"], paths)
    catalogue = json.loads(
        resolve_pin(
            imported["artifacts"]["program_catalogue.json"], repo, label="catalogue"
        ).read_text()
    )
    preparation = {
        row["target_id"]: row
        for row in _read(
            resolve_pin(
                protection["artifacts"]["preparation_view.jsonl.gz"], repo, label="preparation"
            )
        )
    }
    ledgers = []
    if not isinstance(config["program_results"], list) or not config["program_results"]:
        raise ComposeLipidError("precursor audit requires explicit program evidence")
    for pin in config["program_results"]:
        receipt = json.loads(resolve_pin(pin, repo, label="program result").read_text())
        if receipt.get("schema_version") not in LEDGER_FORMATS:
            raise ComposeLipidError("unsupported program evidence format")
        policy = receipt.get("policy", {})
        if policy.get("training_calls") != 0 or policy.get("source_flags_changed") is not False:
            raise ComposeLipidError("program evidence exceeds preparation scope")
        resolve_pin(receipt["config"], repo, label="program config")
        for section in ("inputs", "implementation", "artifacts"):
            for name, record in receipt.get(section, {}).items():
                resolve_pin(record, repo, label=f"program {section}/{name}")
        name, marker = LEDGER_FORMATS[receipt["schema_version"]]
        ledgers.append(
            (resolve_pin(receipt["artifacts"][name], repo, label="program ledger"), marker)
        )
    evidence = (
        {**row, "exact_program": row.get(marker)} for path, marker in ledgers for row in _read(path)
    )
    db_path = resolve_pin(imported["artifacts"]["corpus.sqlite"], repo, label="corpus")
    with sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True) as db:
        # Select metadata explicitly: provider evaluation molecular graphs never enter Python.
        for target, metadata in db.execute(
            "SELECT a.target_id,json_extract(t.payload,'$.primary_metadata') FROM assignments a JOIN targets t USING(target_id) WHERE a.forge_split='train'"
        ):
            preparation[target]["primary_metadata"] = json.loads(metadata)
        evaluation = (
            {
                "target_id": target,
                "family": family,
                "split": split,
                "primary_metadata": json.loads(metadata),
                "test_panels": json.loads(panels),
            }
            for target, family, split, metadata, panels in db.execute(
                "SELECT a.target_id,a.family,a.forge_split,json_extract(t.payload,'$.primary_metadata'),json_extract(a.payload,'$.test_panels') FROM assignments a JOIN targets t USING(target_id) WHERE a.forge_split IN ('calibration','heldout') ORDER BY a.target_id"
            )
        )
        return audit_precursors(preparation, evidence, evaluation, catalogue, guards)


def build_precursor_audit(repo_root: Path, config_path: Path, output_dir: Path) -> dict:
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if output.exists() or not output.is_relative_to(repo) or not config_path.is_relative_to(repo):
        raise ComposeLipidError("precursor audit output must be fresh and inside the repository")
    config = json.loads(config_path.read_text())
    evaluated = _evaluate(repo, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".precursors-", dir=output.parent) as temporary:
        stage = Path(temporary)
        artifacts = {}
        for key in ("rows", "identities", "panels"):
            path = stage / f"{key}.jsonl.gz"
            with path.open("wb") as raw:
                with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as stream:
                    for row in evaluated[key]:
                        stream.write((_json(row) + "\n").encode())
            artifacts[path.name] = {
                "path": (output / path.name).relative_to(repo).as_posix(),
                "sha256": str(sha256_file(path)),
            }
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "partial_precursor_identity_audit_training_unqualified",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": _pin(repo, config_path),
            "inputs": {
                "protection_result": config["protection_result"],
                "program_results": config["program_results"],
            },
            "implementation": {name: _pin(repo, repo / name) for name in IMPLEMENTATION},
            "policy": POLICY,
            "summary": evaluated["summary"],
            "artifacts": artifacts,
        }
        (stage / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        os.rename(stage, output)
    return result


def verify_precursor_audit(repo_root: Path, result_path: Path) -> dict:
    repo = repo_root.resolve()
    result = json.loads((repo / result_path).read_text())
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("status") != "partial_precursor_identity_audit_training_unqualified"
        or result.get("policy") != POLICY
        or set(result.get("implementation", {})) != set(IMPLEMENTATION)
        or set(result.get("artifacts", {}))
        != {f"{key}.jsonl.gz" for key in ("rows", "identities", "panels")}
    ):
        raise ComposeLipidError("precursor audit receipt scope changed")
    for name, pin in result["implementation"].items():
        if resolve_pin(pin, repo, label=name) != (repo / name).resolve():
            raise ComposeLipidError("precursor audit implementation substitution")
    config = json.loads(resolve_pin(result["config"], repo, label="config").read_text())
    if result["inputs"] != {
        "protection_result": config["protection_result"],
        "program_results": config["program_results"],
    }:
        raise ComposeLipidError("precursor audit input substitution")
    expected = _evaluate(repo, config)
    for key in ("rows", "identities", "panels"):
        path = resolve_pin(result["artifacts"][f"{key}.jsonl.gz"], repo, label=key)
        if list(_read(path)) != expected[key]:
            raise ComposeLipidError(f"precursor audit {key} do not reproduce")
    if result["summary"] != expected["summary"]:
        raise ComposeLipidError("precursor audit summary does not reproduce")
    return result
