"""Replay source-adjudicated repeated programs on protected COMPOSE TRAIN only."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.families import RegistryAssemblyAdapter, constitutional_molecule
from forge.assembly.repeated_components import RepeatBounds, replay_repeated_components
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid import _frozen_guards, verify_compose_lipid

CONFIG_SCHEMA = "forge.compose_lipid_repeated_source_config.v1"
RESULT_SCHEMA = "forge.compose_lipid_repeated_source.v1"
IMPLEMENTATION = (
    "forge/assembly/repeated_components.py",
    "forge/assembly/families.py",
    "forge/assembly/registry.py",
    "forge/assembly/program.py",
    "forge/chemistry/reactive_sites.py",
    "forge/corpus/compose_lipid_repeated_source.py",
    "forge/corpus/compose_lipid.py",
    "forge/core/hashing.py",
)
POLICY = {
    "training_calls": 0,
    "generation_calls": 0,
    "heldout_graphs_parsed": False,
    "source_flags_changed": False,
    "fit_population": "protected_train",
    "seed": 0,
}
HOLDS = [
    "source_architecture_conflict",
    "component_dictionary_incomplete",
    "unconditioned_inverse_uniqueness_not_established",
    "global_precursor_protection_incomplete",
    "precursor_holdouts_unqualified",
    "remaining_families_unqualified",
    "repository_full_tests_fail",
]


def _dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _pin(repo: Path, path: Path) -> dict:
    return {"path": path.relative_to(repo).as_posix(), "sha256": str(sha256_file(path))}


def _read_rows(path: Path):
    with gzip.open(path, "rt") as stream:
        for line in stream:
            yield json.loads(line)


def candidate_dictionary(path: Path, requested: set[str]) -> dict:
    """Opaque IDs are lookup keys, never presumed to be SMILES hashes.

    Parse molecular graphs only for the IDs referenced by the already-protected TRAIN.
    Duplicate source IDs are errors even if their structures happen to agree.
    """
    found = {}
    seen = set()
    for row in _read_rows(path):
        identity = row["precursor_id"]
        if identity in seen:
            raise ComposeLipidError("duplicate precursor ID in the pinned source dictionary")
        seen.add(identity)
        if identity in requested:
            canonical, _ = constitutional_molecule(row["constitution"])
            found[identity] = {
                "source_id": identity,
                "canonical_smiles": canonical,
                "constitution_id": hashlib.sha256(canonical.encode()).hexdigest(),
                "source_families": sorted(row["families"]),
                "source_roles": sorted(row["roles"]),
            }
    return found


def _load(repo: Path, config_path: Path):
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("policy") != POLICY:
        raise ComposeLipidError("repeated-source schema or scope changed")
    paths = {name: resolve_pin(pin, repo, label=name) for name, pin in config["inputs"].items()}
    imported = verify_compose_lipid(repo, paths["import_result"])
    import_config = json.loads((repo / imported["config"]["path"]).read_text())
    import_paths = {name: repo / pin["path"] for name, pin in imported["inputs"].items()}
    key = json.loads(import_paths["decomposition_key"].read_text())[config["family"]]
    if (
        key["variable"] != [config["event_count_field"]]
        or any(count != 1 for count in key["roles"].values())
        or key["field_to_role"] != {field: [role] for role, field in config["role_fields"].items()}
        or config["architecture_qualified"] is not False
        or config["training_ready"] is not False
    ):
        raise ComposeLipidError("repeated-source role, event or qualification contract changed")
    adapter = RegistryAssemblyAdapter.from_registry(
        paths["registry"],
        reaction_id=config["reaction_id"],
        expected_sha256=config["inputs"]["registry"]["sha256"],
    )
    if set(adapter.roles) != set(key["roles"]) or config["accumulator_role"] not in adapter.roles:
        raise ComposeLipidError("registry roles differ from source program")
    registry = json.loads(paths["registry"].read_text())
    for name, pin in registry["source_assets"].items():
        resolve_pin(pin, repo, label=name)
    adjudication = json.loads(paths["adjudication"].read_text())
    if adjudication["registry"] != config["inputs"]["registry"]:
        raise ComposeLipidError("source adjudication attributes another registry")
    for name, pin in adjudication["assets"].items():
        resolve_pin(pin, repo, label=name)
    if not adjudication["conflicts"]:
        raise ComposeLipidError("the declared source conflicts cannot be silently removed")
    guards = _frozen_guards(import_config["historical_identity_guards"], import_paths)
    return config, paths, imported, adapter, registry, adjudication, guards


def _replay(adapter, components, target, events, config, registry):
    return replay_repeated_components(
        adapter,
        components,
        target,
        accumulator_role=config["accumulator_role"],
        events=events,
        byproducts_per_event=registry["curation"]["byproducts_per_event"],
        bounds=RepeatBounds(**config["search_bounds"]),
    )


def _controls(adapter, config, registry, adjudication):
    output = {}
    for control in adjudication["source_controls"]:
        label = control["label"]
        if label in output:
            raise ComposeLipidError("duplicated source control")
        result = _replay(
            adapter,
            control["components"],
            control["expected_product"],
            control["events"],
            config,
            registry,
        )
        formula = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(control["expected_product"]))
        if not result["computed_consistency_pass"] or formula != control["expected_formula"]:
            raise ComposeLipidError(f"source control failed: {label}")
        output[label] = {"formula": formula, **result}
    if not output:
        raise ComposeLipidError("independent source controls are required")
    return output


def _evaluate(repo, loaded):
    config, paths, imported, adapter, registry, adjudication, guards = loaded
    controls = _controls(adapter, config, registry, adjudication)
    db_path = repo / imported["artifacts"]["corpus.sqlite"]["path"]
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
        rows = [
            json.loads(row[0])
            for row in db.execute(
                "SELECT t.payload FROM assignments a JOIN targets t USING(target_id) "
                "WHERE a.forge_split='train' AND a.family=? ORDER BY a.target_id",
                (config["family"],),
            )
        ]
    requested = {
        row["primary_metadata"][field] for row in rows for field in config["role_fields"].values()
    }
    dictionary = candidate_dictionary(paths["precursor_dictionary"], requested)
    target_ids = {row["target_id"] for row in rows}
    prior_folds = {key: set() for key in target_ids}
    for record in _read_rows(paths["prior_construction"]):
        if record["target_id"] in prior_folds:
            prior_folds[record["target_id"]].add(record.get("prior_development_split"))
    results = []
    for row in rows:
        metadata = row["primary_metadata"]
        components = []
        missing = []
        for role, field in config["role_fields"].items():
            candidate = dictionary.get(metadata[field])
            if candidate is None or config["family"] not in candidate["source_families"]:
                missing.append(role)
                continue
            components.append(
                {
                    "role": role,
                    **candidate,
                    "historical_fold": guards.effective(candidate["constitution_id"]),
                }
            )
        result = {
            "target_id": row["target_id"],
            "family": config["family"],
            "declared_events": metadata[config["event_count_field"]],
            "declared_site_disposition": metadata.get("site_disposition"),
            "candidate_components": components,
            "missing_roles": sorted(missing),
            "historical_protected_precursor": any(
                c["historical_fold"] in ("calibration", "heldout") for c in components
            ),
            "prior_v5_product_folds": sorted(
                f for f in prior_folds[row["target_id"]] if f is not None
            ),
            "source_training_admissible": row["training_admissible"],
            "training_admitted": False,
            "architecture_qualified": False,
            "computed_consistency_pass": False,
            "status": "missing_candidate_component",
        }
        if not missing:
            replay = _replay(
                adapter,
                {c["role"]: c["canonical_smiles"] for c in components},
                row["constitution"],
                result["declared_events"],
                config,
                registry,
            )
            result.update(replay)
            result["status"] = (
                "computed_consistency_pass"
                if replay["computed_consistency_pass"]
                else "reconstruction_or_site_hold"
            )
        results.append(result)
    return controls, results


def summarize(rows: list[dict]) -> dict:
    counts = Counter()
    events = {}
    failed_checks = Counter()
    prior = Counter()
    for row in rows:
        counts["rows"] += 1
        counts[row["status"]] += 1
        event = str(row["declared_events"])
        events.setdefault(event, Counter())[row["status"]] += 1
        counts["historical_protected_precursor"] += row["historical_protected_precursor"]
        if row["computed_consistency_pass"]:
            counts["consistent_with_protected_precursor"] += row["historical_protected_precursor"]
        for check, passed in row.get("checks", {}).items():
            if not passed:
                failed_checks[check] += 1
        prior.update(row["prior_v5_product_folds"])
    return {
        "counts": dict(sorted(counts.items())),
        "by_declared_event_count": {k: dict(sorted(v.items())) for k, v in sorted(events.items())},
        "failed_checks": dict(sorted(failed_checks.items())),
        "prior_v5_product_fold_overlaps": dict(sorted(prior.items())),
        "architecture_qualified": False,
        "training_admitted": 0,
        "inverse_scope": "conditioned_on_declared_fixed_side_components",
        "precision_scope": "exact computational replay among ID-resolved tuples; chemical precision requires independent source adjudication",
    }


def run_repeated_source(repo_root: Path, config_path: Path, output_dir: Path) -> dict:
    repo = repo_root.resolve()
    config_path = (repo / config_path).resolve()
    output = (repo / output_dir).resolve()
    if output.exists():
        raise ComposeLipidError(f"refusing to overwrite a diagnostic result: {output}")
    loaded = _load(repo, config_path)
    controls, rows = _evaluate(repo, loaded)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".repeated-source-", dir=output.parent) as temporary:
        stage = Path(temporary)
        ledger = stage / "train_replay.jsonl.gz"
        with ledger.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
                for row in rows:
                    stream.write(
                        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                    )
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "diagnostic_complete_architecture_and_training_unqualified",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": _pin(repo, config_path),
            "inputs": loaded[0]["inputs"],
            "implementation": {name: _pin(repo, repo / name) for name in IMPLEMENTATION},
            "runtime": {"rdkit": rdBase.rdkitVersion},
            "policy": POLICY,
            "summary": summarize(rows),
            "source_controls": controls,
            "source_conflicts": loaded[5]["conflicts"],
            "remaining_holds": HOLDS,
            "training_ready": False,
            "artifacts": {
                "train_replay.jsonl.gz": {
                    "path": (output / ledger.name).relative_to(repo).as_posix(),
                    "sha256": str(sha256_file(ledger)),
                }
            },
        }
        _dump(stage / "result.json", result)
        os.rename(stage, output)
    return result


def verify_repeated_source(repo_root: Path, result_path: Path) -> dict:
    """Authenticate all inputs and independently recompute every TRAIN replay."""
    repo = repo_root.resolve()
    result = json.loads((repo / result_path).read_text())
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("training_ready") is not False
        or result.get("policy") != POLICY
        or result.get("remaining_holds") != HOLDS
        or result.get("status") != "diagnostic_complete_architecture_and_training_unqualified"
        or set(result.get("implementation", {})) != set(IMPLEMENTATION)
        or set(result.get("artifacts", {})) != {"train_replay.jsonl.gz"}
    ):
        raise ComposeLipidError("repeated-source result schema, scope or provenance changed")
    for group in ("implementation", "artifacts"):
        for name, pin in result[group].items():
            path = resolve_pin(pin, repo, label=name)
            if group == "implementation" and path != (repo / name).resolve():
                raise ComposeLipidError("implementation pin path substitution")
    config_path = resolve_pin(result["config"], repo, label="config")
    loaded = _load(repo, config_path)
    if (
        result["inputs"] != loaded[0]["inputs"]
        or result["source_conflicts"] != loaded[5]["conflicts"]
    ):
        raise ComposeLipidError("repeated-source inputs or source conflict changed")
    controls, rows = _evaluate(repo, loaded)
    ledger = repo / result["artifacts"]["train_replay.jsonl.gz"]["path"]
    if (
        list(_read_rows(ledger)) != rows
        or result["source_controls"] != controls
        or result["summary"] != summarize(rows)
        or result["runtime"] != {"rdkit": rdBase.rdkitVersion}
    ):
        raise ComposeLipidError("repeated-source replay, runtime or summary changed")
    return result
