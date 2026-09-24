"""Resolve TRAIN-only component constraints and persist a non-admitted precursor view.

This audit consumes the frozen pre-training audit and unchanged registry transforms. Shared
source labels can disambiguate related-transform candidates conditionally, but cannot supply
missing source program definitions, resolve forward selectivity, or grant training admission.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rdkit import rdBase

from forge.assembly.component_constraints import ComponentConstraint, propagate_component_labels
from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.families import LibraryAssemblyError, RegistryAssemblyAdapter
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid import _frozen_guards
from forge.corpus.compose_lipid_pretraining import (
    _dump,
    _pin,
    _rows,
    component_labels,
    verify_pretraining_checks,
)

CONFIG_SCHEMA = "forge.compose_lipid_component_constraints_config.v1"
RESULT_SCHEMA = "forge.compose_lipid_component_constraints.v1"
IMPLEMENTATION = (
    "forge/assembly/component_constraints.py",
    "forge/corpus/compose_lipid_components.py",
    "experiments/phase1/multireaction/compose_lipid_components.py",
)
REPROBE = {"ambiguous_related_transform", "ambiguous_forward_site_class"}
EXACT = {"exact_related_transform", "exact_related_transform_protected_precursor"}


def _write_rows(path: Path, rows) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
            for row in rows:
                stream.write(
                    (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                )


def _read_rows(path: Path):
    with gzip.open(path, "rt") as stream:
        yield from map(json.loads, stream)


def enumerate_candidates(row: dict, program: dict, binding: dict, adapter) -> dict:
    """Keep ambiguous forward candidates in the domain: target membership is not selectivity."""
    labels, complete = component_labels(row, program)
    if not complete:
        raise ComposeLipidError("reprobe requires complete TRAIN component labels")
    try:
        decompositions = adapter.decompose(row["constitution"], maximum_outcomes=binding["limit"])
        candidates = []
        for decomposition in decompositions:
            parts = dict(decomposition.components)
            forward = adapter.forward_products(parts, maximum_outcomes=binding["limit"])
            if forward.saturated:
                return {"status": "bounded_search_abstention", "candidates": []}
            if row["constitution"] not in forward.products:
                raise ComposeLipidError("inverse candidate lost exact forward membership")
            candidates.append(
                {
                    "components": {
                        binding["registry_to_source_roles"][role]: smiles
                        for role, smiles in parts.items()
                    },
                    "forward_unique": forward.products == (row["constitution"],),
                }
            )
    except LibraryAssemblyError as exc:
        return {"status": "bounded_search_abstention", "reason": str(exc), "candidates": []}
    return {"status": "enumerated", "candidates": candidates}


def classify_components(records: list[dict], guards) -> tuple[list[dict], dict]:
    """Produce only exclusions/holds, including all known protected-component replays."""
    constraints = []
    for record in records:
        if not record["candidates"]:
            continue
        roles = sorted(record["labels"])
        if any(set(c["components"]) != set(roles) for c in record["candidates"]):
            raise ComposeLipidError("candidate component roles do not match source labels")
        constraints.append(
            ComponentConstraint(
                record["target_id"],
                tuple((record["family"], role, record["labels"][role]) for role in roles),
                tuple(tuple(c["components"][role] for role in roles) for c in record["candidates"]),
            )
        )
    propagated = propagate_component_labels(constraints)
    conflicts = set(propagated.conflicting_targets)
    witnessed = set(propagated.singleton_component_witness_targets)
    output = []
    families: dict[str, Counter] = {}
    for record in records:
        key, family = record["target_id"], record["family"]
        surviving = propagated.surviving_candidates.get(key, ())
        roles = sorted(record["labels"])
        reason = "unresolved_program_or_components"
        components = []
        if key in conflicts:
            reason = "source_label_structure_conflict"
        elif key in witnessed:
            values = surviving[0]
            chosen = next(
                c
                for c in record["candidates"]
                if tuple(c["components"][role] for role in roles) == values
            )
            for role, smiles in zip(roles, values, strict=True):
                identity = hashlib.sha256(smiles.encode()).hexdigest()
                components.append(
                    {
                        "role": role,
                        "canonical_smiles": smiles,
                        "constitution_id": identity,
                        "historical_fold": guards.effective(identity),
                    }
                )
            if any(c["historical_fold"] in ("calibration", "heldout") for c in components):
                reason = "protected_precursor"
            elif not chosen["forward_unique"]:
                reason = "ambiguous_forward_site_class"
            else:
                reason = "source_program_qualification_missing"
        elif record["candidates"]:
            reason = "component_assignment_unresolved"
        # Direct historical exclusions remain exclusions even if a source-label contradiction
        # later prevents use of their component assignment as a consistency witness.
        known_protected = record["original_status"] == "exact_related_transform_protected_precursor"
        excluded = known_protected or reason == "protected_precursor"
        result = {
            "target_id": key,
            "family": family,
            "original_status": record["original_status"],
            "initial_candidates": len(record["candidates"]),
            "remaining_candidates": len(surviving),
            "singleton_component_witness": key in witnessed,
            "reason": reason,
            "disposition": "quarantine_protected_precursor" if excluded else "hold_unqualified",
            "known_protected_in_prior_audit": known_protected,
            "components": components,
            "training_admitted": False,
            "training_weight": 0.0,
            "source_program_equivalence_qualified": False,
        }
        output.append(result)
        counts = families.setdefault(family, Counter())
        counts["rows"] += 1
        counts[reason] += 1
        counts[result["disposition"]] += 1
        counts["candidate_reduction"] += int(
            len(surviving) < len(record["candidates"]) and key not in conflicts
        )
        counts["new_singleton_witnesses"] += int(len(record["candidates"]) > 1 and key in witnessed)
    return output, {family: dict(counts) for family, counts in sorted(families.items())}


def _inputs(repo: Path, config: dict) -> tuple[dict, dict, dict, object]:
    if (
        set(config) != {"schema_version", "pretraining_audit", "policy"}
        or config["schema_version"] != CONFIG_SCHEMA
    ):
        raise ComposeLipidError("unsupported component constraint configuration")
    if config["policy"] != {
        "fit_split": "train",
        "label_scope": "family_and_role",
        "training_calls": 0,
        "generation_calls": 0,
        "heldout_structure_access": False,
        "seed": 0,
    }:
        raise ComposeLipidError(
            "component constraint policy may not grant training or held-out access"
        )
    audit = verify_pretraining_checks(
        repo, resolve_pin(config["pretraining_audit"], repo, label="pretraining audit")
    )
    prior_config = json.loads((repo / audit["config"]["path"]).read_text())
    imported = json.loads((repo / audit["import_result"]["path"]).read_text())
    import_config = json.loads((repo / imported["config"]["path"]).read_text())
    paths = {name: repo / pin["path"] for name, pin in imported["inputs"].items()}
    guards = _frozen_guards(import_config["historical_identity_guards"], paths)
    return audit, prior_config, imported, guards


def run_component_constraints(repo_root: Path, config_path: Path, output_dir: Path) -> dict:
    repo, started = repo_root.resolve(), time.monotonic()
    config_path = (repo / config_path).resolve()
    config = json.loads(config_path.read_text())
    audit, prior_config, imported, guards = _inputs(repo, config)
    output = (repo / output_dir).resolve()
    output.relative_to(repo)
    if output.exists():
        raise ComposeLipidError("component output already exists; choose a fresh version")
    catalogue = json.loads(
        (repo / imported["artifacts"]["program_catalogue.json"]["path"]).read_text()
    )
    db_path = repo / imported["artifacts"]["corpus.sqlite"]["path"]
    db = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
    adapters, records = {}, []
    try:
        old = _read_rows(repo / audit["artifacts"]["train_checks.jsonl.gz"]["path"])
        for row, saved in zip(_rows(db, "train"), old, strict=True):
            family = row["primary_family"]
            if (row["target_id"], family) != (saved["target_id"], saved["family"]):
                raise ComposeLipidError("pretraining ledger and TRAIN source rows differ")
            probe = saved["related_transform_probe"]
            labels, _ = component_labels(row, catalogue[family])
            record = {
                "target_id": row["target_id"],
                "family": family,
                "original_status": probe["status"],
                "labels": labels,
                "candidates": [],
            }
            if probe["status"] in EXACT:
                record["candidates"] = [
                    {
                        "components": {
                            c["role"]: c["canonical_smiles"] for c in probe["components"]
                        },
                        "forward_unique": True,
                    }
                ]
            elif probe["status"] in REPROBE:
                binding = prior_config["related_transforms"][family]
                if family not in adapters:
                    pin = imported["inputs"][binding["registry_input"]]
                    adapters[family] = RegistryAssemblyAdapter.from_registry(
                        repo / pin["path"],
                        reaction_id=binding["reaction_id"],
                        expected_sha256=pin["sha256"],
                    )
                candidate_result = enumerate_candidates(
                    row, catalogue[family], binding, adapters[family]
                )
                record.update(candidate_result)
            records.append(record)
            if len(records) % 25000 == 0:
                print(f"Component constraints: {len(records)} TRAIN records", flush=True)
    finally:
        db.close()
    if len(records) != audit["train_rows"]:
        raise ComposeLipidError("component population differs from pinned TRAIN audit")
    view, families = classify_components(records, guards)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".components-", dir=output.parent) as temporary:
        work = Path(temporary) / "output"
        work.mkdir()
        _write_rows(work / "candidate_constraints.jsonl.gz", records)
        _write_rows(work / "admission_view.jsonl.gz", view)
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "component_constraints_complete_training_unqualified",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": _pin(repo, config_path),
            "pretraining_audit": config["pretraining_audit"],
            "implementation": {p: _pin(repo, repo / p) for p in IMPLEMENTATION},
            "artifacts": {
                p.name: {
                    "path": (output / p.name).relative_to(repo).as_posix(),
                    "sha256": str(sha256_file(p)),
                }
                for p in sorted(work.iterdir())
            },
            "train_rows": len(records),
            "by_family": families,
            "training_calls": 0,
            "generation_calls": 0,
            "training_admitted_rows": 0,
            "source_programs_qualified": 0,
            "heldout_graphs_parsed": False,
            "source_assignments_modified": False,
            "random_sampling_used": False,
            "seed": 0,
            "evidence_basis": "computed_transform_consistency_conditional_on_source_label_identity",
            "chemical_decomposition_precision": "not_established_no_independent_component_site_truth",
            "limitations": [
                "Shared labels are an assumption, not independent source structure evidence.",
                "Only all-singleton connected components have a demonstrated global assignment.",
                "Forward site ambiguity and missing source program definitions still block training.",
                "Unresolved decompositions can conceal additional protected precursors.",
            ],
            "rdkit_version": rdBase.rdkitVersion,
            "duration_seconds": time.monotonic() - started,
        }
        _dump(work / "result.json", result)
        os.rename(work, output)
    return result


def verify_component_constraints(repo_root: Path, result_path: Path) -> dict:
    repo = repo_root.resolve()
    result = json.loads((repo / result_path).read_text())
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("status") != "component_constraints_complete_training_unqualified"
    ):
        raise ComposeLipidError("unsupported component constraint receipt")
    config = json.loads(resolve_pin(result["config"], repo, label="config").read_text())
    if config["pretraining_audit"] != result["pretraining_audit"]:
        raise ComposeLipidError("pretraining receipt substitution")
    if set(result["implementation"]) != set(IMPLEMENTATION) or set(result["artifacts"]) != {
        "candidate_constraints.jsonl.gz",
        "admission_view.jsonl.gz",
    }:
        raise ComposeLipidError("incomplete component provenance")
    for group in ("implementation", "artifacts"):
        for name, pin in result[group].items():
            path = resolve_pin(pin, repo, label=name)
            if group == "implementation" and path != repo / name:
                raise ComposeLipidError("implementation source substitution")
    audit, _, _, guards = _inputs(repo, config)
    records = list(_read_rows(repo / result["artifacts"]["candidate_constraints.jsonl.gz"]["path"]))
    prior = _read_rows(repo / audit["artifacts"]["train_checks.jsonl.gz"]["path"])
    for record, old in zip(records, prior, strict=True):
        if (record["target_id"], record["family"], record["original_status"]) != (
            old["target_id"],
            old["family"],
            old["related_transform_probe"]["status"],
        ):
            raise ComposeLipidError("constraint population substitution")
    expected_view, families = classify_components(records, guards)
    actual = list(_read_rows(repo / result["artifacts"]["admission_view.jsonl.gz"]["path"]))
    if (
        expected_view != actual
        or families != result["by_family"]
        or len(records) != result["train_rows"]
        or len(records) != audit["train_rows"]
    ):
        raise ComposeLipidError("component consistency recomputation mismatch")
    for field in (
        "training_calls",
        "generation_calls",
        "training_admitted_rows",
        "source_programs_qualified",
        "seed",
    ):
        if type(result[field]) is not int or result[field] != 0:
            raise ComposeLipidError("component audit cannot claim training or source qualification")
    for field in ("heldout_graphs_parsed", "source_assignments_modified", "random_sampling_used"):
        if result[field] is not False:
            raise ComposeLipidError("component audit scope changed")
    return result
