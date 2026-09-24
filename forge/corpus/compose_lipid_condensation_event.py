"""Qualify source-declared single-event condensations with registry-owned atom loss.

Uses the existing source-event receipt schema, controls, protection and exclusions.
The explicit implementation pins distinguish the condensation verifier from the
atom-conserving verifier, whose frozen behavior and historical receipts stay intact.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.condensation_event import check_condensation_event
from forge.assembly.families import LibraryAssemblyError
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus import compose_lipid_source_event as base
from forge.corpus.compose_lipid_source_event import (
    POLICY,
    RESULT_SCHEMA,
    _dump,
    _pin,
    _summary,
    exclude_component_conflicts,
)

IMPLEMENTATION = base.IMPLEMENTATION + (
    "forge/assembly/condensation_event.py",
    "forge/corpus/compose_lipid_condensation_event.py",
)


def _load(repo: Path, config_path: Path):
    loaded = base._load(repo, config_path)
    config, adjudication, *_ = loaded
    path = resolve_pin(config["inputs"]["registry"], repo, label="registry")
    registry = json.loads(path.read_text())
    reaction = next(r for r in registry["reactions"] if r["reaction_id"] == config["reaction_id"])
    byproducts = reaction.get("net_byproducts")
    if not byproducts or byproducts != adjudication["source_contract"].get("net_byproducts"):
        raise ComposeLipidError("condensation source and registry byproduct inventories differ")
    for name, pin in registry["source_assets"].items():
        resolve_pin(pin, repo, label=name)
        if name in adjudication["assets"] and pin != adjudication["assets"][name]:
            raise ComposeLipidError("condensation registry and adjudication source assets differ")
    for name, pin in reaction["implementation"]["parent_registries"].items():
        resolve_pin(pin, repo, label=name)
    resolve_pin(reaction["implementation"]["derivation"], repo, label="registry derivation")
    loaded[-1]["net_byproducts"] = byproducts
    return loaded


def _evaluate(loaded):
    config, adjudication, adapter, reader, guards, order, kwargs = loaded
    controls = {}
    for control in adjudication["source_controls"]:
        label = control["label"]
        if label in controls or control["source_asset"] not in adjudication["assets"]:
            raise ComposeLipidError("source control is duplicated or lacks a primary asset")
        result = check_condensation_event(
            adapter, control["components"], control["expected_product"], **kwargs
        )
        formula = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(control["expected_product"]))
        if not result["computed_consistency_pass"] or formula != control["expected_formula"]:
            raise ComposeLipidError(f"independent source control failed: {label}")
        controls[label] = {"formula": formula, **result}
    if not controls:
        raise ComposeLipidError("independent primary-source controls are required")
    rows = []
    seen = set()
    mapping = adjudication["source_contract"]["registry_to_source_roles"]
    for item in reader.iter_preparation_records(family=config["family"]):
        source, preparation = item["source"], item["preparation"]
        if source["target_id"] in seen or not preparation["eligible_for_program_preparation"]:
            raise ComposeLipidError(
                "duplicate or excluded preparation record reached source replay"
            )
        seen.add(source["target_id"])
        labels = source["primary_metadata"].get("precursor_ids", [])
        if len(labels) != len(order) or any(
            not isinstance(label, str) or not label for label in labels
        ):
            raise ComposeLipidError("complete source precursor labels are required")
        labels = dict(zip(order, labels, strict=True))
        row = {
            "target_id": source["target_id"],
            "constitution_id": preparation["constitution_id"],
            "family": config["family"],
            "source_metadata": source["primary_metadata"],
            "components": [],
            "computed_consistency_pass": False,
            "historical_protected_precursor": False,
            "training_admitted": False,
            "source_bank_assignment_admitted": False,
            "experimental_execution_admitted": False,
        }
        try:
            inverse = adapter.decompose(
                source["constitution"], maximum_outcomes=config["maximum_outcomes"]
            )
            row["inverse_candidate_count"] = len(inverse)
            if len(inverse) == 1:
                components = dict(inverse[0].components)
                for role, smiles in sorted(components.items()):
                    digest = hashlib.sha256(smiles.encode()).hexdigest()
                    row["components"].append(
                        {
                            "role": mapping[role],
                            "source_id": labels[mapping[role]],
                            "canonical_smiles": smiles,
                            "constitution_id": digest,
                            "historical_fold": guards.effective(digest),
                        }
                    )
                row["historical_protected_precursor"] = any(
                    c["historical_fold"] in ("calibration", "heldout") for c in row["components"]
                )
                row.update(
                    check_condensation_event(adapter, components, source["constitution"], **kwargs)
                )
                row["status"] = (
                    "computed_consistent"
                    if row["computed_consistency_pass"]
                    else "excluded_source_or_replay_failure"
                )
            else:
                row["status"] = (
                    "excluded_ambiguous_inverse" if inverse else "excluded_no_exact_inverse"
                )
        except LibraryAssemblyError as exc:
            row["status"] = "excluded_bounded_replay_failure"
            row["error"] = str(exc)
        rows.append(row)
    expected = reader.result["summary"]["by_family"][config["family"]][
        "eligible_for_program_preparation"
    ]
    if len(rows) != expected:
        raise ComposeLipidError("source-event ledger lost preparation rows")
    exclude_component_conflicts(rows)
    return controls, rows


def run_condensation_event(repo_root: Path, config_path: Path, output_dir: Path) -> dict:
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if output.exists() or not output.is_relative_to(repo) or not config_path.is_relative_to(repo):
        raise ComposeLipidError("source-event output must be fresh and inside the repository")
    loaded = _load(repo, config_path)
    controls, rows = _evaluate(loaded)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".source-event-", dir=output.parent) as temporary:
        stage = Path(temporary)
        ledger = stage / "programs.jsonl.gz"
        with ledger.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as stream:
                for row in rows:
                    stream.write(
                        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                    )
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "source_event_evaluated_training_unqualified",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": _pin(repo, config_path),
            "inputs": loaded[0]["inputs"],
            "implementation": {name: _pin(repo, repo / name) for name in IMPLEMENTATION},
            "policy": POLICY,
            "source_controls": controls,
            "summary": _summary(rows, controls),
            "artifacts": {
                ledger.name: {
                    "path": (output / ledger.name).relative_to(repo).as_posix(),
                    "sha256": str(sha256_file(ledger)),
                }
            },
        }
        _dump(stage / "result.json", result)
        os.rename(stage, output)
    return result


def verify_condensation_event(repo_root: Path, result_path: Path) -> dict:
    repo = repo_root.resolve()
    result = json.loads((repo / result_path).read_text())
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("status") != "source_event_evaluated_training_unqualified"
        or result.get("policy") != POLICY
        or set(result.get("implementation", {})) != set(IMPLEMENTATION)
        or set(result.get("artifacts", {})) != {"programs.jsonl.gz"}
    ):
        raise ComposeLipidError("source-event receipt scope or provenance changed")
    for name, pin in result["implementation"].items():
        if resolve_pin(pin, repo, label=name) != (repo / name).resolve():
            raise ComposeLipidError("source-event implementation path substitution")
    ledger = resolve_pin(result["artifacts"]["programs.jsonl.gz"], repo, label="programs")
    loaded = _load(repo, resolve_pin(result["config"], repo, label="config"))
    controls, expected = _evaluate(loaded)
    with gzip.open(ledger, "rt") as stream:
        actual = [json.loads(line) for line in stream]
    if (
        result["inputs"] != loaded[0]["inputs"]
        or actual != expected
        or result["summary"] != _summary(expected, controls)
        or result["source_controls"] != controls
    ):
        raise ComposeLipidError("source-event receipt differs from independently replayed sources")
    return result
