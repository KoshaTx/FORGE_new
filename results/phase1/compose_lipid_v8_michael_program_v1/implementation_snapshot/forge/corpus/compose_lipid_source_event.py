"""Qualify a source-declared single event on the enforced preparation view.

Exact reconstruction admits a computed program witness, never a training row or an
experimental label. Ambiguities and conflicting component labels remain exclusions.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.families import LibraryAssemblyError, RegistryAssemblyAdapter
from forge.assembly.source_event import check_source_event
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid import _frozen_guards
from forge.corpus.compose_lipid_protection import ProtectedPreparationCorpus

CONFIG_SCHEMA = "forge.compose_lipid_source_event_config.v1"
RESULT_SCHEMA = "forge.compose_lipid_source_event.v1"
POLICY = {
    "training_calls": 0,
    "generation_calls": 0,
    "provider_heldout_graphs_parsed": False,
    "source_flags_changed": False,
    "seed": 0,
    "fit_population": "enforced_product_protection_preparation_view",
}
IMPLEMENTATION = (
    "forge/assembly/source_event.py",
    "forge/assembly/repeated_components.py",
    "forge/assembly/families.py",
    "forge/assembly/registry.py",
    "forge/assembly/program.py",
    "forge/chemistry/reactive_sites.py",
    "forge/corpus/compose_lipid_source_event.py",
    "forge/corpus/compose_lipid_protection.py",
    "forge/corpus/compose_lipid.py",
    "forge/corpus/library_splits.py",
    "forge/core/hashing.py",
)
HOLDS = [
    "global_precursor_protection_incomplete",
    "precursor_holdouts_unqualified",
    "source_reagent_bank_and_architecture_assignment_unqualified",
    "remaining_families_unqualified",
    "final_balanced_train_dataset_unbuilt",
    "repository_full_tests_fail",
]


def _pin(repo: Path, path: Path) -> dict:
    return {"path": path.relative_to(repo).as_posix(), "sha256": str(sha256_file(path))}


def _dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _load(repo: Path, config_path: Path):
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("policy") != POLICY:
        raise ComposeLipidError("source-event configuration or scientific scope changed")
    paths = {name: resolve_pin(pin, repo, label=name) for name, pin in config["inputs"].items()}
    if set(paths) != {
        "protection_result",
        "registry",
        "functional_group_registry",
        "adjudication",
        "decomposition_key",
    }:
        raise ComposeLipidError("source-event input set changed")
    bound = config["maximum_outcomes"]
    if type(bound) is not int or bound < 2:
        raise ComposeLipidError("source-event search bound must be at least two")
    adjudication = json.loads(paths["adjudication"].read_text())
    if (
        adjudication.get("schema_version") != "forge.source_event_adjudication.v1"
        or adjudication["family"] != config["family"]
        or adjudication["reaction_id"] != config["reaction_id"]
        or any(
            adjudication[name] != config["inputs"][name]
            for name in ("registry", "functional_group_registry")
        )
        or adjudication["assets"]["decomposition_key"] != config["inputs"]["decomposition_key"]
    ):
        raise ComposeLipidError("source-event adjudication attribution changed")
    for name, pin in adjudication["assets"].items():
        resolve_pin(pin, repo, label=name)
    key = json.loads(paths["decomposition_key"].read_text())[config["family"]]
    contract = adjudication["source_contract"]
    if (
        contract["roles"] != key["roles"]
        or set(key["roles"].values()) != {1}
        or key["variable"]
        or contract["events"] != 1
        or contract["stage_order"] != [config["reaction_id"]]
    ):
        raise ComposeLipidError("source-event contract is not the complete single-event invariant")
    adapter = RegistryAssemblyAdapter.from_registry(
        paths["registry"],
        reaction_id=config["reaction_id"],
        expected_sha256=config["inputs"]["registry"]["sha256"],
    )
    mapping = contract["registry_to_source_roles"]
    if set(mapping) != set(adapter.roles) or sorted(mapping.values()) != sorted(key["roles"]):
        raise ComposeLipidError("source-event roles differ from the source key")
    queries = {}
    for role, ref in contract["query_references"].items():
        if ref["input"] not in ("registry", "functional_group_registry"):
            raise ComposeLipidError("functional-group query must come from a pinned registry")
        registry = json.loads(paths[ref["input"]].read_text())
        reaction = next(r for r in registry["reactions"] if r["reaction_id"] == ref["reaction_id"])
        spec = next(r for r in reaction["reactant_roles"] if r["name"] == ref["role"])
        queries[role] = Chem.MolFromSmarts(spec["required_handle_smarts"])
    reader = ProtectedPreparationCorpus(repo, paths["protection_result"])
    imported = reader.imported
    import_config = json.loads((repo / imported["config"]["path"]).read_text())
    order = import_config["program_bindings"][config["family"]]["precursor_id_order"]
    if len(order) != len(mapping) or set(order) != set(mapping.values()):
        raise ComposeLipidError("positional precursor roles are incomplete or inconsistent")
    if imported["inputs"]["decomposition_key"] != config["inputs"]["decomposition_key"]:
        raise ComposeLipidError("preparation and source-event decomposition keys differ")
    guards = _frozen_guards(
        import_config["historical_identity_guards"],
        {name: repo / pin["path"] for name, pin in imported["inputs"].items()},
    )
    kwargs = {
        "site_contract": contract["site_contract"],
        "role_queries": queries,
        "maximum_outcomes": bound,
    }
    return config, adjudication, adapter, reader, guards, order, kwargs


def exclude_component_conflicts(rows: list[dict]) -> None:
    """One family/role/source label cannot silently acquire multiple constitutions.

    This is a declared label-consistency condition, not a claim that opaque labels are
    chemical hashes. All rows sharing a conflicting label are excluded, without voting.
    """
    identities = defaultdict(set)
    for row in rows:
        for component in row["components"]:
            identities[(row["family"], component["role"], component["source_id"])].add(
                component["constitution_id"]
            )
    conflicts = {key for key, values in identities.items() if len(values) > 1}
    for row in rows:
        row["component_label_conflict"] = any(
            (row["family"], c["role"], c["source_id"]) in conflicts for c in row["components"]
        )
        row["eligible_after_known_exclusions"] = (
            row["computed_consistency_pass"]
            and not row["historical_protected_precursor"]
            and not row["component_label_conflict"]
        )


def _evaluate(loaded):
    config, adjudication, adapter, reader, guards, order, kwargs = loaded
    controls = {}
    for control in adjudication["source_controls"]:
        label = control["label"]
        if label in controls or control["source_asset"] not in adjudication["assets"]:
            raise ComposeLipidError("source control is duplicated or lacks a primary asset")
        result = check_source_event(
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
                    check_source_event(adapter, components, source["constitution"], **kwargs)
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


def _summary(rows, controls):
    return {
        "rows": len(rows),
        "by_status": dict(sorted(Counter(r["status"] for r in rows).items())),
        "check_failures": dict(
            sorted(
                Counter(
                    name for r in rows for name, value in r.get("checks", {}).items() if not value
                ).items()
            )
        ),
        "computed_consistency_pass": sum(r["computed_consistency_pass"] for r in rows),
        "historical_protected_precursor": sum(r["historical_protected_precursor"] for r in rows),
        "component_label_conflicts": sum(r["component_label_conflict"] for r in rows),
        "eligible_after_known_exclusions": sum(r["eligible_after_known_exclusions"] for r in rows),
        "shared_single_event_invariant_qualified": bool(controls)
        and any(r["computed_consistency_pass"] for r in rows),
        "full_source_family_architecture_qualified": False,
        "remaining_holds": HOLDS,
        "training_admitted": 0,
        "training_ready": False,
    }


def run_source_event(repo_root: Path, config_path: Path, output_dir: Path) -> dict:
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


def verify_source_event(repo_root: Path, result_path: Path) -> dict:
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
