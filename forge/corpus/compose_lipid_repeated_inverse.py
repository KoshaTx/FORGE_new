"""Qualify repeated source programs using complete inverse tuples on protected TRAIN."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.families import (
    LibraryAssemblyError,
    RegistryAssemblyAdapter,
    constitutional_molecule,
)
from forge.assembly.repeated_components import RepeatBounds, replay_repeated_components
from forge.assembly.repeated_inverse import infer_repeated_components
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid import _frozen_guards
from forge.corpus.compose_lipid_protection import ProtectedPreparationCorpus
from forge.corpus.compose_lipid_source_event import exclude_component_conflicts

CONFIG_SCHEMA = "forge.compose_lipid_repeated_inverse_config.v1"
RESULT_SCHEMA = "forge.compose_lipid_repeated_inverse.v1"
POLICY = {
    "training_calls": 0,
    "generation_calls": 0,
    "provider_heldout_graphs_parsed": False,
    "source_flags_changed": False,
    "seed": 0,
    "fit_population": "enforced_product_protection_preparation_view",
}
IMPLEMENTATION = (
    "forge/assembly/repeated_inverse.py",
    "forge/assembly/repeated_components.py",
    "forge/assembly/families.py",
    "forge/assembly/registry.py",
    "forge/assembly/program.py",
    "forge/chemistry/reactive_sites.py",
    "forge/corpus/compose_lipid_repeated_inverse.py",
    "forge/corpus/compose_lipid_protection.py",
    "forge/corpus/compose_lipid_source_event.py",
    "forge/corpus/compose_lipid.py",
    "forge/corpus/library_splits.py",
    "forge/core/hashing.py",
)


def _pin(repo: Path, path: Path) -> dict:
    return {"path": path.relative_to(repo).as_posix(), "sha256": str(sha256_file(path))}


def _dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _load(repo: Path, config_path: Path):
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("policy") != POLICY:
        raise ComposeLipidError("repeated-inverse schema or scientific scope changed")
    paths = {name: resolve_pin(pin, repo, label=name) for name, pin in config["inputs"].items()}
    if set(paths) != {"protection_result", "registry", "adjudication", "decomposition_key"}:
        raise ComposeLipidError("repeated-inverse inputs changed")
    registry = json.loads(paths["registry"].read_text())
    source = json.loads(paths["adjudication"].read_text())
    if (
        source.get("schema_version") != "forge.repeated_source_adjudication.v1"
        or source["registry"] != config["inputs"]["registry"]
        or set(source["families"]) != set(config["families"])
    ):
        raise ComposeLipidError("repeated-inverse source attribution or family scope changed")
    for owner, assets in (("registry", registry["source_assets"]), ("source", source["assets"])):
        for name, pin in assets.items():
            resolve_pin(pin, repo, label=f"{owner}.{name}")
    for name in registry["source_assets"].keys() & source["assets"].keys():
        if registry["source_assets"][name] != source["assets"][name]:
            raise ComposeLipidError(f"registry and adjudication source asset differ: {name}")
    if source["assets"]["decomposition_key"] != config["inputs"]["decomposition_key"]:
        raise ComposeLipidError("source adjudication uses another decomposition key")
    key = json.loads(paths["decomposition_key"].read_text())
    adapters, programs = {}, {}
    for family, binding in config["families"].items():
        reaction = next(
            r for r in registry["reactions"] if r["reaction_id"] == binding["reaction_id"]
        )
        for name in ("parent_registry", "functional_group_registry"):
            resolve_pin(reaction["implementation"][name], repo, label=name)
        program = reaction["source_program"]
        if (
            set(key[family]["roles"]) != set(binding["role_fields"])
            or key[family]["field_to_role"]
            != {field: [role] for role, field in binding["role_fields"].items()}
            or key[family]["variable"] != [program["event_count_field"]]
            or program["event_count_field"] != source["source_contract"]["event_count_field"]
            or program["accumulator_role"] != source["source_contract"]["accumulator_role"]
            or program["repeated_roles"] != [source["source_contract"]["side_role"]]
            or binding["interface_stratum"] != program["interface_stratum"]
            or program["maximum_events"]
            != max(source["source_contract"]["source_supported_events"])
        ):
            raise ComposeLipidError("source role, occupancy or interface contract differs")
        adapters[family] = RegistryAssemblyAdapter.from_registry(
            paths["registry"],
            reaction_id=binding["reaction_id"],
            expected_sha256=config["inputs"]["registry"]["sha256"],
        )
        if set(adapters[family].roles) != set(binding["role_fields"]):
            raise ComposeLipidError("source roles differ from the event adapter")
        programs[family] = program
    bounds = RepeatBounds(**config["search_bounds"])
    if bounds.maximum_events < max(p["maximum_events"] for p in programs.values()):
        raise ComposeLipidError("execution bound silently shrinks source event support")
    reader = ProtectedPreparationCorpus(repo, paths["protection_result"])
    imported = reader.imported
    import_config = json.loads((repo / imported["config"]["path"]).read_text())
    if imported["inputs"]["decomposition_key"] != config["inputs"]["decomposition_key"]:
        raise ComposeLipidError("preparation and source decomposition keys differ")
    guards = _frozen_guards(
        import_config["historical_identity_guards"],
        {name: repo / pin["path"] for name, pin in imported["inputs"].items()},
    )
    return config, source, adapters, programs, bounds, reader, guards


def _infer(adapter, target, events, program, bounds):
    return infer_repeated_components(
        adapter, target, accumulator_role=program["accumulator_role"], events=events, bounds=bounds
    )


def _replay(adapter, components, target, events, program, bounds):
    return replay_repeated_components(
        adapter,
        components,
        target,
        accumulator_role=program["accumulator_role"],
        events=events,
        byproducts_per_event=program["net_byproducts_per_event"],
        bounds=bounds,
    )


def _controls(source, adapters, programs, bounds):
    controls = {}
    for kind in ("source_controls", "ambiguity_controls"):
        for control in source[kind]:
            label, family = control["label"], control["family"]
            if label in controls or family not in adapters:
                raise ComposeLipidError("source controls are duplicated or outside family scope")
            adapter, program = adapters[family], programs[family]
            target, events = control["expected_product"], control["events"]
            inferred = _infer(adapter, target, events, program, bounds)
            replayed = _replay(adapter, control["components"], target, events, program, bounds)
            canonical = {
                role: constitutional_molecule(smi)[0] for role, smi in control["components"].items()
            }
            formula = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(target))
            if (
                not inferred["complete_search"]
                or inferred["candidate_components"] != [canonical]
                or formula != control["expected_formula"]
            ):
                raise ComposeLipidError(
                    f"source control identity or complete inverse failed: {label}"
                )
            if kind == "source_controls":
                valid = replayed["computed_consistency_pass"]
            else:
                valid = (
                    replayed["checks"]["complete_search"]
                    and not replayed["checks"]["unique_forward_exact"]
                    and replayed["checks"]["full_element_hydrogen_charge_balance"]
                    and constitutional_molecule(target)[0] in replayed["forward_layers"][-1]
                    and len(replayed["forward_layers"][-1])
                    == control["expected_forward_product_count"]
                )
            if not valid:
                raise ComposeLipidError(
                    f"source control program or ambiguity accounting failed: {label}"
                )
            controls[label] = {
                "family": family,
                "kind": kind,
                "formula": formula,
                "expectation_pass": True,
                "inverse": inferred,
                "replay": replayed,
            }
    if {r["family"] for r in controls.values() if r["kind"] == "source_controls"} != set(adapters):
        raise ComposeLipidError("each family requires an independent positive primary control")
    return controls


def _evaluate(loaded):
    config, source, adapters, programs, bounds, reader, guards = loaded
    controls = _controls(source, adapters, programs, bounds)
    rows = []
    for family, binding in sorted(config["families"].items()):
        seen = set()
        adapter, program = adapters[family], programs[family]
        for item in reader.iter_preparation_records(family=family):
            record, preparation = item["source"], item["preparation"]
            if record["target_id"] in seen or not preparation["eligible_for_program_preparation"]:
                raise ComposeLipidError(
                    "duplicate or excluded preparation row reached repeated inverse"
                )
            seen.add(record["target_id"])
            meta = record["primary_metadata"]
            events = meta.get(program["event_count_field"])
            row = {
                "target_id": record["target_id"],
                "constitution_id": preparation["constitution_id"],
                "family": family,
                "source_metadata": meta,
                "components": [],
                "computed_consistency_pass": False,
                "historical_protected_precursor": False,
                "training_admitted": False,
                "experimental_execution_admitted": False,
                "architecture_assignment_admitted": False,
            }
            if type(events) is not int or any(
                not isinstance(meta.get(field), str) or not meta[field]
                for field in binding["role_fields"].values()
            ):
                row["status"] = "excluded_missing_or_invalid_program_metadata"
            elif (
                events not in source["source_contract"]["source_supported_events"]
                or meta.get("interface_stratum") != binding["interface_stratum"]
            ):
                row["status"] = "excluded_occupancy_or_interface_scope"
            else:
                try:
                    inferred = _infer(adapter, record["constitution"], events, program, bounds)
                    row["inverse"] = inferred
                    candidates = inferred["candidate_components"]
                    if not inferred["complete_search"]:
                        row["status"] = "excluded_incomplete_inverse_search"
                    elif len(candidates) != 1:
                        row["status"] = (
                            "excluded_ambiguous_complete_inverse"
                            if candidates
                            else "excluded_no_complete_inverse"
                        )
                    else:
                        components = candidates[0]
                        for role, smiles in sorted(components.items()):
                            digest = hashlib.sha256(smiles.encode()).hexdigest()
                            row["components"].append(
                                {
                                    "role": role,
                                    "canonical_smiles": smiles,
                                    "constitution_id": digest,
                                    "source_id": meta[binding["role_fields"][role]],
                                    "historical_fold": guards.effective(digest),
                                    "count": 1 if role == program["accumulator_role"] else events,
                                }
                            )
                        row["historical_protected_precursor"] = any(
                            c["historical_fold"] in ("calibration", "heldout")
                            for c in row["components"]
                        )
                        replayed = _replay(
                            adapter, components, record["constitution"], events, program, bounds
                        )
                        row["replay"] = replayed
                        row["computed_consistency_pass"] = replayed["computed_consistency_pass"]
                        row["status"] = (
                            "computed_consistent"
                            if row["computed_consistency_pass"]
                            else "excluded_forward_or_balance_failure"
                        )
                except LibraryAssemblyError as exc:
                    row["status"] = "excluded_replay_error"
                    row["error"] = str(exc)
            rows.append(row)
        if (
            len(seen)
            != reader.result["summary"]["by_family"][family]["eligible_for_program_preparation"]
        ):
            raise ComposeLipidError("repeated-inverse ledger lost preparation rows")
    exclude_component_conflicts(rows)
    return controls, rows


def _summary(rows, families):
    by_family = {}
    for family in sorted(families):
        selected = [row for row in rows if row["family"] == family]
        by_family[family] = {
            "rows": len(selected),
            "by_status": dict(sorted(Counter(r["status"] for r in selected).items())),
            "by_declared_occupancy": dict(
                sorted(
                    Counter(str(r["source_metadata"].get("occupancy")) for r in selected).items()
                )
            ),
            "computed_consistency_pass": sum(r["computed_consistency_pass"] for r in selected),
            "consistent_with_known_protected_precursor": sum(
                r["computed_consistency_pass"] and r["historical_protected_precursor"]
                for r in selected
            ),
            "component_label_conflicts": sum(r["component_label_conflict"] for r in selected),
            "eligible_after_known_exclusions": sum(
                r["eligible_after_known_exclusions"] for r in selected
            ),
            "training_admitted": 0,
        }
    return {
        "by_family": by_family,
        "training_ready": False,
        "training_admitted": 0,
        "remaining_holds": [
            "global_precursor_protection_incomplete",
            "precursor_holdouts_unqualified",
            "source_architecture_and_bank_assignments_incomplete",
            "remaining_family_programs_unqualified",
            "final_balanced_train_dataset_unbuilt",
            "repository_full_tests_fail",
        ],
        "precision_scope": "Primary controls and adversarial checks validate computed programs; corpus-wide chemical precision and experimental selectivity are not calibrated.",
    }


def run_repeated_inverse(repo_root: Path, config_path: Path, output_dir: Path) -> dict:
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if output.exists() or not output.is_relative_to(repo) or not config_path.is_relative_to(repo):
        raise ComposeLipidError("repeated-inverse output must be fresh and inside the repository")
    loaded = _load(repo, config_path)
    controls, rows = _evaluate(loaded)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".repeated-inverse-", dir=output.parent) as temporary:
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
            "status": "repeated_programs_evaluated_training_unqualified",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": _pin(repo, config_path),
            "inputs": loaded[0]["inputs"],
            "implementation": {name: _pin(repo, repo / name) for name in IMPLEMENTATION},
            "policy": POLICY,
            "source_controls": controls,
            "summary": _summary(rows, loaded[0]["families"]),
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


def verify_repeated_inverse(repo_root: Path, result_path: Path) -> dict:
    repo = repo_root.resolve()
    result = json.loads((repo / result_path).read_text())
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("status") != "repeated_programs_evaluated_training_unqualified"
        or result.get("policy") != POLICY
        or set(result.get("implementation", {})) != set(IMPLEMENTATION)
        or set(result.get("artifacts", {})) != {"programs.jsonl.gz"}
    ):
        raise ComposeLipidError("repeated-inverse receipt scope or provenance changed")
    for name, pin in result["implementation"].items():
        if resolve_pin(pin, repo, label=name) != (repo / name).resolve():
            raise ComposeLipidError("repeated-inverse implementation path substitution")
    ledger = resolve_pin(result["artifacts"]["programs.jsonl.gz"], repo, label="programs")
    loaded = _load(repo, resolve_pin(result["config"], repo, label="config"))
    controls, expected = _evaluate(loaded)
    with gzip.open(ledger, "rt") as stream:
        actual = [json.loads(line) for line in stream]
    if (
        result["inputs"] != loaded[0]["inputs"]
        or actual != expected
        or result["summary"] != _summary(expected, loaded[0]["families"])
        or result["source_controls"] != controls
    ):
        raise ComposeLipidError("repeated-inverse receipt differs from complete independent replay")
    return result
