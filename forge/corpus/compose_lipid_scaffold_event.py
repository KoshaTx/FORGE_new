"""Qualify source-scaffold assembly events on the enforced preparation population."""

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
from forge.assembly.precursor_scaffolds import assess_precursor_scaffold
from forge.assembly.repeated_components import RepeatBounds, replay_repeated_components
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid import _frozen_guards
from forge.corpus.compose_lipid_protection import ProtectedPreparationCorpus
from forge.corpus.compose_lipid_source_event import exclude_component_conflicts

CONFIG_SCHEMA = "forge.compose_lipid_scaffold_event_config.v1"
RESULT_SCHEMA = "forge.compose_lipid_scaffold_event.v1"
POLICY = {
    "training_calls": 0,
    "generation_calls": 0,
    "provider_heldout_graphs_parsed": False,
    "source_flags_changed": False,
    "seed": 0,
    "fit_population": "enforced_product_protection_preparation_view",
}
IMPLEMENTATION = (
    "forge/assembly/precursor_scaffolds.py",
    "forge/assembly/repeated_components.py",
    "forge/assembly/families.py",
    "forge/assembly/registry.py",
    "forge/assembly/program.py",
    "forge/assembly/api.py",
    "forge/chemistry/reactive_sites.py",
    "forge/corpus/compose_lipid_scaffold_event.py",
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
        raise ComposeLipidError("scaffold-event configuration or scientific scope changed")
    paths = {name: resolve_pin(pin, repo, label=name) for name, pin in config["inputs"].items()}
    if set(paths) != {"protection_result", "registry", "adjudication", "decomposition_key"}:
        raise ComposeLipidError("scaffold-event inputs changed")
    registry = json.loads(paths["registry"].read_text())
    source = json.loads(paths["adjudication"].read_text())
    if (
        source.get("schema_version") != "forge.scaffold_event_adjudication.v1"
        or source["registry"] != config["inputs"]["registry"]
        or set(source["families"]) != set(config["families"])
    ):
        raise ComposeLipidError("scaffold-event source attribution or family scope changed")
    for owner, assets in (("registry", registry["source_assets"]), ("source", source["assets"])):
        for name, pin in assets.items():
            resolve_pin(pin, repo, label=f"{owner}.{name}")
    if registry["source_assets"] != source["assets"]:
        raise ComposeLipidError("registry and adjudication source assets differ")
    if source["assets"]["decomposition_key"] != config["inputs"]["decomposition_key"]:
        raise ComposeLipidError("source adjudication uses another decomposition key")
    scaffolds = json.loads((repo / source["assets"]["scaffolds.json"]["path"]).read_text())
    controls = json.loads((repo / source["assets"]["controls.json"]["path"]).read_text())
    key = json.loads(paths["decomposition_key"].read_text())
    adapters, reactions = {}, {}
    for family, binding in config["families"].items():
        reaction = next(
            r for r in registry["reactions"] if r["reaction_id"] == binding["reaction_id"]
        )
        for name, pin in reaction["implementation"].items():
            resolve_pin(pin, repo, label=name)
        contract = reaction["precursor_scaffolds"]
        if (
            contract != scaffolds["families"][family]
            or set(key[family]["roles"]) != set(binding["role_fields"])
            or set(key[family]["roles"].values()) != {1}
            or key[family]["variable"]
            or contract["events"] != 1
            or key[family]["field_to_role"]
            != {field: [role] for role, field in binding["role_fields"].items()}
            or not {r["architecture_subfamily"] for r in contract["scaffolds"]}.issubset(
                {r["id"] for r in key[family]["architecture_subfamilies"]}
            )
        ):
            raise ComposeLipidError("source scaffold, role or event-count contract differs")
        adapters[family] = RegistryAssemblyAdapter.from_registry(
            paths["registry"],
            reaction_id=binding["reaction_id"],
            expected_sha256=config["inputs"]["registry"]["sha256"],
        )
        if set(adapters[family].roles) != set(binding["role_fields"]):
            raise ComposeLipidError("source roles differ from the event adapter")
        reactions[family] = reaction
    bounds = RepeatBounds(**config["search_bounds"])
    reader = ProtectedPreparationCorpus(repo, paths["protection_result"])
    imported = reader.imported
    import_config = json.loads((repo / imported["config"]["path"]).read_text())
    if imported["inputs"]["decomposition_key"] != config["inputs"]["decomposition_key"]:
        raise ComposeLipidError("preparation and source decomposition keys differ")
    guards = _frozen_guards(
        import_config["historical_identity_guards"],
        {name: repo / pin["path"] for name, pin in imported["inputs"].items()},
    )
    return config, controls, adapters, reactions, bounds, reader, guards


def _replay(adapter, components, target, reaction, bounds):
    contract = reaction["precursor_scaffolds"]
    return replay_repeated_components(
        adapter,
        components,
        target,
        accumulator_role=next(role for role in adapter.roles if role != contract["precursor_role"]),
        events=contract["events"],
        byproducts_per_event=contract["net_byproducts_per_event"],
        bounds=bounds,
    )


def _controls(controls, adapters, reactions, bounds):
    results = {}
    for c in controls["controls"]:
        family, label = c["family"], c["label"]
        if family not in adapters or label in results:
            raise ComposeLipidError("source control family or label is invalid")
        adapter, reaction = adapters[family], reactions[family]
        canonical = {role: constitutional_molecule(smi)[0] for role, smi in c["components"].items()}
        inferred = adapter.decompose(
            c["expected_product"], maximum_outcomes=bounds.maximum_outcomes
        )
        replay = _replay(adapter, c["components"], c["expected_product"], reaction, bounds)
        scaffold = assess_precursor_scaffold(
            c["components"][reaction["precursor_scaffolds"]["precursor_role"]],
            reaction,
            maximum_matches=bounds.maximum_outcomes,
        )
        formula = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(c["expected_product"]))
        if (
            len(inferred) != 1
            or dict(inferred[0].components) != canonical
            or formula != c["expected_formula"]
            or not replay["computed_consistency_pass"]
            or not scaffold["computed_scaffold_pass"]
        ):
            raise ComposeLipidError(f"primary source control failed: {label}")
        results[label] = {
            "family": family,
            "formula": formula,
            "replay": replay,
            "scaffold": scaffold,
        }
    if {c["family"] for c in results.values()} != set(adapters):
        raise ComposeLipidError("each family needs an independent primary positive control")
    for c in controls["precursor_controls"]:
        if c["family"] not in adapters or c["label"] in results:
            raise ComposeLipidError("precursor control family or label is invalid")
        check = assess_precursor_scaffold(
            c["smiles"], reactions[c["family"]], maximum_matches=bounds.maximum_outcomes
        )
        formula = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(c["smiles"]))
        if (
            not check["computed_scaffold_pass"]
            or check["witnesses"][0]["scaffold_id"] != c["expected_scaffold"]
            or formula != c["expected_formula"]
        ):
            raise ComposeLipidError(f"primary precursor control failed: {c['label']}")
        results[c["label"]] = {"family": c["family"], "formula": formula, "scaffold": check}
    return results


def _evaluate(loaded):
    config, controls, adapters, reactions, bounds, reader, guards = loaded
    source_controls = _controls(controls, adapters, reactions, bounds)
    rows = []
    for family, binding in sorted(config["families"].items()):
        seen = set()
        adapter, reaction = adapters[family], reactions[family]
        for item in reader.iter_preparation_records(family=family):
            record, preparation = item["source"], item["preparation"]
            if record["target_id"] in seen or not preparation["eligible_for_program_preparation"]:
                raise ComposeLipidError(
                    "duplicate or excluded preparation row reached scaffold event"
                )
            seen.add(record["target_id"])
            meta = record["primary_metadata"]
            row = {
                "target_id": record["target_id"],
                "constitution_id": preparation["constitution_id"],
                "family": family,
                "source_metadata": meta,
                "event_count": reaction["precursor_scaffolds"]["events"],
                "source_stage_order": reaction["precursor_scaffolds"]["source_stage_order"],
                "components": [],
                "computed_consistency_pass": False,
                "historical_protected_precursor": False,
                "training_admitted": False,
                "experimental_execution_admitted": False,
            }
            if any(
                not isinstance(meta.get(field), str) or not meta[field]
                for field in binding["role_fields"].values()
            ):
                row["status"] = "excluded_missing_component_labels"
            else:
                try:
                    candidates = adapter.decompose(
                        record["constitution"], maximum_outcomes=bounds.maximum_outcomes
                    )
                    row["inverse_candidate_count"] = len(candidates)
                    if len(candidates) != 1:
                        row["status"] = (
                            "excluded_ambiguous_inverse" if candidates else "excluded_no_inverse"
                        )
                    else:
                        components = dict(candidates[0].components)
                        for role, smiles in sorted(components.items()):
                            digest = hashlib.sha256(smiles.encode()).hexdigest()
                            row["components"].append(
                                {
                                    "role": role,
                                    "canonical_smiles": smiles,
                                    "constitution_id": digest,
                                    "source_id": meta[binding["role_fields"][role]],
                                    "historical_fold": guards.effective(digest),
                                    "count": 1,
                                }
                            )
                        row["historical_protected_precursor"] = any(
                            c["historical_fold"] in ("calibration", "heldout")
                            for c in row["components"]
                        )
                        scaffold = assess_precursor_scaffold(
                            components[reaction["precursor_scaffolds"]["precursor_role"]],
                            reaction,
                            maximum_matches=bounds.maximum_outcomes,
                        )
                        replay = _replay(
                            adapter, components, record["constitution"], reaction, bounds
                        )
                        row.update(scaffold=scaffold, replay=replay)
                        row["computed_consistency_pass"] = (
                            scaffold["computed_scaffold_pass"]
                            and replay["computed_consistency_pass"]
                        )
                        row["status"] = (
                            "computed_consistent"
                            if row["computed_consistency_pass"]
                            else (
                                "excluded_precursor_scaffold"
                                if not scaffold["computed_scaffold_pass"]
                                else "excluded_forward_or_stoichiometry"
                            )
                        )
                        if row["computed_consistency_pass"]:
                            row["computed_architecture_subfamily"] = scaffold["witnesses"][0][
                                "architecture_subfamily"
                            ]
                except LibraryAssemblyError as exc:
                    row.update(status="excluded_replay_error", error=str(exc))
            rows.append(row)
        if (
            len(seen)
            != reader.result["summary"]["by_family"][family]["eligible_for_program_preparation"]
        ):
            raise ComposeLipidError("scaffold-event ledger lost preparation rows")
    exclude_component_conflicts(rows)
    return source_controls, rows


def _summary(rows, families):
    by_family = {}
    for family in sorted(families):
        selected = [r for r in rows if r["family"] == family]
        by_family[family] = {
            "rows": len(selected),
            "by_status": dict(sorted(Counter(r["status"] for r in selected).items())),
            "by_computed_architecture": dict(
                sorted(
                    Counter(
                        r["computed_architecture_subfamily"]
                        for r in selected
                        if r["computed_consistency_pass"]
                    ).items()
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
            "remaining_family_programs_unqualified",
            "final_balanced_train_dataset_unbuilt",
            "repository_full_tests_fail",
        ],
        "precision_scope": "Primary controls and adversarial checks validate source-scaffold transform consistency, not corpus-wide chemical precision, experimental source-bank membership or selectivity.",
    }


def run_scaffold_event(repo_root: Path, config_path: Path, output_dir: Path) -> dict:
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if output.exists() or not output.is_relative_to(repo) or not config_path.is_relative_to(repo):
        raise ComposeLipidError("scaffold-event output must be fresh and inside the repository")
    loaded = _load(repo, config_path)
    controls, rows = _evaluate(loaded)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".scaffold-event-", dir=output.parent) as temporary:
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
            "status": "source_scaffold_events_evaluated_training_unqualified",
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


def verify_scaffold_event(repo_root: Path, result_path: Path) -> dict:
    repo = repo_root.resolve()
    result = json.loads((repo / result_path).read_text())
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("status") != "source_scaffold_events_evaluated_training_unqualified"
        or result.get("policy") != POLICY
        or set(result.get("implementation", {})) != set(IMPLEMENTATION)
        or set(result.get("artifacts", {})) != {"programs.jsonl.gz"}
    ):
        raise ComposeLipidError("scaffold-event receipt scope or provenance changed")
    for name, pin in result["implementation"].items():
        if resolve_pin(pin, repo, label=name) != (repo / name).resolve():
            raise ComposeLipidError("scaffold-event implementation path substitution")
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
        raise ComposeLipidError("scaffold-event receipt differs from complete independent replay")
    return result
