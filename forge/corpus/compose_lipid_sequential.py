"""Source-qualified sequential-program replay on the enforced preparation view."""

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
from forge.assembly.repeated_components import RepeatBounds
from forge.assembly.sequential_program import RegistrySequentialProgram
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid import _frozen_guards
from forge.corpus.compose_lipid_protection import ProtectedPreparationCorpus
from forge.corpus.compose_lipid_source_event import exclude_component_conflicts

CONFIG_SCHEMA = "forge.compose_lipid_sequential_config.v1"
RESULT_SCHEMA = "forge.compose_lipid_sequential.v1"
POLICY = {
    "training_calls": 0,
    "generation_calls": 0,
    "provider_heldout_graphs_parsed": False,
    "source_flags_changed": False,
    "seed": 0,
    "fit_population": "enforced_product_protection_preparation_view",
}
IMPLEMENTATION = (
    "forge/assembly/sequential_program.py",
    "forge/assembly/repeated_components.py",
    "forge/assembly/families.py",
    "forge/assembly/registry.py",
    "forge/assembly/program.py",
    "forge/assembly/api.py",
    "forge/chemistry/reactive_sites.py",
    "forge/corpus/compose_lipid_sequential.py",
    "forge/corpus/compose_lipid_protection.py",
    "forge/corpus/compose_lipid_source_event.py",
    "forge/corpus/compose_lipid.py",
    "forge/corpus/library_splits.py",
    "forge/core/hashing.py",
)


def _pin(repo, path):
    return {"path": path.relative_to(repo).as_posix(), "sha256": str(sha256_file(path))}


def _dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _load(repo, config_path):
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("policy") != POLICY:
        raise ComposeLipidError("sequential configuration or scientific scope changed")
    paths = {name: resolve_pin(pin, repo, label=name) for name, pin in config["inputs"].items()}
    if set(paths) != {"protection_result", "registry", "adjudication", "decomposition_key"}:
        raise ComposeLipidError("sequential input set changed")
    registry = json.loads(paths["registry"].read_text())
    source = json.loads(paths["adjudication"].read_text())
    if (
        source.get("schema_version") != "forge.sequential_program_adjudication.v1"
        or source["registry"] != config["inputs"]["registry"]
        or source["families"] != config["families"]
        or source["assets"] != registry["source_assets"]
    ):
        raise ComposeLipidError("sequential source attribution or family scope changed")
    for name, pin in source["assets"].items():
        resolve_pin(pin, repo, label=name)
    if source["assets"]["decomposition_key"] != config["inputs"]["decomposition_key"]:
        raise ComposeLipidError("sequential source uses another decomposition key")
    key = json.loads(paths["decomposition_key"].read_text())
    contract = json.loads((repo / source["assets"]["stage_contract.json"]["path"]).read_text())
    controls = json.loads(
        (repo / source["assets"]["control_transcriptions.json"]["path"]).read_text()
    )
    if set(config["families"]) != {contract["family"]}:
        raise ComposeLipidError("sequential source contract does not cover declared families")
    expected = {
        k: contract[k]
        for k in (
            "program_id",
            "initial_role",
            "terminal_constraints",
            "product_constraints",
            "architecture_subfamily",
            "scope",
        )
    }
    expected["stages"] = [
        {
            k: s[k]
            for k in (
                "reaction_id",
                "accumulator_role",
                "added_role",
                "net_byproducts",
                "source_step",
            )
        }
        for s in contract["stages"]
    ]
    if registry["programs"] != [expected]:
        raise ComposeLipidError("sequential stages differ from the primary source contract")
    programs = {}
    for family, binding in config["families"].items():
        if (
            binding["program_id"] != contract["program_id"]
            or set(key[family]["roles"]) != set(binding["role_fields"])
            or any(type(n) is not int or n != 1 for n in key[family]["roles"].values())
            or key[family]["variable"]
            or key[family]["field_to_role"] != {v: [k] for k, v in binding["role_fields"].items()}
            or contract["architecture_subfamily"]
            not in {s["id"] for s in key[family]["architecture_subfamilies"]}
        ):
            raise ComposeLipidError(
                "sequential source role, multiplicity or architecture contract differs"
            )
        programs[family] = RegistrySequentialProgram.from_registry(
            paths["registry"],
            program_id=binding["program_id"],
            expected_sha256=config["inputs"]["registry"]["sha256"],
            bounds=RepeatBounds(**config["search_bounds"]),
        )
        if set(programs[family].roles) != set(binding["role_fields"]):
            raise ComposeLipidError("sequential program roles differ from source metadata")
        for reaction in registry["reactions"]:
            for name, pin in reaction["implementation"].items():
                resolve_pin(pin, repo, label=name)
    reader = ProtectedPreparationCorpus(repo, paths["protection_result"])
    imported = reader.imported
    import_config = json.loads((repo / imported["config"]["path"]).read_text())
    if imported["inputs"]["decomposition_key"] != config["inputs"]["decomposition_key"]:
        raise ComposeLipidError("preparation and sequential decomposition keys differ")
    guards = _frozen_guards(
        import_config["historical_identity_guards"],
        {name: repo / pin["path"] for name, pin in imported["inputs"].items()},
    )
    return config, controls, programs, reader, guards


def _evaluate(loaded):
    config, controls, programs, reader, guards = loaded
    results, rows = {}, []
    for family, program in programs.items():
        for control in controls["controls"]:
            label = control["label"]
            if label in results:
                raise ComposeLipidError("duplicate sequential source control")
            replay = program.replay(control["components"], control["expected_product"])
            molecule = Chem.MolFromSmiles(control["expected_product"])
            params = Chem.SmilesParserParams()
            params.removeHs = False
            ion = Chem.CombineMols(molecule, Chem.MolFromSmiles("[H+]", params))
            mass = rdMolDescriptors.CalcExactMolWt(ion)
            if (
                not replay["computed_consistency_pass"]
                or round(mass, 2) != control["source_theoretical_mz"]
            ):
                raise ComposeLipidError(f"independent sequential source control failed: {label}")
            results[label] = {
                "family": family,
                "neutral_formula": rdMolDescriptors.CalcMolFormula(molecule),
                "computed_protonated_mz": mass,
                "replay": replay,
            }
        seen = set()
        for item in reader.iter_preparation_records(family=family):
            source, preparation = item["source"], item["preparation"]
            if source["target_id"] in seen or not preparation["eligible_for_program_preparation"]:
                raise ComposeLipidError("duplicate or protected row reached sequential preparation")
            seen.add(source["target_id"])
            meta = source["primary_metadata"]
            fields = config["families"][family]["role_fields"]
            row = {
                "target_id": source["target_id"],
                "constitution_id": preparation["constitution_id"],
                "family": family,
                "source_metadata": meta,
                "components": [],
                "program_id": program.specification["program_id"],
                "stage_reaction_ids": [s["reaction_id"] for s in program.specification["stages"]],
                "computed_consistency_pass": False,
                "historical_protected_precursor": False,
                "training_admitted": False,
                "experimental_execution_admitted": False,
            }
            if any(
                not isinstance(meta.get(field), str) or not meta[field] for field in fields.values()
            ):
                row["status"] = "excluded_missing_component_labels"
            else:
                inverse = program.infer(source["constitution"])
                row["complete_inverse"] = inverse
                candidates = inverse["candidate_components"]
                if not inverse["complete_search"]:
                    row["status"] = "excluded_search_bound"
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
                                "source_id": meta[fields[role]],
                                "historical_fold": guards.effective(digest),
                                "count": 1,
                            }
                        )
                    row["historical_protected_precursor"] = any(
                        c["historical_fold"] in ("calibration", "heldout")
                        for c in row["components"]
                    )
                    replay = program.replay(components, source["constitution"])
                    row["replay"] = replay
                    row["computed_consistency_pass"] = replay["computed_consistency_pass"]
                    row["status"] = (
                        "computed_consistent"
                        if replay["computed_consistency_pass"]
                        else "excluded_program_or_component_contract"
                    )
                    if replay["computed_consistency_pass"]:
                        row["computed_architecture_subfamily"] = program.specification[
                            "architecture_subfamily"
                        ]
            rows.append(row)
        if (
            len(seen)
            != reader.result["summary"]["by_family"][family]["eligible_for_program_preparation"]
        ):
            raise ComposeLipidError("sequential ledger lost preparation rows")
    if not results:
        raise ComposeLipidError("sequential program requires independent primary controls")
    exclude_component_conflicts(rows)
    return results, rows


def _summary(rows, families):
    by_family = {}
    for family in sorted(families):
        selected = [r for r in rows if r["family"] == family]
        by_family[family] = {
            "rows": len(selected),
            "by_status": dict(sorted(Counter(r["status"] for r in selected).items())),
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
        "precision_scope": "Independent source controls and adversarial tests check the declared program, not experimental source-bank membership, yield, pKa or corpus-wide decomposition precision.",
    }


def run_sequential_program(repo_root: Path, config_path: Path, output_dir: Path) -> dict:
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if output.exists() or not output.is_relative_to(repo) or not config_path.is_relative_to(repo):
        raise ComposeLipidError("sequential output must be fresh and inside the repository")
    loaded = _load(repo, config_path)
    controls, rows = _evaluate(loaded)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sequential-", dir=output.parent) as temporary:
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
            "status": "source_sequential_programs_evaluated_training_unqualified",
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


def verify_sequential_program(repo_root: Path, result_path: Path) -> dict:
    repo = repo_root.resolve()
    result = json.loads((repo / result_path).read_text())
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("status") != "source_sequential_programs_evaluated_training_unqualified"
        or result.get("policy") != POLICY
        or set(result.get("implementation", {})) != set(IMPLEMENTATION)
        or set(result.get("artifacts", {})) != {"programs.jsonl.gz"}
    ):
        raise ComposeLipidError("sequential receipt scope or provenance changed")
    for name, pin in result["implementation"].items():
        if resolve_pin(pin, repo, label=name) != (repo / name).resolve():
            raise ComposeLipidError("sequential implementation path substitution")
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
        raise ComposeLipidError("sequential receipt differs from complete independent replay")
    return result
