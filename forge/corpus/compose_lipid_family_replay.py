"""Replay supplied complete recipes through existing, independently qualified programs.

No inverse decomposition chooses a precursor or its quantity. Other families retain an
explicit qualification gap. All records remain computational evidence, never route labels.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import tempfile
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.condensation_event import check_condensation_event
from forge.assembly.families import LibraryAssemblyError, RegistryAssemblyAdapter
from forge.assembly.precursor_scaffolds import assess_precursor_scaffold
from forge.assembly.source_event import check_source_event
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus import compose_lipid_condensation_event as condensation
from forge.corpus import compose_lipid_scaffold_event as scaffold
from forge.corpus import compose_lipid_sequential as sequential
from forge.corpus import compose_lipid_source_event as event
from forge.corpus import compose_lipid_source_program as passerini
from forge.corpus import compose_lipid_supplied_replay as michael
from forge.corpus.compose_lipid_full_preparation import FullPreparationCorpus
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import compact, rows

CONFIG_SCHEMA = "forge.compose_lipid_family_replay_config.v1"
RESULT_SCHEMA = "forge.compose_lipid_family_replay.v1"
POLICY = {
    "seed": 0,
    "device": "cpu",
    "workers": 1,
    "population": "full_preparation_eligible_intersection_only",
    "components": "supplied_global_ids_roles_quantities_no_product_inference",
    "forward_sites": "unfiltered_exhaustive_with_existing_qualified_bounds",
    "unsupported_family": "pending_source_program_qualification",
    "training_admitted": False,
    "training_calls": 0,
    "experimental_execution_admitted": False,
}
IMPLEMENTATION = tuple(
    sorted(
        set(
            (
                "forge/corpus/compose_lipid_family_replay.py",
                "forge/corpus/compose_lipid_full_preparation.py",
                *michael.IMPLEMENTATION,
                *event.IMPLEMENTATION,
                *condensation.IMPLEMENTATION,
                *scaffold.IMPLEMENTATION,
                *sequential.IMPLEMENTATION,
                *passerini.IMPLEMENTATION,
            )
        )
    )
)


def fixed_components(instances, structures, mapping):
    """Return an exact one-per-role tuple, or an explicit unsupported disposition."""
    if not mapping or len(set(mapping.values())) != len(mapping):
        raise ComposeLipidError("Fixed program role binding is not bijective")
    by_role = {}
    for role, identity, quantity in instances:
        if identity not in structures:
            raise ComposeLipidError(f"Unresolved supplied precursor: {identity}")
        if type(quantity) is not int or quantity < 1:
            raise ComposeLipidError("Invalid supplied precursor quantity")
        if role in by_role:
            return None, "unsupported_source_role_tuple"
        by_role[role] = (structures[identity], quantity)
    if set(by_role) != set(mapping.values()):
        return None, "unsupported_source_role_tuple"
    if any(quantity != 1 for _, quantity in by_role.values()):
        return None, "outside_qualified_program_multiplicity"
    return {role: by_role[source][0] for role, source in mapping.items()}, None


def _scaffold_replay(adapter, reaction, bounds, components, target):
    result = scaffold._replay(adapter, components, target, reaction, bounds)
    check = assess_precursor_scaffold(
        components[reaction["precursor_scaffolds"]["precursor_role"]],
        reaction,
        maximum_matches=bounds.maximum_outcomes,
    )
    result["precursor_scaffold"] = check
    result["checks"]["qualified_complete_precursor_scaffold"] = check["computed_scaffold_pass"]
    result["computed_consistency_pass"] = all(result["checks"].values())
    return result


def _passerini_replay(adapter, contract, components, target):
    result = passerini.check_complete_program(adapter, components, target, contract)
    result["computed_consistency_pass"] = result.pop("qualified")
    return result


def load_executors(repo: Path, paths: dict):
    """Reuse frozen loaders and controls without revising their chemical support."""
    executors, controls = {}, {}
    loaded = michael._load(repo, paths["michael_config"])
    config, _, _, adapters, programs, bounds, checked = loaded
    controls["michael"] = checked
    for family, binding in config["families"].items():
        executors[family] = {
            "kind": "repeated",
            "adapter": adapters[family],
            "binding": binding,
            "program": programs[family],
            "bounds": bounds,
        }
    for key, loader, checker in (
        ("ugi3_config", event._load, check_source_event),
        ("ugi4_config", condensation._load, check_condensation_event),
    ):
        cfg, evidence, adapter, _, _, _, kwargs = loader(repo, paths[key])
        checked = {}
        for control in evidence["source_controls"]:
            result = checker(adapter, control["components"], control["expected_product"], **kwargs)
            formula = rdMolDescriptors.CalcMolFormula(
                Chem.MolFromSmiles(control["expected_product"])
            )
            if not result["computed_consistency_pass"] or formula != control["expected_formula"]:
                raise ComposeLipidError(f"Frozen source control failed: {control['label']}")
            if control["label"] in checked or control["source_asset"] not in evidence["assets"]:
                raise ComposeLipidError("Missing or duplicated source-control attribution")
            checked[control["label"]] = result
        if not checked:
            raise ComposeLipidError("Independent source controls are required")
        controls[cfg["family"]] = checked
        executors[cfg["family"]] = {
            "kind": "fixed",
            "mapping": evidence["source_contract"]["registry_to_source_roles"],
            "run": partial(checker, adapter, **kwargs),
        }
    cfg, _, variant = passerini._load(repo, paths["passerini_config"])
    registry = json.loads(paths["passerini_registry"].read_text())
    if registry["reactions"] != [variant]:
        raise ComposeLipidError("Saved Passerini registry differs from the source-derived variant")
    adapter = RegistryAssemblyAdapter.from_registry(
        paths["passerini_registry"],
        reaction_id=variant["reaction_id"],
        expected_sha256=str(sha256_file(paths["passerini_registry"])),
    )
    controls[cfg["family"]] = passerini._controls(adapter, cfg)
    executors[cfg["family"]] = {
        "kind": "fixed",
        "mapping": cfg["source_contract"]["registry_to_source_roles"],
        "run": partial(_passerini_replay, adapter, cfg["source_contract"]),
    }
    cfg, source_controls, adapters, reactions, bounds, _, _ = scaffold._load(
        repo, paths["reductive_config"]
    )
    controls["reductive"] = scaffold._controls(source_controls, adapters, reactions, bounds)
    for family, adapter in adapters.items():
        executors[family] = {
            "kind": "fixed",
            "mapping": {r: r for r in adapter.roles},
            "run": partial(_scaffold_replay, adapter, reactions[family], bounds),
        }
    cfg, source_controls, programs, _, _ = sequential._load(repo, paths["staar_config"])
    for family, program in programs.items():
        checked = {}
        for control in source_controls["controls"]:
            result = program.replay(control["components"], control["expected_product"])
            mol = Chem.MolFromSmiles(control["expected_product"])
            params = Chem.SmilesParserParams()
            params.removeHs = False
            ion = Chem.CombineMols(mol, Chem.MolFromSmiles("[H+]", params))
            if (
                not result["computed_consistency_pass"]
                or round(rdMolDescriptors.CalcExactMolWt(ion), 2)
                != control["source_theoretical_mz"]
            ):
                raise ComposeLipidError(f"Frozen sequential control failed: {control['label']}")
            if control["label"] in checked:
                raise ComposeLipidError("Duplicated sequential source control")
            checked[control["label"]] = result
        if not checked:
            raise ComposeLipidError("Sequential source controls are required")
        controls[family] = checked
        executors[family] = {
            "kind": "fixed",
            "mapping": {r: r for r in program.roles},
            "run": program.replay,
        }
    return executors, controls


def replay_record(item, structures, executor):
    source, preparation = item["source"], item["preparation"]
    if preparation.get("eligible_for_program_preparation") is not True:
        raise ComposeLipidError("Protected or unassigned target reached family replay")
    if executor is None:
        return {
            "computed_consistency_pass": False,
            "disposition": "pending_source_program_qualification",
        }
    try:
        if executor["kind"] == "repeated":
            result = michael.replay_supplied(
                executor["adapter"],
                item,
                structures,
                executor["binding"],
                executor["program"],
                executor["bounds"],
            )
        else:
            components, reason = fixed_components(
                preparation["component_instances"], structures, executor["mapping"]
            )
            if reason:
                return {"computed_consistency_pass": False, "disposition": reason}
            result = executor["run"](components, source["constitution"])
    except LibraryAssemblyError as exc:
        return {
            "computed_consistency_pass": False,
            "disposition": "unsupported_registry_replay",
            "reason": str(exc),
        }
    if result["computed_consistency_pass"]:
        if not result.get("checks") or not all(result["checks"].values()):
            raise ComposeLipidError("Exact replay lacks complete passing checks")
        result["verified_target_constitution_id"] = hashlib.sha256(
            source["constitution"].encode()
        ).hexdigest()
        if result["verified_target_constitution_id"] != preparation["constitution_id"]:
            # Use the frozen stereo-free canonical identity, never the arbitrary source serialization.
            from forge.assembly.families import constitutional_molecule

            canonical = constitutional_molecule(source["constitution"])[0]
            result["verified_target_constitution_id"] = hashlib.sha256(
                canonical.encode()
            ).hexdigest()
            if result["verified_target_constitution_id"] != preparation["constitution_id"]:
                raise ComposeLipidError(
                    "Supplied replay target differs from authenticated TRAIN identity"
                )
        result["disposition"] = "exact_computed_reconstruction"
    else:
        result.setdefault("disposition", "unresolved_computed_reconstruction")
    return result


def run_family_replay(repo_root: Path, config_path: Path, output_dir: Path):
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if output.exists() or not output.is_relative_to(repo):
        raise ComposeLipidError("Family replay output must be fresh and inside the repository")
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("policy") != POLICY:
        raise ComposeLipidError("Family replay configuration or scientific scope changed")
    if set(config.get("inputs", {})) != {
        "preparation",
        "michael_config",
        "ugi3_config",
        "ugi4_config",
        "passerini_config",
        "passerini_registry",
        "reductive_config",
        "staar_config",
    }:
        raise ComposeLipidError("Family replay input set changed")
    paths = {k: resolve_pin(v, repo, label=k) for k, v in config["inputs"].items()}
    reader = FullPreparationCorpus(repo, paths["preparation"])
    structures = {r["component_id"]: r["constitution"] for r in rows(reader.precursors)}
    executors, controls = load_executors(repo, paths)
    if sorted(executors) != config["qualified_executor_families"]:
        raise ComposeLipidError("Qualified executor family scope changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(Counter)
    with tempfile.TemporaryDirectory(prefix=".family-replay-", dir=output.parent) as temporary:
        stage = Path(temporary)
        ledger = stage / "replay.jsonl.gz"
        with (
            ledger.open("wb") as raw,
            gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as stream,
        ):
            for family in sorted(reader.result["summary"]["by_family"]):
                for item in reader.iter_preparation_records(family=family):
                    result = replay_record(item, structures, executors.get(family))
                    row = {
                        k: item["preparation"][k]
                        for k in (
                            "target_id",
                            "family",
                            "constitution_id",
                            "component_instances",
                            "construction_basis",
                        )
                    }
                    row.update(
                        replay=result,
                        training_admitted=False,
                        experimental_execution_admitted=False,
                    )
                    stream.write((compact(row) + "\n").encode())
                    counts[family]["rows"] += 1
                    counts[family][result["disposition"]] += 1
                    counts[family].update(
                        "failed_" + k for k, v in result.get("checks", {}).items() if not v
                    )
                    if counts[family]["rows"] % 1000 == 0:
                        print(f"{family}: {counts[family]['rows']:,} accounted", flush=True)
                expected = reader.result["summary"]["by_family"][family][
                    "eligible_for_program_preparation"
                ]
                if counts[family]["rows"] != expected:
                    raise ComposeLipidError(f"Family replay coverage changed: {family}")
                print(f"{family}: {dict(counts[family])}", flush=True)
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "supplied_family_replay_complete_training_unqualified",
            "config": pin(repo, config_path),
            "inputs": config["inputs"],
            "implementation": {name: pin(repo, repo / name) for name in IMPLEMENTATION},
            "policy": POLICY,
            "controls": controls,
            "summary": {
                "by_family": {f: dict(c) for f, c in sorted(counts.items())},
                "totals": dict(sum(counts.values(), Counter())),
            },
            "environment": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
            "artifact": {
                "path": str((output / ledger.name).relative_to(repo)),
                "sha256": str(sha256_file(ledger)),
            },
        }
        dump(stage / "result.json", result)
        os.rename(stage, output)
    return result
