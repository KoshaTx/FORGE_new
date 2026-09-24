"""Replay fixed, source-qualified condensation events on current protected TRAIN rows."""

from __future__ import annotations

import gzip
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
from forge.assembly.families import RegistryAssemblyAdapter
from forge.assembly.source_retention import check_retained_condensation
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid_family_replay import replay_record
from forge.corpus.compose_lipid_full_preparation import FullPreparationCorpus
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import compact, rows

CONFIG_SCHEMA = "forge.compose_lipid_fixed_replay_config.v1"
RESULT_SCHEMA = "forge.compose_lipid_fixed_replay.v1"
POLICY = {
    "seed": 0,
    "population": "current_full_preparation_eligible_only",
    "selection_before_molecular_record_decoding": True,
    "component_join": "global_component_id",
    "occupancy_source": "one_supplied_complete_component_per_role_quantity_one",
    "forward_sites": "exhaustive_unfiltered_with_explicit_bounds",
    "retained_handles": "source_role_atom_provenance_excluding_consumed_sites",
    "training_admitted": False,
    "experimental_execution_admitted": False,
    "training_calls": 0,
}
IMPLEMENTATION = (
    "forge/corpus/compose_lipid_fixed_replay.py",
    "forge/corpus/compose_lipid_full_preparation.py",
    "forge/corpus/compose_lipid_family_replay.py",
    "forge/assembly/source_retention.py",
    "forge/assembly/source_event.py",
    "forge/assembly/condensation_event.py",
    "forge/assembly/repeated_components.py",
    "forge/assembly/families.py",
    "forge/assembly/registry.py",
    "forge/assembly/program.py",
    "forge/chemistry/reactive_sites.py",
    "forge/core/hashing.py",
)


def qualify_controls(document, executors):
    checked, covered = {}, defaultdict(set)
    for kind in ("source_controls", "regression_controls"):
        for control in document[kind]:
            label, family = control["label"], control["family"]
            if label in checked or family not in executors:
                raise ComposeLipidError("Duplicate or unconfigured fixed-event control")
            expected = control["expected_computed_consistency_pass"]
            if type(expected) is not bool or (kind == "source_controls" and expected is not True):
                raise ComposeLipidError("Source control must pass; every expectation is boolean")
            if kind == "source_controls" and control["source_asset"] not in document["assets"]:
                raise ComposeLipidError("Source control lacks primary attribution")
            molecule = Chem.MolFromSmiles(control["expected_product"])
            if (
                molecule is None
                or rdMolDescriptors.CalcMolFormula(molecule) != control["expected_formula"]
            ):
                raise ComposeLipidError(f"Fixed control formula differs: {label}")
            result = executors[family]["run"](control["components"], control["expected_product"])
            if (
                result["computed_consistency_pass"] is not expected
                or len(result["forward_products"]) != control["expected_forward_product_count"]
                or any(
                    result["checks"].get(name) is not False
                    for name in control.get("required_failed_checks", [])
                )
            ):
                raise ComposeLipidError(f"Fixed control failed: {label}")
            checked[label] = {"kind": kind, "replay": result}
            covered[family].add(kind)
    if any(covered[f] != {"source_controls", "regression_controls"} for f in executors):
        raise ComposeLipidError("Every fixed family requires source and regression controls")
    return checked


def load_contract(repo: Path, config_path: Path):
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("policy") != POLICY:
        raise ComposeLipidError("Fixed-event replay schema or policy differs")
    if set(config.get("inputs", {})) != {
        "preparation",
        "registry",
        "family_definitions",
        "transform_controls",
    }:
        raise ComposeLipidError("Fixed-event replay input set differs")
    if type(config["maximum_outcomes"]) is not int or config["maximum_outcomes"] < 2:
        raise ComposeLipidError("Fixed-event search bound must be an integer greater than one")
    paths = {k: resolve_pin(v, repo, label=k) for k, v in config["inputs"].items()}
    registry = json.loads(paths["registry"].read_text())
    definitions = json.loads(paths["family_definitions"].read_text())
    document = json.loads(paths["transform_controls"].read_text())
    if (
        document["schema_version"] != "forge.fixed_source_event_adjudication.v1"
        or document["registry"] != config["inputs"]["registry"]
        or set(document["families"]) != set(config["families"])
    ):
        raise ComposeLipidError("Fixed-event control registry or family scope differs")
    for assets in (registry["source_assets"], document["assets"]):
        for name, value in assets.items():
            resolve_pin(value, repo, label=name)
    if any(document["assets"].get(k) != v for k, v in registry["source_assets"].items()):
        raise ComposeLipidError("Fixed-event registry/control source substitution")
    executors = {}
    for family, binding in config["families"].items():
        contract = document["families"][family]
        matches = [r for r in registry["reactions"] if r["reaction_id"] == binding["reaction_id"]]
        if len(matches) != 1 or binding["reaction_id"] != contract["reaction_id"]:
            raise ComposeLipidError("Fixed-event reaction binding differs")
        reaction = matches[0]
        for name, value in reaction["implementation"]["parent_registries"].items():
            resolve_pin(value, repo, label=name)
        resolve_pin(reaction["implementation"]["derivation"], repo, label="derivation")
        roles = binding["registry_to_source_roles"]
        if (
            roles != contract["registry_to_source_roles"]
            or set(roles.values()) != set(definitions[family]["roles"])
            or len(set(roles.values())) != len(roles)
            or definitions[family]["variable"]
            or any(q != 1 for q in definitions[family]["roles"].values())
        ):
            raise ComposeLipidError("Fixed-event complete roles or quantities differ")
        adapter = RegistryAssemblyAdapter.from_registry(
            paths["registry"],
            reaction_id=binding["reaction_id"],
            expected_sha256=config["inputs"]["registry"]["sha256"],
        )
        if set(adapter.roles) != set(roles):
            raise ComposeLipidError("Fixed-event role binding is incomplete")
        queries = {
            r["name"]: Chem.MolFromSmarts(r["required_handle_smarts"])
            for r in reaction["reactant_roles"]
        }
        executors[family] = {
            "kind": "fixed",
            "mapping": roles,
            "run": partial(
                check_retained_condensation,
                adapter,
                site_contract=contract["site_contract"],
                role_queries=queries,
                net_byproducts=reaction["net_byproducts"],
                retained_queries=reaction["retained_queries"],
                maximum_outcomes=config["maximum_outcomes"],
            ),
        }
    controls = qualify_controls(document, executors)
    return config, paths, executors, controls


def run_fixed_replay(repo_root: Path, config_path: Path, output_dir: Path):
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if output.exists() or not output.is_relative_to(repo):
        raise ComposeLipidError("Fixed replay output must be fresh and inside the repository")
    config, paths, executors, controls = load_contract(repo, config_path)
    reader = FullPreparationCorpus(repo, paths["preparation"])
    if reader.result["inputs"]["family_definitions"] != config["inputs"]["family_definitions"]:
        raise ComposeLipidError("Preparation and fixed replay definitions differ")
    structures = {r["component_id"]: r["constitution"] for r in rows(reader.precursors)}
    counts = defaultdict(Counter)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fixed-replay-", dir=output.parent) as temporary:
        stage = Path(temporary)
        ledger = stage / "replay.jsonl.gz"
        with (
            ledger.open("wb") as raw,
            gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as stream,
        ):
            for family, binding in sorted(config["families"].items()):
                for item in reader.iter_preparation_records(family=family):
                    replay = replay_record(item, structures, executors[family])
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
                        reaction_id=binding["reaction_id"],
                        replay=replay,
                        training_admitted=False,
                        experimental_execution_admitted=False,
                    )
                    stream.write((compact(row) + "\n").encode())
                    counts[family]["rows"] += 1
                    counts[family][replay["disposition"]] += 1
                    counts[family].update(
                        "failed_" + k for k, v in replay.get("checks", {}).items() if not v
                    )
                    if counts[family]["rows"] % 500 == 0:
                        print(f'{family}: {counts[family]["rows"]:,} checked', flush=True)
                expected = reader.result["summary"]["by_family"][family][
                    "eligible_for_program_preparation"
                ]
                if counts[family]["rows"] != expected:
                    raise ComposeLipidError("Fixed replay omitted eligible source rows")
                print(f"{family}: {dict(counts[family])}", flush=True)
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "computed_replay_complete_training_unqualified",
            "config": pin(repo, config_path),
            "inputs": config["inputs"],
            "policy": POLICY,
            "implementation": {name: pin(repo, repo / name) for name in IMPLEMENTATION},
            "controls": controls,
            "summary": {
                "by_family": {k: dict(v) for k, v in sorted(counts.items())},
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
