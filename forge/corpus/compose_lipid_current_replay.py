"""Replay qualified supplied programs exclusively through current full-universe protection.

Historical preparation receipts are never substituted for the current protected reader.
Controls, complete role bindings and quantities are checked before any corpus reaction.
"""

from __future__ import annotations

import gzip
import json
import os
import platform
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.families import LibraryAssemblyError, RegistryAssemblyAdapter
from forge.assembly.repeated_components import RepeatBounds, replay_repeated_components
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid_full_preparation import FullPreparationCorpus
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import compact, rows
from forge.corpus.compose_lipid_supplied_replay import replay_supplied

CONFIG_SCHEMA = "forge.compose_lipid_current_replay_config.v1"
RESULT_SCHEMA = "forge.compose_lipid_current_replay.v1"
POLICY = {
    "seed": 0,
    "population": "current_full_preparation_eligible_only",
    "selection_before_molecular_record_decoding": True,
    "occupancy_source": "supplied_instance_quantities_checked_against_metadata_when_present",
    "component_join": "global_component_id",
    "forward_sites": "exhaustive_unfiltered_with_explicit_bounds",
    "training_admitted": False,
    "experimental_execution_admitted": False,
    "training_calls": 0,
}
IMPLEMENTATION = (
    "forge/corpus/compose_lipid_current_replay.py",
    "forge/corpus/compose_lipid_full_preparation.py",
    "forge/corpus/compose_lipid_supplied_replay.py",
    "forge/assembly/repeated_components.py",
    "forge/assembly/families.py",
    "forge/assembly/registry.py",
    "forge/assembly/program.py",
    "forge/chemistry/reactive_sites.py",
    "forge/core/hashing.py",
)


def qualify_controls(document, adapters, programs, bounds):
    """Require positive and explicit ambiguity controls for every configured family."""
    controls, covered = {}, defaultdict(set)
    for kind in ("source_controls", "ambiguity_controls"):
        for control in document[kind]:
            label, family = control["label"], control["family"]
            if label in controls or family not in adapters:
                raise ComposeLipidError("Duplicate control or unconfigured control family")
            adapter, program = adapters[family], programs[family]
            if control["reaction_id"] != adapter.reaction_id:
                raise ComposeLipidError("Control reaction binding differs")
            molecule = Chem.MolFromSmiles(control["expected_product"])
            if (
                molecule is None
                or rdMolDescriptors.CalcMolFormula(molecule) != control["expected_formula"]
            ):
                raise ComposeLipidError(f"Source control formula differs: {label}")
            replay = replay_repeated_components(
                adapter,
                control["components"],
                control["expected_product"],
                accumulator_role=program["accumulator_role"],
                events=control["events"],
                byproducts_per_event=program["net_byproducts_per_event"],
                bounds=bounds,
            )
            expected = kind == "source_controls"
            if (
                control["expected_computed_consistency_pass"] is not expected
                or replay["computed_consistency_pass"] is not expected
                or not replay["checks"]["complete_search"]
                or not replay["checks"]["full_element_hydrogen_charge_balance"]
                or len(replay["forward_layers"][-1]) != control["expected_forward_product_count"]
                or (not expected and control["expected_forward_product_count"] < 2)
            ):
                raise ComposeLipidError(f"Frozen transform control failed: {label}")
            controls[label] = {"kind": kind, "replay": replay}
            covered[family].add(kind)
    if any(covered[f] != {"source_controls", "ambiguity_controls"} for f in adapters):
        raise ComposeLipidError("Every family requires positive and ambiguity controls")
    return controls


def load_contract(repo: Path, config_path: Path):
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("policy") != POLICY:
        raise ComposeLipidError("Current replay schema or scientific policy changed")
    if set(config.get("inputs", {})) != {
        "preparation",
        "registry",
        "family_definitions",
        "transform_controls",
    }:
        raise ComposeLipidError("Current replay input set differs")
    paths = {k: resolve_pin(v, repo, label=k) for k, v in config["inputs"].items()}
    registry = json.loads(paths["registry"].read_text())
    definitions = json.loads(paths["family_definitions"].read_text())
    for name, value in registry["source_assets"].items():
        resolve_pin(value, repo, label=name)
    bounds = RepeatBounds(**config["search_bounds"])
    adapters, programs = {}, {}
    for family, binding in config["families"].items():
        matches = [r for r in registry["reactions"] if r["reaction_id"] == binding["reaction_id"]]
        if len(matches) != 1:
            raise ComposeLipidError("Reaction must resolve exactly once")
        reaction = matches[0]
        for name, value in reaction["implementation"]["parent_registries"].items():
            resolve_pin(value, repo, label=name)
        resolve_pin(reaction["implementation"]["derivation"], repo, label="derivation")
        program, roles = reaction["source_program"], binding["registry_to_source_roles"]
        if (
            set(roles.values()) != set(definitions[family]["roles"])
            or len(set(roles.values())) != len(roles)
            or program["event_count_field"] not in definitions[family]["variable"]
            or bounds.maximum_events < program["maximum_events"]
        ):
            raise ComposeLipidError("Source roles or event support differs")
        adapter = RegistryAssemblyAdapter.from_registry(
            paths["registry"],
            reaction_id=binding["reaction_id"],
            expected_sha256=config["inputs"]["registry"]["sha256"],
        )
        if set(adapter.roles) != set(roles):
            raise ComposeLipidError("Incomplete source role binding")
        adapters[family], programs[family] = adapter, program
    document = json.loads(paths["transform_controls"].read_text())
    if document["registry"] != config["inputs"]["registry"]:
        raise ComposeLipidError("Control registry substitution")
    for name, value in document["assets"].items():
        resolve_pin(value, repo, label=name)
    controls = qualify_controls(document, adapters, programs, bounds)
    return config, paths, adapters, programs, bounds, controls


def run_current_replay(repo_root: Path, config_path: Path, output_dir: Path) -> dict:
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if output.exists() or not output.is_relative_to(repo):
        raise ComposeLipidError("Current replay output must be fresh and inside the repository")
    config, paths, adapters, programs, bounds, controls = load_contract(repo, config_path)
    reader = FullPreparationCorpus(repo, paths["preparation"])
    if reader.result["inputs"]["family_definitions"] != config["inputs"]["family_definitions"]:
        raise ComposeLipidError("Preparation and replay family definitions differ")
    structures = {row["component_id"]: row["constitution"] for row in rows(reader.precursors)}
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(Counter)
    with tempfile.TemporaryDirectory(prefix=".current-replay-", dir=output.parent) as temporary:
        stage = Path(temporary)
        ledger = stage / "replay.jsonl.gz"
        with ledger.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as stream:
                for family, binding in sorted(config["families"].items()):
                    for item in reader.iter_preparation_records(family=family):
                        try:
                            replay = replay_supplied(
                                adapters[family],
                                item,
                                structures,
                                binding,
                                programs[family],
                                bounds,
                            )
                        except LibraryAssemblyError as exc:
                            replay = {
                                "computed_consistency_pass": False,
                                "disposition": "unsupported_registry_replay",
                                "reason": str(exc),
                            }
                        counts[family]["rows"] += 1
                        counts[family][replay["disposition"]] += 1
                        counts[family].update(
                            "failed_" + k for k, v in replay.get("checks", {}).items() if not v
                        )
                        prepared = item["preparation"]
                        stream.write(
                            (
                                compact(
                                    {
                                        "target_id": item["source"]["target_id"],
                                        "family": family,
                                        "constitution_id": prepared["constitution_id"],
                                        "component_instances": prepared["component_instances"],
                                        "construction_basis": prepared["construction_basis"],
                                        "reaction_id": binding["reaction_id"],
                                        "replay": replay,
                                        "training_admitted": False,
                                        "experimental_execution_admitted": False,
                                    }
                                )
                                + "\n"
                            ).encode()
                        )
                        if counts[family]["rows"] % 500 == 0:
                            print(f"{family}: {counts[family]['rows']:,} checked", flush=True)
                    expected = reader.result["summary"]["by_family"][family][
                        "eligible_for_program_preparation"
                    ]
                    if counts[family]["rows"] != expected:
                        raise ComposeLipidError("Replay omitted eligible source rows")
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
