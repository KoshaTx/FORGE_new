"""Replay frozen repeated transforms with supplied role-aware precursor instances.

An exact match is a computed consistency check, not evidence of an experimental
route. No product-based choice of precursor, occupancy or forward site is allowed.
"""

from __future__ import annotations

import gzip
import json
import os
import platform
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import rdBase

from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.families import LibraryAssemblyError, RegistryAssemblyAdapter
from forge.assembly.repeated_components import RepeatBounds, replay_repeated_components
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid_source_view import SourcePreparationCorpus, dump, pin
from forge.corpus.compose_lipid_supplement import compact, rows

CONFIG_SCHEMA = "forge.compose_lipid_supplied_replay_config.v1"
RESULT_SCHEMA = "forge.compose_lipid_supplied_replay.v1"
POLICY = {
    "seed": 0,
    "fit_population": "intersection_old_and_corrected_train_with_product_component_exclusions",
    "component_join": "global_component_id",
    "occupancy_source": "supplied_instance_quantities_checked_against_metadata_when_present",
    "forward_sites": "exhaustive_unfiltered_with_explicit_bounds",
    "scope": "computed_reconstruction_only",
    "training_admitted": False,
    "experimental_execution_admitted": False,
    "training_calls": 0,
}
IMPLEMENTATION = (
    "forge/corpus/compose_lipid_supplied_replay.py",
    "forge/assembly/repeated_components.py",
    "forge/assembly/families.py",
    "forge/assembly/registry.py",
    "forge/assembly/program.py",
    "forge/chemistry/reactive_sites.py",
    "forge/corpus/compose_lipid_source_view.py",
    "forge/corpus/compose_lipid_supplement.py",
    "forge/core/hashing.py",
)


def resolve_components(instances, structures, role_binding, program, metadata):
    """Resolve an explicit complete tuple, preserving every quantity and role."""
    if len(set(role_binding.values())) != len(role_binding):
        raise ComposeLipidError("Source roles cannot be collapsed onto each other")
    by_role = {}
    for role, identity, quantity in instances:
        if identity not in structures:
            raise ComposeLipidError(f"Unknown global component_id: {identity}")
        if type(quantity) is not int or quantity < 1 or role in by_role:
            raise ComposeLipidError(
                "Repeated program requires one positive-quantity identity per role"
            )
        by_role[role] = (structures[identity], quantity)
    if set(by_role) != set(role_binding.values()):
        raise ComposeLipidError("Source instance roles differ from the complete reaction binding")
    accumulator = program["accumulator_role"]
    if by_role[role_binding[accumulator]][1] != 1:
        raise ComposeLipidError("Source accumulator quantity must be exactly one")
    repeated = set(program["repeated_roles"])
    if repeated != set(role_binding) - {accumulator} or not repeated:
        raise ComposeLipidError("Registry repeated roles do not cover the complete tuple")
    quantities = {by_role[role_binding[role]][1] for role in repeated}
    if len(quantities) != 1:
        raise ComposeLipidError("Source repeated roles have inconsistent quantities")
    events = quantities.pop()
    declared = metadata.get(program["event_count_field"])
    if declared is not None and (type(declared) is not int or declared != events):
        raise ComposeLipidError("Supplied quantities disagree with source event metadata")
    return {role: by_role[source][0] for role, source in role_binding.items()}, events


def replay_supplied(adapter, item, structures, binding, program, bounds):
    source, preparation = item["source"], item["preparation"]
    if not preparation["eligible_for_program_preparation"]:
        raise ComposeLipidError("Protected target reached the reaction executor")
    components, events = resolve_components(
        preparation["component_instances"],
        structures,
        binding["registry_to_source_roles"],
        program,
        source.get("primary_metadata", {}),
    )
    if events > program["maximum_events"]:
        return {
            "computed_consistency_pass": False,
            "disposition": "outside_registry_event_support",
            "events": events,
        }
    replay = replay_repeated_components(
        adapter,
        components,
        source["constitution"],
        accumulator_role=program["accumulator_role"],
        events=events,
        byproducts_per_event=program["net_byproducts_per_event"],
        bounds=bounds,
    )
    replay["events"] = events
    replay["disposition"] = (
        "exact_computed_reconstruction"
        if replay["computed_consistency_pass"]
        else "unresolved_computed_reconstruction"
    )
    return replay


def _load(repo: Path, config_path: Path):
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("policy") != POLICY:
        raise ComposeLipidError("Supplied replay schema or scientific scope changed")
    paths = {name: resolve_pin(value, repo, label=name) for name, value in config["inputs"].items()}
    if set(paths) != {"preparation", "registry", "family_definitions", "transform_controls"}:
        raise ComposeLipidError("Supplied replay input set changed")
    registry = json.loads(paths["registry"].read_text())
    for name, value in registry["source_assets"].items():
        resolve_pin(value, repo, label=f"registry.{name}")
    definitions = json.loads(paths["family_definitions"].read_text())
    bounds = RepeatBounds(**config["search_bounds"])
    adapters, programs = {}, {}
    for family, binding in config["families"].items():
        reaction = next(
            r for r in registry["reactions"] if r["reaction_id"] == binding["reaction_id"]
        )
        for name in ("parent_registry", "functional_group_registry"):
            resolve_pin(reaction["implementation"][name], repo, label=name)
        program = reaction["source_program"]
        roles = binding["registry_to_source_roles"]
        if (
            set(roles.values()) != set(definitions[family]["roles"])
            or len(set(roles.values())) != len(roles)
            or program["event_count_field"] not in definitions[family]["variable"]
            or bounds.maximum_events < program["maximum_events"]
        ):
            raise ComposeLipidError("Source family roles or declared event support changed")
        adapter = RegistryAssemblyAdapter.from_registry(
            paths["registry"],
            reaction_id=binding["reaction_id"],
            expected_sha256=config["inputs"]["registry"]["sha256"],
        )
        if set(adapter.roles) != set(roles):
            raise ComposeLipidError("Bound source roles do not cover registry reactants")
        adapters[family], programs[family] = adapter, program
    source = json.loads(paths["transform_controls"].read_text())
    if source["registry"] != config["inputs"]["registry"]:
        raise ComposeLipidError("Transform controls are for a different registry")
    for name, value in source["assets"].items():
        resolve_pin(value, repo, label=f"controls.{name}")
    controls = {}
    for kind in ("source_controls", "ambiguity_controls"):
        for control in source[kind]:
            family = control["family"]
            if family not in adapters:
                raise ComposeLipidError("Configuration omits a required transform-control family")
            program = programs[family]
            replay = replay_repeated_components(
                adapters[family],
                control["components"],
                control["expected_product"],
                accumulator_role=program["accumulator_role"],
                events=control["events"],
                byproducts_per_event=program["net_byproducts_per_event"],
                bounds=bounds,
            )
            expected = kind == "source_controls"
            if (
                replay["computed_consistency_pass"] != expected
                or not replay["checks"]["complete_search"]
            ):
                raise ComposeLipidError(f"Frozen transform control failed: {control['label']}")
            if not expected and (
                len(replay["forward_layers"][-1]) != control["expected_forward_product_count"]
                or not replay["checks"]["full_element_hydrogen_charge_balance"]
            ):
                raise ComposeLipidError("Ambiguity control did not preserve competing outcomes")
            controls[control["label"]] = {"kind": kind, "replay": replay}
    if not controls or not source["source_controls"] or not source["ambiguity_controls"]:
        raise ComposeLipidError("Positive and ambiguity transform controls are mandatory")
    reader = SourcePreparationCorpus(repo, paths["preparation"])
    structures = {row["component_id"]: row["constitution"] for row in rows(reader.precursors)}
    return config, reader, structures, adapters, programs, bounds, controls


def run_supplied_replay(repo_root: Path, config_path: Path, output_dir: Path) -> dict:
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if output.exists() or not output.is_relative_to(repo):
        raise ComposeLipidError("Supplied replay output must be fresh and inside the repository")
    config, reader, structures, adapters, programs, bounds, controls = _load(repo, config_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(Counter)
    with tempfile.TemporaryDirectory(prefix=".supplied-replay-", dir=output.parent) as temporary:
        stage = Path(temporary)
        ledger = stage / "replay.jsonl.gz"
        with ledger.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as stream:
                for family, binding in config["families"].items():
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
                            "failed_" + check
                            for check, passed in replay.get("checks", {}).items()
                            if not passed
                        )
                        row = {
                            "target_id": item["source"]["target_id"],
                            "family": family,
                            "constitution_id": item["preparation"]["constitution_id"],
                            "component_instances": item["preparation"]["component_instances"],
                            "construction_basis": item["preparation"]["construction_basis"],
                            "reaction_id": binding["reaction_id"],
                            "replay": replay,
                            "training_admitted": False,
                            "experimental_execution_admitted": False,
                        }
                        stream.write((compact(row) + "\n").encode())
                        if counts[family]["rows"] % 500 == 0:
                            print(f"{family}: {counts[family]['rows']:,} checked", flush=True)
                    expected = reader.result["summary"]["by_family"][family][
                        "eligible_for_program_preparation"
                    ]
                    if counts[family]["rows"] != expected:
                        raise ComposeLipidError("Replay did not cover every eligible source row")
                    print(f"{family}: {dict(counts[family])}", flush=True)
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "computed_replay_complete_training_unqualified",
            "config": pin(repo, config_path),
            "inputs": config["inputs"],
            "implementation": {name: pin(repo, repo / name) for name in IMPLEMENTATION},
            "policy": POLICY,
            "controls": controls,
            "summary": {
                "by_family": {family: dict(count) for family, count in counts.items()},
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
