"""Replay source-qualified multi-component stages on current protected TRAIN rows."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.families import LibraryAssemblyError, constitutional_molecule
from forge.assembly.grouped_program import RegistryGroupedProgram
from forge.assembly.repeated_components import RepeatBounds
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid_full_preparation import FullPreparationCorpus
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import compact, rows

CONFIG_SCHEMA = "forge.compose_lipid_grouped_replay_config.v1"
RESULT_SCHEMA = "forge.compose_lipid_grouped_replay.v1"
POLICY = {
    "seed": 0,
    "population": "current_full_preparation_eligible_only",
    "selection_before_molecular_record_decoding": True,
    "component_join": "global_component_id",
    "occupancy_source": "declared_source_quantities_equal_documented_stage_event_counts",
    "forward_sites": "exhaustive_unfiltered_with_explicit_bounds",
    "stages": "source_order_grouped_events_complete_forward_and_inverse_no_target_pruning",
    "training_admitted": False,
    "experimental_execution_admitted": False,
    "training_calls": 0,
}
IMPLEMENTATION = (
    "forge/corpus/compose_lipid_grouped_replay.py",
    "forge/corpus/compose_lipid_full_preparation.py",
    "forge/assembly/grouped_program.py",
    "forge/assembly/staged_program.py",
    "forge/assembly/repeated_inverse.py",
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
                raise ComposeLipidError("Duplicate or unconfigured grouped-event control")
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
                raise ComposeLipidError(f"Grouped control formula differs: {label}")
            result = executors[family]["program"].replay(
                control["components"], control["expected_product"]
            )
            if (
                result["computed_consistency_pass"] is not expected
                or list(map(len, result["forward_layers"])) != control["expected_stage_widths"]
                or any(
                    result["checks"].get(name) is not False
                    for name in control.get("required_failed_checks", [])
                )
            ):
                raise ComposeLipidError(f"Grouped control failed: {label}")
            if kind == "source_controls":
                stages = control.get("expected_stage_products", [])
                if len(stages) != len(executors[family]["program"].adapters) or result[
                    "forward_layers"
                ][1:] != [[constitutional_molecule(s)[0]] for s in stages]:
                    raise ComposeLipidError(f"Source stage intermediate differs: {label}")
            checked[label] = {"kind": kind, "replay": result}
            covered[family].add(kind)
    if any(covered[f] != {"source_controls", "regression_controls"} for f in executors):
        raise ComposeLipidError("Every grouped family requires source and regression controls")
    return checked


def load_contract(repo: Path, config_path: Path):
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("policy") != POLICY:
        raise ComposeLipidError("Grouped-event replay schema or policy differs")
    if set(config.get("inputs", {})) != {
        "preparation",
        "registry",
        "family_definitions",
        "transform_controls",
    }:
        raise ComposeLipidError("Grouped-event replay input set differs")
    paths = {k: resolve_pin(v, repo, label=k) for k, v in config["inputs"].items()}
    registry = json.loads(paths["registry"].read_text())
    definitions = json.loads(paths["family_definitions"].read_text())
    document = json.loads(paths["transform_controls"].read_text())
    if (
        document["schema_version"] != "forge.grouped_source_adjudication.v1"
        or document["registry"] != config["inputs"]["registry"]
        or set(document["families"]) != set(config["families"])
    ):
        raise ComposeLipidError("Grouped-event control registry or family scope differs")
    for assets in (registry["source_assets"], document["assets"]):
        for name, value in assets.items():
            resolve_pin(value, repo, label=name)
    if any(document["assets"].get(k) != v for k, v in registry["source_assets"].items()):
        raise ComposeLipidError("Grouped-event registry/control source substitution")
    executors = {}
    if document["families"] != config["families"]:
        raise ComposeLipidError("Grouped source binding differs")
    for family, binding in config["families"].items():
        mapping = binding["registry_to_source_roles"]
        definition = definitions[family]
        program = RegistryGroupedProgram.from_registry(
            paths["registry"],
            program_id=binding["program_id"],
            expected_sha256=config["inputs"]["registry"]["sha256"],
            bounds=RepeatBounds(**config["search_bounds"]),
        )
        if (
            definition["variable"]
            or any(type(q) is not int or q < 1 for q in definition["roles"].values())
            or set(mapping) != set(program.roles)
            or len(set(mapping.values())) != len(mapping)
            or {mapping[r]: q for r, q in program.quantities.items()} != definition["roles"]
        ):
            raise ComposeLipidError("Grouped roles or event quantities differ from the source")
        for reaction in registry["reactions"]:
            for name, value in reaction["implementation"]["parent_registries"].items():
                resolve_pin(value, repo, label=name)
            resolve_pin(reaction["implementation"]["derivation"], repo, label="derivation")
        executors[family] = {"program": program, "mapping": mapping}
    controls = qualify_controls(document, executors)
    return config, paths, executors, controls


def replay_record(item: dict, structures: dict, executor: dict) -> dict:
    source, prepared = item["source"], item["preparation"]
    if prepared.get("eligible_for_program_preparation") is not True:
        raise ComposeLipidError("Protected or unassigned target reached grouped replay")
    supplied = {}
    for role, identity, quantity in prepared["component_instances"]:
        if identity not in structures:
            raise ComposeLipidError(f"Unresolved supplied precursor: {identity}")
        if type(quantity) is not int or quantity < 1:
            raise ComposeLipidError("Invalid supplied precursor quantity")
        if role in supplied:
            return {
                "computed_consistency_pass": False,
                "disposition": "unsupported_source_role_tuple",
            }
        supplied[role] = (identity, quantity)
    if set(supplied) != set(executor["mapping"].values()):
        return {"computed_consistency_pass": False, "disposition": "unsupported_source_role_tuple"}
    parts = {}
    for role, source_role in executor["mapping"].items():
        identity, quantity = supplied[source_role]
        if quantity != executor["program"].quantities[role]:
            return {
                "computed_consistency_pass": False,
                "disposition": "outside_qualified_program_multiplicity",
            }
        parts[role] = structures[identity]
    try:
        result = executor["program"].replay(parts, source["constitution"])
    except LibraryAssemblyError as exc:
        return {
            "computed_consistency_pass": False,
            "disposition": "unsupported_registry_replay",
            "reason": str(exc),
        }
    if result["computed_consistency_pass"]:
        digest = hashlib.sha256(
            constitutional_molecule(source["constitution"])[0].encode()
        ).hexdigest()
        if digest != prepared["constitution_id"]:
            raise ComposeLipidError("Grouped target differs from authenticated TRAIN identity")
        if not result["checks"] or not all(v is True for v in result["checks"].values()):
            raise ComposeLipidError("Grouped exact result lacks complete passing checks")
        result["verified_target_constitution_id"] = digest
        result["disposition"] = "exact_computed_reconstruction"
    else:
        result["disposition"] = "unresolved_computed_reconstruction"
    return result


def run_grouped_replay(repo_root: Path, config_path: Path, output_dir: Path):
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if output.exists() or not output.is_relative_to(repo):
        raise ComposeLipidError("Grouped replay output must be fresh and inside the repository")
    config, paths, executors, controls = load_contract(repo, config_path)
    reader = FullPreparationCorpus(repo, paths["preparation"])
    if reader.result["inputs"]["family_definitions"] != config["inputs"]["family_definitions"]:
        raise ComposeLipidError("Preparation and grouped replay definitions differ")
    structures = {r["component_id"]: r["constitution"] for r in rows(reader.precursors)}
    counts = defaultdict(Counter)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".grouped-replay-", dir=output.parent) as temporary:
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
                        program_id=binding["program_id"],
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
                    raise ComposeLipidError("Grouped replay omitted eligible source rows")
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
