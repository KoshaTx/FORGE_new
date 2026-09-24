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
from forge.assembly.repeated_components import RepeatBounds
from forge.assembly.staged_program import RegistryStagedProgram
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid_full_preparation import FullPreparationCorpus
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import compact, rows

CONFIG_SCHEMA = "forge.compose_lipid_staged_replay_config.v1"
RESULT_SCHEMA = "forge.compose_lipid_staged_replay.v1"
POLICY = {
    "seed": 0,
    "population": "current_full_preparation_eligible_only",
    "selection_before_molecular_record_decoding": True,
    "component_join": "global_component_id",
    "occupancy_source": "declared_source_quantities_expanded_as_identical_component_occurrences",
    "forward_sites": "exhaustive_unfiltered_with_explicit_bounds",
    "stages": "source_order_complete_forward_and_inverse_no_branch_filtering",
    "training_admitted": False,
    "experimental_execution_admitted": False,
    "training_calls": 0,
}
IMPLEMENTATION = (
    "forge/corpus/compose_lipid_staged_replay.py",
    "forge/corpus/compose_lipid_full_preparation.py",
    "forge/assembly/staged_program.py",
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
                raise ComposeLipidError("Duplicate or unconfigured staged-event control")
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
                raise ComposeLipidError(f"Staged control formula differs: {label}")
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
                raise ComposeLipidError(f"Staged control failed: {label}")
            checked[label] = {"kind": kind, "replay": result}
            covered[family].add(kind)
    if any(covered[f] != {"source_controls", "regression_controls"} for f in executors):
        raise ComposeLipidError("Every staged family requires source and regression controls")
    return checked


def load_contract(repo: Path, config_path: Path):
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("policy") != POLICY:
        raise ComposeLipidError("Staged-event replay schema or policy differs")
    if set(config.get("inputs", {})) != {
        "preparation",
        "registry",
        "family_definitions",
        "transform_controls",
    }:
        raise ComposeLipidError("Staged-event replay input set differs")
    paths = {k: resolve_pin(v, repo, label=k) for k, v in config["inputs"].items()}
    registry = json.loads(paths["registry"].read_text())
    definitions = json.loads(paths["family_definitions"].read_text())
    document = json.loads(paths["transform_controls"].read_text())
    if (
        document["schema_version"] != "forge.staged_source_adjudication.v1"
        or document["registry"] != config["inputs"]["registry"]
        or set(document["families"]) != set(config["families"])
    ):
        raise ComposeLipidError("Staged-event control registry or family scope differs")
    for assets in (registry["source_assets"], document["assets"]):
        for name, value in assets.items():
            resolve_pin(value, repo, label=name)
    if any(document["assets"].get(k) != v for k, v in registry["source_assets"].items()):
        raise ComposeLipidError("Staged-event registry/control source substitution")
    executors = {}
    if document["families"] != config["families"]:
        raise ComposeLipidError("Staged source binding differs")
    for family, binding in config["families"].items():
        occurrences = binding["source_role_occurrences"]
        definition = definitions[family]
        if (
            definition["variable"]
            or set(occurrences) != set(definition["roles"])
            or any(
                not isinstance(value, list)
                or not value
                or any(not isinstance(role, str) or not role for role in value)
                or type(definition["roles"][source]) is not int
                or len(value) != definition["roles"][source]
                for source, value in occurrences.items()
            )
        ):
            raise ComposeLipidError("Staged source quantities or roles differ")
        program = RegistryStagedProgram.from_registry(
            paths["registry"],
            program_id=binding["program_id"],
            expected_sha256=config["inputs"]["registry"]["sha256"],
            bounds=RepeatBounds(**config["search_bounds"]),
        )
        terminals = [role for value in occurrences.values() for role in value]
        if len(set(terminals)) != len(terminals) or set(terminals) != set(program.roles):
            raise ComposeLipidError("Staged occurrence roles must cover each terminal exactly once")
        expected_groups = sorted(sorted(value) for value in occurrences.values() if len(value) > 1)
        if (
            sorted(sorted(group) for group in program.specification["equal_component_groups"])
            != expected_groups
        ):
            raise ComposeLipidError("Staged component equality differs from source multiplicity")
        for reaction in registry["reactions"]:
            for name, value in reaction["implementation"]["parent_registries"].items():
                resolve_pin(value, repo, label=name)
            resolve_pin(reaction["implementation"]["derivation"], repo, label="derivation")
        executors[family] = {"program": program, "occurrences": occurrences}
    controls = qualify_controls(document, executors)
    return config, paths, executors, controls


def replay_record(item: dict, structures: dict, executor: dict) -> dict:
    source, prepared = item["source"], item["preparation"]
    if prepared.get("eligible_for_program_preparation") is not True:
        raise ComposeLipidError("Protected or unassigned target reached staged replay")
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
    if set(supplied) != set(executor["occurrences"]):
        return {"computed_consistency_pass": False, "disposition": "unsupported_source_role_tuple"}
    parts = {}
    for role, targets in executor["occurrences"].items():
        identity, quantity = supplied[role]
        if quantity != len(targets):
            return {
                "computed_consistency_pass": False,
                "disposition": "outside_qualified_program_multiplicity",
            }
        for target in targets:
            parts[target] = structures[identity]
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
            raise ComposeLipidError("Staged target differs from authenticated TRAIN identity")
        if not result["checks"] or not all(v is True for v in result["checks"].values()):
            raise ComposeLipidError("Staged exact result lacks complete passing checks")
        result["verified_target_constitution_id"] = digest
        result["disposition"] = "exact_computed_reconstruction"
    else:
        result["disposition"] = "unresolved_computed_reconstruction"
    return result


def run_staged_replay(repo_root: Path, config_path: Path, output_dir: Path):
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if output.exists() or not output.is_relative_to(repo):
        raise ComposeLipidError("Staged replay output must be fresh and inside the repository")
    config, paths, executors, controls = load_contract(repo, config_path)
    reader = FullPreparationCorpus(repo, paths["preparation"])
    if reader.result["inputs"]["family_definitions"] != config["inputs"]["family_definitions"]:
        raise ComposeLipidError("Preparation and staged replay definitions differ")
    structures = {r["component_id"]: r["constitution"] for r in rows(reader.precursors)}
    counts = defaultdict(Counter)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".staged-replay-", dir=output.parent) as temporary:
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
                    raise ComposeLipidError("Staged replay omitted eligible source rows")
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
