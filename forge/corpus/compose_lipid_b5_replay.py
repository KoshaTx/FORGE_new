"""Replay authenticated eligible B5 recipes through separately qualified source profiles."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.families import LibraryAssemblyError, constitutional_molecule
from forge.assembly.repeated_components import RepeatBounds
from forge.assembly.staged_program import RegistryStagedProgram
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid_b5_profiles import B5SourceProfiles
from forge.corpus.compose_lipid_full_preparation import FullPreparationCorpus
from forge.corpus.compose_lipid_original_binding import authenticate_reference, selected_lines
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import compact, rows

CONFIG_SCHEMA = "forge.compose_lipid_b5_replay_config.v1"
RESULT_SCHEMA = "forge.compose_lipid_b5_replay.v1"
POLICY = {
    "seed": 0,
    "population": "current_full_preparation_eligible_only",
    "selection_before_molecular_record_decoding": True,
    "component_join": "global_component_id_roles_and_unit_quantities_preserved",
    "profile_selection": "original_input_metadata_and_complete_precursor_scope_no_target",
    "forward_sites": "exhaustive_unfiltered_with_explicit_bounds",
    "stages": "source_documented_net_composites_complete_forward_and_inverse",
    "wrong_reported_source_labels": "abstain_even_if_generic_transform_matches",
    "training_admitted": False,
    "experimental_execution_admitted": False,
    "training_calls": 0,
}
IMPLEMENTATION = (
    "forge/corpus/compose_lipid_b5_replay.py",
    "forge/corpus/compose_lipid_b5_profiles.py",
    "forge/assembly/component_scope.py",
    "forge/corpus/compose_lipid_original_binding.py",
    "forge/corpus/compose_lipid_full_preparation.py",
    "forge/assembly/staged_program.py",
    "forge/assembly/repeated_inverse.py",
    "forge/assembly/repeated_components.py",
    "forge/assembly/families.py",
    "forge/assembly/registry.py",
    "forge/assembly/program.py",
    "forge/chemistry/reactive_sites.py",
    "forge/core/hashing.py",
)


def qualify_controls(document, registry, profiles, programs):
    checked, covered = {}, set()
    for control in document["controls"]:
        label, profile = control["label"], control["profile"]
        if label in checked:
            raise ComposeLipidError("Duplicate B5 source control")
        mapping = registry["profile_component_scopes"][profile]
        if set(mapping) != set(control["components"]):
            raise ComposeLipidError("Incomplete B5 source-control components")
        scoped = {
            r: profiles.scopes.assess(s, control["components"][r]) for r, s in mapping.items()
        }
        program_id = "source_b5_" + profile
        result = programs[program_id].replay(control["components"], control["expected_product"])
        mol = Chem.MolFromSmiles(control["expected_product"])
        if (
            not all(r["pass"] and r["complete_search"] for r in scoped.values())
            or not result["computed_consistency_pass"]
            or mol is None
            or rdMolDescriptors.CalcMolFormula(mol) != control["expected_formula"]
            or list(map(len, result["forward_layers"])) != control["expected_stage_widths"]
            or result["forward_layers"][1:]
            != [[constitutional_molecule(s)[0]] for s in control["expected_stage_products"]]
        ):
            raise ComposeLipidError(f"Independent B5 source control failed: {label}")
        checked[label] = {"component_scope": scoped, "replay": result}
        covered.add(program_id)
    if covered != set(programs):
        raise ComposeLipidError("A B5 source program lacks independent positive controls")
    return checked


def load_contract(repo: Path, config_path: Path):
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("policy") != POLICY:
        raise ComposeLipidError("B5 replay schema or policy changed")
    if set(config.get("inputs", {})) != {
        "preparation",
        "registry",
        "family_definitions",
        "source_controls",
        "original_tasks",
    }:
        raise ComposeLipidError("B5 replay input set changed")
    paths = {k: resolve_pin(p, repo, label=k) for k, p in config["inputs"].items()}
    registry = json.loads(paths["registry"].read_text())
    profiles = B5SourceProfiles.from_registry(
        paths["registry"], expected_sha256=config["inputs"]["registry"]["sha256"]
    )
    document = json.loads(paths["source_controls"].read_text())
    for group in (
        registry["source_assets"],
        document["assets"],
        registry["complete_scope_curation"]["inputs"],
    ):
        for name, value in group.items():
            resolve_pin(value, repo, label=name)
    resolve_pin(document["implementation"], repo, label="control transcriptions")
    resolve_pin(registry["complete_scope_curation"]["derivation"], repo, label="scope derivation")
    if (
        registry["source_assets"]["control_transcriptions"] != config["inputs"]["source_controls"]
        or registry["complete_scope_curation"]["inputs"]["family_definitions"]
        != config["inputs"]["family_definitions"]
    ):
        raise ComposeLipidError("B5 source-control or formal family registry substitution")
    definitions = json.loads(paths["family_definitions"].read_text())
    definition = definitions[profiles.specification["family"]]
    if (
        definition["roles"] != {"series_head": 1, "series_tail_design": 1}
        or definition["variable"]
        or definition["field_to_role"]["head_id"] != ["series_head"]
    ):
        raise ComposeLipidError("B5 abstract formal role contract changed")
    for reaction in registry["reactions"]:
        for name, value in reaction["implementation"]["parent_registries"].items():
            resolve_pin(value, repo, label=name)
        resolve_pin(reaction["implementation"]["derivation"], repo, label="source reaction overlay")
    programs = {
        p["program_id"]: RegistryStagedProgram.from_registry(
            paths["registry"],
            program_id=p["program_id"],
            expected_sha256=config["inputs"]["registry"]["sha256"],
            bounds=RepeatBounds(**config["search_bounds"]),
        )
        for p in registry["programs"]
    }
    controls = qualify_controls(document, registry, profiles, programs)
    return config, paths, profiles, programs, controls


def authenticated_originals(repo, result_path, config, prepared):
    report = json.loads(result_path.read_text())
    if (
        report.get("schema_version") != "forge.b5_eligible_original_task_extraction.v1"
        or report["inputs"]["preparation"] != config["inputs"]["preparation"]
        or report["selection_before_molecular_record_decoding"] is not True
    ):
        raise ComposeLipidError("B5 original-task population or extraction contract differs")
    for group in ("inputs", "implementation", "task_shards"):
        for name, value in report[group].items():
            resolve_pin(value, repo, label=name)
    path = resolve_pin(report["artifact"], repo, label="eligible original B5 task ledger")
    selected = {}
    requests = defaultdict(lambda: defaultdict(list))
    for row in rows(path):
        target = row["preparation"]["target_id"]
        if target not in prepared or row["preparation"] != prepared[target] or target in selected:
            raise ComposeLipidError(
                "B5 original-task ledger contains noneligible or changed preparation"
            )
        if row["disposition"] != "original_task_bytes_authenticated":
            raise ComposeLipidError("B5 original-task bytes unavailable")
        ref = row["construction"]["provenance"]["task"]
        if row["original_task_file"] != report["task_shards"][ref["path"]]:
            raise ComposeLipidError("B5 original-task shard substitution")
        requests[ref["path"]][ref["line"]].append(target)
        selected[target] = row
    if set(selected) != set(prepared) or len(selected) != report["rows"]:
        raise ComposeLipidError("B5 original-task extraction lost eligible records")
    for source, lines in sorted(requests.items()):
        f = report["task_shards"][source]
        for number, original, digest in selected_lines(
            resolve_pin(f, repo, label=source), set(lines)
        ):
            for target in lines[number]:
                row = selected[target]
                authenticate_reference(
                    row["construction"]["provenance"]["task"],
                    {"source": source, "sha256": f["sha256"]},
                    number,
                    digest,
                )
                if original != row["original_task"]:
                    raise ComposeLipidError("Extracted B5 task differs from original task bytes")
    by_line = {r["construction_source_line"]: t for t, r in prepared.items()}
    source = resolve_pin(report["inputs"]["constructions"], repo, label="original constructions")
    for number, construction, digest in selected_lines(source, set(by_line)):
        row = selected[by_line[number]]
        if construction != row["construction"] or digest != row["construction_payload_sha256"]:
            raise ComposeLipidError("B5 extracted construction differs from original bytes")
    return selected


def replay_record(item, original, structures, profiles, programs):
    prepared = item["preparation"]
    if prepared.get("eligible_for_program_preparation") is not True:
        raise ComposeLipidError("Protected or unassigned target reached B5 replay")
    if (
        prepared != original["preparation"]
        or item["source"].get("primary_metadata", {}) != original["primary_metadata"]
    ):
        raise ComposeLipidError("B5 recipe preparation or source metadata substitution")
    binding = profiles.assess(
        prepared,
        original["construction"],
        original["primary_metadata"],
        original["original_task"],
        structures,
    )
    result = {
        "binding": binding,
        "computed_consistency_pass": False,
        "disposition": binding["disposition"],
        "experimental_execution_admitted": False,
    }
    if not binding["profile_qualified"]:
        return result
    try:
        replay = programs[binding["program_id"]].replay(
            binding["components"], item["source"]["constitution"]
        )
    except LibraryAssemblyError as exc:
        result.update(disposition="unsupported_registry_replay", reason=str(exc))
        return result
    result["replay"] = replay
    result["computed_consistency_pass"] = replay["computed_consistency_pass"]
    result["disposition"] = (
        "exact_source_profile_transform_consistency"
        if result["computed_consistency_pass"]
        else "staged_program_not_exact_unique"
    )
    if result["computed_consistency_pass"]:
        digest = hashlib.sha256(
            constitutional_molecule(item["source"]["constitution"])[0].encode()
        ).hexdigest()
        if digest != prepared["constitution_id"]:
            raise ComposeLipidError(
                "B5 replay target differs from authenticated preparation identity"
            )
        result["verified_target_constitution_id"] = digest
    return result


def run_b5_replay(repo_root: Path, config_path: Path, output_dir: Path):
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if output.exists() or not output.is_relative_to(repo):
        raise ComposeLipidError("B5 replay output must be fresh and inside the repository")
    start = time.monotonic()
    config, paths, profiles, programs, controls = load_contract(repo, config_path)
    implementation = {name: pin(repo, repo / name) for name in IMPLEMENTATION}
    reader = FullPreparationCorpus(repo, paths["preparation"])
    family = profiles.specification["family"]
    prepared = {t: r for t, r in reader.preparation.items() if r["family"] == family}
    originals = authenticated_originals(repo, paths["original_tasks"], config, prepared)
    structures = {r["component_id"]: r["constitution"] for r in rows(reader.precursors)}
    inputs = {**config["inputs"], "precursors": pin(repo, reader.precursors)}
    print(
        f"B5 protected preparation and {len(prepared):,} original tasks authenticated", flush=True
    )
    counts, lanes, failures = Counter(), defaultdict(Counter), Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".b5-replay-", dir=output.parent) as tmp:
        stage = Path(tmp)
        ledger = stage / "replay.jsonl.gz"
        with (
            ledger.open("wb") as raw,
            gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream,
        ):
            for number, item in enumerate(reader.iter_preparation_records(family=family), 1):
                p = item["preparation"]
                original = originals[p["target_id"]]
                result = replay_record(item, original, structures, profiles, programs)
                record = {
                    k: p[k]
                    for k in (
                        "target_id",
                        "family",
                        "constitution_id",
                        "component_instances",
                        "construction_basis",
                    )
                }
                record.update(
                    result=result,
                    training_admitted=False,
                    evidence_basis="computed_transform_consistency",
                    original_task_reference=original["construction"]["provenance"]["task"],
                )
                stream.write((compact(record) + "\n").encode())
                lane = original["primary_metadata"]["design_lane"]
                for counter in (counts, lanes[lane]):
                    counter["rows"] += 1
                    counter["profile_qualified"] += result["binding"]["profile_qualified"]
                    counter["exact"] += result["computed_consistency_pass"]
                    counter[result["disposition"]] += 1
                failures.update(k for k, v in result["binding"]["checks"].items() if not v)
                if number % 250 == 0:
                    print(
                        f"B5 replay {number:,}/{len(prepared):,}; exact={counts['exact']:,}",
                        flush=True,
                    )
        if counts["rows"] != len(prepared):
            raise ComposeLipidError("B5 replay lost eligible records")
        for name, value in implementation.items():
            resolve_pin(value, repo, label=name)
        report = {
            "schema_version": RESULT_SCHEMA,
            "status": "eligible_source_profile_replay_training_unqualified",
            "policy": POLICY,
            "config": pin(repo, config_path),
            "inputs": inputs,
            "implementation": implementation,
            "source_controls": controls,
            "summary": {
                "family": family,
                **dict(counts),
                "by_design_lane": {k: dict(v) for k, v in sorted(lanes.items())},
                "failed_profile_checks": dict(failures),
            },
            "artifact": {
                "path": str((output / ledger.name).relative_to(repo)),
                "sha256": sha256_file(ledger),
            },
            "environment": {
                "python": platform.python_version(),
                "rdkit": rdBase.rdkitVersion,
                "device": "cpu",
                "seed": 0,
            },
            "elapsed_seconds": time.monotonic() - start,
        }
        dump(stage / "result.json", report)
        os.rename(stage, output)
    return report
