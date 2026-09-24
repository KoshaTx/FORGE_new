"""Bind explicitly documented source-lane role vocabularies without changing source rows."""

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
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus import compose_lipid_fixed_replay as canonical
from forge.corpus.compose_lipid_family_replay import replay_record
from forge.corpus.compose_lipid_full_preparation import FullPreparationCorpus
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import compact, rows

CONFIG_SCHEMA = "forge.compose_lipid_fixed_profiles_config.v1"
RESULT_SCHEMA = "forge.compose_lipid_fixed_profiles.v1"
POLICY = {
    **canonical.POLICY,
    "role_binding": "explicit_source_lane_bijection_with_original_instances_retained",
}
IMPLEMENTATION = (*canonical.IMPLEMENTATION, "forge/corpus/compose_lipid_fixed_profiles.py")


def select_executor(item, executor):
    """Select a complete role vocabulary from source metadata, never a product match."""
    declared = [instance[0] for instance in item["preparation"]["component_instances"]]
    if len(declared) != len(set(declared)):
        return executor
    if set(declared) == set(executor["mapping"].values()):
        return executor
    metadata = item["source"].get("primary_metadata", {})
    matches = [
        p
        for p in executor["source_role_profiles"]
        if set(declared) == set(p["registry_to_source_roles"].values())
        and all(metadata.get(k) == v for k, v in p["required_source_metadata"].items())
    ]
    if len(matches) > 1:
        raise ComposeLipidError("Ambiguous source-lane role profile")
    if matches:
        return {**executor, "mapping": matches[0]["registry_to_source_roles"]}
    return executor


def load_contract(repo: Path, config_path: Path):
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("policy") != POLICY:
        raise ComposeLipidError("Source-profile schema or policy differs")
    if set(config.get("inputs", {})) != {
        "preparation",
        "family_definitions",
        "canonical_config",
        "role_profiles",
    }:
        raise ComposeLipidError("Source-profile inputs differ")
    paths = {k: resolve_pin(v, repo, label=k) for k, v in config["inputs"].items()}
    cfg, _, executors, controls = canonical.load_contract(repo, paths["canonical_config"])
    if cfg["families"] != config["families"] or any(
        cfg["inputs"][k] != config["inputs"][k] for k in ("preparation", "family_definitions")
    ):
        raise ComposeLipidError("Source-profile canonical program or population differs")
    document = json.loads(paths["role_profiles"].read_text())
    if document["canonical_config"] != config["inputs"]["canonical_config"]:
        raise ComposeLipidError("Source-profile canonical contract substitution")
    for label, value in document["evidence"].items():
        resolve_pin(value, repo, label=label)
    if set(document["families"]) != set(executors):
        raise ComposeLipidError("Source-profile family scope differs")
    for family, executor in executors.items():
        profiles = document["families"][family]
        identities = set()
        for profile in profiles:
            mapping = profile["registry_to_source_roles"]
            metadata = profile["required_source_metadata"]
            identity = compact(profile)
            if (
                not metadata
                or any(not k or not isinstance(v, str) or not v for k, v in metadata.items())
                or set(mapping) != set(executor["mapping"])
                or any(not isinstance(v, str) or not v for v in mapping.values())
                or len(set(mapping.values())) != len(mapping)
                or identity in identities
            ):
                raise ComposeLipidError("Incomplete or duplicate source-role profile")
            identities.add(identity)
        executor["source_role_profiles"] = profiles
    return config, paths, executors, controls


def run_profile_replay(repo_root: Path, config_path: Path, output_dir: Path):
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
                    replay = replay_record(
                        item, structures, select_executor(item, executors[family])
                    )
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
